"""HYPano2Generate — image -> 360° equirectangular panorama via Qwen-Image-Edit
+ HY-Pano-2 LoRA.

This is the Qwen backend wrapper from
`repo/hyworld2/panogen/pipeline_with_qwen_image.py`, restructured as a
ComfyUI inference node. The HunyuanImage-3 (~80B param) backend is
deliberately not wrapped — see the README for why.
"""

import gc
import logging
import time
from typing import Any

import numpy as np
import torch
from PIL import Image
from comfy_api.latest import io

log = logging.getLogger("hypano2")


def _vram_summary(prefix: str = "") -> str:
    """Compact CUDA memory status — used in node logs to make OOMs traceable."""
    if not torch.cuda.is_available():
        return f"{prefix}cuda: n/a"
    free, total = torch.cuda.mem_get_info()
    used = total - free
    return (
        f"{prefix}cuda free={free / 1e9:.1f}GB used={used / 1e9:.1f}GB "
        f"alloc={torch.cuda.memory_allocated() / 1e9:.1f}GB"
    )


# Upstream prompt templates (verbatim from
# `repo/hyworld2/panogen/pipeline_with_qwen_image.py`). The LoRA was trained
# with these wrappers, so deviating from them degrades quality — exposed as
# an advanced toggle, not removed.
GENERAL_POSITIVE_PREFIX = (
    "Create a **ERP** panoramic expansion of the provided image. "
    "Preserve the original style, lighting, and fine details seamlessly "
    "throughout the extended areas, extend according to: "
)
GENERAL_POSITIVE_SUFFIX = " 8k UHD, masterpiece, razor-sharp details."
GENERAL_NEGATIVE_PROMPT = (
    "低分辨率，低画质，模糊。杂乱的背景，结构扭曲，模糊纹理，物体融合。构图混乱。"
    "过度光滑，画面具有AI感。人脸畸形。巨大物体，巨大建筑，近景特写，近景压迫，比例失调。"
    "车，车辆。画面上方的树叶。"
)


# ---------------------------------------------------------------------------
# Tensor <-> PIL helpers
# ---------------------------------------------------------------------------

def _comfy_image_to_pil(images: torch.Tensor) -> Image.Image:
    """ComfyUI IMAGE (B,H,W,C float[0,1]) -> first-frame PIL.Image (RGB)."""
    if images.dim() == 3:
        images = images.unsqueeze(0)
    if images.dim() != 4 or images.shape[-1] not in (1, 3, 4):
        raise ValueError(
            f"HYPano2Generate: expected IMAGE shape [B,H,W,C], got {tuple(images.shape)}"
        )
    arr = images[0].detach().cpu().clamp(0, 1).numpy()
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    return Image.fromarray(arr)


def _pil_to_comfy_image(img: Image.Image) -> torch.Tensor:
    """PIL.Image -> ComfyUI IMAGE (1,H,W,3 float[0,1])."""
    arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


# ---------------------------------------------------------------------------
# Edge blending — verbatim from upstream
# ---------------------------------------------------------------------------

def circular_blend_edges(image: Image.Image, blend_width: int = 32) -> Image.Image:
    """Cross-fade the left and right edges so the ERP seam disappears.

    Note: the output is cropped by `blend_width` columns on the right (the
    blended region is now redundant with the left). Caller should request
    `width = target + blend_width` if an exact target output width matters.
    """
    if blend_width <= 0:
        return image
    arr = np.array(image)
    for x in range(blend_width):
        arr[:, x, :] = (
            arr[:, -blend_width + x, :] * (1 - x / blend_width)
            + arr[:, x, :] * (x / blend_width)
        )
    return Image.fromarray(arr[:, :-blend_width].astype(np.uint8))


# ---------------------------------------------------------------------------
# Main inference node
# ---------------------------------------------------------------------------

class HYPano2Generate(io.ComfyNode):
    """Run HY-Pano-2 (Qwen backend) on a single input image.

    Caches the diffusers pipeline as a class-level singleton, rebuilds it
    only when the loader handle changes (e.g. user flips dtype or offload).
    """

    _pipeline = None
    _pipeline_key: tuple | None = None
    _patcher = None  # comfy.model_patcher.ModelPatcher wrapping pipe.transformer

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HYPano2Generate",
            display_name="HY-Pano-2 Generate",
            category="HYPano2",
            description=(
                "Expand a single image into a 360° equirectangular panorama "
                "using Qwen-Image-Edit-2509 + the HY-Pano-2 LoRA from Tencent."
            ),
            inputs=[
                io.Custom("HYPANO2_MODEL").Input(
                    "model",
                    tooltip="Model handle from HYPano2LoadModel.",
                ),
                io.Image.Input(
                    "image",
                    tooltip=(
                        "Input image (one frame). The aspect of this image is "
                        "preserved in the prompt's condition thumbnail; the "
                        "output ERP is sized independently via the "
                        "height/width inputs below."
                    ),
                ),
                io.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    tooltip=(
                        "Scene description appended to the upstream positive "
                        "template. Leave empty to keep the default \"extend "
                        "this image\" instruction unchanged."
                    ),
                ),
                io.String.Input(
                    "negative_prompt",
                    default="",
                    multiline=True,
                    tooltip=(
                        "Additional negative cues appended to the upstream "
                        "Chinese-language default negative prompt."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=42, min=0, max=2**31 - 1,
                    tooltip="Random seed for reproducibility.",
                ),
                io.Int.Input(
                    "height",
                    default=960, min=256, max=2048, step=32,
                    tooltip=(
                        "Output ERP height (px). Snapped to a multiple of "
                        "vae_scale_factor*2 internally."
                    ),
                ),
                io.Int.Input(
                    "width",
                    default=1952, min=512, max=4096, step=32,
                    tooltip=(
                        "Output ERP width (px). Snapped to a multiple of "
                        "vae_scale_factor*2 internally. The final output is "
                        "cropped by blend_width on the right."
                    ),
                ),
                io.Int.Input(
                    "num_inference_steps",
                    default=40, min=10, max=100,
                    tooltip="Diffusion denoising steps. Upstream default: 40.",
                ),
                io.Float.Input(
                    "guidance_scale",
                    default=1.0, min=0.0, max=10.0, step=0.1,
                    tooltip=(
                        "Distillation guidance embed. Only active if the "
                        "transformer was trained with guidance embeds "
                        "(Qwen-Image-Edit-2509 is). Upstream default: 1.0."
                    ),
                ),
                io.Float.Input(
                    "true_cfg_scale",
                    default=7.5, min=1.0, max=15.0, step=0.5,
                    tooltip=(
                        "Norm-rescaled true classifier-free guidance scale. "
                        "Setting < 1 disables the unconditional pass. "
                        "Upstream default: 7.5."
                    ),
                ),
                io.Int.Input(
                    "blend_width",
                    default=32, min=0, max=128, step=4,
                    tooltip=(
                        "ERP seam cross-fade width in pixels. The output "
                        "width is reduced by this amount (set to 0 to keep "
                        "the raw output dimensions)."
                    ),
                ),
                io.Float.Input(
                    "crop_border",
                    default=0.0, min=0.0, max=0.25, step=0.01,
                    tooltip=(
                        "Fraction of input border to crop before inference "
                        "(removes JPEG compression rim artefacts)."
                    ),
                ),
                io.Boolean.Input(
                    "use_template",
                    default=True,
                    tooltip=(
                        "Wrap the user prompt in the upstream ERP "
                        "expansion template. Turn off only if you know what "
                        "you're doing — the LoRA was trained with this wrap."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="panorama",
                    tooltip=(
                        "Generated 360° equirectangular panorama. Width may "
                        "be smaller than requested due to seam blending."
                    ),
                ),
            ],
        )

    @classmethod
    @torch.no_grad()
    def execute(
        cls,
        model: Any,
        image: torch.Tensor,
        prompt: str = "",
        negative_prompt: str = "",
        seed: int = 42,
        height: int = 960,
        width: int = 1952,
        num_inference_steps: int = 40,
        guidance_scale: float = 1.0,
        true_cfg_scale: float = 7.5,
        blend_width: int = 32,
        crop_border: float = 0.0,
        use_template: bool = True,
    ):
        if not isinstance(model, dict) or "base_path" not in model:
            raise ValueError(
                f"HYPano2Generate: invalid model handle (expected dict from "
                f"HYPano2LoadModel, got {type(model).__name__})"
            )

        pipe = cls._get_pipeline(model)

        # Tell ComfyUI we're using VRAM so other queued workflows can negotiate
        # eviction against us. With vram_mode=comfy the patcher's load_device
        # == offload_device so this is bookkeeping-only — no extra .to() call.
        if cls._patcher is not None:
            try:
                import comfy.model_management as mm
                mm.load_models_gpu([cls._patcher])
            except ImportError:
                pass

        pil_image = _comfy_image_to_pil(image)
        if crop_border > 0:
            w, h = pil_image.size
            wc, hc = int(crop_border * w), int(crop_border * h)
            pil_image = pil_image.crop((wc, hc, w - wc, h - hc))

        # Compose prompts per the upstream Qwen wrapper.
        if use_template:
            positive = (GENERAL_POSITIVE_PREFIX + prompt + GENERAL_POSITIVE_SUFFIX).strip()
            negative = (GENERAL_NEGATIVE_PROMPT + " " + negative_prompt).strip()
        else:
            positive = prompt.strip()
            negative = negative_prompt.strip()

        log.info(
            "HYPano2Generate: %dx%d steps=%d seed=%d true_cfg=%.1f blend=%d",
            width, height, num_inference_steps, seed, true_cfg_scale, blend_width,
        )
        log.info("HYPano2Generate: %s", _vram_summary("pre-inference "))

        # CPU generator matches the upstream defaults and gives reproducible
        # results regardless of which device the transformer ends up on.
        generator = torch.Generator(device="cpu").manual_seed(int(seed))

        # Step progress: comfy.utils.ProgressBar pushes a 0..N bar into the
        # ComfyUI queue UI and the per-step log line shows wall-clock pace,
        # which is the only signal you get with sequential offload (where
        # each step is silent for ~30-60 s while layers swap CPU<->GPU).
        try:
            import comfy.utils
            pbar = comfy.utils.ProgressBar(int(num_inference_steps))
        except ImportError:
            pbar = None

        t_loop_start = time.perf_counter()
        last_step_t = [t_loop_start]

        def _on_step_end(pipeline, step_idx, timestep, callback_kwargs):
            now = time.perf_counter()
            dt = now - last_step_t[0]
            last_step_t[0] = now
            log.info(
                "HYPano2Generate: step %d/%d  t=%.4f  dt=%.2fs  elapsed=%.1fs",
                step_idx + 1, int(num_inference_steps),
                float(timestep) if hasattr(timestep, "__float__") else timestep,
                dt, now - t_loop_start,
            )
            if pbar is not None:
                pbar.update_absolute(step_idx + 1, int(num_inference_steps))
            # PanoDiffusionPipeline expects a dict-like return; we don't
            # rewrite latents/prompt_embeds, so an empty dict is fine.
            return {}

        log.info("HYPano2Generate: encoding prompt + preparing latents ...")
        try:
            output = pipe(
                image=pil_image,
                prompt=positive,
                negative_prompt=negative or None,
                generator=generator,
                true_cfg_scale=float(true_cfg_scale),
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                num_images_per_prompt=1,
                height=int(height),
                width=int(width),
                callback_on_step_end=_on_step_end,
                callback_on_step_end_tensor_inputs=["latents"],
            ).images[0]
        except torch.cuda.OutOfMemoryError:
            # A failed run leaves the pipeline half-on-GPU; drop it so the next
            # call rebuilds cleanly instead of compounding the OOM.
            log.exception(
                "HYPano2Generate: OOM during inference (%s), tearing down cached pipeline.",
                _vram_summary(),
            )
            cls._teardown()
            raise

        log.info(
            "HYPano2Generate: denoising done in %.1fs (%.2fs/step avg).",
            time.perf_counter() - t_loop_start,
            (time.perf_counter() - t_loop_start) / max(int(num_inference_steps), 1),
        )

        blended = circular_blend_edges(output, blend_width)
        log.info(
            "HYPano2Generate: edge-blend done -> %s  %s",
            blended.size, _vram_summary("post-inference "),
        )
        return io.NodeOutput(_pil_to_comfy_image(blended))

    # ------------------------------------------------------------------
    # Pipeline lifecycle
    # ------------------------------------------------------------------
    @classmethod
    def _get_pipeline(cls, handle: dict):
        dtype_str = handle.get("dtype") or handle.get("torch_dtype", "bf16")
        key = (
            handle["base_path"],
            handle["lora_dir"],
            dtype_str,
        )
        if cls._pipeline is not None and cls._pipeline_key == key:
            return cls._pipeline

        # Tear down any previous instance before loading a new one. A prior
        # OOM can leave the cached pipeline half-on-GPU / half-on-CPU, so
        # this is also the recovery path after a failed run.
        cls._teardown()

        log.info(
            "HYPano2Generate: building PanoDiffusionPipeline (base=%s, dtype=%s)",
            handle["base_path"], dtype_str,
        )
        log.info("HYPano2Generate: %s", _vram_summary("pre-build "))

        # Lazy import — diffusers takes a non-trivial chunk to import; we don't
        # want to pay it at worker spawn time.
        from .hypano2 import PanoDiffusionPipeline

        dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[dtype_str]

        t0 = time.perf_counter()
        pipe = PanoDiffusionPipeline.from_pretrained(
            handle["base_path"],
            torch_dtype=dtype,
        )
        log.info(
            "HYPano2Generate: base model loaded in %.1fs.  %s",
            time.perf_counter() - t0, _vram_summary(),
        )

        # LoRA BEFORE any offload/hook wiring. `accelerate.cpu_offload` (used
        # by enable_model_cpu_offload) and diffusers' group_offloading both
        # install pre-forward hooks that move params between devices; a
        # post-hook `load_lora_weights` would inject LoRA into the wrong
        # device snapshot and either OOM or silently no-op.
        log.info(
            "HYPano2Generate: loading LoRA from %s/%s ...",
            handle["lora_dir"], handle["lora_weight_name"],
        )
        t0 = time.perf_counter()
        pipe.load_lora_weights(
            handle["lora_dir"],
            weight_name=handle["lora_weight_name"],
            torch_dtype=dtype,
        )
        log.info("HYPano2Generate: LoRA loaded in %.1fs.", time.perf_counter() - t0)

        # VAE tiling/slicing caps peak VRAM during decode. Cheap, do it
        # before offload wiring so the hooks see the slicing config.
        if hasattr(pipe, "vae"):
            try:
                pipe.vae.enable_slicing()
                pipe.vae.enable_tiling()
                log.info("HYPano2Generate: VAE slicing + tiling enabled.")
            except AttributeError:
                pass

        # Attention backend follows ComfyUI's startup-time pick: --use-sage-
        # attention / --use-flash-attention launch flags drive both ComfyUI
        # core and our transformer. Single source of truth.
        import comfy.model_management as mm
        if mm.sage_attention_enabled():
            backend = "sage"
        elif mm.xformers_enabled():
            backend = "xformers"
        elif mm.flash_attention_enabled():
            backend = "flash"
        elif mm.pytorch_attention_enabled():
            backend = "native"
        else:
            backend = None
        if backend:
            try:
                pipe.transformer.set_attention_backend(backend)
                log.info("HYPano2Generate: attention backend = %s (from ComfyUI).", backend)
            except Exception as e:
                log.warning("HYPano2Generate: set_attention_backend(%r) failed (%s).", backend, e)

        # Bring text encoder + VAE onto the load device; they're small and
        # run once per generation, so no point streaming them. The
        # transformer is the only thing that needs a memory plan.
        import comfy.model_patcher
        from diffusers.hooks import apply_group_offloading

        load_dev = mm.get_torch_device()
        offload_dev = mm.unet_offload_device()  # GPU on HIGH_VRAM, else CPU
        pipe.text_encoder.to(load_dev)
        pipe.vae.to(load_dev)

        # Auto-budget the transformer's block residency from comfy's
        # free-VRAM reading: aim for ~half of (free - 4 GB activation
        # reserve), so we leave headroom for KV / scheduler / decode peak.
        # On HIGH_VRAM the budget exceeds the block count and group_offload
        # collapses to a single all-resident group — same code path.
        blocks = pipe.transformer.transformer_blocks
        per_block_bytes = sum(p.numel() * p.element_size() for p in blocks[0].parameters())
        free = mm.get_free_memory(load_dev)
        budget = max(0, free - 4 * 1024**3) // 2
        blocks_per_group = min(max(1, int(budget // max(per_block_bytes, 1))), len(blocks))
        log.info(
            "HYPano2Generate: free=%.1fGB per_block=%.0fMB -> blocks_per_group=%d/%d",
            free / 1e9, per_block_bytes / 1e6, blocks_per_group, len(blocks),
        )

        apply_group_offloading(
            pipe.transformer,
            onload_device=load_dev,
            offload_device=offload_dev,
            offload_type="block_level",
            num_blocks_per_group=blocks_per_group,
            use_stream=True,           # overlap H->D with compute
            record_stream=True,        # safe with autograd disabled
            low_cpu_mem_usage=False,
        )

        # ModelPatcher tells comfy's load_models_gpu / cleanup_models that
        # we're holding VRAM. group_offloading already owns the device
        # moves on the transformer's params, so we set load_device ==
        # offload_device to keep comfy from .to()-ing the module behind our
        # back. The size hint reflects what's *actually* resident, not the
        # full ~40 GB transformer, so eviction decisions stay realistic.
        total_params_bytes = sum(p.numel() * p.element_size() for p in pipe.transformer.parameters())
        non_block_bytes = total_params_bytes - per_block_bytes * len(blocks)
        resident_bytes = int(non_block_bytes + per_block_bytes * min(blocks_per_group * 2, len(blocks)))
        cls._patcher = comfy.model_patcher.ModelPatcher(
            pipe.transformer,
            load_device=load_dev,
            offload_device=load_dev,
            size=resident_bytes,
        )
        log.info(
            "HYPano2Generate: pipeline ready. ModelPatcher size=%.1fGB.  %s",
            resident_bytes / 1e9, _vram_summary(),
        )

        cls._pipeline = pipe
        cls._pipeline_key = key
        return pipe

    @classmethod
    def _teardown(cls):
        """Drop the cached pipeline and reclaim VRAM."""
        if cls._patcher is not None:
            try:
                cls._patcher.unpatch_model(device_to=cls._patcher.offload_device)
                cls._patcher.cleanup()
            except Exception:
                pass
            cls._patcher = None
        if cls._pipeline is not None:
            try:
                cls._pipeline.to("cpu")
            except Exception:
                pass
            cls._pipeline = None
            cls._pipeline_key = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Standalone utility node
# ---------------------------------------------------------------------------

class HYPano2BlendEdges(io.ComfyNode):
    """Apply the HY-Pano-2 seam blend to any ERP image.

    Useful when you generate a panorama with another model (or chain through
    HYWM2's SamplePanorama) and want a seamless wrap.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HYPano2BlendEdges",
            display_name="HY-Pano-2 Blend ERP Edges",
            category="HYPano2",
            description=(
                "Cross-fade the left and right edges of an equirectangular "
                "panorama so the seam disappears. Output width = input width "
                "- blend_width."
            ),
            inputs=[
                io.Image.Input("image", tooltip="Equirectangular panorama batch."),
                io.Int.Input(
                    "blend_width",
                    default=32, min=0, max=128, step=4,
                    tooltip="Blend region width in pixels.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(cls, image: torch.Tensor, blend_width: int = 32):
        if image.dim() == 3:
            image = image.unsqueeze(0)
        outs = []
        for i in range(image.shape[0]):
            pil = _comfy_image_to_pil(image[i:i + 1])
            blended = circular_blend_edges(pil, int(blend_width))
            outs.append(_pil_to_comfy_image(blended))
        # All frames have identical dims after blending, so plain cat works.
        return io.NodeOutput(torch.cat(outs, dim=0))
