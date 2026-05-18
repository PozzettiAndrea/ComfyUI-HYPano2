"""HYPano2Sample — full sampling pipeline in a single subprocess-safe node.

Loads UNet + CLIP + VAE + LoRA from disk (cached), encodes prompts via
ComfyUI's native `TextEncodeQwenImageEditPlus`, runs the sampler with
upstream HY-Pano-2's settings (FlowMatchEulerDiscreteScheduler -> euler +
simple, true_cfg_scale=7.5, steps=40), installs upstream's norm-rescaled
CFG as a post-CFG hook on the live model, decodes via VAE, and edge-blends.

All inputs are JSON-safe (strings / scalars / IMAGE tensor), so the node
works across comfy-env's subprocess IPC boundary — unlike a MODEL-passthrough
node, which can't be serialized.
"""

import logging

import numpy as np
import torch
from PIL import Image

import folder_paths
from comfy_api.latest import io

from .generate import circular_blend_edges, _comfy_image_to_pil, _pil_to_comfy_image

log = logging.getLogger("hypano2")


# Upstream prompt templates (verbatim from pipeline_with_qwen_image.py).
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


def _install_norm_rescaled_cfg(model):
    """Attach upstream's norm-rescaled true CFG as a post-CFG hook.

    Upstream: comb = uncond + cfg*(cond - uncond); out = comb * (||cond|| / ||comb||).
    Operating in ComfyUI's denoised x_0 space, we round-trip via
    eps = (x - x_0)/sigma to apply the rescale in the same space upstream uses.
    """
    m = model.clone()

    def _post_cfg(args):
        cond_x0 = args["cond_denoised"]
        comb_x0 = args["denoised"]
        x = args["input"]
        sigma = args["sigma"].reshape(-1, *(1,) * (x.ndim - 1)).to(x.dtype).clamp(min=1e-6)
        cond_eps = (x - cond_x0) / sigma
        comb_eps = (x - comb_x0) / sigma
        cond_norm = torch.linalg.vector_norm(cond_eps, dim=1, keepdim=True)
        comb_norm = torch.linalg.vector_norm(comb_eps, dim=1, keepdim=True).clamp_min(1e-8)
        return x - sigma * (comb_eps * (cond_norm / comb_norm))

    m.set_model_sampler_post_cfg_function(_post_cfg)
    return m


class HYPano2Sample(io.ComfyNode):
    """Single-node HY-Pano-2 sampling pipeline.

    Loads + applies LoRA + encodes + samples + decodes + edge-blends inside
    the subprocess. All inputs/outputs IPC-safe so the node works under
    comfy-env's subprocess isolation.
    """

    # Class-level cache so re-runs with the same files don't reload.
    _cache_key = None
    _cached_model = None
    _cached_clip = None
    _cached_vae = None

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HYPano2Sample",
            display_name="HY-Pano-2 Sample",
            category="HYPano2",
            description=(
                "End-to-end HY-Pano-2 sampling: loads UNet/CLIP/VAE/LoRA "
                "from disk, encodes the prompt against the input image, "
                "runs upstream's sampler (euler/simple, cfg=7.5, 40 steps) "
                "with norm-rescaled CFG, decodes, and edge-blends. "
                "Single node replaces the whole stock-loader stack."
            ),
            inputs=[
                io.Image.Input("image", tooltip="Input image to expand into a panorama."),
                io.String.Input("prompt", multiline=True, default="A bright sunny outdoor day, peaceful atmosphere."),
                io.String.Input("negative_prompt", multiline=True, default=""),
                io.String.Input("unet_filename",
                                default="qwen_image_edit_2509_fp8mixed.safetensors",
                                tooltip="UNet filename under models/diffusion_models/ (or models/unet/). "
                                        "Run HYPano2DownloadModels once to fetch the defaults."),
                io.String.Input("clip_filename",
                                default="qwen_2.5_vl_7b_fp8_scaled.safetensors",
                                tooltip="Qwen-VL text encoder under models/text_encoders/."),
                io.String.Input("vae_filename",
                                default="qwen_image_vae.safetensors",
                                tooltip="Qwen-Image VAE under models/vae/."),
                io.String.Input("lora_filename",
                                default="pytorch_lora_weights.safetensors",
                                tooltip="HY-Pano-2 LoRA under models/loras/."),
                io.Float.Input("lora_strength", default=1.0, min=0.0, max=2.0, step=0.05),
                io.Int.Input("seed", default=42, min=0, max=2**31 - 1),
                io.Int.Input("width", default=1952, min=512, max=4096, step=16),
                io.Int.Input("height", default=960, min=256, max=2048, step=16),
                io.Int.Input("num_inference_steps", default=40, min=10, max=100),
                io.Float.Input("true_cfg_scale", default=7.5, min=1.0, max=15.0, step=0.5),
                io.Int.Input("blend_width", default=32, min=0, max=128, step=4),
                io.Boolean.Input("use_template", default=True,
                                 tooltip="Wrap the prompt with upstream's prefix/suffix."),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(
        cls,
        image,
        prompt,
        negative_prompt,
        unet_filename,
        clip_filename,
        vae_filename,
        lora_filename,
        lora_strength=1.0,
        seed=42,
        width=1952,
        height=960,
        num_inference_steps=40,
        true_cfg_scale=7.5,
        blend_width=32,
        use_template=True,
    ):
        cls._log_runtime_diag()
        model, clip, vae = cls._get_cached_models(
            unet_filename, clip_filename, vae_filename, lora_filename, lora_strength,
        )

        # Compose prompts.
        if use_template:
            pos = (GENERAL_POSITIVE_PREFIX + prompt + GENERAL_POSITIVE_SUFFIX).strip()
            neg = (GENERAL_NEGATIVE_PROMPT + " " + negative_prompt).strip()
        else:
            pos = prompt.strip()
            neg = negative_prompt.strip()

        # Encode prompts via ComfyUI's native TextEncodeQwenImageEditPlus.
        # Both positive AND negative get the conditioning image (matches
        # pipeline_qwen_pano.py:166-180).
        from comfy_extras.nodes_qwen import TextEncodeQwenImageEditPlus
        log.info("HYPano2Sample: encoding positive prompt...")
        pos_cond_out = TextEncodeQwenImageEditPlus.execute(
            clip, pos, vae=vae, image1=image,
        )
        positive = pos_cond_out.result[0] if hasattr(pos_cond_out, "result") else pos_cond_out
        log.info("HYPano2Sample: encoding negative prompt...")
        neg_cond_out = TextEncodeQwenImageEditPlus.execute(
            clip, neg, vae=vae, image1=image,
        )
        negative = neg_cond_out.result[0] if hasattr(neg_cond_out, "result") else neg_cond_out

        # Empty latent. Matches EmptySD3LatentImage's shape — comfy.sample
        # accepts the dict format from common_ksampler.
        import comfy.model_management
        latent_image = torch.zeros(
            [1, 16, height // 8, width // 8],
            device=comfy.model_management.intermediate_device(),
            dtype=comfy.model_management.intermediate_dtype(),
        )
        latent = {"samples": latent_image, "downscale_ratio_spacial": 8}

        # Patch the live ModelPatcher with upstream's norm-rescaled CFG.
        model_patched = _install_norm_rescaled_cfg(model)

        # Run the sampler. Avoid `nodes.common_ksampler` — `nodes` resolves to
        # our pack's nodes/ package (sys.path puts the pack ahead of ComfyUI),
        # so the host `nodes.py` is shadowed. Call comfy.sample.sample directly
        # and inline the bits common_ksampler does.
        log.info(
            "HYPano2Sample: sampling %dx%d steps=%d cfg=%.1f seed=%d ...",
            width, height, num_inference_steps, true_cfg_scale, seed,
        )
        import comfy.sample
        import latent_preview

        latent_image = comfy.sample.fix_empty_latent_channels(
            model_patched, latent["samples"], latent.get("downscale_ratio_spacial", None),
        )
        noise = comfy.sample.prepare_noise(latent_image, seed, None)
        callback = latent_preview.prepare_callback(model_patched, num_inference_steps)
        import comfy.utils
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        samples_tensor = comfy.sample.sample(
            model_patched, noise, num_inference_steps, true_cfg_scale,
            "euler", "simple", positive, negative, latent_image,
            denoise=1.0, disable_noise=False, start_step=None, last_step=None,
            force_full_denoise=False, noise_mask=None,
            callback=callback, disable_pbar=disable_pbar, seed=seed,
        )

        # VAE decode.
        log.info("HYPano2Sample: VAE decode...")
        decoded = vae.decode(samples_tensor)
        # Qwen-Image uses a 3D VAE (Wan21 latent format), so .decode returns
        # (B, T, H, W, C) with T=1 for image. Squeeze T -> (B, H, W, C).
        if decoded.dim() == 5 and decoded.shape[1] == 1:
            decoded = decoded[:, 0]
        elif decoded.dim() == 4 and decoded.shape[-1] not in (1, 3, 4):
            # 2D VAE: (B, C, H, W) -> (B, H, W, C)
            decoded = decoded.movedim(1, -1)
        log.info("HYPano2Sample: decoded shape -> %s", tuple(decoded.shape))

        # Edge blend.
        log.info("HYPano2Sample: edge-blend width=%d", blend_width)
        outs = []
        for i in range(decoded.shape[0]):
            pil = _comfy_image_to_pil(decoded[i : i + 1])
            blended = circular_blend_edges(pil, int(blend_width))
            outs.append(_pil_to_comfy_image(blended))
        out = torch.cat(outs, dim=0)
        log.info("HYPano2Sample: done -> %s", tuple(out.shape))
        return io.NodeOutput(out)

    # --------------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------------
    @classmethod
    def _log_runtime_diag(cls):
        """One-line per-run dump of attention backend + device + VRAM.

        Worker stderr is forwarded to the host as [worker:ComfyUI-HYPano2]
        prefixed log lines, so this becomes visible confirmation that sage
        (or whatever) actually fired.
        """
        import comfy.model_management as mm
        import torch
        if mm.sage_attention_enabled():
            attn = "sage"
        elif mm.flash_attention_enabled():
            attn = "flash"
        elif mm.xformers_enabled():
            attn = "xformers"
        elif mm.pytorch_attention_enabled():
            attn = "pytorch SDPA (FA2 via cuDNN on Ampere)"
        else:
            attn = "split / sub_quad"
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            device = torch.cuda.get_device_name(0)
        else:
            free = total = 0
            device = "CPU"
        log.info(
            "HYPano2Sample runtime: device=%s | attention=%s | vram free=%.1fGB / %.1fGB",
            device, attn, free / 1e9, total / 1e9,
        )

    # --------------------------------------------------------------
    # Model loading & caching
    # --------------------------------------------------------------
    @classmethod
    def _get_cached_models(cls, unet_filename, clip_filename, vae_filename, lora_filename, lora_strength):
        key = (unet_filename, clip_filename, vae_filename, lora_filename, float(lora_strength))
        if cls._cache_key == key and cls._cached_model is not None:
            return cls._cached_model, cls._cached_clip, cls._cached_vae

        import comfy.sd
        import comfy.utils

        def _resolve(folder: str, filename: str, what: str) -> str:
            p = folder_paths.get_full_path(folder, filename)
            if p is None:
                raise RuntimeError(
                    f"HYPano2Sample: {what} {filename!r} not found in "
                    f"models/{folder}/. Run HYPano2DownloadModels first, or "
                    f"place the file there manually."
                )
            return p

        unet_path = _resolve("diffusion_models", unet_filename, "UNet")
        clip_path = _resolve("text_encoders", clip_filename, "CLIP")
        vae_path = _resolve("vae", vae_filename, "VAE")
        lora_path = _resolve("loras", lora_filename, "LoRA") if lora_filename else None

        log.info("HYPano2Sample: loading UNet %s", unet_filename)
        model = comfy.sd.load_diffusion_model(unet_path)

        log.info("HYPano2Sample: loading CLIP %s", clip_filename)
        clip = comfy.sd.load_clip(
            [clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=comfy.sd.CLIPType.QWEN_IMAGE,
        )

        log.info("HYPano2Sample: loading VAE %s", vae_filename)
        vae_sd = comfy.utils.load_torch_file(vae_path)
        vae = comfy.sd.VAE(sd=vae_sd)

        if lora_path and lora_strength != 0.0:
            log.info("HYPano2Sample: loading LoRA %s strength=%.2f", lora_filename, lora_strength)
            lora_sd = comfy.utils.load_torch_file(lora_path)
            model, _ = comfy.sd.load_lora_for_models(model, None, lora_sd, lora_strength, 0)

        cls._cache_key = key
        cls._cached_model = model
        cls._cached_clip = clip
        cls._cached_vae = vae
        return model, clip, vae
