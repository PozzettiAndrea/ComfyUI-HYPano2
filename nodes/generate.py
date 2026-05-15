"""HY-Pano-2 ERP edge blend.

The actual panorama generation runs through ComfyUI's native Qwen-Image-Edit
support (UNETLoader + LoraLoader + TextEncodeQwenImageEditPlus + KSampler) —
see the bundled workflow. The only HY-Pano-2-specific piece left in this
pack is the seamless seam blend, kept here as a standalone IMAGE -> IMAGE
node so it can chain off any sampler output, not just ours.
"""

import numpy as np
import torch
from PIL import Image
from comfy_api.latest import io


def _comfy_image_to_pil(images: torch.Tensor) -> Image.Image:
    """ComfyUI IMAGE (B,H,W,C float[0,1]) -> first-frame PIL.Image (RGB)."""
    if images.dim() == 3:
        images = images.unsqueeze(0)
    if images.dim() != 4 or images.shape[-1] not in (1, 3, 4):
        raise ValueError(
            f"HYPano2BlendEdges: expected IMAGE shape [B,H,W,C], got {tuple(images.shape)}"
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


def circular_blend_edges(image: Image.Image, blend_width: int = 32) -> Image.Image:
    """Cross-fade the left/right edges so the ERP seam disappears.

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


class HYPano2NormRescaledCFG(io.ComfyNode):
    """Patch a MODEL to match HY-Pano-2's upstream CFG formula.

    Upstream `pipeline_qwen_pano.py` doesn't do plain CFG — it does a
    *norm-rescaled* CFG:

        comb = uncond + cfg * (cond - uncond)
        out  = comb * (||cond||_per_token / ||comb||_per_token)

    The rescale forces the post-CFG prediction's magnitude to match the
    conditional prediction's, which prevents the over-saturated / "baked"
    look that standard CFG produces at high scales (we ship cfg=7.5).

    ComfyUI exposes `set_model_sampler_post_cfg_function`, which fires
    after the standard blend. Upstream rescales in noise-prediction space;
    ComfyUI hands us denoised x_0 estimates, so we round-trip via
    `eps = (x - x_0) / sigma` to apply the rescale in the right space.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HYPano2NormRescaledCFG",
            display_name="HY-Pano-2 Norm-Rescaled CFG",
            category="HYPano2",
            inputs=[io.Model.Input("model")],
            outputs=[io.Model.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, model):
        m = model.clone()

        def _post_cfg(args):
            cond_x0 = args["cond_denoised"]
            comb_x0 = args["denoised"]
            x = args["input"]
            sigma = args["sigma"]

            # Broadcast sigma [B] -> [B, 1, ..., 1] across non-batch dims.
            sigma_b = sigma.reshape(-1, *(1,) * (x.ndim - 1)).to(x.dtype)
            # Avoid div-by-zero at the very first / very last step of the
            # schedule where sigma can be tiny.
            sigma_b = sigma_b.clamp(min=1e-6)

            cond_eps = (x - cond_x0) / sigma_b
            comb_eps = (x - comb_x0) / sigma_b

            # Norm per spatial location over the channel dim (latent C).
            # Closest analog to upstream's per-token norm over hidden_dim:
            # each latent (B, h, w) corresponds to one transformer token,
            # and that token's hidden features get linearly projected into
            # the C latent channels.
            cond_norm = torch.linalg.vector_norm(cond_eps, dim=1, keepdim=True)
            comb_norm = torch.linalg.vector_norm(comb_eps, dim=1, keepdim=True).clamp_min(1e-8)
            rescaled_eps = comb_eps * (cond_norm / comb_norm)

            # Back to x_0 space (what ComfyUI's sampler will consume).
            return x - sigma_b * rescaled_eps

        m.set_model_sampler_post_cfg_function(_post_cfg)
        return io.NodeOutput(m)


class HYPano2BlendEdges(io.ComfyNode):
    """Cross-fade the left/right edges of an ERP panorama so the seam disappears."""

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
        return io.NodeOutput(torch.cat(outs, dim=0))
