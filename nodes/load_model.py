"""One-click downloader for the HY-Pano-2 stack.

The actual model loading is done by ComfyUI's native UNETLoader / CLIPLoader
/ VAELoader / LoraLoader — this node just makes sure all four files exist
in the right ComfyUI subdirectories.
"""

import logging
from pathlib import Path

import folder_paths
from comfy_api.latest import io

log = logging.getLogger("hypano2")


# All files come from Comfy-Org's ComfyUI-packaged Qwen mirrors and tencent's
# HY-World-2.0 repo. Files are dropped under each input's standard ComfyUI
# subdirectory so the matching native loaders see them immediately.
_FILES = {
    "diffusion_models": {
        "bf16": ("Comfy-Org/Qwen-Image-Edit_ComfyUI",
                 "split_files/diffusion_models/qwen_image_edit_2509_bf16.safetensors"),
        "fp8":  ("Comfy-Org/Qwen-Image-Edit_ComfyUI",
                 "split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors"),
    },
    "text_encoders": {
        "bf16": ("Comfy-Org/Qwen-Image_ComfyUI",
                 "split_files/text_encoders/qwen_2.5_vl_7b.safetensors"),
        "fp8":  ("Comfy-Org/Qwen-Image_ComfyUI",
                 "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"),
    },
    "vae": {
        "any": ("Comfy-Org/Qwen-Image_ComfyUI",
                "split_files/vae/qwen_image_vae.safetensors"),
    },
    "loras": {
        "any": ("tencent/HY-World-2.0",
                "HY-Pano-2.0/pytorch_lora_weights.safetensors"),
    },
}


def _target_dir(comfy_folder: str) -> Path:
    """Resolve the ComfyUI directory the native loader looks in."""
    paths = folder_paths.get_folder_paths(comfy_folder)
    if not paths:
        raise RuntimeError(
            f"HYPano2DownloadModels: ComfyUI has no '{comfy_folder}' folder configured. "
            f"Check extra_model_paths.yaml or your ComfyUI install."
        )
    return Path(paths[0])


def _download(repo_id: str, filename: str, comfy_folder: str) -> Path:
    """Download `filename` from `repo_id` into ComfyUI's `comfy_folder`.

    Files land flat in the folder (basename only), matching what `UNETLoader`
    and friends list in their dropdowns. Idempotent: if the destination
    already exists with non-zero size we skip the network call.
    """
    from huggingface_hub import hf_hub_download

    dest_dir = _target_dir(comfy_folder)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(filename).name
    if dest.exists() and dest.stat().st_size > 0:
        log.info("HYPano2DownloadModels: present  %s  (%.1f GB)", dest, dest.stat().st_size / 1e9)
        return dest

    log.info("HYPano2DownloadModels: fetching %s : %s  ->  %s", repo_id, filename, dest_dir)
    local = hf_hub_download(repo_id=repo_id, filename=filename)
    # hf_hub_download returns the symlink in its blobs cache; copy or
    # symlink to the ComfyUI dir so loaders find it by basename.
    try:
        dest.symlink_to(local)
    except OSError:
        import shutil
        shutil.copy2(local, dest)
    return dest


class HYPano2DownloadModels(io.ComfyNode):
    """Download Qwen-Image-Edit-2509 + Qwen-VL text encoder + Qwen VAE + HY-Pano-2 LoRA.

    Drops files in `models/diffusion_models/`, `models/text_encoders/`,
    `models/vae/`, `models/loras/`. Pick `bf16` for max quality on a card
    with the headroom, `fp8` to fit a 24 GB consumer card more comfortably.
    Idempotent — re-running with the same `precision` is a no-op.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HYPano2DownloadModels",
            display_name="(Down)Load HY-Pano-2 stack",
            category="HYPano2",
            description=(
                "Downloads Qwen-Image-Edit-2509, the Qwen-VL text encoder, "
                "the Qwen Image VAE, and the HY-Pano-2 LoRA into ComfyUI's "
                "standard subdirs. Use stock UNETLoader / CLIPLoader / "
                "VAELoader / LoraLoader to consume them."
            ),
            inputs=[
                io.Combo.Input(
                    "precision",
                    options=["fp8", "bf16"],
                    default="fp8",
                    tooltip=(
                        "fp8 (default): ~20 GB UNet + ~9 GB text encoder. "
                        "Fits a 24 GB card with the text encoder swapping in "
                        "and out around the sampler. "
                        "bf16: ~41 GB UNet + ~16 GB text encoder. Only viable "
                        "on >=48 GB VRAM cards, or 24 GB VRAM + >=48 GB free "
                        "host RAM."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, precision: str = "bf16"):
        unet  = _download(*_FILES["diffusion_models"][precision], "diffusion_models")
        te    = _download(*_FILES["text_encoders"][precision],    "text_encoders")
        vae   = _download(*_FILES["vae"]["any"],                  "vae")
        lora  = _download(*_FILES["loras"]["any"],                "loras")
        status = (
            f"UNet:  {unet.name}\n"
            f"CLIP:  {te.name}\n"
            f"VAE:   {vae.name}\n"
            f"LoRA:  {lora.name}\n"
            f"Use UNETLoader / CLIPLoader / VAELoader / LoraLoader to load."
        )
        log.info("HYPano2DownloadModels: %s", status.replace("\n", " | "))
        return io.NodeOutput(status)
