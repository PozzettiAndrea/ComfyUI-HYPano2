"""One-click downloader for the HY-Pano-2 stack.

The actual model loading is done by ComfyUI's native UNETLoader / CLIPLoader
/ VAELoader / LoraLoader — this node just makes sure all four files exist
in the right ComfyUI subdirectories.
"""

import logging
import os
from contextlib import contextmanager
from pathlib import Path

import folder_paths
from comfy_api.latest import io

log = logging.getLogger("hypano2")


# huggingface_hub >= 0.30 transparently routes large LFS files through
# Xet (https://huggingface.co/docs/hub/xet) — a content-addressed chunked
# protocol whose client streams bytes through its own pipeline, bypassing
# `huggingface_hub.utils.tqdm`. So neither the console tqdm nor our
# ProgressBar bridge see any updates. Forcing the legacy HTTP path brings
# both back. We only set this if the user hasn't opted in explicitly.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


_FILES = {
    "diffusion_models": {
        "bf16":    ("Comfy-Org/Qwen-Image-Edit_ComfyUI",
                    "split_files/diffusion_models/qwen_image_edit_2509_bf16.safetensors"),
        # fp8 = the scale-augmented hybrid (per-tensor scales recover most of
        # bf16's dynamic range). Same VRAM as raw fp8_e4m3fn, materially better
        # numerics. This is what `precision=fp8` resolves to by default.
        "fp8":     ("Comfy-Org/Qwen-Image-Edit_ComfyUI",
                    "split_files/diffusion_models/qwen_image_edit_2509_fp8mixed.safetensors"),
        # Raw fp8 cast — kept for users who explicitly want it. Lower quality.
        "fp8_raw": ("Comfy-Org/Qwen-Image-Edit_ComfyUI",
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


def _download(repo_id: str, filename: str, comfy_folder: str, expected_size: int = 0) -> Path:
    """Download `filename` from `repo_id` into ComfyUI's `comfy_folder`.

    Idempotent on re-runs: the basename in `comfy_folder` is a symlink to
    the file in HF's cache. We bail out early if that symlink is healthy
    and the size matches (within 0.5% — HF Xet sometimes reports slightly
    different padding). Broken symlinks left by interrupted downloads get
    cleaned up first.
    """
    from huggingface_hub import hf_hub_download

    dest_dir = _target_dir(comfy_folder)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(filename).name

    # Broken symlink from a previous crash -> remove so we can rewrite it.
    if dest.is_symlink() and not dest.exists():
        log.info("HYPano2DownloadModels: stale symlink, removing  %s", dest)
        dest.unlink()

    if dest.exists():
        actual = dest.stat().st_size
        if actual > 0 and (not expected_size or abs(actual - expected_size) < max(expected_size * 0.005, 1024)):
            log.info("HYPano2DownloadModels: present  %s  (%.2f GB)", dest, actual / 1e9)
            return dest
        log.warning(
            "HYPano2DownloadModels: size mismatch on %s (got %.2f GB, expected %.2f GB) - refetching",
            dest, actual / 1e9, expected_size / 1e9,
        )
        dest.unlink()

    log.info("HYPano2DownloadModels: fetching %s : %s  ->  %s", repo_id, filename, dest_dir)
    local = hf_hub_download(repo_id=repo_id, filename=filename)
    # hf_hub_download is itself idempotent — second call hits HF's cache and
    # returns instantly. The work we save by short-circuiting above is just
    # the symlink dance + log spam.
    try:
        dest.symlink_to(local)
    except OSError:
        import shutil
        shutil.copy2(local, dest)
    return dest


def _probe_sizes(manifest):
    """Resolve each (repo, filename) to its remote size via HfApi.

    Returns a list parallel to `manifest` with `size_bytes` appended (0 on
    lookup failure — ProgressBar tolerates a slight under-count).
    """
    from huggingface_hub import HfApi
    api = HfApi()
    by_repo = {}
    for i, (repo_id, fname, folder) in enumerate(manifest):
        by_repo.setdefault(repo_id, []).append((i, fname))
    sizes = [0] * len(manifest)
    for repo_id, items in by_repo.items():
        try:
            info = api.get_paths_info(repo_id=repo_id, paths=[fname for _, fname in items])
            size_by_path = {p.path: getattr(p, "size", 0) or 0 for p in info}
        except Exception as e:
            log.warning("HYPano2DownloadModels: size probe failed for %s (%s)", repo_id, e)
            size_by_path = {}
        for idx, fname in items:
            sizes[idx] = size_by_path.get(fname, 0)
    return [(*m, sz) for m, sz in zip(manifest, sizes)]


@contextmanager
def _hf_progress_into_pbar(pbar, offset_ref, total):
    """Bridge huggingface_hub's internal tqdm into ComfyUI's ProgressBar.

    hf_hub_download writes its own bytes/sec tqdm to the console (we leave
    that alone). We additionally patch tqdm.update so every chunk also
    pushes `offset_ref[0] + tqdm.n` into the ComfyUI queue progress bar.
    """
    # `huggingface_hub.utils.tqdm` exports a `tqdm` class; importing the
    # dotted path resolves the SYMBOL inside `huggingface_hub.utils`, which
    # is the class itself, not the submodule. Grab the class directly to
    # avoid the dotted-attribute confusion.
    from huggingface_hub.utils.tqdm import tqdm as _hf_tqdm
    original = _hf_tqdm.update

    def _patched(self, n=1):
        ret = original(self, n)
        try:
            current = offset_ref[0] + int(self.n or 0)
            if current > total:
                current = total
            pbar.update_absolute(current, total)
        except Exception:
            pass
        return ret

    _hf_tqdm.update = _patched
    try:
        yield
    finally:
        _hf_tqdm.update = original


class HYPano2DownloadModels(io.ComfyNode):
    """Download Qwen-Image-Edit-2509 + Qwen-VL text encoder + Qwen VAE + HY-Pano-2 LoRA.

    Drops files in `models/diffusion_models/`, `models/text_encoders/`,
    `models/vae/`, `models/loras/`. Pick `fp8` (default) to fit a 24 GB card,
    `bf16` for max quality on a ≥48 GB rig. Idempotent — re-runs with the
    same precision are a no-op.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HYPano2DownloadModels",
            display_name="(Down)Load HY-Pano-2 stack",
            category="HYPano2",
            is_output_node=True,
            description=(
                "Downloads Qwen-Image-Edit-2509, the Qwen-VL text encoder, "
                "the Qwen Image VAE, and the HY-Pano-2 LoRA into ComfyUI's "
                "standard subdirs. Use stock UNETLoader / CLIPLoader / "
                "VAELoader / LoraLoader to consume them."
            ),
            inputs=[
                io.Combo.Input(
                    "precision",
                    options=["fp8", "bf16", "fp8_raw"],
                    default="fp8",
                    tooltip=(
                        "fp8 (default): fp8mixed UNet (~20 GB) — fp8 weights "
                        "with per-tensor scale factors that recover most of "
                        "bf16's dynamic range. Plus fp8_scaled text encoder "
                        "(~9 GB). Fits a 24 GB card.\n"
                        "bf16: bf16 UNet (~41 GB) + bf16 text encoder (~16 GB). "
                        "Gold standard, only viable on >=48 GB VRAM rigs.\n"
                        "fp8_raw: unscaled fp8_e4m3fn UNet — same size as fp8 "
                        "but worse numerics. Kept for users who specifically "
                        "want the raw cast."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(display_name="status"),
            ],
        )

    @classmethod
    def execute(cls, precision: str = "fp8"):
        # fp8 and fp8_raw both pair with the fp8_scaled text encoder — the TE
        # only has fp8_scaled and bf16 variants on Comfy-Org's mirror.
        te_precision = "bf16" if precision == "bf16" else "fp8"
        manifest = [
            (*_FILES["diffusion_models"][precision],    "diffusion_models"),
            (*_FILES["text_encoders"][te_precision],    "text_encoders"),
            (*_FILES["vae"]["any"],                     "vae"),
            (*_FILES["loras"]["any"],                   "loras"),
        ]
        sized = _probe_sizes(manifest)
        total = max(sum(s for *_, s in sized), 1)
        log.info(
            "HYPano2DownloadModels: %d files, total %.1f GB to fetch (cached files skipped).",
            len(sized), total / 1e9,
        )

        import comfy.utils
        pbar = comfy.utils.ProgressBar(total)
        offset = [0]
        results = []
        with _hf_progress_into_pbar(pbar, offset, total):
            for repo_id, fname, folder, size in sized:
                dest = _download(repo_id, fname, folder, expected_size=size)
                offset[0] += size
                pbar.update_absolute(offset[0], total)
                results.append((folder, dest.name))

        status = "\n".join(
            f"{folder:>16}: {name}" for folder, name in results
        ) + "\nUse UNETLoader / CLIPLoader / VAELoader / LoraLoaderModelOnly to load."
        log.info("HYPano2DownloadModels: done. %s", status.replace("\n", " | "))
        return io.NodeOutput(status)
