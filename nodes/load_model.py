"""HYPano2LoadModel — resolve the Qwen-Image-Edit base + HY-Pano-2 LoRA on disk
and return a config dict consumed by HYPano2Generate.

The actual pipeline is built lazily inside HYPano2Generate so we don't take
the ~40 GB VRAM hit until the user runs the graph. This mirrors the pattern
used by ComfyUI-HYWM2's LoadHYWM2Model: loader = "download + describe",
inference node = "build pipeline + run forward".
"""

import logging
from contextlib import contextmanager
from pathlib import Path

from comfy_api.latest import io

try:
    from .comfy_utils import get_diffusers_models_path, get_hypano2_models_path
except ImportError:
    from comfy_utils import get_diffusers_models_path, get_hypano2_models_path

log = logging.getLogger("hypano2")


# Default HuggingFace sources matching the upstream Qwen backend defaults
# (see repo/hyworld2/panogen/pipeline_with_qwen_image.py).
DEFAULT_BASE_REPO = "Qwen/Qwen-Image-Edit-2509"
DEFAULT_LORA_REPO = "tencent/HY-World-2.0"
DEFAULT_LORA_SUBFOLDER = "HY-Pano-2.0"
LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"

# Bytes in the LoRA file on HuggingFace (per the model card json).
_LORA_EXPECTED_SIZE = 849_544_392


@contextmanager
def _comfy_hf_progress(total_bytes: int):
    """Wire huggingface_hub's internal tqdm into ComfyUI's ProgressBar.

    huggingface_hub >=1.x exposes a `tqdm_class` kwarg on hf_hub_download,
    but the 0.36.x branch we're pinned to (transformers 4.57.1 caps it at
    <1.0) doesn't — so we monkey-patch the tqdm subclass it uses for the
    duration of the download instead. Every chunk update pushes bytes into
    `comfy.utils.ProgressBar` so the queue UI shows live byte progress for
    the ~810 MB LoRA download.
    """
    try:
        import comfy.utils
        import huggingface_hub.utils.tqdm as hf_tqdm_mod
        import huggingface_hub.file_download as fd
    except ImportError:
        yield
        return

    pbar = comfy.utils.ProgressBar(total_bytes)
    original_update = hf_tqdm_mod.tqdm.update

    def _patched_update(self, n=1):
        ret = original_update(self, n)
        if n and getattr(self, "total", None):
            pbar.update_absolute(min(self.n, total_bytes), total_bytes)
        return ret

    hf_tqdm_mod.tqdm.update = _patched_update
    # `file_download.tqdm` is a re-export of the same class, so patching the
    # class object covers both call sites.
    try:
        yield
    finally:
        hf_tqdm_mod.tqdm.update = original_update


def _download_lora(repo_id: str, subfolder: str) -> Path:
    """Download the HY-Pano-2 LoRA file into ComfyUI/models/hypano2/<subfolder>/.

    Returns the local path to the directory that holds
    `pytorch_lora_weights.safetensors`.
    """
    target_dir = get_hypano2_models_path() / subfolder
    target_dir.mkdir(parents=True, exist_ok=True)
    lora_path = target_dir / LORA_WEIGHT_NAME

    # Tolerate a 10% size deviation (the HF size field is occasionally
    # off-by-a-few-bytes; we mostly want to detect "0-byte stub").
    if lora_path.exists():
        actual = lora_path.stat().st_size
        if abs(actual - _LORA_EXPECTED_SIZE) < _LORA_EXPECTED_SIZE * 0.1:
            log.info("HY-Pano-2 LoRA present at %s (%.1f MB)", lora_path, actual / 1e6)
            return target_dir
        log.warning(
            "Existing LoRA at %s is %.1f MB, expected ~%.1f MB. Redownloading.",
            lora_path, actual / 1e6, _LORA_EXPECTED_SIZE / 1e6,
        )

    from huggingface_hub import hf_hub_download

    log.info("Downloading %s/%s from %s ...", subfolder, LORA_WEIGHT_NAME, repo_id)
    with _comfy_hf_progress(_LORA_EXPECTED_SIZE):
        hf_hub_download(
            repo_id=repo_id,
            filename=f"{subfolder}/{LORA_WEIGHT_NAME}" if subfolder else LORA_WEIGHT_NAME,
            local_dir=str(get_hypano2_models_path()),
        )
    log.info("LoRA downloaded to %s", lora_path)
    return target_dir


def _resolve_base_model(repo_id: str) -> str:
    """Return a local path or repo_id that diffusers can pass to from_pretrained.

    If a local snapshot exists under `ComfyUI/models/diffusers/<repo>/`, use
    that. Otherwise return the bare repo_id and let diffusers manage the HF
    cache (the Qwen-Image-Edit-2509 snapshot is ~40 GB; we don't force a
    re-download into our own folder layout).
    """
    safe_name = repo_id.replace("/", "_")
    local = get_diffusers_models_path() / safe_name
    if (local / "model_index.json").exists():
        log.info("Using local Qwen-Image-Edit snapshot at %s", local)
        return str(local)
    log.info(
        "No local snapshot at %s — diffusers will resolve %s via HF cache.",
        local, repo_id,
    )
    return repo_id


class HYPano2LoadModel(io.ComfyNode):
    """Resolve the Qwen-Image-Edit base + HY-Pano-2 LoRA and return a handle.

    The handle is a small JSON-safe dict consumed by HYPano2Generate, which
    builds the diffusers pipeline lazily.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="HYPano2LoadModel",
            display_name="(Down)Load HY-Pano-2 Model",
            category="HYPano2",
            description=(
                "Resolve Qwen-Image-Edit-2509 + HY-Pano-2 LoRA (downloads the "
                "LoRA from HuggingFace if missing). The diffusers pipeline is "
                "built lazily inside HYPano2Generate."
            ),
            inputs=[
                io.String.Input(
                    "base_model",
                    default=DEFAULT_BASE_REPO,
                    multiline=False,
                    tooltip=(
                        "HuggingFace repo ID (or local path) for the "
                        "Qwen-Image-Edit base. Default: Qwen/Qwen-Image-Edit-2509."
                    ),
                ),
                io.String.Input(
                    "lora_repo",
                    default=DEFAULT_LORA_REPO,
                    multiline=False,
                    tooltip=(
                        "HuggingFace repo containing the HY-Pano-2 LoRA. "
                        "Default: tencent/HY-World-2.0."
                    ),
                ),
                io.String.Input(
                    "lora_subfolder",
                    default=DEFAULT_LORA_SUBFOLDER,
                    multiline=False,
                    tooltip="Subfolder inside the LoRA repo. Default: HY-Pano-2.0.",
                ),
                io.Combo.Input(
                    "torch_dtype",
                    options=["bf16", "fp16"],
                    default="bf16",
                    tooltip=(
                        "Inference dtype. Upstream defaults to bf16 — keep that "
                        "unless your GPU lacks bf16 support."
                    ),
                ),
                io.Boolean.Input(
                    "enable_cpu_offload",
                    default=False,
                    tooltip=(
                        "Use diffusers' enable_model_cpu_offload() to swap "
                        "transformer / VAE / text encoder between CPU and GPU. "
                        "Slower but fits on smaller cards (~24 GB)."
                    ),
                ),
            ],
            outputs=[
                io.Custom("HYPANO2_MODEL").Output(
                    display_name="model",
                    tooltip=(
                        "HY-Pano-2 model handle. Pass to HYPano2Generate."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        base_model: str = DEFAULT_BASE_REPO,
        lora_repo: str = DEFAULT_LORA_REPO,
        lora_subfolder: str = DEFAULT_LORA_SUBFOLDER,
        torch_dtype: str = "bf16",
        enable_cpu_offload: bool = False,
    ):
        log.info(
            "HYPano2LoadModel: base=%s lora=%s/%s dtype=%s offload=%s",
            base_model, lora_repo, lora_subfolder, torch_dtype, enable_cpu_offload,
        )

        lora_dir = _download_lora(lora_repo, lora_subfolder)
        base_path = _resolve_base_model(base_model)

        handle = {
            "base_path": base_path,
            "lora_dir": str(lora_dir),
            "lora_weight_name": LORA_WEIGHT_NAME,
            "torch_dtype": torch_dtype,
            "enable_cpu_offload": bool(enable_cpu_offload),
        }
        return io.NodeOutput(handle)
