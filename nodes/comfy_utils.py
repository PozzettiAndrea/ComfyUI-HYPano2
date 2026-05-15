"""Utility functions for HYPano2 nodes."""

import os
from pathlib import Path

import folder_paths

# Register a "hypano2" model folder under ComfyUI/models/hypano2 for the
# HY-Pano-2 LoRA. The Qwen-Image-Edit base lives under
# ComfyUI/models/diffusers/<repo>, which folder_paths handles by default.
_hypano2_models_dir = os.path.join(folder_paths.models_dir, "hypano2")
os.makedirs(_hypano2_models_dir, exist_ok=True)
folder_paths.add_model_folder_path("hypano2", _hypano2_models_dir)


def get_hypano2_models_path() -> Path:
    """Path to ComfyUI/models/hypano2 (created on first call)."""
    p = Path(folder_paths.models_dir) / "hypano2"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_diffusers_models_path() -> Path:
    """Path to ComfyUI/models/diffusers (created on first call)."""
    p = Path(folder_paths.models_dir) / "diffusers"
    p.mkdir(parents=True, exist_ok=True)
    return p
