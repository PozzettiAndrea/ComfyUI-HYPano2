"""ComfyUI-HYPano2 Prestartup Script."""

from pathlib import Path

from comfy_env import setup_env, copy_files

setup_env()

SCRIPT_DIR = Path(__file__).resolve().parent
COMFYUI_DIR = SCRIPT_DIR.parent.parent

# Copy bundled example images into ComfyUI's `input/` so each `assets/<scene>/`
# subdir shows up in `LoadImage`'s file picker. The recursive `**/*` glob means
# `assets/office/office.jpg` lands at `input/office/office.jpg` and the scene
# subdirs become folders the user can pick from. Same pattern HYWM2 and
# DepthAnythingV3 use.
copy_files(SCRIPT_DIR / "assets", COMFYUI_DIR / "input", "**/*")
