from pathlib import Path
from comfy_env import setup_env, copy_files

setup_env()

SCRIPT_DIR = Path(__file__).resolve().parent
COMFYUI_DIR = SCRIPT_DIR.parent.parent

# Copy assets to input/3d/
copy_files(SCRIPT_DIR / "assets", COMFYUI_DIR / "input")


# The sampling worker patches comfy's attention dispatcher directly inside
# its subprocess (see nodes/force_attention.py), so we don't touch the host
# ComfyUI process here -- setting --use-flash-attention on the host triggers
# its startup validator, which fails on Windows portable where flash-attn
# isn't installed in the main .venv even though it's in the worker env.
