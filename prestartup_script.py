import sys
from pathlib import Path

# Windows portable wraps sys.stderr in a cp1252-encoded LogInterceptor
# (ComfyUI/app/logger.py snapshots stream.encoding at wrap time). Anything
# the comfy-env worker forwards back to the host -- including the upstream
# Chinese negative prompt that flows through our sampling diagnostics --
# would then crash with UnicodeEncodeError. Reconfigure to utf-8 BEFORE
# the wrap so the snapshot picks utf-8 and survives any unicode payload.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

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
