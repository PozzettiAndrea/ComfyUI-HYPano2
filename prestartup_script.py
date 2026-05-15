"""ComfyUI-HYPano2 prestartup: copy bundled example images into ComfyUI/input/.

Each `assets/<scene>/<file>` lands at `ComfyUI/input/<scene>/<file>` so the
LoadImage node's file picker can browse them.
"""

import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
COMFYUI_DIR = SCRIPT_DIR.parent.parent
ASSETS = SCRIPT_DIR / "assets"
TARGET = COMFYUI_DIR / "input"

if ASSETS.is_dir():
    for src in ASSETS.rglob("*"):
        if not src.is_file():
            continue
        dst = TARGET / src.relative_to(ASSETS)
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
