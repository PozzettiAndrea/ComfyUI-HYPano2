import sys
import shutil
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

import folder_paths
from comfy_env import setup_env

setup_env()

# The CONFIGURED input directory, never the code-tree one. ComfyUI Desktop
# (--base-directory) and --input-directory both relocate it, and the load
# nodes only ever scan folder_paths.get_input_directory(). main.py runs
# apply_custom_paths() before prestartup scripts, so this is already resolved.
INPUT = Path(folder_paths.get_input_directory())


def copy_files(src: Path, dst: Path, pattern: str = "*") -> int:
    """Copy bundled assets into a ComfyUI directory. Returns files written.

    Seeds rather than syncs: an existing file is left alone, so a user's
    edited demo asset survives every relaunch. Raises if `src` is missing --
    a typo'd asset directory is a packaging bug, and silence is how it stays
    one.
    """
    src, dst = Path(src), Path(dst)
    if not src.is_dir():
        raise FileNotFoundError(f"asset directory not found: {src}")
    written = 0
    for f in src.glob(pattern):
        if not f.is_file():
            continue
        target = dst / f.relative_to(src)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        written += 1
    return written

SCRIPT_DIR = Path(__file__).resolve().parent

# Copy assets to input/3d/
copy_files(SCRIPT_DIR / "assets", INPUT)


# The sampling worker patches comfy's attention dispatcher directly inside
# its subprocess (see nodes/force_attention.py), so we don't touch the host
# ComfyUI process here -- setting --use-flash-attention on the host triggers
# its startup validator, which fails on Windows portable where flash-attn
# isn't installed in the main .venv even though it's in the worker env.
