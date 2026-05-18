import sys
from pathlib import Path
from comfy_env import setup_env, copy_files

setup_env()

SCRIPT_DIR = Path(__file__).resolve().parent
COMFYUI_DIR = SCRIPT_DIR.parent.parent

# Copy assets to input/3d/
copy_files(SCRIPT_DIR / "assets", COMFYUI_DIR / "input")


# Auto-pick the best available attention backend for the HOST ComfyUI process.
#
# ComfyUI's `comfy.ldm.modules.attention` decides at MODULE IMPORT time which
# kernel to use (sage > xformers > flash > pytorch SDPA), based on whether
# the corresponding CLI flag was passed (`--use-sage-attention` etc.).
# Without a flag it defaults to SDPA even when sage/flash are installed. We
# run before any `comfy.ldm` module imports (main.py line 192, attention
# imports later) so flipping `args.use_sage_attention = True` here makes the
# auto-detect pick it up naturally — same as the user typing the flag.
def _autoselect_attention():
    try:
        from comfy.cli_args import args
    except Exception:
        return None
    # Respect explicit user flags.
    if getattr(args, "use_sage_attention", False) or getattr(args, "use_flash_attention", False):
        return None
    # Sage first — typically fastest on Ampere consumer cards.
    try:
        import sageattention  # noqa: F401
        args.use_sage_attention = True
        return "sage"
    except Exception:
        pass
    # FlashAttention 2 fallback.
    try:
        import flash_attn  # noqa: F401
        args.use_flash_attention = True
        return "flash"
    except Exception:
        pass
    return None


_picked = _autoselect_attention()
if _picked:
    print(f"[ComfyUI-HYPano2] auto-selected attention backend: {_picked}")


# Debug pass: pin to flash to rule sage in or out of the black-image NaN.
try:
    sys.path.insert(0, str(SCRIPT_DIR))
    from nodes.force_attention import force_flash
    force_flash()
except Exception as _e:
    print(f"[ComfyUI-HYPano2] force_flash() skipped: {_e}")
