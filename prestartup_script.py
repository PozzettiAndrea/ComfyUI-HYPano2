from pathlib import Path
from comfy_env import setup_env, copy_files

setup_env()

SCRIPT_DIR = Path(__file__).resolve().parent
COMFYUI_DIR = SCRIPT_DIR.parent.parent

# Copy assets to input/3d/
copy_files(SCRIPT_DIR / "assets", COMFYUI_DIR / "input")


# Force flash attention BEFORE any comfy.ldm import. Set the CLI flag so
# ComfyUI's attention dispatcher picks attention_flash when it later loads,
# matching what `--use-flash-attention` does. Inlined (not imported from our
# nodes/ package) to avoid polluting sys.path with this pack's root, which
# would shadow ComfyUI's top-level `nodes.py` (main.py:477 does
# `import nodes; nodes.init_extra_nodes(...)`).
try:
    import flash_attn  # noqa: F401
    from comfy.cli_args import args as _args
    _args.use_sage_attention = False
    _args.use_flash_attention = True
    print("[ComfyUI-HYPano2] forced --use-flash-attention (debug pass)")
except Exception as _e:
    print(f"[ComfyUI-HYPano2] flash flag not set: {_e}")
