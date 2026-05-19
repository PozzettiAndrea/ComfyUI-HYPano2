"""Pin ComfyUI's attention dispatch to flash_attn 2.

Debug pass: the sage path produced NaN/black output. Switch to flash and
add per-stage tensor diagnostics in sample.py to localize the failure.
Sage stays installed (just not dispatched to) so we can flip back easily.

Call `force_flash()` from BOTH prestartup_script.py (host) and the top of
nodes/sample.py (worker). Both calls are idempotent -- sentinel on the
comfy attention module.
"""

import sys


def _stderr(msg: str) -> None:
    print(f"[HYPano2 force_attention] {msg}", file=sys.stderr, flush=True)


def force_flash() -> None:
    """Pin ComfyUI's attention dispatch to flash_attn 2. Idempotent.

    Patches `optimized_attention` directly instead of flipping the CLI flag
    -- the flag triggers comfy's startup validator and fails on hosts where
    flash-attn lives in the worker env but not the main process.
    """
    try:
        import comfy.ldm.modules.attention as a
    except Exception as e:
        _stderr(f"could not import comfy attention module: {e}")
        return

    if getattr(a, "_HYPANO2_FORCED_FLASH", False):
        return

    if not getattr(a, "FLASH_ATTENTION_IS_AVAILABLE", False):
        _stderr("flash_attn not importable in this env; leaving default")
        return

    a.optimized_attention = a.attention_flash
    a.optimized_attention_masked = a.attention_flash
    a._HYPANO2_FORCED_FLASH = True
    _stderr("forced attention=flash (was sage)")
