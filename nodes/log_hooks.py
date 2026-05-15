"""Wrap comfy.model_management entry points so model lifecycle events show
up loudly in the ComfyUI log. Installed at import time, idempotent.
"""

import logging

log = logging.getLogger("hypano2.mm")


def _summary() -> str:
    import torch
    parts = []
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        parts.append(f"vram free={free / 1e9:.1f}/{total / 1e9:.1f}GB")
    try:
        import psutil
        vm = psutil.virtual_memory()
        parts.append(f"ram free={vm.available / 1e9:.1f}/{vm.total / 1e9:.1f}GB")
    except ImportError:
        pass
    return "  ".join(parts)


def _names(models):
    out = []
    for m in models or []:
        inner = getattr(m, "model", m)
        name = type(inner).__name__
        size_fn = getattr(m, "model_size", None)
        size = (size_fn() if callable(size_fn) else None) or getattr(m, "size", 0) or 0
        out.append(f"{name}({size / 1e9:.1f}GB)" if size else name)
    return ", ".join(out) or "<none>"


def install_hooks():
    import comfy.model_management as mm
    if getattr(mm, "_HYPANO2_HOOKED", False):
        return
    mm._HYPANO2_HOOKED = True

    _orig_load = mm.load_models_gpu
    _orig_free = mm.free_memory
    _orig_unload = mm.unload_all_models
    _orig_cleanup = mm.cleanup_models

    def _hook_load(models, memory_required=0, *a, **kw):
        log.info(
            "[mm] load_models_gpu(%s, want=%.1fGB)  %s",
            _names(models), memory_required / 1e9, _summary(),
        )
        r = _orig_load(models, memory_required, *a, **kw)
        log.info("[mm] load_models_gpu done.             %s", _summary())
        return r

    def _hook_free(memory_required, device, *a, **kw):
        log.info(
            "[mm] free_memory(want=%.1fGB on %s) ...  %s",
            memory_required / 1e9, device, _summary(),
        )
        r = _orig_free(memory_required, device, *a, **kw)
        log.info("[mm] free_memory done.                 %s", _summary())
        return r

    def _hook_unload(*a, **kw):
        log.info("[mm] unload_all_models                  %s", _summary())
        return _orig_unload(*a, **kw)

    def _hook_cleanup(*a, **kw):
        r = _orig_cleanup(*a, **kw)
        log.info("[mm] cleanup_models                     %s", _summary())
        return r

    mm.load_models_gpu = _hook_load
    mm.free_memory = _hook_free
    mm.unload_all_models = _hook_unload
    mm.cleanup_models = _hook_cleanup
    log.info("[mm] hooks installed (hypano2).")


install_hooks()
