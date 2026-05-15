# ComfyUI-HYPano2

ComfyUI wrapper for **HY-Pano 2.0** — Tencent's panorama generator from the HY-World 2.0 pipeline.

Image → 360° equirectangular panorama, via Qwen-Image-Edit-2509 + the HY-Pano-2 LoRA.

| | |
|---|---|
| Upstream | [Tencent-Hunyuan/HY-World-2.0 → `hyworld2/panogen/`](https://github.com/Tencent-Hunyuan/HY-World-2.0/tree/main/hyworld2/panogen) |
| Weights | [tencent/HY-World-2.0 → `HY-Pano-2.0/pytorch_lora_weights.safetensors`](https://huggingface.co/tencent/HY-World-2.0/blob/main/HY-Pano-2.0/pytorch_lora_weights.safetensors) (~810 MB) |
| Base    | [Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) (~40 GB) |
| License | Tencent Hunyuan Community License + Qwen License (read both before redistribution) |

## What's in scope

Only the **Qwen-Image-Edit backend** from upstream is wrapped. The alternative full-stack
HunyuanImage-3 backend (~80B parameters, 32-shard safetensors, ~162 GB) is **out of scope**:
it requires H100/H200-class multi-GPU sharding via `device_map="auto"` and is not realistic
for typical ComfyUI installs.

## Nodes

- **`(Down)Load HY-Pano-2 Model`** — resolves the Qwen base + LoRA on disk. Downloads the ~810 MB LoRA on first run.
- **`HY-Pano-2 Generate`** — image → 360° ERP panorama.
- **`HY-Pano-2 Blend ERP Edges`** — standalone seam blender; works on any ERP panorama (handy
  for chaining with [ComfyUI-HYWM2](https://github.com/PozzettiAndrea/ComfyUI-HYWM2)'s `SamplePanorama`).

## Pairing with ComfyUI-HYWM2

This pack covers the **panorama generation** stage of HY-World 2.0. To go from a generated
panorama to a 3DGS world, chain into [`ComfyUI-HYWM2`](https://github.com/PozzettiAndrea/ComfyUI-HYWM2):

```
LoadImage → HYPano2Generate → HYWM2SamplePanorama → HYWM2Reconstruct → splat / mesh viewers
```

## Why a separate isolated env

Upstream's pipeline imports private helpers from
`diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus`
(`calculate_shift`, `retrieve_timesteps`, `calculate_dimensions`). Minor-version drift in
`diffusers` breaks the import, so we hard-pin `diffusers==0.36.0` together with
`transformers==4.57.1`, `tokenizers==0.22.0`, `safetensors==0.7.0`, and `numpy==2.2.0`.
These would clash with ComfyUI's host venv (and with sibling packs like
`ComfyUI-HYWM2` that pin `numpy<2`), so the nodes run inside an isolated
[`comfy-env`](https://github.com/PozzettiAndrea/comfy-env) subprocess.

## VRAM

VRAM management is **automatic** — we wrap the transformer in
`comfy.model_patcher.ModelPatcher` and use diffusers' `apply_group_offloading` with
a second CUDA stream prefetching the next group of transformer blocks while the
current group's forward runs. Text encoder + VAE stay GPU-resident. On a HIGH_VRAM
machine (`mm.unet_offload_device()` returns the GPU) `apply_group_offloading`
becomes a no-op and the whole pipeline stays loaded; no user toggle needed.

The one tunable on the loader is `blocks_per_group` (default 4) — how many of the
60 transformer blocks live on GPU at once. 4 → ~2.7 GB peak resident for the block
stack on a 24 GB card. Bump it for faster runs if you have VRAM headroom, drop it
if you OOM.

VAE slicing + tiling are always enabled so the final decode step doesn't spike VRAM
at high output resolutions.

The `precision` field (default `auto`) follows the standard ComfyUI pattern:
`mm.should_use_bf16` → bf16 on Ampere+, else `mm.should_use_fp16` → fp16, else fp32.

## Attention kernel

The transformer's attention backend follows **ComfyUI's startup-time detection** —
launch ComfyUI with one of `--use-sage-attention`, `--use-flash-attention`, or no flag
(torch SDPA fallback), and our node automatically calls
`pipe.transformer.set_attention_backend(...)` with the matching diffusers backend
(`sage`, `flash`, `xformers`, or `native`). Single source of truth: no separate
combo box on our node, no settings drift between core ComfyUI samplers and HYPano2.

Both `flash_attn` (v2.8.3) and `sageattention` (v2.2.0) are auto-installed via
[`cuda-wheels`](https://github.com/PozzettiAndrea/cuda-wheels) — prebuilt for cu128 /
py3.13 / torch 2.8, with a native SM 8.6 cubin for Ampere consumer cards (no JIT
fallback on a 3090).

## Citation

```bibtex
@article{hyworld22026,
  title={HY-World 2.0: A Multi-Modal World Model for Reconstructing, Generating, and Simulating 3D Worlds},
  author={Team HY-World},
  journal={arXiv preprint},
  year={2026}
}
```
