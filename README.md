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

Qwen-Image-Edit-2509 in bf16 is ~40 GB resident. The loader exposes an `enable_cpu_offload`
toggle that wires through to diffusers' `enable_model_cpu_offload()` — slower, but fits on
~24 GB cards.

## Citation

```bibtex
@article{hyworld22026,
  title={HY-World 2.0: A Multi-Modal World Model for Reconstructing, Generating, and Simulating 3D Worlds},
  author={Team HY-World},
  journal={arXiv preprint},
  year={2026}
}
```
