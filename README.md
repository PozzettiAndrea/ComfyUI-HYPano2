# ComfyUI-HYPano2

ComfyUI wrapper for **HY-Pano 2.0** — Tencent's panorama generator from the HY-World 2.0 pipeline.

Image → 360° equirectangular panorama, via ComfyUI's native Qwen-Image-Edit-2509 support + the HY-Pano-2 LoRA.

| | |
|---|---|
| Upstream | [Tencent-Hunyuan/HY-World-2.0 → `hyworld2/panogen/`](https://github.com/Tencent-Hunyuan/HY-World-2.0/tree/main/hyworld2/panogen) |
| Weights | [tencent/HY-World-2.0 → `HY-Pano-2.0/pytorch_lora_weights.safetensors`](https://huggingface.co/tencent/HY-World-2.0/blob/main/HY-Pano-2.0/pytorch_lora_weights.safetensors) (~810 MB) |
| Base    | [Comfy-Org/Qwen-Image-Edit_ComfyUI → `qwen_image_edit_2509_bf16.safetensors`](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI) (~41 GB; ~20 GB fp8 variant available) |
| License | Tencent Hunyuan Community License + Qwen License (read both before redistribution) |

## What's in scope

Only the Qwen-Image-Edit backend from upstream is wrapped. The alternative HunyuanImage-3
backend (~80B parameters) is **out of scope** — it requires H100-class multi-GPU sharding
and is not realistic for typical ComfyUI installs.

## How it works

There is no custom inference code in this pack. Everything runs through ComfyUI's native
Qwen-Image-Edit support:

- `UNETLoader` mmaps `qwen_image_edit_2509_*.safetensors` from `models/diffusion_models/`.
- `CLIPLoader` (type=`qwen_image`) loads the Qwen-VL text encoder from `models/text_encoders/`.
- `VAELoader` loads `qwen_image_vae.safetensors` from `models/vae/`.
- `LoraLoaderModelOnly` applies the HY-Pano-2 LoRA from `models/loras/`.
- `TextEncodeQwenImageEditPlus` (from `comfy_extras.nodes_qwen`) is the exact in-tree
  equivalent of diffusers' `QwenImageEditPlusPipeline.encode_prompt` — tokenizes prompt +
  reference images and emits a `reference_latents` conditioning.
- `KSampler` runs the denoising loop.
- `VAEDecode` produces the ERP image.
- `HYPano2BlendEdges` (this pack) cross-fades the wrap-around seam.

Because the pack rides on stock ComfyUI, VRAM/RAM management comes for free:
safetensors mmap → `ModelPatcher.partially_load` → weight-function streaming. Fits on
24 GB cards without `--lowvram`, fits on 32 GB RAM machines without thrashing.

## Nodes

- **`(Down)Load HY-Pano-2 stack`** — one-click downloader. Fetches the four files from
  HuggingFace into `models/diffusion_models/`, `models/text_encoders/`, `models/vae/`,
  `models/loras/`.

  | `precision` | UNet on disk | Text encoder | Fits |
  |---|---|---|---|
  | `fp8` (default) | 20 GB | 9 GB | 24 GB VRAM + 16 GB RAM |
  | `bf16` | 41 GB | 16 GB | ≥48 GB VRAM, or 24 GB VRAM + ≥48 GB free RAM |
- **`HY-Pano-2 Blend ERP Edges`** — cross-fade the left/right edges of an ERP panorama
  so the seam disappears. Standalone IMAGE → IMAGE node; works on any panorama, not
  just ones produced by this workflow (handy for chaining with [`ComfyUI-HYWM2`](https://github.com/PozzettiAndrea/ComfyUI-HYWM2)'s `SamplePanorama`).

## Workflow

`workflows/image_to_panorama.json` wires the full graph using stock loaders.

## Sampler settings

The bundled workflow ships with upstream HY-Pano-2's values:

| KSampler widget | Value | Source |
|---|---|---|
| `steps` | 40 | upstream `pipeline_qwen_pano.py` HY-Pano default |
| `cfg` | 7.5 | maps to upstream `true_cfg_scale` |
| `sampler` | `euler` | upstream uses `FlowMatchEulerDiscreteScheduler` |
| `scheduler` | `simple` | flow-match shift 1.15 set by `comfy.supported_models.QwenImage` |
| `denoise` | 1.0 | full denoise — start from pure noise + reference latents |
| LoRA strength | 1.0 | weights trained against this scale |

Both `TextEncodeQwenImageEditPlus` nodes (positive and negative) receive the
input image via `image1` — upstream encodes the negative prompt against the
same conditioning image so CFG can subtract a matched negative direction.

## Pairing with ComfyUI-HYWM2

```
LoadImage → [Qwen-Image-Edit-2509 + HY-Pano-2 LoRA] → HYPano2BlendEdges
          → HYWM2SamplePanorama → HYWM2Reconstruct → splat / mesh viewers
```

## Citation

```bibtex
@article{hyworld22026,
  title={HY-World 2.0: A Multi-Modal World Model for Reconstructing, Generating, and Simulating 3D Worlds},
  author={Team HY-World},
  journal={arXiv preprint},
  year={2026}
}
```
