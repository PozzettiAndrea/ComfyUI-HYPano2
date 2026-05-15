"""Install hook — pulls flash_attn + sageattention from cuda-wheels into the
host ComfyUI env.

The pack runs in the host env (no per-node pixi isolation), so we sidestep
`comfy_env.install()`'s workspace orchestration and just use `get_wheel_url`
directly to resolve the right prebuilt wheel for this host's (torch, cuda,
python) tuple, then `pip install --no-deps` it.
"""

import subprocess
import sys


CUDA_PACKAGES = ["flash_attn", "sageattention"]


def _host_versions():
    import torch
    torch_version = torch.__version__.split("+")[0]
    cuda_version = torch.version.cuda or "12.8"
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    return torch_version, cuda_version, py_version


def main():
    try:
        from comfy_env import get_wheel_url
    except ImportError:
        print("[hypano2 install] comfy-env not installed yet; installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "comfy-env==0.3.89"])
        from comfy_env import get_wheel_url

    torch_v, cuda_v, py_v = _host_versions()
    print(f"[hypano2 install] host: torch={torch_v} cuda={cuda_v} python={py_v}")

    urls = []
    for pkg in CUDA_PACKAGES:
        url = get_wheel_url(
            pkg,
            torch_version=torch_v,
            cuda_version=cuda_v,
            python_version=py_v,
            log=print,
        )
        if not url:
            print(f"[hypano2 install] no wheel available for {pkg} on this combo; skipping.")
            continue
        urls.append(url)

    if not urls:
        print("[hypano2 install] nothing to install.")
        return

    print(f"[hypano2 install] installing {len(urls)} wheel(s) into host env...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-deps", *urls]
    )
    print("[hypano2 install] done.")


if __name__ == "__main__":
    main()
