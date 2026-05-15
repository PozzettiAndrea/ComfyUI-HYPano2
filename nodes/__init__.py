from . import log_hooks  # noqa: F401  — installs comfy.model_management log hooks
from .load_model import HYPano2DownloadModels
from .generate import HYPano2BlendEdges

NODE_CLASS_MAPPINGS = {
    "HYPano2DownloadModels": HYPano2DownloadModels,
    "HYPano2BlendEdges": HYPano2BlendEdges,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HYPano2DownloadModels": "(Down)Load HY-Pano-2 stack",
    "HYPano2BlendEdges": "HY-Pano-2 Blend ERP Edges",
}
