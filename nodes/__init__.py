from .load_model import HYPano2DownloadModels
from .sample import HYPano2Sample
from .generate import HYPano2BlendEdges

NODE_CLASS_MAPPINGS = {
    "HYPano2DownloadModels": HYPano2DownloadModels,
    "HYPano2Sample": HYPano2Sample,
    "HYPano2BlendEdges": HYPano2BlendEdges,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HYPano2DownloadModels": "(Down)Load HY-Pano-2 stack",
    "HYPano2Sample": "HY-Pano-2 Sample",
    "HYPano2BlendEdges": "HY-Pano-2 Blend ERP Edges",
}
