from .load_model import HYPano2LoadModel
from .generate import HYPano2Generate, HYPano2BlendEdges

NODE_CLASS_MAPPINGS = {
    "HYPano2LoadModel": HYPano2LoadModel,
    "HYPano2Generate": HYPano2Generate,
    "HYPano2BlendEdges": HYPano2BlendEdges,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HYPano2LoadModel": "(Down)Load HY-Pano-2 Model",
    "HYPano2Generate": "HY-Pano-2 Generate",
    "HYPano2BlendEdges": "HY-Pano-2 Blend ERP Edges",
}
