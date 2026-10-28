"""MobileSAM TinyViT weights on TemSeg's YOLO and prompt pipelines."""

from typing import Any, Dict

from app.api.live_models import AvailableModels
from app.models.backends.selection import MOBILE_DEC_PKG, MOBILE_ENC_PKG
from app.models.base_model import ModelConfig
from app.models.impls.yolosam import YoloSam


class MobileSam(YoloSam):
    MODEL_ENUM = AvailableModels.mobile_sam
    # converted TinyViT packages (model_scripts/sam_coreml_direct). Falls
    # back to the torch TinyViT when they are not on disk
    USE_COREML_SAM = True
    COREML_ENC_PKG = MOBILE_ENC_PKG
    COREML_DEC_PKG = MOBILE_DEC_PKG

    def __init__(
        self,
        config: ModelConfig,
        device: str,
        components: Dict[str, Any] | None = None,
    ):
        super().__init__(config, device=device, components=components)
