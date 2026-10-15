"""MobileSAM TinyViT weights on TemSeg's YOLO and prompt pipelines."""

from typing import Any, Dict

from app.api.live_models import AvailableModels
from app.models.base_model import ModelConfig
from app.models.impls.yolosam import YoloSam


class MobileSam(YoloSam):
    MODEL_ENUM = AvailableModels.mobile_sam
    USE_COREML_SAM = False

    def __init__(
        self,
        config: ModelConfig,
        device: str,
        components: Dict[str, Any] | None = None,
    ):
        super().__init__(config, device=device, components=components)
