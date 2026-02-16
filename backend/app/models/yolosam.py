import cv2 as cv
import numpy as np
from typing import Dict, Any

from base_model import Model, SegmentationResult, ModelConfig


class YoloSam(Model):
    def __init__(self, config: ModelConfig):
        super().__init__(config)

    def _load_components(self) -> Dict[str, Any]:
        loaded = {}

        for sub_model in self.config.components:
            print(f"Loading {sub_model.name} from {sub_model.path}")
            loaded[sub_model.name] = f"loaded_from_{sub_model.path}"

        return loaded

    def load_image(self, image_path: str) -> np.ndarray:
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        return img

    def segment(self, image: np.ndarray) -> SegmentationResult:
        # mask = np.zeros(image.shape[:2], dtype=np.uint8)
        #
        # return SegmentationResult(
        #     segmentation_mask=mask,
        #     metadata={"pipeline": self.config.name}
        # )
