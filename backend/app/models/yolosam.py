from base_model import Model, SegmentationResult
import cv2 as cv
import numpy as np

class YoloSam(Model):
    def __init__(self, model_path: str):
        super().__init__("YoloSam", model_path)

        # load model weights here
        # self.model = ...

    def get_model_specs(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
        }

    def load_image(self, image_path: str) -> np.ndarray:
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        return img

    def segment(self, image: np.ndarray) -> SegmentationResult:
        # dummy segmentation
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        return SegmentationResult(
            segmentation_mask=mask,
        )
