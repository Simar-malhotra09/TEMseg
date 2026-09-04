"""YOLO detection via ultralytics' ONNX backend (production path)."""

import numpy as np
from ultralytics import YOLO

from ...base import YoloBackend

CONF = 0.25
IOU = 0.5
MAX_DET = 4000


class OrtYoloBackend(YoloBackend):
    """Wraps an already-loaded ultralytics YOLO(onnx) instance.

    The instance is shared with other pipelines via YoloSam components, so
    this backend must not load its own copy.
    """

    def __init__(self, model: YOLO, device: str):
        self._model = model
        self._device = device

    def detect(self, image_rgb: np.ndarray) -> np.ndarray:
        results = self._model.predict(
            source=image_rgb,
            conf=CONF,
            iou=IOU,
            max_det=MAX_DET,
            verbose=False,
            device=self._device,
        )
        return results[0].boxes.xyxy.cpu().numpy().astype(np.float32)
