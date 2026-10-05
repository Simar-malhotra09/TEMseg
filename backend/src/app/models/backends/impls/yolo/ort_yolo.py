"""YOLO detection via ultralytics' ONNX backend (production path)."""

import numpy as np
from ultralytics import YOLO

from ...base import YoloBackend

CONF = 0.01
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

    def detect(
        self,
        image_rgb: np.ndarray,
        conf: float = CONF,
        iou: float = IOU,
        max_det: int = MAX_DET,
    ) -> np.ndarray:
        results = self._model.predict(
            source=image_rgb,
            conf=conf,
            iou=iou,
            max_det=max_det,
            verbose=False,
            device=self._device,
        )
        return results[0].boxes.xyxy.cpu().numpy().astype(np.float32)
