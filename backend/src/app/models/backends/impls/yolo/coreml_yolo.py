"""YOLO detection via a CoreML .mlpackage (direct pth->coremltools export).

Preprocess: ultralytics LetterBox(640) on RGB, PIL feed (ImageType scales
1/255 in-model). Postprocess: raw (1,5,8400) head output = xywh + class,
conf filter -> stable-sort greedy NMS -> scale_boxes, all numpy.
"""

import time

import cv2 as cv
import numpy as np
from PIL import Image

from app.logutils import get_logger
from ...base import YoloBackend

logger = get_logger("CoreMLYolo")

CONF = 0.25
IOU = 0.5
MAX_DET = 4000
IMGSZ = 640


def letterbox_rgb(im_rgb: np.ndarray) -> np.ndarray:
    """ultralytics LetterBox(640) on an RGB image -> RGB uint8 HWC (PIL feed)."""
    h, w = im_rgb.shape[:2]
    r = IMGSZ / max(h, w)
    new_unpad = (round(w * r), round(h * r))
    dw = (IMGSZ - new_unpad[0]) / 2
    dh = (IMGSZ - new_unpad[1]) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    im = cv.resize(im_rgb, new_unpad, interpolation=cv.INTER_LINEAR)
    im = cv.copyMakeBorder(
        im, top, bottom, left, right, cv.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return np.ascontiguousarray(im)


def numpy_nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float):
    """Greedy NMS matching torchvision semantics (suppress IoU > thr).

    Stable sort: fp16 confs have ties, torchvision breaks them by original
    order deterministically."""
    order = np.argsort(-scores, kind="stable")
    keep = []
    suppressed = np.zeros(len(order), dtype=bool)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    for i in range(len(order)):
        idx = order[i]
        if suppressed[i]:
            continue
        keep.append(idx)
        rest = order[~suppressed]
        x1 = np.maximum(boxes[idx, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[idx, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[idx, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[idx, 3], boxes[rest, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        union = area[idx] + area[rest] - inter
        ious = np.where(union > 0, inter / union, 0)
        suppressed[~suppressed] |= ious > iou_thr
    return keep


class CoreMLYoloBackend(YoloBackend):
    def __init__(self, pkg_path):
        import coremltools as ct

        t0 = time.perf_counter()
        self._model = ct.models.MLModel(str(pkg_path))
        # first predict pays the .mlpackage -> .mlmodelc compile; warm it now
        self._model.predict({"image": Image.fromarray(np.zeros((640, 640, 3), np.uint8))})
        logger.info(f"coreml yolo loaded+compiled in {time.perf_counter() - t0:.1f}s")

    def detect(self, image_rgb: np.ndarray) -> np.ndarray:
        x = letterbox_rgb(image_rgb)
        raw = self._model.predict({"image": Image.fromarray(x)})["var_1362"]
        pred = raw[0]
        h, w = image_rgb.shape[:2]
        xywh, cls = pred[:4].T, pred[4]
        # raw layout is xywh — convert before NMS (ultralytics xywh2xyxy)
        boxes = np.stack(
            [
                xywh[:, 0] - xywh[:, 2] / 2,
                xywh[:, 1] - xywh[:, 3] / 2,
                xywh[:, 0] + xywh[:, 2] / 2,
                xywh[:, 1] + xywh[:, 3] / 2,
            ],
            1,
        )
        cand = np.where(cls > CONF)[0]
        boxes, cls = boxes[cand], cls[cand]
        sel = numpy_nms(boxes, cls, IOU)[:MAX_DET]
        b = boxes[sel].copy()
        gain = min(IMGSZ / h, IMGSZ / w)
        new_unpad = (round(w * gain), round(h * gain))
        pad_x = (IMGSZ - new_unpad[0]) / 2
        pad_y = (IMGSZ - new_unpad[1]) / 2
        b[:, [0, 2]] = (b[:, [0, 2]] - pad_x) / gain
        b[:, [1, 3]] = (b[:, [1, 3]] - pad_y) / gain
        np.clip(b[:, [0, 2]], 0, w, out=b[:, [0, 2]])
        np.clip(b[:, [1, 3]], 0, h, out=b[:, [1, 3]])
        return b
