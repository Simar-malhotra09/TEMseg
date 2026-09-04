"""HACK BRANCH (coreml-dual-pipeline-hack): YoloSam with the dual CoreML
pipeline (YOLO .mlpackage + SAM encoder/decoder .mlpackages) on arm Macs.

This is a deliberate fast-and-hardcoded integration for IN-APP TESTING, not
the final architecture. The proper backend-abstraction refactor comes after
this proves out. Everything not on the main segment() path (point prompts,
segment_batch, FasterYoloSam component reuse) still uses the torch stack,
which stays loaded — so no other endpoint changes behaviour.

What changes in segment():
    YOLO  : best12x.mlpackage (ImageType, PIL letterbox feed) + numpy decode
            (xywh->xyxy, conf 0.25, stable-sort NMS iou 0.5, scale_boxes)
    SAM   : sam_encoder_vit_b_d12_fp32.mlpackage (numpy preprocess) +
            sam_decoder_head64_fp32.mlpackage (static B=64, last chunk
            duplicate-padded — no-op under max-logit union) + numpy
            crop/resize postprocess + running max-logit union with owner
            labels (argmax), threshold > 0.
Numerics vs the torch path measured on the 41-image A/B: encoder 8.6e-6,
decoder logits ~5e-3, YOLO boxes +/-1 on 5/41 boundary images. The known
open question is downstream particle-count parity on those boundary images.

Enable: darwin+arm64 and the three .mlpackage files present in WEIGHTS_DIR.
Force-off with env TEMSEG_COREML=0.
"""

import logging
import os
import platform
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from PIL import Image

from app.logutils import Timer, get_logger
from app.models.impls.yolosam import (
    SAMEmbedding,
    YoloSam,
    YoloSAMSegmentationResult,
)

logger = get_logger("YoloSamCoreML")

WEIGHTS_DIR = Path.home() / "Library" / "Application Support" / "TEMseg" / "weights"
YOLO_PKG = WEIGHTS_DIR / "best12x.mlpackage"
ENC_PKG = WEIGHTS_DIR / "sam_encoder_vit_b_d12_fp32.mlpackage"
DEC_PKG = WEIGHTS_DIR / "sam_decoder_head64_fp32.mlpackage"

IMG_SIZE = 1024
CONF = 0.25
IOU = 0.5
MAX_DET = 4000
B64 = 64
PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
PIXEL_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def coreml_available() -> bool:
    if os.environ.get("TEMSEG_COREML", "").strip().lower() in ("0", "false", "no"):
        return False
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    return all(p.exists() for p in (YOLO_PKG, ENC_PKG, DEC_PKG))


def letterbox_rgb(im_rgb: np.ndarray):
    """ultralytics LetterBox(640) on an RGB image -> RGB uint8 HWC (PIL feed).
    The app's images are already RGB (load_image converts BGR->RGB), unlike
    the benchmark scripts which fed cv2 BGR."""
    h, w = im_rgb.shape[:2]
    r = 640 / max(h, w)
    new_unpad = (round(w * r), round(h * r))
    dw = (640 - new_unpad[0]) / 2
    dh = (640 - new_unpad[1]) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    im = cv.resize(im_rgb, new_unpad, interpolation=cv.INTER_LINEAR)
    im = cv.copyMakeBorder(im, top, bottom, left, right, cv.BORDER_CONSTANT, value=(114, 114, 114))
    return np.ascontiguousarray(im)


def numpy_nms(boxes, scores, iou_thr):
    """Greedy NMS matching torchvision semantics (suppress IoU > thr);
    stable sort for fp16 conf ties."""
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


class YoloSamCoreML(YoloSam):
    """YoloSam with the dual-CoreML hot path in segment(); torch stack stays
    loaded for prompts/batch endpoints and FasterYoloSam component reuse."""

    def __init__(self, config, device: str = "cpu", components=None):
        super().__init__(config, device, components)
        if not coreml_available():
            raise RuntimeError("YoloSamCoreML requested but CoreML packages unavailable")
        import coremltools as ct

        t0 = time.perf_counter()
        self._cm_yolo = ct.models.MLModel(str(YOLO_PKG))
        self._cm_yolo.predict({"image": Image.fromarray(np.zeros((640, 640, 3), np.uint8))})
        logger.info(f"coreml yolo loaded+compiled in {time.perf_counter() - t0:.1f}s")
        t0 = time.perf_counter()
        self._cm_enc = ct.models.MLModel(str(ENC_PKG))
        self._cm_enc.predict({"image": np.zeros((1, 3, 1024, 1024), np.float32)})
        logger.info(f"coreml sam encoder loaded+compiled in {time.perf_counter() - t0:.1f}s")
        t0 = time.perf_counter()
        self._cm_dec = ct.models.MLModel(str(DEC_PKG))
        self._cm_dec.predict({
            "image_embeddings": np.zeros((1, 256, 64, 64), np.float32),
            "boxes": np.zeros((B64, 4), np.float32),
        })
        logger.info(f"coreml sam decoder loaded+compiled in {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------ coreml
    def _cm_yolo_detect(self, image_rgb: np.ndarray) -> np.ndarray:
        x = letterbox_rgb(image_rgb)
        raw = self._cm_yolo.predict({"image": Image.fromarray(x)})["var_1362"]
        pred = raw[0]
        h, w = image_rgb.shape[:2]
        boxes, cls = pred[:4].T, pred[4]
        # raw layout is xywh — convert before NMS (ultralytics xywh2xyxy)
        xywh = boxes
        boxes = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                          xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], 1)
        cand = np.where(cls > CONF)[0]
        boxes, cls = boxes[cand], cls[cand]
        sel = numpy_nms(boxes, cls, IOU)[:MAX_DET]
        b = boxes[sel].copy()
        gain = min(640 / h, 640 / w)
        # pad for scale_boxes: letterbox pads right/bottom by dw,dh (in 640 space)
        new_unpad = (round(w * gain), round(h * gain))
        pad_x = (640 - new_unpad[0]) / 2
        pad_y = (640 - new_unpad[1]) / 2
        b[:, [0, 2]] = (b[:, [0, 2]] - pad_x) / gain
        b[:, [1, 3]] = (b[:, [1, 3]] - pad_y) / gain
        np.clip(b[:, [0, 2]], 0, w, out=b[:, [0, 2]])
        np.clip(b[:, [1, 3]], 0, h, out=b[:, [1, 3]])
        return b

    def _cm_sam_preprocess(self, image_rgb: np.ndarray) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        scale = IMG_SIZE / max(h, w)
        new_w = int(w * scale + 0.5)
        new_h = int(h * scale + 0.5)
        rsz = np.array(
            Image.fromarray(image_rgb).resize((new_w, new_h), Image.BILINEAR)
        ).astype(np.float32)
        x = (rsz - PIXEL_MEAN) / PIXEL_STD
        x = x.transpose(2, 0, 1)[None]
        x = np.pad(x, ((0, 0), (0, 0), (0, IMG_SIZE - new_h), (0, IMG_SIZE - new_w)))
        return np.ascontiguousarray(x, dtype=np.float32)

    def _cm_transform_boxes(self, boxes: np.ndarray, orig_hw) -> np.ndarray:
        h, w = orig_hw
        scale = IMG_SIZE / max(h, w)
        new_w = int(w * scale + 0.5)
        new_h = int(h * scale + 0.5)
        b = boxes.astype(np.float64).copy()
        b[:, [0, 2]] *= new_w / w
        b[:, [1, 3]] *= new_h / h
        return b.astype(np.float32)

    def _cm_decode_union(self, emb: np.ndarray, boxes_t: np.ndarray,
                         orig_hw) -> tuple[np.ndarray, np.ndarray]:
        """Chunked (B=64, duplicate-padded) running max-logit union + owner
        labels. Per-chunk masks are resized in groups of 8 to bound memory."""
        h, w = orig_hw
        scale = IMG_SIZE / max(h, w)
        nw, nh = int(w * scale + 0.5), int(h * scale + 0.5)
        running_max: np.ndarray | None = None
        running_owner: np.ndarray | None = None
        for i in range(0, len(boxes_t), B64):
            chunk = boxes_t[i : i + B64]
            n = len(chunk)
            reps = (B64 + n - 1) // n
            padded = np.tile(chunk, (reps, 1))[:B64].astype(np.float32).copy()
            masks = self._cm_dec.predict(
                {"image_embeddings": emb, "boxes": padded}
            )["masks"]  # (64,1,1024,1024) logits
            chunk_max = None
            chunk_owner = None
            for j in range(0, B64, 8):
                group = masks[j : j + 8, 0, :nh, :nw]
                resized = np.stack(
                    [cv.resize(m, (w, h), interpolation=cv.INTER_LINEAR) for m in group]
                )
                if chunk_max is None:
                    chunk_max = resized.max(axis=0)
                    chunk_owner = resized.argmax(axis=0).astype(np.uint16)
                else:
                    gmax = resized.max(axis=0)
                    gowner = resized.argmax(axis=0).astype(np.uint16)
                    upd = gmax > chunk_max
                    chunk_max = np.maximum(chunk_max, gmax)
                    chunk_owner = np.where(upd, gowner + j, chunk_owner)
            # owner ids: 1-based, global across chunks (i is chunk start)
            if running_max is None:
                running_max = chunk_max
                running_owner = chunk_owner + i + 1
            else:
                upd = chunk_max > running_max
                running_max = np.maximum(running_max, chunk_max)
                running_owner = np.where(upd, chunk_owner + i + 1, running_owner)
        union = (running_max > 0.0).astype(np.uint8)
        owner = np.where(union > 0, running_owner, 0).astype(np.uint16)
        return union, owner

    # ----------------------------------------------------------- segment()
    def segment(
        self,
        image: np.ndarray,
        embedding_cache: SAMEmbedding | None = None,
        encoder_depth: int = 12,
        **kwargs,
    ) -> YoloSAMSegmentationResult:
        logger.info(
            f"[coreml] input image shape: {image.shape}, dtype: {image.dtype}"
        )
        # keep a torch predictor alive so /split & point-prompt endpoints work
        if not hasattr(self, "_predictor"):
            from segment_anything import SamPredictor

            self._predictor = SamPredictor(self.components["sam"])

        with Timer(logger, "yolo_coreml") as t_yolo:
            with t_yolo.step("detect"):
                boxes_np = self._cm_yolo_detect(image)
            t_yolo.field("boxes", len(boxes_np))

        with Timer(logger, "sam_coreml") as t_sam:
            if embedding_cache is not None:
                emb = embedding_cache.features
                if isinstance(emb, torch.Tensor):
                    emb = emb.detach().cpu().numpy()
                emb = np.ascontiguousarray(emb, dtype=np.float32)
                t_sam.field("encode", "cached")
            else:
                with t_sam.step("encode"):
                    x = self._cm_sam_preprocess(image)
                    emb = self._cm_enc.predict({"image": x})["image_embeddings"]
                    t_sam.field("depth", encoder_depth)

            # surface the embedding through the torch predictor for prompts
            self._predictor.features = torch.from_numpy(emb).to(self.device)
            self._predictor.original_size = image.shape[:2]
            self._predictor.input_size = (IMG_SIZE, IMG_SIZE)
            self._predictor.is_image_set = True

            if len(boxes_np) == 0:
                logger.info("[coreml] YOLO found 0 boxes — empty mask, embedding cached")
                return YoloSAMSegmentationResult(
                    segmentation_mask=np.zeros(image.shape[:2], dtype=np.uint8),
                    metadata={"detections": 0, "sam_embedding_cached": True},
                    model=self.MODEL_ENUM,
                    detection_boxes=np.array([]).reshape(0, 4),
                    embedding=SAMEmbedding(
                        features=self._predictor.features,
                        original_size=image.shape[:2],
                        input_size=(IMG_SIZE, IMG_SIZE),
                        encoder_depth=encoder_depth,
                    ),
                )

            with t_sam.step("decode"):
                boxes_t = self._cm_transform_boxes(boxes_np, image.shape[:2])
                combined_mask, owner_labels = self._cm_decode_union(
                    emb, boxes_t, image.shape[:2]
                )
            t_sam.field("mask_px", int(np.count_nonzero(combined_mask)))

        return YoloSAMSegmentationResult(
            segmentation_mask=combined_mask,
            metadata={"detections": len(boxes_np), "encoder_depth": encoder_depth,
                      "backend": "coreml"},
            model=self.MODEL_ENUM,
            embedding=SAMEmbedding(
                features=self._predictor.features,
                original_size=image.shape[:2],
                input_size=(IMG_SIZE, IMG_SIZE),
                encoder_depth=encoder_depth,
            ),
            detection_boxes=boxes_np,
            instance_labels=owner_labels,
        )
