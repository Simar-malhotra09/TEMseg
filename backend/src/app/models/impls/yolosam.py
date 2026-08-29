import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import cv2 as cv
import numpy as np
import torch
from fastapi import APIRouter
from segment_anything import SamPredictor, sam_model_registry
from ultralytics import YOLO

from app.api.live_models import AvailableModels
from app.logutils import Timer, fmt_duration, get_logger
from app.models.base_model import Model, ModelConfig, SegmentationResult

router = APIRouter(prefix="/models/yolosam")
logger = get_logger("YoloSAM")
log_batch = get_logger("YoloSAM", sub="Batch")
SESSIONS_DIR = Path("sessions")


@dataclass
class SAMEmbedding:
    features: torch.Tensor
    original_size: tuple[int, int]
    input_size: tuple[int, int]


@dataclass
class YoloSAMSegmentationResult(SegmentationResult):
    embedding: SAMEmbedding | None = None
    detection_boxes: np.ndarray | None = None


class YoloSam(Model):
    def __init__(self, config: ModelConfig, device: str = "cpu"):
        logger.info("Initializing YoloSam")
        self.device = device
        super().__init__(config)  # calls self._load_components()

    def _load_components(self) -> Dict[str, Any]:
        if not hasattr(self, "config") or not hasattr(self.config, "components"):
            raise ValueError("Invalid config: missing 'components'")

        if not self.config.components:
            raise ValueError("Config contains no components")

        components: Dict[str, Any] = {}

        for comp in self.config.components:
            if not hasattr(comp, "name") or not hasattr(comp, "path"):
                raise ValueError(f"Invalid component structure: {comp}")

            if not isinstance(comp.name, str):
                raise TypeError("Component name must be a string")

            name = comp.name.lower()
            model_path = Path(comp.path)

            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            if not model_path.is_file():
                raise ValueError(f"Model path is not a file: {model_path}")

            if name == "yolo":
                try:
                    model = YOLO(str(model_path))
                    logger.info(f"Loading YOLO model from {model_path.name}")
                except Exception as e:
                    raise RuntimeError(f"Failed to load YOLO: {e}") from e
                components["yolo"] = model

            elif name == "sam":
                if "vit_b" not in sam_model_registry:
                    raise ValueError("SAM registry missing 'vit_b'")

                try:
                    sam = sam_model_registry["vit_b"](checkpoint=str(model_path))
                    logger.info(f"Loading SAM model from {model_path.name}")

                except Exception as e:
                    raise RuntimeError(f"Failed to initialize SAM: {e}") from e

                try:
                    sam.to(device=self.device)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to move SAM to device {self.device}: {e}"
                    ) from e

                components["sam"] = sam

            else:
                raise ValueError(f"Unknown component: {comp.name}")

        return components

    def _empty_device_cache(self) -> None:
        """Release cached allocator memory back to the OS after a forward pass.

        SAM's ViT-B encoder + per-box mask decoding hits a high memory
        watermark; PyTorch's caching allocator holds onto that peak until
        told to release it. Without this, RSS/footprint stays elevated
        indefinitely after the first segment() call.
        """
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

    def load_image(self, image_path: Path) -> np.ndarray:
        if image_path.suffix == ".npy":
            img = np.load(image_path)
        elif image_path.suffix == ".emd":
            import hyperspy.api as hs

            result = hs.load(str(image_path))
            s = result[0] if isinstance(result, list) else result
            img = s.data

        else:
            img = cv.imread(str(image_path), cv.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # normalize to (H, W, 3) uint8
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)  # grayscale → RGB
        elif img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))  # (C,H,W) → (H,W,C)
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)  # (H,W,1) → (H,W,3)
        elif img.shape[2] == 4:
            img = img[:, :, :3]  # drop alpha

        if img.dtype != np.uint8:
            img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(
                "uint8"
            )

        return img

    def segment(
        self, image: np.ndarray, embedding_cache: SAMEmbedding | None = None, **kwargs
    ) -> YoloSAMSegmentationResult:
        logger.info(f"input image shape: {image.shape}, dtype: {image.dtype}")

        # ensure predictor exists for reuse
        if not hasattr(self, "_predictor"):
            self._predictor = SamPredictor(self.components["sam"])
        predictor = self._predictor

        # --- YOLO Detection ---
        # we want to return these boxes
        # and overlay on the image
        # we can see point of failure,
        with Timer(logger, "yolo") as t_yolo:
            with t_yolo.step("predict"):
                results = self.components["yolo"].predict(
                    source=image,
                    conf=0.25,
                    iou=0.5,
                    max_det=4000,
                    verbose=False,
                    device=self.device,
                )
                boxes = results[0].boxes.xyxy

            with t_yolo.step("box_transfer"):
                input_boxes = boxes.to(predictor.device)
                transformed_boxes = predictor.transform.apply_boxes_torch(
                    input_boxes, image.shape[:2]
                )

            t_yolo.field("boxes", len(boxes))

        # Always prime SAM with the image so the embedding is available for
        # downstream point-prompt endpoints (/split, /from-points, /propose-similar)
        # even when YOLO found nothing. Without this, zero-detection images leave
        # the user with no way to manually bootstrap segmentation.
        with Timer(logger, "sam") as t_sam:
            if embedding_cache:
                predictor.features = embedding_cache.features
                predictor.original_size = embedding_cache.original_size
                predictor.input_size = embedding_cache.input_size
                predictor.is_image_set = True
                t_sam.field("encode", "cached")
            else:
                with t_sam.step("encode"):
                    predictor.set_image(image)

            if boxes is None or len(boxes) == 0:
                logger.info(
                    "YOLO found 0 boxes — returning empty mask with SAM embedding cached"
                )
                self._empty_device_cache()
                return YoloSAMSegmentationResult(
                    segmentation_mask=np.zeros(image.shape[:2], dtype=np.uint8),
                    metadata={"detections": 0, "sam_embedding_cached": True},
                    model=AvailableModels.yolosam,
                    detection_boxes=np.array([]).reshape(0, 4),
                    embedding=SAMEmbedding(
                        features=predictor.features,
                        original_size=predictor.original_size,
                        input_size=predictor.input_size,
                    ),
                )

            # Batch boxes to avoid RAM spike on images with many detections.
            # predict_torch allocates (N, 1, H, W) masks; thousands of boxes
            # can exhaust system RAM. We chunk and reduce incrementally.
            box_batch_size = kwargs.get("box_batch_size", 64)
            n_boxes = len(transformed_boxes)

            with t_sam.step("decode"):
                if n_boxes <= box_batch_size:
                    masks, _, _ = predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes,
                        multimask_output=False,
                    )
                    masks_np = masks.cpu().numpy().astype("uint8")
                    combined_mask = np.max(masks_np, axis=0)
                else:
                    logger.info(
                        f"Batching {n_boxes} boxes in chunks of {box_batch_size}"
                    )
                    combined_mask = np.zeros(image.shape[:2], dtype="uint8")
                    for i in range(0, n_boxes, box_batch_size):
                        batch = transformed_boxes[i : i + box_batch_size]
                        masks, _, _ = predictor.predict_torch(
                            point_coords=None,
                            point_labels=None,
                            boxes=batch,
                            multimask_output=False,
                        )
                        batch_np = masks.cpu().numpy().astype("uint8")
                        batch_max = np.max(batch_np, axis=0)
                        combined_mask = np.maximum(combined_mask, batch_max)
                        del masks, batch_np, batch_max
                        self._empty_device_cache()

            t_sam.field("mask_px", int(np.count_nonzero(combined_mask)))

        self._empty_device_cache()

        return YoloSAMSegmentationResult(
            segmentation_mask=combined_mask,
            metadata={"detections": len(boxes)},
            model=AvailableModels.yolosam,
            embedding=SAMEmbedding(
                features=predictor.features,
                original_size=predictor.original_size,
                input_size=predictor.input_size,
            ),
            detection_boxes=boxes.cpu().numpy(),
        )

    def predict_from_prompts(self, prompts: list[dict]) -> np.ndarray:
        """
        Run SAM on a list of {point: [x,y], bbox: [x1,y1,x2,y2]} prompts using
        the predictor that was set during the most recent segment() call.
        Returns a binary uint8 mask (0/1) merged across all prompts.
        """
        if not hasattr(self, "_predictor") or not self._predictor.is_image_set:
            raise RuntimeError("SAM predictor has no image — call segment() first")

        predictor = self._predictor
        h, w = predictor.original_size
        combined = np.zeros((h, w), dtype=np.uint8)

        for prompt in prompts:
            cx, cy = prompt["point"]
            x1, y1, x2, y2 = prompt["bbox"]

            point_coords = torch.tensor(
                [[[cx, cy]]], dtype=torch.float32, device=predictor.device
            )  # (1, 1, 2)
            point_labels = torch.ones((1, 1), dtype=torch.int, device=predictor.device)
            box_tensor = torch.tensor(
                [[x1, y1, x2, y2]], dtype=torch.float32, device=predictor.device
            )
            transformed_box = predictor.transform.apply_boxes_torch(
                box_tensor, predictor.original_size
            )

            masks, _, _ = predictor.predict_torch(
                point_coords=point_coords,
                point_labels=point_labels,
                boxes=transformed_box,
                multimask_output=False,
            )
            mask_np = masks.cpu().numpy().astype(np.uint8).squeeze()
            combined = np.maximum(combined, mask_np)

        return combined

    def segment_batch(
        self, patches: List[np.ndarray], offsets: List[tuple], img_shape: tuple
    ) -> YoloSAMSegmentationResult:
        """
        Batch YOLO detection across all patches, then SAM per patch.
        offsets: [(x1,y1), ...] for stitching back
        """
        h_full, w_full = img_shape
        combined = np.zeros((h_full, w_full), dtype="uint8")

        log_batch.info(f"segment_batch: {len(patches)} patches, output {img_shape}")

        # --- Batch YOLO detection ---
        yolo_model = self.components["yolo"]

        if not hasattr(self, "_predictor"):
            self._predictor = SamPredictor(self.components["sam"])
        predictor = self._predictor

        total_detections = 0
        t_yolo_total = 0.0
        t_sam_total = 0.0
        all_boxes: list[np.ndarray] = []

        with Timer(log_batch, "segment_batch") as t_batch:
            for i, (patch, (x1, y1)) in enumerate(zip(patches, offsets)):
                # YOLO per patch (ONNX model is batch=1)
                t_yolo_start = time.perf_counter()
                results = yolo_model.predict(
                    source=patch,
                    conf=0.25,
                    iou=0.5,
                    max_det=4000,
                    verbose=False,
                    device=self.device,
                )
                t_yolo_patch = time.perf_counter() - t_yolo_start
                t_yolo_total += t_yolo_patch

                boxes = results[0].boxes.xyxy

                if boxes is None or len(boxes) == 0:
                    log_batch.debug(f"patch {i}: no detections, skipping")
                    continue

                log_batch.debug(
                    f"patch {i}: {len(boxes)} detections, yolo={fmt_duration(t_yolo_patch)}"
                )

                # SAM per patch
                t_sam_start = time.perf_counter()
                predictor.set_image(patch)
                input_boxes = boxes.to(predictor.device)
                transformed_boxes = predictor.transform.apply_boxes_torch(
                    input_boxes, patch.shape[:2]
                )

                masks, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
                t_sam_patch = time.perf_counter() - t_sam_start
                t_sam_total += t_sam_patch
                log_batch.debug(f"patch {i}: sam={fmt_duration(t_sam_patch)}")

                patch_mask = masks.cpu().numpy().astype("uint8")
                patch_mask = np.max(patch_mask, axis=0).squeeze()

                x2, y2 = x1 + patch.shape[1], y1 + patch.shape[0]
                combined[y1:y2, x1:x2] = np.maximum(
                    combined[y1:y2, x1:x2], (patch_mask > 0).astype("uint8") * 255
                )

                # Offset boxes to full-image coordinates
                boxes_np = boxes.cpu().numpy()
                boxes_np[:, [0, 2]] += x1
                boxes_np[:, [1, 3]] += y1
                all_boxes.append(boxes_np)

                total_detections += len(boxes)
                self._empty_device_cache()

            t_batch.field("yolo", fmt_duration(t_yolo_total))
            t_batch.field("sam", fmt_duration(t_sam_total))
            t_batch.field("detections", total_detections)

        detection_boxes = (
            np.vstack(all_boxes) if all_boxes else np.array([]).reshape(0, 4)
        )

        return YoloSAMSegmentationResult(
            segmentation_mask=combined,
            metadata={
                "detections": total_detections,
                "yolo_time": round(t_yolo_total, 3),
                "sam_time": round(t_sam_total, 3),
            },
            model=AvailableModels.yolosam,
            detection_boxes=detection_boxes,
        )
