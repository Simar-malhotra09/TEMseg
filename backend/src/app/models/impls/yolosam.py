import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
from fastapi import APIRouter
from segment_anything import SamPredictor, sam_model_registry
from ultralytics import YOLO

from app.api.live_models import AvailableModels
from app.models.base_model import Model, ModelConfig, SegmentationResult

router = APIRouter(prefix="/models/yolosam")
logger = logging.getLogger("routes.models.yolosam")
SESSIONS_DIR = Path("sessions")


@dataclass
class SAMEmbedding:
    features: torch.Tensor
    original_size: tuple[int, int]
    input_size: tuple[int, int]


@dataclass
class YoloSAMSegmentationResult(SegmentationResult):
    embedding: SAMEmbedding | None = None


class YoloSam(Model):
    def __init__(self, config: ModelConfig, device: str = "cpu"):
        logger.info("[YoloSAM] Initializing YoloSam")
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
                    print(f"[YOLOSAM] Loading model_path:{model_path}")
                except Exception as e:
                    raise RuntimeError(f"Failed to load YOLO: {e}") from e
                components["yolo"] = model

            elif name == "sam":
                if "vit_b" not in sam_model_registry:
                    raise ValueError("SAM registry missing 'vit_b'")

                try:
                    sam = sam_model_registry["vit_b"](checkpoint=str(model_path))
                    print(f"[YOLOSAM] Loading model_path:{model_path}")

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

    def load_image(self, image_path: Path) -> np.ndarray:
        if image_path.suffix == ".npy":
            img = np.load(image_path)
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
        logger.info(f"[YoloSAM] input image shape: {image.shape}, dtype: {image.dtype}")

        # ensure predictor exists — created once, reused across calls
        if not hasattr(self, "_predictor"):
            self._predictor = SamPredictor(self.components["sam"])
        predictor = self._predictor

        # --- YOLO Detection ---
        t0 = time.perf_counter()
        results = self.components["yolo"].predict(
            source=image, conf=0.25, iou=0.5, max_det=4000, verbose=False
        )
        t1 = time.perf_counter()
        boxes = results[0].boxes.xyxy
        logger.info(f"[YoloSAM-Yolo] predict={t1 - t0:.3f}s | boxes={len(boxes)}")

        t2 = time.perf_counter()
        input_boxes = boxes.to(predictor.device)
        transformed_boxes = predictor.transform.apply_boxes_torch(
            input_boxes, image.shape[:2]
        )
        t3 = time.perf_counter()
        logger.info(f"[YoloSAM-Yolo] box_transfer={t3 - t2:.3f}s")

        yolo_time_elapsed = t3 - t0
        logger.info(f"[YoloSAM-Yolo] detected {len(boxes)} boxes")

        if boxes is None or len(boxes) == 0:
            return YoloSAMSegmentationResult(
                segmentation_mask=np.zeros(image.shape[:2], dtype=np.uint8),
                metadata={"detections": 0},
                model=AvailableModels.yolosam,
            )

        # --- SAM Segmentation ---
        sam_start = time.perf_counter()

        if embedding_cache:
            predictor.features = embedding_cache.features
            predictor.original_size = embedding_cache.original_size
            predictor.input_size = embedding_cache.input_size
            predictor.is_image_set = True
            logger.info("[YoloSAM-SAM] Using cached SAM embedding")
        else:
            predictor.set_image(image)
            logger.info("[YoloSAM-SAM] Encoding image with SAM")

        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        masks_np = masks.cpu().numpy().astype("uint8")
        combined_mask = np.max(masks_np, axis=0)

        sam_time_elapsed = time.perf_counter() - sam_start

        logger.info(
            f"[YoloSAM] output mask shape: {combined_mask.shape}, input was: {image.shape[:2]}"
        )
        logger.info(f"[YoloSAM] Yolo took {yolo_time_elapsed:.4f}s")
        logger.info(f"[YoloSAM] SAM took {sam_time_elapsed:.4f}s")

        return YoloSAMSegmentationResult(
            segmentation_mask=combined_mask,
            metadata={"detections": len(boxes)},
            model=AvailableModels.yolosam,
            embedding=SAMEmbedding(
                features=predictor.features,
                original_size=predictor.original_size,
                input_size=predictor.input_size,
            ),
        )

    def segment_batch(
        self, patches: List[np.ndarray], offsets: List[tuple], img_shape: tuple
    ) -> YoloSAMSegmentationResult:
        """
        Batch YOLO detection across all patches, then SAM per patch.
        offsets: [(x1,y1), ...] for stitching back
        """
        h_full, w_full = img_shape
        combined = np.zeros((h_full, w_full), dtype="uint8")

        logger.info(
            f"Starting batch segmentation: {len(patches)} patches, output size {img_shape}"
        )

        # batch YOLO — one call for all patches
        yolo_model = self.components["yolo"]
        logger.info("Running YOLO batch detection")
        all_results = yolo_model.predict(
            source=patches, conf=0.25, iou=0.5, max_det=4000
        )

        sam_model = self.components["sam"]
        predictor = SamPredictor(sam_model)

        total_detections = 0

        for i, (patch, result, (x1, y1)) in enumerate(
            zip(patches, all_results, offsets)
        ):
            boxes = result.boxes.xyxy

            if boxes is None or len(boxes) == 0:
                logger.info(f"Patch {i}: no detections")
                continue

            logger.info(f"Patch {i}: {len(boxes)} detections")

            # SAM still per-patch but YOLO was batched
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

            patch_mask = masks.cpu().numpy().astype("uint8")
            patch_mask = np.max(patch_mask, axis=0).squeeze()

            x2, y2 = x1 + patch.shape[1], y1 + patch.shape[0]

            combined[y1:y2, x1:x2] = np.maximum(
                combined[y1:y2, x1:x2], (patch_mask > 0).astype("uint8") * 255
            )

            total_detections += len(boxes)

        logger.info(f"Segmentation complete: {total_detections} total detections")

        return YoloSAMSegmentationResult(
            segmentation_mask=combined,
            metadata={
                "detections": total_detections,
                # "yolo_time_elapsed": yolo_time_elapsed,
                # "sam_time_elapsed": sam_time_elapsed
            },
            model=AvailableModels.yolosam,
        )

    def plot(self, image, combined_mask):
        if combined_mask is None:
            print("Warning: No masks to visualize.")
            return None

        # squeeze extra dimensions
        mask_to_plot = combined_mask.squeeze()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image)  # show original image

        # overlay mask
        ax.imshow(
            np.ma.masked_where(mask_to_plot == 0, mask_to_plot),
            cmap="nipy_spectral",
            alpha=0.5,
        )

        ax.axis("off")
        plt.tight_layout(pad=0)
        plt.show()
