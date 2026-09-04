"""YoloSam: YOLO detection + SAM segmentation, wired through model backends.

The pipeline logic lives here; the inference stacks (torch vs CoreML,
platform selection) live in app.models.backends. This class consumes the
YoloBackend/SamBackend interfaces only.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import cv2 as cv
import numpy as np
import torch
from fastapi import APIRouter

from app.api.live_models import AvailableModels
from app.logutils import Timer, get_logger
from app.models.backends.base import SamEmbedding
from app.models.backends.selection import choose_backends
from app.models.base_model import Model, ModelConfig, SegmentationResult

router = APIRouter(prefix="/models/yolosam")
logger = get_logger("YoloSAM")
log_batch = get_logger("YoloSAM", sub="Batch")
SESSIONS_DIR = Path("sessions")


@dataclass
class YoloSAMSegmentationResult(SegmentationResult):
    embedding: SamEmbedding | None = None
    detection_boxes: np.ndarray | None = None
    # uint16 ownership map: pixel value = 1-based index of the YOLO box whose
    # decode won that pixel (highest full-res logit), 0 = background.
    instance_labels: np.ndarray | None = None


class YoloSam(Model):
    # Which AvailableModels enum results report; FasterYoloSam overrides this
    # so the shared segment()/segment_batch() implementations stay reusable.
    MODEL_ENUM = AvailableModels.yolosam

    def __init__(
        self,
        config: ModelConfig,
        device: str = "cpu",
        components: Dict[str, Any] | None = None,
        _faster: bool = False,
    ):
        logger.info("Initializing YoloSam")
        self.device = device
        self._shared_components = components
        super().__init__(config)  # calls self._load_components()
        self._yolo, self._sam, self._prompt_sam = choose_backends(
            self.components, device, faster=_faster
        )

    @property
    def _predictor(self):
        """SamPredictor of a torch SAM backend, if one is loaded.

        Kept for the point-prompt endpoints in masks.py; None when no torch
        predictor exists."""
        src = self._prompt_sam if self._prompt_sam is not None else self._sam
        return getattr(src, "predictor", None)

    def _load_components(self) -> Dict[str, Any]:
        # Reuse components already loaded by another pipeline instance
        # Both FYS and YoloSAM share the same components
        if self._shared_components is not None:
            logger.info("Reusing shared model components")
            return self._shared_components

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
                from ultralytics import YOLO

                try:
                    model = YOLO(str(model_path))
                    logger.info(f"Loading YOLO model from {model_path.name}")
                except Exception as e:
                    raise RuntimeError(f"Failed to load YOLO: {e}") from e
                components["yolo"] = model

            elif name == "sam":
                from segment_anything import sam_model_registry

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
        self,
        image: np.ndarray,
        embedding_cache: SamEmbedding | None = None,
        encoder_depth: int = 12,
        **kwargs,
    ) -> YoloSAMSegmentationResult:
        logger.info(f"input image shape: {image.shape}, dtype: {image.dtype}")

        # YOLO Detection
        with Timer(logger, "yolo") as t_yolo:
            with t_yolo.step("predict"):
                boxes = self._yolo.detect(image)
            t_yolo.field("boxes", len(boxes))

        # Always prime SAM with the image so the embedding is available for
        # downstream point-prompt endpoints (/split, /from-points, /propose-similar)
        # even when YOLO found nothing.
        with Timer(logger, "sam") as t_sam:
            if embedding_cache:
                emb = embedding_cache
                t_sam.field("encode", "cached")
            else:
                with t_sam.step("encode"):
                    emb = self._sam.encode(image, encoder_depth)
                    t_sam.field("depth", encoder_depth)

            # point-prompt endpoints use the torch predictor; keep it in sync
            # with whichever backend produced the embedding
            if self._prompt_sam is not None and self._prompt_sam is not self._sam:
                self._prompt_sam.sync_embedding(emb)

            if len(boxes) == 0:
                logger.info(
                    "YOLO found 0 boxes — returning empty mask with SAM embedding cached"
                )
                return YoloSAMSegmentationResult(
                    segmentation_mask=np.zeros(image.shape[:2], dtype=np.uint8),
                    metadata={"detections": 0, "sam_embedding_cached": True},
                    model=self.MODEL_ENUM,
                    detection_boxes=np.array([]).reshape(0, 4),
                    embedding=emb,
                )

            box_batch_size = kwargs.get("box_batch_size", 64)

            with t_sam.step("decode"):
                combined_mask, owner_labels = self._sam.decode_union(
                    emb, boxes, box_batch_size
                )

            t_sam.field("mask_px", int(np.count_nonzero(combined_mask)))

        return YoloSAMSegmentationResult(
            segmentation_mask=combined_mask,
            metadata={"detections": len(boxes), "encoder_depth": encoder_depth},
            model=self.MODEL_ENUM,
            embedding=emb,
            detection_boxes=boxes,
            instance_labels=owner_labels,
        )

    def predict_from_prompts(self, prompts: list[dict]) -> np.ndarray:
        """
        Run SAM on a list of {point: [x,y], bbox: [x1,y1,x2,y2]} prompts using
        the predictor that was set during the most recent segment() call.
        Returns a binary uint8 mask (0/1) merged across all prompts.
        """
        return self._sam.predict_prompts(prompts)

    def segment_batch(
        self,
        patches: List[np.ndarray],
        offsets: List[tuple],
        img_shape: tuple,
        encoder_depth: int = 12,
    ) -> YoloSAMSegmentationResult:
        """
        Batch YOLO detection across all patches, then SAM per patch.
        offsets: [(x1,y1), ...] for stitching back
        encoder_depth truncates the SAM ViT-B encoder per patch (< 12).
        """
        h_full, w_full = img_shape
        combined = np.zeros((h_full, w_full), dtype="uint8")
        combined_labels = np.zeros((h_full, w_full), dtype=np.uint16)

        log_batch.info(f"segment_batch: {len(patches)} patches, output {img_shape}")

        total_detections = 0
        all_boxes: list[np.ndarray] = []
        label_offset = 0  # owner ids keep global box order across patches

        for i, (patch, (x1, y1)) in enumerate(zip(patches, offsets)):
            boxes_np = self._yolo.detect(patch)
            if len(boxes_np) == 0:
                log_batch.debug(f"patch {i}: no detections, skipping")
                continue
            log_batch.debug(f"patch {i}: {len(boxes_np)} detections")

            emb = self._sam.encode(patch, encoder_depth)
            patch_union, patch_labels = self._sam.decode_union(emb, boxes_np, 64)

            x2, y2 = x1 + patch.shape[1], y1 + patch.shape[0]
            combined[y1:y2, x1:x2] = np.maximum(
                combined[y1:y2, x1:x2], (patch_union > 0).astype("uint8") * 255
            )
            if patch_labels is not None:
                lbl = combined_labels[y1:y2, x1:x2]
                hit = patch_union > 0
                lbl[hit] = patch_labels[hit] + label_offset

            # Offset boxes to full-image coordinates
            boxes_np[:, [0, 2]] += x1
            boxes_np[:, [1, 3]] += y1
            all_boxes.append(boxes_np)

            label_offset += len(boxes_np)
            total_detections += len(boxes_np)
            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps":
                torch.mps.empty_cache()

        detection_boxes = (
            np.vstack(all_boxes) if all_boxes else np.array([]).reshape(0, 4)
        )

        return YoloSAMSegmentationResult(
            segmentation_mask=combined,
            metadata={
                "detections": total_detections,
                "encoder_depth": encoder_depth,
            },
            model=self.MODEL_ENUM,
            detection_boxes=detection_boxes,
            instance_labels=combined_labels if total_detections else None,
        )
