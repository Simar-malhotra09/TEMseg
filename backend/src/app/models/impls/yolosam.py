"""YoloSam: YOLO detection + SAM segmentation, wired through model backends.

The pipeline logic lives here; the inference stacks (torch vs CoreML,
platform selection) live in app.models.backends. This class consumes the
YoloBackend/SamBackend interfaces only.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from fastapi import APIRouter

from app.api.live_models import AvailableModels
from app.logutils import Timer, get_logger
from app.models.backends.base import SamEmbedding
from app.models.backends.selection import (
    choose_backends,
    coreml_available,
    device_allocated_bytes,
    process_peak_rss_bytes,
    ram_since,
)
from app.models.base_model import Model, ModelConfig, SegmentationResult
from app.models.helpers.image_io import load_rgb_uint8

router = APIRouter(prefix="/models/yolosam")
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
    USE_COREML_SAM = True

    def __init__(
        self,
        config: ModelConfig,
        device: str = "cpu",
        components: Dict[str, Any] | None = None,
        _faster: bool = False,
    ):
        self._logger = get_logger(self.MODEL_ENUM.value)
        self._batch_logger = get_logger(self.MODEL_ENUM.value, sub="Batch")
        self._logger.info("Initializing %s", self.MODEL_ENUM.value)
        self.device = device
        self._shared_components = components
        super().__init__(config)  # calls self._load_components()
        self._yolo, self._sam, self._prompt_sam = choose_backends(
            self.components,
            device,
            faster=_faster,
            use_coreml_sam=self.USE_COREML_SAM,
        )

    @property
    def _predictor(self):
        """SamPredictor of a torch SAM backend, if one is loaded.

        Kept for the point-prompt endpoints in masks.py; None when no torch
        predictor exists."""
        src = self._prompt_sam if self._prompt_sam is not None else self._sam
        return getattr(src, "predictor", None)

    def sync_prompt_embedding(self, emb: SamEmbedding) -> None:
        """Point the torch prompt predictor at a cached embedding (point-prompt
        endpoints restore state without re-encoding; emb.features is numpy,
        sync_embedding converts it to a device tensor)."""
        src = self._prompt_sam if self._prompt_sam is not None else self._sam
        src.sync_embedding(emb)

    def _load_components(self) -> Dict[str, Any]:
        if not hasattr(self, "config") or not hasattr(self.config, "components"):
            raise ValueError("Invalid config: missing 'components'")

        if not self.config.components:
            raise ValueError("Config contains no components")

        components: Dict[str, Any] = dict(self._shared_components or {})
        if components:
            self._logger.info("Reusing shared model components")

        for comp in self.config.components:
            if not hasattr(comp, "name") or not hasattr(comp, "path"):
                raise ValueError(f"Invalid component structure: {comp}")

            if not isinstance(comp.name, str):
                raise TypeError("Component name must be a string")

            name = comp.name.lower()
            if name in components:
                continue
            model_path = Path(comp.path)

            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            if not model_path.is_file():
                raise ValueError(f"Model path is not a file: {model_path}")

            if name == "yolo":
                from ultralytics import YOLO

                if coreml_available():
                    # the CoreML yolo backend serves detections; the onnx
                    # session would be loaded and never used
                    self._logger.info("CoreML yolo active — skipping ONNX yolo load")
                    continue
                try:
                    model = YOLO(str(model_path), task="detect")
                    self._logger.info("Loading YOLO model from %s", model_path.name)
                except Exception as e:
                    raise RuntimeError(f"Failed to load YOLO: {e}") from e
                components["yolo"] = model

            elif name == "sam":
                from segment_anything import sam_model_registry

                if "vit_b" not in sam_model_registry:
                    raise ValueError("SAM registry missing 'vit_b'")

                ram_before = (
                    process_peak_rss_bytes(),
                    device_allocated_bytes(self.device),
                )
                try:
                    sam = sam_model_registry["vit_b"](checkpoint=str(model_path))
                    self._logger.info("Loading SAM model from %s", model_path.name)

                except Exception as e:
                    raise RuntimeError(f"Failed to initialize SAM: {e}") from e

                try:
                    sam.to(device=self.device)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to move SAM to device {self.device}: {e}"
                    ) from e

                sam.ram_bytes = ram_since(ram_before, self.device)
                components["sam"] = sam

            elif name == "microsam_em":
                from segment_anything import sam_model_registry

                if "vit_b" not in sam_model_registry:
                    raise ValueError("SAM registry missing 'vit_b'")
                ram_before = (
                    process_peak_rss_bytes(),
                    device_allocated_bytes(self.device),
                )
                checkpoint = torch.load(
                    model_path, map_location="cpu", weights_only=False
                )
                if not isinstance(checkpoint, dict):
                    raise ValueError("micro-SAM EM checkpoint is not a dictionary")
                state = checkpoint.get("model_state", checkpoint)
                if not isinstance(state, dict):
                    raise ValueError("micro-SAM EM checkpoint has no state dictionary")
                state = {
                    key.removeprefix("sam."): value for key, value in state.items()
                }
                sam = sam_model_registry["vit_b"]()
                sam.load_state_dict(state)
                sam.to(device=self.device)
                sam.ram_bytes = ram_since(ram_before, self.device)
                components["sam"] = sam

            elif name == "mobile_sam":
                from mobile_sam import sam_model_registry

                ram_before = (
                    process_peak_rss_bytes(),
                    device_allocated_bytes(self.device),
                )
                sam = sam_model_registry["vit_t"](checkpoint=str(model_path))
                sam.to(device=self.device)
                sam.ram_bytes = ram_since(ram_before, self.device)
                components["sam"] = sam

            else:
                raise ValueError(f"Unknown component: {comp.name}")

        return components

    def load_image(self, image_path: Path) -> np.ndarray:
        return load_rgb_uint8(image_path)

    def segment(
        self,
        image: np.ndarray,
        embedding_cache: SamEmbedding | None = None,
        encoder_depth: int = 12,
        yolo_params: dict | None = None,
        **kwargs,
    ) -> YoloSAMSegmentationResult:
        self._logger.info("input image shape: %s, dtype: %s", image.shape, image.dtype)

        # YOLO Detection
        with Timer(self._logger, "yolo") as t_yolo:
            with t_yolo.step("predict"):
                boxes = self._yolo.detect(image, **(yolo_params or {}))
            t_yolo.field("boxes", len(boxes))

        # Always prime SAM with the image so the embedding is available for
        # downstream point-prompt endpoints (/split, /from-points, /propose-similar)
        # even when YOLO found nothing.
        with Timer(self._logger, "sam") as t_sam:
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
                self._logger.info(
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
        yolo_params: dict | None = None,
    ) -> YoloSAMSegmentationResult:
        """
        Batch YOLO detection across all patches, then SAM per patch.
        offsets: [(x1,y1), ...] for stitching back
        encoder_depth truncates the SAM ViT-B encoder per patch (< 12).
        """
        h_full, w_full = img_shape
        combined = np.zeros((h_full, w_full), dtype="uint8")
        combined_labels = np.zeros((h_full, w_full), dtype=np.uint16)

        self._batch_logger.info(
            "segment_batch: %s patches, output %s", len(patches), img_shape
        )

        total_detections = 0
        all_boxes: list[np.ndarray] = []
        label_offset = 0  # owner ids keep global box order across patches

        for i, (patch, (x1, y1)) in enumerate(zip(patches, offsets)):
            boxes_np = self._yolo.detect(patch, **(yolo_params or {}))
            if len(boxes_np) == 0:
                self._batch_logger.debug("patch %s: no detections, skipping", i)
                continue
            self._batch_logger.debug("patch %s: %s detections", i, len(boxes_np))

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
