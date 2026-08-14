import cv2 as cv
import numpy as np
import torch
import logging
from fastapi import APIRouter
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from typing import Dict, Any
from pathlib import Path
from app.api.live_models import AvailableModels
from app.models.base_model import Model, ModelConfig, SegmentationResult
from app.models.helpers.maskrcnn_utils import group_boxes_and_masks

router = APIRouter(prefix="/models/maskrcnn")
logger = logging.getLogger("routes.models.maskrcnn")
SESSIONS_DIR = Path("sessions")


class MaskRCNN(Model):
    def __init__(
        self, config: ModelConfig, model_id: AvailableModels, device: str = "cpu"
    ):
        self.device = device
        self.model_id = model_id
        super().__init__(config)

    def _load_components(self) -> Dict[str, Any]:
        if not hasattr(self, "config") or not hasattr(self.config, "components"):
            raise ValueError("Invalid config: missing 'components' attribute")

        if not self.config.components:
            raise ValueError("Config contains no components")

        for comp in self.config.components:
            if not hasattr(comp, "name") or not hasattr(comp, "path"):
                raise ValueError(f"Invalid component structure: {comp}")

            if not isinstance(comp.name, str):
                raise TypeError("Component name must be a string")

            if comp.name.lower() != "maskrcnn":
                continue

            model_path = Path(comp.path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            if not model_path.is_file():
                raise ValueError(f"Model path is not a file: {model_path}")

            model = self._build_model()
            if model is None:
                raise RuntimeError("_build_model() returned None")

            try:
                checkpoint = torch.load(model_path, map_location=self.device)
            except Exception as e:
                raise RuntimeError(f"Failed to load checkpoint: {e}") from e

            if not isinstance(checkpoint, dict):
                raise ValueError("Checkpoint is not a valid state_dict")

            try:
                model.load_state_dict(checkpoint)
            except Exception as e:
                raise RuntimeError(f"State dict mismatch: {e}") from e

            try:
                model.to(self.device)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to move model to device {self.device}: {e}"
                ) from e

            model.eval()

            return {"maskrcnn": model}

        raise ValueError("No 'maskrcnn' component found in config")

    def _build_model(self, num_classes=2):
        model = maskrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, 256, num_classes
        )
        return model

    def load_image(self, image_path: Path) -> np.ndarray:
        if image_path.suffix == ".npy":
            img = np.load(image_path)
        elif image_path.suffix == ".emd":
            import hyperspy.api as hs

            result = hs.load(str(image_path))
            s = result[0] if isinstance(result, list) else result
            img = s.data

        else:
            img = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to load image: {image_path}")
            img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

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

        return img.astype(np.float32) / 255.0

    # def load_image(self, image_path: str) -> np.ndarray:
    #     img = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
    #     if img is None:
    #         raise ValueError(f"Failed to load image: {image_path}")
    #     img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    #     return img.astype(np.float32) / 255.0

    def segment(self, image: np.ndarray) -> SegmentationResult:
        model = self.components["maskrcnn"]
        tensor = torch.from_numpy(image).permute(2, 0, 1).to(self.device)

        with torch.no_grad():
            prediction = model([tensor])[0]

        keep = prediction["scores"] > 0.95
        results = {
            "boxes": prediction["boxes"][keep].cpu().numpy(),
            "scores": prediction["scores"][keep].cpu().numpy(),
            "masks": prediction["masks"][keep].cpu().numpy(),
        }

        results = group_boxes_and_masks(results)

        if len(results["masks"]) == 0:
            combined = np.zeros(image.shape[:2], dtype="uint8")
        else:
            combined = (np.max(results["masks"].squeeze(1), axis=0) > 0.5).astype(
                "uint8"
            )

        return SegmentationResult(
            segmentation_mask=combined,
            metadata={"detections": len(results["boxes"])},
            model=self.model_id,
        )
