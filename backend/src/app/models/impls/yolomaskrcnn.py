import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

import cv2 as cv
import numpy as np
import torch
from fastapi import APIRouter
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.roi_heads import (
    maskrcnn_inference,
    paste_masks_in_image,
)
from torchvision.models.detection.transform import resize_boxes
from ultralytics import YOLO

from app.api.live_models import AvailableModels
from app.models.base_model import Model, ModelConfig, SegmentationResult

router = APIRouter(prefix="/models/yolomaskrcnn")
logger = logging.getLogger("routes.models.yolomaskrcnn")


class YoloMaskRCNN(Model):
    """YOLO for detection + Mask R-CNN mask head for segmentation.

    YOLO's boxes are used as the region proposals (unmodified), and only the
    Mask R-CNN backbone + mask head are run to produce the masks. The RPN and
    box head are never used at inference time.
    """

    def __init__(
        self, config: ModelConfig, model_id: AvailableModels, device: str = "cpu"
    ):
        self.device = device
        self.model_id = model_id
        super().__init__(config)

    def _load_components(self) -> Dict[str, Any]:
        if not hasattr(self, "config") or not self.config.components:
            raise ValueError("Invalid config: missing 'components'")

        components: Dict[str, Any] = {}

        for comp in self.config.components:
            if not hasattr(comp, "name") or not hasattr(comp, "path"):
                raise ValueError(f"Invalid component structure: {comp}")

            name = comp.name.lower()
            model_path = Path(comp.path)

            if not model_path.exists() or not model_path.is_file():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            if name == "yolo":
                components["yolo"] = YOLO(str(model_path), task="detect")
            elif name == "maskrcnn":
                model = self._build_model()
                try:
                    checkpoint = torch.load(
                        model_path, map_location="cpu", weights_only=False
                    )
                except Exception as e:
                    raise RuntimeError(f"Failed to load checkpoint: {e}") from e

                if not isinstance(checkpoint, dict):
                    raise ValueError("Checkpoint is not a valid state_dict")

                model.load_state_dict(checkpoint)
                model.to(self.device)
                model.eval()
                components["maskrcnn"] = model

        if "yolo" not in components or "maskrcnn" not in components:
            raise ValueError(
                "YoloMaskRCNN config must contain 'yolo' and 'maskrcnn' components"
            )
        return components

    def _build_model(self, num_classes: int = 2) -> torch.nn.Module:
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

        # normalize to (H, W, 3) uint8
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        if img.dtype != np.uint8:
            img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(
                "uint8"
            )
        return img

    def segment(self, image: np.ndarray) -> SegmentationResult:
        yolo = self.components["yolo"]
        model = self.components["maskrcnn"]

        results = yolo.predict(
            source=image,
            conf=0.25,
            iou=0.5,
            max_det=4000,
            verbose=False,
            device=self.device,
        )
        boxes = results[0].boxes.xyxy.cpu().numpy()

        h, w = image.shape[:2]
        if len(boxes) == 0:
            return SegmentationResult(
                segmentation_mask=np.zeros((h, w), dtype="uint8"),
                metadata={"detections": 0},
                model=self.model_id,
            )

        tensor = (
            torch.from_numpy(image.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .to(self.device)
        )
        yb = torch.as_tensor(boxes, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            images, _ = model.transform([tensor], None)
            features = model.backbone(images.tensors)
            if isinstance(features, torch.Tensor):
                features = OrderedDict([("0", features)])

            resized_h, resized_w = images.image_sizes[0]
            proposals = resize_boxes(yb, (h, w), (resized_h, resized_w))

            # mask head only, on YOLO's unmodified boxes
            mask_features = model.roi_heads.mask_roi_pool(
                features, [proposals], images.image_sizes
            )
            mask_features = model.roi_heads.mask_head(mask_features)
            mask_logits = model.roi_heads.mask_predictor(mask_features)

            labels = [torch.ones(len(boxes), dtype=torch.int64, device=self.device)]
            mask_prob = maskrcnn_inference(mask_logits, labels)[0]
            masks_in_orig = paste_masks_in_image(mask_prob, yb, (h, w))

        combined = (masks_in_orig.cpu().numpy()[:, 0] > 0.5).astype("uint8")
        combined = combined.max(axis=0)

        return SegmentationResult(
            segmentation_mask=combined,
            metadata={"detections": len(boxes)},
            model=self.model_id,
        )
