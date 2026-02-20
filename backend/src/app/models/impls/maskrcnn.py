import cv2 as cv
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from typing import Dict, Any
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.backends.backend_agg import FigureCanvasAgg
from app.models.base_model import Model, ModelConfig, SegmentationResult
from app.models.helpers.maskrcnn_utils import group_boxes_and_masks

class MaskRCNN(Model):
    def __init__(self, config: ModelConfig, device: str = "cpu"):
        self.device = device
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
                raise RuntimeError(f"Failed to move model to device {self.device}: {e}") from e

            model.eval()

            return {"maskrcnn": model}

        raise ValueError("No 'maskrcnn' component found in config")

    def _build_model(self, num_classes=2):
        model = maskrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
        return model

    def load_image(self, image_path: str) -> np.ndarray:
        img = cv.imread(str(image_path), cv.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        return img.astype(np.float32) / 255.0

    def segment(self, image: np.ndarray) -> SegmentationResult:
        model = self.components["maskrcnn"]
        tensor = torch.from_numpy(image).permute(2, 0, 1).to(self.device)

        with torch.no_grad():
            prediction = model([tensor])[0]

        keep = prediction['scores'] > 0.95
        results = {
            'boxes': prediction['boxes'][keep].cpu().numpy(),
            'scores': prediction['scores'][keep].cpu().numpy(),
            'masks': prediction['masks'][keep].cpu().numpy(),
        }

        results = group_boxes_and_masks(results)

        if len(results['masks']) == 0:
            combined = np.zeros(image.shape[:2], dtype="uint8")
        else:
            combined = (np.max(results['masks'].squeeze(1), axis=0) > 0.5).astype("uint8")

        return SegmentationResult(
            segmentation_mask=combined,
            metadata={"detections": len(results['boxes'])}
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
        ax.imshow(np.ma.masked_where(mask_to_plot == 0, mask_to_plot),
                  cmap='nipy_spectral', alpha=0.5)

        ax.axis("off")
        plt.tight_layout(pad=0)
        plt.show()
