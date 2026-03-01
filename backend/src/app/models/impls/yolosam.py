import cv2 as cv
import numpy as np
import torch
from typing import Dict, Any
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pathlib import Path 
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

from app.models.base_model import Model, SegmentationResult, ModelConfig


class YoloSam(Model):
    def __init__(self, config: ModelConfig, device: str = "cpu"):
        self.device = device
        super().__init__(config)

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
                except Exception as e:
                    raise RuntimeError(f"Failed to load YOLO: {e}") from e
                components["yolo"] = model

            elif name == "sam":
                if "vit_b" not in sam_model_registry:
                    raise ValueError("SAM registry missing 'vit_b'")

                try:
                    sam = sam_model_registry["vit_b"](checkpoint=str(model_path))
                except Exception as e:
                    raise RuntimeError(f"Failed to initialize SAM: {e}") from e

                try:
                    sam.to(device=self.device)
                except Exception as e:
                    raise RuntimeError(f"Failed to move SAM to device {self.device}: {e}") from e

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
            img = np.stack([img] * 3, axis=-1)       # grayscale → RGB
        elif img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))        # (C,H,W) → (H,W,C)
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)           # (H,W,1) → (H,W,3)
        elif img.shape[2] == 4:
            img = img[:, :, :3]                       # drop alpha

        if img.dtype != np.uint8:
            img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype("uint8")

        return img

    def segment(self, image: np.ndarray) -> SegmentationResult:

        # --- YOLO Detection ---
        yolo_model = self.components["yolo"]

        results = yolo_model.predict(
            source=image,
            conf=0.25,
            iou=0.5,
            max_det=4000
        )

        boxes = results[0].boxes.xyxy
        if boxes is None or len(boxes) == 0:
            return SegmentationResult(
                segmentation_mask=np.zeros(image.shape[:2], dtype=np.uint8),
                metadata={"detections": 0},
                model= "YoloSAM"
            )

        # --- SAM Segmentation ---
        sam_model = self.components["sam"]
        predictor = SamPredictor(sam_model)

        predictor.set_image(image)

        input_boxes = boxes.to(predictor.device)
        transformed_boxes = predictor.transform.apply_boxes_torch(
            input_boxes, image.shape[:2]
        )

        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        masks_np = masks.cpu().numpy().astype("uint8")

        # Combine masks into single mask
        combined_mask = np.max(masks_np, axis=0)

        # self.plot(image, combined_mask)

        return SegmentationResult(
            segmentation_mask=combined_mask,
            metadata={"detections": len(boxes)},
            model="YoloSAM"
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
