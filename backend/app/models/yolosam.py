import cv2 as cv
import numpy as np
import torch
from typing import Dict, Any
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.backends.backend_agg import FigureCanvasAgg

from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

from base_model import Model, SegmentationResult, ModelConfig


class YoloSam(Model):
    def __init__(self, config: ModelConfig, device: str = "cpu"):
        self.device = device
        super().__init__(config)

    def _load_components(self) -> Dict[str, Any]:
        components = {}

        for comp in self.config.components:

            if comp.name.lower() == "yolo":
                print(f"Loading YOLO from {comp.path}")
                components["yolo"] = YOLO(comp.path)

            elif comp.name.lower() == "sam":
                print(f"Loading SAM from {comp.path}")
                sam = sam_model_registry["vit_b"](checkpoint=comp.path)
                sam.to(device=self.device)
                components["sam"] = sam

            else:
                raise ValueError(f"Unknown component: {comp.name}")

        return components

    def load_image(self, image_path: str) -> np.ndarray:
        img = cv.imread(image_path, cv.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
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
                metadata={"detections": 0}
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

        self.plot(image, combined_mask)

        return SegmentationResult(
            segmentation_mask=combined_mask,
            metadata={"detections": len(boxes)}
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
