import numpy as np
from pydantic import BaseModel
import cv2 as cv
from typing import List
from pathlib import Path 
from typing import List 
import logging
from fastapi import APIRouter

from app.models.base_model import SegmentationResult
 
router = APIRouter(prefix="/utils")
logger = logging.getLogger("routes.utils")
SESSIONS_DIR = Path("sessions")

class Box(BaseModel):
    id: str
    x: float
    y: float
    width: float
    height: float

def normalize_mask(mask) -> np.ndarray:
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    elif mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask.squeeze(-1)

    assert mask.ndim == 2, f"Unexpected mask shape: {mask.shape}"
    return (mask > 0).astype("uint8") * 255


def blackout_regions(img: np.ndarray, regions: List[Box], save_path:Path | str) -> np.ndarray:
    """
    Black out rectangular regions in an image.
    Optionally saves the result for verification.
    """
    img_out = img.copy()
    h, w = img_out.shape[:2]

    for box in regions:
        x1 = max(0, min(int(box.x), w))
        x2 = max(0, min(int(box.x + box.width), w))
        y1 = max(0, min(int(box.y), h))
        y2 = max(0, min(int(box.y + box.height), h))

        img_out[y1:y2, x1:x2] = 0

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_img = (img_out * 255).astype("uint8") if img_out.dtype == np.float32 else img_out
        cv.imwrite(str(save_path), save_img)
        print(f"Blacked-out image saved to {save_path}")

    return img_out

def inverse_blackout_regions(img: np.ndarray, regions: List[Box], save_path:Path | str)-> np.ndarray:
    """
    Inverse of the blackout_regions function: 
    So there's two ways you can go about this:
    a. Black out everything else, 
    b. Seperate each as patches, and batch run seg. 

    I assume b. will work much better across models. 
    This same applied to blackout function as well. 

    For now, we will just get the a. working. 
    """

    img_out= np.zeros_like(img)
    h, w = img_out.shape[:2]
    for box in regions:
        x1 = max(0, min(int(box.x), w))
        x2 = max(0, min(int(box.x + box.width), w))
        y1 = max(0, min(int(box.y), h))
        y2 = max(0, min(int(box.y + box.height), h))
        img_out[y1:y2, x1:x2] = img[y1:y2, x1:x2]  # copy original pixels back in

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_img = img_out.copy()
        if save_img.dtype in (np.float32, np.float64):
            mn, mx = save_img.min(), save_img.max()
            save_img = ((save_img - mn) / (mx - mn + 1e-8) * 255).astype("uint8") if mx > mn else np.zeros_like(save_img, dtype="uint8")
        cv.imwrite(str(save_path), save_img)
        print(f"Inverse blacked-out image saved to {save_path}")

    return img_out

import numpy as np
import cv2 as cv

def colorize_components_inplace(mask: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Converts a binary mask (0/1 or 0/255) into a colored
    connected-components image.

    Returns a 3-channel uint8 image.
    """
    binary = (mask > 0).astype(np.uint8)

    num_labels, labels = cv.connectedComponents(binary)

    colored = np.zeros((*labels.shape, 3), dtype=np.uint8)

    rng = np.random.default_rng(seed)
    colors = rng.integers(0, 255, size=(num_labels, 3), dtype=np.uint8)

    for label in range(1, num_labels):  # skip background
        colored[labels == label] = colors[label]

    return colored



def extract_instances(mask: np.ndarray) -> list:
    """
    Connected components → simplified polygon contours per instance.
    Returns list of dicts matching the Instance schema.
    """
    # ensure binary
    binary = (mask > 0).astype("uint8")

    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(binary, connectivity=8)

    instances = []
    for label_id in range(1, num_labels):  # skip 0 = background
        # isolate this component
        component = (labels == label_id).astype("uint8") * 255

        area = int(stats[label_id, cv.CC_STAT_AREA])

        # skip noise — particles smaller than 50px
        # if area < 50:
        #     continue

        contours, _ = cv.findContours(component, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # simplify contour — epsilon controls how many vertices remain
        epsilon = 0.01 * cv.arcLength(contours[0], True)
        approx = cv.approxPolyDP(contours[0], epsilon, True)

        # squeeze to [[x,y], ...] and convert to float
        contour_pts = approx.squeeze().tolist()

        # approxPolyDP can return a 1D array if only 2 points — skip degenerate
        if not isinstance(contour_pts[0], list):
            continue

        x = int(stats[label_id, cv.CC_STAT_LEFT])
        y = int(stats[label_id, cv.CC_STAT_TOP])
        w = int(stats[label_id, cv.CC_STAT_WIDTH])
        h = int(stats[label_id, cv.CC_STAT_HEIGHT])

        instances.append({
            "id": label_id,
            "contour": contour_pts,
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "area": area,
        })

    return instances


def rasterize_instances(instances: list, h: int, w: int) -> np.ndarray:
    """
    Convert polygon instances back to a binary mask.
    """
    mask = np.zeros((h, w), dtype="uint8")

    for inst in instances:
        pts = np.array(inst.contour, dtype=np.int32).reshape((-1, 1, 2))
        cv.fillPoly(mask, [pts], 255)

    return mask


def save_debug_overlay(orig_path: Path, instances: list, save_path: Path):
    """Draw instance polygons + IDs overlaid on original image for visual verification."""
    img = cv.imread(str(orig_path))
    if img is None:
        # try npy
        arr = np.load(str(orig_path))
        arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype("uint8")
        img = cv.cvtColor(arr, cv.COLOR_GRAY2BGR)

    colors = [
        (255, 50, 50), (50, 255, 50), (50, 50, 255),
        (255, 255, 50), (255, 50, 255), (50, 255, 255),
    ]

    for i, inst in enumerate(instances):
        color = colors[i % len(colors)]
        pts = np.array(inst["contour"], dtype=np.int32).reshape((-1, 1, 2))

        # draw filled polygon with transparency
        overlay = img.copy()
        cv.fillPoly(overlay, [pts], color)
        img = cv.addWeighted(img, 0.7, overlay, 0.3, 0)

        # draw contour outline
        cv.polylines(img, [pts], isClosed=True, color=color, thickness=2)

        # draw instance ID at centroid
        cx = int(np.mean([p[0] for p in inst["contour"]]))
        cy = int(np.mean([p[1] for p in inst["contour"]]))
        cv.putText(img, str(inst["id"]), (cx, cy),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv.imwrite(str(save_path), img)


def batch_seg_patches(
    img: np.ndarray,
    regions: List[Box],  
    model,          # pre loaded model instance
    ) -> SegmentationResult:
    """
    The function blackout_regions and inverse_blackout_regions take one image 
    as input, and blackout regions in-place(in a copy of the image) and 
    return back a single image. 

    A better way might be to send patches of images that we want to keep 
    instead, batch seg and stitch it back again. 

    Key things to keep in mind are latency concerns and robustness of stitch.
    """
    h, w = img.shape[:2]
    combined = np.zeros((h, w), dtype="uint8")
    total_detections = 0

    for box in regions:
        # clamp to image bounds
        x1 = max(0, min(int(box.x), w))
        y1 = max(0, min(int(box.y), h))
        x2 = max(0, min(int(box.x + box.width), w))
        y2 = max(0, min(int(box.y + box.height), h))

        if x2 <= x1 or y2 <= y1:
            logger.warning(f"[BATCH] Degenerate patch skipped: {box}")
            continue

        patch = img[y1:y2, x1:x2]
        logger.info(f"[BATCH] Running seg on patch shape: {patch.shape}, offset: ({x1},{y1})")

        result = model.segment(patch)
        total_detections += result.metadata.get("detections", 0) if result.metadata else 0
        patch_mask = result.segmentation_mask


        # squeeze if needed
        if patch_mask.ndim == 3 and patch_mask.shape[0] == 1:
            patch_mask = patch_mask.squeeze(0)

        logger.info(f"[BATCH] Patch mask shape: {patch_mask.shape}, expected: ({y2-y1},{x2-x1})")

        # resize mask back to patch dims if model changed size
        if patch_mask.shape != (y2 - y1, x2 - x1):
            patch_mask = cv.resize(
                patch_mask.astype("uint8"),
                (x2 - x1, y2 - y1),
                interpolation=cv.INTER_NEAREST  # nearest for binary masks
            )

        # stitch back at correct offset
        combined[y1:y2, x1:x2] = np.maximum(
            combined[y1:y2, x1:x2],
            (patch_mask > 0).astype("uint8") * 255
        )

    return SegmentationResult(
        segmentation_mask=combined,
        metadata={"detections": total_detections},
        model="YoloSAM"
    )
