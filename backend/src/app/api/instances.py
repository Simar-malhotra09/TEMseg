import logging
import json
import math
import numpy as np
import cv2 as cv
from pathlib import Path
from scipy import ndimage

logger = logging.getLogger(__name__)

MIN_INSTANCE_AREA = 50

# Contour simplification (cv.approxPolyDP) epsilon, as a fraction of contour
# perimeter. Scaled by how much of the image the instance covers, so tiny
# particles get aggressively simplified (fewer vertices to clutter the UI)
# while large, significant ones keep more shape detail.
MIN_EPSILON_FRAC = 0.015  # applied to instances at/above SIZE_REF_AREA_FRAC
MAX_EPSILON_FRAC = 0.035  # applied to vanishingly small instances
SIZE_REF_AREA_FRAC = 0.05  # area fraction (of full image) considered "large"


def _simplification_epsilon(perimeter: float, area: int, image_area: int) -> float:
    """approxPolyDP epsilon for a contour, relative to how large the instance is vs the image."""
    area_frac = area / image_area
    significance = min(1.0, math.sqrt(area_frac / SIZE_REF_AREA_FRAC))
    epsilon_frac = (
        MAX_EPSILON_FRAC - (MAX_EPSILON_FRAC - MIN_EPSILON_FRAC) * significance
    )
    return epsilon_frac * perimeter


def extract_instances(
    mask: np.ndarray,
    session_dir: Path,
    save: bool = True,
) -> tuple[list[dict], np.ndarray]:
    """
    Run connected-component labeling on a binary mask.
    Returns (instances, labeled_mask).

    If save=True, writes:
      - instances.npy  : labeled integer mask (uint16), pixel value = instance ID
      - instances.json : list of {id, contour, bbox, area} dicts
    """
    logger.info("[INSTANCES] Starting instance extraction")

    # ensure binary uint8
    binary = (mask > 0).astype(np.uint8)

    labeled, n_components = ndimage.label(binary)
    logger.info(f"[INSTANCES] Found {n_components} connected components")
    image_area = binary.shape[0] * binary.shape[1]

    instances = []
    for inst_id in range(1, n_components + 1):
        component = (labeled == inst_id).astype(np.uint8)
        contours, _ = cv.findContours(component, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.debug(f"[INSTANCES] Component {inst_id} had no contours, skipping")
            continue

        area = int(np.sum(component))
        perimeter = cv.arcLength(contours[0], True)
        epsilon = _simplification_epsilon(perimeter, area, image_area)
        approx = cv.approxPolyDP(contours[0], epsilon, True)

        # squeeze to [[x,y], ...] and convert to float
        contour= approx.squeeze()
        if contour.ndim < 2 or len(contour) < 3:
            logger.debug(f"[INSTANCES] Component {inst_id} contour too small ({len(contour)} pts), skipping")
            continue

        x, y, w, h = cv.boundingRect(contours[0])
        if area < MIN_INSTANCE_AREA:
            logger.debug(f"[INSTANCES] Component {inst_id} too small ({area}px), skipping")
            continue
        instances.append({
            "id": inst_id,
            "contour": contour.tolist(),
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "area": area,
        })

    logger.info(f"[INSTANCES] Extracted {len(instances)} valid instances (skipped {n_components - len(instances)})")

    if save:
        npy_path = session_dir / "instances.npy"
        json_path = session_dir / "instances.json"

        np.save(npy_path, labeled.astype(np.uint16))
        logger.info(f"[INSTANCES] Saved labeled mask → {npy_path}")

        with open(json_path, "w") as f:
            json.dump(instances, f)
        logger.info(f"[INSTANCES] Saved instance metadata → {json_path} ({len(instances)} instances)")

    return instances, labeled


def load_instances(session_dir: Path) -> tuple[list[dict], np.ndarray] | None:
    """
    Load instances from disk. Returns (instances, labeled_mask) or None if not found.
    """
    npy_path = session_dir / "instances.npy"
    json_path = session_dir / "instances.json"

    if not npy_path.exists() or not json_path.exists():
        logger.info(f"[INSTANCES] No saved instances found in {session_dir}")
        return None

    labeled = np.load(npy_path)
    with open(json_path) as f:
        instances = json.load(f)

    logger.info(f"[INSTANCES] Loaded {len(instances)} instances from disk")
    return instances, labeled


def save_instances(session_dir: Path, instances: list[dict], labeled: np.ndarray) -> None:
    """Persist updated instances and labeled mask back to disk."""
    npy_path = session_dir / "instances.npy"
    json_path = session_dir / "instances.json"

    np.save(npy_path, labeled.astype(np.uint16))
    with open(json_path, "w") as f:
        json.dump(instances, f)

    logger.info(f"[INSTANCES] Saved {len(instances)} instances → {session_dir}")


def rasterize_instances(instances: list[dict], shape: tuple[int, int]) -> np.ndarray:
    """
    Rebuild a labeled mask from instance contours.
    Used after edits (split, vertex drag) to regenerate mask.png.
    """
    logger.info(f"[INSTANCES] Rasterizing {len(instances)} instances into {shape} mask")
    labeled = np.zeros(shape, dtype=np.uint16)
    for inst in instances:
        contour = np.array(inst["contour"], dtype=np.int32)
        cv.fillPoly(labeled, [contour], color=inst["id"])
    return labeled


def colorize_labeled_mask(labeled: np.ndarray) -> np.ndarray:
    """Convert a labeled integer mask to a colorized uint8 BGR image for saving as mask.png."""
    colored = np.zeros((*labeled.shape, 3), dtype=np.uint8)
    ids = np.unique(labeled)
    ids = ids[ids > 0]
    for inst_id in ids:
        hue = int((inst_id * 37) % 180)  # spread hues
        hsv_color = np.uint8([[[hue, 220, 220]]])
        bgr = cv.cvtColor(hsv_color, cv.COLOR_HSV2BGR)[0][0]
        colored[labeled == inst_id] = bgr
    return colored
