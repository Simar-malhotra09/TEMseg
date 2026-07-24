import logging
import json
import math
import time
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
MIN_EPSILON_FRAC = 0.005  # applied to instances at/above SIZE_REF_AREA_FRAC
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
    epsilon_scale: float = 1.0,
) -> tuple[list[dict], np.ndarray]:
    """
    Run connected-component labeling on a binary mask.
    Returns (instances, labeled_mask).

    If save=True, writes:
      - instances.npy  : labeled integer mask (uint16), pixel value = instance ID
      - instances.json : list of {id, contour, bbox, area} dicts

    epsilon_scale relaxes (value < 1.0) or tightens (value > 1.0) polygon
    simplification. SAM masks are typically high quality, so YOLO-SAM uses a
    smaller scale to retain more vertices.
    """
    t_start = time.perf_counter()

    # ensure binary uint8
    binary = (mask > 0).astype(np.uint8)

    labeled, n_components = ndimage.label(binary)
    image_area = binary.shape[0] * binary.shape[1]

    # bounding-box slice per label, so each component is scanned/compared
    # only within its own crop instead of against the full frame
    slices = ndimage.find_objects(labeled)

    instances = []
    skipped = 0
    for inst_id, sl in enumerate(slices, start=1):
        if sl is None:
            skipped += 1
            continue
        y_off, x_off = sl[0].start, sl[1].start
        component = (labeled[sl] == inst_id).astype(np.uint8)
        contours, _ = cv.findContours(
            component, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            skipped += 1
            continue

        area = int(np.sum(component))
        perimeter = cv.arcLength(contours[0], True)
        epsilon = _simplification_epsilon(perimeter, area, image_area) * epsilon_scale
        approx = cv.approxPolyDP(contours[0], epsilon, True)

        # squeeze to [[x,y], ...], then shift back from crop-local to full-image coords
        contour = approx.squeeze()
        if contour.ndim < 2 or len(contour) < 3:
            skipped += 1
            continue
        contour = contour + np.array([x_off, y_off])

        x, y, w, h = cv.boundingRect(contours[0])
        x, y = x + x_off, y + y_off
        if area < MIN_INSTANCE_AREA:
            skipped += 1
            continue
        instances.append(
            {
                "id": inst_id,
                "contour": contour.tolist(),
                "bbox": {"x": x, "y": y, "w": w, "h": h},
                "area": area,
            }
        )

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        f"Extracted {len(instances)} particles from mask "
        f"({skipped} skipped) in {elapsed_ms:.1f}ms"
    )

    if save:
        npy_path = session_dir / "instances.npy"
        json_path = session_dir / "instances.json"

        np.save(npy_path, labeled.astype(np.uint16))

        with open(json_path, "w") as f:
            json.dump(instances, f)
        logger.info(f"Saved {len(instances)} particle outlines to {session_dir.name}")

    return instances, labeled


def load_instances(session_dir: Path) -> tuple[list[dict], np.ndarray] | None:
    """
    Load instances from disk. Returns (instances, labeled_mask) or None if not found.
    """
    t_start = time.perf_counter()
    npy_path = session_dir / "instances.npy"
    json_path = session_dir / "instances.json"

    if not npy_path.exists() or not json_path.exists():
        return None

    labeled = np.load(npy_path)
    with open(json_path) as f:
        instances = json.load(f)

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        f"Loaded {len(instances)} saved particles for {session_dir.name} "
        f"in {elapsed_ms:.1f}ms"
    )
    return instances, labeled


def save_instances(
    session_dir: Path, instances: list[dict], labeled: np.ndarray
) -> None:
    """Persist updated instances and labeled mask back to disk."""
    t_start = time.perf_counter()
    npy_path = session_dir / "instances.npy"
    json_path = session_dir / "instances.json"

    np.save(npy_path, labeled.astype(np.uint16))
    with open(json_path, "w") as f:
        json.dump(instances, f)

    elapsed_ms = (time.perf_counter() - t_start) * 1000
    logger.info(
        f"Saved {len(instances)} particles for {session_dir.name} in {elapsed_ms:.1f}ms"
    )


def rasterize_instances(instances: list[dict], shape: tuple[int, int]) -> np.ndarray:
    """
    Rebuild a labeled mask from instance contours.
    Used after edits (split, vertex drag) to regenerate mask.png.
    """
    logger.info(f"Building mask from {len(instances)} particles")
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
