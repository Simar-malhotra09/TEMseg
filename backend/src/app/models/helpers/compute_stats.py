import numpy as np
import cv2 as cv
import math


''' These are all AI generated to start and I assume not super helpful at all ! '''
def _prepare_mask(mask: np.ndarray) -> np.ndarray:
    """
    Ensure mask is binary uint8 (0 or 255).
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # normalize to 0 / 255
    mask = (mask > 0).astype(np.uint8) * 255
    return mask


def compute_particle_count(mask: np.ndarray) -> int:
    mask = _prepare_mask(mask)

    num_labels, _ = cv.connectedComponents(mask)
    return num_labels - 1  # subtract background


def compute_avg_size(mask: np.ndarray) -> float:
    mask = _prepare_mask(mask)

    num_labels, labels = cv.connectedComponents(mask)
    if num_labels <= 1:
        return 0.0

    areas = []
    for label in range(1, num_labels):
        area = np.sum(labels == label)
        areas.append(area)

    return float(np.mean(areas))


def compute_avg_circularity(mask: np.ndarray) -> float:
    mask = _prepare_mask(mask)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.0

    circularities = []

    for cnt in contours:
        area = cv.contourArea(cnt)
        perimeter = cv.arcLength(cnt, True)

        if perimeter == 0:
            continue

        circularity = (4 * math.pi * area) / (perimeter ** 2)
        circularities.append(circularity)

    if not circularities:
        return 0.0

    return float(np.mean(circularities))


def compute_coverage(mask: np.ndarray) -> float:
    mask = _prepare_mask(mask)

    total_pixels = mask.size
    foreground_pixels = np.count_nonzero(mask)

    return float(foreground_pixels / total_pixels)
