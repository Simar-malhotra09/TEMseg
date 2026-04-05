"""
Particle statistics from a segmentation mask.

All spatial measurements are returned in two forms:
  - *_px   — always present, in pixel units
  - *_real — present only when pixel_size is provided, in physical units

The caller (segment router / export) reads metadata.json from the session
to get pixel_size + pixel_unit, and passes them in.

Future: user can manually input pixel_size from the frontend.
"""

import numpy as np
import cv2 as cv
import math
from typing import Optional
from scipy import stats as sp_stats


def _prepare_mask(mask: np.ndarray) -> np.ndarray:
    """Ensure mask is binary uint8 (0 or 255)."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    return (mask > 0).astype(np.uint8) * 255


def _equivalent_diameter_px(area_px: float) -> float:
    """Diameter of a circle with the same area."""
    if area_px <= 0:
        return 0.0
    return 2.0 * math.sqrt(area_px / math.pi)


def _fit_ellipse_safe(cnt):
    """Fit ellipse if contour has enough points. Returns (major, minor) axes lengths or None."""
    if len(cnt) < 5:
        return None
    ellipse = cv.fitEllipse(cnt)
    # ellipse = ((cx, cy), (width, height), angle)
    axes = ellipse[1]
    major = max(axes)
    minor = min(axes)
    return (major, minor)


def _analyze_particles(mask: np.ndarray):
    """
    Extract per-particle measurements from a binary mask.
    Returns a list of dicts, one per particle.
    """
    mask = _prepare_mask(mask)
    num_labels, labels = cv.connectedComponents(mask)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    particles = []

    for cnt in contours:
        area_px = cv.contourArea(cnt)
        if area_px < 1:
            continue

        perimeter_px = cv.arcLength(cnt, True)
        diameter_px = _equivalent_diameter_px(area_px)

        # circularity: 1.0 = perfect circle, <1 = irregular
        circularity = 0.0
        if perimeter_px > 0:
            circularity = (4 * math.pi * area_px) / (perimeter_px**2)
            circularity = min(circularity, 1.0)  # clamp numerical noise

        # aspect ratio from fitted ellipse
        aspect_ratio = 1.0
        major_px = diameter_px
        minor_px = diameter_px
        ellipse = _fit_ellipse_safe(cnt)
        if ellipse is not None:
            major_px, minor_px = ellipse
            aspect_ratio = major_px / minor_px if minor_px > 0 else 1.0

        # bounding box
        x, y, w, h = cv.boundingRect(cnt)

        # shape classification based on circularity + aspect ratio
        shape = _classify_shape(circularity, aspect_ratio)

        particles.append(
            {
                "area_px": float(area_px),
                "perimeter_px": float(perimeter_px),
                "diameter_px": float(diameter_px),
                "major_axis_px": float(major_px),
                "minor_axis_px": float(minor_px),
                "circularity": float(circularity),
                "aspect_ratio": float(aspect_ratio),
                "shape": shape,
                "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            }
        )

    return particles


def _classify_shape(circularity: float, aspect_ratio: float) -> str:
    """
    Simple shape classification:
      - circular:   high circularity, low aspect ratio
      - elongated:  high aspect ratio (rods, wires)
      - irregular:  everything else
    """
    if circularity > 0.85 and aspect_ratio < 1.3:
        return "circular"
    elif aspect_ratio > 2.0:
        return "elongated"
    else:
        return "irregular"


def _fit_distributions(diameters: list[float]) -> dict:
    """
    Fit lognormal, normal, and Weibull distributions to diameter data.
    Returns best-fit model + all fitted parameters + goodness-of-fit.
    Requires at least 10 particles for meaningful fits.
    """
    if len(diameters) < 10:
        return {"reliable": False, "reason": "fewer than 10 particles"}

    arr = np.array(diameters)
    results = {}

    # Normal
    try:
        mu, std = sp_stats.norm.fit(arr)
        ks_stat, ks_p = sp_stats.kstest(arr, "norm", args=(mu, std))
        results["normal"] = {
            "params": {"mean": float(mu), "std": float(std)},
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_p),
        }
    except Exception:
        pass

    # Lognormal
    try:
        shape, loc, scale = sp_stats.lognorm.fit(arr, floc=0)
        mu_log = float(np.log(scale))
        sigma_log = float(shape)
        ks_stat, ks_p = sp_stats.kstest(arr, "lognorm", args=(shape, loc, scale))
        results["lognormal"] = {
            "params": {
                "mu_log": mu_log,
                "sigma_log": sigma_log,
                "geometric_mean": float(scale),
            },
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_p),
        }
    except Exception:
        pass

    # Weibull
    try:
        c, loc, scale = sp_stats.weibull_min.fit(arr, floc=0)
        ks_stat, ks_p = sp_stats.kstest(arr, "weibull_min", args=(c, loc, scale))
        results["weibull"] = {
            "params": {"shape": float(c), "scale": float(scale)},
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_p),
        }
    except Exception:
        pass

    if not results:
        return {"reliable": False, "reason": "all fits failed"}

    # pick best model by highest KS p-value (least evidence against fit)
    best = max(results.items(), key=lambda x: x[1]["ks_pvalue"])

    return {
        "reliable": True,
        "best_model": best[0],
        "fits": results,
    }


# Aggregate stats

def compute_stats(
    mask: np.ndarray,
    pixel_size: Optional[float] = None,
    pixel_unit: Optional[str] = None,
) -> dict:
    """
    Compute aggregate particle statistics from a segmentation mask.

    Args:
        mask: 2D array, non-zero = foreground
        pixel_size: physical size of one pixel (e.g. 0.245 for 0.245 nm/px)
        pixel_unit: unit string (e.g. "nm", "µm", "Å")

    Returns dict with:
        - unit info
        - aggregate stats (count, avg size, distributions, etc.)
        - per-particle list for histograms / export
    """
    mask = _prepare_mask(mask)
    particles = _analyze_particles(mask)

    has_scale = pixel_size is not None and pixel_size > 0
    scale = pixel_size if has_scale else 1.0
    scale_sq = scale * scale
    unit = pixel_unit if has_scale else "px"

    # --- Aggregate ---
    count = len(particles)
    total_pixels = mask.size
    foreground_pixels = int(np.count_nonzero(mask))
    coverage = foreground_pixels / total_pixels if total_pixels > 0 else 0.0

    if count == 0:
        return {
            "pixel_size": pixel_size,
            "pixel_unit": pixel_unit,
            "unit": unit,
            "has_scale": has_scale,
            "particle_count": 0,
            "coverage": 0.0,
            "avg_area_px": 0.0,
            "avg_diameter_px": 0.0,
            "avg_circularity": 0.0,
            "avg_aspect_ratio": 0.0,
            "avg_size": 0.0,
            "particles": [],
            "shape_distribution": {},
            "size_stats": {},
        }

    areas_px = [p["area_px"] for p in particles]
    diameters_px = [p["diameter_px"] for p in particles]
    circularities = [p["circularity"] for p in particles]
    aspect_ratios = [p["aspect_ratio"] for p in particles]
    shapes = [p["shape"] for p in particles]

    # shape distribution
    shape_counts = {}
    for s in shapes:
        shape_counts[s] = shape_counts.get(s, 0) + 1
    shape_distribution = {
        k: {"count": v, "fraction": v / count} for k, v in shape_counts.items()
    }

    # size stats
    areas_real = [a * scale_sq for a in areas_px]
    diameters_real = [d * scale for d in diameters_px]
    distribution_fits_dim = _fit_distributions(diameters_real)
    distribution_fits_area= _fit_distributions(areas_real)

    size_stats = {
        "area_mean": float(np.mean(areas_real)),
        "area_std": float(np.std(areas_real)),
        "area_min": float(np.min(areas_real)),
        "area_max": float(np.max(areas_real)),
        "area_median": float(np.median(areas_real)),
        "diameter_mean": float(np.mean(diameters_real)),
        "diameter_std": float(np.std(diameters_real)),
        "diameter_min": float(np.min(diameters_real)),
        "diameter_max": float(np.max(diameters_real)),
        "diameter_median": float(np.median(diameters_real)),
        "unit": unit,
    }

    # add real-unit fields to each particle
    for p in particles:
        p["area_real"] = p["area_px"] * scale_sq
        p["perimeter_real"] = p["perimeter_px"] * scale
        p["diameter_real"] = p["diameter_px"] * scale
        p["major_axis_real"] = p["major_axis_px"] * scale
        p["minor_axis_real"] = p["minor_axis_px"] * scale

    # --- backward compat fields (used by existing frontend) ---
    avg_size = float(np.mean(areas_px))  # kept in pixels for compat

    return {
        # scale info
        "pixel_size": pixel_size,
        "pixel_unit": pixel_unit,
        "unit": unit,
        "has_scale": has_scale,
        # aggregate (backward compat)
        "particle_count": count,
        "coverage": float(coverage),
        "avg_size": avg_size,
        "avg_circularity": float(np.mean(circularities)),
        "avg_aspect_ratio": float(np.mean(aspect_ratios)),
        # detailed aggregate
        "avg_area_px": float(np.mean(areas_px)),
        "avg_diameter_px": float(np.mean(diameters_px)),
        "avg_area_real": float(np.mean(areas_real)) if has_scale else None,
        "avg_diameter_real": float(np.mean(diameters_real)) if has_scale else None,
        # distributions
        "size_stats": size_stats,
        "shape_distribution": shape_distribution,
        "distribution_fits_diameter": distribution_fits_dim,
        "distribution_fits_area": distribution_fits_area,
        # per-particle (for histograms, export)
        "particles": particles,
    }


# backward compatibility
def compute_particle_count(mask: np.ndarray) -> int:
    mask = _prepare_mask(mask)
    num_labels, _ = cv.connectedComponents(mask)
    return num_labels - 1


def compute_avg_size(mask: np.ndarray) -> float:
    mask = _prepare_mask(mask)
    num_labels, labels = cv.connectedComponents(mask)
    if num_labels <= 1:
        return 0.0
    areas = [np.sum(labels == label) for label in range(1, num_labels)]
    return float(np.mean(areas))


def compute_avg_circularity(mask: np.ndarray) -> float:
    mask = _prepare_mask(mask)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    circs = []
    for cnt in contours:
        area = cv.contourArea(cnt)
        perim = cv.arcLength(cnt, True)
        if perim == 0:
            continue
        circs.append((4 * math.pi * area) / (perim**2))
    return float(np.mean(circs)) if circs else 0.0


def compute_coverage(mask: np.ndarray) -> float:
    mask = _prepare_mask(mask)
    return float(np.count_nonzero(mask) / mask.size)


