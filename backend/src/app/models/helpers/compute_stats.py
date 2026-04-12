import numpy as np
import cv2 as cv
import math
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


# shape distribution
# this needs to be a lot more robust 
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


# pdf distribution
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
def compute_stats_from_instances(
    instances: list[dict],
    mask: np.ndarray,
    pixel_size: float | None = None,
    pixel_unit: str | None = None,
    labeled_mask: np.ndarray | None= None # mask with each pixel being the particle id, 0 for bg
) -> dict:
    """
    Compute stats using pre-extracted instances (from instances.json).
    This ensures particle IDs match between stats and the canvas overlay.
    """
    has_scale = pixel_size is not None and pixel_size > 0
    scale = pixel_size if has_scale else 1.0
    scale_sq = scale * scale if scale else 1.0
    unit = pixel_unit if has_scale else "px"

    total_pixels = mask.size
    foreground_pixels = int(np.count_nonzero(mask > 0))
    coverage = foreground_pixels / total_pixels if total_pixels > 0 else 0.0

    particles = []
    for inst in instances:
        inst_id = inst["id"]

        # get contour: prefer exact pixels from labeled mask
        if labeled_mask is not None:
            component = (labeled_mask == inst_id).astype(np.uint8)
            exact_contours, _ = cv.findContours(
                component, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
            )
            if exact_contours:
                cnt = exact_contours[0]
                area_px = float(np.sum(component))  # pixel count is ground truth
                perimeter_px = float(cv.arcLength(cnt, True))
            else:
                cnt = np.array(inst["contour"], dtype=np.int32)
                area_px = float(inst["area"])
                perimeter_px = float(cv.arcLength(cnt, True))
        else:
            cnt = np.array(inst["contour"], dtype=np.int32)
            area_px = float(inst["area"])
            perimeter_px = float(cv.arcLength(cnt, True))

        if area_px < 1 or len(cnt) < 3:
            continue

        diameter_px = _equivalent_diameter_px(area_px)

        # circularity from exact contour
        circularity = 0.0
        if perimeter_px > 0:
            circularity = (4 * math.pi * area_px) / (perimeter_px ** 2)
            circularity = min(circularity, 1.0)

        # aspect ratio
        aspect_ratio = 1.0
        major_px = diameter_px
        minor_px = diameter_px
        ellipse = _fit_ellipse_safe(cnt)
        if ellipse is not None:
            major_px, minor_px = ellipse
            aspect_ratio = major_px / minor_px if minor_px > 0 else 1.0

        shape = _classify_shape(circularity, aspect_ratio)

        p = {
            "id": inst_id,
            "area_px": area_px,
            "perimeter_px": perimeter_px,
            "diameter_px": diameter_px,
            "major_axis_px": float(major_px),
            "minor_axis_px": float(minor_px),
            "circularity": float(circularity),
            "aspect_ratio": float(aspect_ratio),
            "shape": shape,
            "bbox": inst["bbox"],
        }

        if has_scale:
            p["area_real"] = area_px * scale_sq
            p["perimeter_real"] = perimeter_px * scale
            p["diameter_real"] = diameter_px * scale
            p["major_axis_real"] = float(major_px) * scale
            p["minor_axis_real"] = float(minor_px) * scale

        particles.append(p)

    count = len(particles)

    if count == 0:
        return {
            "pixel_size": pixel_size, "pixel_unit": pixel_unit,
            "unit": unit, "has_scale": has_scale,
            "particle_count": 0, "coverage": coverage,
            "avg_size": 0.0, "avg_circularity": 0.0, "avg_aspect_ratio": 0.0,
            "avg_area_px": 0.0, "avg_diameter_px": 0.0,
            "avg_area_real": None, "avg_diameter_real": None,
            "particles": [], "shape_distribution": {}, "size_stats": {},
            "distribution_fits_diameter": {"reliable": False, "reason": "no particles"},
            "distribution_fits_area": {"reliable": False, "reason": "no particles"},
        }

    areas_px = [p["area_px"] for p in particles]
    diameters_px = [p["diameter_px"] for p in particles]
    circularities = [p["circularity"] for p in particles]
    aspect_ratios = [p["aspect_ratio"] for p in particles]
    shapes = [p["shape"] for p in particles]

    areas_real = [a * scale_sq for a in areas_px]
    diameters_real = [d * scale for d in diameters_px]

    shape_counts = {}
    for s in shapes:
        shape_counts[s] = shape_counts.get(s, 0) + 1
    shape_distribution = {k: {"count": v, "fraction": v / count} for k, v in shape_counts.items()}

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

    distribution_fits_diameter = _fit_distributions(diameters_real)
    distribution_fits_area = _fit_distributions(areas_real)

    return {
        "pixel_size": pixel_size, "pixel_unit": pixel_unit,
        "unit": unit, "has_scale": has_scale,
        "particle_count": count, "coverage": float(coverage),
        "avg_size": float(np.mean(areas_px)),
        "avg_circularity": float(np.mean(circularities)),
        "avg_aspect_ratio": float(np.mean(aspect_ratios)),
        "avg_area_px": float(np.mean(areas_px)),
        "avg_diameter_px": float(np.mean(diameters_px)),
        "avg_area_real": float(np.mean(areas_real)) if has_scale else None,
        "avg_diameter_real": float(np.mean(diameters_real)) if has_scale else None,
        "size_stats": size_stats,
        "shape_distribution": shape_distribution,
        "distribution_fits_diameter": distribution_fits_diameter,
        "distribution_fits_area": distribution_fits_area,
        "particles": particles,
    }
