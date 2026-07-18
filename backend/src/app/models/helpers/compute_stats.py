import operator
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, get_args

import numpy as np
import cv2 as cv
import math
from scipy import stats as sp_stats

from app.models.helpers.settings import settings

ShapeMetric = Literal["circularity", "aspect_ratio", "solidity", "n_vertices"]
_SHAPE_METRICS = get_args(
    ShapeMetric
)  # server side source of truth, derived from the type above
ShapeOperator = Literal["<", "<=", ">", ">=", "==", "!="]

_SHAPE_OPERATORS: dict[ShapeOperator, Callable[[float, float], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}

# metrics compared with rounding tolerance when op is "==" / "!="
_FLOAT_SHAPE_METRICS = {"circularity", "aspect_ratio", "solidity"}


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


# single conditon
@dataclass(frozen=True)
class ShapeCondition:
    metric: ShapeMetric
    op: ShapeOperator
    value: float


# multiple conditions; Rule true => all conditions also true
@dataclass(frozen=True)
class ShapeRule:
    label: str
    conditions: list[ShapeCondition]


# full config of conditions and rules
@dataclass(frozen=True)
class ShapeClassificationConfig:
    default_shape: str
    rules: list[ShapeRule]


def _parse_shape_condition(raw: dict) -> ShapeCondition:
    metric = raw["metric"]
    if metric not in _SHAPE_METRICS:
        raise ValueError(
            f"Unknown shape metric in shape_config.toml: {metric!r}. Valid metrics: {_SHAPE_METRICS}"
        )
    op = raw["op"]
    if op not in _SHAPE_OPERATORS:
        raise ValueError(f"Unknown operator in shape_config.toml: {op!r}")
    return ShapeCondition(metric=metric, op=op, value=float(raw["value"]))


def _parse_shape_rule(raw: dict) -> ShapeRule:
    return ShapeRule(
        label=raw["label"],
        conditions=[_parse_shape_condition(c) for c in raw["conditions"]],
    )


def load_shape_classification_config(path: Path) -> ShapeClassificationConfig:
    """Load shape classification rules from a TOML file (see shape_config.toml)."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return ShapeClassificationConfig(
        default_shape=raw["default_shape"],
        rules=[_parse_shape_rule(r) for r in raw.get("rules", [])],
    )


def _shape_metric_value(
    metric: ShapeMetric,
    circularity: float,
    aspect_ratio: float,
    solidity: float,
    n_vertices: int,
) -> float:
    if metric == "circularity":
        return circularity
    if metric == "aspect_ratio":
        return aspect_ratio
    if metric == "solidity":
        return solidity
    return float(n_vertices)


def _shape_rule_matches(
    rule: ShapeRule,
    circularity: float,
    aspect_ratio: float,
    solidity: float,
    n_vertices: int,
) -> bool:
    for cond in rule.conditions:
        actual = _shape_metric_value(
            cond.metric, circularity, aspect_ratio, solidity, n_vertices
        )
        compare = _SHAPE_OPERATORS[cond.op]
        if cond.op in ("==", "!=") and cond.metric in _FLOAT_SHAPE_METRICS:
            if not compare(round(actual, 2), round(cond.value, 2)):
                return False
        elif not compare(actual, cond.value):
            return False
    return True


# shape distribution
def _classify_shape(
    circularity: float,
    aspect_ratio: float,
    solidity: float,
    n_vertices: int,
    config: ShapeClassificationConfig,
) -> str:
    """
    Shape classification driven by user-editable rules (shape_config.toml).
    Categories chosen for TEM nanoparticle relevance.
    """
    for rule in config.rules:
        if _shape_rule_matches(rule, circularity, aspect_ratio, solidity, n_vertices):
            return rule.label
    return config.default_shape


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
    labeled_mask: np.ndarray
    | None = None,  # mask with each pixel being the particle id, 0 for bg
) -> dict:
    """
    Compute stats using pre-extracted instances (from instances.json).
    This ensures particle IDs match between stats and the canvas overlay.
    """

    # I added this placeholder '-' in commit 836855bd05011e60c1d96038b00240566da4756f
    # this is hacky and needs to be fixed later.
    has_scale = pixel_size != "-" and pixel_size is not None and pixel_size > 0
    scale = pixel_size if has_scale else 1.0
    scale_sq = scale * scale if scale else 1.0
    unit = pixel_unit if has_scale else "px"

    total_pixels = mask.size
    foreground_pixels = int(np.count_nonzero(mask > 0))
    coverage = foreground_pixels / total_pixels if total_pixels > 0 else 0.0

    particles = []

    # load shape classification rules once, outside the per-instance loop
    shape_config = load_shape_classification_config(settings.SHAPE_CONFIG_PATH)

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
                area_px = float(np.sum(component))
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

        # circularity
        circularity = 0.0
        if perimeter_px > 0:
            circularity = (4 * math.pi * area_px) / (perimeter_px**2)
            circularity = min(circularity, 1.0)

        # solidity = area / convex hull area
        solidity = 1.0
        try:
            hull = cv.convexHull(cnt)
            hull_area = cv.contourArea(hull)
            if hull_area > 0:
                solidity = area_px / hull_area
                solidity = min(solidity, 1.0)
        except Exception:
            pass

        # convexity = convex hull perimeter / perimeter
        convexity = 1.0
        try:
            hull = cv.convexHull(cnt)
            hull_perim = cv.arcLength(hull, True)
            if perimeter_px > 0:
                convexity = hull_perim / perimeter_px
                convexity = min(convexity, 1.0)
        except Exception:
            pass

        # rectangularity = area / bounding rect area
        x, y, w, h = cv.boundingRect(cnt)
        rect_area = w * h
        rectangularity = area_px / rect_area if rect_area > 0 else 0.0

        # aspect ratio from ellipse fit or min area rect
        aspect_ratio = 1.0
        major_px = diameter_px
        minor_px = diameter_px
        ellipse = _fit_ellipse_safe(cnt)
        if ellipse is not None:
            major_px, minor_px = ellipse
            aspect_ratio = major_px / minor_px if minor_px > 0 else 1.0

        # vertex count from simplified contour (for faceted shape detection)
        epsilon = 0.02 * perimeter_px
        approx = cv.approxPolyDP(cnt, epsilon, True)
        n_vertices = len(approx)

        shape = _classify_shape(
            circularity, aspect_ratio, solidity, n_vertices, shape_config
        )

        p = {
            "id": inst_id,
            "area_px": area_px,
            "perimeter_px": perimeter_px,
            "diameter_px": diameter_px,
            "major_axis_px": float(major_px),
            "minor_axis_px": float(minor_px),
            "circularity": float(circularity),
            "solidity": float(solidity),
            "convexity": float(convexity),
            "rectangularity": float(rectangularity),
            "aspect_ratio": float(aspect_ratio),
            "n_vertices": n_vertices,
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
            "pixel_size": pixel_size,
            "pixel_unit": pixel_unit,
            "unit": unit,
            "has_scale": has_scale,
            "particle_count": 0,
            "coverage": coverage,
            "avg_size": 0.0,
            "avg_circularity": 0.0,
            "avg_aspect_ratio": 0.0,
            "avg_area_px": 0.0,
            "avg_diameter_px": 0.0,
            "avg_area_real": None,
            "avg_diameter_real": None,
            "particles": [],
            "shape_distribution": {},
            "size_stats": {},
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
    shape_distribution = {
        k: {"count": v, "fraction": v / count} for k, v in shape_counts.items()
    }

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
        "pixel_size": pixel_size,
        "pixel_unit": pixel_unit,
        "unit": unit,
        "has_scale": has_scale,
        "particle_count": count,
        "coverage": float(coverage),
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
