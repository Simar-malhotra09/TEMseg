from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from app.api.live_models import AvailableModels
from app.logutils import get_logger

logger = get_logger("batch")

VALID_IMAGE_EXTENSIONS = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".npy", ".emd"}


class ExportPreset(str, Enum):
    full = "full"
    no_png = "no_png"
    instances = "instances"


EXPORT_PRESET_ITEMS: dict[ExportPreset, list[str]] = {
    ExportPreset.full: [
        "original_image",
        "seg_mask_png",
        "seg_mask_npy",
        "instances_json",
        "stats_csv",
        "coco_json",
    ],
    ExportPreset.no_png: [
        "original_image",
        "seg_mask_npy",
        "instances_json",
        "stats_csv",
        "coco_json",
    ],
    ExportPreset.instances: ["instances_json"],
}


VALID_EXPORT_ITEMS = {
    "original_image",
    "seg_mask_png",
    "seg_mask_npy",
    "instances_json",
    "stats_csv",
    "coco_json",
}


MODEL_NAME_MAP: dict[str, AvailableModels] = {
    "yolosam": AvailableModels.yolosam,
    "yolomaskrcnn": AvailableModels.yolomaskrcnn,
    "maskrcnn-synthetic": AvailableModels.maskrcnn_synthetic,
}


def resolve_model_name(name: str) -> AvailableModels:
    key = name.lower().strip()
    if key not in MODEL_NAME_MAP:
        valid = ", ".join(MODEL_NAME_MAP.keys())
        raise ValueError(f"Unknown model '{name}'. Valid: {valid}")
    return MODEL_NAME_MAP[key]


def normalize_extension(value: str | None) -> str | None:
    """Normalize an --extension value to a lowercase suffix with a leading dot.

    Returns None for empty/None values, meaning "accept any supported image
    extension".
    """
    if value is None:
        return None
    ext = value.strip().lower()
    if not ext:
        return None
    if not ext.startswith("."):
        ext = "." + ext
    return ext


def resolve_export_items(value: str) -> list[str]:
    """Resolve --export value: either a preset name or a comma-separated item list."""
    try:
        preset = ExportPreset(value.lower().strip())
        return EXPORT_PRESET_ITEMS[preset]
    except ValueError:
        pass

    items = [i.strip() for i in value.split(",") if i.strip()]
    unknown = set(items) - VALID_EXPORT_ITEMS
    if unknown:
        raise ValueError(f"Unknown export items: {unknown}")
    return items


def apply_include(items: list[str], include: list[str]) -> list[str]:
    """Add --include items to a resolved export item list. Unknown items or
    items already present are skipped with a warning."""
    result = list(items)
    for item in include:
        if item not in VALID_EXPORT_ITEMS:
            logger.warning(f"--include: unknown export item '{item}', ignoring")
        elif item in result:
            logger.warning(f"--include: '{item}' already in export items, ignoring")
        else:
            result.append(item)
    return result


def apply_exclude(items: list[str], exclude: list[str]) -> list[str]:
    """Remove --exclude items from a resolved export item list. Unknown items
    or items not present in the list are skipped with a warning."""
    result = list(items)
    for item in exclude:
        if item not in VALID_EXPORT_ITEMS:
            logger.warning(f"--exclude: unknown export item '{item}', ignoring")
        elif item not in result:
            logger.warning(f"--exclude: '{item}' not in export items, ignoring")
        else:
            result.remove(item)
    return result


@dataclass(frozen=True)
class BatchConfig:
    input_dir: Path
    output_dir: Path
    model_name: str
    export_items: list[str]
    quiet: bool
    extension: str | None = None


@dataclass
class ImageResult:
    filename: str
    success: bool
    particle_count: int
    coverage: float
    avg_area: float
    avg_diameter: float
    avg_circularity: float
    avg_aspect_ratio: float
    unit: str
    time_elapsed: float
    error: str | None
