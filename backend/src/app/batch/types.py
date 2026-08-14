from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from app.api.live_models import AvailableModels

VALID_IMAGE_EXTENSIONS = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".npy", ".emd"}


class ExportPreset(str, Enum):
    full = "full"
    masks = "masks"
    stats = "stats"


EXPORT_PRESET_ITEMS: dict[ExportPreset, list[str]] = {
    ExportPreset.full: [
        "original_image",
        "seg_mask_png",
        "seg_mask_npy",
        "instances_json",
        "stats_csv",
        "coco_json",
    ],
    ExportPreset.masks: ["seg_mask_png", "seg_mask_npy"],
    ExportPreset.stats: ["stats_csv", "instances_json"],
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
    "maskrcnn": AvailableModels.maskrcnn,
    "maskrcnn-synthetic": AvailableModels.maskrcnn_synthetic,
}


def resolve_model_name(name: str) -> AvailableModels:
    key = name.lower().strip()
    if key not in MODEL_NAME_MAP:
        valid = ", ".join(MODEL_NAME_MAP.keys())
        raise ValueError(f"Unknown model '{name}'. Valid: {valid}")
    return MODEL_NAME_MAP[key]


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


@dataclass(frozen=True)
class BatchConfig:
    input_dir: Path
    output_dir: Path
    model_name: str
    export_items: list[str]
    quiet: bool


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
