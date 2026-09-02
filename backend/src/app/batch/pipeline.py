import gc
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import cv2 as cv
import numpy as np
import torch

from app.api.instances import colorize_labeled_mask, extract_instances
from app.api.live_models import AvailableModels
from app.api.model_registry import get_device
from app.api.routers.images import _extract_metadata
from app.api.utils import normalize_mask
from app.batch.types import (
    VALID_IMAGE_EXTENSIONS,
    BatchConfig,
    ImageResult,
    resolve_model_name,
)
from app.models.base_model import Model
from app.models.helpers.compute_stats import compute_stats_from_instances
from app.models.helpers.config import (
    house_synthetic_config,
    nano_config,
    yolomaskrcnn_config,
)
from app.models.impls.fasteryolosam import FasterYoloSam
from app.models.impls.maskrcnn import MaskRCNN
from app.models.impls.yolomaskrcnn import YoloMaskRCNN
from app.models.impls.yolosam import YoloSam
from app.logutils import get_logger

logger = get_logger("batch")

_CVAT_IMPORT_README: bytes = b"""\
Importing into CVAT
===================

CVAT requires labels to be defined in the task BEFORE importing annotations.

Steps:
  1. Create a new task in CVAT.
  2. In the "Labels" step, add a label named exactly:  particle
  3. Upload original_image.png as the task data.
  4. Open the task, go to Actions > Upload annotations.
  5. Choose format "COCO 1.0" and select annotations.coco.json.

Label Studio
============
  1. Create a project with an Image classification/segmentation template.
  2. In Settings > Labeling Interface, add a PolygonLabels tag with value="particle".
  3. Import > Upload files, select annotations.coco.json.
"""


def build_model(model_id: AvailableModels) -> Model:
    device = get_device()
    if model_id == AvailableModels.yolosam:
        return YoloSam(nano_config, device=device)
    elif model_id == AvailableModels.fasteryolosam:
        return FasterYoloSam(nano_config, device=device)
    elif model_id == AvailableModels.yolomaskrcnn:
        return YoloMaskRCNN(
            yolomaskrcnn_config, AvailableModels.yolomaskrcnn, device=device
        )
    elif model_id == AvailableModels.maskrcnn_synthetic:
        return MaskRCNN(
            house_synthetic_config, AvailableModels.maskrcnn_synthetic, device=device
        )
    raise ValueError(f"No builder for model: {model_id}")


def _to_uint8_bgr(img: np.ndarray) -> np.ndarray:
    arr = img
    if arr.dtype != np.uint8:
        arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(
            np.uint8
        )
    return cv.cvtColor(arr, cv.COLOR_RGB2BGR)


def _build_coco_json(instances: list[dict], width: int, height: int) -> bytes:
    coco: dict = {
        "info": {"description": "TEM particle segmentation export"},
        "images": [
            {
                "id": 1,
                "file_name": "original_image.png",
                "width": width,
                "height": height,
            }
        ],
        "categories": [{"id": 1, "name": "particle", "supercategory": ""}],
        "annotations": [],
    }

    for inst in instances:
        contour: list[list[int]] = inst["contour"]
        flat: list[float] = [coord for pt in contour for coord in pt]
        bbox_d: dict = inst["bbox"]
        bbox: list[float] = [
            float(bbox_d["x"]),
            float(bbox_d["y"]),
            float(bbox_d["w"]),
            float(bbox_d["h"]),
        ]
        coco["annotations"].append(
            {
                "id": inst["id"],
                "image_id": 1,
                "category_id": 1,
                "segmentation": [flat],
                "area": float(inst["area"]),
                "bbox": bbox,
                "iscrowd": 0,
            }
        )

    return json.dumps(coco, indent=2).encode("utf-8")


def _write_stats_csv(stats: dict, image_output_dir: Path) -> None:
    with open(image_output_dir / "stats.json", "w") as f:
        json.dump(stats, f)

    particles = stats.get("particles", [])
    unit = stats.get("unit", "px")
    has_scale = stats.get("has_scale", False)

    header = (
        f"id,area_{unit}{'²' if unit != 'px' else ''},eq_diameter_{unit},"
        f"perimeter_{unit},circularity,solidity,aspect_ratio,n_vertices,shape\n"
    )
    rows = []
    for p in particles:
        area = p.get("area_real", p["area_px"]) if has_scale else p["area_px"]
        diam = (
            p.get("diameter_real", p["diameter_px"]) if has_scale else p["diameter_px"]
        )
        perim = (
            p.get("perimeter_real", p["perimeter_px"])
            if has_scale
            else p["perimeter_px"]
        )
        rows.append(
            f"{p['id']},{area:.4f},{diam:.4f},{perim:.4f},"
            f"{p['circularity']:.4f},{p.get('solidity', 0):.4f},"
            f"{p['aspect_ratio']:.4f},{p.get('n_vertices', 0)},{p['shape']}\n"
        )

    (image_output_dir / "stats.csv").write_text(header + "".join(rows))


def _write_exports(
    image_path: Path,
    image_output_dir: Path,
    export_items: list[str],
    img: np.ndarray,
    labeled: np.ndarray,
    instances: list[dict],
    stats: dict,
) -> None:
    if "original_image" in export_items:
        shutil.copy2(image_path, image_output_dir / f"original{image_path.suffix}")

    if "seg_mask_png" in export_items:
        colored = colorize_labeled_mask(labeled)
        cv.imwrite(str(image_output_dir / "mask.png"), colored)

    # instances.npy / instances.json are always written by extract_instances();
    # drop the ones that weren't requested.
    if "seg_mask_npy" not in export_items:
        (image_output_dir / "instances.npy").unlink(missing_ok=True)
    if "instances_json" not in export_items:
        (image_output_dir / "instances.json").unlink(missing_ok=True)

    if "stats_csv" in export_items:
        _write_stats_csv(stats, image_output_dir)

    if "coco_json" in export_items:
        height, width = labeled.shape[:2]
        coco_bytes = _build_coco_json(instances, width, height)
        (image_output_dir / "annotations.coco.json").write_bytes(coco_bytes)
        cv.imwrite(str(image_output_dir / "original_image.png"), _to_uint8_bgr(img))
        (image_output_dir / "IMPORT_INSTRUCTIONS.txt").write_bytes(_CVAT_IMPORT_README)


def process_image(
    image_path: Path,
    model: Model,
    model_id: AvailableModels,
    image_output_dir: Path,
    export_items: list[str],
) -> ImageResult:
    t_start = time.perf_counter()
    image_output_dir.mkdir(parents=True, exist_ok=True)

    img = model.load_image(image_path)

    meta = _extract_metadata(image_path, image_path.name)
    pixel_size = meta.get("pixel_size")
    pixel_unit = meta.get("pixel_unit")

    result = model.segment(img)

    mask = normalize_mask(result.segmentation_mask)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    binary = (mask > 0).astype(np.uint8)
    epsilon_scale = (
        0.75
        if model_id in (AvailableModels.yolosam, AvailableModels.fasteryolosam)
        else 1.0
    )
    instances, labeled = extract_instances(
        binary, image_output_dir, save=True, epsilon_scale=epsilon_scale
    )

    stats = compute_stats_from_instances(
        instances,
        mask,
        pixel_size=pixel_size,
        pixel_unit=pixel_unit,
        labeled_mask=labeled,
    )

    _write_exports(
        image_path, image_output_dir, export_items, img, labeled, instances, stats
    )

    time_elapsed = time.perf_counter() - t_start

    return ImageResult(
        filename=image_path.name,
        success=True,
        particle_count=stats["particle_count"],
        coverage=stats["coverage"],
        avg_area=stats["avg_area_px"],
        avg_diameter=stats["avg_diameter_px"],
        avg_circularity=stats["avg_circularity"],
        avg_aspect_ratio=stats["avg_aspect_ratio"],
        unit=stats["unit"],
        time_elapsed=time_elapsed,
        error=None,
    )


def _clear_device_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()


def run_batch(config: BatchConfig) -> list[ImageResult]:
    model_id = resolve_model_name(config.model_name)
    model = build_model(model_id)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p
        for p in config.input_dir.iterdir()
        if p.suffix.lower() in VALID_IMAGE_EXTENSIONS
        and (config.extension is None or p.suffix.lower() == config.extension)
    )
    if not images:
        raise ValueError(f"No valid images found in {config.input_dir}")

    results: list[ImageResult] = []
    total = len(images)

    for i, image_path in enumerate(images, start=1):
        if not config.quiet:
            print(f"[{i}/{total}] Processing {image_path.name}...", file=sys.stderr)

        image_output_dir = config.output_dir / image_path.stem
        try:
            result = process_image(
                image_path, model, model_id, image_output_dir, config.export_items
            )
        except Exception as e:
            logger.warning(
                f"Failed to process {image_path.name}:\n{traceback.format_exc()}"
            )
            result = ImageResult(
                filename=image_path.name,
                success=False,
                particle_count=0,
                coverage=0.0,
                avg_area=0.0,
                avg_diameter=0.0,
                avg_circularity=0.0,
                avg_aspect_ratio=0.0,
                unit="",
                time_elapsed=0.0,
                error=str(e),
            )

        results.append(result)
        _clear_device_cache()

    return results
