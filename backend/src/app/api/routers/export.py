import io
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel

from app.logutils import Timer, get_logger, ui_event
from app.api.utils import _session_dir
from app.models.helpers.stats_export import build_stats_csv

logger = get_logger("export")
router = APIRouter(prefix="/export", tags=["export"])


class ExportRequest(BaseModel):
    items: list[str]


VALID_ITEMS = {
    "original_image",
    "seg_mask_png",
    "seg_mask_npy",
    "refined_mask_png",
    "refined_mask_npy",
    "instances_json",
    "stats_csv",
    "coco_json",
}


# curr we save the org file format (npy, tiff, png, jpg ) and
# a org preview which we use to mount on workspace.
# instead of the org file format we could just save a npy file I guess.
def _find_original_image(session_dir: Path) -> Path | None:
    """Return the display preview used for COCO image export."""
    p = session_dir / "original_preview.png"
    return p if p.exists() else None


def _find_source_image(session_dir: Path) -> Path | None:
    sources = sorted(path for path in session_dir.glob("org_*") if path.is_file())
    return sources[0] if sources else None


def _analysis_manifest(session_dir: Path, requested_items: list[str]) -> bytes:
    metadata_path = session_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    stats_path = session_dir / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    source = _find_source_image(session_dir)
    if source is None:
        raise HTTPException(status_code=404, detail="Source image is missing")
    with source.open("rb") as source_file:
        source_sha256 = hashlib.file_digest(source_file, "sha256").hexdigest()
    labeled_path = session_dir / "instances.npy"
    mask_sha256 = None
    if labeled_path.exists():
        with labeled_path.open("rb") as labeled_file:
            mask_sha256 = hashlib.file_digest(labeled_file, "sha256").hexdigest()
    manifest = {
        "schema_version": 1,
        "source_name": source.name.removeprefix("org_"),
        "source_sha256": source_sha256,
        "image_shape": metadata.get("image_shape"),
        "pixel_size": metadata.get("pixel_size"),
        "pixel_unit": metadata.get("pixel_unit"),
        "calibration": metadata.get("calibration"),
        "analysis": metadata.get("analysis"),
        "shape_config_sha256": stats.get("shape_config_sha256"),
        "labeled_mask_sha256": mask_sha256,
        "particle_count": stats.get("particle_count"),
        "edge_particle_count": stats.get("edge_particle_count"),
        "requested_items": requested_items,
    }
    return json.dumps(manifest, indent=2).encode("utf-8")


def _image_dimensions(session_dir: Path) -> tuple[int, int]:
    """Return (width, height) for the session image."""
    meta_path = session_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        shape = meta.get("image_shape")
        if shape and len(shape) == 2:
            height, width = shape
            return int(width), int(height)
    preview = session_dir / "original_preview.png"
    if preview.exists():
        with Image.open(preview) as img:
            return img.width, img.height
    # instances.npy is always present and shares image spatial dimensions
    mask: np.ndarray = np.load(session_dir / "instances.npy")
    height, width = mask.shape[:2]
    return int(width), int(height)


def _build_coco_json(session_dir: Path) -> bytes:
    """
    Build a COCO-format annotation JSON from instances.json.

    Segmentation is stored as a flat polygon [x1,y1,x2,y2,...] per instance,
    matching COCO's single-polygon convention (one polygon per crowd=0 annotation).
    """
    instances_path = session_dir / "instances.json"
    if not instances_path.exists():
        raise HTTPException(
            status_code=404, detail="instances.json not found for coco_json export"
        )

    with open(instances_path) as f:
        instances: list[dict] = json.load(f)

    width, height = _image_dimensions(session_dir)

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


@router.post("/{session_id}")
async def export_session(session_id: str, body: ExportRequest):
    """
    Build a ZIP of requested export items and stream it to the client.

    Available items:
      original_image    — original source file
      seg_mask_png      — mask.png
      seg_mask_npy      — instances.npy (labeled integer mask)
      refined_mask_png  — mask.png (same file, post-refinement if saved)
      refined_mask_npy  — instances.npy (post-refinement if saved)
      instances_json    — instances.json
      stats_csv         — currently does not account for client side modifications, specifically refinements
    """
    t = Timer(logger, "export")
    logger.info(f"export | session={session_id} | items={body.items}")

    # validate requested items
    unknown = set(body.items) - VALID_ITEMS
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown export items: {unknown}")

    if not body.items:
        raise HTTPException(status_code=400, detail="No items requested")

    session_dir = _session_dir(session_id)

    # map each requested item to a (zip_filename, source_path_or_bytes) tuple
    # source can be a Path (read from disk) or bytes (generated in memory)
    entries: list[tuple[str, Path | bytes]] = []

    for item in body.items:
        if item == "original_image":
            path = _find_source_image(session_dir)
            if path is None:
                logger.warning(f"original_image not found | session={session_id}")
                continue
            entries.append((f"source_image{path.suffix.lower()}", path))
            logger.info(f"Adding original_image | {path.name}")

        elif item == "seg_mask_png":
            path = session_dir / "mask.png"
            if not path.exists():
                logger.warning(f"mask.png not found | session={session_id}")
                continue
            entries.append(("seg_mask.png", path))
            logger.info("Adding seg_mask_png")

        elif item == "seg_mask_npy":
            path = session_dir / "instances.npy"
            if not path.exists():
                logger.warning(
                    f"instances.npy not found for seg_mask_npy | session={session_id}"
                )
                continue
            entries.append(("seg_mask.npy", path))
            logger.info("Adding seg_mask_npy")

        elif item == "refined_mask_png":
            # refined mask overwrites mask.png after save — same file
            path = session_dir / "mask.png"
            if not path.exists():
                logger.warning(
                    f"mask.png not found for refined_mask_png | session={session_id}"
                )
                continue
            entries.append(("refined_mask.png", path))
            logger.info("Adding refined_mask_png")

        elif item == "refined_mask_npy":
            path = session_dir / "instances.npy"
            if not path.exists():
                logger.warning(
                    f"instances.npy not found for refined_mask_npy | session={session_id}"
                )
                continue
            entries.append(("refined_mask.npy", path))
            logger.info("Adding refined_mask_npy")

        elif item == "instances_json":
            path = session_dir / "instances.json"
            if not path.exists():
                logger.warning(f"instances.json not found | session={session_id}")
                continue
            entries.append(("instances.json", path))
            logger.info("Adding instances_json")

        elif item == "coco_json":
            try:
                coco_bytes = _build_coco_json(session_dir)
            except HTTPException:
                logger.warning(
                    f"coco_json skipped — instances.json missing | session={session_id}"
                )
                continue
            image_path = _find_original_image(session_dir)
            if image_path:
                entries.append(("original_image.png", image_path))
            entries.append(("annotations.coco.json", coco_bytes))
            entries.append(("IMPORT_INSTRUCTIONS.txt", _CVAT_IMPORT_README))
            logger.info(
                f"Adding coco_json | {len(json.loads(coco_bytes)['annotations'])} annotations"
            )

        elif item == "stats_csv":
            stats_path = session_dir / "stats.json"
            if not stats_path.exists():
                logger.warning(f"stats.json not found | session={session_id}")
                continue

            with open(stats_path) as f:
                stats = json.load(f)

            entries.append(("stats.csv", build_stats_csv(stats)))
            logger.info(
                f"Adding stats_csv | {len(stats.get('particles', []))} particles"
            )

    if not entries:
        logger.warning(f"No files found to export | session={session_id}")
        raise HTTPException(
            status_code=404, detail="None of the requested items exist for this session"
        )

    entries.append(
        ("analysis_manifest.json", _analysis_manifest(session_dir, body.items))
    )

    # build zip in memory
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for zip_name, source in entries:
            if isinstance(source, bytes):
                zf.writestr(zip_name, source)
            else:
                zf.write(source, arcname=zip_name)

    buf.seek(0)
    t.field("files", len(entries))
    t.field("zip_kb", round(buf.getbuffer().nbytes / 1024, 1))
    t.stop()

    ui_event(
        "EXPORT_READY",
        "Your export is ready — check your downloads folder.",
        level="info",
    )

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename=temseg_export_{session_id[:8]}.zip"
        },
    )


@router.post("/{session_id}/hist_csv")
async def export_histogram_csv(session_id: str, metric: str = "diameter"):
    """
    Return a CSV containing histogram-sized data for the given metric.

    The CSV contains per-particle values for the requested metric followed by
    summary statistics (count, mean, std, median) and the best distribution fit.
    """
    metric = metric.lower().strip()
    if metric not in {"diameter", "area"}:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid metric '{metric}'; use 'diameter' or 'area'",
        )

    logger.info(f"POST export histogram csv | session={session_id} | metric={metric}")

    session_dir = _session_dir(session_id)
    stats_path = session_dir / "stats.json"
    if not stats_path.exists():
        logger.warning(f"stats.json not found | session={session_id}")
        raise HTTPException(status_code=404, detail="stats.json not found")

    with open(stats_path) as f:
        stats = json.load(f)

    particles = stats.get("particles", [])
    unit = stats.get("unit", "px")
    has_scale = stats.get("has_scale", False)
    size_stats = stats.get("size_stats", {})
    fits = stats.get(f"distribution_fits_{metric}", {"reliable": False})

    value_key = f"{metric}_real" if has_scale else f"{metric}_px"
    unit_label = unit if metric == "diameter" else f"{unit}²"
    header = f"id,{metric}_{unit_label}\n"

    rows = []
    for p in particles:
        value = p.get(value_key)
        if value is None:
            value = p.get(f"{metric}_px", 0)
        rows.append(f"{p['id']},{value:.4f}")

    # summary statistics after particle data
    rows.append("")
    rows.append(f"Number of particles,{len(particles)}")
    rows.append(f"Mean,{size_stats.get(f'{metric}_mean', 0):.4f} {unit_label}")
    rows.append(
        f"Standard deviation,{size_stats.get(f'{metric}_std', 0):.4f} {unit_label}"
    )
    rows.append(f"Median,{size_stats.get(f'{metric}_median', 0):.4f} {unit_label}")

    if fits.get("reliable") and fits.get("best_model"):
        fit = fits.get("fits", {}).get(fits["best_model"])
        if fit:
            params = ", ".join(f"{k}={v:.4f}" for k, v in fit.get("params", {}).items())
            rows.append(f"Best fit,{fits['best_model']} ({params})")

    csv_bytes = (header + "\n".join(rows)).encode("utf-8")
    logger.info(f"histogram CSV ready | {len(particles)} particles | metric={metric}")

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=temseg_histogram_{metric}_{session_id[:8]}.csv"
        },
    )
