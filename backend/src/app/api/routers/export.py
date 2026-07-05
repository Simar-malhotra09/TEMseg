import io
import json
import logging
import zipfile
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel

logger = logging.getLogger("routes.export")
router = APIRouter(prefix="/export", tags=["export"])

SESSIONS_DIR = Path("sessions")


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


def _session_dir(session_id: str) -> Path:
    d = SESSIONS_DIR / session_id
    if not d.exists():
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return d


# curr we save the org file format (npy, tiff, png, jpg ) and
# a org preview which we use to mount on workspace.
# instead of the org file format we could just save a npy file I guess.
def _find_original_image(session_dir: Path) -> Path | None:
    """Original image is saved as original_preview.png for all formats."""
    p = session_dir / "original_preview.png"
    return p if p.exists() else None


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


@router.post("/{session_id}")
async def export_session(session_id: str, body: ExportRequest):
    """
    Build a ZIP of requested export items and stream it to the client.

    Available items:
      original_image    — original_preview.png
      seg_mask_png      — mask.png
      seg_mask_npy      — instances.npy (labeled integer mask)
      refined_mask_png  — mask.png (same file, post-refinement if saved)
      refined_mask_npy  — instances.npy (post-refinement if saved)
      instances_json    — instances.json
      stats_csv         — currently does not account for client side modifications, specifically refinements
    """
    logger.info(f"[EXPORT] POST export | session={session_id} | items={body.items}")

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
            path = _find_original_image(session_dir)
            if path is None:
                logger.warning(
                    f"[EXPORT] original_image not found | session={session_id}"
                )
                continue
            entries.append(("original_image.png", path))
            logger.info(f"[EXPORT] Adding original_image | {path.name}")

        elif item == "seg_mask_png":
            path = session_dir / "mask.png"
            if not path.exists():
                logger.warning(f"[EXPORT] mask.png not found | session={session_id}")
                continue
            entries.append(("seg_mask.png", path))
            logger.info("[EXPORT] Adding seg_mask_png")

        elif item == "seg_mask_npy":
            path = session_dir / "instances.npy"
            if not path.exists():
                logger.warning(
                    f"[EXPORT] instances.npy not found for seg_mask_npy | session={session_id}"
                )
                continue
            entries.append(("seg_mask.npy", path))
            logger.info("[EXPORT] Adding seg_mask_npy")

        elif item == "refined_mask_png":
            # refined mask overwrites mask.png after save — same file
            path = session_dir / "mask.png"
            if not path.exists():
                logger.warning(
                    f"[EXPORT] mask.png not found for refined_mask_png | session={session_id}"
                )
                continue
            entries.append(("refined_mask.png", path))
            logger.info("[EXPORT] Adding refined_mask_png")

        elif item == "refined_mask_npy":
            path = session_dir / "instances.npy"
            if not path.exists():
                logger.warning(
                    f"[EXPORT] instances.npy not found for refined_mask_npy | session={session_id}"
                )
                continue
            entries.append(("refined_mask.npy", path))
            logger.info("[EXPORT] Adding refined_mask_npy")

        elif item == "instances_json":
            path = session_dir / "instances.json"
            if not path.exists():
                logger.warning(
                    f"[EXPORT] instances.json not found | session={session_id}"
                )
                continue
            entries.append(("instances.json", path))
            logger.info("[EXPORT] Adding instances_json")

        elif item == "coco_json":
            try:
                coco_bytes = _build_coco_json(session_dir)
            except HTTPException:
                logger.warning(
                    f"[EXPORT] coco_json skipped — instances.json missing | session={session_id}"
                )
                continue
            # COCO task expects the image alongside the annotation file
            image_path = _find_original_image(session_dir)
            if image_path:
                entries.append(("original_image.png", image_path))
            entries.append(("annotations.coco.json", coco_bytes))
            logger.info(
                f"[EXPORT] Adding coco_json | {len(json.loads(coco_bytes)['annotations'])} annotations"
            )

        elif item == "stats_csv":
            stats_path = session_dir / "stats.json"
            if not stats_path.exists():
                logger.warning(f"[EXPORT] stats.json not found | session={session_id}")
                continue

            with open(stats_path) as f:
                stats = json.load(f)

            # build per-particle CSV
            particles = stats.get("particles", [])
            unit = stats.get("unit", "px")
            has_scale = stats.get("has_scale", False)

            header = f"id,area_{unit}{'²' if unit != 'px' else ''},eq_diameter_{unit},perimeter_{unit},circularity,solidity,aspect_ratio,n_vertices,shape\n"
            rows = []
            for p in particles:
                area = p.get("area_real", p["area_px"]) if has_scale else p["area_px"]
                diam = (
                    p.get("diameter_real", p["diameter_px"])
                    if has_scale
                    else p["diameter_px"]
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

            csv_bytes = (header + "".join(rows)).encode("utf-8")
            entries.append(("stats.csv", csv_bytes))
            logger.info(f"[EXPORT] Adding stats_csv | {len(particles)} particles")

    if not entries:
        logger.warning(f"[EXPORT] No files found to export | session={session_id}")
        raise HTTPException(
            status_code=404, detail="None of the requested items exist for this session"
        )

    # build zip in memory
    logger.info(f"[EXPORT] Building ZIP | {len(entries)} files")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for zip_name, source in entries:
            if isinstance(source, bytes):
                zf.writestr(zip_name, source)
            else:
                zf.write(source, arcname=zip_name)

    buf.seek(0)
    zip_size_kb = buf.getbuffer().nbytes / 1024
    logger.info(f"[EXPORT] ZIP ready | {zip_size_kb:.1f} KB | session={session_id}")

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename=temseg_export_{session_id[:8]}.zip"
        },
    )
