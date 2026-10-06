import json
import hashlib
import math
import uuid
import re
import shutil
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, Request, Form, HTTPException
from fastapi.responses import FileResponse
import numpy as np
import cv2 as cv
from tifffile import TiffFileError
from app.logutils import get_logger, ui_event
from app.api.utils import validate_session_id
from app.api.routers.sessions import update_mru, create_group
from app.models.helpers.image_io import load_rgb_uint8
from pydantic import BaseModel
from typing import Literal

router = APIRouter(prefix="/images")
logger = get_logger("images")
logger_meta = get_logger("images", sub="meta")
SESSIONS_DIR = Path("sessions")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy", ".emd"}
PIXEL_UNIT_ALIASES = {
    "pm": "pm",
    "nm": "nm",
    "um": "µm",
    "µm": "µm",
    "μm": "µm",
    "micron": "µm",
    "microns": "µm",
    "mm": "mm",
    "m": "m",
    "a": "Å",
    "å": "Å",
    "angstrom": "Å",
    "ångström": "Å",
}


def _validated_upload_name(filename: str | None) -> str:
    if not filename or filename in {".", ".."}:
        raise HTTPException(status_code=400, detail="Image filename is required")
    if "/" in filename or "\\" in filename:
        raise HTTPException(
            status_code=400, detail="Image filename must not contain a path"
        )
    if len(filename) > 200 or any(
        ord(char) < 32 or ord(char) == 127 for char in filename
    ):
        raise HTTPException(
            status_code=400,
            detail="Image filename is too long or contains control characters",
        )
    if Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported image file type")
    return filename


class UpdateMetadataRequest(BaseModel):
    pixel_size: float | None = None
    pixel_unit: str | None = None
    calibration_source: Literal["manual", "scale_bar"] = "manual"
    scale_bar_pixels: float | None = None
    scale_bar_length: float | None = None


def _extract_metadata(filepath: Path, filename: str) -> dict:
    """
    Extract metadata from the uploaded file.
    """
    fname = filename.lower()
    meta = {
        "file_path": str(filepath),
        "file_name": str(Path(filename).stem),
        "original_format": Path(filename).suffix.lstrip("."),
        "image_shape": None,
    }

    try:
        if fname.endswith(".emd"):
            from rsciio.emd import file_reader

            signals = file_reader(str(filepath))
            s = signals[0]

            meta["image_shape"] = list(s["data"].shape)

            # axes are dicts, not HyperSpy objects
            if "axes" in s:
                meta["axes"] = []
                for ax in s["axes"]:
                    meta["axes"].append(
                        {
                            "name": ax.get("name"),
                            "scale": float(ax["scale"]) if ax.get("scale") else None,
                            "offset": float(ax["offset"]) if ax.get("offset") else None,
                            "units": ax.get("units"),
                            "size": ax.get("size"),
                        }
                    )
                spatial_axes = {
                    str(axis["name"]).lower(): axis
                    for axis in meta["axes"]
                    if str(axis["name"]).lower() in {"x", "y"}
                }
                x_axis = spatial_axes.get("x")
                y_axis = spatial_axes.get("y")
                if x_axis and y_axis:
                    x_scale = x_axis.get("scale")
                    y_scale = y_axis.get("scale")
                    x_unit = PIXEL_UNIT_ALIASES.get(
                        str(x_axis.get("units") or "").strip().lower()
                    )
                    y_unit = PIXEL_UNIT_ALIASES.get(
                        str(y_axis.get("units") or "").strip().lower()
                    )
                    if (
                        x_scale is not None
                        and y_scale is not None
                        and math.isfinite(x_scale)
                        and math.isfinite(y_scale)
                        and x_scale > 0
                        and y_scale > 0
                        and x_unit
                        and x_unit == y_unit
                        and math.isclose(x_scale, y_scale, rel_tol=1e-5)
                    ):
                        meta["pixel_size"] = x_scale
                        meta["pixel_unit"] = x_unit
                        meta["calibration"] = {"source": "embedded"}
                    else:
                        meta["calibration_warning"] = (
                            "Embedded x and y pixel scales differ or are incomplete. "
                            "Set the pixel size before using physical measurements."
                        )

            # original metadata is already a dict
            if "original_metadata" in s:
                meta["original_metadata"] = s["original_metadata"]
            if "metadata" in s:
                meta["hyperspy_metadata"] = s["metadata"]

            logger_meta.info(
                f"EMD: pixel_size={meta.get('pixel_size')} {meta.get('pixel_unit')}"
            )

        elif fname.endswith((".tif", ".tiff")):
            import tifffile
            from tifffile import TiffFileError

            try:
                tif = tifffile.TiffFile(str(filepath))
            except TiffFileError as e:
                logger_meta.warning(f"Not a valid TIFF file: {e}")
            else:
                page = tif.pages[0]
                meta["image_shape"] = [int(page.shape[0]), int(page.shape[1])]

                # Store all TIFF tags
                meta["tiff_tags"] = {}
                for tag in page.tags.values():
                    try:
                        val = tag.value
                        # Make JSON-serializable
                        if isinstance(val, (bytes, np.ndarray)):
                            val = str(val)[:500]
                        elif isinstance(val, tuple):
                            val = list(val)
                        meta["tiff_tags"][tag.name] = val
                    except Exception:
                        pass

        elif fname.endswith(".npy"):
            arr = np.load(str(filepath))
            meta["image_shape"] = list(arr.shape)

        else:
            img = cv.imread(str(filepath))
            if img is not None:
                meta["image_shape"] = [img.shape[0], img.shape[1]]

    except Exception as e:
        logger_meta.warning(f"Metadata extraction failed (non-fatal): {e}")

    if "pixel_size" not in meta:
        meta["pixel_size"] = None
    if "pixel_unit" not in meta:
        meta["pixel_unit"] = None

    return meta


@router.get("/{session_id}/preview")
async def get_preview(session_id: str):
    path = SESSIONS_DIR / validate_session_id(session_id) / "original_preview.png"
    if not path.exists():
        return {"error": "No preview found"}
    return FileResponse(path)


@router.get("/{session_id}/metadata")
async def get_metadata(session_id: str):
    """Return session metadata (pixel scale, format, etc.)."""
    path = SESSIONS_DIR / validate_session_id(session_id) / "metadata.json"
    if not path.exists():
        return {"pixel_size": None, "pixel_unit": None}
    with open(path) as f:
        return json.load(f)


@router.put("/{session_id}/metadata")
async def update_metadata(session_id: str, req: UpdateMetadataRequest):
    """Update pixel size / unit in session metadata and recompute stats if available."""
    session_dir = SESSIONS_DIR / validate_session_id(session_id)
    meta_path = session_dir / "metadata.json"
    if not meta_path.exists():
        return {"error": "Session metadata not found"}

    with open(meta_path) as f:
        meta = json.load(f)

    if (
        req.pixel_size is None
        or not math.isfinite(req.pixel_size)
        or req.pixel_size <= 0
    ):
        raise HTTPException(
            status_code=422, detail="Pixel size must be a positive finite number"
        )
    unit_key = req.pixel_unit.strip().lower() if req.pixel_unit else ""
    unit = PIXEL_UNIT_ALIASES.get(unit_key)
    if unit is None:
        raise HTTPException(
            status_code=422,
            detail="Use a supported length unit: pm, nm, µm, Å, mm, or m",
        )
    calibration: dict[str, str | float] = {"source": req.calibration_source}
    if req.calibration_source == "scale_bar":
        pixels = req.scale_bar_pixels
        length = req.scale_bar_length
        if (
            pixels is None
            or length is None
            or not all(math.isfinite(value) and value > 0 for value in (pixels, length))
        ):
            raise HTTPException(
                status_code=422, detail="Scale bar length and pixel span are required"
            )
        if not math.isclose(req.pixel_size, length / pixels, rel_tol=1e-6):
            raise HTTPException(
                status_code=422, detail="Pixel size does not match the scale bar"
            )
        calibration["scale_bar_pixels"] = pixels
        calibration["scale_bar_length"] = length

    meta["pixel_size"] = req.pixel_size
    meta["pixel_unit"] = unit
    meta["calibration"] = calibration
    meta.pop("calibration_warning", None)

    # Compute before persisting the new scale so failed stats cannot leave stale units.
    stats_path = session_dir / "stats.json"
    stats = None
    if stats_path.exists():
        from app.api.instances import load_instances
        from app.models.helpers.compute_stats import compute_stats_from_instances

        cached = load_instances(session_dir)
        if cached is None:
            raise HTTPException(
                status_code=409, detail="Cannot update scale: particle data is missing"
            )
        instances, labeled = cached
        binary = (labeled > 0).astype(np.uint8)
        stats = compute_stats_from_instances(
            instances,
            binary,
            pixel_size=req.pixel_size,
            pixel_unit=unit,
            labeled_mask=labeled,
        )

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    if stats is not None:
        with open(stats_path, "w") as f:
            json.dump(stats, f)
    return {"metadata": meta, "stats": stats}


def _save_uploaded_image(file: UploadFile, session_id: str) -> dict:
    """Persist one uploaded file into its session dir; returns preview + info.
    Returns {"error": ...} when the format can't be loaded."""
    filename = _validated_upload_name(file.filename)
    session_dir = SESSIONS_DIR / validate_session_id(session_id)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    staged = SESSIONS_DIR / f".upload-{uuid.uuid4().hex}{Path(filename).suffix.lower()}"
    with staged.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info("Upload session: %s, filename: %s", session_id, file.filename)

    try:
        display = load_rgb_uint8(staged)
    except (ValueError, TiffFileError, ImportError, OSError) as error:
        staged.unlink(missing_ok=True)
        logger.warning("Could not decode image %s: %s", filename, error)
        return {"error": f"Could not read {filename}: {error}"}

    session_dir.mkdir(parents=True, exist_ok=False)
    dest = session_dir / f"org_{filename}"
    staged.replace(dest)

    preview_path = session_dir / "original_preview.png"
    if not cv.imwrite(str(preview_path), cv.cvtColor(display, cv.COLOR_RGB2BGR)):
        raise OSError(f"Could not save image preview: {preview_path}")
    preview_url = f"/images/{session_id}/preview"

    metadata = _extract_metadata(dest, filename)
    metadata["image_shape"] = [int(display.shape[0]), int(display.shape[1])]
    with dest.open("rb") as source:
        metadata["source_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
    meta_path = session_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "filename": filename,
        "preview_url": preview_url,
        "image_info": {
            "file_path": metadata.get("file_path"),
            "file_name": metadata.get("file_name"),
            "image_shape": metadata.get("image_shape"),
            "original_format": metadata.get("original_format"),
            "pixel_size": metadata.get("pixel_size"),
            "pixel_unit": metadata.get("pixel_unit"),
        },
    }


@router.post("/upload")
async def upload_image(request: Request, file: UploadFile = File(...)):
    session_id = uuid.uuid4().hex[:8]
    result = _save_uploaded_image(file, session_id)
    if "error" in result:
        return result
    update_mru(session_id)
    # YOLO is warmed at backend startup (see api/main.py lifespan) — the ONNX
    # graph is static-shape so warming against the uploaded image buys nothing.
    ui_event(
        "IMAGE_LOADED",
        "Image loaded — ready to segment.",
        level="info",
    )
    return {"session_id": session_id, **result}


def _natural_name_key(upload: UploadFile) -> list:
    name = upload.filename or ""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


@router.post("/upload-many")
async def upload_many_images(
    request: Request,
    files: list[UploadFile] = File(...),
    group_name: str | None = Form(None),
):
    """One session per file, plus a group tying them together in filename order.
    Unloadable files are skipped and reported in `warnings`."""
    if not files:
        raise HTTPException(status_code=400, detail="No files received")

    sessions: list[dict] = []
    warnings: list[dict] = []
    for f in sorted(files, key=_natural_name_key):
        ext = Path(f.filename or "").suffix.lower()
        if ext not in IMAGE_EXTENSIONS:
            warnings.append({"filename": f.filename, "error": "unsupported file type"})
            continue
        session_id = uuid.uuid4().hex[:8]
        result = _save_uploaded_image(f, session_id)
        if "error" in result:
            warnings.append({"filename": f.filename, "error": result["error"]})
            continue
        sessions.append({"session_id": session_id, **result})

    if not sessions:
        return {"error": "No loadable images in upload", "warnings": warnings}

    group = None
    if group_name and group_name.strip():
        group = create_group(group_name.strip(), [s["session_id"] for s in sessions])

    update_mru(sessions[0]["session_id"])
    ui_event(
        "IMAGES_LOADED",
        f"Loaded {len(sessions)} image(s)"
        + (f" into group {group['name']!r}" if group else ""),
        level="info",
    )
    return {"group": group, "sessions": sessions, "warnings": warnings}


""" 
Send response back to user for a given session id. 
Just sends the first file for now.
"""


@router.get("/{session_id}/file")
async def get_image(session_id: str):
    session_dir = SESSIONS_DIR / validate_session_id(session_id)
    image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy", ".emd"}
    files = [f for f in session_dir.glob("org_*") if f.suffix.lower() in image_exts]

    if not files:
        return {"error": f"No file exists for session id: {session_id}"}

    update_mru(session_id)

    file = files[0]

    logger.info(
        "Tried to get an image. session_id: %s, filename: %s",
        session_id,
        file.name,
    )

    return FileResponse(file)


@router.get("/{session_id}/mask")
async def get_mask(session_id: str):
    mask_path = SESSIONS_DIR / validate_session_id(session_id) / "mask.png"
    return FileResponse(mask_path)


@router.post("/{session_id}/gt")
async def upload_ground_truth(session_id: str, file: UploadFile = File(...)):
    filename = _validated_upload_name(file.filename)
    session_dir = SESSIONS_DIR / validate_session_id(session_id)
    if not session_dir.is_dir():
        raise HTTPException(status_code=404, detail="Session not found")

    dest = session_dir / f"ground_truth_{filename}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(
        "Tried to upload an ground truth for session_id: %s, filename: %s",
        session_id,
        file.filename,
    )
    return {"session_id": session_id, "filename": file.filename}


@router.get("/{session_id}/instances-debug")
async def get_instances_debug(session_id: str):
    path = SESSIONS_DIR / validate_session_id(session_id) / "instances_debug.png"
    if not path.exists():
        return {"error": "No debug image — call /masks/{session_id}/instances first"}
    return FileResponse(path)


@router.get("/{session_id}/yolo-boxes-debug")
async def get_yolo_boxes_debug(session_id: str):
    path = SESSIONS_DIR / validate_session_id(session_id) / "yolo_boxes_debug.png"
    if not path.exists():
        return {
            "error": "No YOLO boxes debug image — run segmentation with YoloSAM first"
        }
    return FileResponse(path)


@router.get("/{session_id}/sam-debug")
async def get_sam_debug(session_id: str):
    """Accumulated SAM prompt debug overlay, written by masks.from_points.
    Every click appends its point and its three candidate boundaries."""
    path = SESSIONS_DIR / validate_session_id(session_id) / "sam_debug.png"
    if not path.exists():
        return {"error": "No SAM debug image — click points in augment mode first"}
    return FileResponse(path)


@router.get("/{session_id}/proposals-debug")
async def get_proposals_debug(session_id: str):
    path = SESSIONS_DIR / validate_session_id(session_id) / "proposals_debug.png"
    if not path.exists():
        return {
            "error": "No proposals debug image — call /masks/{session_id}/propose-similar first"
        }
    return FileResponse(path)
