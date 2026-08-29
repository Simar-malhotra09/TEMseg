import json
import uuid
import shutil
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import FileResponse
import numpy as np
import cv2 as cv
from concurrent.futures import ThreadPoolExecutor
import asyncio
import sys
import os
from app.api.live_models import AvailableModels
from app.logutils import get_logger
from pydantic import BaseModel

router = APIRouter(prefix="/images")
logger = get_logger("images")
logger_rsciio = get_logger("images", sub="rsciio")
logger_meta = get_logger("images", sub="meta")
logger_upload = get_logger("images", sub="upload")
SESSIONS_DIR = Path("sessions")


class UpdateMetadataRequest(BaseModel):
    pixel_size: float | None = None
    pixel_unit: str | None = None


def _ensure_rsciio_plugins():
    """
    Fix rsciio IO plugin discovery in frozen (PyInstaller) apps.

    HyperSpy 2.x discovers format readers via entry points at import time.
    In a frozen app, entry points don't work, so we manually walk the rsciio
    package directory for specifications.yaml files and inject them into
    BOTH rsciio.IO_PLUGINS and hyperspy.io.IO_PLUGINS.
    """
    import hyperspy.io
    import rsciio
    import yaml

    # If HyperSpy already has a full set of plugins, nothing to do.
    # (In a working non-frozen env, this would typically be 30+ plugins)
    if len(getattr(hyperspy.io, "IO_PLUGINS", [])) > 5:
        return

    logger_rsciio.info("Plugin registry incomplete — scanning for specifications.yaml")

    # Find the rsciio package directory (works inside _MEIPASS too)
    rsciio_dir = Path(rsciio.__file__).parent
    logger_rsciio.info(f"rsciio dir: {rsciio_dir}")

    # Also check _MEIPASS paths in case rsciio.__file__ doesn't point there
    search_dirs = [rsciio_dir]
    if getattr(sys, "frozen", False):
        bundle = Path(sys._MEIPASS)
        for candidate in [bundle / "rsciio", bundle / "lib" / "rsciio"]:
            if candidate.exists() and candidate != rsciio_dir:
                search_dirs.append(candidate)

    plugins = []
    seen_names = set()

    for base in search_dirs:
        if not base.exists():
            continue
        for dirpath, _, filenames in os.walk(str(base)):
            if "specifications.yaml" not in filenames:
                continue
            spec_file = os.path.join(dirpath, "specifications.yaml")
            try:
                with open(spec_file, "r") as f:
                    specs = yaml.safe_load(f)
                plugin_name = os.path.basename(dirpath)
                specs["api"] = f"rsciio.{plugin_name}"

                # Deduplicate
                name = specs.get("name", plugin_name)
                if name not in seen_names:
                    plugins.append(specs)
                    seen_names.add(name)
            except Exception as e:
                logger_rsciio.warning(f"Failed to load {spec_file}: {e}")

    logger_rsciio.info(f"Found {len(plugins)} plugins from specifications.yaml")

    # Log which formats we found (especially check for emd)
    for p in plugins:
        exts = p.get("file_extensions", [])
        logger_rsciio.info(f"  {p.get('name', '?')}: {exts}")

    # Inject into BOTH registries
    rsciio.IO_PLUGINS.clear()
    rsciio.IO_PLUGINS.extend(plugins)

    hyperspy.io.IO_PLUGINS.clear()
    hyperspy.io.IO_PLUGINS.extend(plugins)

    # Verify EMD is there
    emd_found = any(
        "emd" in [e.lower() for e in p.get("file_extensions", [])]
        for p in hyperspy.io.IO_PLUGINS
    )
    if emd_found:
        logger_rsciio.info(
            f"EMD reader registered successfully ({len(plugins)} total plugins)"
        )
    else:
        logger_rsciio.error("EMD reader NOT found after registration!")


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
            _ensure_rsciio_plugins()
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
                if meta["axes"]:
                    meta["pixel_size"] = meta["axes"][0].get("scale")
                    meta["pixel_unit"] = meta["axes"][0].get("units")

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

    if not meta.__contains__("pixel_size"):
        meta["pixel_size"] = "-"
    if not meta.__contains__("pixel_unit"):
        meta["pixel_unit"] = "-"

    return meta


@router.get("/{session_id}/preview")
async def get_preview(session_id: str):
    path = SESSIONS_DIR / session_id / "original_preview.png"
    if not path.exists():
        return {"error": "No preview found"}
    return FileResponse(path)


@router.get("/{session_id}/metadata")
async def get_metadata(session_id: str):
    """Return session metadata (pixel scale, format, etc.)."""
    path = SESSIONS_DIR / session_id / "metadata.json"
    if not path.exists():
        return {"pixel_size": None, "pixel_unit": None}
    with open(path) as f:
        return json.load(f)


@router.put("/{session_id}/metadata")
async def update_metadata(session_id: str, req: UpdateMetadataRequest):
    """Update pixel size / unit in session metadata and recompute stats if available."""
    session_dir = SESSIONS_DIR / session_id
    meta_path = session_dir / "metadata.json"
    if not meta_path.exists():
        return {"error": "Session metadata not found"}

    with open(meta_path) as f:
        meta = json.load(f)

    if req.pixel_size is not None:
        meta["pixel_size"] = req.pixel_size
    if req.pixel_unit is not None:
        meta["pixel_unit"] = req.pixel_unit

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Recompute stats if we have instances + mask so the UI stays consistent
    stats_path = session_dir / "stats.json"
    if stats_path.exists():
        try:
            from app.api.instances import load_instances
            from app.models.helpers.compute_stats import compute_stats_from_instances

            cached = load_instances(session_dir)
            if cached is not None:
                instances, labeled = cached
                binary = (labeled > 0).astype(np.uint8)
                stats = compute_stats_from_instances(
                    instances,
                    binary,
                    pixel_size=meta.get("pixel_size"),
                    pixel_unit=meta.get("pixel_unit"),
                    labeled_mask=labeled,
                )
                with open(stats_path, "w") as f:
                    json.dump(stats, f)
                logger_meta.info(
                    f"Recomputed stats after pixel_size update | session={session_id}"
                )
                return {"metadata": meta, "stats": stats}
        except Exception as e:
            logger_meta.warning(
                f"Stats recompute failed after pixel_size update: {e}"
            )

    return {"metadata": meta}


@router.post("/upload")
async def upload_image(request: Request, file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())[:4]
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    dest = session_dir / f"org_{file.filename}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info("Upload session: %s, filename: %s", session_id, file.filename)

    metadata = _extract_metadata(dest, file.filename)
    meta_path = session_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(
        f"Metadata saved: pixel_size={metadata.get('pixel_size')}, unit={metadata.get('pixel_unit')}"
    )

    fname = file.filename.lower()
    arr = None
    preview_url = f"/images/{session_id}/file"

    logger.info(f"preview url: {preview_url}")

    if fname.endswith(".npy"):
        arr = np.load(str(dest))
    elif fname.endswith((".tif", ".tiff")):
        import tifffile
        from tifffile import TiffFileError

        try:
            arr = tifffile.imread(str(dest))
        except TiffFileError as e:
            logger.warning(f"Invalid TIFF/TIF file: {e}")
            return {"error": "Not a valid TIFF/TIF file!"}
    elif fname.endswith(".emd"):
        try:
            _ensure_rsciio_plugins()
            from rsciio.emd import file_reader

            signals = file_reader(str(dest))
            s = signals[0]
            arr = s["data"]
        except ImportError:
            logger.warning("HyperSpy not available — cannot load EMD files")
            return {
                "error": "EMD format requires HyperSpy which is not available in this build"
            }
    elif fname.endswith((".png", ".jpg", ".jpeg")):
        img = cv.imread(str(dest))
        if img is not None:
            # save as preview directly
            preview_path = session_dir / "original_preview.png"
            cv.imwrite(str(preview_path), img)
            preview_url = f"/images/{session_id}/preview"
    if arr is not None:
        logger.info(f"img shape: {arr.shape}")

        arr_min, arr_max = arr.min(), arr.max()
        display = (
            ((arr - arr_min) / (arr_max - arr_min + 1e-8) * 255).astype("uint8")
            if arr_max > arr_min
            else np.zeros_like(arr, dtype="uint8")
        )

        preview_path = session_dir / "original_preview.png"
        cv.imwrite(str(preview_path), display)
        preview_url = f"/images/{session_id}/preview"

    async def warm_for_shape():
        yolosam = request.app.state.models.get(AvailableModels.yolosam.value)
        if not yolosam:
            return
        try:
            img = yolosam.load_image(dest)
            logger_upload.info(f"Warming YOLO for shape: {img.shape}")
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as pool:
                await loop.run_in_executor(
                    pool,
                    lambda: yolosam.components["yolo"].predict(
                        source=img,
                        verbose=False,
                        conf=0.25,
                        device=yolosam.device,
                    ),
                )
            logger_upload.info("YOLO warmup complete")
        except Exception as e:
            logger_upload.warning(f"Warmup failed (non-fatal): {e}")

    asyncio.create_task(warm_for_shape())

    return {
        "session_id": session_id,
        "filename": file.filename,
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


""" 
Send response back to user for a given session id. 
Just sends the first file for now.
"""


@router.get("/{session_id}/file")
async def get_image(session_id: str):
    session_dir = SESSIONS_DIR / session_id
    image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy", ".emd"}
    files = [f for f in session_dir.glob("org_*") if f.suffix.lower() in image_exts]

    if not files:
        return {"error": f"No file exists for session id: {session_id}"}

    file = files[0]

    logger.info(
        "Tried to get an image. session_id: %s, filename: %s",
        session_id,
        file.name,
    )

    return FileResponse(file)


@router.get("/{session_id}/mask")
async def get_mask(session_id: str):
    mask_path = SESSIONS_DIR / session_id / "mask.png"
    return FileResponse(mask_path)


@router.post("/{session_id}/gt")
async def upload_ground_truth(session_id: str, file: UploadFile = File(...)):
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    dest = session_dir / file.filename
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
    path = SESSIONS_DIR / session_id / "instances_debug.png"
    if not path.exists():
        return {"error": "No debug image — call /masks/{session_id}/instances first"}
    return FileResponse(path)


@router.get("/{session_id}/yolo-boxes-debug")
async def get_yolo_boxes_debug(session_id: str):
    path = SESSIONS_DIR / session_id / "yolo_boxes_debug.png"
    if not path.exists():
        return {
            "error": "No YOLO boxes debug image — run segmentation with YoloSAM first"
        }
    return FileResponse(path)


@router.get("/{session_id}/proposals-debug")
async def get_proposals_debug(session_id: str):
    path = SESSIONS_DIR / session_id / "proposals_debug.png"
    if not path.exists():
        return {
            "error": "No proposals debug image — call /masks/{session_id}/propose-similar first"
        }
    return FileResponse(path)
