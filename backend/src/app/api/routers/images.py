from typing import Annotated
import json
import uuid, shutil
from pathlib import Path 
from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import FileResponse
import logging
import numpy as np 
import cv2 as cv
from concurrent.futures import ThreadPoolExecutor
import asyncio
from app.api.live_models import AvailableModels

router = APIRouter(prefix="/images")
logger = logging.getLogger("routes.images")
SESSIONS_DIR = Path("sessions")


def _extract_metadata(filepath: Path, filename: str) -> dict:
    """
    Extract all available metadata from the uploaded file.
    """
    fname = filename.lower()
    meta = {
        "original_format": Path(filename).suffix.lstrip("."),
        "image_shape": None,
    }

    try:
        if fname.endswith(".emd"):
            import hyperspy.api as hs
            result = hs.load(str(filepath))
            s = result[0] if isinstance(result, list) else result
            meta["image_shape"] = list(s.data.shape)

            # Store all axes calibration
            meta["axes"] = []
            for ax in s.axes_manager.signal_axes:
                meta["axes"].append({
                    "name": ax.name,
                    "scale": float(ax.scale) if ax.scale else None,
                    "offset": float(ax.offset) if ax.offset else None,
                    "units": ax.units if ax.units else None,
                    "size": ax.size,
                })

            # Convenience fields
            if meta["axes"]:
                meta["pixel_size"] = meta["axes"][0].get("scale")
                meta["pixel_unit"] = meta["axes"][0].get("units")

            # Store all original metadata from the file
            if hasattr(s, "original_metadata"):
                try:
                    meta["original_metadata"] = s.original_metadata.as_dictionary()
                except Exception:
                    meta["original_metadata"] = str(s.original_metadata)

            if hasattr(s, "metadata"):
                try:
                    meta["hyperspy_metadata"] = s.metadata.as_dictionary()
                except Exception:
                    meta["hyperspy_metadata"] = str(s.metadata)

            logger.info(f"[META] EMD: pixel_size={meta.get('pixel_size')} {meta.get('pixel_unit')}")

        elif fname.endswith((".tif", ".tiff")):
            import tifffile
            with tifffile.TiffFile(str(filepath)) as tif:
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
        logger.warning(f"[META] Metadata extraction failed (non-fatal): {e}")

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


@router.post("/upload")
async def upload_image(request: Request, file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())[:4]
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    dest = session_dir / f"org_{file.filename}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info("[IMG] Upload session: %s, filename: %s", session_id, file.filename)

    metadata = _extract_metadata(dest, file.filename)
    meta_path = session_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"[IMG] Metadata saved: pixel_size={metadata.get('pixel_size')}, unit={metadata.get('pixel_unit')}")

    fname = file.filename.lower()
    arr = None
    preview_url = f"/images/{session_id}/file"

    logger.info(f"[IMG] preview url : {preview_url}")

    if fname.endswith(".npy"):
        arr = np.load(str(dest))
    elif fname.endswith((".tif", ".tiff")):
        import tifffile
        arr = tifffile.imread(str(dest))
    elif fname.endswith(".emd"):
        import hyperspy.api as hs
        result = hs.load(str(dest))
        s = result[0] if isinstance(result, list) else result
        arr = s.data

    if arr is not None:
        logger.info(f"[IMG] img shape: {arr.shape}")

        arr_min, arr_max = arr.min(), arr.max()
        display = ((arr - arr_min) / (arr_max - arr_min + 1e-8) * 255).astype("uint8") \
            if arr_max > arr_min else np.zeros_like(arr, dtype="uint8")

        preview_path = session_dir / "original_preview.png"
        cv.imwrite(str(preview_path), display)
        preview_url = f"/images/{session_id}/preview"

    async def warm_for_shape():
        yolosam = request.app.state.models.get(AvailableModels.yolosam.value)
        if not yolosam:
            return
        try:
            img = yolosam.load_image(dest)
            logger.info(f"[UPLOAD] Warming YOLO for shape: {img.shape}")
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as pool:
                await loop.run_in_executor(
                    pool,
                    lambda: yolosam.components["yolo"].predict(
                        source=img, verbose=False, conf=0.25
                    )
                )
            logger.info("[UPLOAD] YOLO warmup complete")
        except Exception as e:
            logger.warning(f"[UPLOAD] Warmup failed (non-fatal): {e}")

    asyncio.create_task(warm_for_shape())

    return {
        "session_id": session_id,
        "filename": file.filename,
        "preview_url": preview_url,
        "metadata": metadata,
    }


''' 
Send response back to user for a given session id. 
Just sends the first file for now.
'''
@router.get("/{session_id}/file")
async def get_image(session_id: str):
    session_dir = SESSIONS_DIR / session_id
    files = list(session_dir.glob("*"))

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
