import logging
from pathlib import Path

import cv2 as cv
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel

from app.models.helpers import rf_cache

router = APIRouter(prefix="/rf")
logger = logging.getLogger("routes.rf")
SESSIONS_DIR = Path("sessions")


class TrainRequest(BaseModel):
    session_id: str
    min_area: int = 50


@router.post("/train")
async def train_rf(req: TrainRequest):
    """Manually trigger RF training on an already-segmented session."""
    session_dir = SESSIONS_DIR / req.session_id
    if not session_dir.exists():
        return {"error": "Invalid session"}

    orig_files = list(session_dir.glob("org_*"))
    if not orig_files:
        return {"error": "No image found in session"}

    mask_path = session_dir / "mask.png"
    if not mask_path.exists():
        return {"error": "No mask found — run segmentation first"}

    # load image via cv2 (mask.png is always a uint8 BGR/gray)
    image_bgr = cv.imread(str(orig_files[0]))
    if image_bgr is None:
        return {"error": f"Could not read image: {orig_files[0]}"}
    image = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)

    mask_gray = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)
    if mask_gray is None:
        return {"error": "Could not read mask"}

    rf_cache.evict(req.session_id)
    rf_cache.get_or_train(req.session_id, image, mask_gray, min_area=req.min_area)

    return {"status": "trained", "session_id": req.session_id}


@router.delete("/cache/{session_key}")
async def evict_rf_cache(session_key: str):
    """Evict the cached RF classifier for a session."""
    rf_cache.evict(session_key)
    return {"status": "evicted", "session_key": session_key}
