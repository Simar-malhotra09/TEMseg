import logging
import time
from pathlib import Path

import cv2 as cv
import numpy as np
from fastapi import APIRouter, Request  # noqa: F401 — Request needed for app.state
from pydantic import BaseModel

from app.api.instances import extract_instances
from app.api.live_models import AvailableModels
from app.models.helpers import rf_cache

router = APIRouter(prefix="/rf")
logger = logging.getLogger("routes.rf")
SESSIONS_DIR = Path("sessions")


class TrainRequest(BaseModel):
    session_id: str
    min_area: int = 50


class ProposeRequest(BaseModel):
    session_id: str
    top_n: int = 5


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


@router.post("/propose")
async def rf_propose(req: ProposeRequest, request: Request):
    """
    Run RF pixel classifier and return missed regions as proposals directly.
    No SAM — the RF mask IS the segmentation.
    """
    t0 = time.perf_counter()

    session_dir = SESSIONS_DIR / req.session_id
    if not session_dir.exists():
        return {"error": "Invalid session"}

    orig_files = list(session_dir.glob("org_*"))
    if not orig_files:
        return {"error": "No image found in session"}

    mask_path = session_dir / "mask.png"
    if not mask_path.exists():
        return {"error": "No mask — run segmentation first"}

    model_inst = request.app.state.models.get(AvailableModels.yolosam)
    if model_inst is None:
        return {"error": "YoloSAM model not available"}

    image = model_inst.load_image(orig_files[0])

    mask_gray = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)
    if mask_gray is None:
        return {"error": "Could not read mask"}
    binary_mask = (mask_gray > 0).astype(np.uint8)

    rf = rf_cache.get_or_train(req.session_id, image, binary_mask)
    missed = rf.predict_missed_mask(image, binary_mask)

    if not np.any(missed):
        return {"proposals": [], "message": "RF found no missed regions", "elapsed": time.perf_counter() - t0}

    instances, _ = extract_instances(missed, session_dir, save=False)
    for inst in instances:
        inst["source"] = "rf"

    elapsed = time.perf_counter() - t0
    logger.info(f"[RF-Propose] {len(instances)} proposals in {elapsed:.2f}s")
    return {"proposals": instances, "elapsed": elapsed}


@router.delete("/cache/{session_key}")
async def evict_rf_cache(session_key: str):
    """Evict the cached RF classifier for a session."""
    rf_cache.evict(session_key)
    return {"status": "evicted", "session_key": session_key}
