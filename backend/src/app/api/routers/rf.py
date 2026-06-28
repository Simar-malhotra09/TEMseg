import logging
import time
from pathlib import Path

import cv2 as cv
import numpy as np
from fastapi import APIRouter, Request
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
    Run RF recovery and return missed regions as SAM-segmented proposals.
    Requires a prior /segment call so the SAM embedding is cached.
    Returns proposals in the same shape as /propose-similar so the
    frontend accept/reject flow works unchanged.
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

    embedding = request.app.state.embedding_cache.get(req.session_id)
    if embedding is None:
        return {"error": "SAM embedding not cached — run segmentation first"}

    model_inst = request.app.state.models.get(AvailableModels.yolosam)
    if model_inst is None:
        return {"error": "YoloSAM model not available"}

    image = model_inst.load_image(orig_files[0])

    mask_gray = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)
    if mask_gray is None:
        return {"error": "Could not read mask"}
    binary_mask = (mask_gray > 0).astype(np.uint8)

    rf = rf_cache.get_or_train(req.session_id, image, binary_mask)
    prompts = rf.get_prompts(image, binary_mask, top_n=req.top_n)

    if not prompts:
        return {"proposals": [], "message": "RF found no missed regions", "elapsed": time.perf_counter() - t0}

    logger.info(f"[RF-Propose] {len(prompts)} prompts for session={req.session_id}")

    # restore SAM predictor from cache so predict_from_prompts works
    if not hasattr(model_inst, "_predictor"):
        from segment_anything import SamPredictor
        model_inst._predictor = SamPredictor(model_inst.components["sam"])
    predictor = model_inst._predictor
    predictor.features = embedding.features
    predictor.original_size = embedding.original_size
    predictor.input_size = embedding.input_size
    predictor.is_image_set = True

    extra_mask = model_inst.predict_from_prompts(prompts)

    # keep only pixels not already in the existing mask
    extra_only = extra_mask & ~binary_mask

    if not np.any(extra_only):
        return {"proposals": [], "message": "All RF regions already covered by mask", "elapsed": time.perf_counter() - t0}

    instances, _ = extract_instances(extra_only, session_dir, save=False)

    # tag so the frontend can display a different colour / label if desired
    for inst in instances:
        inst["source"] = "rf"

    elapsed = time.perf_counter() - t0
    logger.info(f"[RF-Propose] returning {len(instances)} proposals in {elapsed:.2f}s")
    return {"proposals": instances, "elapsed": elapsed}


@router.delete("/cache/{session_key}")
async def evict_rf_cache(session_key: str):
    """Evict the cached RF classifier for a session."""
    rf_cache.evict(session_key)
    return {"status": "evicted", "session_key": session_key}
