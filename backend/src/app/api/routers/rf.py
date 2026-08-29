from pathlib import Path

import cv2 as cv
import numpy as np
from fastapi import APIRouter, Request  # noqa: F401 — Request needed for app.state
from pydantic import BaseModel

from app.api.instances import extract_instances, load_instances
from app.api.live_models import AvailableModels
from app.api.utils import Stroke, mask_iou, strokes_to_mask
from app.logutils import Timer, get_logger
from app.models.helpers import rf_cache

router = APIRouter(prefix="/rf")
logger = get_logger("rf")
SESSIONS_DIR = Path("sessions")

# A user-marked bg scribble that's too small gives the RF almost no sense of
# what background looks like, so anything even slightly different from the
# handful of sampled pixels drifts to the foreground side.
# Below this floor we refuse to train rather than return junk.
MIN_BG_FRACTION = 0.04


class TrainRequest(BaseModel):
    session_id: str
    min_area: int | None = None  # None => derive from segmented instances


class ProposeRequest(BaseModel):
    session_id: str
    top_n: int = 5
    bg_scribbles: list[Stroke] | None = None
    # reject a proposal if IoU with any already-committed instance exceeds this
    # (matches /masks/{id}/propose-similar's iou_dedupe default)
    iou_dedupe: float = 0.2


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
        return {"error": "No mask found. Run segmentation first"}

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
    """
    t = Timer(logger, "rf propose")

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

    bg_mask = (
        strokes_to_mask(req.bg_scribbles, mask_gray.shape) if req.bg_scribbles else None
    )
    # Require marked background unconditionally.
    min_bg_px = int(MIN_BG_FRACTION * mask_gray.size)
    marked_px = int(bg_mask.sum()) if bg_mask is not None else 0
    if marked_px < min_bg_px:
        return {
            "error": (
                f"Not enough background marked ({marked_px}px, need at least "
                f"{min_bg_px}px). Mark a few background areas first so the RF "
                f"has both classes to train on."
            )
        }

    # Always retrain fresh: the cache is keyed only on session_id, so a stale
    # entry from a previous call would silently ignore mask edits (refine)
    # and any newly-drawn bg_scribbles.
    rf_cache.evict(req.session_id)
    rf = rf_cache.get_or_train(req.session_id, image, binary_mask, bg_mask=bg_mask)
    missed = rf.predict_missed_mask(image, binary_mask)

    if not np.any(missed):
        return {
            "proposals": [],
            "message": "RF found no missed regions",
            "elapsed": t.elapsed,
        }

    instances, _ = extract_instances(missed, session_dir, save=False)

    # Dedupe against the *rasterized final contour*, not the raw missed-pixel
    # mask. extract_instances traces contours with RETR_EXTERNAL, which only
    # follows a component's outer boundary and ignores holes. A ring/halo of
    # RF false positives around an already-segmented particle is a donut: the
    # raw missed mask correctly has a hole where that particle sits (excluded
    # via ~(mask>0) above), but its external-only contour collapses into a
    # solid filled disc — which is what actually gets rendered/committed, and
    # it swallows the particle whole even though the raw mask never touched
    # it. Rasterizing the committed contour (same way rasterize_instances
    # does) before the IoU check catches that.
    cached_existing = load_instances(session_dir)
    if cached_existing is not None:
        existing_instances, existing_labeled = cached_existing
        existing_ids = [inst["id"] for inst in existing_instances]
        deduped = []
        for inst in instances:
            contour = np.array(inst["contour"], dtype=np.int32)
            proposal_mask = np.zeros(existing_labeled.shape, dtype=np.uint8)
            cv.fillPoly(proposal_mask, [contour], color=1)
            proposal_mask = proposal_mask.astype(bool)
            worst_iou = max(
                (
                    mask_iou(proposal_mask, existing_labeled == eid)
                    for eid in existing_ids
                ),
                default=0.0,
            )
            if worst_iou > req.iou_dedupe:
                continue
            deduped.append(inst)
        n_rejected = len(instances) - len(deduped)
        if n_rejected:
            logger.info(
                f"Removed {n_rejected} RF proposal(s) that overlapped existing particles"
            )
        instances = deduped

    for inst in instances:
        inst["source"] = "rf"

    t.field("proposals", len(instances))
    t.stop()
    return {"proposals": instances, "elapsed": t.elapsed}


@router.delete("/cache/{session_key}")
async def evict_rf_cache(session_key: str):
    """Evict the cached RF classifier for a session."""
    rf_cache.evict(session_key)
    return {"status": "evicted", "session_key": session_key}
