from typing import List, Dict, Any
from pathlib import Path
from fastapi import APIRouter, Request
import logging
import time
import cv2 as cv
import numpy as np

# from app.models.base_model import  StatType, StatsConfig
from app.api.live_models import AvailableModels
from app.api.utils import (
    Box,
    normalize_mask,
    blackout_regions,
    inverse_blackout_regions,
    colorize_components_inplace,
    batch_seg_patches,
)
from app.api.instances import extract_instances


from pydantic import BaseModel


class SegmentRequest(BaseModel):
    session_id: str
    model: AvailableModels
    regions: List[Box] = None
    blackout: bool = False
    inverse_blackout: bool = False
    colorize: bool = True  # colorize masks for overlay


class SegmentResponse(BaseModel):
    model: AvailableModels
    mask_url: str
    metadata: Dict[str, Any]
    stats: dict
    time_elapsed: float


router = APIRouter(prefix="/segment")
logger = logging.getLogger("routes.segment")
SESSIONS_DIR = Path("sessions")
# logger.disabled= True


@router.post("/")
async def segment(req: SegmentRequest, request: Request):
    t_req_start = time.perf_counter()

    model_inst = request.app.state.models.get(req.model.value)
    cache = request.app.state.embedding_cache
    embedding = cache.get(req.session_id)

    if not model_inst:
        return {"error": f"Model {req.model} not found"}

    if req.blackout and req.inverse_blackout:
        raise ValueError(
            "Only one of blackout_regions or inverse_blackout_regions may be True."
        )

    logger.info(f"[SEG] Request received for session: {req.session_id}")
    logger.info(f"[SEG] Model requested: {req.model}")
    logger.info(f"[SEG] Regions selected : {len(req.regions)}")
    logger.info(f"[SEG] Regions are being {'blacked_out' if req.blackout else 'kept'}")
    logger.info(f"[SEG] colorize: {req.colorize}")

    session_dir = SESSIONS_DIR / req.session_id
    if not session_dir.exists():
        return {"error": "Invalid session"}

    orig_files = list(session_dir.glob("org_*"))
    if not orig_files:
        return {"error": "No image found"}

    image_path = orig_files[0]

    # ── image load ───────────────────────────────────────────────
    t0 = time.perf_counter()
    img = model_inst.load_image(image_path)
    t_load = time.perf_counter() - t0
    logger.info(f"[SEG-TIMING] image_load={t_load:.3f}s  shape={img.shape}")

    # ── inference ────────────────────────────────────────────────
    t1 = time.perf_counter()

    if req.inverse_blackout:
        if req.model == AvailableModels.yolosam:
            logger.info(f"[SEG] Batch seg on {len(req.regions)} patches")
            result = batch_seg_patches(img, req.regions, model_inst)
        elif req.model == AvailableModels.maskrcnn:
            img = inverse_blackout_regions(
                img,
                req.regions,
                save_path=f"sessions/{req.session_id}/inverse_blackout_check.png",
            )
            result = model_inst.segment(img)
        else:
            return {"error": f"Unsupported model {req.model}"}
    else:
        if req.blackout:
            logger.info(f"[SEG] Applying blackout to {len(req.regions)} regions")
            img = blackout_regions(
                img,
                req.regions,
                save_path=f"sessions/{req.session_id}/blackout_check.png",
            )
        if req.model == AvailableModels.yolosam:
            result = model_inst.segment(img, embedding)
        else:
            result = model_inst.segment(img)

    t_inference = time.perf_counter() - t1
    logger.info(f"[SEG-TIMING] inference={t_inference:.3f}s")

    # ── validation ───────────────────────────────────────────────
    if req.model != result.model:
        return {
            "error": f"Mismatch between requested model {req.model} and result {result.model}"
        }
    if result.segmentation_mask is None:
        return {"error": "Segmentation returned no mask"}

    # ── mask normalize + colorize ────────────────────────────────
    t2 = time.perf_counter()
    mask = normalize_mask(result.segmentation_mask)
    # morphological cleanup to remove single-pixel noise and close small gaps
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    if mask is None or mask.size == 0:
        return {"error": "Mask is empty"}

    if not np.any(mask):
        return {
            "mask_url": None,
            "metadata": result.metadata,
            "stats": {},
            "model": req.model,
            "warning": "Mask contains no detected particles",
        }

    save_mask = colorize_components_inplace(mask) if req.colorize else mask
    t_colorize = time.perf_counter() - t2
    logger.info(f"[SEG-TIMING] normalize+colorize={t_colorize:.3f}s")

    # ── save mask ────────────────────────────────────────────────
    t3 = time.perf_counter()
    mask_path = session_dir / "mask.png"
    success = cv.imwrite(str(mask_path), save_mask)
    t_save = time.perf_counter() - t3
    logger.info(f"[SEG-TIMING] mask_save={t_save:.3f}s")

    if not success:
        return {"error": "Failed to save mask"}

    # ── cache embedding ──────────────────────────────────────────
    if hasattr(result, "embedding") and result.embedding is not None:
        cache[req.session_id] = result.embedding
        logger.info(f"[SEG] Cached SAM embedding for session {req.session_id}")

    elapsed = time.perf_counter() - t_req_start

    # ── save the instances in instance.json ──────────────────────────────────────────
    logger.info(f"[SEG] Pre-computing instances for session={req.session_id}")
    try:
        binary = (mask > 0).astype(np.uint8)
        instances, labeld = extract_instances(binary, session_dir, save=True)
        logger.info(f"[SEG] Pre-computed {len(instances)} instances and saved to disk")
    except Exception as e:
        # non-fatal — instances can be recomputed on demand in GET /masks/.../instances
        logger.warning(f"[SEG] Instance pre-computation failed (non-fatal): {e}")

    # ── stats ────────────────────────────────────────────────────
    t4 = time.perf_counter()

    # load pixel scale from session metadata (if available)
    pixel_size = None
    pixel_unit = None
    meta_path = session_dir / "metadata.json"
    if meta_path.exists():
        import json

        with open(meta_path) as f:
            meta = json.load(f)
        pixel_size = meta.get("pixel_size")
        pixel_unit = meta.get("pixel_unit")

    # stats_results = compute_stats_from_instances(
    #     instances, mask, pixel_size=pixel_size, pixel_unit=pixel_unit
    # )

    stats_results: dict = model_inst.compute_stats(
        instances, mask, pixel_size, pixel_unit, labeld
    )

    t_stats = time.perf_counter() - t4
    logger.info(
        f"[SEG-TIMING] stats={t_stats:.3f}s | pixel_size={pixel_size} {pixel_unit}"
    )

    logger.info(
        f"[SEG-TIMING] TOTAL={elapsed:.3f}s  "
        f"(load={t_load:.3f} | infer={t_inference:.3f} | "
        f"colorize={t_colorize:.3f} | save={t_save:.3f} | stats={t_stats:.3f})"
    )

    # save stats to file
    if stats_results:
        stats_path = session_dir / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats_results, f)
        logger.info(f"[SEG] Stats saved to {stats_path}")

    return SegmentResponse(
        mask_url=f"/images/{req.session_id}/mask",
        metadata=result.metadata,
        stats=stats_results,
        model=req.model,
        time_elapsed=elapsed,
    )
