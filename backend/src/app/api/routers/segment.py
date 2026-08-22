from typing import List, Dict, Any
from pathlib import Path
from fastapi import APIRouter, Request
import logging
import time
import cv2 as cv
import numpy as np

# from app.models.base_model import  StatType, StatsConfig
from app.api.live_models import AvailableModels
from app.api.model_registry import get_or_load_model
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
    debug_boxes_url: str | None = None


router = APIRouter(prefix="/segment")
logger = logging.getLogger("routes.segment")
SESSIONS_DIR = Path("sessions")
# logger.disabled= True


@router.post("/")
async def segment(req: SegmentRequest, request: Request):
    t_req_start = time.perf_counter()

    model_inst = get_or_load_model(request.app.state.models, req.model)
    cache = request.app.state.embedding_cache
    embedding = cache.get(req.session_id)

    if req.blackout and req.inverse_blackout:
        raise ValueError(
            "Only one of blackout_regions or inverse_blackout_regions may be True."
        )

    action = (
        "blackout"
        if req.blackout
        else "keep"
        if not req.inverse_blackout
        else "patch-seg"
    )
    logger.info(
        f"Segmentation request: session={req.session_id}, model={req.model.value}, "
        f"regions={len(req.regions)}, mode={action}"
    )

    session_dir = SESSIONS_DIR / req.session_id
    if not session_dir.exists():
        return {"error": "Invalid session"}

    orig_files = list(session_dir.glob("org_*"))
    if not orig_files:
        return {"error": "No image found"}

    image_path = orig_files[0]

    # load image
    t0 = time.perf_counter()
    img = model_inst.load_image(image_path)
    t_load = time.perf_counter() - t0
    logger.info(f"Loaded image in {t_load * 1000:.1f}ms, shape {img.shape}")

    # Resolve req bodies before inferencing
    t1 = time.perf_counter()

    if req.inverse_blackout:
        # inverese blackout reqs to keep only the selected regions
        # and ignore all else. Suitable for batch segmentation
        if req.model == AvailableModels.yolosam:
            logger.info(f"Running batch segmentation on {len(req.regions)} patches")
            result = batch_seg_patches(img, req.regions, model_inst)
        # yolomaskrcnn and maskrcnn_synthetic do not support batch segmentation
        elif req.model in (
            AvailableModels.yolomaskrcnn,
            AvailableModels.maskrcnn_synthetic,
        ):
            img = inverse_blackout_regions(
                img,
                req.regions,
                save_path=f"sessions/{req.session_id}/inverse_blackout_check.png",
            )
            result = model_inst.segment(img)
        else:
            return {"error": f"Unsupported model {req.model}"}

    else:
        # blackout reqs ignores the req selected regions and keeps
        # everything else.
        if req.blackout:
            logger.info(f"Applying blackout to {len(req.regions)} regions")
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
    logger.info(f"Model inference took {t_inference * 1000:.1f}ms")

    # validation model which served req
    if req.model != result.model:
        return {
            "error": f"Mismatch between requested model {req.model} and result {result.model}"
        }
    if result.segmentation_mask is None:
        return {"error": "Segmentation returned no mask"}

    # mask normalize + colorize
    t2 = time.perf_counter()
    mask = normalize_mask(result.segmentation_mask)
    # morphological cleanup to remove single-pixel noise and close small gaps
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    if mask is None or mask.size == 0:
        return {"error": "Mask is empty"}

    if not np.any(mask):
        # Cache embedding even on empty result so user can bootstrap manually
        # via /masks/{id}/from-points and /masks/{id}/propose-similar.
        if hasattr(result, "embedding") and result.embedding is not None:
            cache[req.session_id] = result.embedding
            logger.info(
                f"Cached SAM embedding for session {req.session_id} (no particles detected)"
            )
        return {
            "mask_url": None,
            "metadata": result.metadata,
            "stats": {},
            "model": req.model,
            "warning": "No particles were detected! ",
        }

    save_mask = colorize_components_inplace(mask) if req.colorize else mask
    t_colorize = time.perf_counter() - t2
    logger.info(f"Prepared result mask in {t_colorize * 1000:.1f}ms")

    # save mask
    t3 = time.perf_counter()
    mask_path = session_dir / "mask.png"
    success = cv.imwrite(str(mask_path), save_mask)
    t_save = time.perf_counter() - t3
    logger.info(f"Saved mask image in {t_save * 1000:.1f}ms")

    if not success:
        return {"error": "Failed to save mask"}

    # ── save YOLO debug boxes overlay ────────────────────────────
    debug_boxes_url = None
    if (
        hasattr(result, "detection_boxes")
        and result.detection_boxes is not None
        and len(result.detection_boxes) > 0
    ):
        debug_img = img.copy()
        if debug_img.ndim == 2:
            debug_img = cv.cvtColor(debug_img, cv.COLOR_GRAY2BGR)
        elif debug_img.shape[2] == 3 and debug_img.dtype != np.uint8:
            debug_img = (
                (debug_img - debug_img.min())
                / (debug_img.max() - debug_img.min() + 1e-8)
                * 255
            ).astype(np.uint8)
        for box in result.detection_boxes.astype(int):
            x1, y1, x2, y2 = box
            cv.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        boxes_path = session_dir / "yolo_boxes_debug.png"
        cv.imwrite(str(boxes_path), debug_img)
        debug_boxes_url = f"/images/{req.session_id}/yolo-boxes-debug"
        logger.info(f"Saved detection debug overlay to {boxes_path.name}")

    # ── cache embedding ──────────────────────────────────────────
    if hasattr(result, "embedding") and result.embedding is not None:
        cache[req.session_id] = result.embedding
        logger.info(f"Cached SAM embedding for session {req.session_id}")

    # evict stale RF so /rf/propose trains fresh on the new mask
    from app.models.helpers import rf_cache

    rf_cache.evict(req.session_id)

    elapsed = time.perf_counter() - t_req_start

    # ── save the instances in instance.json ──────────────────────────────────────────
    logger.info(f"Extracting particles for session {req.session_id}")
    try:
        binary = (mask > 0).astype(np.uint8)
        sam_epsilon_scale = 0.75 if result.model == AvailableModels.yolosam else 1.0
        instances, labeld = extract_instances(
            binary, session_dir, save=True, epsilon_scale=sam_epsilon_scale
        )
        logger.info(f"Extracted and saved {len(instances)} particles")
    except Exception as e:
        # non-fatal — instances can be recomputed on demand in GET /masks/.../instances
        logger.warning(f"Particle extraction skipped (non-fatal): {e}")

    # comute stats
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
        f"Computed statistics in {t_stats * 1000:.1f}ms "
        f"(pixel size {pixel_size} {pixel_unit})"
    )

    logger.info(
        f"Segmentation complete in {elapsed * 1000:.1f}ms "
        f"(load {t_load * 1000:.1f}ms | inference {t_inference * 1000:.1f}ms | "
        f"mask {t_colorize * 1000:.1f}ms | save {t_save * 1000:.1f}ms | stats {t_stats * 1000:.1f}ms)"
    )

    # save stats to file
    if stats_results:
        stats_path = session_dir / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats_results, f)
        logger.info(f"Saved statistics to {stats_path.name}")

    return SegmentResponse(
        mask_url=f"/images/{req.session_id}/mask",
        metadata=result.metadata,
        stats=stats_results,
        model=req.model,
        time_elapsed=elapsed,
        debug_boxes_url=debug_boxes_url,
    )
