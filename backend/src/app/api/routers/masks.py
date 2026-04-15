from fastapi import APIRouter, HTTPException, Request
from pathlib import Path
from pydantic import BaseModel
from typing import List, Tuple
import numpy as np
import cv2 as cv
import logging
import time
from app.api.live_models import AvailableModels

# from app.api.utils import extract_instances, rasterize_instances, save_debug_overlay
from app.api.instances import (
    load_instances,
    extract_instances,
    save_instances,
    rasterize_instances,
    colorize_labeled_mask,
)

import json
from app.models.helpers.compute_stats import compute_stats_from_instances

router = APIRouter(prefix="/masks")
logger = logging.getLogger("routes.masks")
SESSIONS_DIR = Path("sessions")


def _session_dir(session_id: str) -> Path:
    d = SESSIONS_DIR / session_id
    if not d.exists():
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return d


class Instance(BaseModel):
    id: int
    contour: List[Tuple[float, float]]  # [[x,y], ...]
    bbox: dict  # {x, y, w, h}
    area: float


class SaveInstancesRequest(BaseModel):
    instances: list[dict]


class SplitRequest(BaseModel):
    instance_id: int
    points: list[list[float]]  # [[x, y], [x, y], ...]


class InstancesResponse(BaseModel):
    instances: List[Instance]


@router.post("/{session_id}/instances")
async def get_instances(session_id: str):
    logger.info(f"[MASKS] GET instances | session={session_id}")
    session_dir = _session_dir(session_id)

    # fast path — already on disk
    cached = load_instances(session_dir)
    if cached is not None:
        instances, _ = cached
        logger.info(f"[MASKS] Returning {len(instances)} cached instances")
        return {"instances": instances}

    # recompute from mask.png
    mask_path = session_dir / "mask.png"
    if not mask_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No segmentation mask found — run segmentation first",
        )

    logger.info("[MASKS] No cached instances, recomputing from mask.png")
    mask_bgr = cv.imread(str(mask_path))
    if mask_bgr is None:
        raise HTTPException(status_code=500, detail="Failed to read mask.png")

    binary = (cv.cvtColor(mask_bgr, cv.COLOR_BGR2GRAY) > 0).astype(np.uint8)
    instances, labeled = extract_instances(binary, session_dir, save=True)

    logger.info(f"[MASKS] Recomputed and saved {len(instances)} instances")
    return {"instances": instances}


@router.put("/{session_id}/instances")
async def api_save_instances(session_id: str, req: SaveInstancesRequest):
    logger.info(
        f"[MASKS] POST save | session={session_id} | {len(req.instances)} instances"
    )
    session_dir = _session_dir(session_id)

    cached = load_instances(session_dir)
    if cached is not None:
        _, labeled_old = cached
        shape = labeled_old.shape
    else:
        mask_path = session_dir / "mask.png"
        if not mask_path.exists():
            raise HTTPException(
                status_code=404, detail="No mask found to infer image shape"
            )
        mask_bgr = cv.imread(str(mask_path))
        shape = mask_bgr.shape[:2]

    logger.info(f"[MASKS] Rasterizing into shape {shape}")
    labeled = rasterize_instances(req.instances, shape)
    save_instances(session_dir, req.instances, labeled)

    colored = colorize_labeled_mask(labeled)
    mask_path = session_dir / "mask.png"
    cv.imwrite(str(mask_path), colored)
    logger.info(f"[MASKS] Regenerated mask.png | path={mask_path}")

    # recompute stats from updated instances
    pixel_size = None
    pixel_unit = None
    meta_path = session_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        pixel_size = meta.get("pixel_size")
        pixel_unit = meta.get("pixel_unit")

    binary = (labeled > 0).astype(np.uint8)
    stats = compute_stats_from_instances(
        req.instances,
        binary,
        pixel_size=pixel_size,
        pixel_unit=pixel_unit,
        labeled_mask=labeled,
    )
    # save new stats
    stats_path = session_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f)

    logger.info(f"[MASKS] Stats saved to {stats_path}")
    logger.info(f"[MASKS] Recomputed stats | {stats['particle_count']} particles")

    return {"mask_url": f"/images/{session_id}/mask", "stats": stats}


@router.post("/{session_id}/split")
async def split_instance(session_id: str, body: SplitRequest, request: Request):
    """
    Split a single instance into N instances using SAM point prompts.
    One point per expected particle. Uses cached SAM embedding — same session only.

    Steps:
      1. Load instances.npy to get the original blob mask for constraining SAM
      2. Restore SAM predictor from embedding cache
      3. For each point, run SAM predict with a single foreground point
      4. AND opr each result against the original blob mask to prevent bleed
      5. Extract contours from each result
      6. Assign new IDs, update instances list, save to disk
      7. Regenerate mask.png
    """
    t_start = time.time()
    logger.info(
        f"[SPLIT] POST split | session={session_id} | instance_id={body.instance_id} | {len(body.points)} points"
    )

    session_dir = _session_dir(session_id)

    # ── 1. load instances ────────────────────────────────────────────────────
    cached = load_instances(session_dir)
    if cached is None:
        raise HTTPException(
            status_code=404,
            detail="No instances found — run segmentation and enter refine mode first",
        )

    instances, labeled = cached

    # find the instance being split
    target = next((i for i in instances if i["id"] == body.instance_id), None)
    if target is None:
        raise HTTPException(
            status_code=404, detail=f"Instance {body.instance_id} not found"
        )

    logger.info(
        f"[SPLIT] Target instance {body.instance_id} | area={target['area']} | bbox={target['bbox']}"
    )

    # extract original blob mask — used to constrain SAM output
    blob_mask = (labeled == body.instance_id).astype(np.uint8)
    logger.info(f"[SPLIT] Blob mask | nonzero pixels={int(np.sum(blob_mask))}")

    # ── 2. get SAM predictor from cache ──────────────────────────────────────
    embedding_cache = getattr(request.app.state, "embedding_cache", {})
    embedding = embedding_cache.get(session_id)

    if embedding is None:
        raise HTTPException(
            status_code=400,
            detail="SAM embedding not cached for this session — re-run segmentation to restore it",
        )

    logger.info(f"[SPLIT] SAM embedding found in cache | session={session_id}")

    # get yolosam model from app state
    models = request.app.state.models
    yolosam = models.get(AvailableModels.yolosam.value)
    if yolosam is None:
        raise HTTPException(status_code=500, detail="YoloSAM model not loaded")

    # restore predictor state from cached embedding
    predictor = yolosam._predictor
    predictor.features = embedding.features
    predictor.original_size = embedding.original_size
    predictor.input_size = embedding.input_size
    predictor.is_image_set = True
    logger.info("[SPLIT] SAM predictor state restored from cache")

    # ── 3. run SAM for each point ─────────────────────────────────────────────
    new_masks = []
    for i, (px, py) in enumerate(body.points):
        logger.info(
            f"[SPLIT] Running SAM predict | point {i + 1}/{len(body.points)} | ({px:.1f}, {py:.1f})"
        )
        t_pred = time.time()

        point_coords = np.array([[px, py]])
        point_labels = np.array([1])  # 1 = foreground

        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,  # SAM returns 3 masks, we take best
        )

        # take highest-scoring mask
        best_idx = int(np.argmax(scores))
        raw_mask = masks[best_idx].astype(np.uint8)

        logger.info(
            f"[SPLIT] SAM predict done | point {i + 1} | score={scores[best_idx]:.3f} | elapsed={time.time() - t_pred:.2f}s"
        )

        # ── 4. constrain to original blob ────────────────────────────────────
        constrained = cv.bitwise_and(raw_mask, blob_mask)
        pixel_count = int(np.sum(constrained))
        logger.info(
            f"[SPLIT] Constrained mask | point {i + 1} | pixels={pixel_count} (raw={int(np.sum(raw_mask))})"
        )

        if pixel_count < 10:
            logger.warning(
                f"[SPLIT] Point {i + 1} produced near-empty mask after constraint — skipping"
            )
            continue

        new_masks.append(constrained)

    if not new_masks:
        raise HTTPException(
            status_code=400,
            detail="All split points produced empty masks — try placing points more centrally",
        )

    logger.info(
        f"[SPLIT] Got {len(new_masks)} valid masks from {len(body.points)} points"
    )

    # ── 5. extract contours from each mask ───────────────────────────────────
    max_id = max(i["id"] for i in instances)
    new_instances = []

    for i, mask in enumerate(new_masks):
        new_id = max_id + i + 1
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.warning(f"[SPLIT] No contour found for mask {i + 1}, skipping")
            continue

        largest = max(contours, key=cv.contourArea)
        epsilon = 0.01 * cv.arcLength(largest, True)
        contour = cv.approxPolyDP(largest, epsilon, True).squeeze()

        if contour.ndim < 2 or len(contour) < 3:
            logger.warning(f"[SPLIT] Contour {i + 1} too small, skipping")
            continue

        x, y, w, h = cv.boundingRect(contours[0])
        area = int(np.sum(mask))

        inst = {
            "id": new_id,
            "contour": contour.tolist(),
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "area": area,
        }
        new_instances.append(inst)
        logger.info(
            f"[SPLIT] New instance | id={new_id} | area={area} | contour_pts={len(contour)}"
        )

    if not new_instances:
        raise HTTPException(
            status_code=400, detail="Could not extract valid contours from split masks"
        )

    # ── 6. update instances list and labeled mask ─────────────────────────────
    updated_instances = [
        i for i in instances if i["id"] != body.instance_id
    ] + new_instances
    logger.info(
        f"[SPLIT] Updated instance list | before={len(instances)} | after={len(updated_instances)}"
    )

    # update labeled mask — clear old blob, paint new ones
    labeled[labeled == body.instance_id] = 0
    for inst in new_instances:
        contour = np.array(inst["contour"], dtype=np.int32)
        cv.fillPoly(labeled, [contour], color=inst["id"])

    save_instances(session_dir, updated_instances, labeled)

    # ── 7. regenerate mask.png ────────────────────────────────────────────────
    colored = colorize_labeled_mask(labeled)
    mask_path = session_dir / "mask.png"
    cv.imwrite(str(mask_path), colored)
    logger.info("[SPLIT] Regenerated mask.png")

    t_total = time.time() - t_start
    logger.info(
        f"[SPLIT] Complete | {len(new_instances)} new instances | total={t_total:.2f}s"
    )

    return {
        "instances": new_instances,
        "mask_url": f"/images/{session_id}/mask",
    }
