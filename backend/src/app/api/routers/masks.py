from fastapi import APIRouter, HTTPException, Request
from pathlib import Path
from pydantic import BaseModel
from typing import List, Tuple
import numpy as np
import cv2 as cv
import time
from app.api.live_models import AvailableModels
from app.api.utils import mask_iou

# from app.api.utils import extract_instances, rasterize_instances, save_debug_overlay
from app.api.instances import (
    load_instances,
    extract_instances,
    save_instances,
    rasterize_instances,
    colorize_labeled_mask,
    next_free_id,
)

import json
from app.logutils import Timer, fmt_duration, get_logger, ui_event
from app.models.helpers.compute_stats import compute_stats_from_instances

router = APIRouter(prefix="/masks")
logger = get_logger("masks")
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


class FromBoxesRequest(BaseModel):
    # image-space boxes: each one is segmented by SAM with a box prompt and
    # returned as a proposal. Boxes should tightly contain a single particle.
    boxes: list[list[float]]  # [[x0, y0, x1, y1], ...]
    # not-yet-committed proposals from prior box-drags in this session; same
    # dedup-mask role as in FromPointsRequest
    pending: list[dict] = []
    # reject SAM masks whose area > this fraction of the full image
    max_image_fraction: float = 0.05
    # absolute minimum area in pixels
    min_area: int = 100


class FromPointsRequest(BaseModel):
    # image-space points where each click should become a particle
    points: list[list[float]]  # [[x, y], ...]
    # not-yet-committed proposals from prior clicks in this bootstrap session.
    # Painted into the dedup mask so a new click in/near a pending proposal
    # is rejected instead of producing an overlapping mask.
    pending: list[dict] = []
    # reject SAM masks whose area > this fraction of the full image
    # (kills "select everything" failures on background clicks)
    max_image_fraction: float = 0.05
    # absolute minimum area in pixels: below this, the click likely landed
    # on a texture/noise patch rather than a particle
    min_area: int = 100


class ProposeSimilarRequest(BaseModel):
    # seed-finding method:
    #   "cosine" — default. Cosine similarity vs a mean SAM-embedding prototype.
    #   "ncc"   — template-match an averaged image patch. Was tried as a fix
    #             for cosine's misplaced seeds but performed worse in practice
    #             across both monomorphic and polymorphic samples. Retained
    #             behind this knob for future revisit.
    method: str = "cosine"
    # peak floor for seed candidates (0..1, higher = stricter). Both methods
    # produce normalized scores so the same default works for either.
    sim_threshold: float = 0.75
    # cap on number of returned proposals (cost control)
    max_proposals: int = 50
    # NMS radius in pixels between accepted seed peaks
    nms_distance: int = 20
    # reject proposal if its area < this * median(existing areas)
    min_area_ratio: float = 0.3
    # reject proposal if IoU with any existing instance > this
    iou_dedupe: float = 0.2
    # reject proposal whose area > this fraction of the full image
    # as it kills SAM's "select everything" failure mode for ambiguous prompts
    max_image_fraction: float = 0.05


@router.get("/{session_id}/stats")
async def get_stats(session_id: str):
    """Return cached stats.json for this session. 404 if not yet computed."""
    session_dir = _session_dir(session_id)
    stats_path = session_dir / "stats.json"
    if not stats_path.exists():
        raise HTTPException(status_code=404, detail="No stats — run /segment first")
    with open(stats_path) as f:
        return json.load(f)


@router.post("/{session_id}/instances")
async def get_instances(session_id: str):
    session_dir = _session_dir(session_id)

    # fast path — already on disk
    cached = load_instances(session_dir)
    if cached is not None:
        instances, _ = cached
        return {"instances": instances}

    # recompute from mask.png
    mask_path = session_dir / "mask.png"
    if not mask_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No segmentation mask found — run segmentation first",
        )

    logger.info(f"Recomputing particles from mask for session {session_id}")
    mask_bgr = cv.imread(str(mask_path))
    if mask_bgr is None:
        raise HTTPException(status_code=500, detail="Failed to read mask.png")

    t = Timer(logger, "recompute particles")
    binary = (cv.cvtColor(mask_bgr, cv.COLOR_BGR2GRAY) > 0).astype(np.uint8)
    instances, labeled = extract_instances(binary, session_dir, save=True)
    t.field("particles", len(instances))
    t.stop()
    return {"instances": instances}


@router.put("/{session_id}/instances")
async def api_save_instances(session_id: str, req: SaveInstancesRequest):
    t = Timer(logger, "save instances")
    session_dir = _session_dir(session_id)

    cached = load_instances(session_dir)
    if cached is not None:
        _, labeled_old = cached
        shape = labeled_old.shape
    else:
        # No prior instances — happens when user is annotating from scratch
        # (zero-detection case). Prefer mask.png if /segment was run, else fall
        # back to original_preview.png which is always present post-upload.
        shape = None
        mask_path = session_dir / "mask.png"
        if mask_path.exists():
            mask_bgr = cv.imread(str(mask_path))
            if mask_bgr is not None:
                shape = mask_bgr.shape[:2]
        if shape is None:
            preview_path = session_dir / "original_preview.png"
            if preview_path.exists():
                preview = cv.imread(str(preview_path))
                if preview is not None:
                    shape = preview.shape[:2]
        if shape is None:
            raise HTTPException(
                status_code=404,
                detail="No mask or preview found to infer image shape",
            )

    labeled = rasterize_instances(req.instances, shape)
    save_instances(session_dir, req.instances, labeled)

    colored = colorize_labeled_mask(labeled)
    mask_path = session_dir / "mask.png"
    cv.imwrite(str(mask_path), colored)

    # evict stale RF so a later /rf/propose trains fresh on the edited mask
    from app.models.helpers import rf_cache

    rf_cache.evict(session_id)

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

    t.field("particles", len(req.instances))
    t.field("total", stats["particle_count"])
    t.stop()

    ui_event(
        "MASKS_SAVED",
        "Edited masks saved.",
        level="info",
    )
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
    t = Timer(logger, "split")
    logger.info(
        f"Splitting particle {body.instance_id} into {len(body.points)} parts for session {session_id}"
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

    logger.info(f"Original particle area {target['area']}px, bbox {target['bbox']}")

    # extract original blob mask — used to constrain SAM output
    blob_mask = (labeled == body.instance_id).astype(np.uint8)

    # ── 2. get SAM predictor from cache ──────────────────────────────────────
    embedding_cache = getattr(request.app.state, "embedding_cache", {})
    embedding = embedding_cache.get(session_id)

    if embedding is None:
        raise HTTPException(
            status_code=400,
            detail="SAM embedding not cached for this session — re-run segmentation to restore it",
        )

    # get yolosam model from app state
    models = request.app.state.models
    yolosam = models.get(AvailableModels.yolosam.value)
    if yolosam is None:
        raise HTTPException(status_code=500, detail="YoloSAM model not loaded")

    # restore predictor state from cached embedding
    yolosam.sync_prompt_embedding(embedding)
    predictor = yolosam._predictor
    logger.info("Restored SAM model state from cache")

    # ── 3. run SAM for each point ─────────────────────────────────────────────
    new_masks = []
    for i, (px, py) in enumerate(body.points):
        t_pred = time.perf_counter()

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

        pred_dt = time.perf_counter() - t_pred

        # ── 4. constrain to original blob ────────────────────────────────────
        constrained = cv.bitwise_and(raw_mask, blob_mask)
        pixel_count = int(np.sum(constrained))

        if pixel_count < 10:
            logger.warning(
                f"Split point {i + 1} produced too small a mask ({pixel_count}px) — skipping"
            )
            continue

        logger.debug(
            f"split point {i + 1}/{len(body.points)} predicted={fmt_duration(pred_dt)} "
            f"({pixel_count}px after constraint)"
        )
        new_masks.append(constrained)

    if not new_masks:
        raise HTTPException(
            status_code=400,
            detail="All split points produced empty masks — try placing points more centrally",
        )

    logger.info(
        f"Got {len(new_masks)} valid masks from {len(body.points)} split points"
    )

    # ── 5. extract contours from each mask ───────────────────────────────────
    # the instance being split is going away, so its id is free to reuse
    used_ids = {i["id"] for i in instances if i["id"] != body.instance_id}
    new_instances = []

    for i, mask in enumerate(new_masks):
        new_id = next_free_id(used_ids)
        used_ids.add(new_id)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.warning(f"No contour found for split mask {i + 1}, skipping")
            continue

        largest = max(contours, key=cv.contourArea)
        epsilon = 0.01 * cv.arcLength(largest, True)
        contour = cv.approxPolyDP(largest, epsilon, True).squeeze()

        if contour.ndim < 2 or len(contour) < 3:
            logger.warning(f"Split result {i + 1} too small, skipping")
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
            f"Created new particle {new_id} with area {area}px from split point {i + 1}"
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
        f"Updated particle list from {len(instances)} to {len(updated_instances)}"
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

    t.field("created", len(new_instances))
    t.stop()

    return {
        "instances": new_instances,
        "mask_url": f"/images/{session_id}/mask",
    }


# /masks/{session_id}/from-boxes
#
# Box-prompt SAM. User drags a tight rectangle around a particle; the box
# becomes a SAM prompt (multimask_output=False, since the box disambiguates
# scale on its own). Returns proposals only ie the same pendingProposals flow as
# /from-points and /propose-similar. SAM is dramatically more reliable with
# box prompts than point prompts on small / clumped / OOD particles, so this
# is the preferred bootstrap path when SAM-with-point fails.


@router.post("/{session_id}/from-boxes")
async def from_boxes(session_id: str, body: FromBoxesRequest, request: Request):
    t = Timer(logger, "from-boxes")
    logger.info(
        f"Box-prompt segmentation for session {session_id}: {len(body.boxes)} box(es)"
    )

    if not body.boxes:
        raise HTTPException(status_code=400, detail="No boxes provided")

    session_dir = _session_dir(session_id)

    cached = load_instances(session_dir)
    if cached is not None:
        instances, labeled = cached
        labeled = labeled.copy()
    else:
        # Same fallback chain as /from-points — required because the user may
        # bootstrap with boxes before /segment has ever run.
        shape = None
        mask_path = session_dir / "mask.png"
        if mask_path.exists():
            m = cv.imread(str(mask_path))
            if m is not None:
                shape = m.shape[:2]
        if shape is None:
            preview = session_dir / "original_preview.png"
            if preview.exists():
                p = cv.imread(str(preview))
                if p is not None:
                    shape = p.shape[:2]
        if shape is None:
            raise HTTPException(
                status_code=400,
                detail="Cannot determine image shape — run /segment first",
            )
        instances = []
        labeled = np.zeros(shape, dtype=np.uint16)

    for p in body.pending:
        contour = np.array(p.get("contour", []), dtype=np.int32)
        pid = int(p.get("id", 0))
        if contour.ndim == 2 and len(contour) >= 3 and pid > 0:
            cv.fillPoly(labeled, [contour], color=pid)

    embedding_cache = getattr(request.app.state, "embedding_cache", {})
    embedding = embedding_cache.get(session_id)
    if embedding is None:
        raise HTTPException(
            status_code=400,
            detail="SAM embedding not cached — re-run /segment first to encode the image",
        )

    yolosam = request.app.state.models.get(AvailableModels.yolosam.value)
    if yolosam is None:
        raise HTTPException(status_code=500, detail="YoloSAM model not loaded")

    yolosam.sync_prompt_embedding(embedding)
    predictor = yolosam._predictor

    h_img, w_img = embedding.original_size
    image_area = float(h_img * w_img)
    max_area = body.max_image_fraction * image_area

    existing_mask = labeled > 0
    used_ids = {i["id"] for i in instances} | {
        int(p.get("id", 0)) for p in body.pending
    }
    proposals: list[dict] = []
    rejected: list[dict] = []

    for i, box in enumerate(body.boxes):
        if len(box) != 4:
            rejected.append({"index": i, "reason": "malformed box"})
            continue
        x0, y0, x1, y1 = (float(c) for c in box)
        # normalize order in case the user dragged right-to-left or up-to-down
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        # clamp to image bounds
        x0 = max(0.0, x0)
        y0 = max(0.0, y0)
        x1 = min(float(w_img - 1), x1)
        y1 = min(float(h_img - 1), y1)
        if (x1 - x0) < 4 or (y1 - y0) < 4:
            rejected.append({"index": i, "reason": "box too small"})
            continue

        # SAM's strongest prompt is box+positive-point. The box constrains
        # scale, the point disambiguates "which particle" when the box
        # incidentally clips a neighbor (typical in tight clumps).
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        try:
            masks, scores, _ = predictor.predict(
                point_coords=np.array([[cx, cy]]),
                point_labels=np.array([1]),
                box=np.array([x0, y0, x1, y1]),
                multimask_output=False,
            )
        except Exception as e:
            logger.warning(f"SAM prediction failed for box {i}: {e}")
            rejected.append({"index": i, "reason": f"SAM error: {e}"})
            continue

        constrained = (masks[0].astype(np.uint8) > 0) & (~existing_mask)
        area = int(constrained.sum())
        if area < body.min_area:
            rejected.append({"index": i, "reason": f"area={area} < min"})
            continue
        if area > max_area:
            rejected.append({"index": i, "reason": f"area={area} > max"})
            continue

        constrained_u8 = constrained.astype(np.uint8)
        contours, _ = cv.findContours(
            constrained_u8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            rejected.append({"index": i, "reason": "no contour"})
            continue
        largest = max(contours, key=cv.contourArea)
        epsilon = 0.01 * cv.arcLength(largest, True)
        approx = cv.approxPolyDP(largest, epsilon, True).squeeze()
        if approx.ndim < 2 or len(approx) < 3:
            rejected.append({"index": i, "reason": "degenerate contour"})
            continue
        x, y, w, h = cv.boundingRect(largest)

        next_id = next_free_id(used_ids)
        inst = {
            "id": next_id,
            "contour": approx.tolist(),
            "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "area": area,
            "sam_score": float(scores[0]),
            "source": "box",
        }
        proposals.append(inst)
        cv.fillPoly(labeled, [approx.astype(np.int32)], color=next_id)
        existing_mask = existing_mask | constrained
        used_ids.add(next_id)

    if not proposals:
        raise HTTPException(
            status_code=400,
            detail=f"All {len(body.boxes)} boxes rejected: {rejected}",
        )

    t.field("proposals", f"{len(proposals)}/{len(body.boxes)}")
    t.field("rejected", len(rejected))
    t.stop()

    return {
        "proposals": proposals,
        "rejected": rejected,
        "elapsed": round(t.elapsed, 3),
    }


# /masks/{session_id}/from-points
#
# Bootstrap mode for "YOLO found nothing". User clicks 1-N particles, each
# click is turned into a *proposal* via SAM single-point predict. The proposal
# is returned only (no disk write). The frontend overlays it; the user can
# reject individual proposals or accept the batch, at which point the existing
# PUT /masks/.../instances commits them (save + mask.png + stats in one pass).


def _pick_smallest_safe_mask(
    masks: np.ndarray,
    min_area: int,
    max_area: float,
) -> tuple[int, int, np.ndarray] | None:
    """From SAM's 3 multimask outputs, return the smallest mask within
    [min_area, max_area]. Returns (idx, area, mask) or None."""
    candidates = []
    for k in range(masks.shape[0]):
        m = masks[k].astype(np.uint8)
        a = int(m.sum())
        if min_area <= a <= max_area:
            candidates.append((a, k, m))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    a, k, m = candidates[0]
    return k, a, m


@router.post("/{session_id}/from-points")
async def from_points(session_id: str, body: FromPointsRequest, request: Request):
    t = Timer(logger, "from-points")
    logger.info(
        f"Point-prompt segmentation for session {session_id}: {len(body.points)} click(s)"
    )

    if not body.points:
        raise HTTPException(status_code=400, detail="No points provided")

    session_dir = _session_dir(session_id)

    # load existing instances (may be empty after a zero-YOLO segment)
    cached = load_instances(session_dir)
    if cached is not None:
        instances, labeled = cached
        labeled = labeled.copy()  # avoid mutating the cached array with pending paint
    else:
        # No instances on disk yet. infer shape from mask.png if present,
        # otherwise from the original_preview.png.
        shape = None
        mask_path = session_dir / "mask.png"
        if mask_path.exists():
            m = cv.imread(str(mask_path))
            if m is not None:
                shape = m.shape[:2]
        if shape is None:
            preview = session_dir / "original_preview.png"
            if preview.exists():
                p = cv.imread(str(preview))
                if p is not None:
                    shape = p.shape[:2]
        if shape is None:
            raise HTTPException(
                status_code=400,
                detail="Cannot determine image shape — run /segment first",
            )
        instances = []
        labeled = np.zeros(shape, dtype=np.uint16)

    # paint pending (uncommitted) proposals into the dedup mask so a new click
    # that lands on/near one is rejected rather than producing an overlap.
    for p in body.pending:
        contour = np.array(p.get("contour", []), dtype=np.int32)
        pid = int(p.get("id", 0))
        if contour.ndim == 2 and len(contour) >= 3 and pid > 0:
            cv.fillPoly(labeled, [contour], color=pid)

    # restore SAM predictor state from cached embedding
    embedding_cache = getattr(request.app.state, "embedding_cache", {})
    embedding = embedding_cache.get(session_id)
    if embedding is None:
        raise HTTPException(
            status_code=400,
            detail="SAM embedding not cached — re-run /segment first to encode the image",
        )

    yolosam = request.app.state.models.get(AvailableModels.yolosam.value)
    if yolosam is None:
        raise HTTPException(status_code=500, detail="YoloSAM model not loaded")

    yolosam.sync_prompt_embedding(embedding)
    predictor = yolosam._predictor

    h_img, w_img = embedding.original_size
    image_area = float(h_img * w_img)
    max_area = body.max_image_fraction * image_area
    logger.info(
        f"Image size {h_img}x{w_img}, accepted particle area "
        f"{body.min_area}-{int(max_area)}px"
    )

    # SAM predict per click
    existing_mask = labeled > 0
    used_ids = {i["id"] for i in instances} | {
        int(p.get("id", 0)) for p in body.pending
    }
    proposals: list[dict] = []
    rejected: list[dict] = []

    # Build a size prior from disk + pending if any. With ≥1 example, swap each
    # click's SAM call from multimask=True to a box-sweep whose widths come from
    # sqrt(area) percentiles of the prior, mirroring /propose-similar's per-seed
    # logic. This makes click-produced masks track the user's annotated scale
    # rather than depending on SAM's arbitrary multimask scale lottery.
    prior_areas = [float(i["area"]) for i in instances] + [
        float(p["area"]) for p in body.pending if p.get("area")
    ]
    has_prior = len(prior_areas) > 0
    if has_prior:
        sqrt_areas = np.sqrt(np.array(prior_areas, dtype=float))
        if len(sqrt_areas) >= 4:
            box_base = np.percentile(sqrt_areas, [25, 50, 75])
        else:
            med = float(np.median(sqrt_areas))
            box_base = np.array([0.7 * med, 1.0 * med, 1.3 * med])
        box_widths = box_base * 1.2
        log_median = float(np.log(max(float(np.median(prior_areas)), 1.0)))
        logger.info(
            f"Using size prior from {len(prior_areas)} existing particles, "
            f"box widths={box_widths.round(1).tolist()}"
        )
    else:
        box_widths = None
        log_median = None
        logger.info("No existing size examples, using SAM scale picker")

    for i, point in enumerate(body.points):
        if len(point) != 2:
            rejected.append({"index": i, "reason": "malformed point"})
            continue
        px, py = float(point[0]), float(point[1])
        if not (0 <= px < w_img and 0 <= py < h_img):
            rejected.append({"index": i, "reason": "point out of bounds"})
            continue

        constrained: np.ndarray | None = None
        area = 0
        chosen_sam_score = 0.0

        if has_prior:
            # Box-sweep around the click, widths from sqrt(area) percentiles.
            # multimask=False since box already disambiguates scale.
            best_log_dist = float("inf")
            for k, bw in enumerate(box_widths):
                half = float(bw) / 2.0
                x0 = max(0.0, px - half)
                y0 = max(0.0, py - half)
                x1 = min(float(w_img - 1), px + half)
                y1 = min(float(h_img - 1), py + half)
                try:
                    masks, scores, _ = predictor.predict(
                        point_coords=np.array([[px, py]]),
                        point_labels=np.array([1]),
                        box=np.array([x0, y0, x1, y1]),
                        multimask_output=False,
                    )
                except Exception as e:
                    logger.warning(
                        f"SAM prediction failed for click {i} at box size {k}: {e}"
                    )
                    continue
                m = (masks[0].astype(np.uint8) > 0) & (~existing_mask)
                a = int(m.sum())
                if a < body.min_area or a > max_area:
                    continue
                log_dist = abs(np.log(max(a, 1.0)) - log_median)
                if log_dist < best_log_dist:
                    best_log_dist = log_dist
                    constrained = m
                    area = a
                    chosen_sam_score = float(scores[0])
            if constrained is None:
                rejected.append(
                    {"index": i, "reason": "no box variant within area window"}
                )
                continue
        else:
            # No prior exists. fall back to the original multimask + smallest-in-window pick.
            try:
                masks, scores, _ = predictor.predict(
                    point_coords=np.array([[px, py]]),
                    point_labels=np.array([1]),
                    multimask_output=True,
                )
            except Exception as e:
                logger.warning(f"SAM prediction failed for click {i}: {e}")
                rejected.append({"index": i, "reason": f"SAM error: {e}"})
                continue
            pick = _pick_smallest_safe_mask(masks, body.min_area, max_area)
            if pick is None:
                rejected.append(
                    {"index": i, "reason": "no SAM output within area window"}
                )
                continue
            best_idx, _, raw = pick
            constrained = (raw > 0) & (~existing_mask)
            area = int(constrained.sum())
            if area < body.min_area:
                rejected.append(
                    {"index": i, "reason": f"after constraint area={area} < min"}
                )
                continue
            chosen_sam_score = float(scores[best_idx])

        constrained_u8 = constrained.astype(np.uint8)
        contours, _ = cv.findContours(
            constrained_u8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            rejected.append({"index": i, "reason": "no contour"})
            continue
        largest = max(contours, key=cv.contourArea)
        epsilon = 0.01 * cv.arcLength(largest, True)
        approx = cv.approxPolyDP(largest, epsilon, True).squeeze()
        if approx.ndim < 2 or len(approx) < 3:
            rejected.append({"index": i, "reason": "degenerate contour"})
            continue
        x, y, w, h = cv.boundingRect(largest)

        next_id = next_free_id(used_ids)
        inst = {
            "id": next_id,
            "contour": approx.tolist(),
            "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "area": area,
            "sam_score": chosen_sam_score,
            "source": "click",
        }
        proposals.append(inst)
        # paint into labeled + existing_mask so subsequent clicks (within this
        # same request) don't overlap. Not persisted to disk — caller commits.
        cv.fillPoly(labeled, [approx.astype(np.int32)], color=next_id)
        existing_mask = existing_mask | constrained
        used_ids.add(next_id)

    if not proposals:
        raise HTTPException(
            status_code=400,
            detail=f"All {len(body.points)} clicks rejected: {rejected}",
        )

    t.field("proposals", f"{len(proposals)}/{len(body.points)}")
    t.field("rejected", len(rejected))
    t.stop()

    return {
        "proposals": proposals,
        "rejected": rejected,
        "elapsed": round(t.elapsed, 3),
    }


# /masks/{session_id}/propose-similar
#
# Use the user's already-annotated instances as in-context examples to find
# more visually-similar particles in the SAME image. Returns proposals only
# nothing is written to disk. The frontend overlays them; the user accepts /
# rejects, then commits the kept ones via the existing PUT /masks/.../instances.
#
# Algorithm:
#   1. Reuse cached SAM embedding (predictor.features, shape (1, 256, 64, 64)).
#   2. Build a prototype: average L2-normalized embedding vectors taken inside
#      each existing instance mask (downsampled to embedding grid).
#   3. Cosine-similarity map between prototype and every embedding pixel.
#   4. Upsample to image res; mask out regions inside existing instances.
#   5. Pick local maxima above sim_threshold via greedy NMS — these are seeds.
#   6. For each seed, run SAM predict() with a single foreground point.
#   7. Constrain to "not already annotated" regions; reject by area / IoU.
#   8. Return surviving candidates as instance dicts (with provisional IDs).


def _embedding_feature_map(predictor) -> np.ndarray:
    """Pull SAM image features into a (H_e, W_e, C) L2-normalized numpy array."""
    feats = predictor.features  # (1, C, H_e, W_e) torch tensor
    feats_np = feats.squeeze(0).cpu().numpy()  # (C, H_e, W_e)
    feats_np = np.transpose(feats_np, (1, 2, 0))  # (H_e, W_e, C)
    norms = np.linalg.norm(feats_np, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return feats_np / norms


def _instance_prototype(
    feat_map: np.ndarray, labeled: np.ndarray, instance_ids: list[int]
) -> np.ndarray:
    """Mean L2-normalized feature vector across pixels inside the given instances."""
    h_e, w_e, _ = feat_map.shape
    labeled_small = cv.resize(
        labeled.astype(np.int32), (w_e, h_e), interpolation=cv.INTER_NEAREST
    )
    inside = np.isin(labeled_small, instance_ids)
    if not inside.any():
        raise ValueError(
            "No example pixels mapped onto embedding grid — instances too small"
        )
    vecs = feat_map[inside]
    proto = vecs.mean(axis=0)
    n = np.linalg.norm(proto)
    if n < 1e-8:
        raise ValueError("Prototype norm is zero — embedding likely degenerate")
    return proto / n


def _greedy_peak_nms(
    score_map: np.ndarray, threshold: float, min_distance: int, max_peaks: int
) -> list[tuple[int, int]]:
    """Return up to max_peaks (y, x) coordinates of local maxima above threshold."""
    candidates = np.argwhere(score_map >= threshold)
    if len(candidates) == 0:
        return []
    scores = score_map[candidates[:, 0], candidates[:, 1]]
    order = np.argsort(-scores)
    picked: list[tuple[int, int]] = []
    d2 = min_distance * min_distance
    for idx in order:
        y, x = int(candidates[idx, 0]), int(candidates[idx, 1])
        if any((y - py) ** 2 + (x - px) ** 2 < d2 for py, px in picked):
            continue
        picked.append((y, x))
        if len(picked) >= max_peaks:
            break
    return picked


def _build_avg_particle_template(
    image_gray: np.ndarray,
    instances: list[dict],
    labeled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build an averaged particle patch + alignment mask from existing instances.

    For each instance we take a square crop centered on its bbox, resize to a
    common side length derived from median(sqrt(area)), and average. The mask
    is the average of per-instance binary masks (instance pixels = 1, else 0),
    used as the matchTemplate mask so off-particle pixels don't contribute to
    the NCC score.

    Returns (template_uint8, mask_float32, side_px).
    """
    areas = [float(i["area"]) for i in instances]
    # Target side: rough diameter from median area, padded ~40% for context.
    side = int(round(np.sqrt(max(np.median(areas), 1.0)) * 1.4))
    side = max(side, 16)
    if side % 2 == 0:
        side += 1  # odd so the center maps cleanly

    patches: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    h_img, w_img = image_gray.shape[:2]
    for inst in instances:
        bbox = inst["bbox"]
        w, h = int(bbox["w"]), int(bbox["h"])
        if w < 4 or h < 4:
            continue
        cx = int(bbox["x"]) + w // 2
        cy = int(bbox["y"]) + h // 2
        half = max(w, h) // 2 + 2  # square crop, slight pad
        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        x1 = min(w_img, cx + half)
        y1 = min(h_img, cy + half)
        patch = image_gray[y0:y1, x0:x1]
        inst_mask = (labeled[y0:y1, x0:x1] == int(inst["id"])).astype(np.float32)
        if patch.shape[0] < 4 or patch.shape[1] < 4 or not inst_mask.any():
            continue
        patch_r = cv.resize(patch, (side, side), interpolation=cv.INTER_AREA)
        mask_r = cv.resize(inst_mask, (side, side), interpolation=cv.INTER_LINEAR)
        patches.append(patch_r.astype(np.float32))
        masks.append(mask_r)

    if not patches:
        raise ValueError(
            "No usable patches for template — instances too small or empty"
        )

    template = np.stack(patches).mean(axis=0).astype(np.uint8)
    template_mask = np.stack(masks).mean(axis=0).astype(np.float32)
    # Binarize the mask a bit so partial-coverage pixels don't dilute matches.
    template_mask = (template_mask > 0.4).astype(np.float32)
    return template, template_mask, side


def _ncc_score_map(
    image_gray: np.ndarray,
    template: np.ndarray,
    template_mask: np.ndarray,
) -> np.ndarray:
    """Run cv.matchTemplate with mask and pad to image-space coordinates.

    cv.matchTemplate output is (H - h_t + 1, W - w_t + 1) with each pixel
    indexing the *top-left* of the matched window. We center the score at
    the window center by padding with -1 (below any threshold) on all sides.
    """
    score = cv.matchTemplate(
        image_gray, template, cv.TM_CCORR_NORMED, mask=template_mask
    )
    # NaNs can appear where mask covers all-zero regions; clamp them.
    score = np.where(np.isfinite(score), score, -1.0)
    h_t, w_t = template.shape
    top = h_t // 2
    bottom = h_t - 1 - top
    left = w_t // 2
    right = w_t - 1 - left
    return cv.copyMakeBorder(
        score, top, bottom, left, right, cv.BORDER_CONSTANT, value=-1.0
    )


@router.post("/{session_id}/propose-similar")
async def propose_similar(
    session_id: str, body: ProposeSimilarRequest, request: Request
):
    t = Timer(logger, "propose-similar")
    logger.info(
        f"Finding similar particles for session {session_id}: "
        f"threshold={body.sim_threshold}, max={body.max_proposals}, "
        f"min spacing={body.nms_distance}px"
    )

    session_dir = _session_dir(session_id)

    #  1. load existing instances
    cached = load_instances(session_dir)
    if cached is None:
        raise HTTPException(
            status_code=400,
            detail="No existing instances — annotate at least one particle first",
        )
    instances, labeled = cached
    if not instances:
        raise HTTPException(
            status_code=400,
            detail="Existing instances list is empty — annotate at least one particle first",
        )

    existing_ids = [i["id"] for i in instances]
    existing_areas = [i["area"] for i in instances]
    median_area = float(np.median(existing_areas))
    min_area = max(50.0, body.min_area_ratio * median_area)

    # Box-prompt sweep widths derived from existing particle size distribution.
    # The seed point fixes location; varying box width lets SAM produce a mask
    # at multiple scales without using multimask_output (which was picking
    # arbitrary scales unrelated to our prior).
    sqrt_areas = np.sqrt(np.array(existing_areas, dtype=float))
    if len(sqrt_areas) >= 4:
        box_base = np.percentile(sqrt_areas, [25, 50, 75])
    else:
        # not enough samples for stable percentiles — use multiplicative widths
        med = float(np.median(sqrt_areas))
        box_base = np.array([0.7 * med, 1.0 * med, 1.3 * med])
    # mild padding so SAM has context at the box edge (tight boxes clip)
    box_widths = box_base * 1.2
    logger.info(
        f"Using {len(instances)} example particles, median area {median_area:.0f}px, "
        f"box widths={box_widths.round(1).tolist()}"
    )

    # 2. cached SAM embedding
    embedding_cache = getattr(request.app.state, "embedding_cache", {})
    embedding = embedding_cache.get(session_id)
    if embedding is None:
        raise HTTPException(
            status_code=400,
            detail="SAM embedding not cached — re-run segmentation to restore it",
        )

    models = request.app.state.models
    yolosam = models.get(AvailableModels.yolosam.value)
    if yolosam is None:
        raise HTTPException(status_code=500, detail="YoloSAM model not loaded")

    yolosam.sync_prompt_embedding(embedding)
    predictor = yolosam._predictor

    h_img, w_img = embedding.original_size
    logger.info(f"Image size {h_img}x{w_img}, restored SAM embedding")

    #  3. similarity map
    method = body.method.lower()
    if method not in ("ncc", "cosine"):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown method '{body.method}' (expected 'ncc' or 'cosine')",
        )

    if method == "cosine":
        feat_map = _embedding_feature_map(predictor)  # (H_e, W_e, C)
        try:
            proto = _instance_prototype(feat_map, labeled, existing_ids)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        sim_small = feat_map @ proto  # (H_e, W_e)
        logger.info(
            f"Cosine similarity range [{sim_small.min():.3f}, {sim_small.max():.3f}] "
            f"(mean {sim_small.mean():.3f})"
        )
        sim = cv.resize(sim_small, (w_img, h_img), interpolation=cv.INTER_LINEAR)
    else:
        # NCC: template-match an averaged particle patch against the actual
        # image. Avoids SAM's pretrained ViT embedding, which isn't TEM-aware.
        preview_path = session_dir / "original_preview.png"
        if not preview_path.exists():
            raise HTTPException(
                status_code=500,
                detail="original_preview.png missing — cannot run NCC",
            )
        image_bgr = cv.imread(str(preview_path))
        if image_bgr is None:
            raise HTTPException(
                status_code=500, detail="Failed to read original_preview.png"
            )
        if image_bgr.shape[:2] != (h_img, w_img):
            image_bgr = cv.resize(image_bgr, (w_img, h_img))
        image_gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
        try:
            template, template_mask, side_px = _build_avg_particle_template(
                image_gray, instances, labeled
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        logger.info(
            f"NCC template side={side_px}px built from {len(instances)} examples"
        )
        sim = _ncc_score_map(image_gray, template, template_mask)
        logger.info(
            f"NCC score range [{sim.min():.3f}, {sim.max():.3f}] "
            f"(mean {sim.mean():.3f})"
        )

    # mask out regions inside any existing instance (both methods)
    existing_any = (labeled > 0).astype(np.uint8)
    dilated = cv.dilate(
        existing_any, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    )
    sim[dilated > 0] = -1.0

    # 4. NMS to pick seed points
    peaks = _greedy_peak_nms(
        sim, body.sim_threshold, body.nms_distance, body.max_proposals
    )
    logger.info(f"Found {len(peaks)} candidate locations above similarity threshold")

    if not peaks:
        return {
            "proposals": [],
            "median_area": median_area,
            "sim_threshold": body.sim_threshold,
            "message": "No regions above similarity threshold",
        }

    #  5. SAM predict per seed, dedupe
    existing_mask = existing_any.astype(bool)
    proposals: list[dict] = []
    used_ids = set(existing_ids)
    image_area = float(h_img * w_img)
    max_area = body.max_image_fraction * image_area
    log_median = np.log(max(median_area, 1.0))

    for i, (py, px) in enumerate(peaks):
        point_coords = np.array([[float(px), float(py)]])
        point_labels = np.array([1])

        # Sweep box widths around the seed; SAM with point+box, one mask each.
        # The box constrains scale, so multimask_output=False is sufficient.
        best_box_idx = -1
        best_area = 0
        best_constrained = None
        best_log_dist = float("inf")
        best_sam_score = 0.0
        for k, bw in enumerate(box_widths):
            half = float(bw) / 2.0
            x0 = max(0.0, px - half)
            y0 = max(0.0, py - half)
            x1 = min(float(w_img - 1), px + half)
            y1 = min(float(h_img - 1), py + half)
            box = np.array([x0, y0, x1, y1])
            try:
                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box,
                    multimask_output=False,
                )
            except Exception as e:
                logger.warning(
                    f"SAM prediction failed for candidate {i} at box size {k}: {e}"
                )
                continue
            m = (masks[0].astype(np.uint8) > 0) & (~existing_mask)
            a = int(m.sum())
            if a < min_area or a > max_area:
                continue
            log_dist = abs(np.log(max(a, 1.0)) - log_median)
            if log_dist < best_log_dist:
                best_log_dist = log_dist
                best_box_idx = k
                best_area = a
                best_constrained = m
                best_sam_score = float(scores[0])
        if best_box_idx < 0 or best_constrained is None:
            continue
        constrained = best_constrained.astype(np.uint8)
        area = best_area

        # IoU against every existing instance (uses labeled mask, no per-inst loop)
        # cheap proxy: overlap-to-area ratio with the existing union, already handled by constrain
        # full IoU dedupe: compare this proposal to each existing instance individually
        proposal_mask = constrained.astype(bool)
        # quick rejection if a single existing instance dominates
        worst_iou = 0.0
        for eid in existing_ids:
            inst_mask = labeled == eid
            iou = mask_iou(proposal_mask, inst_mask)
            if iou > worst_iou:
                worst_iou = iou
            if iou > body.iou_dedupe:
                break
        if worst_iou > body.iou_dedupe:
            continue

        # contour
        contours, _ = cv.findContours(
            constrained, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        largest = max(contours, key=cv.contourArea)
        epsilon = 0.01 * cv.arcLength(largest, True)
        approx = cv.approxPolyDP(largest, epsilon, True).squeeze()
        if approx.ndim < 2 or len(approx) < 3:
            continue
        x, y, w, h = cv.boundingRect(largest)

        # similarity score that produced this seed (for ranking in UI)
        seed_sim = float(sim[py, px]) if sim[py, px] > 0 else 0.0

        next_id = next_free_id(used_ids)
        proposals.append(
            {
                "id": next_id,
                "contour": approx.tolist(),
                "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                "area": area,
                "sam_score": best_sam_score,
                "similarity": seed_sim,
                "seed": [int(px), int(py)],
                "box_width": float(box_widths[best_box_idx]),
            }
        )
        # block this region from being re-proposed by later seeds
        existing_mask = np.logical_or(existing_mask, proposal_mask)
        used_ids.add(next_id)

    # 6. debug overlay PNG
    # Renders: original image + existing instances (green) + proposals (yellow)
    # + seed points (red). Saved to session_dir for the user to inspect.
    # Prefer original_preview.png as the org_*.emd source can't be read by cv.imread.
    debug_url = None
    preview_path = session_dir / "original_preview.png"
    if not preview_path.exists():
        orig_files = list(session_dir.glob("org_*"))
        preview_path = orig_files[0] if orig_files else None

    if preview_path and proposals:
        try:
            orig = cv.imread(str(preview_path))
            if orig is None:
                logger.warning(
                    f"Could not read preview image {preview_path.name}, skipping debug overlay"
                )
            else:
                if orig.shape[:2] != (h_img, w_img):
                    orig = cv.resize(orig, (w_img, h_img))
                overlay = orig.copy()
                # existing instances in green outline
                for eid in existing_ids:
                    inst_contour_mask = (labeled == eid).astype(np.uint8)
                    c, _ = cv.findContours(
                        inst_contour_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
                    )
                    cv.drawContours(overlay, c, -1, (0, 200, 0), 2)
                # proposals in yellow outline + ID label
                for p in proposals:
                    pts = np.array(p["contour"], dtype=np.int32).reshape(-1, 1, 2)
                    cv.polylines(overlay, [pts], True, (0, 255, 255), 3)
                    sx, sy = p["seed"]
                    cv.circle(overlay, (sx, sy), 6, (0, 0, 255), -1)
                    cv.putText(
                        overlay,
                        f"#{p['id']} sim={p['similarity']:.2f}",
                        (p["bbox"]["x"], max(p["bbox"]["y"] - 6, 12)),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )
                debug_path = session_dir / "proposals_debug.png"
                cv.imwrite(str(debug_path), overlay)
                debug_url = f"/images/{session_id}/proposals-debug"
                logger.info(f"Saved proposal debug overlay to {debug_path.name}")
        except Exception as e:
            logger.warning(f"Could not save proposal debug overlay: {e}")

    t.field("proposals", len(proposals))
    t.field("candidates", len(peaks))
    t.stop()

    return {
        "proposals": proposals,
        "median_area": median_area,
        "sim_threshold": body.sim_threshold,
        "method": method,
        "seed_count": len(peaks),
        "debug_overlay_path": str(session_dir / "proposals_debug.png")
        if debug_url
        else None,
        "debug_overlay_url": debug_url,
        "elapsed": round(t.elapsed, 3),
    }
