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
    # absolute minimum area in pixels — below this, the click likely landed
    # on a texture/noise patch rather than a particle
    min_area: int = 100


class ProposeSimilarRequest(BaseModel):
    # cosine-sim floor for seed candidates (0..1, higher = stricter)
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
    # — kills SAM's "select everything" failure mode for ambiguous prompts
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


# ─────────────────────────────────────────────────────────────────────────────
# /masks/{session_id}/from-points
#
# Bootstrap mode for "YOLO found nothing" — user clicks 1-N particles, each
# click is turned into a *proposal* via SAM single-point predict. The proposal
# is returned only (no disk write). The frontend overlays it; the user can
# reject individual proposals or accept the batch, at which point the existing
# PUT /masks/.../instances commits them (save + mask.png + stats in one pass).
# ─────────────────────────────────────────────────────────────────────────────


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
    t_start = time.time()
    logger.info(f"[FROM-POINTS] session={session_id} | {len(body.points)} click(s)")

    if not body.points:
        raise HTTPException(status_code=400, detail="No points provided")

    session_dir = _session_dir(session_id)

    # ── load existing instances (may be empty after a zero-YOLO segment) ──────
    cached = load_instances(session_dir)
    if cached is not None:
        instances, labeled = cached
        labeled = labeled.copy()  # avoid mutating the cached array with pending paint
    else:
        # No instances on disk yet — infer shape from mask.png if present,
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

    # ── restore SAM predictor state from cached embedding ────────────────────
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

    predictor = yolosam._predictor
    predictor.features = embedding.features
    predictor.original_size = embedding.original_size
    predictor.input_size = embedding.input_size
    predictor.is_image_set = True

    h_img, w_img = embedding.original_size
    image_area = float(h_img * w_img)
    max_area = body.max_image_fraction * image_area
    logger.info(
        f"[FROM-POINTS] Image {h_img}x{w_img} | area window "
        f"[{body.min_area}, {int(max_area)}]px"
    )

    # ── SAM predict per click ────────────────────────────────────────────────
    existing_mask = labeled > 0
    all_ids = [i["id"] for i in instances] + [int(p.get("id", 0)) for p in body.pending]
    next_id = (max(all_ids) + 1) if all_ids else 1
    proposals: list[dict] = []
    rejected: list[dict] = []

    for i, point in enumerate(body.points):
        if len(point) != 2:
            rejected.append({"index": i, "reason": "malformed point"})
            continue
        px, py = float(point[0]), float(point[1])
        if not (0 <= px < w_img and 0 <= py < h_img):
            rejected.append({"index": i, "reason": "point out of bounds"})
            continue

        try:
            masks, scores, _ = predictor.predict(
                point_coords=np.array([[px, py]]),
                point_labels=np.array([1]),
                multimask_output=True,
            )
        except Exception as e:
            logger.warning(f"[FROM-POINTS] SAM predict failed at click {i}: {e}")
            rejected.append({"index": i, "reason": f"SAM error: {e}"})
            continue

        pick = _pick_smallest_safe_mask(masks, body.min_area, max_area)
        if pick is None:
            rejected.append({"index": i, "reason": "no SAM output within area window"})
            continue
        best_idx, area, raw = pick

        # drop pixels already inside another instance (including ones just created)
        constrained = (raw > 0) & (~existing_mask)
        area = int(constrained.sum())
        if area < body.min_area:
            rejected.append(
                {"index": i, "reason": f"after constraint area={area} < min"}
            )
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

        inst = {
            "id": next_id,
            "contour": approx.tolist(),
            "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "area": area,
            "sam_score": float(scores[best_idx]),
            "source": "click",
        }
        proposals.append(inst)
        # paint into labeled + existing_mask so subsequent clicks (within this
        # same request) don't overlap. Not persisted to disk — caller commits.
        cv.fillPoly(labeled, [approx.astype(np.int32)], color=next_id)
        existing_mask = existing_mask | constrained
        next_id += 1

    if not proposals:
        raise HTTPException(
            status_code=400,
            detail=f"All {len(body.points)} clicks rejected: {rejected}",
        )

    t_total = time.time() - t_start
    logger.info(
        f"[FROM-POINTS] Proposed {len(proposals)} / {len(body.points)} clicks "
        f"| rejected={len(rejected)} | total={t_total:.2f}s"
    )

    return {
        "proposals": proposals,
        "rejected": rejected,
        "elapsed": round(t_total, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# /masks/{session_id}/propose-similar
#
# Use the user's already-annotated instances as in-context examples to find
# more visually-similar particles in the SAME image. Returns proposals only —
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
# ─────────────────────────────────────────────────────────────────────────────


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


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = int(np.logical_and(mask_a, mask_b).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(mask_a, mask_b).sum())
    return inter / union if union else 0.0


@router.post("/{session_id}/propose-similar")
async def propose_similar(
    session_id: str, body: ProposeSimilarRequest, request: Request
):
    t_start = time.time()
    logger.info(
        f"[PROPOSE] session={session_id} | thr={body.sim_threshold} | "
        f"max={body.max_proposals} | nms={body.nms_distance}px"
    )

    session_dir = _session_dir(session_id)

    # ── 1. load existing instances ───────────────────────────────────────────
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
        f"[PROPOSE] {len(instances)} examples | median area={median_area:.0f}px | "
        f"floor={min_area:.0f}px | box widths={box_widths.round(1).tolist()}"
    )

    # ── 2. cached SAM embedding ──────────────────────────────────────────────
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

    predictor = yolosam._predictor
    predictor.features = embedding.features
    predictor.original_size = embedding.original_size
    predictor.input_size = embedding.input_size
    predictor.is_image_set = True

    h_img, w_img = embedding.original_size
    logger.info(f"[PROPOSE] Image size {h_img}x{w_img} | embedding restored")

    # ── 3. prototype + similarity map ────────────────────────────────────────
    feat_map = _embedding_feature_map(predictor)  # (H_e, W_e, C)
    try:
        proto = _instance_prototype(feat_map, labeled, existing_ids)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    sim_small = feat_map @ proto  # (H_e, W_e)
    logger.info(
        f"[PROPOSE] sim range [{sim_small.min():.3f}, {sim_small.max():.3f}] | "
        f"mean={sim_small.mean():.3f}"
    )

    # upsample to image resolution
    sim = cv.resize(sim_small, (w_img, h_img), interpolation=cv.INTER_LINEAR)

    # mask out regions inside any existing instance
    existing_any = (labeled > 0).astype(np.uint8)
    # dilate a bit so seeds don't land on instance borders
    dilated = cv.dilate(
        existing_any, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    )
    sim[dilated > 0] = -1.0

    # ── 4. NMS to pick seed points ───────────────────────────────────────────
    peaks = _greedy_peak_nms(
        sim, body.sim_threshold, body.nms_distance, body.max_proposals
    )
    logger.info(f"[PROPOSE] {len(peaks)} seed peaks above threshold")

    if not peaks:
        return {
            "proposals": [],
            "median_area": median_area,
            "sim_threshold": body.sim_threshold,
            "message": "No regions above similarity threshold",
        }

    # ── 5. SAM predict per seed, dedupe ──────────────────────────────────────
    existing_mask = existing_any.astype(bool)
    proposals: list[dict] = []
    max_id = max(existing_ids)
    next_id = max_id + 1
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
                logger.warning(f"[PROPOSE] SAM predict failed at seed {i} box {k}: {e}")
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
            logger.debug(
                f"[PROPOSE] seed {i} rejected: no box variant within area window "
                f"[{min_area:.0f}, {max_area:.0f}]"
            )
            continue
        constrained = best_constrained.astype(np.uint8)
        area = best_area

        # IoU against every existing instance (uses labeled mask, no per-inst loop)
        # cheap proxy: overlap-to-area ratio with the existing union — already handled by constrain
        # full IoU dedupe: compare this proposal to each existing instance individually
        proposal_mask = constrained.astype(bool)
        # quick rejection if a single existing instance dominates
        worst_iou = 0.0
        for eid in existing_ids:
            inst_mask = labeled == eid
            iou = _mask_iou(proposal_mask, inst_mask)
            if iou > worst_iou:
                worst_iou = iou
            if iou > body.iou_dedupe:
                break
        if worst_iou > body.iou_dedupe:
            logger.debug(f"[PROPOSE] seed {i} rejected: IoU {worst_iou:.2f}")
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
        next_id += 1

    # ── 6. debug overlay PNG for visual verification ─────────────────────────
    # Renders: original image + existing instances (green) + proposals (yellow)
    # + seed points (red). Saved to session_dir for the user to inspect.
    # Prefer original_preview.png — the org_*.emd source can't be read by cv.imread.
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
                    f"[PROPOSE] cv.imread returned None for {preview_path} — "
                    f"skipping debug overlay"
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
                logger.info(f"[PROPOSE] Debug overlay saved → {debug_path}")
        except Exception as e:
            logger.warning(f"[PROPOSE] Debug overlay render failed: {e}")

    t_total = time.time() - t_start
    logger.info(
        f"[PROPOSE] Returning {len(proposals)} proposals from {len(peaks)} seeds | "
        f"total={t_total:.2f}s"
    )

    return {
        "proposals": proposals,
        "median_area": median_area,
        "sim_threshold": body.sim_threshold,
        "seed_count": len(peaks),
        "debug_overlay_path": str(session_dir / "proposals_debug.png")
        if debug_url
        else None,
        "debug_overlay_url": debug_url,
        "elapsed": round(t_total, 3),
    }
