import asyncio
import math
from pathlib import Path

import cv2 as cv
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel
from skimage.feature import blob_log

from app.api.instances import extract_instances, load_instances
from app.api.live_models import AvailableModels
from app.api.model_registry import get_or_load_model, model_operation
from app.api.routers.masks import FromPointsRequest, box_sweep_widths, from_points
from app.api.routers.sessions import get_group
from app.api.utils import Stroke, mask_iou, strokes_to_mask, validate_session_id
from app.logutils import Timer, get_logger
from app.models.helpers import batch_learning, rf_cache
from app.models.helpers.image_io import load_rgb_uint8
from app.models.impls.rf_recovery import RFRecovery
from app.models.impls.yolosam import YoloSam

router = APIRouter(prefix="/rf", dependencies=[Depends(model_operation)])
logger = get_logger("rf")
SESSIONS_DIR = Path("sessions")

# A user-marked bg scribble that's too small gives the RF almost no sense of
# what background looks like, so anything even slightly different from the
# handful of sampled pixels drifts to the foreground side.
# Below this floor we refuse to train rather than return junk.
MIN_BG_FRACTION = 0.04

# Flip to False to inspect the raw RF contours. When True, each proposal gets
# the convex hull of its raw output: no in and out boundary wobble, and later
# manual refinement works on a simpler polygon.
USE_CONVEX_HULL = True


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


class BatchTeachRequest(BaseModel):
    session_id: str
    strokes: list[Stroke]


class BatchInstancesRequest(BaseModel):
    session_id: str
    instance_ids: list[int]


class BatchProposeRequest(BaseModel):
    session_id: str


def _batch_session(group_id: str, session_id: str) -> Path:
    group = get_group(group_id)
    if session_id not in group["session_ids"]:
        raise HTTPException(
            status_code=400,
            detail=f"Session {session_id} is not in group {group_id}",
        )
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Session {session_id} is missing")
    return session_dir


def _require_batch(group_id: str) -> None:
    get_group(group_id)
    if not batch_learning.is_enabled(group_id):
        raise HTTPException(
            status_code=400,
            detail=f"Enable RF learning for group {group_id} first",
        )


def _original_image(session_dir: Path) -> Path:
    image_path = next(session_dir.glob("org_*"), None)
    if image_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"No original image found in session {session_dir.name}",
        )
    return image_path


@router.post("/batches/{group_id}/enable")
def enable_batch(group_id: str) -> dict[str, bool]:
    get_group(group_id)
    batch_learning.enable(group_id)
    return {"enabled": True}


@router.get("/batches/{group_id}")
def batch_status(group_id: str) -> dict[str, object]:
    group = get_group(group_id)
    active = batch_learning.examples(group_id)
    member_ids = set(group["session_ids"])
    for example in active:
        if example.session_id not in member_ids:
            raise HTTPException(
                status_code=409,
                detail=f"Example {example.id} belongs to a session outside group {group_id}",
            )
    return {
        "enabled": batch_learning.is_enabled(group_id),
        "foreground_count": sum(e.label == "foreground" for e in active),
        "background_count": sum(e.label == "background" for e in active),
        "examples": [e.model_dump(exclude={"strokes", "image_sha256"}) for e in active],
    }


def _teach_batch(
    group_id: str, body: BatchTeachRequest, label: batch_learning.PixelLabel
) -> dict[str, object]:
    _require_batch(group_id)
    session_dir = _batch_session(group_id, body.session_id)
    image_path = _original_image(session_dir)
    image = YoloSam.load_image(image_path)
    try:
        example = batch_learning.add_strokes(
            group_id, body.session_id, label, body.strokes, image, image_path
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"example_id": example.id, "status": batch_status(group_id)}


@router.post("/batches/{group_id}/foreground")
def teach_foreground(group_id: str, body: BatchTeachRequest) -> dict[str, object]:
    return _teach_batch(group_id, body, "foreground")


@router.post("/batches/{group_id}/background")
def teach_background(group_id: str, body: BatchTeachRequest) -> dict[str, object]:
    return _teach_batch(group_id, body, "background")


@router.post("/batches/{group_id}/foreground-instances")
def teach_foreground_instances(
    group_id: str, body: BatchInstancesRequest
) -> dict[str, object]:
    _require_batch(group_id)
    session_dir = _batch_session(group_id, body.session_id)
    image_path = _original_image(session_dir)
    image = YoloSam.load_image(image_path)
    try:
        example = batch_learning.add_instances(
            group_id, body.session_id, body.instance_ids, session_dir, image, image_path
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"example_id": example.id, "status": batch_status(group_id)}


@router.delete("/batches/{group_id}/examples/{example_id}")
def retract_example(group_id: str, example_id: str) -> dict[str, object]:
    _require_batch(group_id)
    try:
        batch_learning.retract(group_id, example_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return batch_status(group_id)


@router.post("/train")
async def train_rf(req: TrainRequest):
    """Manually trigger RF training on an already-segmented session."""
    session_dir = SESSIONS_DIR / validate_session_id(req.session_id)
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

    session_dir = SESSIONS_DIR / validate_session_id(req.session_id)
    if not session_dir.exists():
        return {"error": "Invalid session"}

    orig_files = list(session_dir.glob("org_*"))
    if not orig_files:
        return {"error": "No image found in session"}

    mask_path = session_dir / "mask.png"
    if not mask_path.exists():
        return {"error": "No mask — run segmentation first"}

    image = load_rgb_uint8(orig_files[0])

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

    return _proposals_from_missed(missed, session_dir, req.iou_dedupe, "rf", t)


def _particle_centers(
    image: np.ndarray, missed: np.ndarray, known_areas: list[int], limit: int
) -> list[list[float]]:
    median_area = float(np.median(known_areas)) if known_areas else 500.0
    radius = max(12.0, math.sqrt(median_area / math.pi))
    count, labels, stats, _ = cv.connectedComponentsWithStats(missed, 8)
    component_ids = sorted(
        range(1, count), key=lambda idx: int(stats[idx, 4]), reverse=True
    )
    centers: list[list[float]] = []

    for component_id in component_ids:
        x, y, width, height, _ = (int(value) for value in stats[component_id])
        pad = math.ceil(radius)
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(image.shape[1], x + width + pad)
        y1 = min(image.shape[0], y + height + pad)
        gray = (
            cv.cvtColor(image[y0:y1, x0:x1], cv.COLOR_RGB2GRAY).astype(np.float32)
            / 255.0
        )
        blobs = blob_log(
            1.0 - gray,
            min_sigma=0.35 * radius,
            max_sigma=0.9 * radius,
            num_sigma=8,
            threshold=0.03,
            overlap=0.2,
        )
        candidates = sorted(blobs, key=lambda blob: float(blob[2]), reverse=True)
        selected: list[tuple[int, int]] = []
        for local_y, local_x, _ in candidates:
            px = min(image.shape[1] - 1, x0 + round(float(local_x)))
            py = min(image.shape[0] - 1, y0 + round(float(local_y)))
            if labels[py, px] != component_id:
                continue
            if any((px - sx) ** 2 + (py - sy) ** 2 < radius**2 for sx, sy in selected):
                continue
            selected.append((px, py))
            centers.append([float(px), float(py)])
            if len(selected) >= 12 or len(centers) >= limit:
                break
        if len(centers) >= limit:
            break
    return centers


def _batch_missed(
    group_id: str, session_id: str
) -> tuple[Path, np.ndarray, np.ndarray, int]:
    """Train the batch RF on active examples and return its missed pixels."""
    _require_batch(group_id)
    session_dir = _batch_session(group_id, session_id)
    mask_path = session_dir / "mask.png"
    if not mask_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"Segment session {session_id} before batch RF recovery",
        )
    mask_gray = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)
    if mask_gray is None:
        raise HTTPException(status_code=500, detail=f"Cannot read {mask_path}")
    image = YoloSam.load_image(_original_image(session_dir))
    if image.shape[:2] != mask_gray.shape:
        raise HTTPException(
            status_code=400,
            detail=f"Image and mask dimensions differ for session {session_id}",
        )
    group = get_group(group_id)
    try:
        features, labels = batch_learning.training_data(
            group_id, set(group["session_ids"]), SESSIONS_DIR
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    min_area, _ = rf_cache.derive_min_area(session_id)
    rf = RFRecovery(min_area=min_area)
    rf.train_labeled_features(features, labels)
    missed = rf.predict_missed_mask(image, (mask_gray > 0).astype(np.uint8))
    return session_dir, image, missed, min_area


@router.get("/batches/{group_id}/debug-overlay")
def batch_debug_overlay(group_id: str, session_id: str) -> Response:
    """Paint RF missed pixels green, prompt centers yellow, SAM boxes red."""
    session_dir, image, missed, _ = _batch_missed(group_id, session_id)
    cached = load_instances(session_dir)
    known_areas: list[int] = (
        [int(instance["area"]) for instance in cached[0]] if cached is not None else []
    )
    centers = _particle_centers(image, missed, known_areas, 50)

    base = image if image.dtype == np.uint8 else np.clip(image, 0, 255).astype(np.uint8)
    canvas = cv.cvtColor(np.ascontiguousarray(base), cv.COLOR_RGB2BGR)
    green = canvas.copy()
    green[missed > 0] = (0, 255, 0)
    canvas = cv.addWeighted(green, 0.45, canvas, 0.55, 0)

    # same box sweep from_points sends to SAM, one box per width per center
    widths = box_sweep_widths([float(area) for area in known_areas])
    if widths is not None:
        height, width = canvas.shape[:2]
        for px, py in centers:
            for box_width in widths:
                half = float(box_width) / 2.0
                x0 = int(max(0.0, px - half))
                y0 = int(max(0.0, py - half))
                x1 = int(min(float(width - 1), px + half))
                y1 = int(min(float(height - 1), py + half))
                cv.rectangle(canvas, (x0, y0), (x1, y1), (0, 0, 255), 1)
    for px, py in centers:
        cv.circle(canvas, (int(px), int(py)), 3, (0, 255, 255), -1)

    ok, png = cv.imencode(".png", canvas)
    if not ok:
        raise HTTPException(status_code=500, detail="PNG encode failed")
    return Response(content=png.tobytes(), media_type="image/png")


@router.post("/batches/{group_id}/propose")
async def batch_propose(
    group_id: str, body: BatchProposeRequest, request: Request
) -> dict[str, object]:
    session_dir, image, missed, min_area = _batch_missed(group_id, body.session_id)
    embedding_cache = getattr(request.app.state, "embedding_cache", None)
    if embedding_cache is None:
        raise HTTPException(
            status_code=503, detail="SAM embedding cache is unavailable"
        )
    if body.session_id not in embedding_cache:
        sam = await asyncio.to_thread(
            get_or_load_model,
            request.app.state.models,
            request.app.state.model_components,
            AvailableModels.mobile_sam,
        )
        embedding_cache[body.session_id] = await asyncio.to_thread(
            sam._sam.encode,
            image,
            12,
        )
    cached = load_instances(session_dir)
    known_areas: list[int] = (
        [int(instance["area"]) for instance in cached[0]] if cached is not None else []
    )
    centers = _particle_centers(image, missed, known_areas, 50)
    if not centers:
        return {
            "proposals": [],
            "message": "RF found no clear unsegmented particle centers",
            "examples_used": len(batch_learning.examples(group_id)),
        }
    result = await from_points(
        body.session_id,
        FromPointsRequest(
            points=centers,
            pending=[],
            max_image_fraction=0.05,
            min_area=max(100, min_area),
        ),
        request,
    )
    proposals = [
        {**proposal, "source": "batch_rf_sam"} for proposal in result["proposals"]
    ]
    return {
        "proposals": proposals,
        "elapsed": result["elapsed"],
        "examples_used": len(batch_learning.examples(group_id)),
        "centers_tried": len(centers),
        "centers_rejected": len(result["rejected"]),
    }


def _proposals_from_missed(
    missed: np.ndarray, session_dir: Path, iou_dedupe: float, source: str, t: Timer
) -> dict[str, object]:

    if not np.any(missed):
        return {
            "proposals": [],
            "message": "RF found no missed regions",
            "elapsed": t.elapsed,
        }

    instances, _ = extract_instances(missed, session_dir, save=False)

    if USE_CONVEX_HULL:
        # hull over main + satellite points, so the outline covers the whole
        # raw output. bbox and area follow the hull, so what the client shows
        # is what a later accept commits.
        for inst in instances:
            pts = np.array(inst["contour"], dtype=np.int32).reshape(-1, 2)
            for extra in inst.get("extra_contours") or []:
                pts = np.vstack([pts, np.array(extra, dtype=np.int32).reshape(-1, 2)])
            hull = cv.convexHull(pts).reshape(-1, 2)
            x, y, w, h = cv.boundingRect(hull)
            inst["contour"] = hull.tolist()
            inst["extra_contours"] = []
            inst["bbox"] = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
            inst["area"] = int(cv.contourArea(hull))

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
            if worst_iou > iou_dedupe:
                continue
            deduped.append(inst)
        n_rejected = len(instances) - len(deduped)
        if n_rejected:
            logger.info(
                f"Removed {n_rejected} RF proposal(s) that overlapped existing particles"
            )
        instances = deduped

    for inst in instances:
        inst["source"] = source

    t.field("proposals", len(instances))
    t.stop()
    return {"proposals": instances, "elapsed": t.elapsed}


@router.delete("/cache/{session_key}")
async def evict_rf_cache(session_key: str):
    """Evict the cached RF classifier for a session."""
    rf_cache.evict(session_key)
    return {"status": "evicted", "session_key": session_key}
