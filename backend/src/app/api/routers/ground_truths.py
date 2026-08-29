from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import cv2 as cv
from app.api.utils import Box, blackout_regions, inverse_blackout_regions
from app.scripts.compare_gt import normalize_mask, compute_metrics
from app.logutils import get_logger

router = APIRouter(prefix="/gt")

logger = get_logger("gt")

SESSIONS_DIR = Path("sessions")

class ComputeRequest(BaseModel):
    blackout: bool = False
    inverse_blackout: bool = False
    regions: List[Box] = []

class GTResponse(BaseModel):
    warnings: list[str] = []
    scores: Optional[dict] = None



@router.post("/{session_id}")
async def upload_gt(session_id: str, file: UploadFile = File(...)):
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        return {"error": "Invalid session"}

    warnings = []
    gt_mask = normalize_mask(file)
    logger.info(f'gt_mask dtype:{gt_mask.dtype}')
    logger.info(f"min: {gt_mask.min()}, max: {gt_mask.max()}")
    orig_files = list(session_dir.glob("org_*"))
    if orig_files:
        orig_path = orig_files[0]
        
        # handle npy originals
        if orig_path.suffix == ".npy":
            orig = np.load(str(orig_path))
        else:
            orig = cv.imread(str(orig_path), cv.IMREAD_GRAYSCALE)
        
        if orig is None:
            logger.warning(f"Could not read original image: {orig_path}")
        elif orig.shape[:2] != gt_mask.shape[:2]:  # compare H,W only, ignore channels
            warnings.append(f"GT dimensions {gt_mask.shape} don't match image {orig.shape}")

    np.save(str(session_dir / "gt_mask.npy"), gt_mask)
    cv.imwrite(f"sessions/{session_id}/gt_mask.png", 
               (gt_mask > 0).astype(np.uint8) * 255)
    return GTResponse(warnings=warnings, scores=None)


@router.post("/{session_id}/compute")
async def compute_gt(session_id: str, req: ComputeRequest):
    blackout = req.blackout
    inverse_blackout = req.inverse_blackout
    regions = req.regions

    logger.info(f"Request received for session: {session_id}")
    logger.info(f"Regions selected : {len(req.regions)}")
    mode = "blacked_out" if req.blackout else "kept"
    logger.info(f"Regions are being {mode}")

    if blackout and inverse_blackout:
        raise ValueError("Only one of blackout_regions or inverse_blackout_regions may be True.")

    session_dir = SESSIONS_DIR / session_id
    gt_path   = session_dir / "gt_mask.npy"
    mask_path = session_dir / "mask.png"

    if not gt_path.exists():
        return {"error": "No GT uploaded"}
    if not mask_path.exists():
        return {"error": "No segmentation mask found"}

    gt_mask = np.load(str(gt_path))
    pred    = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)
    pred = (pred > 127).astype(np.uint8)   # convert 0/255 to 0/1

    # apply same operations to gt before computing
    if blackout:
        gt_mask = blackout_regions(gt_mask, regions,None)

    if inverse_blackout:
        gt_mask = inverse_blackout_regions(gt_mask, regions,None)

    preview = (gt_mask * 255).astype("uint8")
    cv.imwrite(str(session_dir / "gt_preview.png"), preview)


    scores = compute_metrics(gt_mask, pred)
    return {"scores": scores}


@router.get("/{session_id}/preview")
async def gt_preview(session_id: str):
    path = SESSIONS_DIR / session_id / "gt_preview.png"
    if not path.exists():
        return {"error": "No GT preview"}
    return FileResponse(path)
