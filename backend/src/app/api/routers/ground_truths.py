from fastapi import APIRouter, UploadFile, File, Body
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import cv2 as cv 
from app.api.utils import blackout_regions
from app.scripts.compare_gt import normalize_mask, compute_metrics
import logging
from app.api.utils import Box,blackout_regions

router = APIRouter(prefix="/gt")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("routes.gt")

SESSIONS_DIR = Path("sessions")

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
            logger.warning(f"[GT] Could not read original image: {orig_path}")
        elif orig.shape[:2] != gt_mask.shape[:2]:  # compare H,W only, ignore channels
            warnings.append(f"GT dimensions {gt_mask.shape} don't match image {orig.shape}")

    np.save(str(session_dir / "gt_mask.npy"), gt_mask)
    cv.imwrite(f"sessions/{session_id}/gt_mask.png", 
               (gt_mask > 0).astype(np.uint8) * 255)
    return GTResponse(warnings=warnings, scores=None)


@router.post("/{session_id}/compute")
async def compute_gt(session_id: str, regions: List[Box] = Body(default=[])):
    session_dir = SESSIONS_DIR / session_id
    gt_path   = session_dir / "gt_mask.npy"
    mask_path = session_dir / "mask.png"

    if not gt_path.exists():
        return {"error": "No GT uploaded"}
    if not mask_path.exists():
        return {"error": "No segmentation mask found"}

    gt_mask = np.load(str(gt_path))
    pred    = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)

    # apply same blackout regions to GT before comparing
    gt_mask = blackout_regions(gt_mask, regions,None)

    cv.imwrite(f"sessions/{session_id}/gt_mask_blackout_check.png", 
               (gt_mask > 0).astype(np.uint8) * 255)

    scores = compute_metrics(gt_mask, pred)
    return {"scores": scores}
