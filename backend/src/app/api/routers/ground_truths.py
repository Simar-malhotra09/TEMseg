from fastapi import APIRouter, UploadFile, File
from pathlib import Path
from pydantic import BaseModel
from typing import Optional
import numpy as np
import cv2 as cv 
from app.scripts.compare_gt import normalize_mask, compute_metrics
import logging

router = APIRouter(prefix="/gt")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("routes.gt")

SESSIONS_DIR = Path("sessions")

class GTResponse(BaseModel):
    warnings: list[str] = []
    scores: Optional[dict] = None


@router.post("/{session_id}")
async def upload_gt(session_id: str, file: UploadFile = File(...)):
    logger.info(f"[GT] Upload request for session: {session_id}")
    logger.info(f"[GT] Filename received: {file.filename}")

    session_dir = SESSIONS_DIR / session_id
    logger.info(f"[GT] Session dir resolved to: {session_dir}")

    if not session_dir.exists():
        logger.warning(f"[GT] Session directory does not exist: {session_dir}")
        return {"error": "Invalid session"}

    warnings = []

    # Normalize
    logger.info("[GT] Normalizing mask...")
    gt_mask = normalize_mask(file)
    logger.info(f"[GT] Mask shape: {gt_mask.shape}, dtype: {gt_mask.dtype}")

    # Validate against original
    orig_files = list(session_dir.glob("org_*"))
    logger.info(f"[GT] Found original files: {orig_files}")

    if orig_files:
        orig = cv.imread(str(orig_files[0]), cv.IMREAD_GRAYSCALE)
        logger.info(f"[GT] Original image shape: {orig.shape}")

        if orig.shape != gt_mask.shape:
            warning_msg = f"GT dimensions {gt_mask.shape} don't match image {orig.shape}"
            warnings.append(warning_msg)
            logger.warning(f"[GT] {warning_msg}")
    else:
        logger.warning("[GT] No original image found to validate against.")

    # Save GT
    gt_path = session_dir / "gt_mask.npy"
    np.save(str(gt_path), gt_mask)
    logger.info(f"[GT] Saved ground truth to: {gt_path}")

    # Check if segmentation already done
    mask_path = session_dir / "mask.png"
    logger.info(f"[GT] Checking for existing prediction at: {mask_path}")

    scores = None
    if mask_path.exists():
        logger.info("[GT] Prediction found. Computing metrics...")
        pred = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)
        scores = compute_metrics(gt_mask, pred)
        logger.info(f"[GT] Computed scores: {scores}")
    else:
        logger.info("[GT] No prediction mask found. Skipping metric computation.")

    logger.info("[GT] Upload completed successfully.")
    return GTResponse(warnings=warnings, scores=scores)
