from typing import Annotated
import uuid, shutil
from pathlib import Path 
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse
import logging
import torch 
import cv2 as cv
import numpy as np
from app.models.base_model import SubModelConfig, ModelConfig
from app.models.impls.yolosam import YoloSam
from app.models.helpers.config import nano_config
from pydantic import BaseModel

class SegmentRequest(BaseModel):
    session_id: str
    model: str


router = APIRouter(prefix="/segment")
logger = logging.getLogger("routes.segment")
SESSIONS_DIR = Path("sessions")

def normalize_mask(mask) -> np.ndarray:
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)       # (1,H,W) → (H,W)
    elif mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask.squeeze(-1)      # (H,W,1) → (H,W)
    assert mask.ndim == 2, f"Unexpected mask shape: {mask.shape}"
    return (mask > 0).astype("uint8") * 255

# maybe make an enum for model to make it more robust
@router.post("/")
async def segment(req: SegmentRequest):
    session_dir = SESSIONS_DIR / req.session_id

    if not session_dir.exists():
        return {"error": "Invalid session"}

    files = list(session_dir.iterdir())
    if not files:
        return {"error": "No image found"}

    # choose model based on req.model (for now just yolosam)
    if req.model != "yolosam":
        return {"error": "Unsupported model"}

    model_inst = YoloSam(nano_config, device="cpu")

    img = model_inst.load_image(files[0])
    result = model_inst.segment(img)

    # save mask
    mask_path = session_dir / "mask.png"
    mask = normalize_mask(result.segmentation_mask)
    print("mask shape:", mask.shape, "dtype:", mask.dtype, "max:", mask.max())
    success = cv.imwrite(str(mask_path), mask)
    print("imwrite success:", success)

    return {
        "mask_url": f"/images/{req.session_id}/mask",  
        "metadata": result.metadata
    }
