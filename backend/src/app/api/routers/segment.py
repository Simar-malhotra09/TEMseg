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
from app.models.impls.maskrcnn import MaskRCNN
from app.models.helpers.config import nano_config, house_config
from pydantic import BaseModel
from enum import Enum

class ModelType(str, Enum):
    yolosam = "yolosam"
    maskrcnn = "maskrcnn"


class SegmentRequest(BaseModel):
    session_id: str
    model: ModelType


router = APIRouter(prefix="/segment")
logger = logging.getLogger("routes.segment")
SESSIONS_DIR = Path("sessions")


def normalize_mask(mask) -> np.ndarray:
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    elif mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask.squeeze(-1)

    assert mask.ndim == 2, f"Unexpected mask shape: {mask.shape}"
    return (mask > 0).astype("uint8") * 255


@router.post("/")
async def segment(req: SegmentRequest):
    session_dir = SESSIONS_DIR / req.session_id

    if not session_dir.exists():
        return {"error": "Invalid session"}

    files = list(session_dir.iterdir())
    if not files:
        return {"error": "No image found"}

    image_path = files[0]

    if req.model == ModelType.yolosam:
        model_inst = YoloSam(nano_config, device="cpu")

    elif req.model == ModelType.maskrcnn:
        model_inst = MaskRCNN(house_config, device="cpu")

    else:
        return {"error": "Unsupported model"}

    img = model_inst.load_image(image_path)
    result = model_inst.segment(img)

    mask = normalize_mask(result.segmentation_mask)

    mask_path = session_dir / "mask.png"
    success = cv.imwrite(str(mask_path), mask)

    if not success:
        return {"error": "Failed to save mask"}

    return {
        "mask_url": f"/images/{req.session_id}/mask",
        "metadata": result.metadata
    }
