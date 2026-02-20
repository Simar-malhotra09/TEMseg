from typing import Annotated
import uuid, shutil
from pathlib import Path 
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse
import logging
import torch 
import cv2 as cv
from app.models.base_model import SubModelConfig, ModelConfig
from app.models.impls.yolosam import YoloSam
from app.models.helpers.config import nano_config



router = APIRouter(prefix="/segment")
logger = logging.getLogger("routes.segment")
SESSIONS_DIR = Path("sessions")



# maybe make an enum for model to make it more robust
@router.post("/")
async def segment(session_id: str, model: str):
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        return {"error": "Invalid session"}

    files = list(session_dir.iterdir())
    if not files:
        return {"error": "No image found"}

    model_inst = YoloSam(nano_config, device="cpu")
    img = model_inst.load_image(files[0])
    result = model_inst.segment(img)

    # save mask
    mask_path = session_dir / "mask.png"
    cv.imwrite(str(mask_path), result.segmentation_mask * 255)

    return {
        "mask_url": f"{session_id}/mask",
        "metadata": result.metadata
    }
