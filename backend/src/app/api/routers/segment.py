from typing import Annotated, List
import uuid, shutil
from pathlib import Path 
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse
import logging
import torch 
import cv2 as cv
import numpy as np
from app.models.base_model import SubModelConfig, ModelConfig, StatType, StatsConfig, StatsResult
from app.models.impls.yolosam import YoloSam
from app.models.impls.maskrcnn import MaskRCNN
from app.models.helpers.config import nano_config, house_config
from app.api.live_models import AvailableModels

from pydantic import BaseModel
from enum import Enum


class Box(BaseModel):
    id: str
    x: float
    y: float
    width: float
    height: float

class SegmentRequest(BaseModel):
    session_id: str
    model: AvailableModels
    blackout_regions: List[Box]=None



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

def blackout_regions(img: np.ndarray, regions: List[Box], save_path: str | Path = None) -> np.ndarray:
    """
    Black out rectangular regions in an image.
    Optionally saves the result for verification.
    """
    img_out = img.copy()
    h, w = img_out.shape[:2]

    for box in regions:
        x1 = max(0, min(int(box.x), w))
        x2 = max(0, min(int(box.x + box.width), w))
        y1 = max(0, min(int(box.y), h))
        y2 = max(0, min(int(box.y + box.height), h))

        img_out[y1:y2, x1:x2] = 0

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv.imwrite(str(save_path), img_out)
        print(f"Blacked-out image saved to {save_path}")

    return img_out

@router.post("/")
async def segment(req: SegmentRequest):
    print(f"Seg request:{req} ")
    session_dir = SESSIONS_DIR / req.session_id

    if not session_dir.exists():
        return {"error": "Invalid session"}

    files = list(session_dir.iterdir())
    if not files:
        return {"error": "No image found"}

    image_path = files[0]

    if req.model == AvailableModels.yolosam:
        model_inst = YoloSam(nano_config, device="cpu")

    elif req.model == AvailableModels.maskrcnn:
        model_inst = MaskRCNN(house_config, device="cpu")

    else:
        return {"error": "Unsupported model"}

    img = model_inst.load_image(image_path)
    if req.blackout_regions:
        print(f"Blacking out {len(req.blackout_regions)} regions! ")
        img= blackout_regions(img, req.blackout_regions,  save_path=f"sessions/{req.session_id}/blackout_check.png")

    result = model_inst.segment(img)

    if req.model != result.model:
        return {"error": f"Mismatch between requested model {req.model} and model used {result.model}"}

    if result.segmentation_mask is None:
        return {"error": "Segmentation returned no mask"}

    mask = normalize_mask(result.segmentation_mask)

    if mask is None or mask.size == 0:
        return {"error": "Mask is empty"}

    # check mask has foreground pixels
    if not np.any(mask):
        return {
            "mask_url": None,
            "metadata": result.metadata,
            "stats": {},
            "model": req.model,
            "warning": "Mask contains no detected particles"
        }

    mask_path = session_dir / "mask.png"
    success = cv.imwrite(str(mask_path), mask)

    if not success:
        return {"error": "Failed to save mask"}


    # Default stats config for now.
    stats_config = StatsConfig(
        enabled={
            StatType.PARTICLE_COUNT,
            StatType.AVG_SIZE,
            StatType.AVG_CIRCULARITY,
            StatType.COVERAGE,
        }
    )

    stats_results = model_inst.compute_stats(mask, stats_config)


    if not stats_results or not getattr(stats_results, "values", None):
        return {"error": "Failed to compute stats"}

    return {
        "mask_url": f"/images/{req.session_id}/mask",
        "metadata": result.metadata,
        "stats": stats_results.values,
        "model": req.model
    }
