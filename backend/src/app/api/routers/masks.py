from fastapi import APIRouter
from pathlib import Path
from pydantic import BaseModel
from typing import List, Tuple
import numpy as np
import cv2 as cv
import logging
from app.api.utils import extract_instances, rasterize_instances, save_debug_overlay

router = APIRouter(prefix="/masks")
logger = logging.getLogger("routes.masks")
SESSIONS_DIR = Path("sessions")

class Instance(BaseModel):
    id: int
    contour: List[Tuple[float, float]]  # [[x,y], ...]
    bbox: dict                           # {x, y, w, h}
    area: float

class InstancesResponse(BaseModel):
    instances: List[Instance]

class SaveInstancesRequest(BaseModel):
    session_id: str
    instances: List[Instance]


@router.post("/{session_id}/instances")
async def get_instances(session_id: str) -> InstancesResponse:
    session_dir = SESSIONS_DIR / session_id
    mask_path = session_dir / "mask.png"

    if not mask_path.exists():
        return {"error": "No mask found — run segmentation first"}

    mask = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)
    instances = extract_instances(mask)

   # save overlay on original image
    orig_files = list(session_dir.glob("org*"))
    if orig_files:
        save_debug_overlay(
            orig_path=orig_files[0],
            instances=instances,
            save_path=session_dir / "instances_debug.png"
        )
    logger.info(f"[MASKS] Found {len(instances)} instances for session {session_id}")
    return InstancesResponse(instances=instances)


@router.put("/{session_id}/instances")
async def save_instances(session_id: str, req: SaveInstancesRequest):
    session_dir = SESSIONS_DIR / session_id
    mask_path = session_dir / "mask.png"

    if not mask_path.exists():
        return {"error": "No mask found"}

    # read original to get dimensions
    original = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)
    h, w = original.shape[:2]

    # rasterize polygons back to binary mask
    new_mask = rasterize_instances(req.instances, h, w)

    # save as colorized mask
    colored = cv.applyColorMap(new_mask, cv.COLORMAP_TURBO)
    colored[new_mask == 0] = 0
    cv.imwrite(str(mask_path), colored)

    logger.info(f"[MASKS] Saved {len(req.instances)} instances for session {session_id}")
    return {"mask_url": f"/images/{session_id}/mask", "instance_count": len(req.instances)}
