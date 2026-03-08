from typing import List, Dict, Any
from pathlib import Path 
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse
import logging
import time
import cv2 as cv
import numpy as np
from app.models.base_model import SubModelConfig, ModelConfig, StatType, StatsConfig, StatsResult
from app.models.impls.yolosam import YoloSam
from app.models.impls.maskrcnn import MaskRCNN
from app.models.helpers.config import nano_config, house_config
from app.api.live_models import AvailableModels
from app.api.utils import Box, normalize_mask, blackout_regions, inverse_blackout_regions, colorize_components_inplace, batch_seg_patches
from pydantic import BaseModel



class SegmentRequest(BaseModel):
    session_id: str
    model: AvailableModels
    regions:List[Box]= None
    blackout: bool = False
    inverse_blackout: bool = False
    colorize: bool= True # colorize masks for overlay

class SegmentResponse(BaseModel):
    model: AvailableModels
    mask_url: str 
    metadata: Dict[str, Any]
    stats: Dict
    time_elapsed:float
    

router = APIRouter(prefix="/segment")
logger = logging.getLogger("routes.segment")
SESSIONS_DIR = Path("sessions")
logger.disabled= True


@router.post("/")
async def segment(req: SegmentRequest, request: Request):
    model_inst = request.app.state.models.get(req.model.value)
    cache = request.app.state.embedding_cache

    # return embedding else None
    embedding = cache.get(req.session_id)  

    logging.info(f"Model chosen: {model_inst}")
    if not model_inst:
        return {"error": f"Model {req.model} not found"}

    start_time = time.perf_counter()

    if req.blackout and req.inverse_blackout:
        raise ValueError("Only one of blackout_regions or inverse_blackout_regions may be True.")

    logger.info(f"[SEG] Request received for session: {req.session_id}")
    logger.info(f"[SEG] Model requested: {req.model}")
    logger.info(f"[SEG] Regions selected : {len(req.regions)}")
    mode = "blacked_out" if req.blackout else "kept"
    logger.info(f"[SEG] Regions are being {mode}")
    logger.info(f"[SEG] colorize: {req.colorize}")

    session_dir = SESSIONS_DIR / req.session_id
    logger.info(f"[SEG] Session dir resolved to: {session_dir}")

    if not session_dir.exists():
        logger.warning("[SEG] Invalid session directory")
        return {"error": "Invalid session"}

    orig_files = list(session_dir.glob("org_*"))
    logger.info(f"[SEG] Found original files: {orig_files}")

    if not orig_files:
        logger.warning("[SEG] No original image found")
        return {"error": "No image found"}

    image_path = orig_files[0]
    logger.info(f"[SEG] Using image: {image_path}")

    # Model selection
    # if req.model == AvailableModels.yolosam:
    #     logger.info("[SEG] Initializing YoloSAM model")
    #     model_inst = YoloSam(nano_config, device="cpu")
    #
    # elif req.model == AvailableModels.maskrcnn:
    #     logger.info("[SEG] Initializing MaskRCNN model")
    #     model_inst = MaskRCNN(house_config, device="cpu")
    #
    # else:
    #     logger.error(f"[SEG] Unsupported model requested: {req.model}")
    #     return {"error": "Unsupported model"}

    # Load image
    logger.info("[SEG] Loading image...")
    img = model_inst.load_image(image_path)



    # Apply inverse blackout if needed
    if req.inverse_blackout:

        # if yolosam: use batch seg
        if req.model == AvailableModels.yolosam:
            logger.info(f"[SEG] Batch seg on {len(req.regions)} patches")
            result = batch_seg_patches(img, req.regions, model_inst)
            mask= result.segmentation_mask

        # maskrcnn currently doesn't impl batch seg!
        elif req.model== AvailableModels.maskrcnn:
            img= inverse_blackout_regions(
                    img, 
                    req.regions,
                    save_path=f"sessions/{req.session_id}/inverse_blackout_check.png"
                    )
            result= model_inst.segment(img)
            mask = result.segmentation_mask

        else:
            logger.error(f"[SEG] Unsupported model for inverse blackout: {req.model}")
            return {"error": f"Unsupported model {req.model}"}   
    else:
        # Apply blackout if needed
        if req.blackout:
            logger.info(f"[SEG] Applying blackout to {len(req.regions)} regions")
            img = blackout_regions(
                img,
                req.regions,
                save_path=f"sessions/{req.session_id}/blackout_check.png"
            )

        # for blackout seg or when neither applied


        if req.model== AvailableModels.yolosam:
            # we cache embds for images and use them if available since
            # SAM's segment operations are expensive. 
            # This'll only work within a session and not across. 
            result = model_inst.segment(img, embedding)
            mask = result.segmentation_mask
        else:
            result = model_inst.segment(img)
            mask = result.segmentation_mask



    if req.model != result.model:
        logger.error("[SEG] Model mismatch between request and result")
        return {
            "error": f"Mismatch between requested model {req.model} and model used {result.model}"
        }

    if result.segmentation_mask is None:
        logger.error("[SEG] Segmentation returned no mask")
        return {"error": "Segmentation returned no mask"}

    logger.info("[SEG] Normalizing mask...")
    mask = normalize_mask(result.segmentation_mask)

    if req.colorize:
        save_mask = colorize_components_inplace(mask)
    else:
        save_mask = mask


    if mask is None or mask.size == 0:
        logger.error("[SEG] Mask normalization failed or mask empty")
        return {"error": "Mask is empty"}

    if not np.any(mask):
        logger.warning("[SEG] Mask contains no foreground pixels")
        return {
            "mask_url": None,
            "metadata": result.metadata,
            "stats": {},
            "model": req.model,
            "warning": "Mask contains no detected particles"
        }

    mask_path = session_dir / "mask.png"
    logger.info(f"[SEG] Saving mask to: {mask_path}")
    success= cv.imwrite(str(mask_path), save_mask)

    if not success:
        logger.error("[SEG] Failed to save mask")
        return {"error": "Failed to save mask"}

    logger.info("[SEG] Computing stats...")

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
        logger.error("[SEG] Failed to compute stats")
        return {"error": "Failed to compute stats"}

    logger.info(f"[SEG] Stats computed: {stats_results.values}")
    if result.embedding:
        cache[req.session_id] = result.embedding
        logger.info(f"[SEG] Cached SAM embedding for session {req.session_id}")

    logger.info("[SEG] Segmentation completed successfully")

    elapsed = time.perf_counter() - start_time
    logger.info(f"[SEG] Total request time: {elapsed:.4f}s")

    
    return SegmentResponse(
        mask_url=f"/images/{req.session_id}/mask",
        metadata=result.metadata,
        stats= stats_results.values,
        model=  req.model,
        time_elapsed= elapsed,
        )
