from contextlib import asynccontextmanager
from cv2 import CALIB_FIX_FOCAL_LENGTH
import numpy as np
from fastapi import FastAPI
import time 
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import asyncio
from fastapi.middleware.cors import CORSMiddleware
from app.api.routers import images, segment, ground_truths, masks
from app.api.live_models import AvailableModels
from app.models.impls.yolosam import YoloSam
from app.models.helpers.config import nano_config, house_config
from app.models.impls.maskrcnn import MaskRCNN

from typing import List
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# parent logger for all routes
routes_logger = logging.getLogger("routes")
routes_logger.setLevel(logging.INFO)

# file handler for routes
file_handler = logging.FileHandler("routes.log")
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(formatter)
routes_logger.addHandler(file_handler)

@asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_old_sessions()

    # initalize on startup, 
    app.state.models = {
        AvailableModels.yolosam: YoloSam(nano_config, device="cpu"),
        AvailableModels.maskrcnn: MaskRCNN(house_config, device="cpu"),
    }

    app.state.warmed_up = False

    # Make a dummy req to warm up SAM in the YoloSAM pipeline
    # warmup up in a diff thread to prevent blocking of server
    async def warmup():
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            await loop.run_in_executor(pool, app.state.models[AvailableModels.yolosam].segment, dummy)
        app.state.warmed_up = True
        routes_logger.info("Models warmed up")

    asyncio.create_task(warmup())
    yield



app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(images.router)
app.include_router(segment.router)
app.include_router(ground_truths.router)
app.include_router(masks.router)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/models", response_model=List[str])
def get_models():
    return [model.value for model in AvailableModels]


def cleanup_old_sessions(max_age_hours: int = 24):
    sessions_dir = Path("sessions")
    
    if not sessions_dir.exists():
        return
    now = time.time()
    for session in sessions_dir.iterdir():
        if not session.is_dir():
            continue
        age = now - session.stat().st_mtime
        if age > max_age_hours * 3600:
            shutil.rmtree(session)
            routes_logger.info(f"Cleaned up session: {session.name}")
