from contextlib import asynccontextmanager
from cv2 import CALIB_FIX_FOCAL_LENGTH
from fastapi import FastAPI
import time 
import shutil
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from app.api.routers import images, segment, ground_truths, masks
from app.api.live_models import AvailableModels
from app.models.impls.yolosam import YoloSam
from app.models.helpers.config import nano_config, house_config
from app.models.impls.maskrcnn import MaskRCNN

from typing import List
import logging

@asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_old_sessions()
    # load on startup, available for entire app lifetime
    app.state.models = {
        AvailableModels.yolosam: YoloSam(nano_config, device="cpu"),
        AvailableModels.maskrcnn: MaskRCNN(house_config, device="cpu"),
    }
    yield


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
        age = now - session.stat().st_mtime
        if age > max_age_hours * 3600:
            shutil.rmtree(session)
            logger.info(f"Cleaned up session: {session.name}")
