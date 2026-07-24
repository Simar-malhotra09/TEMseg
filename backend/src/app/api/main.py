from contextlib import asynccontextmanager
from fastapi import FastAPI
import time
import shutil
from pathlib import Path
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware
from app.api.routers import images, segment, ground_truths, masks, export, rf, config
from app.api.live_models import AvailableModels
from app.api.model_registry import get_or_load_model

from typing import List
import logging

import os
import platform


LOG_FORMAT = "%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATEFMT,
)

# resolve a writable, per-user log dir (never inside the app bundle)
system = platform.system()
if system == "Windows":
    base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    log_dir = os.path.join(base, "TEMseg", "logs")
elif system == "Darwin":
    log_dir = os.path.expanduser("~/Library/Logs/TEMseg")
else:  # Linux
    log_dir = os.path.expanduser("~/.local/state/TEMseg/logs")

os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, "routes.log")

# parent logger for all routes
routes_logger = logging.getLogger("routes")
routes_logger.setLevel(logging.INFO)
formatter = logging.Formatter(LOG_FORMAT, datefmt=DATEFMT)

# file handler for routes
if not routes_logger.handlers:
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    routes_logger.addHandler(file_handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # cleanup_old_sessions(force=True)

    # initalize on startup: only YoloSAM, the default/better model, is loaded
    # eagerly. MaskRCNN is lazily loaded on first use (see model_registry.py)
    # to avoid holding both models in RAM at once.
    app.state.models = {}
    get_or_load_model(app.state.models, AvailableModels.yolosam)
    app.state.embedding_cache: Dict[str, Dict] = {}
    app.state.warmed_up = False
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
app.include_router(export.router)
app.include_router(rf.router)
app.include_router(config.router)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/models", response_model=List[str])
def get_models():
    return [model.value for model in AvailableModels]


def cleanup_old_sessions(max_age_hours: int = 24, force=False):
    sessions_dir = Path("sessions")

    if not sessions_dir.exists():
        return

    now = time.time()
    if force:
        max_age_hours = 0
    for session in sessions_dir.iterdir():
        if not session.is_dir():
            continue
        age = now - session.stat().st_mtime
        if age > max_age_hours * 3600:
            shutil.rmtree(session)
            routes_logger.info(f"Cleaned up old session: {session.name}")
