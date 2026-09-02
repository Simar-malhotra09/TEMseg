from contextlib import asynccontextmanager
from fastapi import FastAPI
import asyncio
import time
import shutil
from pathlib import Path
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from app.api.routers import images, segment, ground_truths, masks, export, rf, config
from app.api.live_models import AvailableModels
from app.api.model_registry import get_or_load_model

from typing import List

from app.logutils import get_logger, init_logging

init_logging()
log = get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # cleanup_old_sessions(force=True)

    # initalize on startup: only YoloSAM, the default/better model, is loaded
    # eagerly. YoloMaskRCNN is lazily loaded on first use (see model_registry.py)
    # to avoid holding both models in RAM at once.
    app.state.models = {}
    get_or_load_model(app.state.models, AvailableModels.yolosam)
    app.state.embedding_cache: Dict[str, Dict] = {}

    # Warm YOLO at startup instead of on image upload. The ONNX graph has a
    # static [1,3,640,640] input, so any dummy image triggers the one-time
    # ~5.7s CoreML compile and the result covers every uploaded shape.
    # FasterYoloSam shares this YOLO session via the registry, so one warm
    # serves both pipelines. Runs on a thread so lifespan isn't blocked;
    # segment requests await the event rather than racing the compile.
    app.state.yolo_warm = asyncio.Event()

    def _warm_yolo():
        try:
            yolosam = app.state.models[AvailableModels.yolosam]
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            yolosam.components["yolo"].predict(
                source=dummy, verbose=False, conf=0.25, device=yolosam.device
            )
            log.info("YOLO startup warmup complete")
        except Exception as e:
            log.warning(f"YOLO warmup failed (non-fatal): {e}")

    async def _background_warmup():
        try:
            await asyncio.to_thread(_warm_yolo)
        finally:
            app.state.yolo_warm.set()

    asyncio.create_task(_background_warmup())
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
            log.info(f"Cleaned up old session: {session.name}")
