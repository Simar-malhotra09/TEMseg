from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import asyncio
import time
import shutil
from pathlib import Path
from typing import Dict
from uuid import uuid4
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from app.api.routers import (
    images,
    segment,
    ground_truths,
    masks,
    export,
    rf,
    config,
    sessions,
)
from app.api.live_models import AvailableModels
from app.api.model_registry import get_device, get_or_load_model

from typing import List

from app.logutils import (
    get_logger,
    init_logging,
    set_request_id,
    ui_event,
    get_ui_events,
)

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
    ui_event(
        "MODELS_READY",
        "Analysis engine is ready — upload an image to begin.",
        level="info",
    )

    # Warm the selected yolo backend at startup instead of on first upload
    # (classic: one-time ORT CoreML-EP compile; coreml backend warms in its
    # own __init__). Runs on a thread so lifespan isn't blocked; segment
    # requests await the event rather than racing the compile.
    app.state.yolo_warm = asyncio.Event()

    def _warm_yolo():
        try:
            yolosam = app.state.models[AvailableModels.yolosam]
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            yolosam._yolo.detect(dummy)
            log.info("YOLO startup warmup complete")
        except Exception as e:
            log.warning(f"YOLO warmup failed (non-fatal): {e}")
            ui_event(
                "STARTUP_WARNING",
                "The analysis engine started, but first use may be slow. "
                "If images take a long time to process, restart the app.",
                level="warning",
            )

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


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Tag every log line produced while serving this request with req=<id>."""
    rid = uuid4().hex[:6]
    set_request_id(rid)
    response = await call_next(request)
    response.headers["X-Request-Id"] = rid
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    ui_event(
        "BACKEND_ERROR",
        "Something went wrong while processing your request. "
        "Please try again — if this keeps happening, restart the app.",
        level="error",
    )
    log.error(f"Unhandled error on {request.url.path}: {exc!r}")
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


app.include_router(images.router)
app.include_router(segment.router)
app.include_router(ground_truths.router)
app.include_router(masks.router)
app.include_router(export.router)
app.include_router(rf.router)
app.include_router(config.router)
app.include_router(sessions.router)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/models", response_model=List[str])
def get_models():
    return [model.value for model in AvailableModels]


@app.get("/ui/events")
def ui_events(since: int = 0):
    """User-facing event feed for the frontend status panel."""
    return {"events": get_ui_events(since)}


def _engine_of(backend) -> str | None:
    if backend is None:
        return None
    name = type(backend).__name__.lower()
    if "coreml" in name:
        return "coreml"
    if "ort" in name or "onnx" in name:
        return "onnx"
    return "torch"


def _weight_entry(path: Path) -> dict:
    if path.is_dir():
        size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    elif path.exists():
        size = path.stat().st_size
    else:
        size = 0
    return {
        "name": path.name,
        "present": path.exists(),
        "size_mb": round(size / (1024 * 1024), 1),
    }


@app.get("/system/info")
def system_info(request: Request):
    """For settings panel in client"""
    from app.logutils import default_log_dir
    from app.models.backends.selection import (
        DEC_PKG,
        ENC_PKG,
        YOLO_PKG,
        coreml_available,
        process_peak_rss_bytes,
    )
    from app.models.helpers.settings import settings

    device = get_device()
    yolosam = request.app.state.models.get(AvailableModels.yolosam.value)
    coreml = coreml_available()

    def role(key, name, backend, weight_paths):
        engine = _engine_of(backend)
        return {
            "key": key,
            "name": name,
            "loaded": backend is not None,
            "engine": engine,
            "runs_on": (
                "apple neural engine" if engine == "coreml" else device
            ) if backend is not None else None,
            "ram_bytes": (
                getattr(backend, "ram_bytes", None) if backend is not None else None
            ),
            "weights": [_weight_entry(Path(p)) for p in weight_paths],
        }

    yolo_backend = yolosam._yolo if yolosam is not None else None
    sam_backend = yolosam._sam if yolosam is not None else None
    models = [
        role(
            "yolo",
            "yolo",
            yolo_backend,
            [YOLO_PKG] if coreml else [settings.YOLO_MODEL_PATH],
        ),
        role(
            "sam",
            "sam",
            sam_backend,
            [ENC_PKG, DEC_PKG] if coreml else [settings.SAM_MODEL_PATH],
        ),
    ]
    if coreml:
        # classic path serves prompts from the same torch sam; only coreml
        # keeps a separate prompt model
        prompt_backend = yolosam._prompt_sam if yolosam is not None else None
        models.append(
            role(
                "prompt_sam",
                "sam (prompts)",
                prompt_backend,
                [settings.SAM_MODEL_PATH],
            )
        )
    return {
        "device": device,
        "log_dir": str(default_log_dir()),
        "weights_dir": str(settings.WEIGHTS_DIR),
        "peak_rss_bytes": process_peak_rss_bytes(),
        "models": models,
    }


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
