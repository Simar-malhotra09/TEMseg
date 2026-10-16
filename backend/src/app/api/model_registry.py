from threading import Lock

import torch

from app.api.live_models import AvailableModels
from app.logutils import get_logger
from app.models.helpers.config import (
    house_synthetic_config,
    mobile_sam_pipeline_config,
    microsam_em_pipeline_config,
    nano_config,
    yolomaskrcnn_config,
)
from app.models.impls.fasteryolosam import FasterYoloSam
from app.models.impls.maskrcnn import MaskRCNN
from app.models.impls.mobile_sam import MobileSam
from app.models.impls.microsam_em import MicroSamEm
from app.models.impls.yolomaskrcnn import YoloMaskRCNN
from app.models.impls.yolosam import YoloSam

logger = get_logger("registry")
_MODEL_LOAD_LOCK = Lock()


def _build_fasteryolosam(models: dict, device: str) -> FasterYoloSam:
    """Build FasterYoloSam on stock YoloSam's components when available.

    Both pipelines use the same weight files (nano_config), so reusing the
    loaded dict means one YOLO ONNX session (one ~5.7s CoreML compile) and one
    SAM vit_b copy in RAM instead of a duplicated pair. Falls back to a fresh
    load if YoloSam has never been instantiated.
    """
    base = models.get(AvailableModels.yolosam)
    if base is not None:
        logger.info("FasterYoloSam reusing YoloSam components (shared load)")
        return FasterYoloSam(nano_config, device=device, components=base.components)
    return FasterYoloSam(nano_config, device=device)


def _build_yolomaskrcnn(models: dict, device: str) -> YoloMaskRCNN:
    """Build YoloMaskRCNN sharing YoloSam's detector when available, so both
    model families run one yolo backend (no duplicate load/compile)."""
    base = models.get(AvailableModels.yolosam)
    if base is not None:
        logger.info("YoloMaskRCNN reusing YoloSam components (shared detector)")
        return YoloMaskRCNN(
            yolomaskrcnn_config,
            AvailableModels.yolomaskrcnn,
            device=device,
            components=base.components,
        )
    return YoloMaskRCNN(
        yolomaskrcnn_config, AvailableModels.yolomaskrcnn, device=device
    )


def _build_microsam_em(models: dict, device: str) -> MicroSamEm:
    base = models.get(AvailableModels.yolosam)
    shared_yolo = (
        {"yolo": base.components["yolo"]}
        if base is not None and "yolo" in base.components
        else None
    )
    return MicroSamEm(
        microsam_em_pipeline_config,
        device=device,
        components=shared_yolo,
    )


def _build_mobile_sam(models: dict, device: str) -> MobileSam:
    base = models.get(AvailableModels.yolosam)
    shared_yolo = (
        {"yolo": base.components["yolo"]}
        if base is not None and "yolo" in base.components
        else None
    )
    return MobileSam(
        mobile_sam_pipeline_config,
        device=device,
        components=shared_yolo,
    )


_MODEL_BUILDERS = {
    AvailableModels.yolosam: lambda device: YoloSam(nano_config, device=device),
    AvailableModels.maskrcnn_synthetic: lambda device: MaskRCNN(
        house_synthetic_config, AvailableModels.maskrcnn_synthetic, device=device
    ),
}


def get_device() -> str:
    """Pick the best available device at startup."""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Using device: {device}")
    return device


def get_or_load_model(models: dict, model: AvailableModels):
    """Return the cached model instance, lazily instantiating it on first use."""
    with _MODEL_LOAD_LOCK:
        if model not in models:
            device = get_device()
            logger.info("Lazily loading model", extra={"model": model.value})
            if model is AvailableModels.fasteryolosam:
                models[model] = _build_fasteryolosam(models, device)
            elif model is AvailableModels.mobile_sam:
                models[model] = _build_mobile_sam(models, device)
            elif model is AvailableModels.microsam_em:
                models[model] = _build_microsam_em(models, device)
            elif model is AvailableModels.yolomaskrcnn:
                models[model] = _build_yolomaskrcnn(models, device)
            else:
                builder = _MODEL_BUILDERS.get(model)
                if builder is None:
                    raise ValueError(f"Unknown model: {model}")
                models[model] = builder(device)
        return models[model]
