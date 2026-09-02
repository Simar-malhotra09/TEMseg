import torch

from app.api.live_models import AvailableModels
from app.logutils import get_logger
from app.models.helpers.config import (
    house_synthetic_config,
    nano_config,
    yolomaskrcnn_config,
)
from app.models.impls.fastyolosam import FastYoloSam
from app.models.impls.maskrcnn import MaskRCNN
from app.models.impls.yolomaskrcnn import YoloMaskRCNN
from app.models.impls.yolosam import YoloSam

logger = get_logger("registry")


def _build_fastyolosam(models: dict, device: str) -> FastYoloSam:
    """Build FastYoloSam on stock YoloSam's components when available.

    Both pipelines use the same weight files (nano_config), so reusing the
    loaded dict means one YOLO ONNX session (one ~5.7s CoreML compile) and one
    SAM vit_b copy in RAM instead of a duplicated pair. Falls back to a fresh
    load if YoloSam has never been instantiated.
    """
    base = models.get(AvailableModels.yolosam)
    if base is not None:
        logger.info("FastYoloSam reusing YoloSam components (shared load)")
        return FastYoloSam(nano_config, device=device, components=base.components)
    return FastYoloSam(nano_config, device=device)


_MODEL_BUILDERS = {
    AvailableModels.yolosam: lambda device: YoloSam(nano_config, device=device),
    AvailableModels.yolomaskrcnn: lambda device: YoloMaskRCNN(
        yolomaskrcnn_config, AvailableModels.yolomaskrcnn, device=device
    ),
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
    if model not in models:
        device = get_device()
        if model is AvailableModels.fastyolosam:
            models[model] = _build_fastyolosam(models, device)
        else:
            builder = _MODEL_BUILDERS.get(model)
            if builder is None:
                raise ValueError(f"Unknown model: {model}")
            logger.info(f"Lazily loading {model} on demand")
            models[model] = builder(device)
    return models[model]
