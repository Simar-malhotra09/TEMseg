import logging

import torch

from app.api.live_models import AvailableModels
from app.models.helpers.config import house_config, nano_config
from app.models.impls.maskrcnn import MaskRCNN
from app.models.impls.yolosam import YoloSam

logger = logging.getLogger("routes")

_MODEL_BUILDERS = {
    AvailableModels.yolosam: lambda device: YoloSam(nano_config, device=device),
    AvailableModels.maskrcnn: lambda device: MaskRCNN(house_config, device=device),
}


def get_device() -> str:
    """Pick the best available device at startup."""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"[STARTUP] Using device: {device}")
    return device


def get_or_load_model(models: dict, model: AvailableModels):
    """Return the cached model instance, lazily instantiating it on first use."""
    if model not in models:
        builder = _MODEL_BUILDERS.get(model)
        if builder is None:
            raise ValueError(f"Unknown model: {model}")
        logger.info(f"[MODEL] Lazily loading {model} on demand")
        models[model] = builder(get_device())
    return models[model]
