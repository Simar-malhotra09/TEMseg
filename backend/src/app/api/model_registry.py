import torch

from app.api.live_models import AvailableModels
from app.logutils import get_logger
from app.models.helpers.config import (
    house_synthetic_config,
    nano_config,
    yolomaskrcnn_config,
)
from app.models.impls.fasteryolosam import FasterYoloSam
from app.models.impls.maskrcnn import MaskRCNN
from app.models.impls.yolomaskrcnn import YoloMaskRCNN
from app.models.impls.yolosam import YoloSam

logger = get_logger("registry")


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


def _build_yolosam(device: str):
    """HACK BRANCH: dual-CoreML yolosam on arm Macs when the .mlpackages
    exist (TEMSEG_COREML=0 forces the classic torch path)."""
    from app.models.impls.yolosam_coreml import YoloSamCoreML, coreml_available

    if coreml_available():
        logger.info("yolosam: using dual CoreML backend (coreml-dual-pipeline-hack)")
        return YoloSamCoreML(nano_config, device=device)
    return YoloSam(nano_config, device=device)


_MODEL_BUILDERS = {
    AvailableModels.yolosam: _build_yolosam,
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
        if model is AvailableModels.fasteryolosam:
            models[model] = _build_fasteryolosam(models, device)
        else:
            builder = _MODEL_BUILDERS.get(model)
            if builder is None:
                raise ValueError(f"Unknown model: {model}")
            logger.info(f"Lazily loading {model} on demand")
            models[model] = builder(device)
    return models[model]
