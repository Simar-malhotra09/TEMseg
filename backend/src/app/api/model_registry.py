import gc
from threading import Condition, RLock
from typing import Generator, TypedDict

import torch
from fastapi import HTTPException, Request

from app.api.live_models import AvailableModels
from app.logutils import get_logger
from app.models.backends.selection import evict_sam_backends
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
_MODEL_LOAD_LOCK = RLock()

SEGMENTOR_GROUPS: dict[str, tuple[str, tuple[AvailableModels, ...], str]] = {
    "sam_vit_b": (
        "SAM ViT-B",
        (AvailableModels.yolosam, AvailableModels.fasteryolosam),
        "sam",
    ),
    "mobile_sam": ("MobileSAM", (AvailableModels.mobile_sam,), "sam"),
    "microsam_em": (
        "micro-SAM EM ViT-B",
        (AvailableModels.microsam_em,),
        "sam",
    ),
    "yolomaskrcnn": (
        "YOLO Mask R-CNN",
        (AvailableModels.yolomaskrcnn,),
        "maskrcnn",
    ),
    "maskrcnn_synthetic": (
        "Mask R-CNN Synthetic",
        (AvailableModels.maskrcnn_synthetic,),
        "maskrcnn",
    ),
}

MODEL_SEGMENTOR_GROUP = {
    model: group_id
    for group_id, (_, models, _) in SEGMENTOR_GROUPS.items()
    for model in models
}


class SegmentorGroupStatus(TypedDict):
    id: str
    name: str
    models: list[str]
    loaded: bool
    can_eject: bool
    ram_bytes: int | None


def _component_registry_key(group_id: str) -> str:
    return f"segmentor:{group_id}"


def _shared_components_for_model(
    model: AvailableModels, shared_components: dict[str, object]
) -> dict[str, object]:
    components: dict[str, object] = {}
    yolo_component = shared_components.get("detector:yolo_component")
    if yolo_component is not None and model is not AvailableModels.maskrcnn_synthetic:
        components["yolo"] = yolo_component

    group_id = MODEL_SEGMENTOR_GROUP[model]
    segmentor_component = shared_components.get(_component_registry_key(group_id))
    if segmentor_component is not None:
        _, _, component_name = SEGMENTOR_GROUPS[group_id]
        if group_id == "mobile_sam":
            components["mobile_sam"] = segmentor_component
            components["sam"] = segmentor_component
        elif group_id == "microsam_em":
            components["microsam_em"] = segmentor_component
            components["sam"] = segmentor_component
        else:
            components[component_name] = segmentor_component
    return components


def _register_model_components(
    model_id: AvailableModels,
    model_instance: object,
    shared_components: dict[str, object],
) -> None:
    model_components = getattr(model_instance, "components")
    yolo_component = model_components.get("yolo")
    if yolo_component is not None:
        shared_components.setdefault("detector:yolo_component", yolo_component)

    yolo_backend = getattr(model_instance, "_yolo", None)
    if yolo_backend is not None:
        shared_components.setdefault("detector:yolo_backend", yolo_backend)

    group_id = MODEL_SEGMENTOR_GROUP[model_id]
    _, _, component_name = SEGMENTOR_GROUPS[group_id]
    segmentor_component = model_components.get(component_name)
    if segmentor_component is None and group_id in ("mobile_sam", "microsam_em"):
        segmentor_component = model_components.get("sam")
    if segmentor_component is not None:
        shared_components[_component_registry_key(group_id)] = segmentor_component


def get_segmentor_group_status(
    shared_components: dict[str, object],
) -> list[SegmentorGroupStatus]:
    loaded_group_ids = {
        group_id
        for group_id in SEGMENTOR_GROUPS
        if _component_registry_key(group_id) in shared_components
    }
    return [
        {
            "id": group_id,
            "name": name,
            "models": [model.value for model in models],
            "loaded": group_id in loaded_group_ids,
            "can_eject": group_id in loaded_group_ids and len(loaded_group_ids) > 1,
            "ram_bytes": getattr(
                shared_components.get(_component_registry_key(group_id)),
                "ram_bytes",
                None,
            ),
        }
        for group_id, (name, models, _) in SEGMENTOR_GROUPS.items()
    ]


def model_operation(request: Request) -> Generator[None, None, None]:
    """Keep model memory stable for the duration of a model-backed request."""
    app_state = request.app.state
    condition: Condition = app_state.model_lifecycle_condition
    with condition:
        if app_state.model_unloading:
            raise HTTPException(
                status_code=409,
                detail="A model is being unloaded. Retry this request shortly.",
            )
        app_state.active_model_operations += 1
    try:
        yield
    finally:
        with condition:
            app_state.active_model_operations -= 1
            condition.notify_all()


def _build_fasteryolosam(
    device: str, shared_components: dict[str, object]
) -> FasterYoloSam:
    return FasterYoloSam(
        nano_config,
        device=device,
        components=_shared_components_for_model(
            AvailableModels.fasteryolosam, shared_components
        ),
    )


def _build_yolomaskrcnn(
    device: str, shared_components: dict[str, object]
) -> YoloMaskRCNN:
    return YoloMaskRCNN(
        yolomaskrcnn_config,
        AvailableModels.yolomaskrcnn,
        device=device,
        components=_shared_components_for_model(
            AvailableModels.yolomaskrcnn, shared_components
        ),
    )


def _build_model(
    model: AvailableModels,
    device: str,
    shared_components: dict[str, object],
) -> object:
    if model is AvailableModels.yolosam:
        return YoloSam(
            nano_config,
            device=device,
            components=_shared_components_for_model(model, shared_components),
        )
    if model is AvailableModels.fasteryolosam:
        return _build_fasteryolosam(device, shared_components)
    if model is AvailableModels.mobile_sam:
        return MobileSam(
            mobile_sam_pipeline_config,
            device=device,
            components=_shared_components_for_model(model, shared_components),
        )
    if model is AvailableModels.microsam_em:
        return MicroSamEm(
            microsam_em_pipeline_config,
            device=device,
            components=_shared_components_for_model(model, shared_components),
        )
    if model is AvailableModels.yolomaskrcnn:
        return _build_yolomaskrcnn(device, shared_components)
    if model is AvailableModels.maskrcnn_synthetic:
        return MaskRCNN(
            house_synthetic_config,
            AvailableModels.maskrcnn_synthetic,
            device=device,
        )
    raise ValueError(f"Unknown model: {model}")


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


def get_or_load_model(
    models: dict,
    shared_components: dict[str, object],
    model: AvailableModels,
) -> object:
    """Return one cached model, sharing detector and segmentor components."""
    with _MODEL_LOAD_LOCK:
        if model not in models:
            logger.info("Lazily loading model", extra={"model": model.value})
            model_instance = _build_model(model, get_device(), shared_components)
            models[model] = model_instance
            _register_model_components(model, model_instance, shared_components)
        return models[model]


def eject_segmentor_group(
    models: dict,
    shared_components: dict[str, object],
    group_id: str,
) -> None:
    """Unload one segmentor group while retaining shared detector components."""
    if group_id not in SEGMENTOR_GROUPS:
        raise KeyError(f"Unknown segmentor group: {group_id}")
    loaded_groups = [
        key
        for key in SEGMENTOR_GROUPS
        if _component_registry_key(key) in shared_components
    ]
    if group_id not in loaded_groups:
        raise ValueError(f"{SEGMENTOR_GROUPS[group_id][0]} is not loaded")
    if len(loaded_groups) <= 1:
        raise RuntimeError("At least one segmentor group must remain loaded")

    with _MODEL_LOAD_LOCK:
        component_key = _component_registry_key(group_id)
        segmentor_component = shared_components.pop(component_key)
        for model in SEGMENTOR_GROUPS[group_id][1]:
            models.pop(model, None)
        if SEGMENTOR_GROUPS[group_id][2] == "sam":
            evict_sam_backends(segmentor_component)
        del segmentor_component
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available() and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        logger.info("Ejected segmentor group", extra={"group": group_id})
