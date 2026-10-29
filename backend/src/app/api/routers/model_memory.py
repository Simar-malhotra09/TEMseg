from fastapi import APIRouter, HTTPException, Request

from app.api.model_registry import (
    SEGMENTOR_GROUPS,
    eject_segmentor_group,
    get_segmentor_group_status,
)

router = APIRouter(prefix="/models/memory", tags=["model memory"])


@router.post("/{group_id}/eject")
def eject_model_segmentor(group_id: str, request: Request) -> dict:
    if group_id not in SEGMENTOR_GROUPS:
        raise HTTPException(status_code=404, detail="Unknown segmentor group")

    app_state = request.app.state
    condition = app_state.model_lifecycle_condition
    with condition:
        if app_state.model_unloading:
            raise HTTPException(
                status_code=409, detail="A model is already being unloaded"
            )
        if app_state.active_model_operations > 0:
            raise HTTPException(
                status_code=409,
                detail="Wait for current segmentation or SAM tools to finish before ejecting a model.",
            )
        app_state.model_unloading = True

    try:
        eject_segmentor_group(
            app_state.models,
            app_state.model_components,
            group_id,
        )
        ejected_models = set(SEGMENTOR_GROUPS[group_id][1])
        for session_id, model_id in list(app_state.embedding_model_cache.items()):
            if model_id in ejected_models:
                app_state.embedding_model_cache.pop(session_id)
                app_state.embedding_cache.pop(session_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    finally:
        with condition:
            app_state.model_unloading = False
            condition.notify_all()

    return {
        "segmentor_groups": get_segmentor_group_status(app_state.model_components),
        "ejected_group": group_id,
    }
