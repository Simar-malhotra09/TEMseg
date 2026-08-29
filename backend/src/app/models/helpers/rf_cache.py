from pathlib import Path

import numpy as np

from app.api.instances import MIN_INSTANCE_AREA, load_instances
from app.logutils import get_logger
from app.models.impls.rf_recovery import RFRecovery

logger = get_logger("RFRecovery", sub="cache")

SESSIONS_DIR = Path("sessions")

# Fraction of the existing (model-segmented) median particle area used as the
# RF's missed-region floor.
_MIN_AREA_RATIO = 0.3

_cache: dict[str, RFRecovery] = {}


def _derive_min_area(session_key: str) -> tuple[int, str]:
    """
    Min area for RF-recovered regions, derived from particles the model has
    already segmented for this session. Falls back to the app-wide minimum
    instance area when the session has no prior segmented objects to derive
    from.

    Returns (min_area, source) where source is "median" when derived from prior
    instances, "fallback" when there were none to derive from.
    """
    cached = load_instances(SESSIONS_DIR / session_key)
    if cached is None:
        return MIN_INSTANCE_AREA, "fallback"
    instances, _ = cached
    if not instances:
        return MIN_INSTANCE_AREA, "fallback"
    median_area = float(np.median([inst["area"] for inst in instances]))
    return max(MIN_INSTANCE_AREA, int(_MIN_AREA_RATIO * median_area)), "median"


def get_or_train(
    session_key: str,
    image: np.ndarray,
    mask: np.ndarray,
    min_area: int | None = None,
    bg_mask: np.ndarray | None = None,
) -> RFRecovery:
    if session_key not in _cache:
        if min_area is not None:
            resolved_min_area, min_area_source = min_area, "explicit"
        else:
            resolved_min_area, min_area_source = _derive_min_area(session_key)
        logger.info(
            f"Training new RFRecovery for session={session_key} "
            f"min_area={resolved_min_area} min_area_source={min_area_source}"
        )
        rf = RFRecovery(min_area=resolved_min_area)
        rf.train(image, mask, bg_mask)
        _cache[session_key] = rf
    return _cache[session_key]


def update(session_key: str, image: np.ndarray, mask: np.ndarray) -> None:
    if session_key not in _cache:
        logger.warning(
            f"No entry for session={session_key}, skipping update"
        )
        return
    _cache[session_key].update(image, mask)


def evict(session_key: str) -> None:
    removed = _cache.pop(session_key, None)
    if removed is not None:
        logger.info(f"Evicted session={session_key}")
    else:
        logger.debug(f"Evict called for unknown session={session_key}")
