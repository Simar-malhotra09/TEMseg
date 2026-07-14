import logging

import numpy as np

from app.models.impls.rf_recovery import RFRecovery

logger = logging.getLogger(__name__)

_DEFAULT_MIN_AREA = 50

_cache: dict[str, RFRecovery] = {}


def get_or_train(
    session_key: str,
    image: np.ndarray,
    mask: np.ndarray,
    min_area: int = _DEFAULT_MIN_AREA,
    bg_mask: np.ndarray | None = None,
) -> RFRecovery:
    if session_key not in _cache:
        logger.info(f"[RF-Cache] Training new RFRecovery for session={session_key}")
        rf = RFRecovery(min_area=min_area)
        rf.train(image, mask, bg_mask)
        _cache[session_key] = rf
    return _cache[session_key]


def update(session_key: str, image: np.ndarray, mask: np.ndarray) -> None:
    if session_key not in _cache:
        logger.warning(
            f"[RF-Cache] No entry for session={session_key}, skipping update"
        )
        return
    _cache[session_key].update(image, mask)


def evict(session_key: str) -> None:
    removed = _cache.pop(session_key, None)
    if removed is not None:
        logger.info(f"[RF-Cache] Evicted session={session_key}")
    else:
        logger.debug(f"[RF-Cache] Evict called for unknown session={session_key}")
