"""Platform/capability probe: pick the inference backends for yolosam.

arm Macs with the CoreML .mlpackages present get CoreML yolo+sam with a
torch SAM kept alongside for point-prompt endpoints. Everything else gets
the classic onnx+pth torch stack. TEMSEG_COREML=0 forces classic.

Backends are stateless per call (encode/decode take everything as args),
so instances are cached process-wide keyed by assets+device: YoloSam and
FasterYoloSam share one set of loaded/compiled CoreML models instead of
re-paying the load+compile warmup on every model-class switch.
"""

import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict

from app.models.helpers.settings import settings

from .base import SamBackend, YoloBackend
from .impls.sam.coreml_sam import CoreMLSamBackend
from .impls.sam.torch_sam import FasterTorchSamBackend, TorchSamBackend
from .impls.yolo.coreml_yolo import CoreMLYoloBackend
from .impls.yolo.ort_yolo import OrtYoloBackend

YOLO_PKG = settings.WEIGHTS_DIR / "best12x.mlpackage"
ENC_PKG = settings.WEIGHTS_DIR / "sam_encoder_vit_b_d12_fp32.mlpackage"
DEC_PKG = settings.WEIGHTS_DIR / "sam_decoder_head16_fp32.mlpackage"

_backend_cache: dict[tuple, tuple[YoloBackend, SamBackend, SamBackend | None]] = {}


def coreml_available() -> bool:
    if os.environ.get("TEMSEG_COREML", "").strip().lower() in ("0", "false", "no"):
        return False
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    return all(p.exists() for p in (YOLO_PKG, ENC_PKG, DEC_PKG))


def choose_backends(
    components: Dict[str, Any], device: str, faster: bool = False
) -> tuple[YoloBackend, SamBackend, SamBackend | None]:
    """(yolo, sam, prompt_sam). prompt_sam is a torch SAM for point-prompt
    endpoints; it may be the same object as sam on the classic path.
    Cached: same key -> same instances, no recompile."""
    if coreml_available():
        key = ("coreml", str(YOLO_PKG), str(ENC_PKG), str(DEC_PKG),
               id(components["sam"]), device)
        if key in _backend_cache:
            return _backend_cache[key]
        yolo: YoloBackend = CoreMLYoloBackend(YOLO_PKG)
        sam: SamBackend = CoreMLSamBackend(ENC_PKG, DEC_PKG, device)
        prompt_sam = TorchSamBackend(components["sam"], device)
        cached = (yolo, sam, prompt_sam)
        _backend_cache[key] = cached
        return cached
    key = ("classic", faster, id(components["yolo"]), id(components["sam"]), device)
    if key in _backend_cache:
        return _backend_cache[key]
    yolo = OrtYoloBackend(components["yolo"], device)
    sam_cls = FasterTorchSamBackend if faster else TorchSamBackend
    sam = sam_cls(components["sam"], device)
    cached = (yolo, sam, None)
    _backend_cache[key] = cached
    return cached
