"""Platform/capability probe: pick the inference backends for yolosam.

arm Macs with the CoreML .mlpackages present get CoreML yolo+sam with a
torch SAM kept alongside for point-prompt endpoints. Everything else gets
the classic onnx+pth torch stack. TEMSEG_COREML=0 forces classic.

coremltools must be importable too: packaged builds don't bundle it yet
(spec hiddenimports), so those fall back to classic until the mlpackages
ship via the manifest and the spec bundles coremltools.

Backends are stateless per call (encode/decode take everything as args),
so instances are cached process-wide keyed by assets+device: YoloSam and
FasterYoloSam share one set of loaded/compiled CoreML models instead of
re-paying the load+compile warmup on every model-class switch.
"""

import importlib.util
import os
import platform
import sys
from typing import Any, Dict

from app.logutils import get_logger
from app.models.helpers.settings import settings

from .base import SamBackend, YoloBackend
from .impls.sam.coreml_sam import CoreMLSamBackend
from .impls.sam.torch_sam import FasterTorchSamBackend, TorchSamBackend
from .impls.yolo.coreml_yolo import CoreMLYoloBackend
from .impls.yolo.ort_yolo import OrtYoloBackend

logger = get_logger("Backends")

YOLO_PKG = settings.WEIGHTS_DIR / "best12x.mlpackage"
ENC_PKG = settings.WEIGHTS_DIR / "sam_encoder_vit_b_d12_fp32.mlpackage"
DEC_PKG = settings.WEIGHTS_DIR / "sam_decoder_head16_fp32.mlpackage"

_backend_cache: dict[tuple, tuple[YoloBackend, SamBackend, SamBackend | None]] = {}


def process_peak_rss_bytes() -> int | None:
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak) if sys.platform == "darwin" else int(peak) * 1024
    except Exception:
        return None


def device_allocated_bytes(device: str) -> int | None:
    # MPS/CUDA weights live outside process RSS, so measure device-side too.
    try:
        import torch

        if device == "mps" and torch.backends.mps.is_available():
            return int(torch.mps.current_allocated_memory())
        if device == "cuda" and torch.cuda.is_available():
            return int(torch.cuda.memory_allocated())
    except Exception:
        return None
    return None


def ram_since(
    before: tuple[int | None, int | None], device: str
) -> int | None:
    rss_after = process_peak_rss_bytes()
    alloc_after = device_allocated_bytes(device)
    if before[1] is not None and alloc_after is not None:
        return max(0, alloc_after - before[1])
    if before[0] is not None and rss_after is not None:
        return max(0, rss_after - before[0])
    return None


def coreml_available() -> bool:
    if os.environ.get("TEMSEG_COREML", "").strip().lower() in ("0", "false", "no"):
        return False
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    if importlib.util.find_spec("coremltools") is None:
        logger.info("coremltools not importable — using classic backends")
        return False
    return all(p.exists() for p in (YOLO_PKG, ENC_PKG, DEC_PKG))


def choose_yolo_backend(components: Dict[str, Any], device: str) -> YoloBackend:
    """The yolo backend for a given platform, cached by assets+device so
    every model family (yolosam, yolomaskrcnn) shares one detector."""
    if coreml_available():
        key = ("coreml-yolo", str(YOLO_PKG))
    else:
        key = ("classic-yolo", id(components["yolo"]), device)
    if key not in _backend_cache:
        backend = (
            CoreMLYoloBackend(YOLO_PKG)
            if coreml_available()
            else OrtYoloBackend(components["yolo"], device)
        )
        _backend_cache[key] = backend
    return _backend_cache[key]


def choose_backends(
    components: Dict[str, Any], device: str, faster: bool = False
) -> tuple[YoloBackend, SamBackend, SamBackend | None]:
    """(yolo, sam, prompt_sam). prompt_sam is a torch SAM for point-prompt
    endpoints; it may be the same object as sam on the classic path.
    Cached: same key -> same instances, no recompile."""
    yolo = choose_yolo_backend(components, device)
    if coreml_available():
        key = ("coreml", str(YOLO_PKG), str(ENC_PKG), str(DEC_PKG),
               id(components["sam"]), device)
        if key in _backend_cache:
            return _backend_cache[key]
        sam: SamBackend = CoreMLSamBackend(ENC_PKG, DEC_PKG, device)
        # ram_bytes only makes sense for in-process backends: CoreML memory
        # lives in the ANE driver process, so its RSS delta would be a lie.
        # The torch sam's number is measured at the weight load site
        # (yolosam components build) and stashed on the module itself.
        prompt_sam = TorchSamBackend(components["sam"], device)
        prompt_sam.ram_bytes = getattr(components["sam"], "ram_bytes", None)
        cached = (yolo, sam, prompt_sam)
        _backend_cache[key] = cached
        return cached
    key = ("classic", faster, id(components["yolo"]), id(components["sam"]), device)
    if key in _backend_cache:
        return _backend_cache[key]
    sam_cls = FasterTorchSamBackend if faster else TorchSamBackend
    sam = sam_cls(components["sam"], device)
    sam.ram_bytes = getattr(components["sam"], "ram_bytes", None)
    cached = (yolo, sam, None)
    _backend_cache[key] = cached
    return cached
