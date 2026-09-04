"""Platform/capability probe: pick the inference backends for yolosam.

arm Macs with the CoreML .mlpackages present get CoreML yolo+sam with a
torch SAM kept alongside for point-prompt endpoints. Everything else gets
the classic onnx+pth torch stack. TEMSEG_COREML=0 forces classic.
"""

import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict

from .base import SamBackend, YoloBackend
from .impls.sam.coreml_sam import CoreMLSamBackend
from .impls.sam.torch_sam import FasterTorchSamBackend, TorchSamBackend
from .impls.yolo.coreml_yolo import CoreMLYoloBackend
from .impls.yolo.ort_yolo import OrtYoloBackend

WEIGHTS_DIR = Path.home() / "Library" / "Application Support" / "TEMseg" / "weights"
YOLO_PKG = WEIGHTS_DIR / "best12x.mlpackage"
ENC_PKG = WEIGHTS_DIR / "sam_encoder_vit_b_d12_fp32.mlpackage"
DEC_PKG = WEIGHTS_DIR / "sam_decoder_head64_fp32.mlpackage"


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
    endpoints; it may be the same object as sam on the classic path."""
    if coreml_available():
        yolo: YoloBackend = CoreMLYoloBackend(YOLO_PKG)
        sam: SamBackend = CoreMLSamBackend(ENC_PKG, DEC_PKG, device)
        prompt_sam = TorchSamBackend(components["sam"], device)
        return yolo, sam, prompt_sam
    yolo = OrtYoloBackend(components["yolo"], device)
    sam_cls = FasterTorchSamBackend if faster else TorchSamBackend
    sam = sam_cls(components["sam"], device)
    return yolo, sam, None
