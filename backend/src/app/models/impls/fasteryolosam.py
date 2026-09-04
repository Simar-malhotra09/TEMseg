"""FasterYoloSam: YoloSam whose SAM backend carries the measured decode
optimisations (fused running max-logit union, dense-PE cache, SDPA decoder).

The optimisations live in FasterTorchSamBackend (backends/impls/sam) so the
decode loop and pacing hook are backend concerns; this class is now just
backend selection. On platforms where the selected SAM backend is not torch
(e.g. CoreML on arm Macs) the optimisations don't apply and FasterYoloSam
behaves identically to YoloSam — the two models only differ where the torch
decode runs (Windows today).
"""

from typing import Any, Dict

from app.api.live_models import AvailableModels
from app.logutils import get_logger
from app.models.backends.impls.sam.torch_sam import (
    USE_DECODER_SDPA,
    USE_DENSE_PE_CACHE,
    USE_FUSED_UNION,
)
from app.models.base_model import ModelConfig
from app.models.impls.yolosam import YoloSam

logger = get_logger("FasterYoloSAM")
log_batch = get_logger("FasterYoloSAM", sub="Batch")


class FasterYoloSam(YoloSam):
    MODEL_ENUM = AvailableModels.fasteryolosam

    def __init__(
        self,
        config: ModelConfig,
        device: str = "cpu",
        components: Dict[str, Any] | None = None,
    ):
        super().__init__(config, device, components=components, _faster=True)
        logger.info(
            "FasterYoloSam active (fused_union=%s, decoder_sdpa=%s, dense_pe_cache=%s)",
            USE_FUSED_UNION,
            USE_DECODER_SDPA,
            USE_DENSE_PE_CACHE,
        )
