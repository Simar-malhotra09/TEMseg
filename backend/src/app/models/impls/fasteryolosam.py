"""
FasterYoloSAM optimizations.

The goal is to reduce SAM decoding time and memory usage while keeping
the output identical to the original YoloSAM implementation whenever
possible. Since the per-box-instance-identity change, both models share the
chunked max-logit decode loop in YoloSam._decode_union_with_ownership;
this subclass only hooks in the measured decode optimisations below.

1. Fused mask union

   The original implementation copies every chunk of masks from MPS to
   the CPU and combines them there. This creates roughly 0.5 GB of
   intermediate CPU transfers.

   FasterYoloSAM instead upsamples each chunk on the device and keeps a
   running pixel-wise maximum at full resolution. Only the final result
   is copied to the CPU and thresholded.

   This produces the same binary mask as the original implementation:
   the original takes the union of individually thresholded masks,
   while FasterYoloSAM takes the maximum logit first and thresholds once.
   Since thresholding at zero is monotonic, these produce the same set
   of pixels.

   (Since the ownership change this fused form is also what the base
   YoloSam uses, because the running max is where the owner labels are
   computed.)


2. Fused attention in the mask decoder

   The mask decoder uses PyTorch's
   scaled_dot_product_attention() instead of manually computing
   attention with q @ k.T, softmax, and @ v.

   See https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html for more information.

   The change is scoped to FasterYoloSAM and restored afterward, so a
   normal YoloSAM running in the same process keeps its original
   attention implementation.

   Mask decode time:

   6.82s -> 6.24s

   The improvement seems to be modest since our token length is pretty small
   shouldn't have any side effects.

   The fused attention kernel can produce slightly different floating-point
   rounding, which can change pixels whose logits are extremely close
   to the threshold.

   Set TEMSEG_FASTSAM_DECODER_SDPA=0 to disable this optimization and
   restore bit-exact output.

3. Cache the dense positional encoding

   prompt_encoder.get_dense_pe() always returns the same tensor for a
   given model. It depends only on the model weights, not on the image
   or prompts.

   The original implementation recomputes it for every prediction.
   FasterYoloSAM caches it instead.

   This saves about 17-19 ms per image.

Optimizations tested and rejected:

* ViT encoder SDPA:
  Slower on MPS (1.96s -> 2.09s).
  Run: 20260831_034631.

* FP16/BF16 autocast:
  Slower (1.96s -> 2.22s) or provided no meaningful improvement.
  BF16 weight casting also slightly worsened output parity.
  Runs: 20260831_034154 and 20260831_034447.

* Low-resolution mask union:
  Taking the maximum at 256x256 and upsampling once at the end is not
  equivalent to upsampling each mask and then taking the maximum.

  On the dense reference image this produced about 21,000 extra fringe
  pixels (~10%) and an IoU of 0.9101. More importantly, those extra
  pixels caused neighboring particles to merge during instance
  extraction.

  Pixel-level IoU is therefore not enough here; the extracted instances
  also need to match. The full-resolution union preserves exact parity
  while still providing most of the memory and transfer improvements.

  The remaining cost is the per-chunk upsampling, which takes about
  0.26s total.

Set TEMSEG_FASTSAM_FUSED_UNION=0 for the legacy per-chunk CPU union (debug;
in that mode per-box ownership labels are not produced and instance
extraction falls back to connected components).

Use model_scripts/ to benchmark YoloSAM and
FasterYoloSAM on the same image(s) and compare runtime, masks, and extracted
instance counts.
"""

import os
from contextlib import contextmanager
from typing import Any, Dict

import numpy as np
import torch
from segment_anything import SamPredictor
from segment_anything.modeling import transformer as sam_transformer

from app.api.live_models import AvailableModels
from app.logutils import get_logger
from app.models.base_model import ModelConfig
from app.models.impls.yolosam import YoloSam

logger = get_logger("FasterYoloSAM")
log_batch = get_logger("FasterYoloSAM", sub="Batch")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


USE_FUSED_UNION = _env_flag("TEMSEG_FASTSAM_FUSED_UNION", True)
USE_DECODER_SDPA = _env_flag("TEMSEG_FASTSAM_DECODER_SDPA", True)
USE_DENSE_PE_CACHE = _env_flag("TEMSEG_FASTSAM_DENSE_PE_CACHE", True)


def _decoder_sdpa_attention_forward(self, q, k, v):
    """transformer.Attention.forward with the manual softmax fused into SDPA.

    Replace manual softmax(q @ k^T / sqrt(head_dim)) @ v with
    F.scaled_dot_product_attention which uses kernels. Applies ONLY to the mask
    decoder's TwoWayTransformer since the ViT encoder's Attention is a different
    class (relative-position bias) and measured SLOWER with SDPA on MPS.
    """
    q = self.q_proj(q)
    k = self.k_proj(k)
    v = self.v_proj(v)

    q = self._separate_heads(q, self.num_heads)
    k = self._separate_heads(k, self.num_heads)
    v = self._separate_heads(v, self.num_heads)

    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    out = self._recombine_heads(out)
    return self.out_proj(out)


@contextmanager
def _decoder_sdpa_scope():
    """
    Patch sam_transformer.Attention.forward to SDPA for the decode only.
    """
    original = sam_transformer.Attention.forward
    sam_transformer.Attention.forward = _decoder_sdpa_attention_forward
    try:
        yield
    finally:
        sam_transformer.Attention.forward = original


class FasterYoloSam(YoloSam):
    """
    YoloSam with the measured SAM-decode optimisations. See module docstring.

    Segment flow (YOLO, embeddings, stitching) is inherited from YoloSam;
    only the decode loop and the per-chunk pacing hook differ.
    """

    MODEL_ENUM = AvailableModels.fasteryolosam

    def __init__(
        self,
        config: ModelConfig,
        device: str = "cpu",
        components: Dict[str, Any] | None = None,
    ):
        self._dense_pe: torch.Tensor | None = None
        super().__init__(config, device, components=components)
        logger.info(
            "FasterYoloSam active (fused_union=%s, decoder_sdpa=%s, dense_pe_cache=%s)",
            USE_FUSED_UNION,
            USE_DECODER_SDPA,
            USE_DENSE_PE_CACHE,
        )

    def _sync_device(self) -> None:
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

    # Deliberately NOT empty_cache (unlike the base hook). Flushing the
    # allocator after every chunk keeps idle RSS low but forces the next call
    # to re-claim ~750MB from the OS — a 1–2s tax that hides the decode gains
    # between client calls. The caching allocator is not a leak: the high-water
    # mark is the steady-state cost of density like this image, and holding it
    # makes every subsequent segment in the client cheaper. The batch pipeline
    # still flushes per patch, where long runs genuinely need the memory.
    def _after_decode_chunk(self) -> None:
        # Pacing sync per chunk: without it the driver queues all chunks
        # before any completes, so ~200MB of buffers pile up mid-loop and
        # segment-to-segment timings swing by ±1.5s on MPS. Syncing per chunk
        # lets the allocator reuse the last chunk's buffers exactly like
        # base's .cpu() pacing, and was the fastest decode variant measured.
        self._sync_device()

    def _get_dense_pe(self, sam) -> torch.Tensor:
        """prompt_encoder.get_dense_pe(), cached.

        The dense positional encoding is a CONSTANT tensor since it depends only on
        the prompt-encoder weights and never on the image or prompts, yet base
        SAM recomputes it on every prediction. Caching
        saves ~17–19ms per chunk per image.
        """
        if self._dense_pe is None or not USE_DENSE_PE_CACHE:
            pe = sam.prompt_encoder.get_dense_pe()
            if USE_DENSE_PE_CACHE:
                self._dense_pe = pe
            return pe
        return self._dense_pe

    def _decode_union_with_ownership(
        self,
        predictor: SamPredictor,
        transformed_boxes: torch.Tensor,
        chunk: int,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        sam = predictor.model
        n = len(transformed_boxes)

        if not USE_FUSED_UNION:
            # Legacy debug path: per-chunk CPU union of thresholded masks.
            # No running max-logit, so ownership labels are unavailable and
            # callers fall back to connected-component extraction.
            union = np.zeros(predictor.original_size, dtype="uint8")
            with torch.no_grad():
                for i in range(0, n, chunk):
                    masks, _, _ = predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes[i : i + chunk],
                        multimask_output=False,
                    )
                    batch = masks.cpu().numpy().astype("uint8")
                    union = np.maximum(union, batch[:, 0].max(axis=0))
            return union, None

        image_pe = self._get_dense_pe(sam)
        running_max: torch.Tensor | None = None
        running_owner: torch.Tensor | None = None
        with torch.no_grad():
            for i in range(0, n, chunk):
                boxes_chunk = transformed_boxes[i : i + chunk]
                sparse, dense = sam.prompt_encoder(
                    points=None, boxes=boxes_chunk, masks=None
                )
                if USE_DECODER_SDPA:
                    with _decoder_sdpa_scope():
                        low_res, _ = sam.mask_decoder(
                            image_embeddings=predictor.features,
                            image_pe=image_pe,
                            sparse_prompt_embeddings=sparse,
                            dense_prompt_embeddings=dense,
                            multimask_output=False,
                        )
                else:
                    low_res, _ = sam.mask_decoder(
                        image_embeddings=predictor.features,
                        image_pe=image_pe,
                        sparse_prompt_embeddings=sparse,
                        dense_prompt_embeddings=dense,
                        multimask_output=False,
                    )
                full = sam.postprocess_masks(
                    low_res, predictor.input_size, predictor.original_size
                )
                chunk_max, chunk_owner = full.squeeze(1).max(dim=0)
                if running_max is None:
                    running_max = chunk_max
                    running_owner = chunk_owner + i + 1
                else:
                    update = chunk_max > running_max
                    running_max = torch.maximum(running_max, chunk_max)
                    running_owner = torch.where(
                        update, chunk_owner + i + 1, running_owner
                    )
                self._after_decode_chunk()

        union_t = running_max > sam.mask_threshold
        owner_t = torch.where(union_t, running_owner, torch.zeros_like(running_owner))
        return (
            union_t.cpu().numpy().astype("uint8"),
            owner_t.cpu().numpy().astype(np.uint16),
        )
