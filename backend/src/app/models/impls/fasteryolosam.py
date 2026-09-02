"""
FasterYoloSAM optimizations.

The goal is to reduce SAM decoding time and memory usage while keeping
the output identical to the original YoloSAM implementation whenever
possible.

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

Use model_scripts/ to benchmark YoloSAM and
FasterYoloSAM on the same image(s) and compare runtime, masks, and extracted
instance counts.
"""

import os
import time
from contextlib import contextmanager
from typing import Any, Dict, List

import numpy as np
import torch
from segment_anything import SamPredictor
from segment_anything.modeling import transformer as sam_transformer

from app.api.live_models import AvailableModels
from app.logutils import Timer, fmt_duration, get_logger
from app.models.base_model import ModelConfig
from app.models.impls.yolosam import (
    SAMEmbedding,
    YoloSAMSegmentationResult,
    YoloSam,
)

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

    The segment()/segment_batch() overrides reuse the base YOLO + embedding
    flow verbatim; only the mask-decode/union step differs.
    """

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

    # SAM decode optimisations
    def _get_predictor(self) -> SamPredictor:
        if not hasattr(self, "_predictor"):
            self._predictor = SamPredictor(self.components["sam"])
        return self._predictor

    def _sync_device(self) -> None:
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

    def _get_dense_pe(self, sam) -> torch.Tensor:
        """prompt_encoder.get_dense_pe(), cached.

        The dense positional encoding is a CONSTANT tensor since it depends only on
        the prompt-encoder weights and never on the image or prompts, yet base
        SAM recomputes it on every predict_torch() call. Caching
        saves ~17–19ms per chunk per image.
        """
        if self._dense_pe is None or not USE_DENSE_PE_CACHE:
            pe = sam.prompt_encoder.get_dense_pe()
            if USE_DENSE_PE_CACHE:
                self._dense_pe = pe
            return pe
        return self._dense_pe

    def _union_decode(
        self, predictor: SamPredictor, transformed_boxes: torch.Tensor, chunk: int
    ) -> np.ndarray:
        """
        Union of all per-box masks as a uint8 0/1 array at original size.
        """
        sam = predictor.model
        image_pe = self._get_dense_pe(sam)
        n = len(transformed_boxes)

        if not USE_FUSED_UNION:
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
            return union

        running_fullres = None
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
                chunk_max = full.amax(dim=(0, 1), keepdim=True)  # (1,1,H,W)
                running_fullres = (
                    chunk_max
                    if running_fullres is None
                    else torch.maximum(running_fullres, chunk_max)
                )
                # Pacing sync per chunk: without it the driver queues all
                # chunks before any completes, so ~200MB of buffers pile up
                # mid-loop and segment-to-segment timings swing by ±1.5s on
                # MPS. Syncing per chunk lets the allocator reuse the last
                # chunk's buffers exactly like base's .cpu() pacing, and was
                # the fastest decode variant measured
                self._sync_device()

        mask = running_fullres > sam.mask_threshold
        return mask.cpu().numpy().astype("uint8").squeeze()

    # pipeline overrides (YOLO + embedding flow identical to base)
    def segment(
        self,
        image: np.ndarray,
        embedding_cache=None,
        encoder_depth: int = 12,
        **kwargs,
    ):
        logger.info(f"input image shape: {image.shape}, dtype: {image.dtype}")

        predictor = self._get_predictor()

        with Timer(logger, "yolo") as t_yolo:
            with t_yolo.step("predict"):
                results = self.components["yolo"].predict(
                    source=image,
                    conf=0.25,
                    iou=0.5,
                    max_det=4000,
                    verbose=False,
                    device=self.device,
                )
                boxes = results[0].boxes.xyxy

            with t_yolo.step("box_transfer"):
                input_boxes = boxes.to(predictor.device)
                transformed_boxes = predictor.transform.apply_boxes_torch(
                    input_boxes, image.shape[:2]
                )

            t_yolo.field("boxes", len(boxes))

        with Timer(logger, "sam") as t_sam:
            if embedding_cache:
                predictor.features = embedding_cache.features
                predictor.original_size = embedding_cache.original_size
                predictor.input_size = embedding_cache.input_size
                predictor.is_image_set = True
                t_sam.field("encode", "cached")
            else:
                with t_sam.step("encode"):
                    self._encode_image(predictor, image, encoder_depth)
                    t_sam.field("depth", encoder_depth)

            if boxes is None or len(boxes) == 0:
                logger.info(
                    "YOLO found 0 boxes — returning empty mask with SAM "
                    "embedding cached"
                )
                return YoloSAMSegmentationResult(
                    segmentation_mask=np.zeros(image.shape[:2], dtype=np.uint8),
                    metadata={"detections": 0, "sam_embedding_cached": True},
                    model=AvailableModels.fasteryolosam,
                    detection_boxes=np.array([]).reshape(0, 4),
                    embedding=SAMEmbedding(
                        features=predictor.features,
                        original_size=predictor.original_size,
                        input_size=predictor.input_size,
                        encoder_depth=encoder_depth,
                    ),
                )

            box_batch_size = kwargs.get("box_batch_size", 64)

            with t_sam.step("decode"):
                combined_mask = self._union_decode(
                    predictor, transformed_boxes, box_batch_size
                )

            t_sam.field("mask_px", int(np.count_nonzero(combined_mask)))

        # Deliberately NO torch.mps.empty_cache() here. Stock flushes the
        # allocator after every segment (and per chunk), which keeps idle RSS
        # low but forces the next call to re-claim ~750MB from the OS — a
        # 1–2s tax that hides the decode gains between client calls. The
        # caching allocator is not a leak: the ~750MB high-water mark is the
        # steady-state cost of density like this image, and holding it makes
        # every subsequent segment in the client cheaper. segment_batch()
        # still flushes per patch, where long runs genuinely need the memory.
        return YoloSAMSegmentationResult(
            segmentation_mask=combined_mask,
            metadata={"detections": len(boxes), "encoder_depth": encoder_depth},
            model=AvailableModels.fasteryolosam,
            embedding=SAMEmbedding(
                features=predictor.features,
                original_size=predictor.original_size,
                input_size=predictor.input_size,
                encoder_depth=encoder_depth,
            ),
            detection_boxes=boxes.cpu().numpy(),
        )

    def segment_batch(
        self,
        patches: List[np.ndarray],
        offsets: List[tuple],
        img_shape: tuple,
        encoder_depth: int = 12,
    ) -> YoloSAMSegmentationResult:
        h_full, w_full = img_shape
        combined = np.zeros((h_full, w_full), dtype="uint8")

        log_batch.info(f"segment_batch: {len(patches)} patches, output {img_shape}")

        yolo_model = self.components["yolo"]
        predictor = self._get_predictor()

        total_detections = 0
        t_yolo_total = 0.0
        t_sam_total = 0.0
        all_boxes: List[np.ndarray] = []

        with Timer(log_batch, "segment_batch") as t_batch:
            for i, (patch, (x1, y1)) in enumerate(zip(patches, offsets)):
                t_yolo_start = time.perf_counter()
                results = yolo_model.predict(
                    source=patch,
                    conf=0.25,
                    iou=0.5,
                    max_det=4000,
                    verbose=False,
                    device=self.device,
                )
                t_yolo_patch = time.perf_counter() - t_yolo_start
                t_yolo_total += t_yolo_patch

                boxes = results[0].boxes.xyxy

                if boxes is None or len(boxes) == 0:
                    log_batch.debug(f"patch {i}: no detections, skipping")
                    continue

                log_batch.debug(
                    f"patch {i}: {len(boxes)} detections, "
                    f"yolo={fmt_duration(t_yolo_patch)}"
                )

                t_sam_start = time.perf_counter()
                self._encode_image(predictor, patch, encoder_depth)
                input_boxes = boxes.to(predictor.device)
                transformed_boxes = predictor.transform.apply_boxes_torch(
                    input_boxes, patch.shape[:2]
                )
                patch_union = self._union_decode(predictor, transformed_boxes, 64)
                t_sam_patch = time.perf_counter() - t_sam_start
                t_sam_total += t_sam_patch
                log_batch.debug(f"patch {i}: sam={fmt_duration(t_sam_patch)}")

                x2, y2 = x1 + patch.shape[1], y1 + patch.shape[0]
                combined[y1:y2, x1:x2] = np.maximum(
                    combined[y1:y2, x1:x2], (patch_union > 0).astype("uint8") * 255
                )

                boxes_np = boxes.cpu().numpy()
                boxes_np[:, [0, 2]] += x1
                boxes_np[:, [1, 3]] += y1
                all_boxes.append(boxes_np)

                total_detections += len(boxes)
                self._empty_device_cache()

            t_batch.field("yolo", fmt_duration(t_yolo_total))
            t_batch.field("sam", fmt_duration(t_sam_total))
            t_batch.field("detections", total_detections)

        detection_boxes = (
            np.vstack(all_boxes) if all_boxes else np.array([]).reshape(0, 4)
        )

        return YoloSAMSegmentationResult(
            segmentation_mask=combined,
            metadata={
                "detections": total_detections,
                "yolo_time": round(t_yolo_total, 3),
                "sam_time": round(t_sam_total, 3),
                "encoder_depth": encoder_depth,
            },
            model=AvailableModels.fasteryolosam,
            detection_boxes=detection_boxes,
        )
