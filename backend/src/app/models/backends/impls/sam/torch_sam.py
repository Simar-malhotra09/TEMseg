"""SAM on torch (MPS/CUDA/CPU): production predictor + chunked union decode.

FasterTorchSamBackend carries the FasterYoloSam decode optimizations
(fused running max-logit union, dense-PE cache, SDPA decoder) so the model
class itself shrinks to backend selection.
"""

import os
from contextlib import contextmanager
from typing import List

import numpy as np
import torch
from segment_anything import SamPredictor
from segment_anything.modeling import transformer as sam_transformer

from ...base import SamBackend, SamEmbedding


def encode_block_subset(
    sam: torch.nn.Module, image_tensor: torch.Tensor, block_indices: List[int]
) -> torch.Tensor:
    """Preprocessed tensor -> SAM neck over an explicit ordered block subset.

    Same forward path as ImageEncoderViT.forward but runs `block_indices`
    (0-based) instead of all 12 transformer blocks. Output shape equals the
    full-depth path so the decoder is unaffected.
    """
    enc = sam.image_encoder
    h = enc.patch_embed(image_tensor)
    if enc.pos_embed is not None:
        h = h + enc.pos_embed
    for i in block_indices:
        h = enc.blocks[i](h)
    return enc.neck(h.permute(0, 3, 1, 2))


class TorchSamBackend(SamBackend):
    def __init__(self, sam: torch.nn.Module, device: str):
        self._model = sam
        self._device = device
        self._predictor = SamPredictor(sam)

    @property
    def predictor(self) -> SamPredictor:
        return self._predictor

    def _sync_device(self) -> None:
        if self._device == "cuda":
            torch.cuda.synchronize()
        elif self._device == "mps":
            torch.mps.synchronize()

    def _after_decode_chunk(self) -> None:
        if self._device == "cuda":
            torch.cuda.empty_cache()
        elif self._device == "mps":
            torch.mps.empty_cache()

    def encode(self, image_rgb: np.ndarray, encoder_depth: int = 12) -> SamEmbedding:
        if encoder_depth == 12:
            self._predictor.set_image(image_rgb)
            return self._current_embedding(encoder_depth)
        sam = self._model
        n_blocks = len(sam.image_encoder.blocks)
        if not 1 <= encoder_depth < n_blocks:
            raise ValueError(
                f"encoder_depth must be in 1..{n_blocks - 1} (or 12 for full),"
                f" is {encoder_depth}"
            )
        self._predictor.reset_image()
        self._predictor.original_size = image_rgb.shape[:2]
        with torch.no_grad():
            input_image = self._predictor.transform.apply_image(image_rgb)
            input_image_torch = torch.as_tensor(input_image, device=self._device)
            input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None]
            self._predictor.input_size = tuple(input_image_torch.shape[-2:])
            self._predictor.features = encode_block_subset(
                sam, sam.preprocess(input_image_torch), list(range(encoder_depth))
            )
        self._predictor.is_image_set = True
        return self._current_embedding(encoder_depth)

    def _current_embedding(self, encoder_depth: int) -> SamEmbedding:
        return SamEmbedding(
            features=self._predictor.features.detach().cpu().numpy(),
            original_size=self._predictor.original_size,
            input_size=self._predictor.input_size,
            encoder_depth=encoder_depth,
        )

    def sync_embedding(self, emb: SamEmbedding) -> None:
        """Point the predictor at an embedding produced elsewhere (e.g. the
        CoreML encoder) so prompt endpoints and decode share one state."""
        self._predictor.features = torch.from_numpy(
            np.ascontiguousarray(emb.features, dtype=np.float32)
        ).to(self._device)
        self._predictor.original_size = emb.original_size
        self._predictor.input_size = emb.input_size
        self._predictor.is_image_set = True

    def decode_union(
        self, emb: SamEmbedding, boxes_xyxy: np.ndarray, box_batch: int = 64
    ) -> tuple[np.ndarray, np.ndarray]:
        self.sync_embedding(emb)
        predictor = self._predictor
        boxes = torch.from_numpy(boxes_xyxy.astype(np.float32)).to(self._device)
        transformed = predictor.transform.apply_boxes_torch(boxes, emb.original_size)
        return self._decode(predictor, transformed, box_batch)

    def _decode(
        self, predictor: SamPredictor, transformed_boxes: torch.Tensor, chunk: int
    ) -> tuple[np.ndarray, np.ndarray]:
        sam = self._model
        image_pe = sam.prompt_encoder.get_dense_pe()
        n = len(transformed_boxes)
        running_max: torch.Tensor | None = None
        running_owner: torch.Tensor | None = None
        with torch.no_grad():
            for i in range(0, n, chunk):
                sparse, dense = sam.prompt_encoder(
                    points=None, boxes=transformed_boxes[i : i + chunk], masks=None
                )
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

    def predict_prompts(self, prompts: list[dict]) -> np.ndarray:
        predictor = self._predictor
        if not predictor.is_image_set:
            raise RuntimeError("SAM predictor has no image — call segment() first")
        h, w = predictor.original_size
        combined = np.zeros((h, w), dtype=np.uint8)
        for prompt in prompts:
            cx, cy = prompt["point"]
            x1, y1, x2, y2 = prompt["bbox"]
            point_coords = torch.tensor(
                [[[cx, cy]]], dtype=torch.float32, device=self._device
            )
            point_labels = torch.ones((1, 1), dtype=torch.int, device=self._device)
            box_tensor = torch.tensor(
                [[x1, y1, x2, y2]], dtype=torch.float32, device=self._device
            )
            transformed_box = predictor.transform.apply_boxes_torch(
                box_tensor, predictor.original_size
            )
            masks, _, _ = predictor.predict_torch(
                point_coords=point_coords,
                point_labels=point_labels,
                boxes=transformed_box,
                multimask_output=False,
            )
            mask_np = masks.cpu().numpy().astype(np.uint8).squeeze()
            combined = np.maximum(combined, mask_np)
        return combined


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


USE_FUSED_UNION = _env_flag("TEMSEG_FASTSAM_FUSED_UNION", True)
USE_DECODER_SDPA = _env_flag("TEMSEG_FASTSAM_DECODER_SDPA", True)
USE_DENSE_PE_CACHE = _env_flag("TEMSEG_FASTSAM_DENSE_PE_CACHE", True)


def _decoder_sdpa_attention_forward(self, q, k, v):
    """manual-softmax attention replaced with SDPA kernels (decode only)."""
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
    original = sam_transformer.Attention.forward
    sam_transformer.Attention.forward = _decoder_sdpa_attention_forward
    try:
        yield
    finally:
        sam_transformer.Attention.forward = original


class FasterTorchSamBackend(TorchSamBackend):
    """TorchSamBackend with the FasterYoloSam decode optimizations."""

    def __init__(self, sam: torch.nn.Module, device: str):
        super().__init__(sam, device)
        self._dense_pe: torch.Tensor | None = None

    def _after_decode_chunk(self) -> None:
        # Deliberately no empty_cache: flushing per chunk forces the next call
        # to re-claim ~750MB from the OS. Sync paces the chunk queue instead.
        self._sync_device()

    def _get_dense_pe(self) -> torch.Tensor:
        if self._dense_pe is None or not USE_DENSE_PE_CACHE:
            pe = self._model.prompt_encoder.get_dense_pe()
            if USE_DENSE_PE_CACHE:
                self._dense_pe = pe
            return pe
        return self._dense_pe

    def _decode(
        self, predictor: SamPredictor, transformed_boxes: torch.Tensor, chunk: int
    ) -> tuple[np.ndarray, np.ndarray | None]:
        sam = self._model
        n = len(transformed_boxes)

        if not USE_FUSED_UNION:
            # legacy debug path: per-chunk CPU union, no ownership labels
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

        image_pe = self._get_dense_pe()
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
