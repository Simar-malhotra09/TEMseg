"""SAM on CoreML: encoder + B=16 decoder .mlpackages with numpy/MPS glue.

Decoder is static-batch B=16 (see model_scripts/coreml_export); partial
chunks are duplicate-padded, which is a no-op under the max-logit union.
Postprocess (crop 1024^2 to prepad + bilinear resize to original) runs on
MPS via torch interpolate — production's kernel, bit-identical — with the
per-chunk max/argmax reduce done pairwise elementwise (MPS dim-0 reductions
and numpy axis-0 argmax both take slow kernels; elementwise ops are exact
and fast). Point prompts unsupported.
"""

import time

import numpy as np
import torch
from PIL import Image

from app.logutils import get_logger
from ...base import SamBackend, SamEmbedding

logger = get_logger("CoreMLSam")

IMG_SIZE = 1024
B = 16
PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
PIXEL_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def preprocess_1024(image_rgb: np.ndarray) -> np.ndarray:
    """resize-longest-1024, mean/std norm, zero-pad -> (1,3,1024,1024) fp32."""
    h, w = image_rgb.shape[:2]
    scale = IMG_SIZE / max(h, w)
    new_w = int(w * scale + 0.5)
    new_h = int(h * scale + 0.5)
    rsz = np.array(
        Image.fromarray(image_rgb).resize((new_w, new_h), Image.BILINEAR)
    ).astype(np.float32)
    x = (rsz - PIXEL_MEAN) / PIXEL_STD
    x = x.transpose(2, 0, 1)[None]
    x = np.pad(x, ((0, 0), (0, 0), (0, IMG_SIZE - new_h), (0, IMG_SIZE - new_w)))
    return np.ascontiguousarray(x, dtype=np.float32)


class CoreMLSamBackend(SamBackend):
    def __init__(self, enc_pkg, dec_pkg, device: str):
        import coremltools as ct

        t0 = time.perf_counter()
        self._enc = ct.models.MLModel(str(enc_pkg))
        self._enc.predict({"image": np.zeros((1, 3, 1024, 1024), np.float32)})
        logger.info(f"coreml sam encoder loaded+compiled in {time.perf_counter() - t0:.1f}s")
        t0 = time.perf_counter()
        self._dec = ct.models.MLModel(str(dec_pkg))
        self._dec.predict(
            {
                "image_embeddings": np.zeros((1, 256, 64, 64), np.float32),
                "boxes": np.zeros((B, 4), np.float32),
            }
        )
        logger.info(f"coreml sam decoder loaded+compiled in {time.perf_counter() - t0:.1f}s")
        self._device = torch.device(device)

    def encode(self, image_rgb: np.ndarray, encoder_depth: int = 12) -> SamEmbedding:
        x = preprocess_1024(image_rgb)
        emb = self._enc.predict({"image": x})["image_embeddings"]
        return SamEmbedding(
            features=np.ascontiguousarray(emb, dtype=np.float32),
            original_size=image_rgb.shape[:2],
            input_size=(IMG_SIZE, IMG_SIZE),
            encoder_depth=encoder_depth,
        )

    def decode_union(
        self, emb: SamEmbedding, boxes_xyxy: np.ndarray, box_batch: int = B
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w = emb.original_size
        scale = IMG_SIZE / max(h, w)
        nw, nh = int(w * scale + 0.5), int(h * scale + 0.5)
        boxes_t = boxes_xyxy.astype(np.float64).copy()
        boxes_t[:, [0, 2]] *= nw / w
        boxes_t[:, [1, 3]] *= nh / h

        running_max: torch.Tensor | None = None
        running_owner: torch.Tensor | None = None
        for i in range(0, len(boxes_t), B):
            chunk = boxes_t[i : i + B].astype(np.float32)
            n = len(chunk)
            reps = (B + n - 1) // n
            padded = np.tile(chunk, (reps, 1))[:B].copy()
            masks = self._dec.predict(
                {"image_embeddings": emb.features, "boxes": padded}
            )["masks"]  # (B,1,1024,1024) logits
            groups = []
            for j in range(0, B, 8):
                group = masks[j : j + 8, 0, :nh, :nw]
                gt = torch.from_numpy(
                    np.ascontiguousarray(group, dtype=np.float32)
                ).unsqueeze(1).to(self._device)
                groups.append(
                    torch.nn.functional.interpolate(
                        gt, size=(h, w), mode="bilinear", align_corners=False
                    ).squeeze(1)
                )
            chunk_logits = torch.cat(groups, 0)
            if running_max is None:
                running_max = chunk_logits[0].clone()
                running_owner = torch.full_like(
                    running_max, i + 1, dtype=torch.float32
                )
                start = 1
            else:
                start = 0
            for j in range(start, chunk_logits.shape[0]):
                m = chunk_logits[j]
                upd = m > running_max
                running_max = torch.where(upd, m, running_max)
                running_owner = torch.where(upd, float(i + j + 1), running_owner)

        union_t = running_max > 0.0
        owner = torch.where(union_t, running_owner, torch.zeros_like(running_owner))
        return (
            union_t.cpu().numpy().astype("uint8"),
            owner.cpu().numpy().astype(np.uint16),
        )
