"""Shared contracts for model backends.

The app (routers, model impls) talks only to these interfaces. Concrete
inference stacks live under impls/ and are chosen by selection.py.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class SamEmbedding:
    """SAM image embedding. numpy-first so any backend can consume it."""

    features: np.ndarray  # (1, 256, 64, 64) float32
    original_size: tuple[int, int]
    input_size: tuple[int, int]
    encoder_depth: int = 12


class YoloBackend(ABC):
    """Object detection: image in, boxes out."""

    @abstractmethod
    def detect(self, image_rgb: np.ndarray) -> np.ndarray:
        """(H,W,3) uint8 RGB -> boxes_xyxy (N,4) float32 in image space."""


class SamBackend(ABC):
    """SAM encode + boxes-only union decode (+ optional point prompts)."""

    @abstractmethod
    def encode(self, image_rgb: np.ndarray, encoder_depth: int = 12) -> SamEmbedding: ...

    @abstractmethod
    def decode_union(
        self, emb: SamEmbedding, boxes_xyxy: np.ndarray, box_batch: int = 64
    ) -> tuple[np.ndarray, np.ndarray]:
        """(union uint8, owner_labels uint16) at original-image size."""

    def predict_prompts(self, prompts: list[dict]) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} does not support point prompts")
