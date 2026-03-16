from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np

from app.models.helpers.compute_stats import (
    compute_avg_circularity,
    compute_avg_size,
    compute_coverage,
    compute_particle_count,
)

""" Stores individual models"""


@dataclass
class SubModelConfig:
    name: str
    path: str | Path


""" 
Stores multiple Sub models 
example:
    The nano pipeline uses
    Yolo for object detection 
    SAM for segmentation
"""


@dataclass
class ModelConfig:
    name: str
    components: List[SubModelConfig]


@dataclass
class SegmentationResult:
    segmentation_mask: np.ndarray
    model: str
    metadata: Dict[str, Any] | None = None


"""
    List all available stats
"""


class StatType(str, Enum):
    PARTICLE_COUNT = "particle_count"
    AVG_SIZE = "avg_size"
    AVG_CIRCULARITY = "avg_circularity"
    COVERAGE = "coverage"


@dataclass
class StatsConfig:
    enabled: Set[StatType]


@dataclass
class StatsResult:
    values: dict[StatType, float]


class Model(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.components = self._load_components()

    @abstractmethod
    def _load_components(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def load_image(self, image_path: Path) -> np.ndarray:
        pass

    @abstractmethod
    def segment(self, image: np.ndarray, **kwargs) -> SegmentationResult:
        pass

    def get_model_specs(self) -> None:
        print(f"\nModel: {self.config.name}")
        print("-" * 40)
        for comp in self.config.components:
            print(f"Component: {comp.name}")
            print(f"Path:      {comp.path}")
            print("-" * 40)

    def compute_stats(self, mask, config: StatsConfig) -> StatsResult:
        results = {}

        if StatType.PARTICLE_COUNT in config.enabled:
            results[StatType.PARTICLE_COUNT] = compute_particle_count(mask)

        if StatType.AVG_SIZE in config.enabled:
            results[StatType.AVG_SIZE] = compute_avg_size(mask)

        if StatType.AVG_CIRCULARITY in config.enabled:
            results[StatType.AVG_CIRCULARITY] = compute_avg_circularity(mask)

        if StatType.COVERAGE in config.enabled:
            results[StatType.COVERAGE] = compute_coverage(mask)

        return StatsResult(values=results)
