from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Literal

from app.models.helpers.compute_stats import compute_stats_from_instances
import numpy as np

# from app.models.helpers.compute_stats import (
#     compute_avg_circularity,
#     compute_avg_size,
#     compute_coverage,
#     compute_particle_count,
# )

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

class Particle(BaseModel):
    id: int
    # pixel-space
    area_px: float
    perimeter_px: float
    diameter_px: float
    major_axis_px: float
    minor_axis_px: float

    # shape
    circularity: float
    aspect_ratio: float
    shape: str  

    bbox: dict

    # optional real-world units
    area_real: Optional[float] = None
    perimeter_real: Optional[float] = None
    diameter_real: Optional[float] = None
    major_axis_real: Optional[float] = None
    minor_axis_real: Optional[float] = None

class ShapeStats(BaseModel):
    count: int
    fraction: float

class SizeStats(BaseModel):
    area_mean: float
    area_std: float
    area_min: float
    area_max: float
    area_median: float

    diameter_mean: float
    diameter_std: float
    diameter_min: float
    diameter_max: float
    diameter_median: float

    unit: str

class StatsResponse(BaseModel):
    # scale info
    pixel_size: Optional[float]
    pixel_unit: Optional[str]
    unit: str
    has_scale: bool

    # aggregate (compat)
    particle_count: int
    coverage: float
    avg_size: float
    avg_circularity: float
    avg_aspect_ratio: float

    # detailed aggregate
    avg_area_px: float
    avg_diameter_px: float
    avg_area_real: Optional[float] = None
    avg_diameter_real: Optional[float] = None

    # distributions
    size_stats: SizeStats
    shape_distribution: Dict[str, ShapeStats]
    distribution_fits_diameter: Dict = Field(default_factory=dict)
    distribution_fits_area: Dict = Field(default_factory=dict)

    # per-particle
    particles: List[Particle]

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


    def compute_stats(
        self,
        instances: List[Dict[str, Any]],
        mask: np.ndarray,
        pixel_size: float | None = None,
        pixel_unit: str | None = None
    ) -> StatsResponse:   

        stats_results = compute_stats_from_instances(
            instances, mask,
            pixel_size=pixel_size,
            pixel_unit=pixel_unit
        )

        return StatsResponse.model_validate(stats_results)
