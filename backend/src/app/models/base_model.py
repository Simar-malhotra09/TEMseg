from abc import ABC, abstractmethod
import cv2 as cv
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any

''' Stores individual models'''
@dataclass
class SubModelConfig:
    name: str
    path: str

''' 
Stores multiple Sub models 
example:
    The nano pipeline uses
    Yolo for object detection 
    SAM for segmentation
'''
@dataclass
class ModelConfig:
    name: str
    components: List[SubModelConfig]


@dataclass
class SegmentationResult:
    segmentation_mask: np.ndarray
    model: str
    metadata: Dict[str, Any] | None = None


class Model(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.components = self._load_components()

    @abstractmethod
    def _load_components(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def load_image(self, image_path: str) -> np.ndarray:
        pass

    @abstractmethod
    def segment(self, image: np.ndarray) -> SegmentationResult:
        pass

    def get_model_specs(self) -> None:
        print(f"\nModel: {self.config.name}")
        print("-" * 40)
        for comp in self.config.components:
            print(f"Component: {comp.name}")
            print(f"Path:      {comp.path}")
            print("-" * 40)


