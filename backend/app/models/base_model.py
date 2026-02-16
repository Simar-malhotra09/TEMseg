from abc import ABC, abstractmethod
import cv2 as cv
import numpy as np
from dataclasses import dataclass
from typing import List

'''
A simple way to structure this would be defining x,y,width and height.
But our objects are not uniform in any sense.
What we could do is, return a numpy arr of same h,w as img with each
pixel representing an object id.
'''
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np


@dataclass
class SubModelConfig:
    name: str
    path: str


@dataclass
class ModelConfig:
    name: str
    components: List[SubModelConfig]


@dataclass
class SegmentationResult:
    segmentation_mask: np.ndarray
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


