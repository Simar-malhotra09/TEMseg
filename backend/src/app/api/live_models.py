from enum import Enum


class AvailableModels(str, Enum):
    yolosam = "YoloSAM"
    maskrcnn = "MaskRCNN"
    maskrcnn_synthetic = "MaskRCNN-Synthetic"
