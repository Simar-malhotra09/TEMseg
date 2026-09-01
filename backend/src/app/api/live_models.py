from enum import Enum


class AvailableModels(str, Enum):
    yolosam = "YoloSAM"
    fastyolosam = "FastYoloSAM"
    yolomaskrcnn = "YoloMaskRCNN"
    maskrcnn_synthetic = "MaskRCNN-Synthetic"
