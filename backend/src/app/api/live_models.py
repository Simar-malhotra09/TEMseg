from enum import Enum


class AvailableModels(str, Enum):
    yolosam = "YoloSAM"
    fasteryolosam = "FasterYoloSAM"
    yolomaskrcnn = "YoloMaskRCNN"
    maskrcnn_synthetic = "MaskRCNN-Synthetic"
