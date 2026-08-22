from enum import Enum


class AvailableModels(str, Enum):
    yolosam = "YoloSAM"
    yolomaskrcnn = "YoloMaskRCNN"
    maskrcnn_synthetic = "MaskRCNN-Synthetic"
