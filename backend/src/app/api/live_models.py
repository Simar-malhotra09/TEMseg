from enum import Enum


class AvailableModels(str, Enum):
    yolosam = "YoloSAM"
    fasteryolosam = "FasterYoloSAM"
    mobile_sam = "MobileSAM"
    microsam_em = "micro-SAM EM (ViT-B)"
    yolomaskrcnn = "YoloMaskRCNN"
    maskrcnn_synthetic = "MaskRCNN-Synthetic"
