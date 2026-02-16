from base_model import SubModelConfig, ModelConfig
from yolosam import YoloSam
from config import nano_config

model = YoloSam(nano_config, device="cpu")

model.get_model_specs()

img = model.load_image("./0032.tif")
result = model.segment(img)
model.plot(img, result.segmentation_mask)
print(result.metadata)
