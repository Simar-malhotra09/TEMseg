from yolosam import YoloSam
from config import nano_config

model = YoloSam(nano_config)

model.get_model_specs()

# img = model.load_image("test.png")
# result = model.segment(img)
