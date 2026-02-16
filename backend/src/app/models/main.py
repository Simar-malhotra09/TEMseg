from pathlib import Path
from app.models.base_model import SubModelConfig, ModelConfig
from app.models.yolosam import YoloSam
from app.models.config import nano_config


#backend/src
BASE_DIR = Path(__file__).resolve().parents[2]  
DATA_DIR = BASE_DIR / "data"


model = YoloSam(nano_config, device="cpu")

model.get_model_specs()

img = model.load_image(DATA_DIR / "0032.tif")
result = model.segment(img)
model.plot(img, result.segmentation_mask)
print(result.metadata)
