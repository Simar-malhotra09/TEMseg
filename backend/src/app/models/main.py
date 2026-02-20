from pathlib import Path
from app.models.base_model import SubModelConfig, ModelConfig
from app.models.impls.yolosam import YoloSam
from app.models.impls.maskrcnn import MaskRCNN
from app.models.helpers.config import nano_config, house_config
import torch
import matplotlib.pyplot as plt
#backend/src
BASE_DIR = Path(__file__).resolve().parents[2]  
DATA_DIR = BASE_DIR / "data"



# model = MaskRCNN(house_config, device="cpu")
# print(torch.cuda.is_available())
# model.get_model_specs()
# img = model.load_image(DATA_DIR / "0032.tif")
# result = model.segment(img)
# model.plot(img, result.segmentation_mask)
# print(result.metadata)
#
# model = YoloSam(nano_config, device="cpu")
#
# print(torch.cuda.is_available())
# model.get_model_specs()
# img = model.load_image(DATA_DIR / "0032.tif")
# result = model.segment(img)
# model.plot(img, result.segmentation_mask)
# print(result.metadata)

#
model1 = MaskRCNN(house_config, device="cpu")
print(torch.cuda.is_available())
model1.get_model_specs()

img1= model1.load_image(DATA_DIR / "0032.tif")
result1 = model1.segment(img1)

model2 = YoloSam(nano_config, device="cpu")
print(torch.cuda.is_available())
model2.get_model_specs()

img2= model2.load_image(DATA_DIR / "0032.tif")
result2 = model2.segment(img2)
if isinstance(result2.segmentation_mask, torch.Tensor):
    result2.segmentation_mask = result2.segmentation_mask.detach().cpu().numpy()

if result2.segmentation_mask.ndim == 3 and result2.segmentation_mask.shape[0] == 1:
    result2.segmentation_mask = result2.segmentation_mask.squeeze(0)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(img1)
axes[0].imshow(result1.segmentation_mask, alpha=0.5)
axes[0].set_title("MaskRCNN")
axes[0].axis("off")

axes[1].imshow(img2)
axes[1].imshow(result2.segmentation_mask, alpha=0.5)
axes[1].set_title("YoloSam")
axes[1].axis("off")

plt.tight_layout()
plt.show()

print(result1.metadata)
print(result2.metadata)

