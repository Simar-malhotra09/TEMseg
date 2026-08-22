from pathlib import Path
from app.models.base_model import StatType, StatsConfig
from app.models.impls.yolosam import YoloSam
from app.models.impls.yolomaskrcnn import YoloMaskRCNN
from app.models.helpers.config import nano_config, yolomaskrcnn_config
from app.api.live_models import AvailableModels
from app.models.helpers.compute_stats import (
    compute_particle_count,
    compute_avg_size,
    compute_avg_circularity,
    compute_coverage,
)
import torch

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"


stats_config = StatsConfig(
    enabled={
        StatType.PARTICLE_COUNT,
        StatType.AVG_SIZE,
        StatType.AVG_CIRCULARITY,
        StatType.COVERAGE,
    }
)


def compute_stats(mask, config: StatsConfig):
    results = {}

    if StatType.PARTICLE_COUNT in config.enabled:
        results["particle_count"] = compute_particle_count(mask)

    if StatType.AVG_SIZE in config.enabled:
        results["avg_size"] = compute_avg_size(mask)

    if StatType.AVG_CIRCULARITY in config.enabled:
        results["avg_circularity"] = compute_avg_circularity(mask)

    if StatType.COVERAGE in config.enabled:
        results["coverage"] = compute_coverage(mask)

    return results


model1 = YoloMaskRCNN(yolomaskrcnn_config, AvailableModels.yolomaskrcnn, device="cpu")
print(torch.cuda.is_available())
model1.get_model_specs()

img1 = model1.load_image(DATA_DIR / "0032.tif")
result1 = model1.segment(img1)

model2 = YoloSam(nano_config, device="cpu")
print(torch.cuda.is_available())
model2.get_model_specs()

img2 = model2.load_image(DATA_DIR / "0032.tif")
result2 = model2.segment(img2)
if isinstance(result2.segmentation_mask, torch.Tensor):
    result2.segmentation_mask = result2.segmentation_mask.detach().cpu().numpy()

if result2.segmentation_mask.ndim == 3 and result2.segmentation_mask.shape[0] == 1:
    result2.segmentation_mask = result2.segmentation_mask.squeeze(0)

stats1 = compute_stats(result1.segmentation_mask, stats_config)
stats2 = compute_stats(result2.segmentation_mask, stats_config)

print("YoloMaskRCNN stats:", stats1)
print("YoloSam stats:", stats2)
