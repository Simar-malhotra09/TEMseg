from app.models.base_model import SubModelConfig, ModelConfig
from app.models.helpers.settings import settings


yolo_config = SubModelConfig("yolo", settings.YOLO_MODEL_PATH)
sam_config = SubModelConfig("sam", settings.SAM_MODEL_PATH)
mobile_sam_config = SubModelConfig("mobile_sam", settings.MOBILE_SAM_MODEL_PATH)
microsam_em_config = SubModelConfig("microsam_em", settings.MICROSAM_EM_MODEL_PATH)
maskrcnn_config = SubModelConfig("maskrcnn", settings.MASKRCNN_MODEL_PATH)
maskrcnn_synthetic_config = SubModelConfig(
    "maskrcnn", settings.MASKRCNN_SYNTHETIC_MODEL_PATH
)

nano_config = ModelConfig(name="nano_pipeline", components=[yolo_config, sam_config])
microsam_em_pipeline_config = ModelConfig(
    name="microsam_em_pipeline", components=[yolo_config, microsam_em_config]
)
mobile_sam_pipeline_config = ModelConfig(
    name="mobile_sam_pipeline", components=[yolo_config, mobile_sam_config]
)

yolomaskrcnn_config = ModelConfig(
    name="yolomaskrcnn_pipeline", components=[yolo_config, maskrcnn_config]
)

house_synthetic_config = ModelConfig(
    name="house_pipeline_synthetic", components=[maskrcnn_synthetic_config]
)
