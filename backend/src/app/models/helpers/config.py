from app.models.base_model import SubModelConfig, ModelConfig
from app.models.helpers.settings import settings


yolo_config = SubModelConfig("yolo", settings.YOLO_MODEL_PATH)
sam_config = SubModelConfig("sam", settings.SAM_MODEL_PATH)
maskrcnn_config= SubModelConfig("maskrcnn", settings.MASKRCNN_MODEL_PATH)

nano_config = ModelConfig(
    name="nano_pipeline",
    components=[yolo_config, sam_config]
)

house_config= ModelConfig(
        name="house_pipeline",
        components=[maskrcnn_config]
        )

