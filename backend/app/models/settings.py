import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH")
    SAM_MODEL_PATH = os.environ.get("SAM_MODEL_PATH")

settings = Settings()


