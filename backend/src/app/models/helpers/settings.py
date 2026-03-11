from dotenv import load_dotenv
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[4]  # repo root
load_dotenv(ROOT / ".env")

class Settings:
    YOLO_MODEL_PATH = ROOT / os.getenv("YOLO_MODEL_PATH")
    SAM_MODEL_PATH = ROOT / os.getenv("SAM_MODEL_PATH")
    MASKRCNN_MODEL_PATH = ROOT / os.getenv("MASKRCNN_MODEL_PATH")

settings = Settings()
