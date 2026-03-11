from pathlib import Path
import os

def find_root(marker: str = "pyproject.toml") -> Path:
    """Walk up from this file until we find the marker file."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find project root (looking for {marker})")

ROOT = find_root()

class Settings:
    WEIGHTS_DIR = ROOT / "weights"
    YOLO_MODEL_PATH    = WEIGHTS_DIR / "best12x.onnx"
    SAM_MODEL_PATH     = WEIGHTS_DIR / "sam_vit_b_01ec64.pth"
    MASKRCNN_MODEL_PATH = WEIGHTS_DIR / "maskrcnn_best_model.pth"

settings = Settings()
