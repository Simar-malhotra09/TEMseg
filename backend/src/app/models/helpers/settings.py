"""
Weight path resolution — works in three contexts:

1. FROZEN (PyInstaller .app):
   ~/Library/Application Support/TEMseg/weights/

2. DEV with TEMSEG_WEIGHTS_DIR env var override:
   whatever the env var points to

3. DEV default:
   backend/weights/  (resolved via parents[4] from this file)
"""

import sys
import platform
from pathlib import Path


def _app_support_weights_dir() -> Path:
    """Platform-appropriate user data directory for weights."""
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "TEMseg" / "weights"
    elif system == "Windows":
        app_data = Path.home() / "AppData" / "Local" / "TEMseg" / "weights"
        return app_data
    else:
        # Linux / other
        return Path.home() / ".local" / "share" / "TEMseg" / "weights"


def _is_frozen() -> bool:
    """True when running inside a PyInstaller bundle."""
    return getattr(sys, "frozen", False)




WEIGHTS_DIR = _app_support_weights_dir()


class Settings:
    WEIGHTS_DIR         = WEIGHTS_DIR
    YOLO_MODEL_PATH     = WEIGHTS_DIR / "best12x.onnx"
    SAM_MODEL_PATH      = WEIGHTS_DIR / "sam_vit_b_01ec64.pth"
    MASKRCNN_MODEL_PATH = WEIGHTS_DIR / "maskrcnn_best_model.pth"

    @classmethod
    def weights_present(cls) -> bool:
        """Check if all required weight files exist."""
        return all(p.exists() for p in [
            cls.YOLO_MODEL_PATH,
            cls.SAM_MODEL_PATH,
            cls.MASKRCNN_MODEL_PATH,
        ])

    @classmethod
    def missing_weights(cls) -> list[str]:
        """Return list of missing weight filenames."""
        missing = []
        for name, path in [
            ("best12x.onnx", cls.YOLO_MODEL_PATH),
            ("sam_vit_b_01ec64.pth", cls.SAM_MODEL_PATH),
            ("maskrcnn_best_model.pth", cls.MASKRCNN_MODEL_PATH),
        ]:
            if not path.exists():
                missing.append(name)
        return missing


settings = Settings()
