"""
Weight path resolution:

1. FROZEN (PyInstaller .app / .exe):
   macOS : ~/Library/Application Support/TEMseg/weights/
   Windows: ~/AppData/Local/TEMseg/weights/

2. DEV with TEMSEG_WEIGHTS_DIR env var override:
   whatever the env var points to

3. DEV default:
   backend/weights/  (resolved via parents[4] from this file)
"""

import os
import sys
import platform
from pathlib import Path


def _app_support_weights_dir() -> Path:
    """Platform-appropriate user data directory for weights."""
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / "TEMseg" / "weights"
    elif system == "Windows":
        return Path.home() / "AppData" / "Local" / "TEMseg" / "weights"
    else:
        # Linux / other
        return Path.home() / ".local" / "share" / "TEMseg" / "weights"


def _dev_weights_dir() -> Path:
    """Dev fallback: backend/weights/ relative to this file."""
    return Path(__file__).resolve().parents[4] / "weights"


def _resolve_weights_dir() -> Path:
    """
    Resolve weights directory with priority:
      1. TEMSEG_WEIGHTS_DIR env var
      2. Frozen app bundle -> platform-specific app data dir
      3. Dev -> backend/weights/
    """
    if env_dir := os.environ.get("TEMSEG_WEIGHTS_DIR"):
        return Path(env_dir)

    if getattr(sys, "frozen", False):
        return _app_support_weights_dir()

    return _dev_weights_dir()


def app_support_shape_config_path() -> Path:
    """Platform-appropriate user override location for shape_config.toml."""
    system = platform.system()
    if system == "Darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "TEMseg"
            / "shape_config.toml"
        )
    elif system == "Windows":
        return Path.home() / "AppData" / "Local" / "TEMseg" / "shape_config.toml"
    else:
        return Path.home() / ".local" / "share" / "TEMseg" / "shape_config.toml"


def bundled_shape_config_path() -> Path:
    """Default shape_config.toml shipped next to compute_stats.py."""
    return Path(__file__).resolve().parent / "shape_config.toml"


def resolve_shape_config_path() -> Path:
    """
    Resolve shape classification config with priority:
      1. TEMSEG_SHAPE_CONFIG env var
      2. User override in app-support dir, if present
      3. Bundled default -> helpers/shape_config.toml

    Resolved fresh on every call rather than cached since the override file can
    be written or deleted at runtime via the /config/shape-rules endpoints,
    and a cached path would keep serving stale rules until process restart.
    """
    if env_path := os.environ.get("TEMSEG_SHAPE_CONFIG"):
        return Path(env_path)

    override = app_support_shape_config_path()
    if override.exists():
        return override

    return bundled_shape_config_path()


WEIGHTS_DIR = _resolve_weights_dir()


class Settings:
    WEIGHTS_DIR = WEIGHTS_DIR
    YOLO_MODEL_PATH = WEIGHTS_DIR / "best12x.onnx"
    SAM_MODEL_PATH = WEIGHTS_DIR / "sam_vit_b_01ec64.pth"
    MASKRCNN_MODEL_PATH = WEIGHTS_DIR / "maskrcnn_best_model.pth"
    MASKRCNN_SYNTHETIC_MODEL_PATH = WEIGHTS_DIR / "maskrcnn_best_model_synthetic.pth"

    @property
    def SHAPE_CONFIG_PATH(self) -> Path:
        return resolve_shape_config_path()

    @classmethod
    def weights_present(cls) -> bool:
        """Check if all required weight files exist."""
        return all(
            p.exists()
            for p in [
                cls.YOLO_MODEL_PATH,
                cls.SAM_MODEL_PATH,
                cls.MASKRCNN_MODEL_PATH,
                cls.MASKRCNN_SYNTHETIC_MODEL_PATH,
            ]
        )

    @classmethod
    def missing_weights(cls) -> list[str]:
        """Return list of missing weight filenames."""
        missing = []
        for name, path in [
            ("best12x.onnx", cls.YOLO_MODEL_PATH),
            ("sam_vit_b_01ec64.pth", cls.SAM_MODEL_PATH),
            ("maskrcnn_best_model.pth", cls.MASKRCNN_MODEL_PATH),
            ("maskrcnn_best_model_synthetic.pth", cls.MASKRCNN_SYNTHETIC_MODEL_PATH),
        ]:
            if not path.exists():
                missing.append(name)
        return missing


settings = Settings()
