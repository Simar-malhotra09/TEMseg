"""
install.py — run once on a fresh machine to set up the backend.

Usage:
  python install.py

What it does:
  1. Detects platform (Windows/Mac/Linux) and GPU availability
  2. Installs the correct torch + torchvision build for your hardware
  3. Installs all other dependencies from pyproject.toml
  4. Checks for required model weights and tells you where to get them
"""

import subprocess
import sys
import platform
from pathlib import Path

from torch import onnx
from src.app.models.helpers.settings import Settings



def run(cmd: list[str], **kwargs) -> int:
    print(f"\n  >> {' '.join(cmd)}")
    result = subprocess.run(cmd, **kwargs)
    return result.returncode


def pip(*args):
    return run([sys.executable, "-m", "pip", *args])


def detect_device() -> str:
    """
    Returns 'cuda', 'mps', or 'cpu'.
    Tries to import torch if already installed to check cuda availability.
    Falls back to platform heuristics.
    """
    # check for NVIDIA GPU on Windows/Linux via nvidia-smi
    if platform.system() in ("Windows", "Linux"):
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True
        )
        if result.returncode == 0:
            print("  [detect] NVIDIA GPU found via nvidia-smi")
            return "cuda"

    # check for Apple Silicon
    if platform.machine() == "arm64":
        print("  [detect] Apple Silicon detected → MPS")
        return "mps"

    print("  [detect] No GPU detected → CPU")
    return "cpu"


def torch_install_cmd(device: str) -> list[str]:
    """
    Returns the pip install command for the correct torch build.
    Uses torch 2.5.1 — latest stable with broad CUDA/platform support.
    """
    base = ["torch==2.5.1", "torchvision==0.20.1"]

    if device == "cuda":
        # CUDA 12.1 — works on most modern NVIDIA GPUs (GTX 10xx and newer)
        # if you have an older GPU with CUDA 11.8 support only, change cu121 → cu118
        index = "https://download.pytorch.org/whl/cu121"
        return [
            sys.executable, "-m", "pip", "install",
            *base,
            "--index-url", index,
        ]
    elif device == "mps":
        # standard PyPI torch has MPS support built in for Apple Silicon
        index = "https://download.pytorch.org/whl/cpu"
        return [
            sys.executable, "-m", "pip", "install",
            *base,
            "--index-url", index,
        ]
    else:
        # CPU only
        index = "https://download.pytorch.org/whl/cpu"
        return [
            sys.executable, "-m", "pip", "install",
            *base,
            "--index-url", index,
        ]

def onnx_install_cmd(device:str):
    if device == "cuda":
        pip("install", "onnxruntime-gpu")
    else:
        pip("install", "onnxruntime")


def is_lfs_pointer(path: Path) -> bool:
    """LFS pointer files are <200 bytes and start with 'version https://git-lfs'"""
    if path.stat().st_size > 1000:
        return False
    try:
        content = path.read_text(errors="ignore")
        return content.startswith("version https://git-lfs")
    except Exception:
        return False

def check_weights(settings):
    """
    Checks for required model weight files and prints download instructions if missing.
    """
    from pathlib import Path

    weights = {
        "SAM ViT-B": Path(settings.SAM_MODEL_PATH),
        "YOLO ": Path(settings.YOLO_MODEL_PATH),
        "MaskRCNN": Path(settings.MASKRCNN_MODEL_PATH)
    }

    missing = []
    lfs_pointers = []

    print("\n  [weights] Checking model weights...")
    for name, path in weights.items():
        if not path.exists():
            print(f"  [weights] MISSING  {name} -> {path}")
            missing.append((name, path))
        elif is_lfs_pointer(path):
            print(f"  [weights] LFS POINTER  {name} -> {path}  (not downloaded yet)")
            lfs_pointers.append((name, path))
        else:
            size_mb = path.stat().st_size / 1_000_000
            print(f"  [weights]  ok  {name} -> {path}  ({size_mb:.0f} MB)")

    if lfs_pointers:
        print("\n  -- LFS pointer files found -- run these to download actual weights:")
        print("\n    git lfs install")
        print("    git lfs pull\n")
        print("  If git-lfs is not installed: https://git-lfs.com")
        print("  If on a network that blocks LFS, download manually:")
        # for name, path in lfs_pointers:
        #     if "SAM" in name:
        #         print(f"\n    SAM ViT-B (~375 MB):")
        #         print(f"    https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        #         print(f"    Save to: {path.resolve()}")
        #     else:
        #         print(f"\n    {name}: copy from another machine to {path.resolve()}")
        return False

    if missing:
        print("\n  ── Missing weights ──────────────────────────────────────────")
        for name, path in missing:
            if "SAM" in name:
                print(f"\n  {name}:")
                print(f"    Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
                print(f"    Save to:       {path}")
            elif "YOLO" in name:
                print(f"\n  {name}:")
                print(f"    This is your trained YOLO model — copy it from your Mac.")
                print(f"    Save to: {path}")
        print()
        return False
    return True


def main():

    settings = Settings()

    print("=" * 60)
    print("  TEM Particle Seg — Backend Setup")
    print(f"  Python {sys.version}")
    print(f"  Platform: {platform.system()} {platform.machine()}")
    print("=" * 60)

    # step 1 — detect device
    print("\n[1/4] Detecting hardware...")
    device = detect_device()
    print(f"  → Installing torch for device: {device}")

    # step 2 — install torch
    print(f"\n[2/4] Installing torch 2.5.1 + torchvision 0.20.1 ({device})...")
    cmd = torch_install_cmd(device)
    onnx_install_cmd(device)
    code = run(cmd)
    if code != 0:
        print("\n  ERROR: torch install failed. Check your internet connection and try again.")
        sys.exit(1)

    # step 3 — install everything else from pyproject.toml
    print("\n[3/4] Installing project dependencies...")
    code = pip("install", "-e", ".[dev]")
    if code != 0:
        print("\n  ERROR: dependency install failed.")
        sys.exit(1)

    # step 4 — check weights
    print("\n[4/4] Checking model weights...")
    weights_ok = check_weights(settings)

    # done
    print("\n" + "=" * 60)
    if weights_ok:
        print("  ✓ Setup complete! Start the backend with:")
        print()
        print("    python start.py")
        print()
        print("  Or manually:")
        print()
        print("    uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000")
    else:
        print("  ⚠ Setup mostly complete — download missing weights above,")
        print("  then run:  python start.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
