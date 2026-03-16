import subprocess
import sys
import platform
import shutil
from pathlib import Path

from src.app.models.helpers.settings import Settings


def run(cmd: list[str], **kwargs) -> int:
    print(f"\n  >> {' '.join(cmd)}")
    result = subprocess.run(cmd, **kwargs)
    return result.returncode


def get_uv() -> list[str] | None:
    """Return ['uv', 'pip'] if uv is available, else None."""
    if shutil.which("uv"):
        return ["uv", "pip"]
    return None


def pip_run(*args):
    uv = get_uv()
    if uv:
        return run([*uv, *args])
    return run([sys.executable, "-m", "pip", *args])


def detect_device() -> str:
    if platform.system() in ("Windows", "Linux"):
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            print("  [detect] NVIDIA GPU found via nvidia-smi")
            return "cuda"
    if platform.machine() == "arm64":
        print("  [detect] Apple Silicon detected → MPS")
        return "mps"
    print("  [detect] No GPU detected → CPU")
    return "cpu"


def install_torch(device: str) -> int:
    base = ["torch==2.5.1", "torchvision==0.20.1"]
    index = (
        "https://download.pytorch.org/whl/cu121"
        if device == "cuda"
        else "https://download.pytorch.org/whl/cpu"
    )
    return pip_run("install", *base, "--index-url", index)


def install_onnx(device: str):
    pkg = "onnxruntime-gpu" if device == "cuda" else "onnxruntime"
    pip_run("install", pkg)


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
        "MaskRCNN": Path(settings.MASKRCNN_MODEL_PATH),
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
        return False

    if missing:
        print("\n  ── Missing weights ──────────────────────────────────────────")
        for name, path in missing:
            if "SAM" in name:
                print(f"\n  {name}:")
                print(
                    "    Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
                )
                print(f"    Save to:       {path}")
            elif "YOLO" in name:
                print(f"\n  {name}:")
                print("    This is your trained YOLO model — copy it from your Mac.")
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
    code = install_torch(device)
    install_onnx(device)
    if code != 0:
        print(
            "\n  ERROR: torch install failed. Check your internet connection and try again."
        )
        sys.exit(1)

    # step 3 — install everything else from pyproject.toml
    print("\n[3/4] Installing project dependencies...")
    if shutil.which("uv"):
        code = run(["uv", "sync", "--extra", "dev"])
    else:
        code = run([sys.executable, "-m", "pip", "install", "-e", ".[dev]"])
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
