"""Build the macOS application variants with doit."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterator, NamedTuple

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv" / "bin" / "python"
PYINSTALLER = ROOT / ".venv" / "bin" / "pyinstaller"


class BuildVariant(NamedTuple):
    name: str
    spec: Path
    app_name: str
    required_modules: tuple[str, ...]


TIER2 = BuildVariant(
    "tier2",
    Path("scripts/mac/tier2/temseg_tier2.spec"),
    "TEMseg",
    (
        "torch", "torchvision", "numpy", "scipy", "sklearn", "skimage",
        "cv2", "PIL", "tifffile", "onnxruntime", "ultralytics",
        "segment_anything", "hyperspy", "rsciio", "fastapi", "uvicorn",
        "pydantic", "webview", "PyInstaller",
    ),
)
COREML = BuildVariant(
    "coreml",
    Path("scripts/mac/coreml/temseg_coreml.spec"),
    "TEMseg-coreml",
    (
        "torch", "torchvision", "numpy", "scipy", "sklearn", "skimage",
        "cv2", "PIL", "tifffile", "onnxruntime", "ultralytics",
        "segment_anything", "hyperspy", "rsciio", "coremltools", "fastapi",
        "uvicorn", "pydantic", "webview", "PyInstaller",
    ),
)


def files_under(directory: Path) -> list[str]:
    ignored_parts = {".next", "node_modules", "out", "__pycache__"}
    return sorted(
        str(path.relative_to(ROOT))
        for path in directory.rglob("*")
        if path.is_file() and not ignored_parts.intersection(path.parts)
    )


def frontend_inputs() -> list[str]:
    return files_under(ROOT / "frontend")


def install_frontend() -> None:
    if not (ROOT / "frontend" / "node_modules").is_dir():
        subprocess.run(["npm", "ci"], cwd=ROOT / "frontend", check=True)


def build_frontend() -> None:
    subprocess.run(["npm", "run", "build"], cwd=ROOT / "frontend", check=True)


def publish_unblock_script() -> None:
    dist_dir = ROOT / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "scripts/mac/unblock.sh", dist_dir / "unblock.sh")


def variant_inputs(variant: BuildVariant) -> list[str]:
    fixed_inputs = [
        "dodo.py", "launcher.py", "rthook_hyperspy.py", "weight_manifest.json",
        "pyproject.toml", "uv.lock", str(variant.spec),
        "scripts/mac/unblock.sh",
    ]
    backend_inputs = files_under(ROOT / "backend" / "src")
    return sorted(set(fixed_inputs + backend_inputs))


def safe_branch_name() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    sanitized = "".join(
        character if character.isalnum() else "-"
        for character in result.stdout.strip()
    ).strip("-")
    return sanitized or "unknown"


def check_environment(variant: BuildVariant) -> None:
    if os.sys.platform != "darwin":
        raise RuntimeError("macOS app builds must run on macOS")
    if not PYINSTALLER.is_file():
        raise FileNotFoundError(
            f"PyInstaller is missing at {PYINSTALLER}; install it in .venv first"
        )
    missing = [
        module
        for module in variant.required_modules
        if subprocess.run(
            [
                str(PYTHON), "-c",
                "import importlib.util, sys; "
                f"sys.exit(0 if importlib.util.find_spec('{module}') else 1)",
            ],
            cwd=ROOT,
            check=False,
        ).returncode != 0
    ]
    if missing:
        raise RuntimeError(
            f"Missing .venv modules for {variant.name}: {', '.join(missing)}"
        )


def build_variant(variant: BuildVariant) -> None:
    check_environment(variant)
    (ROOT / ".env").touch(exist_ok=True)

    dist_dir = ROOT / "dist" / variant.name
    work_dir = ROOT / "build" / variant.name
    dist_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(PYINSTALLER), str(variant.spec),
            "--distpath", str(dist_dir),
            "--workpath", str(work_dir),
            "--noconfirm",
        ],
        cwd=ROOT,
        check=True,
    )

    rsciio_path = subprocess.run(
        [
            str(PYTHON), "-c",
            "import rsciio; from pathlib import Path; "
            "print(Path(rsciio.__file__).parent)",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_app = dist_dir / f"{variant.app_name}.app"
    rsciio_destination = (
        source_app / "Contents" / "Frameworks" / "rsciio"
    )
    if not source_app.is_dir():
        raise FileNotFoundError(f"PyInstaller did not produce {source_app}")
    if rsciio_destination.exists():
        shutil.rmtree(rsciio_destination)
    shutil.copytree(rsciio_path, rsciio_destination)

    target_app = dist_dir / f"{variant.app_name}-{safe_branch_name()}.app"
    if target_app.exists():
        shutil.rmtree(target_app)
    source_app.rename(target_app)


def clean_variant(variant: BuildVariant) -> None:
    for path in (ROOT / "build" / variant.name, ROOT / "dist" / variant.name):
        if path.exists():
            shutil.rmtree(path)


def task_frontend() -> dict[str, object]:
    return {
        "file_dep": frontend_inputs(),
        "targets": ["frontend/out/index.html"],
        "actions": [install_frontend, build_frontend],
        "clean": ["rm -rf frontend/out"],
    }


def task_unblock() -> dict[str, object]:
    return {
        "file_dep": ["scripts/mac/unblock.sh"],
        "targets": ["dist/unblock.sh"],
        "actions": [publish_unblock_script],
    }


def task_build() -> Iterator[dict[str, object]]:
    for variant in (TIER2, COREML):
        yield {
            "name": variant.name,
            "task_dep": ["frontend", "unblock"],
            "file_dep": variant_inputs(variant),
            "targets": [f"dist/{variant.name}/{variant.app_name}-{safe_branch_name()}.app"],
            "actions": [(build_variant, [variant])],
        }


def task_purge() -> Iterator[dict[str, object]]:
    for variant in (TIER2, COREML):
        yield {
            "name": variant.name,
            "actions": [(clean_variant, [variant])],
        }
