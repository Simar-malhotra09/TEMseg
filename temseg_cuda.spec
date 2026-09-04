# -*- mode: python ; coding: utf-8 -*-
"""
TEMseg PyInstaller spec for windows CUDA build

Usage:
    pyinstaller temseg_cuda.spec

Differences from temseg.spec (CPU build):
    - CUDA runtime DLLs are NOT stripped. 
    - Expects onnxruntime-gpu (not onnxruntime) installed in the venv before building.
    - Windows-only; produces dist/TEMseg-cuda/ one-dir bundle

Prerequisites:
    - NVIDIA driver (CUDA 12.1 compatible) on the build machine is NOT required,
      but onnxruntime-gpu must be installed: uv pip install onnxruntime-gpu
      (onnxruntime and onnxruntime-gpu cannot coexist, so uninstall cpu variant first)
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata, collect_all

block_cipher = None

ROOT = Path(SPECPATH)

IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform == "win32"

if IS_MAC:
    raise RuntimeError("temseg_cuda.spec is Windows-only. Use temseg.spec on macOS.")

ICON_PATH = str(ROOT / "assets" / "temseg_icon.ico") if IS_WIN else None

_required_inputs = [
    str(ROOT / "launcher.py"),
    str(ROOT / "weight_manifest.json"),
    str(ROOT / "rthook_hyperspy.py"),
    str(ROOT / "frontend" / "out" / "index.html"),
    str(ROOT / "backend" / "src"),
]
if ICON_PATH:
    _required_inputs.append(ICON_PATH)
_missing = [p for p in _required_inputs if not Path(p).exists()]
if _missing:
    print("[temseg_cuda.spec] FATAL. missing required input(s):", file=sys.stderr)
    for p in _missing:
        print(f"  - {p}", file=sys.stderr)
    raise SystemExit(1)

# Data files

datas = [
    (str(ROOT / "frontend" / "out"), "frontend_out"),
    (str(ROOT / "backend" / "src"), "backend_src"),
    (str(ROOT / "weight_manifest.json"), "."),
    # Shape rules config, read by the bundled fallback in settings.py
    (str(ROOT / "backend" / "src" / "app" / "models" / "helpers" / "shape_config.toml"),
     "app/models/helpers"),
]

datas += collect_data_files("torch")
datas += collect_data_files("torchvision")
datas += collect_data_files("numpy")
datas += collect_data_files("certifi")

datas += collect_data_files("hyperspy")
datas += copy_metadata("hyperspy")

_hs_datas, _hs_binaries, _hs_hiddenimports = collect_all("hyperspy")
datas += _hs_datas
extra_binaries = _hs_binaries
extra_hiddenimports = _hs_hiddenimports

datas += collect_data_files("rosettasciio")
datas += copy_metadata("rosettasciio")

datas += collect_data_files("rsciio")
import rsciio
_rsciio_dir = Path(rsciio.__file__).parent
for _subdir in _rsciio_dir.iterdir():
    if _subdir.is_dir() and not _subdir.name.startswith("_"):
        datas += [(str(_subdir), f"rsciio/{_subdir.name}")]

# Hidden imports

hiddenimports = [
    "uvicorn",
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    "fastapi",
    "starlette",
    "starlette.routing",
    "starlette.middleware",
    "starlette.middleware.cors",
    "pydantic",
    "multipart",
    "python_multipart",
    "webview",
    "numpy",
    "numpy.core",
    "numpy.core._multiarray_umath",
    "numpy.core.multiarray",
    "numpy.core.numeric",
    "numpy.core._methods",
    "numpy.lib",
    "numpy.lib.format",
    "numpy.fft",
    "numpy.linalg",
    "numpy.random",
    "cv2",
    "PIL",
    "PIL.Image",
    "scipy",
    "scipy.ndimage",
    "scipy.spatial",
    "tifffile",
    "skimage",
    "skimage.measure",
    "torch",
    "torchvision",
    "onnxruntime",
    "ultralytics",
    "segment_anything",
    "hyperspy.io",
    "hyperspy.io.plugins",
    "hyperspy.signal",
    "hyperspy.signals",
    "hyperspy._components",
    "hyperspy.datasets",
    "hyperspy.utils",
    "hyperspy.misc",
    "hyperspy.extensions",
    "app",
    "app.api",
    "app.api.main",
    "app.api.routers",
    "app.api.routers.images",
    "app.api.routers.segment",
    "app.api.routers.masks",
    "app.api.routers.ground_truths",
    "app.api.routers.export",
    "app.api.utils",
    "app.api.instances",
    "app.api.live_models",
    "app.models",
    "app.models.base_model",
    "app.models.impls",
    "app.models.impls.yolosam",
    "app.models.impls.maskrcnn",
    "app.models.helpers",
    "app.models.helpers.config",
    "app.models.helpers.settings",
    "app.scripts",
    "app.scripts.compare_gt",
    # Windows WebView2
    "pythoncom",
    "pywintypes",
    "win32api",
    "win32con",
    "win32com",
    "win32com.client",
]

hiddenimports += collect_submodules("numpy")
hiddenimports += collect_submodules("webview")
hiddenimports += collect_submodules("torch")
hiddenimports += collect_submodules("torchvision")
hiddenimports += collect_submodules("onnxruntime")
hiddenimports += collect_submodules("ultralytics")
hiddenimports += collect_submodules("hyperspy")
hiddenimports += collect_submodules("rosettasciio")
hiddenimports += collect_submodules("rsciio")
hiddenimports += extra_hiddenimports

hiddenimports = list(set(hiddenimports))

# Excludes

excludes = [
    "tkinter",
    "notebook",
    "jupyter",
    "IPython",
    "pytest",
    "setuptools",
    "pip",
    "triton",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "sip",
    "PyQt5.sip",
]

# Analysis

a = Analysis(
    [str(ROOT / "launcher.py")],
    pathex=[
        str(ROOT / "backend" / "src"),
    ],
    binaries=extra_binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(ROOT / "rthook_hyperspy.py")],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# CUDA libs.
# cudart64_*.dll, cublas64_*.dll, cudnn*.dll etc. stay in the bundle so
# torch.cuda.is_available() returns True on NVIDIA hardware.

# Post-analysis size pruning (same as CPU spec)

_exclude_data_patterns = [
    "/sessions/",
    "backend_src/sessions",
    "hyperspy/tests/",
    "rsciio/tests/",
    "numpy/_core/tests/",
    "numpy/f2py/tests/",
    "numpy/lib/tests/",
    "numpy/random/tests/",
    "numpy/typing/tests/",
    "pint/testsuite/",
    "backend_src/app/api/tests/",
    ".egg-info/",
    "_polars_runtime_32/",
    "_polars_runtime",
]

_pruned_count = 0
_pruned_bytes = 0

def _should_exclude_data(src):
    src_str = str(src)
    for pat in _exclude_data_patterns:
        if pat in src_str:
            return True
    return False

new_datas = []
for entry in a.datas:
    src = entry[1] if len(entry) > 1 else entry[0]
    if _should_exclude_data(src):
        _pruned_count += 1
        try:
            _pruned_bytes += Path(src).stat().st_size
        except Exception:
            pass
    else:
        new_datas.append(entry)

a.datas = new_datas

print(f"[TEMseg-cuda.spec] Pruned {_pruned_count} data entries (~{_pruned_bytes // (1024*1024)}MB)")

# Build

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TEMseg",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    icon=ICON_PATH,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="TEMseg-cuda",
)
