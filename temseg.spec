# -*- mode: python ; coding: utf-8 -*-
"""
TEMseg PyInstaller spec — macOS Apple Silicon (.app bundle)

Usage:
    pyinstaller temseg.spec

What this bundles:
    - launcher.py as the entry point
    - backend/src/ as backend_src (the FastAPI app + models)
    - frontend/out/ as frontend_out (static Next.js build)
    - weight_manifest.json (for first-run download)
    - DOES NOT bundle model weights — downloaded on first launch

Notes:
    - PyTorch, ONNX Runtime, OpenCV etc. are collected automatically
    - Hidden imports are listed explicitly for modules that PyInstaller misses
    - The app sets CWD to backend_src at runtime so relative imports work
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata, collect_all

block_cipher = None

ROOT = Path(SPECPATH)  # directory containing this .spec file

# ---------------------------------------------------------------------------
# Data files to bundle
# ---------------------------------------------------------------------------

datas = [
    # Static frontend build
    (str(ROOT / "frontend" / "out"), "frontend_out"),
    # Backend source (FastAPI app, models, routers)
    (str(ROOT / "backend" / "src"), "backend_src"),
    # Weight manifest
    (str(ROOT / "weight_manifest.json"), "."),
]

# Collect data files that PyTorch / torchvision / numpy need at runtime
datas += collect_data_files("torch")
datas += collect_data_files("torchvision")
datas += collect_data_files("numpy")
datas += collect_data_files("certifi")

datas += collect_data_files("hyperspy")
datas += copy_metadata("hyperspy")

# collect_all() returns a 3-tuple: (datas, binaries, hiddenimports)
# — it must be unpacked, NOT appended directly to datas.
_hs_datas, _hs_binaries, _hs_hiddenimports = collect_all("hyperspy")
datas += _hs_datas
extra_binaries = _hs_binaries
extra_hiddenimports = _hs_hiddenimports

datas += collect_data_files("rosettasciio")
datas += copy_metadata("rosettasciio")

# rsciio: we need ALL subdirectories (emd/, tiff/, dm3/, etc.) because each
# contains a specifications.yaml that HyperSpy uses to register the format.
# collect_data_files("rsciio") should grab them, but we also add an explicit
# recursive glob to be safe.
datas += collect_data_files("rsciio")
import rsciio
_rsciio_dir = Path(rsciio.__file__).parent
for _subdir in _rsciio_dir.iterdir():
    if _subdir.is_dir() and not _subdir.name.startswith("_"):
        # Bundle each format subdirectory (emd/, tiff/, bruker/, etc.)
        datas += [(str(_subdir), f"rsciio/{_subdir.name}")]
# ---------------------------------------------------------------------------
# Hidden imports PyInstaller tends to miss
# ---------------------------------------------------------------------------

hiddenimports = [
    # FastAPI / Starlette / Uvicorn
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
    # PyWebView
    "webview",
    # NumPy internals (PyInstaller misses these)
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
    # Image / science stack
    "cv2",
    "PIL",
    "PIL.Image",
    "scipy",
    "scipy.ndimage",
    "scipy.spatial",
    "tifffile",
    "skimage",
    "skimage.measure",
    # ML
    "torch",
    "torchvision",
    "onnxruntime",
    "ultralytics",
    "segment_anything",
    # HyperSpy (lazy imports PyInstaller misses)
    "hyperspy.io",
    "hyperspy.io.plugins",
    "hyperspy.signal",
    "hyperspy.signals",
    "hyperspy._components",
    "hyperspy.datasets",
    "hyperspy.utils",
    "hyperspy.misc",
    "hyperspy.extensions",
    # Backend app modules (PyInstaller can't trace string-based uvicorn import)
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
]

# Collect ALL submodules for packages that have many dynamic imports
hiddenimports += collect_submodules("numpy")
hiddenimports += collect_submodules("webview")
hiddenimports += collect_submodules("torch")
hiddenimports += collect_submodules("torchvision")
hiddenimports += collect_submodules("onnxruntime")
hiddenimports += collect_submodules("ultralytics")
hiddenimports += collect_submodules("hyperspy")
hiddenimports += collect_submodules("rosettasciio")
hiddenimports += collect_submodules("rsciio")  # actual package providing IO plugins
hiddenimports += extra_hiddenimports  # from collect_all("hyperspy")
# Deduplicate
hiddenimports = list(set(hiddenimports))

# ---------------------------------------------------------------------------
# Excludes — things we don't need, to reduce bundle size
# ---------------------------------------------------------------------------

excludes = [
    "tkinter",
    "notebook",
    "jupyter",
    "IPython",
    "pytest",
    "setuptools",
    "pip",
    # CUDA runtime libs are stripped from binaries below, but the Python
    # module torch.cuda must stay — PyTorch imports it at init.
    "triton",
    # Qt — PyWebView on macOS uses native WebKit, not Qt
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "sip",
    "PyQt5.sip",
]

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

a = Analysis(
    [str(ROOT / "launcher.py")],
    pathex=[
        str(ROOT / "backend" / "src"),
    ],
    binaries=extra_binaries,  # from collect_all("hyperspy")
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

# ---------------------------------------------------------------------------
# Remove CUDA libs if they snuck in (can save ~1GB)
# ---------------------------------------------------------------------------

cuda_prefixes = ("libcuda", "libnvrtc", "libcublas", "libcudnn", "libcufft",
                 "libcurand", "libcusparse", "libcusolver", "libnccl",
                 "libnvToolsExt", "libcudart")

a.binaries = [
    b for b in a.binaries
    if not any(b[0].startswith(prefix) for prefix in cuda_prefixes)
]

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

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
    upx=False,      # UPX can break signed binaries on Mac
    console=False,   # no terminal window
    icon=str(ROOT / "temseg_icon.icns"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="TEMseg",
)

app = BUNDLE(
    coll,
    name="TEMseg.app",
    icon=str(ROOT / "temseg_icon.icns"),
    bundle_identifier="com.temseg.app",
    info_plist={
        "CFBundleDisplayName": "TEMseg",
        "CFBundleShortVersionString": "0.1.0",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "12.0",
    },
)
