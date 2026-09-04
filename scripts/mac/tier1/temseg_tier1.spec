# -*- mode: python ; coding: utf-8 -*-
# TEMseg PyInstaller spec - TIER 1 (macOS .app bundle, build-speed optimised)
#
# Drop-in replacement for <repo-root>/temseg.spec. Tier 1 keeps the produced
# app bundle byte-comparable with the original spec: only redundant build WORK
# is removed, nothing that changes what ships. See README.md in this directory.
#
# Changes vs temseg.spec:
#   1. ROOT is computed three levels up from this file's location
#      (scripts/mac/tier1/) instead of SPECPATH; all other paths unchanged.
#   2. macOS: the COLLECT step is skipped; BUNDLE now consumes the exact same
#      inputs directly (exe, a.binaries, a.zipfiles, a.datas). BUNDLE accepts
#      EXE + raw TOCs exactly like COLLECT (PyInstaller 6.21.0,
#      building/osx.py), PKG dependencies pass through identically, and
#      append_pkg=True means no PKG entry ever reached the COLLECT TOC - so
#      the two bundle layouts are equivalent. The old chain built dist/TEMseg
#      and then copied everything again into dist/TEMseg.app; we never shipped
#      dist/TEMseg. This removes one full-app copy per build.
#   3. Removed collect_data_files("rosettasciio") and
#      collect_submodules("rosettasciio"): both import-resolving calls use the
#      MODULE name, and "rosettasciio" is not importable (the import package is
#      "rsciio"). Verified: both return 0 entries. copy_metadata() stays - it
#      resolves by DISTRIBUTION name.
#   4. Removed duplicated hyperspy collection: collect_all("hyperspy") already
#      includes the data files, metadata and all hyperspy submodules that the
#      two manual lines re-collected.
#   5. Removed the manual rsciio subdirectory loop: collect_data_files("rsciio")
#      already recurses; the loop re-added the same files (deduped by PyInstaller's
#      TOC normalisation). Loose .py sources are additionally irrelevant because
#      (a) rsciio modules ship compiled in the PYZ via the hiddenimports below
#      and (b) the build script's step-4b copy still mirrors the full directory
#      into Contents/Frameworks/ as a safety net.
#
# Everything else (datas destinations, hiddenimports, excludes, CUDA-stripping,
# data pruning, EXE/BUNDLE settings, Info.plist) matches temseg.spec.
"""
TEMseg PyInstaller spec: cross-platform (macOS .app bundle / Windows .exe one-dir)

Usage:
    pyinstaller temseg_tier1.spec

What this bundles:
    - launcher.py as the entry point
    - backend/src/ as backend_src (the FastAPI app + models)
    - frontend/out/ as frontend_out (static Next.js build)
    - weight_manifest.json (for first-run download)
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata, collect_all

block_cipher = None

# TIER1: repo root three levels up from this spec (original spec used
# ROOT = Path(SPECPATH) because it lives at the repo root itself).
SPEC_DIR = Path(globals().get("SPECPATH", str(Path.cwd()))).resolve()
ROOT = SPEC_DIR.parents[2]  # <root>/scripts/mac/tier1 -> <root>/

IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform == "win32"

# Platform-specific icons
if IS_MAC:
    ICON_PATH = str(ROOT / "assets" / "temseg_icon.icns")
elif IS_WIN:
    ICON_PATH = str(ROOT / "assets" / "temseg_icon.ico")
else:
    ICON_PATH = None

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
    print("[temseg_tier1.spec] FATAL. missing required input(s):", file=sys.stderr)
    for p in _missing:
        print(f"  - {p}", file=sys.stderr)
    raise SystemExit(1)

# Data files to bundle

datas = [
    # Static frontend build
    (str(ROOT / "frontend" / "out"), "frontend_out"),
    # Backend source (FastAPI app, models, routers)
    (str(ROOT / "backend" / "src"), "backend_src"),
    # Weight manifest
    (str(ROOT / "weight_manifest.json"), "."),
    # Shape rules config, read by the bundled fallback in settings.py
    (str(ROOT / "backend" / "src" / "app" / "models" / "helpers" / "shape_config.toml"),
     "app/models/helpers"),
]

# Collect data files that PyTorch / torchvision / numpy need at runtime
datas += collect_data_files("torch")
datas += collect_data_files("torchvision")
datas += collect_data_files("numpy")
datas += collect_data_files("certifi")

# TIER1: manual hyperspy data/metadata lines removed - collect_all() below
# subsumes them.

# collect_all() returns a 3-tuple: (datas, binaries, hiddenimports)
#, it must be unpacked, NOT appended directly to datas.
_hs_datas, _hs_binaries, _hs_hiddenimports = collect_all("hyperspy")
datas += _hs_datas
extra_binaries = _hs_binaries
extra_hiddenimports = _hs_hiddenimports

# TIER1: collect_data_files("rosettasciio") removed (no-op - see header).
# The metadata name is the DISTRIBUTION name, so this one stays:
datas += copy_metadata("rosettasciio")

# rsciio: we need ALL subdirectories (emd/, tiff/, dm3/, etc.) because each
# contains a specifications.yaml that HyperSpy uses to register the format.
# TIER1: collect_data_files("rsciio") recurses already; the manual per-subdir
# loop and `import rsciio` that used to live here were pure duplicates.
datas += collect_data_files("rsciio")

# Hidden imports PyInstaller tends to miss

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
# TIER1: collect_submodules("rosettasciio") removed (no-op - see header).
hiddenimports += collect_submodules("rsciio")  # actual package providing IO plugins
hiddenimports += extra_hiddenimports  # from collect_all("hyperspy")

# Windows-only: pywebview uses MS Edge WebView2 backend, which pulls pywin32 COM bits
if IS_WIN:
    hiddenimports += [
        "pythoncom",
        "pywintypes",
        "win32api",
        "win32con",
        "win32com",
        "win32com.client",
    ]

# Deduplicate
hiddenimports = list(set(hiddenimports))

# Excludes to reduce bundle size

excludes = [
    "tkinter",
    "notebook",
    "jupyter",
    "IPython",
    "pytest",
    "setuptools",
    "pip",
    # CUDA runtime libs are stripped from binaries below, but the Python
    # module torch.cuda must stay since pytorch calls at init.
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


# Linux / macOS naming (lib*.so / lib*.dylib)
_cuda_unix = ("libcuda", "libnvrtc", "libcublas", "libcudnn", "libcufft",
              "libcurand", "libcusparse", "libcusolver", "libnccl",
              "libnvToolsExt", "libcudart")

# Windows naming (versioned .dll, e.g. cudart64_12.dll, cublas64_12.dll)
_cuda_win = ("cudart64_", "cublas64_", "cublasLt64_", "cudnn", "cufft64_",
             "curand64_", "cusparse64_", "cusolver64_", "nvrtc64_",
             "nvrtc-builtins64_", "nccl64_", "nvToolsExt64_")

cuda_prefixes = _cuda_unix + _cuda_win

def _is_cuda_lib(name: str) -> bool:
    base = Path(name).name.lower()
    return any(base.startswith(p.lower()) for p in cuda_prefixes)

a.binaries = [b for b in a.binaries if not _is_cuda_lib(b[0])]

# Post-analysis size pruning

# Patterns to exclude from a.datas (src_path check)
_exclude_data_patterns = [
    # dev session data — user runtime state, not app code
    "/sessions/",
    "backend/src/sessions/",
    "backend_src/sessions/",
    # test directories across packages
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

def _should_exclude_data(entry):
    src = entry[1] if len(entry) > 1 else entry[0]
    dest = entry[0]
    src_str, dest_str = str(src), str(dest)
    for pat in _exclude_data_patterns:
        if pat in src_str or pat in dest_str:
            return True
    return False

new_datas = []
for entry in a.datas:
    # a.datas entries are (dest_name, src_path, type) or (dest_name, src_path)
    if _should_exclude_data(entry):
        _pruned_count += 1
        try:
            _pruned_bytes += Path(src).stat().st_size
        except Exception:
            pass
    else:
        new_datas.append(entry)

a.datas = new_datas

print(f"[TEMseg.spec] Pruned {_pruned_count} data entries (~{_pruned_bytes // (1024*1024)}MB)")

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
    upx=False,      # UPX can break signed binaries on Mac
    console=True,   # matches temseg.spec
    icon=ICON_PATH,
)

# TIER1: only build the COLLECT (onedir staging) on Windows. On macOS the
# COLLECT output (dist/TEMseg) is never used - only the .app is shipped - and
# both COLLECT and BUNDLE unconditionally wipe and rewrite their full output
# trees, so the old chain copied the entire app twice per build. Feeding the
# BUNDLE directly from the EXE + TOCs produces the same .app contents (with
# exclude_binaries=True and append_pkg=True, the COLLECT TOC and the direct
# EXE+TOC list are the same set of entries - verified against PyInstaller
# 6.21.0 sources).
if IS_WIN:
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=False,
        name="TEMseg",
    )

# macOS .app bundle
if IS_MAC:
    app = BUNDLE(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        name="TEMseg.app",
        icon=ICON_PATH,
        bundle_identifier="com.temseg.app",
        info_plist={
            "CFBundleDisplayName": "TEMseg",
            "CFBundleShortVersionString": "0.1.0",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "12.0",
        },
    )
