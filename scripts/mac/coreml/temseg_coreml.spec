# -*- mode: python ; coding: utf-8 -*-
# TEMseg PyInstaller spec - COREML VARIANT (macOS .app, CoreML inference path)
#
# = temseg_tier2.spec (identical tier-1/tier-2 slimming; see
# ../tier2/temseg_tier2.spec for the full rationale) PLUS the CoreML runtime:
#   * collect_all("coremltools") — mlpackage loading at runtime (with its
#     native libs, data and hidden imports). Adds ~70MB (coremltools + sympy
#     deps) over the classic build — that's the whole point of having a
#     separate classic build.
#   * a coreml_variant.marker at the bundle root: launcher.py uses it to (a)
#     allow the coreml-variant weight_manifest entries (the mlpackages) and
#     (b) skip the frozen-classic TEMSEG_COREML=0 preset.
# Everything else (torch/ort/ultralytics collections, CUDA stripping, data
# pruning, EXE/BUNDLE) matches temseg_tier2.spec — torch stays bundled until
# the prompt decoder + resize glue no longer need it.
#
# LESSONS LEARNED (verified with clean-build + launch smoke tests):
#   * `torch.testing` must NOT be excluded. torch/autograd/gradcheck.py does
#     `import torch.testing` at module top level (line 11), and torch.autograd
#     is imported at torch init.
#   * `numpy.f2py` / `numpy.distutils` must NOT be excluded. scipy's
#     array_api_compat layer clones the numpy module at import time and that
#     triggers numpy.__getattr__ -> `import numpy.f2py` (see
#     scipy/_lib/array_api_compat/_internal.py). So numpy collection is left
#     EXACTLY as in tier 1.
#
# Changes vs tier 1 (temseg_tier1.spec):
#   1. Removed collect_data_files("torch"): _pyinstaller_hooks_contrib/stdhooks/
#      hook-torch.py already collects torch datas (with dev headers excluded via
#      excludes=["**/*.h", "**/*.hpp", "**/*.cuh", ...]). Same data files,
#      smaller bundle (no include/*.h etc.), one fewer duplicate collection.
#   2. Removed collect_submodules("torch"): hook-torch.py already sets
#      hiddenimports = collect_submodules("torch"). Removing ours removes a
#      duplicate ~2180-module walk during spec evaluation. (The hook still
#      walks them once during Analysis - the saving is the duplicate work.)
#   3. Removed collect_submodules("hyperspy"): collect_all("hyperspy") above
#      already returns the full submodule list (collect_all = collect_data_files
#      + collect_dynamic_libs + collect_submodules). Pure duplicate work.
#   4. `excludes` gains a small set of verified-safe subtrees:
#      tensorboard / torch.utils.tensorboard (try/except + TYPE_CHECKING only)
#      and the hyperspy.tests / rsciio.tests test suites. See the excludes
#      block for the reasoning.
#
# Everything else (numpy, torchvision, onnxruntime, ultralytics collections,
# hiddenimports, CUDA-stripping, data pruning, EXE/BUNDLE settings) matches
# temseg_tier1.spec.
"""
TEMseg PyInstaller spec: cross-platform (macOS .app bundle / Windows .exe one-dir)

Usage:
    pyinstaller temseg_tier2.spec

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

# Repo root three levels up from this spec (scripts/mac/tier2/ -> <root>/).
SPEC_DIR = Path(globals().get("SPECPATH", str(Path.cwd()))).resolve()
ROOT = SPEC_DIR.parents[2]

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
    print("[temseg_coreml.spec] FATAL. missing required input(s):", file=sys.stderr)
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

# Collect data files that torchvision / numpy / certifi need at runtime.
# TIER2: collect_data_files("torch") removed - hook-torch.py collects torch
#        datas (excluding dev headers).
datas += collect_data_files("torchvision")
datas += collect_data_files("numpy")
datas += collect_data_files("certifi")

# collect_all() returns a 3-tuple: (datas, binaries, hiddenimports)
# It must be unpacked, NOT appended directly to datas.
_hs_datas, _hs_binaries, _hs_hiddenimports = collect_all("hyperspy")
datas += _hs_datas
extra_binaries = _hs_binaries
extra_hiddenimports = _hs_hiddenimports

# rosettasciio metadata (distribution name) stays.
datas += copy_metadata("rosettasciio")

# --- CoreML variant additions (hiddenimports appended below, once the
# hiddenimports list exists) ---
_ct_datas, _ct_binaries, _ct_hiddenimports = collect_all("coremltools")
datas += _ct_datas
extra_binaries += _ct_binaries
# Marker file: launcher.py uses it to allow coreml-variant manifest entries
# (the mlpackages) and to skip the frozen-classic TEMSEG_COREML=0 preset.
_coreml_marker = SPEC_DIR / "coreml_variant.marker"
_coreml_marker.write_text("coreml build")
datas += [(str(_coreml_marker), ".")]

# rsciio: we need ALL subdirectories (emd/, tiff/, dm3/, etc.) because each
# contains a specifications.yaml that HyperSpy uses to register the format.
# collect_data_files("rsciio") recurses; the build script's step-4b copy is a
# further safety net.
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

# Collect ALL submodules for packages that have many dynamic imports.
# numpy is left unfiltered (exactly as tier 1): scipy's array_api_compat
# triggers `import numpy.f2py` at import time, so numpy.f2py/distutils must
# remain bundled.
hiddenimports += collect_submodules("numpy")
hiddenimports += collect_submodules("webview")
# TIER2: collect_submodules("torch") removed - hook-torch.py supplies
#        hiddenimports = collect_submodules("torch") already.
hiddenimports += collect_submodules("torchvision")
hiddenimports += collect_submodules("onnxruntime")
hiddenimports += collect_submodules("ultralytics")
# TIER2: collect_submodules("hyperspy") removed - collect_all("hyperspy")
#        above already provides the full list in extra_hiddenimports.
hiddenimports += collect_submodules("rsciio")  # actual package providing IO plugins
hiddenimports += extra_hiddenimports  # from collect_all("hyperspy")
hiddenimports += _ct_hiddenimports  # coreml variant: coremltools submodules

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
    # --- TIER2 additions (each verified against installed sources) ---
    # tensorboard: ultralytics imports torch.utils.tensorboard inside
    # try/except (utils/callbacks/tensorboard.py -> SummaryWriter=None), and
    # torch only references it under TYPE_CHECKING (torch/monitor/__init__.py).
    "tensorboard",
    "torch.utils.tensorboard",
    # Pure test suites that are only in the module graph because
    # collect_submodules("hyperspy") / collect_submodules("rsciio") add them
    # as hiddenimports. Never imported by runtime code.
    "hyperspy.tests",
    "rsciio.tests",
    # NOTE: torch.testing is deliberately NOT excluded - torch/autograd/
    # gradcheck.py does `import torch.testing` at module top level and
    # torch.autograd is imported at torch init. torch.utils.benchmark and
    # torch.distributed are also deliberately NOT excluded
    # (torch/profiler/_pattern_matcher.py and ultralytics/utils/torch_utils.py
    # import them unconditionally).
    #
    # NOTE: numpy.f2py / numpy.distutils are deliberately NOT excluded - scipy's
    # array_api_compat layer clones the numpy module at import time and triggers
    # `import numpy.f2py` via numpy.__getattr__.
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

print(f"[temseg_coreml.spec] Pruned {_pruned_count} data entries (~{_pruned_bytes // (1024*1024)}MB)")

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
    console=True,   # matches temseg.spec / temseg_tier1.spec
    icon=ICON_PATH,
)

# Only build the COLLECT (onedir staging) on Windows. On macOS the COLLECT
# output (dist/TEMseg) is never used - only the .app is shipped - and both
# COLLECT and BUNDLE unconditionally wipe and rewrite their full output trees,
# so the old chain copied the entire app twice per build. Feeding the BUNDLE
# directly from the EXE + TOCs produces the same .app contents.
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
        name="TEMseg-coreml.app",
        icon=ICON_PATH,
        bundle_identifier="com.temseg.coreml",
        info_plist={
            "CFBundleDisplayName": "TEMseg CoreML",
            "CFBundleShortVersionString": "0.1.0",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "12.0",
        },
    )
