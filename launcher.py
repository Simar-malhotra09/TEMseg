"""
launcher.py — TEMseg desktop application launcher.
"""

import os

os.environ["YOLO_AUTOINSTALL"] = "False"

import json
import hashlib
import os
import platform
import sys
import threading
import time
import urllib.request
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import webview


try:
    # frozen build: app.logutils is bundled into the PYZ directly
    from app.logutils import get_logger, init_logging
except ModuleNotFoundError:
    # dev checkout: backend/src isn't on sys.path yet
    sys.path.insert(0, str(Path(__file__).parent / "backend" / "src"))
    from app.logutils import get_logger, init_logging

init_logging()
log = get_logger("launcher")

if getattr(sys, "frozen", False):
    import ssl
    import certifi

    os.environ["SSL_CERT_FILE"] = certifi.where()
    ssl._create_default_https_context = ssl.create_default_context
    try:
        import yaml
        import rsciio

        if not rsciio.IO_PLUGINS:
            bundle = Path(sys._MEIPASS)
            rsciio_data = bundle / "rsciio"
            if not rsciio_data.exists():
                # check Resources
                rsciio_data = bundle.parent / "Resources" / "rsciio"
            if rsciio_data.exists():
                for sub, _, _ in os.walk(str(rsciio_data)):
                    specsf = os.path.join(sub, "specifications.yaml")
                    if os.path.isfile(specsf):
                        with open(specsf, "r") as stream:
                            specs = yaml.safe_load(stream)
                            specs["api"] = "rsciio.%s" % os.path.split(sub)[1]
                            rsciio.IO_PLUGINS.append(specs)
                log.info(f"Manually loaded {len(rsciio.IO_PLUGINS)} rsciio IO plugins")
    except Exception as e:
        log.warning(f"Failed to load rsciio plugins: {e}")


def _is_frozen() -> bool:
    return getattr(sys, "frozen", False)


# Windows: ensure Microsoft Edge WebView2 runtime is installed before pywebview
# starts. PyWebView's EdgeChromium backend fails opaquely without it.

# Microsoft Evergreen Bootstrapper (small ~2MB downloader that pulls the full runtime)
WEBVIEW2_BOOTSTRAPPER_URL = "https://go.microsoft.com/fwlink/p/?LinkId=2124703"

# Registry GUID for the WebView2 Runtime client
WEBVIEW2_CLIENT_GUID = "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"


def _webview2_installed() -> bool:
    """Check HKLM (system) and HKCU (per-user) for a non-empty pv value."""
    if platform.system() != "Windows":
        return True

    import winreg  # stdlib, Windows-only

    subkeys = [
        # 64-bit Windows: machine-wide install lives under WOW6432Node
        (
            winreg.HKEY_LOCAL_MACHINE,
            rf"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{WEBVIEW2_CLIENT_GUID}",
        ),
        # 32-bit Windows fallback
        (
            winreg.HKEY_LOCAL_MACHINE,
            rf"SOFTWARE\Microsoft\EdgeUpdate\Clients\{WEBVIEW2_CLIENT_GUID}",
        ),
        # Per-user install
        (
            winreg.HKEY_CURRENT_USER,
            rf"SOFTWARE\Microsoft\EdgeUpdate\Clients\{WEBVIEW2_CLIENT_GUID}",
        ),
    ]

    for root, path in subkeys:
        try:
            with winreg.OpenKey(root, path) as key:
                pv, _ = winreg.QueryValueEx(key, "pv")
                if pv and pv != "0.0.0.0":
                    return True
        except OSError:
            continue
    return False


def _msgbox(text: str, title: str, style: int) -> int:
    """Native Win32 MessageBox (no GUI deps)."""
    import ctypes

    return ctypes.windll.user32.MessageBoxW(0, text, title, style)


def _ensure_webview2_runtime() -> bool:
    """
    Windows-only: download and silently install WebView2 if missing.
    Returns True if the runtime is (now) available, False if user declined or install failed.
    No-op (returns True) on non-Windows.
    """
    if platform.system() != "Windows":
        return True
    if _webview2_installed():
        return True

    log.info("WebView2 runtime not detected — prompting user to install.")

    # MB_YESNO | MB_ICONINFORMATION
    choice = _msgbox(
        "TEMseg needs the Microsoft Edge WebView2 runtime (a small Windows "
        "component) to display its interface.\n\n"
        "Click Yes to download and install it now (~2 MB download, silent install).",
        "TEMseg — One-time setup",
        0x00000004 | 0x00000040,
    )
    if choice != 6:  # IDYES = 6
        return False

    import tempfile
    import subprocess

    tmp_dir = Path(tempfile.mkdtemp(prefix="temseg_wv2_"))
    installer = tmp_dir / "MicrosoftEdgeWebview2Setup.exe"

    try:
        log.info(f"Downloading WebView2 bootstrapper to {installer}")
        req = urllib.request.Request(
            WEBVIEW2_BOOTSTRAPPER_URL,
            headers={"User-Agent": "TEMseg/1.0"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            installer.write_bytes(resp.read())

        log.info("Running WebView2 installer silently...")
        # /silent /install — no UI; bootstrapper fetches and installs the full runtime
        result = subprocess.run(
            [str(installer), "/silent", "/install"],
            timeout=600,
        )
        if result.returncode != 0:
            log.error(f"WebView2 installer exited with code {result.returncode}")
            return False

    except Exception:
        log.exception("WebView2 install failed")
        return False
    finally:
        try:
            installer.unlink(missing_ok=True)
            tmp_dir.rmdir()
        except Exception:
            pass

    return _webview2_installed()


def _bundle_dir() -> Path:
    """PyInstaller sets sys._MEIPASS to the temp extract dir."""
    if _is_frozen():
        return Path(sys._MEIPASS)
    return Path(__file__).parent


def _project_root() -> Path:
    return Path(__file__).parent


def _frontend_out_dir() -> Path:
    if _is_frozen():
        return _bundle_dir() / "frontend_out"
    return _project_root() / "frontend" / "out"


def _weights_dir() -> Path:
    """Where weights live at runtime — resolved by backend settings
    (env override > platform app-data dir), same in dev and frozen builds."""
    from app.models.helpers.settings import settings

    return settings.WEIGHTS_DIR


def _manifest_path() -> Path:
    if _is_frozen():
        return _bundle_dir() / "weight_manifest.json"
    return _project_root() / "weight_manifest.json"


def _backend_src_dir() -> Path:
    """Backend source dir — needed for CWD so relative 'sessions/' works."""
    if _is_frozen():
        return _bundle_dir() / "backend_src"
    return _project_root() / "backend" / "src"


def _load_manifest() -> list[dict]:
    mp = _manifest_path()
    log.info("Weight manifest found @ %s", mp)
    if not mp.exists():
        log.warning(f"Weight manifest not found at {mp}")
        return []
    with open(mp) as f:
        data = json.load(f)
    return data.get("weights", [])


def _verify_sha256(filepath: Path, expected: str) -> bool:
    """Check SHA256 of a downloaded file. Skip if placeholder."""
    if not expected or "PLACEHOLDER" in expected.upper():
        return True  # can't verify, assume ok
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        log.warning(
            "SHA256 mismatch for %s: expected %s, got %s",
            filepath.name, expected, actual,
        )
    return actual == expected


def _platform_matches(platforms: list) -> bool:
    """Entry platforms filter: darwin-arm64 = this machine (arm Mac)."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _coreml_variant_enabled() -> bool:
    """CoreML build ships a coreml_variant.marker in the bundle; classic
    builds and dev resolve from TEMSEG_COREML (default: on)."""
    root = _bundle_dir() if _is_frozen() else _project_root()
    if (root / "coreml_variant.marker").exists():
        return True
    return os.environ.get("TEMSEG_COREML", "").strip().lower() not in ("0", "false", "no")


def _entry_applies(entry: dict) -> bool:
    platforms = entry.get("platforms")
    if platforms and not _platform_matches(platforms):
        return False
    if entry.get("variant") == "coreml" and not _coreml_variant_enabled():
        return False
    return True


def _apply_variant_defaults() -> None:
    """Frozen classic builds force the classic pipeline (no coremltools
    bundled); the CoreML build ships a marker file instead."""
    if _is_frozen() and not (_bundle_dir() / "coreml_variant.marker").exists():
        os.environ.setdefault("TEMSEG_COREML", "0")


def check_and_download_weights(progress_callback=None) -> tuple[bool, str]:
    """
    Ensure all weights exist in the weights dir.
    Returns (success: bool, message: str).

    progress_callback(filename, percent) is called during download.
    """
    weights_dir = _weights_dir()
    weights_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest()
    if not manifest:
        # No manifest — check if weights exist anyway (dev mode)
        expected = [
            "best12x.onnx",
            "sam_vit_b_01ec64.pth",
            "maskrcnn_best_model.pth",
            "maskrcnn_best_model_synthetic.pth",
        ]
        missing = [f for f in expected if not (weights_dir / f).exists()]
        if missing:
            return False, f"Missing weights and no manifest to download from: {missing}"
        return True, "All weights present"

    for entry in manifest:
        filename = entry["filename"]
        if not _entry_applies(entry):
            log.info(f"Skipping {filename} (variant/platform)")
            continue
        url = entry.get("url", "")
        sha256 = entry.get("sha256", "")
        dest = weights_dir / filename
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists():
            if _verify_sha256(dest, sha256):
                log.info(f"Weight OK: {filename}")
                continue
            else:
                log.warning(f"Checksum mismatch for {filename}, re-downloading")
                dest.unlink()

        if not url or "PLACEHOLDER" in url.upper():
            return False, (
                f"Weight '{filename}' is missing and no download URL is configured.\n"
                f"Please place it manually in:\n{weights_dir}"
            )

        log.info(f"Downloading {filename} from {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "TEMseg/1.0"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                tmp = dest.with_suffix(".tmp")
                with open(tmp, "wb") as out:
                    while True:
                        chunk = resp.read(1024 * 256)  # 256KB chunks
                        if not chunk:
                            break
                        out.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback and total > 0:
                            pct = int(downloaded / total * 100)
                            progress_callback(filename, pct)

                # verify
                if not _verify_sha256(tmp, sha256):
                    tmp.unlink()
                    return False, f"Checksum verification failed for {filename}"

                tmp.rename(dest)
                log.info(f"Downloaded: {filename}")

        except Exception as e:
            return False, f"Failed to download {filename}: {e}"

    return True, "All weights ready"


FRONTEND_PORT = 3001


def _make_spa_handler(directory: str):
    """Create an SPA handler class bound to the given directory."""

    class SPAHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)

        def do_GET(self):
            requested = Path(directory) / self.path.lstrip("/")
            if not requested.exists() or requested.is_dir():
                self.path = "/index.html"
            super().do_GET()

        def log_message(self, format, *args):
            pass

    return SPAHandler


def start_frontend_server():
    out_dir = _frontend_out_dir()
    if not out_dir.exists():
        log.error(f"Frontend build not found at {out_dir}")
        return
    handler = _make_spa_handler(str(out_dir))
    server = HTTPServer(("localhost", FRONTEND_PORT), handler)
    log.info(f"Frontend serving on :{FRONTEND_PORT}")
    server.serve_forever()


# Backend (in-process via uvicorn)

BACKEND_PORT = 8080


def start_backend_server():
    """Run the FastAPI app via uvicorn in the current thread."""
    import uvicorn

    # Set CWD so that relative paths (sessions/) resolve correctly
    backend_src = _backend_src_dir()
    os.chdir(str(backend_src))

    # If frozen, tell settings.py where weights are
    if _is_frozen():
        os.environ["TEMSEG_WEIGHTS_DIR"] = str(_weights_dir())

    # Add backend src to sys.path so imports work
    src_str = str(backend_src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=BACKEND_PORT,
        log_level="info",
    )


# Wait helpers


def wait_for_server(port: int, timeout: int = 60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://localhost:{port}", timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    return False


# PyWebView JS API bridge


class Api:
    """Exposed to JS as window.pywebview.api"""

    def __init__(self, window_ref):
        self._window = window_ref

    def export_zip(
        self,
        session_id: str,
        items: list[str],
        training_samples: list[dict] | None = None,
    ) -> dict:
        """
        Called from JS: window.pywebview.api.export_zip(sessionId, items, trainingSamples)
        Fetches ZIP from backend, opens native Save dialog, writes to disk.
        """
        try:
            url = f"http://localhost:{BACKEND_PORT}/export/{session_id}"
            payload = json.dumps(
                {"items": items, "training_samples": training_samples}
            ).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                zip_bytes = resp.read()

            is_training_only = items == ["training_data"]
            save_filename = (
                f"training_data_{session_id[:8]}.zip"
                if is_training_only
                else f"temseg_export_{session_id[:8]}.zip"
            )
            save_path = self._window.create_file_dialog(
                webview.SAVE_DIALOG,
                save_filename=save_filename,
                file_types=("ZIP archive (*.zip)",),
            )

            if not save_path:
                return {"success": False, "error": "cancelled"}

            dest = save_path if isinstance(save_path, str) else save_path[0]
            Path(dest).write_bytes(zip_bytes)
            return {"success": True, "path": dest}

        except Exception as e:
            return {"success": False, "error": str(e)}


# Loading window (shown during weight download / model init)

LOADING_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0f0f12;
    color: #c8c6c1;
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100vh;
    overflow: hidden;
  }
  .bg-grid {
    position: absolute; inset: 0;
    background-image:
      linear-gradient(rgba(126,232,162,0.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(126,232,162,0.04) 1px, transparent 1px);
    background-size: 32px 32px;
    pointer-events: none;
  }
  .container {
    position: relative;
    text-align: center;
    max-width: 440px;
    padding: 2rem;
    z-index: 1;
  }
  .brand {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.2rem;
  }
  .logo-mark {
    width: 28px; height: 28px;
    border: 2px solid #7ee8a2;
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    color: #7ee8a2;
    font-weight: 700;
    font-size: 13px;
  }
  h1 { font-size: 1.5rem; font-weight: 600; color: #e8e6e1; letter-spacing: -0.02em; }
  .spinner-wrap {
    display: flex; align-items: center; justify-content: center;
    gap: 6px; margin: 1.2rem 0 0.6rem;
  }
  .dot {
    width: 7px; height: 7px;
    background: #7ee8a2;
    border-radius: 50%;
    animation: bounce 1.2s infinite ease-in-out;
  }
  .dot:nth-child(2) { animation-delay: 0.15s; opacity: 0.7; }
  .dot:nth-child(3) { animation-delay: 0.3s; opacity: 0.5; }
  @keyframes bounce {
    0%, 80%, 100% { transform: translateY(0); }
    40% { transform: translateY(-10px); }
  }
  .status {
    font-size: 0.9rem;
    color: #888;
    margin-top: 0.4rem;
    min-height: 1.5em;
  }
  .progress-outer {
    width: 100%; height: 5px;
    background: #1a1a1e;
    border-radius: 3px;
    margin-top: 0.9rem;
    overflow: hidden;
    border: 1px solid #1f1f23;
  }
  .progress-inner {
    height: 100%; width: 0%;
    background: linear-gradient(90deg, #7ee8a2, #5bc48a);
    border-radius: 3px;
    transition: width 0.35s ease;
  }
  .steps {
    display: flex;
    justify-content: space-between;
    margin-top: 0.6rem;
    font-size: 0.7rem;
    color: #444;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .steps .active { color: #7ee8a2; }
  .error {
    color: #e87e7e;
    font-size: 0.85rem;
    margin-top: 1rem;
    white-space: pre-wrap;
    line-height: 1.4;
    background: rgba(232,126,126,0.06);
    padding: 10px 12px;
    border-radius: 6px;
    border: 1px solid rgba(232,126,126,0.15);
  }
  .footer {
    margin-top: 1.2rem;
    font-size: 0.7rem;
    color: #333;
  }
</style>
</head>
<body>
<div class="bg-grid"></div>
<div class="container">
  <div class="brand">
    <div class="logo-mark">T</div>
    <h1>TEMseg</h1>
  </div>
  <div class="spinner-wrap">
    <div class="dot"></div>
    <div class="dot"></div>
    <div class="dot"></div>
  </div>
  <div class="status" id="status">Checking model weights…</div>
  <div class="progress-outer"><div class="progress-inner" id="bar"></div></div>
  <div class="steps" id="steps">
    <span id="step-weights">Weights</span>
    <span id="step-backend">Backend</span>
    <span id="step-models">Models</span>
    <span id="step-ready">Ready</span>
  </div>
  <div class="error" id="error"></div>
  <div class="footer">First launch may take a few minutes</div>
</div>
<script>
  function setStatus(msg) {
    document.getElementById("status").textContent = msg;
  }
  function setProgress(pct) {
    document.getElementById("bar").style.width = pct + "%";
    if (pct < 30) { setStep("weights"); }
    else if (pct < 60) { setStep("backend"); }
    else if (pct < 90) { setStep("models"); }
    else { setStep("ready"); }
  }
  function setStep(id) {
    ["weights","backend","models","ready"].forEach(s => {
      document.getElementById("step-"+s).classList.toggle("active", s === id);
    });
  }
  function setError(msg) {
    document.getElementById("error").textContent = msg;
    document.querySelector(".spinner-wrap").style.display = "none";
  }
</script>
</body>
</html>
"""


# Main


def main():
    log.info("Starting TEMseg...")

    _apply_variant_defaults()

    headless = os.environ.get("TEMSEG_HEADLESS") == "1"

    # Phase 0: Windows-only — ensure WebView2 runtime exists
    # PyWebView's EdgeChromium backend silently fails without it on fresh
    # Windows installs. Skipped in headless mode (no GUI).
    if not headless and not _ensure_webview2_runtime():
        _msgbox(
            "TEMseg cannot start without the Microsoft Edge WebView2 runtime.\n\n"
            "Please install it manually from:\n"
            "https://developer.microsoft.com/microsoft-edge/webview2/",
            "TEMseg — Setup incomplete",
            0x00000010,  # MB_ICONERROR
        )
        sys.exit(1)

    def _run_startup(ui_eval) -> bool:
        """Start weights check + backend + frontend servers.

        ui_eval(js) updates the loading UI in GUI mode and is a no-op in
        headless mode. Returns True once everything is ready.
        """
        log.info("Starting up: weights check, then backend + frontend servers")
        try:
            # Check / download weights
            def on_progress(filename, pct):
                ui_eval(
                    f'setStatus("Downloading {filename}… {pct}%"); setProgress({pct});'
                )

            ok, msg = check_and_download_weights(progress_callback=on_progress)
            if not ok:
                log.error("Setup incomplete: %s", msg)
                ui_eval(f"setError({json.dumps(msg)}); setStatus('Setup incomplete');")
                return False

            ui_eval('setStatus("Starting backend…"); setProgress(0);')

            # Phase 2: Start servers
            backend_thread = threading.Thread(target=start_backend_server, daemon=True)
            backend_thread.start()

            frontend_thread = threading.Thread(
                target=start_frontend_server, daemon=True
            )
            frontend_thread.start()

            # Wait for backend
            ui_eval('setStatus("Loading models…");')
            if not wait_for_server(BACKEND_PORT, timeout=120):
                log.error("Backend failed to start.")
                ui_eval('setError("Backend failed to start. Check console for errors.");')
                return False

            ui_eval('setStatus("Almost ready…"); setProgress(80);')
            if not wait_for_server(FRONTEND_PORT, timeout=15):
                log.error("Frontend server failed to start.")
                ui_eval('setError("Frontend server failed to start.");')
                return False

            ui_eval('setProgress(100); setStatus("Ready!");')
            time.sleep(0.3)
            return True

        except Exception as e:
            log.exception("Startup failed")
            ui_eval(f"setError({json.dumps(str(e))});")
            return False

    if headless:
        # No GUI: run the same startup sequence and keep serving. Used by the
        # API suite (scripts/mac/test_api.sh) and the nightly cron, which
        # have no WindowServer/Aqua session to create a pywebview window.
        log.info("Headless mode (TEMSEG_HEADLESS=1) — skipping GUI")
        _run_startup(lambda js: log.debug("ui: %s", js))
        while True:
            time.sleep(3600)
        return

    # GUI mode: loading window + weight check
    loading_window = webview.create_window(
        title="TEMseg — Loading",
        html=LOADING_HTML,
        width=480,
        height=260,
        resizable=False,
    )

    def startup():
        """Runs after the loading window is visible."""
        if not _run_startup(loading_window.evaluate_js):
            return  # keep loading window open so the user can read the error

        # Phase 3: Open main window, close loader
        main_window = webview.create_window(
            title="TEMseg",
            url=f"http://localhost:{FRONTEND_PORT}/workspace",
            width=1400,
            height=900,
            min_size=(900, 600),
        )
        main_window.expose(Api(main_window).export_zip)

        # Close loading window after main is created
        loading_window.destroy()

    # Run startup after window is shown
    webview.start(startup, debug=False)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()
