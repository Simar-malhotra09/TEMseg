"""
launcher.py — TEMseg desktop application launcher.
"""
import os
os.environ["YOLO_AUTOINSTALL"] = "False"

import json
import hashlib
import logging
import os
import platform
import sys
import threading
import time
import urllib.request
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import webview


logging.basicConfig(
    level=logging.INFO,
    format="[launcher] %(message)s",
)
log = logging.getLogger("launcher")

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


# ---------------------------------------------------------------------------
# Windows: ensure Microsoft Edge WebView2 runtime is installed before pywebview
# starts. PyWebView's EdgeChromium backend fails opaquely without it.
# ---------------------------------------------------------------------------

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
        (winreg.HKEY_LOCAL_MACHINE,
         rf"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{WEBVIEW2_CLIENT_GUID}"),
        # 32-bit Windows fallback
        (winreg.HKEY_LOCAL_MACHINE,
         rf"SOFTWARE\Microsoft\EdgeUpdate\Clients\{WEBVIEW2_CLIENT_GUID}"),
        # Per-user install
        (winreg.HKEY_CURRENT_USER,
         rf"SOFTWARE\Microsoft\EdgeUpdate\Clients\{WEBVIEW2_CLIENT_GUID}"),
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
    """Where weights live at runtime."""
    if _is_frozen():
        system = platform.system()
        if system == "Darwin":
            return (
                Path.home() / "Library" / "Application Support" / "TEMseg" / "weights"
            )
        elif system == "Windows":
            return Path.home() / "AppData" / "Local" / "TEMseg" / "weights"
        else:
            return Path.home() / ".local" / "share" / "TEMseg" / "weights"

    return _project_root()


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
    log.info("Weight manifest found @ ", mp)
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
    return h.hexdigest() == expected


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
        expected = ["best12x.onnx", "sam_vit_b_01ec64.pth", "maskrcnn_best_model.pth"]
        missing = [f for f in expected if not (weights_dir / f).exists()]
        if missing:
            return False, f"Missing weights and no manifest to download from: {missing}"
        return True, "All weights present"

    for entry in manifest:
        filename = entry["filename"]
        url = entry.get("url", "")
        sha256 = entry.get("sha256", "")
        dest = weights_dir / filename

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


# ---------------------------------------------------------------------------
# Backend (in-process via uvicorn)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Wait helpers
# ---------------------------------------------------------------------------


def wait_for_server(port: int, timeout: int = 60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://localhost:{port}", timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    return False


# ---------------------------------------------------------------------------
# PyWebView JS API bridge
# ---------------------------------------------------------------------------


class Api:
    """Exposed to JS as window.pywebview.api"""

    def __init__(self, window_ref):
        self._window = window_ref

    def export_zip(self, session_id: str, items: list[str]) -> dict:
        """
        Called from JS: window.pywebview.api.export_zip(sessionId, items)
        Fetches ZIP from backend, opens native Save dialog, writes to disk.
        """
        try:
            url = f"http://localhost:{BACKEND_PORT}/export/{session_id}"
            payload = json.dumps({"items": items}).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                zip_bytes = resp.read()

            save_path = self._window.create_file_dialog(
                webview.SAVE_DIALOG,
                save_filename=f"temseg_export_{session_id[:8]}.zip",
                file_types=("ZIP archive (*.zip)",),
            )

            if not save_path:
                return {"success": False, "error": "cancelled"}

            dest = save_path if isinstance(save_path, str) else save_path[0]
            Path(dest).write_bytes(zip_bytes)
            return {"success": True, "path": dest}

        except Exception as e:
            return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Loading window (shown during weight download / model init)
# ---------------------------------------------------------------------------

LOADING_HTML = """
<!DOCTYPE html>
<html>
<head>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #1a1a2e;
    color: #e0e0e0;
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100vh;
  }
  .container { text-align: center; max-width: 420px; padding: 2rem; }
  h1 { font-size: 1.6rem; margin-bottom: 0.5rem; color: #ffffff; }
  .status { font-size: 0.95rem; color: #a0a0b8; margin-top: 1rem; min-height: 1.4em; }
  .progress-outer {
    width: 100%; height: 6px; background: #2a2a4a;
    border-radius: 3px; margin-top: 0.8rem; overflow: hidden;
  }
  .progress-inner {
    height: 100%; width: 0%; background: #6c63ff;
    border-radius: 3px; transition: width 0.3s ease;
  }
  .error { color: #ff6b6b; font-size: 0.9rem; margin-top: 1rem; white-space: pre-wrap; }
</style>
</head>
<body>
<div class="container">
  <h1>TEMseg</h1>
  <div class="status" id="status">Checking model weights…</div>
  <div class="progress-outer"><div class="progress-inner" id="bar"></div></div>
  <div class="error" id="error"></div>
</div>
<script>
  function setStatus(msg) {
    document.getElementById("status").textContent = msg;
  }
  function setProgress(pct) {
    document.getElementById("bar").style.width = pct + "%";
  }
  function setError(msg) {
    document.getElementById("error").textContent = msg;
  }
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    log.info("Starting TEMseg...")

    # --- Phase 0: Windows-only — ensure WebView2 runtime exists ---
    # PyWebView's EdgeChromium backend silently fails without it on fresh Windows installs.
    if not _ensure_webview2_runtime():
        _msgbox(
            "TEMseg cannot start without the Microsoft Edge WebView2 runtime.\n\n"
            "Please install it manually from:\n"
            "https://developer.microsoft.com/microsoft-edge/webview2/",
            "TEMseg — Setup incomplete",
            0x00000010,  # MB_ICONERROR
        )
        sys.exit(1)

    # --- Phase 1: Loading window + weight check ---
    loading_window = webview.create_window(
        title="TEMseg — Loading",
        html=LOADING_HTML,
        width=480,
        height=260,
        resizable=False,
    )

    def startup():
        """Runs after the loading window is visible."""
        log.info("[STARTUP]")
        try:
            # Check / download weights
            def on_progress(filename, pct):
                loading_window.evaluate_js(
                    f'setStatus("Downloading {filename}… {pct}%"); setProgress({pct});'
                )

            ok, msg = check_and_download_weights(progress_callback=on_progress)
            if not ok:
                loading_window.evaluate_js(f"setError({json.dumps(msg)});")
                loading_window.evaluate_js('setStatus("Setup incomplete");')
                return  # keep window open so user can read error

            loading_window.evaluate_js(
                'setStatus("Starting backend…"); setProgress(0);'
            )

            # --- Phase 2: Start servers ---
            backend_thread = threading.Thread(target=start_backend_server, daemon=True)
            backend_thread.start()

            frontend_thread = threading.Thread(
                target=start_frontend_server, daemon=True
            )
            frontend_thread.start()

            # Wait for backend
            loading_window.evaluate_js('setStatus("Loading models…");')
            if not wait_for_server(BACKEND_PORT, timeout=120):
                loading_window.evaluate_js(
                    'setError("Backend failed to start. Check console for errors.");'
                )
                return

            loading_window.evaluate_js('setStatus("Almost ready…"); setProgress(80);')
            if not wait_for_server(FRONTEND_PORT, timeout=15):
                loading_window.evaluate_js(
                    'setError("Frontend server failed to start.");'
                )
                return

            loading_window.evaluate_js('setProgress(100); setStatus("Ready!");')
            time.sleep(0.3)

            # --- Phase 3: Open main window, close loader ---
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

        except Exception as e:
            log.exception("Startup failed")
            loading_window.evaluate_js(f"setError({json.dumps(str(e))});")

    # Run startup after window is shown
    webview.start(startup, debug=False)


if __name__ == "__main__":
    main()
