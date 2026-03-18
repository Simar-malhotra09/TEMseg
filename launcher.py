"""
launcher.py -- minimal PyWebView proof of concept.
Starts the FastAPI backend, serves the static frontend, opens a desktop window.
"""

import subprocess
import sys
import time
import threading
import urllib.request
import urllib.error
import webview
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

ROOT = Path(__file__).parent
FRONTEND_OUT = ROOT / "frontend" / "out"
BACKEND_DIR = ROOT / "backend" / "src"
BACKEND_PORT = 8000
FRONTEND_PORT = 3001  # different from dev port to avoid confusion




class SPAHandler(SimpleHTTPRequestHandler):
    """Serve index.html for any path that doesn't match a real file.
    Required for Next.js static export with client-side routing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_OUT), **kwargs)

    def do_GET(self):
        # check if the requested path maps to a real file in out/
        requested = Path(FRONTEND_OUT) / self.path.lstrip("/")
        if not requested.exists() or requested.is_dir():
            # fall back to index.html — let the client router handle it
            self.path = "/index.html"
        super().do_GET()

    def log_message(self, format, *args):
        pass  # suppress per-request logs


def serve_frontend():
    server = HTTPServer(("localhost", FRONTEND_PORT), SPAHandler)
    server.serve_forever()


def start_backend():
    """Start FastAPI backend as a subprocess."""
    venv_python = ROOT / "backend" / "env312" / "Scripts" / "python.exe"
    if not venv_python.exists():
        # Mac/Linux path
        venv_python = ROOT / "backend" / "env312" / "bin" / "python"
    if not venv_python.exists():
        # fall back to current interpreter
        venv_python = Path(sys.executable)

    cmd = [
        str(venv_python),
        "-m",
        "uvicorn",
        "app.api.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(BACKEND_PORT),
    ]
    return subprocess.Popen(cmd, cwd=BACKEND_DIR)


def wait_for_server(port: int, timeout: int = 30):
    """Poll until server responds or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://localhost:{port}", timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    # start backend
    print("[launcher] Starting backend...")
    backend = start_backend()

    # start frontend static server in background thread
    print("[launcher] Starting frontend static server...")
    t = threading.Thread(target=serve_frontend, daemon=True)
    t.start()

    # wait for both
    print("[launcher] Waiting for backend...")
    if not wait_for_server(BACKEND_PORT):
        print("[launcher] ERROR: backend did not start in time")
        backend.terminate()
        sys.exit(1)

    print("[launcher] Waiting for frontend...")
    if not wait_for_server(FRONTEND_PORT):
        print("[launcher] ERROR: frontend did not start in time")
        backend.terminate()
        sys.exit(1)

    print("[launcher] Both ready — opening window")

    LAUNCH_PATH= f"http://localhost:{FRONTEND_PORT}/workspace"
    print(f"[launcher] Opening window at {LAUNCH_PATH}")
    window = webview.create_window(
        title="TEMseg",
        url=LAUNCH_PATH,
        width=1400,
        height=900,
        min_size=(900, 600),
    )
    webview.start()

    # window closed — kill backend
    print("[launcher] Window closed, stopping backend...")
    backend.terminate()


if __name__ == "__main__":
    main()
