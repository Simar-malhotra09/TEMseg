"""
start.py — starts both the FastAPI backend and Next.js frontend.

Usage:
  python start.py
  python start.py --port 8080 --frontend-port 3000
  python start.py --no-reload
  python start.py --backend-only
  python start.py --frontend-only
"""

import argparse
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


def stream_output(proc: subprocess.Popen, prefix: str):
    """Stream a process's stdout to console with a label prefix."""
    for line in iter(proc.stdout.readline, b""):
        print(f"  [{prefix}] {line.decode(errors='replace').rstrip()}", flush=True)


def find_frontend_dir() -> Path | None:
    root = Path(__file__).parent
    for candidate in ["../frontend", "frontend"]:
        p = (root / candidate).resolve()
        if (p / "package.json").exists():
            return p
    return None


def npm_cmd() -> str:
    """On Windows npm is npm.cmd, elsewhere it's npm."""
    return "npm.cmd" if platform.system() == "Windows" else "npm"


def cleanup(processes: list[subprocess.Popen]):
    """Terminate all child processes gracefully, then force-kill if needed."""
    for proc in processes:
        if proc.poll() is None:
            if platform.system() == "Windows":
                proc.terminate()
            else:
                proc.send_signal(signal.SIGTERM)
    time.sleep(1)
    for proc in processes:
        if proc.poll() is None:
            proc.kill()


def _wait_for_backend(port: int, timeout: int = 30):
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://localhost:{port}/models", timeout=1)
            print("[start] Backend ready.")
            return
        except Exception:
            time.sleep(0.5)
    print("[start] WARNING: backend did not respond in time, starting frontend anyway.")


def main():
    parser = argparse.ArgumentParser(description="Start TEM seg — backend + frontend")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--frontend-port", type=int, default=3000)
    parser.add_argument("--no-reload", action="store_true")
    parser.add_argument("--backend-only", action="store_true")
    parser.add_argument("--frontend-only", action="store_true")
    args = parser.parse_args()

    processes: list[subprocess.Popen] = []

    # ── backend ───────────────────────────────────────────────────────────────
    if not args.frontend_only:
        backend_cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "app.api.main:app",
            "--host",
            args.host,
            "--port",
            str(args.port),
        ]
        if not args.no_reload:
            backend_cmd.append("--reload")

        print(f"[backend]  Starting on http://localhost:{args.port}")
        backend = subprocess.Popen(
            backend_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=Path(__file__).parent / "src",
        )
        processes.append(backend)
        threading.Thread(
            target=stream_output, args=(backend, "backend"), daemon=True
        ).start()

        # wait for backend to signal ready before starting frontend
        print("[start] Waiting for backend to be ready...")
        _wait_for_backend(args.port)

    # ── frontend ──────────────────────────────────────────────────────────────
    if not args.backend_only:
        frontend_dir = find_frontend_dir()

        if frontend_dir is None:
            print("[frontend] WARNING: could not find frontend directory — skipping.")
            print(
                "[frontend] Expected a 'frontend/' or 'web/' folder with package.json."
            )
        else:
            print(f"[frontend] Found at {frontend_dir}")

            # auto-install node_modules on first run
            if not (frontend_dir / "node_modules").exists():
                print(
                    "[frontend] node_modules missing — running npm install (first time only)..."
                )
                result = subprocess.run([npm_cmd(), "install"], cwd=frontend_dir)
                if result.returncode != 0:
                    print("[frontend] ERROR: npm install failed. Is Node.js installed?")
                    cleanup(processes)
                    sys.exit(1)
                print("[frontend] npm install complete.")

            frontend_env = {**os.environ, "PORT": str(args.frontend_port)}

            print("\n")
            print('*'*100)
            print(f"[frontend] Starting on http://localhost:{args.frontend_port}/workspace")
            print('*'*100)
            print("\n")

            frontend = subprocess.Popen(
                [npm_cmd(), "run", "dev"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=frontend_dir,
                env=frontend_env,
            )
            processes.append(frontend)
            threading.Thread(
                target=stream_output, args=(frontend, "frontend"), daemon=True
            ).start()

    if not processes:
        print(
            "Nothing started — check your setup or use --backend-only / --frontend-only."
        )
        sys.exit(1)

    print("\n  Both servers running. Press Ctrl+C to stop all.\n")

    # ── watch — exit if either process dies unexpectedly ─────────────────────
    try:
        while True:
            for proc in processes:
                if proc.poll() is not None:
                    print(
                        f"\n  A server exited unexpectedly (code {proc.returncode}). Shutting everything down."
                    )
                    cleanup(processes)
                    sys.exit(1)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Ctrl+C received — stopping all servers...")
        cleanup(processes)
        print("  Done.")


if __name__ == "__main__":
    main()
