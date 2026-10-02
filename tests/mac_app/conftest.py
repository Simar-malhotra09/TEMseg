from __future__ import annotations

import os
import socket
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import cast

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
API_URL = "http://127.0.0.1:8080"
FRONTEND_URL = "http://127.0.0.1:3001"
PORTS = (8080, 3001)
STARTUP_TIMEOUT_SECONDS = 180


@dataclass
class DesktopApp:
    app_path: Path
    expected_engine: str
    process: subprocess.Popen[bytes]
    client: httpx.Client
    sessions: list[str] = field(default_factory=list)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("packaged macOS app")
    group.addoption(
        "--app",
        action="store",
        type=Path,
        required=True,
        help="Path to a built .app bundle",
    )
    group.addoption(
        "--expected-engine",
        action="store",
        choices=("onnx", "coreml"),
        required=True,
        help="Inference engine the packaged app must report for YOLO",
    )
    group.addoption("--quick", action="store_true", help="Run the core API checks")
    group.addoption("--full", action="store_true", help="Include recovery and config checks")
    group.addoption(
        "--all-models",
        action="store_true",
        help="Run segmentation with the other bundled models",
    )


def assert_ports_available() -> None:
    for port in PORTS:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.5)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                pytest.fail(
                    f"Port {port} has an active listener; stop the other app before testing"
                )


def executable_in_bundle(app_path: Path) -> Path:
    executable_dir = app_path / "Contents" / "MacOS"
    executables = [
        path
        for path in executable_dir.iterdir()
        if path.is_file() and os.access(path, os.X_OK)
    ]
    if len(executables) != 1:
        pytest.fail(
            f"Expected one executable in {executable_dir}, found {executables}"
        )
    return executables[0]


def process_log_tail(log_path: Path) -> str:
    if not log_path.exists():
        return "No application log was written."
    return log_path.read_text(encoding="utf-8", errors="replace")[-4000:]


@pytest.fixture(scope="session")
def desktop_app(request: pytest.FixtureRequest) -> Iterator[DesktopApp]:
    app_argument = cast(Path, request.config.getoption("--app"))
    app_path = (
        (Path.cwd() / app_argument).resolve()
        if not app_argument.is_absolute()
        else app_argument
    )
    if not app_path.is_dir() or app_path.suffix != ".app":
        pytest.fail(f"--app must point to an existing .app bundle: {app_path}")

    expected_engine = cast(str, request.config.getoption("--expected-engine"))
    executable = executable_in_bundle(app_path)
    assert_ports_available()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = REPO_ROOT / "build" / "mac_api_tests" / f"{run_id}_{os.getpid()}"
    log_dir.mkdir(parents=True, exist_ok=False)
    log_path = log_dir / "app.log"
    log_file = log_path.open("wb")

    process_environment = os.environ.copy()
    process_environment["TEMSEG_HEADLESS"] = "1"
    process = subprocess.Popen(
        [str(executable)],
        cwd=REPO_ROOT,
        env=process_environment,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    client = httpx.Client(
        base_url=API_URL,
        timeout=httpx.Timeout(180.0, connect=3.0),
    )
    app = DesktopApp(
        app_path=app_path,
        expected_engine=expected_engine,
        process=process,
        client=client,
    )

    try:
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if process.poll() is not None:
                pytest.fail(
                    f"App exited during startup with code {process.returncode}. "
                    f"Log: {log_path}\n{process_log_tail(log_path)}"
                )
            try:
                api_response = client.get("/")
                frontend_response = httpx.get(FRONTEND_URL, timeout=2.0)
                if api_response.status_code == 200 and frontend_response.status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(2)
        else:
            pytest.fail(
                f"App did not start its API and frontend within "
                f"{STARTUP_TIMEOUT_SECONDS}s. Log: {log_path}\n"
                f"{process_log_tail(log_path)}"
            )

        print(f"Testing {app_path}; app log: {log_path}")
        yield app
    finally:
        cleanup_errors: list[str] = []
        for session_id in app.sessions:
            response = client.delete(f"/sessions/{session_id}")
            if response.status_code != 200:
                cleanup_errors.append(
                    f"Could not remove test session {session_id}: "
                    f"HTTP {response.status_code}, {response.text}"
                )
        client.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        log_file.close()
        if cleanup_errors:
            pytest.fail("\n".join(cleanup_errors))
