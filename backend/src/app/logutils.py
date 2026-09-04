"""
logutils — the single place TEMseg configures logging.

Three things live here:

1. get_logger(component, sub=None)
   Every module logs through this. The component string doubles as the
   bracket tag rendered on every line:   [YoloSAM.SAM] encode=41ms

2. Timer
   Time-taken as a first-class citizen. Per-step lines go to DEBUG
   (toggle with TEMSEG_LOG_LEVEL=DEBUG), the cumulative summary always
   goes to INFO:
       with Timer(log, "segment") as t:
           with t.step("load"):      ...
           with t.step("inference"): ...
           t.field("detections", 12)
   # INFO: [segment] segment | load=41ms inference=1.12s | total=1.21s detections=12

3. init_logging()
   One idempotent setup call (launcher.py and app.api.main both call it;
   second call is a no-op). Console: "[component] message".
   File: full timestamped format at <default_log_dir()>/temseg.log.

Level resolution: explicit arg > TEMSEG_LOG_LEVEL env > DEBUG when
running unfrozen (dev) / INFO when frozen (packaged). The level applies
to the "temseg" namespace only — root stays at INFO so chatty
third-party libs (asyncio, matplotlib, numba) don't flood the console.
"""

import contextvars
import json
import logging
import os
import sys
import time
from collections import deque
from contextlib import contextmanager
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

_NAMESPACE = "temseg"
_LOG_FILE_NAME = "temseg.log"
_UI_LOG_FILE_NAME = "ui.log"
_CONSOLE_FORMAT = "[%(component)s] %(req)s%(message)s"
_FILE_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)s [%(component)s] %(req)s%(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_handler_marker = "_temseg_handler"
_LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_LOG_BACKUPS = 5

# per-request correlation id; every log line made while a request is being
# served carries it so interleaved requests can be untangled
_req_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "temseg_req_id", default=None
)

# user-facing event stream: JSON lines in ui.log + in-memory ring buffer
# that /ui/events serves to the frontend status panel
_ui_events: deque[dict] = deque(maxlen=200)
_ui_seq = 0

_logger_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def set_request_id(rid: str | None) -> None:
    _req_id.set(rid)


class _ComponentFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, "component"):
            record.component = record.name
        record.req = f"req={_req_id.get()} " if _req_id.get() else ""
        return super().format(record)


class _ComponentAdapter(logging.LoggerAdapter):
    """Injects the bracket tag into every record from this component."""

    def process(self, msg, kwargs):
        extra = kwargs.setdefault("extra", {})
        extra["component"] = self.extra["component"]
        return msg, kwargs


def get_logger(component: str, sub: str | None = None) -> _ComponentAdapter:
    tag = f"{component}.{sub}" if sub else component
    name = f"{_NAMESPACE}.{component.lower()}"
    if sub:
        name += f".{sub.lower()}"
    return _ComponentAdapter(logging.getLogger(name), {"component": tag})


def fmt_duration(seconds: float) -> str:
    """Coherent units everywhere: <1s -> '41.2ms', >=1s -> '1.21s'."""
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    return f"{seconds:.2f}s"


class Timer:
    """
    Measures a multi-step operation. Two usage styles:

        with Timer(log, "segment") as t:      # summary logged on exit
            with t.step("inference"):  ...    # per-step line at DEBUG
            t.field("detections", 12)         # extra key=value in summary

    or imperative (for flat function bodies where a `with` block would
    re-indent everything):

        t = Timer(log, "from-boxes")          # starts immediately
        ...                                   # early exits just skip the summary
        t.field("proposals", 5)
        elapsed = t.elapsed
        t.stop()                              # emits the summary once

    On exception inside a `with` block: FastAPI's HTTPException is request
    flow control, logged at DEBUG ("aborted"); anything else is a real
    failure, logged at WARNING, before the exception propagates.
    """

    def __init__(self, log, name: str, summary_level: int = logging.INFO):
        self._log = log
        self._name = name
        self._summary_level = summary_level
        self._steps: list[tuple[str, float]] = []
        self._fields: list[tuple[str, object]] = []
        self._t0 = time.perf_counter()
        self._stopped = False

    def __enter__(self) -> "Timer":
        return self

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._t0

    @contextmanager
    def step(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self._steps.append((name, dt))
            self._log.debug("step=%s dt=%s", name, fmt_duration(dt))

    def field(self, key: str, value) -> None:
        """Attach `key=value` to the summary line (counts, shapes, ...)."""
        self._fields.append((key, value))

    def stop(self) -> None:
        if not self._stopped:
            self._stopped = True
            self._emit(None)

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self._stopped:
            self._stopped = True
            self._emit(exc_type, exc)
        return False

    def _emit(self, exc_type, exc=None) -> None:
        total = time.perf_counter() - self._t0
        if exc_type is not None:
            if exc_type.__name__ == "HTTPException":
                self._log.debug(
                    "%s aborted (HTTP %s) after %s",
                    self._name,
                    getattr(exc, "status_code", "?"),
                    fmt_duration(total),
                )
            else:
                self._log.warning(
                    "%s failed (%s) after %s: %s",
                    self._name,
                    exc_type.__name__,
                    fmt_duration(total),
                    exc,
                )
            return
        parts = " ".join(f"{name}={fmt_duration(dt)}" for name, dt in self._steps)
        extras = " ".join(f"{key}={value}" for key, value in self._fields)
        msg = self._name
        if parts:
            msg += f" | {parts}"
        msg += f" | total={fmt_duration(total)}"
        if extras:
            msg += f" | {extras}"
        self._log.log(self._summary_level, "%s", msg)


def default_log_dir() -> Path:
    """Writable per-user log dir (never inside the app bundle)."""
    system = sys.platform
    if system == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
        return base / "TEMseg" / "logs"
    if system == "darwin":
        return Path.home() / "Library" / "Logs" / "TEMseg"
    return Path.home() / ".local" / "state" / "TEMseg" / "logs"


def _resolve_level(level) -> int:
    if level is not None:
        return level
    env = os.environ.get("TEMSEG_LOG_LEVEL", "").lower()
    if env in _logger_levels:
        return _logger_levels[env]
    return logging.INFO if getattr(sys, "frozen", False) else logging.DEBUG


def init_logging(log_dir: Path | None = None, level: int | None = None) -> Path:
    """
    Configure console + file logging once for the whole process.
    Idempotent: later calls (e.g. app.api.main after launcher.py) no-op.
    Returns the log file path.
    """
    root = logging.getLogger()
    if any(getattr(h, _handler_marker, False) for h in root.handlers):
        existing = default_log_dir() / _LOG_FILE_NAME if log_dir is None else log_dir / _LOG_FILE_NAME
        return existing

    app_level = _resolve_level(level)
    console_formatter = _ComponentFormatter(_CONSOLE_FORMAT)
    file_formatter = _ComponentFormatter(_FILE_FORMAT, datefmt=_DATEFMT)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(console_formatter)

    if log_dir is None:
        log_dir = default_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / _LOG_FILE_NAME
    file_handler = RotatingFileHandler(
        log_path, maxBytes=_LOG_MAX_BYTES, backupCount=_LOG_BACKUPS
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    for handler in (console, file_handler):
        setattr(handler, _handler_marker, True)
        root.addHandler(handler)
    root.setLevel(logging.INFO)

    logging.getLogger(_NAMESPACE).setLevel(app_level)

    _init_ui_stream(log_dir)
    return log_path


def _init_ui_stream(log_dir: Path) -> None:
    """User-facing event stream: rotated JSON-lines file, no console noise."""
    ui_logger = logging.getLogger(f"{_NAMESPACE}.ui")
    if any(getattr(h, _handler_marker, False) for h in ui_logger.handlers):
        return
    handler = RotatingFileHandler(
        log_dir / _UI_LOG_FILE_NAME, maxBytes=_LOG_MAX_BYTES, backupCount=_LOG_BACKUPS
    )
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    setattr(handler, _handler_marker, True)
    ui_logger.addHandler(handler)
    ui_logger.propagate = False
    ui_logger.setLevel(logging.INFO)


def ui_event(
    code: str,
    message: str,
    level: str = "info",
    progress: float | None = None,
    **fields,
) -> None:
    """Emit a user-facing event: JSON line in ui.log + /ui/events ring buffer.

    These are the ONLY lines a non-technical user should ever see — plain
    language, no stack traces, no timings. Dev detail stays in temseg.log.
    """
    global _ui_seq
    _ui_seq += 1
    event = {
        "id": _ui_seq,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "level": level,
        "code": code,
        "message": message,
        "progress": progress,
        **fields,
    }
    _ui_events.append(event)
    logging.getLogger(f"{_NAMESPACE}.ui").log(
        _logger_levels.get(level, logging.INFO), json.dumps(event)
    )


def get_ui_events(since: int = 0) -> list[dict]:
    """Ring-buffer events with id > since, oldest first (for /ui/events)."""
    return [e for e in _ui_events if e["id"] > since]
