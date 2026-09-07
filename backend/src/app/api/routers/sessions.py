import json
import time
from pathlib import Path

from fastapi import APIRouter

from app.api.utils import SESSIONS_DIR
from app.logutils import get_logger

logger = get_logger("sessions")
router = APIRouter(tags=["sessions"])

MRU_K_SESSIONS_FILE = Path("mru_k_sessions.json")
MRU_K_VAL = 10


def _read_mru() -> list[dict]:
    try:
        return json.loads(MRU_K_SESSIONS_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def update_mru(session_id: str) -> None:
    """Record a session as just-used. Best-effort: never raises."""
    meta_path = SESSIONS_DIR / session_id / "metadata.json"
    if not meta_path.exists():
        logger.warning(f"MRU skip — no metadata.json | session={session_id}")
        return
    meta = json.loads(meta_path.read_text())
    entry = {
        "session_id": session_id,
        "file_name": f"{meta.get('file_name', '')}.{meta.get('original_format', '')}",
        "last_updated_at": time.time(),
    }
    entries = [e for e in _read_mru() if e["session_id"] != session_id]
    entries.insert(0, entry)
    MRU_K_SESSIONS_FILE.write_text(json.dumps(entries[:MRU_K_VAL], indent=2))


@router.get("/sessions/recent")
def recent_sessions() -> list[dict]:
    """MRU list for the client's 'Recently opened' tab; drops deleted sessions."""
    # size is computed at read time so old MRU entries work without migration
    entries = []
    for e in _read_mru():
        d = SESSIONS_DIR / e["session_id"]
        if not d.exists():
            continue
        org_file = next(d.glob("org_*"), None)
        entries.append({
            **e,
            "file_size_bytes": org_file.stat().st_size if org_file else None,
        })
    return entries
