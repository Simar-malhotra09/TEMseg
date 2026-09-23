import json
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.api.utils import SESSIONS_DIR
from app.logutils import get_logger

logger = get_logger("sessions")
router = APIRouter(tags=["sessions"])

MRU_K_SESSIONS_FILE = Path("mru_k_sessions.json")
GROUPS_FILE = Path("groups.json")
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


def _read_groups() -> list[dict]:
    try:
        return json.loads(GROUPS_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def create_group(name: str, session_ids: list[str]) -> dict:
    """Record an ordered group of session ids; sessions themselves stay flat."""
    groups = _read_groups()
    group = {
        "id": uuid.uuid4().hex[:6],
        "name": name,
        "created_at": time.time(),
        "session_ids": session_ids,
    }
    groups.insert(0, group)
    GROUPS_FILE.write_text(json.dumps(groups, indent=2))
    return group


def _group_sessions(g: dict) -> list[dict]:
    """Session rows for a group; drops sessions whose dir no longer exists."""
    rows = []
    for sid in g["session_ids"]:
        d = SESSIONS_DIR / sid
        if not d.exists():
            continue
        file_name = sid
        meta_path = d / "metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError:
                meta = {}
            file_name = str(meta.get("file_name", sid))
            if meta.get("original_format"):
                file_name = f"{file_name}.{meta['original_format']}"
        rows.append({
            "session_id": sid,
            "file_name": file_name,
            "preview_url": f"/images/{sid}/preview",
        })
    return rows


def _group_response(g: dict) -> dict:
    sessions = _group_sessions(g)
    return {
        **g,
        "sessions": sessions,
        "session_count": len(sessions),
    }


@router.get("/groups")
def list_groups() -> list[dict]:
    return [_group_response(g) for g in _read_groups()]


@router.get("/groups/{group_id}")
def get_group(group_id: str):
    for g in _read_groups():
        if g["id"] == group_id:
            return _group_response(g)
    raise HTTPException(status_code=404, detail=f"Group {group_id} not found")
