import json
import shutil
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.utils import SESSIONS_DIR
from app.logutils import get_logger

logger = get_logger("sessions")
router = APIRouter(tags=["sessions"])

MRU_K_SESSIONS_FILE = Path("mru_k_sessions.json")
GROUPS_FILE = Path("groups.json")
MRU_K_VAL = 10


class RenameRequest(BaseModel):
    name: str


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
            # rebuilt at read time so renames show up without a re-upload
            "file_name": _session_display_name(e["session_id"]) or e["file_name"],
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


def _session_display_name(session_id: str) -> str | None:
    """Current display name (with extension) from metadata.json, or None."""
    meta_path = SESSIONS_DIR / session_id / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return None
    name = str(meta.get("file_name") or session_id)
    if meta.get("original_format"):
        name = f"{name}.{meta['original_format']}"
    return name


def _group_sessions(g: dict) -> list[dict]:
    """Session rows for a group; drops sessions whose dir no longer exists."""
    rows = []
    for sid in g["session_ids"]:
        if not (SESSIONS_DIR / sid).exists():
            continue
        rows.append({
            "session_id": sid,
            "file_name": _session_display_name(sid) or sid,
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


def _write_groups(groups: list[dict]) -> None:
    GROUPS_FILE.write_text(json.dumps(groups, indent=2))


def _clean_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name is empty")
    return name


@router.patch("/sessions/{session_id}")
def rename_session(session_id: str, req: RenameRequest):
    """Rename the display name in metadata.json; the file on disk is untouched."""
    meta_path = SESSIONS_DIR / session_id / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    meta = json.loads(meta_path.read_text())
    name = _clean_name(req.name)
    fmt = str(meta.get("original_format") or "").lower()
    if fmt and name.lower().endswith(f".{fmt}"):
        name = name[: -(len(fmt) + 1)]
    name = _clean_name(name)
    meta["file_name"] = name
    meta_path.write_text(json.dumps(meta, indent=2))
    return {"session_id": session_id, "file_name": name}


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """Remove the session dir and drop it from the MRU and every group.
    A group left with no sessions is removed too."""
    session_dir = SESSIONS_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    shutil.rmtree(session_dir)

    mru = _read_mru()
    kept_mru = [e for e in mru if e["session_id"] != session_id]
    if len(kept_mru) != len(mru):
        MRU_K_SESSIONS_FILE.write_text(json.dumps(kept_mru, indent=2))

    changed = False
    kept_groups = []
    for g in _read_groups():
        if session_id in g["session_ids"]:
            g["session_ids"] = [s for s in g["session_ids"] if s != session_id]
            changed = True
            if not g["session_ids"]:
                continue
        kept_groups.append(g)
    if changed:
        _write_groups(kept_groups)
    return {"session_id": session_id}


@router.patch("/groups/{group_id}")
def rename_group(group_id: str, req: RenameRequest):
    groups = _read_groups()
    for g in groups:
        if g["id"] == group_id:
            g["name"] = _clean_name(req.name)
            _write_groups(groups)
            return _group_response(g)
    raise HTTPException(status_code=404, detail=f"Group {group_id} not found")


@router.delete("/groups/{group_id}")
def delete_group(group_id: str):
    """Drop the group record only; its sessions stay as individual sessions."""
    groups = _read_groups()
    kept = [g for g in groups if g["id"] != group_id]
    if len(kept) == len(groups):
        raise HTTPException(status_code=404, detail=f"Group {group_id} not found")
    _write_groups(kept)
    return {"deleted": group_id}
