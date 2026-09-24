import json

from app.api.routers import sessions as s


def test_group_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "GROUPS_FILE", tmp_path / "groups.json")
    g = s.create_group("batch A", ["s1", "s2"])
    assert g["id"] and g["name"] == "batch A"
    assert g["session_ids"] == ["s1", "s2"]

    groups = s._read_groups()
    assert len(groups) == 1 and groups[0]["id"] == g["id"]

    # listing drops session ids whose dirs are gone
    resp = s._group_response(g)
    assert resp["session_count"] == 0


def test_rename_and_delete_session_prune_mru_and_groups(tmp_path, monkeypatch):
    monkeypatch.setattr(s, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(s, "GROUPS_FILE", tmp_path / "groups.json")
    monkeypatch.setattr(s, "MRU_K_SESSIONS_FILE", tmp_path / "mru.json")
    sd = tmp_path / "sessions" / "s1"
    sd.mkdir(parents=True)
    (sd / "metadata.json").write_text(
        json.dumps({"file_name": "img", "original_format": "tif"})
    )
    s.MRU_K_SESSIONS_FILE.write_text(json.dumps([
        {"session_id": "s1", "file_name": "img.tif"},
        {"session_id": "gone", "file_name": "x.png"},
    ]))
    s.create_group("batch", ["s1"])

    resp = s.rename_session("s1", s.RenameRequest(name="particle 01.tif"))
    assert resp["file_name"] == "particle 01"
    assert s._session_display_name("s1") == "particle 01.tif"

    s.delete_session("s1")
    assert not sd.exists()
    assert s._read_mru() == [{"session_id": "gone", "file_name": "x.png"}]
    # group lost its only session, so the record goes too
    assert s._read_groups() == []
