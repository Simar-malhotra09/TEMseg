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
