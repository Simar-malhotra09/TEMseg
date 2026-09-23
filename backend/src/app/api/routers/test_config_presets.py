import pytest
from fastapi import HTTPException

from app.api.routers.config import _preset_path


def test_preset_name_guard():
    assert _preset_path("rods v2").name == "rods v2.toml"
    assert _preset_path("This is mine!").name == "This is mine!.toml"
    assert _preset_path("  padded  ").name == "padded.toml"
    for bad in ["", " ", "../evil", "a/b", "a\\b", "..", ".", "a\x00b"]:
        with pytest.raises(HTTPException):
            _preset_path(bad)
