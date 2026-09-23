import pytest
from fastapi import HTTPException

from app.api.routers.config import _preset_path


def test_preset_name_guard():
    assert _preset_path("rods v2").name == "rods v2.toml"
    for bad in ["", " ", "../evil", "a/b", "\\win", "a\\b"]:
        with pytest.raises(HTTPException):
            _preset_path(bad)
