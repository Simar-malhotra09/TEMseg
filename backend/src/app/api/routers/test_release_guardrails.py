import io
import csv
import hashlib
import importlib
import json
import zipfile

import httpx
import numpy as np
import pytest
import tifffile

from app.logutils import init_logging
from app.models.helpers.stats_export import build_stats_csv


@pytest.mark.asyncio
async def test_local_api_and_scientific_image_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    init_logging(log_dir=tmp_path)
    app = importlib.import_module("app.api.main").app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://localhost:8080"
    ) as client:
        remote = await client.get("/", headers={"Origin": "https://example.org"})
        assert remote.status_code == 403

        bad_name = await client.post(
            "/images/upload", files={"file": ("../escape.tif", b"bad", "image/tiff")}
        )
        assert bad_name.status_code == 400

        bad_image = await client.post(
            "/images/upload", files={"file": ("broken.tif", b"bad", "image/tiff")}
        )
        assert "error" in bad_image.json()
        assert not list((tmp_path / "sessions").iterdir())

        raw = io.BytesIO()
        tifffile.imwrite(raw, np.array([[1000, 2000], [3000, 4000]], dtype=np.uint16))
        upload = await client.post(
            "/images/upload",
            files={"file": ("tem16.tif", raw.getvalue(), "image/tiff")},
            headers={"Origin": "http://localhost:3001"},
        )
        assert upload.status_code == 200, upload.text
        session_id = upload.json()["session_id"]
        preview = await client.get(f"/images/{session_id}/preview")
        assert preview.status_code == 200

        invalid_scale = await client.put(
            f"/images/{session_id}/metadata",
            json={"pixel_size": 0, "pixel_unit": "nm"},
        )
        assert invalid_scale.status_code == 422
        scale = await client.put(
            f"/images/{session_id}/metadata",
            json={
                "pixel_size": 0.5,
                "pixel_unit": "nm",
                "calibration_source": "scale_bar",
                "scale_bar_pixels": 100,
                "scale_bar_length": 50,
            },
        )
        assert scale.status_code == 200, scale.text
        assert scale.json()["metadata"]["calibration"]["source"] == "scale_bar"

        export = await client.post(
            f"/export/{session_id}", json={"items": ["original_image"]}
        )
        assert export.status_code == 200, export.text
        with zipfile.ZipFile(io.BytesIO(export.content)) as archive:
            assert archive.read("source_image.tif") == raw.getvalue()
            manifest = json.loads(archive.read("analysis_manifest.json"))
        assert manifest["calibration"]["scale_bar_pixels"] == 100
        assert manifest["source_sha256"] == hashlib.sha256(raw.getvalue()).hexdigest()

        labeled = np.array([[0, 1], [1, 0]], dtype=np.uint16)
        session_dir = tmp_path / "sessions" / session_id
        np.save(session_dir / "instances.npy", labeled)
        mask_bytes = io.BytesIO()
        np.save(mask_bytes, labeled)
        ground_truth = await client.post(
            f"/gt/{session_id}",
            files={"file": ("same-mask.npy", mask_bytes.getvalue())},
        )
        assert ground_truth.status_code == 200
        score = await client.post(
            f"/gt/{session_id}/compute",
            json={"blackout": False, "inverse_blackout": False, "regions": []},
        )
        assert score.status_code == 200
        assert score.json()["scores"]["iou"] > 0.9999


def test_stats_csv_quotes_shape_labels_and_blocks_formulas():
    particle = {
        "id": 1,
        "area_px": 12.0,
        "diameter_px": 3.9,
        "perimeter_px": 13.0,
        "circularity": 0.9,
        "aspect_ratio": 1.1,
        "shape": '=HYPERLINK("https://example.org","x")',
    }
    rows = list(
        csv.reader(io.StringIO(build_stats_csv({"particles": [particle]}).decode()))
    )
    assert rows[1][8] == '\'=HYPERLINK("https://example.org","x")'
    assert len(rows[1]) == len(rows[0])
