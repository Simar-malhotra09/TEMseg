from __future__ import annotations

import io
import zipfile
from typing import cast

import httpx
import pytest

from conftest import DesktopApp, REPO_ROOT

FIXTURES = REPO_ROOT / "scripts" / "mac" / "fixtures"


def response_object(response: httpx.Response) -> dict[str, object]:
    payload = response.json()
    assert isinstance(payload, dict), f"Expected JSON object, got {payload!r}"
    return cast(dict[str, object], payload)


def upload_fixture(app: DesktopApp, filename: str) -> tuple[str, dict[str, object]]:
    fixture_path = FIXTURES / filename
    with fixture_path.open("rb") as fixture:
        response = app.client.post(
            "/images/upload",
            files={"file": (filename, fixture, "application/octet-stream")},
        )
    assert response.status_code == 200, response.text
    payload = response_object(response)
    session_id = payload.get("session_id")
    image_info = payload.get("image_info")
    assert isinstance(session_id, str) and session_id
    assert isinstance(image_info, dict)
    app.sessions.append(session_id)
    return session_id, cast(dict[str, object], image_info)


def test_health_and_models(desktop_app: DesktopApp) -> None:
    response = desktop_app.client.get("/")
    assert response.status_code == 200, response.text
    assert response_object(response).get("message") == "Hello World"

    models_response = desktop_app.client.get("/models")
    assert models_response.status_code == 200, models_response.text
    models = models_response.json()
    assert isinstance(models, list)
    assert {"YoloSAM", "YoloMaskRCNN", "MaskRCNN-Synthetic"}.issubset(models)

    frontend_response = httpx.get("http://127.0.0.1:3001/", timeout=10.0)
    assert frontend_response.status_code == 200, frontend_response.text
    assert "<html" in frontend_response.text.lower()

    workspace_response = httpx.get(
        "http://127.0.0.1:3001/workspace", timeout=10.0
    )
    assert workspace_response.status_code == 200, workspace_response.text


@pytest.mark.parametrize(
    ("filename", "expected_format"),
    (
        ("sample.emd", "emd"),
        ("sample.tif", "tif"),
        ("sample.jpg", "jpg"),
        ("mask.npy", "npy"),
    ),
)
def test_supported_image_uploads(
    desktop_app: DesktopApp,
    filename: str,
    expected_format: str,
) -> None:
    _, image_info = upload_fixture(desktop_app, filename)
    assert image_info.get("original_format") == expected_format
    assert image_info.get("image_shape")


def test_segmentation_backend_stats_and_export(desktop_app: DesktopApp) -> None:
    session_id, image_info = upload_fixture(desktop_app, "sample.emd")
    assert image_info.get("original_format") == "emd"
    assert image_info.get("pixel_size")

    for path in (f"/images/{session_id}/metadata", f"/images/{session_id}/preview"):
        response = desktop_app.client.get(path)
        assert response.status_code == 200, response.text

    segment_response = desktop_app.client.post(
        "/segment/",
        json={
            "session_id": session_id,
            "model": "YoloSAM",
            "blackout": False,
            "inverse_blackout": False,
            "regions": [],
            "colorize": True,
        },
    )
    assert segment_response.status_code == 200, segment_response.text
    segment_payload = response_object(segment_response)
    mask_url = segment_payload.get("mask_url")
    assert isinstance(mask_url, str) and mask_url

    mask_response = desktop_app.client.get(mask_url)
    assert mask_response.status_code == 200
    assert mask_response.content.startswith(b"\x89PNG\r\n\x1a\n")

    instances_response = desktop_app.client.post(f"/masks/{session_id}/instances")
    assert instances_response.status_code == 200, instances_response.text
    instances = response_object(instances_response).get("instances")
    assert isinstance(instances, list) and instances

    stats_response = desktop_app.client.get(f"/masks/{session_id}/stats")
    assert stats_response.status_code == 200, stats_response.text
    assert isinstance(stats_response.json(), dict)

    system_response = desktop_app.client.get("/system/info")
    assert system_response.status_code == 200, system_response.text
    system_info = response_object(system_response)
    backend_rows = system_info.get("models")
    assert isinstance(backend_rows, list)
    backends = {
        str(row.get("key")): row.get("engine")
        for row in cast(list[dict[str, object]], backend_rows)
    }
    assert backends.get("yolo") == desktop_app.expected_engine
    expected_sam_engine = (
        "coreml" if desktop_app.expected_engine == "coreml" else "torch"
    )
    assert backends.get("sam") == expected_sam_engine
    if desktop_app.expected_engine == "coreml":
        assert backends.get("prompt_sam") == "torch"

    export_response = desktop_app.client.post(
        f"/export/{session_id}",
        json={
            "items": [
                "original_image",
                "seg_mask_png",
                "seg_mask_npy",
                "instances_json",
                "stats_csv",
                "coco_json",
            ]
        },
    )
    assert export_response.status_code == 200, export_response.text
    with zipfile.ZipFile(io.BytesIO(export_response.content)) as archive:
        assert archive.testzip() is None
        assert {
            "original_image.png",
            "seg_mask.png",
            "seg_mask.npy",
            "instances.json",
            "stats.csv",
            "annotations.coco.json",
        }.issubset(archive.namelist())

    histogram_response = desktop_app.client.post(
        f"/export/{session_id}/hist_csv?metric=diameter"
    )
    assert histogram_response.status_code == 200, histogram_response.text
    assert histogram_response.content


def test_full_mode_recovery_and_ground_truth(
    desktop_app: DesktopApp,
    pytestconfig: pytest.Config,
) -> None:
    if not pytestconfig.getoption("--full"):
        pytest.skip("pass --full to run the extended API checks")

    session_id, _ = upload_fixture(desktop_app, "sample.emd")
    segment_response = desktop_app.client.post(
        "/segment/",
        json={
            "session_id": session_id,
            "model": "YoloSAM",
            "blackout": False,
            "inverse_blackout": False,
            "regions": [],
            "colorize": True,
        },
    )
    assert segment_response.status_code == 200, segment_response.text

    instances_response = desktop_app.client.post(
        f"/masks/{session_id}/instances"
    )
    assert instances_response.status_code == 200, instances_response.text
    instance_rows = response_object(instances_response).get("instances")
    assert isinstance(instance_rows, list) and instance_rows
    first_instance = cast(dict[str, object], instance_rows[0])
    bbox = cast(dict[str, object], first_instance["bbox"])
    x = int(cast(float, bbox["x"]) + cast(float, bbox["w"]) / 2)
    y = int(cast(float, bbox["y"]) + cast(float, bbox["h"]) / 2)

    refine_requests = (
        (
            "from-points",
            {
                "points": [[x, y]],
                "pending": [],
                "max_image_fraction": 0.05,
                "min_area": 100,
            },
        ),
        (
            "from-boxes",
            {
                "boxes": [[x - 20, y - 20, x + 20, y + 20]],
                "pending": [],
                "max_image_fraction": 0.05,
                "min_area": 100,
            },
        ),
        (
            "propose-similar",
            {
                "method": "cosine",
                "sim_threshold": 0.75,
                "max_proposals": 5,
                "nms_distance": 20,
            },
        ),
    )
    for endpoint, body in refine_requests:
        response = desktop_app.client.post(
            f"/masks/{session_id}/{endpoint}", json=body
        )
        assert response.status_code in (200, 400), response.text

    instance_id = first_instance.get("id")
    assert isinstance(instance_id, int)
    split_response = desktop_app.client.post(
        f"/masks/{session_id}/split",
        json={"instance_id": instance_id, "points": [[x, y]]},
    )
    assert split_response.status_code in (200, 400), split_response.text

    gt_path = FIXTURES / "mask.npy"
    with gt_path.open("rb") as gt_file:
        gt_response = desktop_app.client.post(
            f"/gt/{session_id}", files={"file": (gt_path.name, gt_file)}
        )
    assert gt_response.status_code == 200, gt_response.text
    compute_response = desktop_app.client.post(
        f"/gt/{session_id}/compute",
        json={"blackout": False, "inverse_blackout": False, "regions": []},
    )
    assert compute_response.status_code in (200, 400), compute_response.text

    shape_rules_response = desktop_app.client.get("/config/shape-rules")
    assert shape_rules_response.status_code == 200, shape_rules_response.text


def test_all_models_mode(
    desktop_app: DesktopApp,
    pytestconfig: pytest.Config,
) -> None:
    if not pytestconfig.getoption("--all-models"):
        pytest.skip("pass --all-models to run the additional segmentation models")

    session_id, _ = upload_fixture(desktop_app, "sample.emd")
    for model_name in ("YoloMaskRCNN", "MaskRCNN-Synthetic"):
        response = desktop_app.client.post(
            "/segment/",
            json={
                "session_id": session_id,
                "model": model_name,
                "blackout": False,
                "inverse_blackout": False,
                "regions": [],
                "colorize": False,
            },
        )
        assert response.status_code == 200, response.text
