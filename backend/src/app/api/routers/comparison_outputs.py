import json
import re
import shutil
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.api.utils import SESSIONS_DIR, validate_session_id


class ComparisonOutputEntry(TypedDict):
    run_id: str
    model: str
    time_elapsed: float


class ComparisonView(TypedDict):
    base_model: str | None
    pane_models: list[str]


class ComparisonManifest(TypedDict):
    schema_version: int
    outputs: list[ComparisonOutputEntry]
    view: ComparisonView


class ComparisonViewRequest(BaseModel):
    base_model: str | None
    pane_models: list[str]


router = APIRouter(prefix="/sessions", tags=["comparison outputs"])
RUN_ID_PATTERN = re.compile(r"[a-f0-9]{32}\Z")


def _manifest_path(session_dir: Path) -> Path:
    return session_dir / "comparisons" / "manifest.json"


def _get_session_dir(session_id: str) -> Path:
    session_dir = SESSIONS_DIR / validate_session_id(session_id)
    if not session_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return session_dir


def _read_manifest(session_dir: Path) -> ComparisonManifest:
    path = _manifest_path(session_dir)
    if path.exists():
        return json.loads(path.read_text())
    manifest: ComparisonManifest = {
        "schema_version": 1,
        "outputs": [],
        "view": {"base_model": None, "pane_models": []},
    }
    metadata_path = session_dir / "metadata.json"
    mask_path = session_dir / "mask.png"
    if metadata_path.exists() and mask_path.exists():
        metadata = json.loads(metadata_path.read_text())
        model = metadata.get("analysis", {}).get("model")
        if model:
            entry = _snapshot_output(session_dir, model, 0.0)
            manifest["outputs"].append(entry)
            manifest["view"] = {"base_model": model, "pane_models": [model]}
    _write_manifest(session_dir, manifest)
    return manifest


def _write_manifest(session_dir: Path, manifest: ComparisonManifest) -> None:
    path = _manifest_path(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"manifest-{uuid4().hex}.tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2))
    temporary_path.replace(path)


def _snapshot_output(
    session_dir: Path, model: str, time_elapsed: float
) -> ComparisonOutputEntry:
    comparison_dir = session_dir / "comparisons"
    runs_dir = comparison_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex
    temporary_dir = runs_dir / f".{run_id}.tmp"
    output_dir = runs_dir / run_id
    temporary_dir.mkdir()
    shutil.copy2(session_dir / "mask.png", temporary_dir / "mask.png")
    for filename in ("instances.npy", "stats.json"):
        source = session_dir / filename
        if source.exists():
            shutil.copy2(source, temporary_dir / filename)
    temporary_dir.replace(output_dir)
    return {"run_id": run_id, "model": model, "time_elapsed": time_elapsed}


def prepare_append_snapshot(session_dir: Path) -> None:
    _read_manifest(session_dir)


def save_segment_output(
    session_dir: Path, model: str, time_elapsed: float, append: bool
) -> None:
    previous_manifest = _read_manifest(session_dir)
    manifest = (
        previous_manifest
        if append
        else {
            "schema_version": 1,
            "outputs": [],
            "view": {"base_model": None, "pane_models": []},
        }
    )
    old_entries = previous_manifest["outputs"]
    replaced_entries = [entry for entry in old_entries if entry["model"] == model]
    retained_entries = [entry for entry in old_entries if entry["model"] != model]
    new_entry = _snapshot_output(session_dir, model, time_elapsed)
    if append:
        outputs = [*retained_entries, new_entry]
        previous_view = manifest["view"]
        pane_models = [item for item in previous_view["pane_models"] if item != model]
        if previous_view["base_model"] is None:
            base_model = model
        else:
            base_model = previous_view["base_model"]
        if len(pane_models) < 4:
            pane_models.append(model)
        view = {"base_model": base_model, "pane_models": pane_models}
    else:
        outputs = [new_entry]
        view = {"base_model": model, "pane_models": [model]}
        replaced_entries = old_entries
    next_manifest: ComparisonManifest = {
        "schema_version": 1,
        "outputs": outputs,
        "view": view,
    }
    _write_manifest(session_dir, next_manifest)
    live_run_ids = {entry["run_id"] for entry in outputs}
    for entry in replaced_entries:
        if entry["run_id"] not in live_run_ids:
            old_run_dir = session_dir / "comparisons" / "runs" / entry["run_id"]
            if old_run_dir.exists():
                shutil.rmtree(old_run_dir)


def _get_output_dir(session_dir: Path, run_id: str) -> Path:
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise HTTPException(status_code=400, detail="Invalid comparison output ID")
    manifest = _read_manifest(session_dir)
    if not any(entry["run_id"] == run_id for entry in manifest["outputs"]):
        raise HTTPException(status_code=404, detail="Comparison output not found")
    return session_dir / "comparisons" / "runs" / run_id


@router.get("/{session_id}/comparison-outputs")
def get_comparison_outputs(session_id: str) -> dict:
    session_dir = _get_session_dir(session_id)
    manifest = _read_manifest(session_dir)
    outputs = []
    for entry in manifest["outputs"]:
        output_dir = session_dir / "comparisons" / "runs" / entry["run_id"]
        stats_path = output_dir / "stats.json"
        outputs.append(
            {
                **entry,
                "stats": json.loads(stats_path.read_text())
                if stats_path.exists()
                else None,
                "mask_url": f"/sessions/{session_id}/comparison-outputs/{entry['run_id']}/mask",
                "labels_url": (
                    f"/sessions/{session_id}/comparison-outputs/{entry['run_id']}/labels"
                    if (output_dir / "instances.npy").is_file()
                    else None
                ),
            }
        )
    return {"outputs": outputs, "view": manifest["view"]}


@router.get("/{session_id}/comparison-outputs/{run_id}/mask")
def get_comparison_mask(session_id: str, run_id: str) -> FileResponse:
    session_dir = _get_session_dir(session_id)
    output_path = _get_output_dir(session_dir, run_id) / "mask.png"
    if not output_path.is_file():
        raise HTTPException(status_code=404, detail="Comparison mask not found")
    return FileResponse(output_path, media_type="image/png")


@router.get("/{session_id}/comparison-outputs/{run_id}/labels")
def get_comparison_labels(session_id: str, run_id: str) -> FileResponse:
    session_dir = _get_session_dir(session_id)
    output_path = _get_output_dir(session_dir, run_id) / "instances.npy"
    if not output_path.is_file():
        raise HTTPException(status_code=404, detail="Comparison labels not found")
    return FileResponse(output_path, media_type="application/octet-stream")


@router.put("/{session_id}/comparison-outputs/view")
def save_comparison_view(session_id: str, req: ComparisonViewRequest) -> dict:
    session_dir = _get_session_dir(session_id)
    if len(req.pane_models) > 4 or len(set(req.pane_models)) != len(req.pane_models):
        raise HTTPException(status_code=422, detail="Choose up to four unique models")
    if req.base_model is not None and req.base_model not in req.pane_models:
        raise HTTPException(
            status_code=422, detail="The base model must be in the pane selection"
        )
    manifest = _read_manifest(session_dir)
    available_models = {entry["model"] for entry in manifest["outputs"]}
    if not set(req.pane_models).issubset(available_models):
        raise HTTPException(
            status_code=422, detail="Pane selection includes an unavailable model"
        )
    manifest["view"] = {"base_model": req.base_model, "pane_models": req.pane_models}
    _write_manifest(session_dir, manifest)
    return {"view": manifest["view"]}
