"""Persistent, user-confirmed pixel examples for RF learning within a group."""

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

import cv2 as cv
import numpy as np
from pydantic import BaseModel

from app.api.instances import load_instances
from app.api.utils import Stroke, strokes_to_mask
from app.models.impls.rf_recovery import sample_labeled_features

DATA_DIR = Path("rf_batches")
SCHEMA_VERSION = 2
MAX_PIXELS_PER_EXAMPLE = 4_000
MAX_PIXELS_PER_CLASS = 50_000
PixelLabel = Literal["foreground", "background"]


class BatchExample(BaseModel):
    id: str
    group_id: str
    session_id: str
    label: PixelLabel
    created_at: str
    image_sha256: str
    image_shape: tuple[int, int]
    pixel_count: int
    feature_count: int
    source_kind: Literal["strokes", "instances", "group_masks"]
    strokes: list[Stroke]
    instance_ids: list[int]


class TeachEvent(BaseModel):
    version: int
    action: Literal["teach"]
    example: BatchExample


class RetractEvent(BaseModel):
    version: int
    action: Literal["retract"]
    example_id: str
    created_at: str


def _batch_dir(group_id: str) -> Path:
    return DATA_DIR / group_id


def is_enabled(group_id: str) -> bool:
    return (_batch_dir(group_id) / "events.jsonl").is_file()


def enable(group_id: str) -> None:
    batch_dir = _batch_dir(group_id)
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "events.jsonl").touch(exist_ok=True)


def _append_event(group_id: str, event: TeachEvent | RetractEvent) -> None:
    log_path = _batch_dir(group_id) / "events.jsonl"
    if not log_path.is_file():
        raise FileNotFoundError(f"RF learning is not enabled for group {group_id}")
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(event.model_dump_json() + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def examples(group_id: str) -> list[BatchExample]:
    log_path = _batch_dir(group_id) / "events.jsonl"
    if not log_path.is_file():
        return []
    active: dict[str, BatchExample] = {}
    for line_number, raw in enumerate(log_path.read_text().splitlines(), start=1):
        event_data: dict[str, object] = json.loads(raw)
        if event_data.get("version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported RF learning event version in {log_path}:{line_number}"
            )
        if event_data.get("action") == "teach":
            event = TeachEvent.model_validate(event_data)
            active[event.example.id] = event.example
        elif event_data.get("action") == "retract":
            event = RetractEvent.model_validate(event_data)
            active.pop(event.example_id, None)
        else:
            raise ValueError(f"Unknown RF learning action in {log_path}:{line_number}")
    return list(active.values())


def add_strokes(
    group_id: str,
    session_id: str,
    label: PixelLabel,
    strokes: list[Stroke],
    image: np.ndarray,
    image_path: Path,
) -> BatchExample:
    if not strokes:
        raise ValueError("Mark at least one region before teaching the batch")
    height, width = image.shape[:2]
    for stroke in strokes:
        if len(stroke.points) < 2 or len(stroke.points) % 2 != 0:
            raise ValueError(f"Stroke {stroke.id} has invalid point coordinates")
        if not np.isfinite(stroke.points).all() or not np.isfinite(stroke.stroke_width):
            raise ValueError(f"Stroke {stroke.id} contains a non-finite value")
        if stroke.stroke_width <= 0:
            raise ValueError(f"Stroke {stroke.id} has a non-positive width")
        if any(
            x < 0 or x >= width or y < 0 or y >= height
            for x, y in zip(stroke.points[::2], stroke.points[1::2])
        ):
            raise ValueError(f"Stroke {stroke.id} has points outside the image")

    labeled_mask = strokes_to_mask(strokes, (height, width))
    return _save_example(
        group_id,
        session_id,
        label,
        "strokes",
        strokes,
        [],
        labeled_mask,
        image,
        image_path,
    )


def add_instances(
    group_id: str,
    session_id: str,
    instance_ids: list[int],
    session_dir: Path,
    image: np.ndarray,
    image_path: Path,
) -> BatchExample:
    if not instance_ids or len(instance_ids) != len(set(instance_ids)):
        raise ValueError("Select one or more distinct saved particle IDs")
    saved = load_instances(session_dir)
    if saved is None:
        raise ValueError(f"No saved particle masks in session {session_id}")
    instances, labeled = saved
    known_ids = {int(instance["id"]) for instance in instances}
    if not set(instance_ids).issubset(known_ids):
        raise ValueError(f"Some particle IDs are not saved in session {session_id}")
    if labeled.shape != image.shape[:2]:
        raise ValueError(
            f"Saved particle mask and image differ in session {session_id}"
        )
    interior = cv.erode(
        (labeled > 0).astype(np.uint8),
        cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)),
        iterations=2,
    )
    labeled_mask = (interior > 0) & np.isin(labeled, instance_ids)
    return _save_example(
        group_id,
        session_id,
        "foreground",
        "instances",
        [],
        instance_ids,
        labeled_mask,
        image,
        image_path,
    )


def _save_example(
    group_id: str,
    session_id: str,
    label: PixelLabel,
    source_kind: Literal["strokes", "instances"],
    strokes: list[Stroke],
    instance_ids: list[int],
    labeled_mask: np.ndarray,
    image: np.ndarray,
    image_path: Path,
) -> BatchExample:
    height, width = image.shape[:2]
    pixel_count = int(labeled_mask.sum())
    with image_path.open("rb") as stream:
        image_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    for previous in examples(group_id):
        if previous.session_id != session_id:
            continue
        if previous.image_sha256 != image_sha256:
            raise ValueError(
                f"Original image changed since example {previous.id}; remove that example"
            )
        if previous.label == label:
            continue
        region_path = _batch_dir(group_id) / f"{previous.id}_region.npz"
        with np.load(region_path, allow_pickle=False) as region:
            previous_mask = region["mask"]
        if previous_mask.shape != labeled_mask.shape:
            raise ValueError(f"Stored region shape differs for example {previous.id}")
        if np.any(labeled_mask & (previous_mask > 0)):
            raise ValueError(
                f"Marked pixels overlap confirmed {previous.label} example {previous.id}"
            )
    features = sample_labeled_features(image, labeled_mask, MAX_PIXELS_PER_EXAMPLE)
    example_id = uuid4().hex
    example = BatchExample(
        id=example_id,
        group_id=group_id,
        session_id=session_id,
        label=label,
        created_at=datetime.now(timezone.utc).isoformat(),
        image_sha256=image_sha256,
        image_shape=(height, width),
        pixel_count=pixel_count,
        feature_count=len(features),
        source_kind=source_kind,
        strokes=strokes,
        instance_ids=instance_ids,
    )
    feature_path = _batch_dir(group_id) / f"{example_id}.npy"
    with feature_path.open("wb") as stream:
        np.save(stream, features, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    region_path = _batch_dir(group_id) / f"{example_id}_region.npz"
    with region_path.open("wb") as stream:
        np.savez_compressed(stream, mask=labeled_mask.astype(np.uint8))
        stream.flush()
        os.fsync(stream.fileno())
    _append_event(
        group_id,
        TeachEvent(version=SCHEMA_VERSION, action="teach", example=example),
    )
    return example


def retract(group_id: str, example_id: str) -> None:
    if example_id not in {example.id for example in examples(group_id)}:
        raise ValueError(
            f"RF learning example {example_id} is not active in group {group_id}"
        )
    _append_event(
        group_id,
        RetractEvent(
            version=SCHEMA_VERSION,
            action="retract",
            example_id=example_id,
            created_at=datetime.now(timezone.utc).isoformat(),
        ),
    )


def retract_session(group_id: str, session_id: str) -> None:
    for example in examples(group_id):
        if example.session_id == session_id:
            retract(group_id, example.id)


def remove_group(group_id: str) -> None:
    batch_dir = _batch_dir(group_id)
    if batch_dir.exists():
        shutil.rmtree(batch_dir)


def training_data(
    group_id: str, session_ids: set[str], sessions_dir: Path
) -> tuple[np.ndarray, np.ndarray]:
    class_features: dict[PixelLabel, list[np.ndarray]] = {
        "foreground": [],
        "background": [],
    }
    for example in examples(group_id):
        if example.session_id not in session_ids:
            raise ValueError(
                f"Example {example.id} refers to a session outside group {group_id}"
            )
        image_path = next((sessions_dir / example.session_id).glob("org_*"), None)
        if image_path is None:
            raise FileNotFoundError(
                f"Original image for example {example.id} is missing; remove that example"
            )
        with image_path.open("rb") as stream:
            current_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if current_hash != example.image_sha256:
            raise ValueError(
                f"Original image changed for example {example.id}; remove that example"
            )
        feature_path = _batch_dir(group_id) / f"{example.id}.npy"
        features = np.load(feature_path, allow_pickle=False)
        if features.shape != (example.feature_count, 37):
            raise ValueError(f"Invalid stored features for example {example.id}")
        if not np.isfinite(features).all():
            raise ValueError(f"Stored features are not finite for example {example.id}")
        class_features[example.label].append(features)

    missing = [label for label, rows in class_features.items() if not rows]
    if missing:
        raise ValueError(f"Teach the batch {', '.join(missing)} before RF recovery")

    sampled: dict[PixelLabel, np.ndarray] = {}
    for label, rows in class_features.items():
        combined = np.vstack(rows)
        if len(combined) > MAX_PIXELS_PER_CLASS:
            indices = np.random.default_rng(42).choice(
                len(combined), size=MAX_PIXELS_PER_CLASS, replace=False
            )
            combined = combined[indices]
        sampled[label] = combined

    features = np.vstack((sampled["background"], sampled["foreground"]))
    labels = np.concatenate(
        (
            np.zeros(len(sampled["background"]), dtype=np.int8),
            np.ones(len(sampled["foreground"]), dtype=np.int8),
        )
    )
    return features, labels
