
import pytest
import time
from fastapi.testclient import TestClient
from pathlib import Path
from app.api.live_models import AvailableModels
from app.api.main import app
from app.models.base_model import ModelConfig
from app.models.impls.yolosam import YoloSam
from dataclasses import dataclass
import cv2 as cv
import numpy as np
import torch
import logging
import time
from typing import List
from fastapi import APIRouter
from typing import Dict, Any
from pathlib import Path 
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from app.models.base_model import Model, SegmentationResult, ModelConfig
from app.models.helpers.config import nano_config, house_config

# @pytest.fixture(scope="session", autouse=True)
# def setup_app():
#     with TestClient(app) as c:
#         yield c 


# consts
SESSIONS_DIR = Path("sessions")
N_IMAGES = 5

def get_sessions(n: int):
    sessions = [p for p in SESSIONS_DIR.iterdir() if p.is_dir()]
    if len(sessions) < n:
        pytest.exit(f"Not enough session dirs. Needed {n}, found {len(sessions)}")
    return sessions[:n]


def load_image(image_path: Path) -> np.ndarray:
    if image_path.suffix == ".npy":
        img = np.load(image_path)
    else:
        img = cv.imread(str(image_path), cv.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # normalize to (H, W, 3) uint8
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)       # grayscale → RGB
    elif img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))        # (C,H,W) → (H,W,C)
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)           # (H,W,1) → (H,W,3)
    elif img.shape[2] == 4:
        img = img[:, :, :3]                       # drop alpha

    if img.dtype != np.uint8:
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype("uint8")

    return img


def create_yolo_inst(config: ModelConfig):

    for comp in config.components:
        name = comp.name.lower()
        model_path = Path(comp.path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if not model_path.is_file():
            raise ValueError(f"Model path is not a file: {model_path}")

        if name == "yolo":
            try:
                model = YOLO(str(model_path))
                print(f"[TYPE] model: {type(model)}")
                model.export(format="onnx", int8= True, simplify= True,imgsz=640, opset=12)
            except Exception as e:
                raise RuntimeError(f"Failed to load YOLO: {e}") from e
            return model


def run_yolo():
    sessions=get_sessions(N_IMAGES)
    print(len(sessions))
    yolo= create_yolo_inst(nano_config)

    for i, session in enumerate(sessions): 
        t0= time.perf_counter()

        orig_files = list(session.glob("org_*"))
        if not orig_files:
            pytest.skip(f"No original image in session ")
        img= load_image(orig_files[0])
        print("IMG SHAPE:", img.shape)

        t1= time.perf_counter()
        elapsed= t1-t0
        
        results = yolo.predict(source=img,conf=0.25, iou=0.5, max_det=4000)
        boxes = results[0].boxes.xyxy


        print("\n")
        print(f"[IMG LOAD] {elapsed}")
        print(f"[DETECT] {time.perf_counter() -t1}")
        print("\n")






def test_create_yolo_inst():
    t0= time.perf_counter()
    create_yolo_inst(nano_config)
    t1= time.perf_counter()
    print(f"elapsed: {t1-t0}")

def test_run_yolo():
    run_yolo()
        

