import numpy as np
import cv2
from scipy.spatial.distance import cdist
from fastapi import UploadFile
import io 
def normalize_mask(file: UploadFile) -> np.ndarray:
    """Accept npy, png, tiff — always return binary (H,W) uint8 array."""
    data = file.file.read()
    name = file.filename.lower()

    if name.endswith(".npy"):
        mask = np.load(io.BytesIO(data))
    elif name.endswith((".png", ".tif", ".tiff")):
        arr = np.frombuffer(data, np.uint8)
        mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    elif name.endswith(".json"):
        # COCO format — impl later
        raise NotImplementedError("COCO JSON not yet supported")
    else:
        raise ValueError(f"Unsupported format: {name}")

    # normalize to binary 0/1
    mask = (mask > 0).astype("uint8")

    # sanity check — if more than 2 unique values after binarization something is off
    return mask

def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    gt = (gt > 0).astype(bool)
    pred = (pred > 0).astype(bool)

    return {
        "iou": iou_score(gt, pred),
        # "dice": dice_coefficient(gt, pred),
        # "hausdorff_95": hausdorff_95(gt, pred),
    }

def dice_coefficient(y_true, y_pred) -> float:
    intersection = np.logical_and(y_true, y_pred).sum()
    return 2.0 * intersection / (y_true.sum() + y_pred.sum() + 1e-8)

def iou_score(y_true, y_pred) -> float:
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return float(intersection / (union + 1e-8))

def hausdorff_95(y_true, y_pred) -> float:
    true_points = np.argwhere(y_true)
    pred_points = np.argwhere(y_pred)
    if len(true_points) == 0 or len(pred_points) == 0:
        return float("nan")
    distances_fw = cdist(true_points, pred_points).min(axis=1)
    distances_bw = cdist(pred_points, true_points).min(axis=1)
    return float(np.percentile(np.concatenate([distances_fw, distances_bw]), 95))
