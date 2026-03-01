import numpy as np
from pydantic import BaseModel
import cv2 as cv
from typing import List
from pathlib import Path 

class Box(BaseModel):
    id: str
    x: float
    y: float
    width: float
    height: float

def normalize_mask(mask) -> np.ndarray:
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.squeeze(0)
    elif mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask.squeeze(-1)

    assert mask.ndim == 2, f"Unexpected mask shape: {mask.shape}"
    return (mask > 0).astype("uint8") * 255

def blackout_regions(img: np.ndarray, regions: List[Box], save_path:Path | str) -> np.ndarray:
    """
    Black out rectangular regions in an image.
    Optionally saves the result for verification.
    """
    img_out = img.copy()
    h, w = img_out.shape[:2]

    for box in regions:
        x1 = max(0, min(int(box.x), w))
        x2 = max(0, min(int(box.x + box.width), w))
        y1 = max(0, min(int(box.y), h))
        y2 = max(0, min(int(box.y + box.height), h))

        img_out[y1:y2, x1:x2] = 0

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_img = (img_out * 255).astype("uint8") if img_out.dtype == np.float32 else img_out
        cv.imwrite(str(save_path), save_img)
        print(f"Blacked-out image saved to {save_path}")

    return img_out
