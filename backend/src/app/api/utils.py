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

def inverse_blackout_regions(img: np.ndarray, regions: List[Box], save_path:Path | str)-> np.ndarray:
    """
    Inverse of the blackout_regions function: 
    So there's two ways you can go about this:
    a. Black out everything else, 
    b. Seperate each as patches, and batch run seg. 

    I assume b. will work much better across models. 
    This same applied to blackout function as well. 

    For now, we will just get the a. working. 
    """

    img_out= np.zeros_like(img)
    h, w = img_out.shape[:2]
    for box in regions:
        x1 = max(0, min(int(box.x), w))
        x2 = max(0, min(int(box.x + box.width), w))
        y1 = max(0, min(int(box.y), h))
        y2 = max(0, min(int(box.y + box.height), h))
        img_out[y1:y2, x1:x2] = img[y1:y2, x1:x2]  # copy original pixels back in

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_img = img_out.copy()
        if save_img.dtype in (np.float32, np.float64):
            mn, mx = save_img.min(), save_img.max()
            save_img = ((save_img - mn) / (mx - mn + 1e-8) * 255).astype("uint8") if mx > mn else np.zeros_like(save_img, dtype="uint8")
        cv.imwrite(str(save_path), save_img)
        print(f"Inverse blacked-out image saved to {save_path}")

    return img_out

import numpy as np
import cv2 as cv

def colorize_components_inplace(mask: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Converts a binary mask (0/1 or 0/255) into a colored
    connected-components image.

    Returns a 3-channel uint8 image.
    """
    binary = (mask > 0).astype(np.uint8)

    num_labels, labels = cv.connectedComponents(binary)

    colored = np.zeros((*labels.shape, 3), dtype=np.uint8)

    rng = np.random.default_rng(seed)
    colors = rng.integers(0, 255, size=(num_labels, 3), dtype=np.uint8)

    for label in range(1, num_labels):  # skip background
        colored[labels == label] = colors[label]

    return colored
