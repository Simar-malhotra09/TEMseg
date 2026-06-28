import logging
import pickle
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

MAX_TRAIN_PIXELS = 500_000
_DEFAULT_MIN_AREA = 50


def _extract_features(image: np.ndarray) -> np.ndarray:
    """Per-pixel texture features from an RGB uint8 image. Returns (H, W, 6) float32."""
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY).astype(np.float32)

    blur3 = cv.GaussianBlur(gray, (3, 3), 0)
    blur7 = cv.GaussianBlur(gray, (7, 7), 0)

    gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)

    lap = cv.Laplacian(gray, cv.CV_32F)

    # local std via variance formula: E[X²] - E[X]²
    blur5 = cv.GaussianBlur(gray, (5, 5), 0)
    blur_sq = cv.GaussianBlur(gray**2, (5, 5), 0)
    local_std = np.sqrt(np.maximum(blur_sq - blur5**2, 0))

    return np.stack([gray, blur3, blur7, grad_mag, lap, local_std], axis=-1)


def _subsample(X: np.ndarray, y: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    if len(X) <= n:
        return X, y
    idx = np.random.default_rng(42).choice(len(X), n, replace=False)
    return X[idx], y[idx]


class RFRecovery:
    def __init__(self, min_area: int = _DEFAULT_MIN_AREA) -> None:
        self.min_area = min_area
        self._rf: RandomForestClassifier | None = None
        self._X_accum: list[np.ndarray] = []
        self._y_accum: list[np.ndarray] = []

    def _build_xy(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        feats = _extract_features(image)
        X = feats.reshape(-1, feats.shape[-1])
        y = (mask > 0).reshape(-1).astype(np.uint8)
        return _subsample(X, y, MAX_TRAIN_PIXELS)

    def _fit(self) -> None:
        X_all = np.vstack(self._X_accum)
        y_all = np.concatenate(self._y_accum)
        X_all, y_all = _subsample(X_all, y_all, MAX_TRAIN_PIXELS)

        self._rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42,
        )
        self._rf.fit(X_all, y_all)
        logger.info(f"[RFRecovery] Fitted on {len(X_all)} pixels ({int(y_all.sum())} fg)")

    def train(self, image: np.ndarray, mask: np.ndarray) -> None:
        X, y = self._build_xy(image, mask)
        self._X_accum = [X]
        self._y_accum = [y]
        self._fit()

    def update(self, image: np.ndarray, mask: np.ndarray) -> None:
        X, y = self._build_xy(image, mask)
        self._X_accum.append(X)
        self._y_accum.append(y)
        self._fit()

    def get_prompts(
        self, image: np.ndarray, mask: np.ndarray, top_n: int = 5
    ) -> list[dict]:
        if self._rf is None:
            raise RuntimeError("RFRecovery must be trained before calling get_prompts")

        feats = _extract_features(image)
        proba = self._rf.predict_proba(feats.reshape(-1, feats.shape[-1]))[:, 1]
        pred_map = (proba > 0.5).astype(np.uint8).reshape(image.shape[:2])

        # only care about regions not already covered by the current mask
        missed = pred_map & ~((mask > 0).astype(np.uint8))

        labeled, n = ndimage.label(missed)

        prompts: list[dict] = []
        for comp_id in range(1, n + 1):
            comp = labeled == comp_id
            area = int(comp.sum())
            if area < self.min_area:
                continue
            ys, xs = np.where(comp)
            prompts.append(
                {
                    "point": [int(np.mean(xs)), int(np.mean(ys))],
                    "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    "area": area,
                }
            )

        # sort largest→smallest, drop bottom 20% (noise), return top_n
        prompts.sort(key=lambda p: p["area"], reverse=True)
        if len(prompts) > 5:
            prompts = prompts[: max(1, int(len(prompts) * 0.8))]

        return prompts[:top_n]

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "rf": self._rf,
                    "X": self._X_accum,
                    "y": self._y_accum,
                    "min_area": self.min_area,
                },
                f,
            )
        logger.info(f"[RFRecovery] Saved → {path}")

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._rf = state["rf"]
        self._X_accum = state["X"]
        self._y_accum = state["y"]
        self.min_area = state["min_area"]
        logger.info(f"[RFRecovery] Loaded ← {path}")
