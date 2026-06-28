import logging
import pickle
import time
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

MAX_TRAIN_PIXELS = 500_000
_DEFAULT_MIN_AREA = 50
# Predict at this fraction of the original resolution to keep predict_proba fast.
# 0.25 → 512×512 on a 2048×2048 image (64× fewer pixels than full res).
_PREDICT_SCALE = 0.25


def _extract_features(image: np.ndarray) -> np.ndarray:
    """Per-pixel texture features from an RGB uint8 image. Returns (H, W, 6) float32."""
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY).astype(np.float32)

    blur3 = cv.GaussianBlur(gray, (3, 3), 0)
    blur7 = cv.GaussianBlur(gray, (7, 7), 0)

    gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)

    lap = cv.Laplacian(gray, cv.CV_32F)

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

        t0 = time.perf_counter()

        # Downscale before feature extraction so predict_proba runs on ~64× fewer pixels.
        h_orig, w_orig = image.shape[:2]
        h_small = max(1, int(h_orig * _PREDICT_SCALE))
        w_small = max(1, int(w_orig * _PREDICT_SCALE))
        img_small = cv.resize(image, (w_small, h_small), interpolation=cv.INTER_AREA)
        mask_small = cv.resize(
            (mask > 0).astype(np.uint8), (w_small, h_small), interpolation=cv.INTER_NEAREST
        )

        feats = _extract_features(img_small)
        t_feat = time.perf_counter()

        proba = self._rf.predict_proba(feats.reshape(-1, feats.shape[-1]))[:, 1]
        t_pred = time.perf_counter()

        pred_map = (proba > 0.5).astype(np.uint8).reshape(h_small, w_small)
        missed = pred_map & ~mask_small

        labeled, n = ndimage.label(missed)
        t_label = time.perf_counter()

        logger.info(
            f"[RFRecovery] get_prompts: feat={t_feat-t0:.3f}s "
            f"predict={t_pred-t_feat:.3f}s label={t_label-t_pred:.3f}s "
            f"pixels={h_small*w_small} components={n}"
        )

        # Scale factor to map small-image coords back to original
        sx = w_orig / w_small
        sy = h_orig / h_small
        # min_area is in original-image pixels; scale it to small-image pixels
        min_area_small = max(1, int(self.min_area / (sx * sy)))

        prompts: list[dict] = []
        for comp_id in range(1, n + 1):
            comp = labeled == comp_id
            area_small = int(comp.sum())
            if area_small < min_area_small:
                continue
            ys, xs = np.where(comp)
            # scale coordinates back to original image space
            cx = int(np.mean(xs) * sx)
            cy = int(np.mean(ys) * sy)
            x1 = int(xs.min() * sx)
            y1 = int(ys.min() * sy)
            x2 = min(w_orig - 1, int(xs.max() * sx))
            y2 = min(h_orig - 1, int(ys.max() * sy))
            area_orig = int(area_small * sx * sy)
            prompts.append(
                {
                    "point": [cx, cy],
                    "bbox": [x1, y1, x2, y2],
                    "area": area_orig,
                }
            )

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
