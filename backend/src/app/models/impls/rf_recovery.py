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

# Predict (and train) at this fraction of the original resolution.
# Keeps predict_proba fast while preserving enough spatial detail.
_PREDICT_SCALE = 0.25

# Gaussian sigmas applied at _PREDICT_SCALE resolution.
# At 0.25× a 2048px image → 512px; particles are ~7–54px → σ covers
# the blob-detection range for typical TEM particles.
_SIGMAS = (1.0, 2.0, 4.0, 8.0, 16.0)


def _hessian_eigenvalues(blurred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Eigenvalues of the Hessian matrix at each pixel.
    λ_max captures ridges/blobs (high curvature); λ_min captures edges.
    These are the features that make Labkit work for blob detection.
    """
    Hxx = cv.Sobel(blurred, cv.CV_32F, 2, 0, ksize=3)
    Hxy = cv.Sobel(blurred, cv.CV_32F, 1, 1, ksize=3)
    Hyy = cv.Sobel(blurred, cv.CV_32F, 0, 2, ksize=3)
    half_trace = (Hxx + Hyy) / 2.0
    disc = np.sqrt(np.maximum(((Hxx - Hyy) / 2.0) ** 2 + Hxy ** 2, 0.0))
    return half_trace + disc, half_trace - disc  # λ_large, λ_small


def _extract_features(image: np.ndarray) -> np.ndarray:
    """
    Labkit-style multi-scale features: Gaussian, gradient magnitude,
    Laplacian-of-Gaussian, and Hessian eigenvalues at each of _SIGMAS.
    Returns (H, W, n_features) float32.
    """
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY).astype(np.float32)
    channels: list[np.ndarray] = []

    for sigma in _SIGMAS:
        blurred = cv.GaussianBlur(gray, (0, 0), sigma)

        # Gaussian smoothing — captures local mean intensity at this scale
        channels.append(blurred)

        # Gradient magnitude — captures local edge strength
        gx = cv.Sobel(blurred, cv.CV_32F, 1, 0, ksize=3)
        gy = cv.Sobel(blurred, cv.CV_32F, 0, 1, ksize=3)
        channels.append(np.sqrt(gx ** 2 + gy ** 2))

        # Laplacian of Gaussian — blob detector / second-order response
        channels.append(cv.Laplacian(blurred, cv.CV_32F))

        # Hessian eigenvalues — critical for detecting blob-shaped particles
        lam_large, lam_small = _hessian_eigenvalues(blurred)
        channels.append(lam_large)
        channels.append(lam_small)

    # 5 features × 5 scales = 25 features total
    return np.stack(channels, axis=-1)


def _to_small(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Downscale image and mask to _PREDICT_SCALE for consistent feature extraction."""
    h, w = image.shape[:2]
    h_s = max(1, int(h * _PREDICT_SCALE))
    w_s = max(1, int(w * _PREDICT_SCALE))
    img_s = cv.resize(image, (w_s, h_s), interpolation=cv.INTER_AREA)
    mask_s = cv.resize(
        (mask > 0).astype(np.uint8), (w_s, h_s), interpolation=cv.INTER_NEAREST
    )
    return img_s, mask_s


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
        img_s, mask_s = _to_small(image, mask)
        feats = _extract_features(img_s)
        X = feats.reshape(-1, feats.shape[-1])
        y = mask_s.reshape(-1).astype(np.uint8)
        return _subsample(X, y, MAX_TRAIN_PIXELS)

    def _fit(self) -> None:
        X_all = np.vstack(self._X_accum)
        y_all = np.concatenate(self._y_accum)
        X_all, y_all = _subsample(X_all, y_all, MAX_TRAIN_PIXELS)

        self._rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,  # grow full trees — Labkit default
            n_jobs=-1,
            class_weight="balanced",
            random_state=42,
        )
        self._rf.fit(X_all, y_all)
        logger.info(
            f"[RFRecovery] Fitted on {len(X_all)} pixels "
            f"({int(y_all.sum())} fg / {int((y_all == 0).sum())} bg)"
        )

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

        img_s, mask_s = _to_small(image, mask)
        h_orig, w_orig = image.shape[:2]
        h_s, w_s = img_s.shape[:2]
        sx, sy = w_orig / w_s, h_orig / h_s

        feats = _extract_features(img_s)
        t_feat = time.perf_counter()

        proba = self._rf.predict_proba(feats.reshape(-1, feats.shape[-1]))[:, 1]
        t_pred = time.perf_counter()

        pred_map = (proba > 0.5).astype(np.uint8).reshape(h_s, w_s)
        missed = pred_map & ~mask_s

        labeled, n = ndimage.label(missed)
        t_label = time.perf_counter()

        logger.info(
            f"[RFRecovery] get_prompts: feat={t_feat-t0:.3f}s "
            f"predict={t_pred-t_feat:.3f}s label={t_label-t_pred:.3f}s "
            f"components={n}"
        )

        min_area_s = max(1, int(self.min_area / (sx * sy)))
        prompts: list[dict] = []

        for comp_id in range(1, n + 1):
            comp = labeled == comp_id
            area_s = int(comp.sum())
            if area_s < min_area_s:
                continue
            ys, xs = np.where(comp)
            prompts.append({
                "point": [int(np.mean(xs) * sx), int(np.mean(ys) * sy)],
                "bbox": [
                    int(xs.min() * sx), int(ys.min() * sy),
                    min(w_orig - 1, int(xs.max() * sx)),
                    min(h_orig - 1, int(ys.max() * sy)),
                ],
                "area": int(area_s * sx * sy),
            })

        prompts.sort(key=lambda p: p["area"], reverse=True)
        if len(prompts) > 5:
            prompts = prompts[: max(1, int(len(prompts) * 0.8))]

        return prompts[:top_n]

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {"rf": self._rf, "X": self._X_accum, "y": self._y_accum, "min_area": self.min_area},
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
