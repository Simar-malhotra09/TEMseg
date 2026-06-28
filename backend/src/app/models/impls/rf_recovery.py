"""
RF pixel classifier for recovering missed detections from YOLO+SAM output.
Mirrors the Labkit filter bank approach: structure-tensor eigenvalues +
multi-scale Gaussian/LoG/gradient at full image resolution.

Critical design choices (vs naive approach):
- Labels are extracted from the mask with an uncertain border zone excluded:
    certain fg = eroded instances (2 px in from SAM boundary)
    certain bg = pixels > 10 px away from any particle
    everything in between = unlabeled (not used for training)
  This prevents SAM's noisy boundaries from poisoning the RF.
- Feature extraction and prediction both run at full resolution so the
  sigma values (1, 2, 4, 8) correspond to real particle scales.
"""

import logging
import pickle
import time
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, gaussian_laplace
from skimage.color import rgb2gray
from skimage.feature import structure_tensor, structure_tensor_eigenvalues
from skimage.morphology import dilation, disk, erosion, remove_small_objects
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

logger = logging.getLogger(__name__)

MAX_TRAIN_PIXELS = 500_000
_DEFAULT_MIN_AREA = 200   # px² at full resolution; below this → noise
_BG_DILATION = 10         # px — confident bg must be this far from any particle
_FG_EROSION = 2           # px — confident fg excludes noisy SAM boundary
_SIGMAS = (1.0, 2.0, 4.0, 8.0)


def _extract_features(image: np.ndarray) -> np.ndarray:
    """
    Per-pixel feature matrix at full resolution.
    21 features: raw intensity + (Gaussian, grad mag, LoG, ST-eig×2) × 4 sigmas.
    Returns (H*W, 21) float32.
    """
    gray = rgb2gray(image) if image.ndim == 3 else image.astype(np.float64)
    lo, hi = gray.min(), gray.max()
    gray = ((gray - lo) / (hi - lo + 1e-8)).astype(np.float32)

    channels: list[np.ndarray] = [gray]

    for sigma in _SIGMAS:
        g = gaussian_filter(gray, sigma)

        channels.append(g)

        gy, gx = np.gradient(g)
        channels.append(np.sqrt(gx ** 2 + gy ** 2).astype(np.float32))

        channels.append(gaussian_laplace(gray, sigma).astype(np.float32))

        Axx, Axy, Ayy = structure_tensor(g, sigma=sigma)
        eigs = structure_tensor_eigenvalues(np.array([Axx, Axy, Ayy]))
        channels.append(eigs[0].astype(np.float32))
        channels.append(eigs[1].astype(np.float32))

    feat_stack = np.stack(channels, axis=-1)  # (H, W, 21)
    H, W, _ = feat_stack.shape
    return feat_stack.reshape(H * W, -1)


def _labels_from_mask(mask: np.ndarray) -> np.ndarray:
    """
    Build confident pixel labels, leaving the uncertain boundary zone unlabeled.

    Returns flat int8 array: 1 = certain fg, 0 = certain bg, -1 = unlabeled.
    The unlabeled ring (within _BG_DILATION px of any particle) prevents the RF
    from training on SAM's noisy segmentation boundaries.
    """
    binary = mask > 0
    labels = np.full(binary.size, -1, dtype=np.int8)

    eroded_fg = erosion(binary, disk(_FG_EROSION))
    labels[eroded_fg.ravel()] = 1

    dilated_fg = dilation(binary, disk(_BG_DILATION))
    labels[(~dilated_fg).ravel()] = 0

    return labels


def _subsample(X: np.ndarray, y: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    if len(X) <= n:
        return X, y
    X, y = shuffle(X, y, random_state=42)
    return X[:n], y[:n]


class RFRecovery:
    def __init__(self, min_area: int = _DEFAULT_MIN_AREA) -> None:
        self.min_area = min_area
        self._rf: RandomForestClassifier | None = None
        self._X_accum: list[np.ndarray] = []
        self._y_accum: list[np.ndarray] = []

    def _build_xy(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        features = _extract_features(image)
        labels = _labels_from_mask(mask)
        labeled = labels != -1
        return _subsample(features[labeled], labels[labeled], MAX_TRAIN_PIXELS)

    def _fit(self) -> None:
        X_all = np.vstack(self._X_accum)
        y_all = np.concatenate(self._y_accum)
        X_all, y_all = _subsample(X_all, y_all, MAX_TRAIN_PIXELS)

        self._rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42,
        )
        self._rf.fit(X_all, y_all)
        logger.info(
            f"[RFRecovery] fitted {len(X_all)} px "
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
            raise RuntimeError("RFRecovery: call train() first")

        t0 = time.perf_counter()
        h, w = image.shape[:2]

        features = _extract_features(image)
        t_feat = time.perf_counter()

        probs = self._rf.predict_proba(features)[:, 1].reshape(h, w)
        t_pred = time.perf_counter()

        missed = (probs > 0.6) & ~(mask > 0)
        missed = remove_small_objects(missed, min_size=self.min_area)

        labeled, n = ndimage.label(missed)
        t_label = time.perf_counter()

        logger.info(
            f"[RFRecovery] feat={t_feat-t0:.2f}s predict={t_pred-t_feat:.2f}s "
            f"label={t_label-t_pred:.2f}s components={n}"
        )

        prompts: list[dict] = []
        for comp_id in range(1, n + 1):
            comp = labeled == comp_id
            ys, xs = np.where(comp)
            prompts.append({
                "point": [int(np.mean(xs)), int(np.mean(ys))],
                "bbox": [int(xs.min()), int(ys.min()), min(w - 1, int(xs.max())), min(h - 1, int(ys.max()))],
                "area": int(comp.sum()),
            })

        prompts.sort(key=lambda p: p["area"], reverse=True)
        return prompts[:top_n]

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"rf": self._rf, "X": self._X_accum, "y": self._y_accum, "min_area": self.min_area}, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._rf = state["rf"]
        self._X_accum = state["X"]
        self._y_accum = state["y"]
        self.min_area = state["min_area"]
