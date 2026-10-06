from pathlib import Path

import cv2 as cv
import numpy as np
import tifffile


def load_image_array(image_path: Path) -> np.ndarray:
    """Read one image without discarding the source bit depth."""
    suffix = image_path.suffix.lower()
    if suffix == ".npy":
        return np.load(image_path, allow_pickle=False)
    if suffix == ".emd":
        from rsciio.emd import file_reader

        signals = file_reader(str(image_path))
        if not signals:
            raise ValueError(f"EMD contains no image: {image_path}")
        image = np.asarray(signals[0]["data"])
        if image.ndim != 2:
            raise ValueError(
                f"EMD contains a stack or scan. Select one image before import: {image_path}"
            )
        return image
    if suffix in {".tif", ".tiff"}:
        with tifffile.TiffFile(image_path) as tiff:
            series = tiff.series[0]
            if series.axes not in {"YX", "YXS", "YXC", "CYX"}:
                raise ValueError(
                    f"TIFF axes {series.axes} describe a stack or unsupported image: {image_path}"
                )
            return series.asarray()

    image = cv.imread(str(image_path), cv.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    if image.ndim == 3 and image.shape[2] == 3:
        return cv.cvtColor(image, cv.COLOR_BGR2RGB)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv.cvtColor(image, cv.COLOR_BGRA2RGBA)
    return image


def to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Make a display image from one 2D grayscale or RGB image."""
    if (
        image.ndim == 3
        and image.shape[0] in {1, 3, 4}
        and image.shape[2]
        not in {
            1,
            3,
            4,
        }
    ):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 2:
        rgb = np.repeat(image[:, :, None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] in {1, 3, 4}:
        rgb = np.repeat(image, 3, axis=2) if image.shape[2] == 1 else image[:, :, :3]
    else:
        raise ValueError(
            f"Expected one 2D image or RGB image, received shape {image.shape}"
        )

    if rgb.shape[0] == 0 or rgb.shape[1] == 0:
        raise ValueError("Image has no pixels")
    if not np.isfinite(rgb).all():
        raise ValueError("Image contains NaN or infinite pixel values")
    if rgb.dtype == np.uint8:
        return rgb

    low = float(np.min(rgb))
    high = float(np.max(rgb))
    if high == low:
        return np.zeros(rgb.shape, dtype=np.uint8)
    return (
        ((rgb.astype(np.float32) - low) * (255.0 / (high - low)))
        .clip(0, 255)
        .astype(np.uint8)
    )


def load_rgb_uint8(image_path: Path) -> np.ndarray:
    """Read one image and use the same normalization for preview and inference."""
    return to_rgb_uint8(load_image_array(image_path))
