"""Download and verify optional model weights."""

import os
import urllib.error
import urllib.request
from pathlib import Path

import xxhash

from app.logutils import get_logger

logger = get_logger("weights")


def ensure_xxh128_weight(path: Path, url: str, expected_hash: str) -> Path:
    """Download one public model file and verify its upstream XXH128 digest."""
    if path.exists():
        if _hash_file(path) == expected_hash:
            return path
        raise ValueError(f"Model weight checksum mismatch: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: OSError | urllib.error.URLError | ValueError | None = None
    for attempt in range(1, 4):
        temporary_path = path.with_suffix(f"{path.suffix}.{attempt}.part")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "TEMseg/1.0"})
            with urllib.request.urlopen(request, timeout=300) as response:
                with temporary_path.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            actual_hash = _hash_file(temporary_path)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Model weight checksum mismatch for {path.name}: "
                    f"expected {expected_hash}, got {actual_hash}"
                )
            os.replace(temporary_path, path)
            logger.info("Optional model weight downloaded", extra={"path": str(path)})
            return path
        except (OSError, urllib.error.URLError, ValueError) as error:
            last_error = error
            logger.warning(
                "Optional model weight download failed",
                extra={"path": str(path), "attempt": attempt, "error": str(error)},
            )
            temporary_path.unlink(missing_ok=True)

    raise RuntimeError(
        f"Could not download {path.name} from {url} after 3 attempts: {last_error}"
    )


def _hash_file(path: Path) -> str:
    digest = xxhash.xxh128()
    with path.open("rb") as model_file:
        while chunk := model_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
