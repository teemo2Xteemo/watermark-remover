from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from watermark_remover.exceptions import InputValidationError
from watermark_remover.io.validate import sniff_image_format, validate_input_path

_ENCODE_EXT = {
    ".png": ".png",
    ".jpg": ".jpg",
    ".jpeg": ".jpg",
    ".webp": ".webp",
}
_PNG_PARAMS = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
_JPEG_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), 95]


def read_image(path: Path) -> np.ndarray:
    """Read an image as BGR uint8 with shape (H, W, 3)."""
    src = validate_input_path(path)
    sniff_image_format(src)
    payload = np.fromfile(str(src), dtype=np.uint8)
    image = cv2.imdecode(payload, cv2.IMREAD_COLOR)
    if image is None:
        raise InputValidationError(f"failed to decode image: {src.name}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[2] != 3:
        raise InputValidationError(f"expected 3-channel image: {src.name}")
    return np.ascontiguousarray(image, dtype=np.uint8)


def write_bytes_atomic(path: Path, data: bytes) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        with tmp.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, dest)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def write_image_atomic(path: Path, image: np.ndarray) -> None:
    """Encode image and replace the destination atomically (tmp → fsync → replace)."""
    dest = Path(path)
    suffix = dest.suffix.lower()
    encode_ext = _ENCODE_EXT.get(suffix)
    if encode_ext is None:
        raise InputValidationError(f"unsupported output format '{dest.suffix}'")
    params = _PNG_PARAMS if encode_ext == ".png" else _JPEG_PARAMS
    ok, buffer = cv2.imencode(encode_ext, image, params)
    if not ok:
        raise InputValidationError(f"failed to encode image: {dest.name}")
    write_bytes_atomic(dest, buffer.tobytes())
