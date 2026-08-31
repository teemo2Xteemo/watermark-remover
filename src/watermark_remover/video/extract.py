from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from watermark_remover.exceptions import InputValidationError
from watermark_remover.io.video import open_capture


def extract_frames(path: Path) -> Iterator[tuple[int, np.ndarray]]:
    """Yield `(frame_index, frame_bgr_uint8)` one frame at a time. Does not buffer the video."""
    capture = open_capture(Path(path))
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise InputValidationError(f"expected BGR frames from video: {Path(path).name}")
            # Copy: OpenCV reuses the capture buffer on the next read().
            yield index, np.ascontiguousarray(frame.copy(), dtype=np.uint8)
            index += 1
    finally:
        capture.release()


def read_first_frame(path: Path) -> np.ndarray:
    """Decode only the first BGR frame. Does not read the rest of the file."""
    capture = open_capture(Path(path))
    try:
        ok, frame = capture.read()
        if not ok or frame is None:
            raise InputValidationError(f"cannot read first frame: {Path(path).name}")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise InputValidationError(f"expected BGR frames from video: {Path(path).name}")
        return np.ascontiguousarray(frame.copy(), dtype=np.uint8)
    finally:
        capture.release()


def read_frame_at(path: Path, frame_idx: int) -> np.ndarray:
    """Seek and decode a single BGR frame. Does not read neighboring frames into memory."""
    capture = open_capture(Path(path))
    try:
        if int(frame_idx) > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise InputValidationError(
                f"cannot read frame {int(frame_idx)}: {Path(path).name}"
            )
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise InputValidationError(f"expected BGR frames from video: {Path(path).name}")
        return np.ascontiguousarray(frame.copy(), dtype=np.uint8)
    finally:
        capture.release()
