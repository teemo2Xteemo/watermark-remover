from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

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
