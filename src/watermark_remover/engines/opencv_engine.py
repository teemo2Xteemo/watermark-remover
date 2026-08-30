from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from watermark_remover.engines.base import InpaintEngine
from watermark_remover.exceptions import EngineError, MaskError
from watermark_remover.masks.base import validate_mask_array

_METHODS = {
    "telea": cv2.INPAINT_TELEA,
    "ns": cv2.INPAINT_NS,
}


def _pin_seeds() -> None:
    np.random.seed(0)
    cv2.setRNGSeed(0)


class OpenCVInpaintEngine(InpaintEngine):
    def __init__(self, radius: int, method: Literal["telea", "ns"]) -> None:
        if radius < 1:
            raise EngineError("OpenCV inpaint radius must be >= 1")
        if method not in _METHODS:
            raise EngineError(f"unknown OpenCV inpaint method: {method!r}")
        self._radius = int(radius)
        self._method: Literal["telea", "ns"] = method

    def process(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise EngineError("image must be BGR uint8 with shape (H, W, 3)")
        binary = validate_mask_array(mask)
        if binary.shape != image.shape[:2]:
            raise MaskError(
                f"mask shape {binary.shape} does not match image {image.shape[:2]}"
            )
        _pin_seeds()
        try:
            result = cv2.inpaint(image, binary, self._radius, _METHODS[self._method])
        except cv2.error as exc:
            raise EngineError("OpenCV inpaint failed") from exc
        if result.dtype != np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(result)
