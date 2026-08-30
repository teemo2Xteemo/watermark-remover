from __future__ import annotations

from typing import Literal

import numpy as np

from watermark_remover.config import Settings
from watermark_remover.engines.base import InpaintEngine
from watermark_remover.engines.opencv_engine import OpenCVInpaintEngine
from watermark_remover.exceptions import EngineError

EngineName = Literal["opencv", "lama", "auto"]


def get_engine(
    name: EngineName,
    mask: np.ndarray,
    settings: Settings,
) -> InpaintEngine:
    if name == "lama":
        raise EngineError(
            "LaMa engine is not implemented yet. "
            "Download weights later with: python scripts/download_models.py"
        )
    if name == "auto":
        # M4 will route large masks to LaMa; M1 always selects OpenCV.
        _ = _mask_area_ratio(mask) < settings.mask_area_threshold
        return _opencv_engine(settings)
    if name == "opencv":
        return _opencv_engine(settings)
    raise EngineError(f"unknown engine: {name!r}")


def _opencv_engine(settings: Settings) -> OpenCVInpaintEngine:
    return OpenCVInpaintEngine(
        radius=settings.opencv_radius,
        method=settings.opencv_method,
    )


def _mask_area_ratio(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask)) / float(mask.size)
