from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from watermark_remover.config import Settings
from watermark_remover.engines.base import InpaintEngine
from watermark_remover.engines.lama_engine import LaMaInpaintEngine
from watermark_remover.engines.opencv_engine import OpenCVInpaintEngine
from watermark_remover.exceptions import EngineError

EngineName = Literal["opencv", "lama", "auto"]


def get_engine(
    name: EngineName,
    mask: np.ndarray,
    settings: Settings,
) -> InpaintEngine:
    if name == "opencv":
        return _opencv_engine(settings)
    if name == "lama":
        return _lama_engine(settings)
    if name == "auto":
        if _mask_area_ratio(mask) < settings.mask_area_threshold:
            return _opencv_engine(settings)
        return _lama_engine(settings)
    raise EngineError(f"unknown engine: {name!r}")


def _opencv_engine(settings: Settings) -> OpenCVInpaintEngine:
    return OpenCVInpaintEngine(
        radius=settings.opencv_radius,
        method=settings.opencv_method,
    )


def _lama_engine(settings: Settings) -> LaMaInpaintEngine:
    weights = (
        Path(settings.lama_weights)
        if settings.lama_weights is not None
        else Path(settings.model_dir) / "lama.onnx"
    )
    return LaMaInpaintEngine(weights_path=weights, device=_lama_device())


def _lama_device() -> str:
    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" in ort.get_available_providers():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _mask_area_ratio(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask)) / float(mask.size)
