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
ResolvedEngineName = Literal["opencv", "lama"]

# Placeholder files (e.g. 19-byte stubs) are not valid ONNX; test fixture tiny.onnx is larger.
_MIN_WEIGHTS_BYTES = 128
_SETUP_COMMAND = "python scripts/download_models.py"


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


def resolved_engine_name(engine: InpaintEngine) -> ResolvedEngineName:
    if isinstance(engine, LaMaInpaintEngine):
        return "lama"
    return "opencv"


def resolve_lama_weights(settings: Settings) -> Path:
    """Return a usable ONNX path. Skip tiny placeholders and try MODEL_DIR/lama.onnx."""
    primary = (
        Path(settings.lama_weights)
        if settings.lama_weights is not None
        else Path(settings.model_dir) / "lama.onnx"
    )
    fallback = Path(settings.model_dir) / "lama.onnx"
    if _is_usable_weights(primary):
        return primary
    primary_exists = False
    try:
        primary_exists = primary.is_file()
    except OSError:
        primary_exists = False
    if primary_exists and not _same_path(primary, fallback) and _is_usable_weights(fallback):
        return fallback
    raise EngineError(f"LaMa weights not found ({primary.name}). Run: {_SETUP_COMMAND}")


def _is_usable_weights(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= _MIN_WEIGHTS_BYTES
    except OSError:
        return False


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return str(left).replace("\\", "/").lower() == str(right).replace("\\", "/").lower()


def _lama_engine(settings: Settings) -> LaMaInpaintEngine:
    return LaMaInpaintEngine(
        weights_path=resolve_lama_weights(settings),
        device=_lama_device(),
        tile_size=settings.tile_size,
        tile_overlap=settings.tile_overlap,
    )


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
