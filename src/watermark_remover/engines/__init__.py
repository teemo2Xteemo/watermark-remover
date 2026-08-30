from __future__ import annotations

from watermark_remover.engines.base import InpaintEngine
from watermark_remover.engines.lama_engine import LaMaInpaintEngine
from watermark_remover.engines.opencv_engine import OpenCVInpaintEngine
from watermark_remover.engines.registry import get_engine
from watermark_remover.engines.tiling import TiledInpaint

__all__ = [
    "InpaintEngine",
    "LaMaInpaintEngine",
    "OpenCVInpaintEngine",
    "TiledInpaint",
    "get_engine",
]
