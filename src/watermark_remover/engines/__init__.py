from __future__ import annotations

from watermark_remover.engines.base import InpaintEngine
from watermark_remover.engines.opencv_engine import OpenCVInpaintEngine
from watermark_remover.engines.registry import get_engine

__all__ = ["InpaintEngine", "OpenCVInpaintEngine", "get_engine"]
