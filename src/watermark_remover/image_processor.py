from __future__ import annotations

from typing import Literal

import numpy as np

from watermark_remover.config import Settings
from watermark_remover.engines.registry import get_engine
from watermark_remover.engines.tiling import TiledInpaint
from watermark_remover.exceptions import EngineError
from watermark_remover.masks.base import validate_mask_array, validate_mask_coverage
from watermark_remover.masks.manual import ManualMaskProvider

EngineName = Literal["opencv", "lama", "auto"]


class ImageProcessor:
    def process(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        engine_name: EngineName,
        config: Settings,
        *,
        allow_empty_mask: bool = False,
        allow_full_mask: bool = False,
    ) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise EngineError("image must be BGR uint8 with shape (H, W, 3)")
        binary = validate_mask_array(mask)
        provider = ManualMaskProvider(binary)
        resolved = provider.get_mask(image, frame_idx=0)
        validate_mask_coverage(
            resolved,
            allow_empty_mask=allow_empty_mask,
            allow_full_mask=allow_full_mask,
        )
        engine = get_engine(engine_name, resolved, config)
        height, width = image.shape[:2]
        if height > config.tile_size or width > config.tile_size:
            return TiledInpaint(config.tile_size, config.tile_overlap).process(
                image, resolved, engine
            )
        return engine.process(image, resolved)
