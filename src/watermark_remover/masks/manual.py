from __future__ import annotations

import numpy as np

from watermark_remover.exceptions import MaskError
from watermark_remover.masks.base import MaskProvider, validate_mask_array


class ManualMaskProvider(MaskProvider):
    def __init__(self, mask: np.ndarray) -> None:
        self._mask = validate_mask_array(mask)

    def get_mask(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        del frame_idx
        if frame.ndim < 2:
            raise MaskError("frame must have at least 2 dimensions")
        frame_hw = (int(frame.shape[0]), int(frame.shape[1]))
        if self._mask.shape != frame_hw:
            raise MaskError(
                f"mask shape {self._mask.shape} does not match frame {frame_hw}"
            )
        return self._mask
