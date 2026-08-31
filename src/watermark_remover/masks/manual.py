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


class KeyframeMaskProvider(MaskProvider):
    """Hold-last keyframes: latest mask whose timestamp `t` is ≤ current time."""

    def __init__(self, keyframes: list[tuple[float, np.ndarray]], fps: float) -> None:
        if fps <= 0:
            raise MaskError("fps must be positive")
        if not keyframes:
            raise MaskError("at least one keyframe is required")
        ordered = sorted(
            ((float(t), validate_mask_array(mask)) for t, mask in keyframes),
            key=lambda item: item[0],
        )
        self._keyframes = ordered
        self._fps = float(fps)

    def get_mask(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        if frame.ndim < 2:
            raise MaskError("frame must have at least 2 dimensions")
        frame_hw = (int(frame.shape[0]), int(frame.shape[1]))
        t = float(frame_idx) / self._fps
        mask = self._mask_at(t)
        if mask.shape != frame_hw:
            raise MaskError(
                f"mask shape {mask.shape} does not match frame {frame_hw}"
            )
        return mask

    def _mask_at(self, t: float) -> np.ndarray:
        chosen = self._keyframes[0][1]
        for key_t, key_mask in self._keyframes:
            if key_t <= t:
                chosen = key_mask
            else:
                break
        return chosen
