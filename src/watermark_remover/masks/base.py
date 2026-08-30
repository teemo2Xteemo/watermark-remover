from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from watermark_remover.exceptions import MaskError


@dataclass(frozen=True)
class MaskCandidate:
    mask: np.ndarray
    confidence: float
    method: str
    bbox: tuple[int, int, int, int]


class MaskProvider(ABC):
    @abstractmethod
    def get_mask(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """Return a binary mask uint8 {0, 255} with shape (H, W) == frame[:2]."""


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        if mask.shape[2] == 1:
            gray = mask[:, :, 0]
        else:
            gray = mask.max(axis=2)
    elif mask.ndim == 2:
        gray = mask
    else:
        raise MaskError("mask must have shape (H, W) or (H, W, C)")
    return np.where(gray > 127, np.uint8(255), np.uint8(0))


def validate_mask_array(mask: np.ndarray) -> np.ndarray:
    binary = binarize_mask(mask)
    unique = set(np.unique(binary).tolist())
    if not unique.issubset({0, 255}):
        raise MaskError("mask values must be {0, 255}")
    return np.ascontiguousarray(binary, dtype=np.uint8)


def validate_mask_coverage(
    mask: np.ndarray,
    *,
    allow_empty_mask: bool = False,
    allow_full_mask: bool = False,
) -> None:
    if mask.size == 0:
        raise MaskError("mask has zero size")
    filled = int(np.count_nonzero(mask))
    if filled == 0 and not allow_empty_mask:
        raise MaskError("empty mask; pass --allow-empty-mask to override")
    if filled == mask.size and not allow_full_mask:
        raise MaskError("full-image mask; pass --allow-full-mask to override")
