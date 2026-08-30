from __future__ import annotations

import numpy as np

from watermark_remover.engines.base import InpaintEngine
from watermark_remover.exceptions import EngineError, MaskError
from watermark_remover.masks.base import validate_mask_array

_DEFAULT_TILE_SIZE = 512
_DEFAULT_OVERLAP = 32


class TiledInpaint:
    """Split a frame into overlapping tiles, run any InpaintEngine, blend seams."""

    def __init__(
        self,
        tile_size: int = _DEFAULT_TILE_SIZE,
        overlap: int = _DEFAULT_OVERLAP,
    ) -> None:
        if tile_size < 1:
            raise EngineError("tile_size must be >= 1")
        if overlap < 0:
            raise EngineError("tile overlap must be >= 0")
        if overlap >= tile_size:
            raise EngineError("tile overlap must be smaller than tile_size")
        self._tile_size = int(tile_size)
        self._overlap = int(overlap)

    @property
    def tile_size(self) -> int:
        return self._tile_size

    @property
    def overlap(self) -> int:
        return self._overlap

    def process(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        engine: InpaintEngine,
    ) -> np.ndarray:
        """Inpaint `image` (H, W, 3) BGR uint8 using overlapping tiles.

        Overlap regions are blended with weights that sum to 1 so a tile is
        never double-applied. Unmasked pixels are copied from `image`.
        """
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise EngineError("image must be BGR uint8 with shape (H, W, 3)")
        binary = validate_mask_array(mask)
        if binary.shape != image.shape[:2]:
            raise MaskError(f"mask shape {binary.shape} does not match image {image.shape[:2]}")

        height, width = image.shape[:2]
        if height <= self._tile_size and width <= self._tile_size:
            return np.ascontiguousarray(engine.process(image, binary))

        acc = np.zeros((height, width, 3), dtype=np.float64)
        weight_sum = np.zeros((height, width), dtype=np.float64)
        y_origins = _tile_origins(height, self._tile_size, self._overlap)
        x_origins = _tile_origins(width, self._tile_size, self._overlap)

        for y0 in y_origins:
            y1 = min(y0 + self._tile_size, height)
            for x0 in x_origins:
                x1 = min(x0 + self._tile_size, width)
                tile_mask = binary[y0:y1, x0:x1]
                if not np.any(tile_mask):
                    continue
                tile_image = image[y0:y1, x0:x1]
                inpainted = engine.process(tile_image, tile_mask)
                if inpainted.shape != tile_image.shape or inpainted.dtype != np.uint8:
                    raise EngineError(
                        "tiled engine must return uint8 with the same shape as the tile"
                    )
                weights = _tile_blend_weights(
                    tile_h=y1 - y0,
                    tile_w=x1 - x0,
                    y0=y0,
                    x0=x0,
                    height=height,
                    width=width,
                    overlap=self._overlap,
                )
                weights = weights * (tile_mask > 0)
                acc[y0:y1, x0:x1] += inpainted.astype(np.float64) * weights[..., None]
                weight_sum[y0:y1, x0:x1] += weights

        result = image.astype(np.float64)
        covered = weight_sum > 0
        result[covered] = acc[covered] / weight_sum[covered][:, None]
        unmasked = binary == 0
        result[unmasked] = image[unmasked]
        return np.ascontiguousarray(np.clip(np.rint(result), 0, 255).astype(np.uint8))


def _tile_origins(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    step = tile_size - overlap
    last = length - tile_size
    origins = list(range(0, last + 1, step))
    if origins[-1] != last:
        origins.append(last)
    return origins


def _tile_blend_weights(
    *,
    tile_h: int,
    tile_w: int,
    y0: int,
    x0: int,
    height: int,
    width: int,
    overlap: int,
) -> np.ndarray:
    """Linear ramps on interior overlaps; full weight on image borders."""
    wy = np.ones(tile_h, dtype=np.float64)
    wx = np.ones(tile_w, dtype=np.float64)
    if overlap > 0:
        if y0 > 0:
            n = min(overlap, tile_h)
            wy[:n] = np.linspace(0.0, 1.0, n, endpoint=True)
        if y0 + tile_h < height:
            n = min(overlap, tile_h)
            wy[-n:] = np.linspace(1.0, 0.0, n, endpoint=True)
        if x0 > 0:
            n = min(overlap, tile_w)
            wx[:n] = np.linspace(0.0, 1.0, n, endpoint=True)
        if x0 + tile_w < width:
            n = min(overlap, tile_w)
            wx[-n:] = np.linspace(1.0, 0.0, n, endpoint=True)
    weights = wy[:, None] * wx[None, :]
    return np.maximum(weights, 1e-8)
