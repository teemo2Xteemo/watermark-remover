from __future__ import annotations

import numpy as np
import pytest

from watermark_remover.engines.base import InpaintEngine
from watermark_remover.engines.tiling import TiledInpaint
from watermark_remover.exceptions import EngineError, MaskError


class _ConstantEngine(InpaintEngine):
    def __init__(self, value: int = 100) -> None:
        self._value = value
        self.calls: list[tuple[int, int]] = []

    def process(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self.calls.append((int(image.shape[0]), int(image.shape[1])))
        return np.full_like(image, self._value)


class _IdentityEngine(InpaintEngine):
    def process(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(image.copy())


class _OriginPaintEngine(InpaintEngine):
    """Paint the mask with a stamp derived from the tile mean so adjacent tiles differ."""

    def process(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = image.copy()
        stamp = int(np.mean(image[:, :, 0]))
        out[mask == 255] = (stamp, 40, 200)
        return out


def _gradient_image(height: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = (xx % 256).astype(np.uint8)
    image[:, :, 1] = (yy % 256).astype(np.uint8)
    image[:, :, 2] = 80
    return image


def test_tiling_output_shape_dtype_and_bounds() -> None:
    image = _gradient_image(40, 56)
    mask = np.zeros((40, 56), dtype=np.uint8)
    mask[8:32, 10:40] = 255
    out = TiledInpaint(tile_size=24, overlap=4).process(image, mask, _ConstantEngine(90))
    assert out.shape == image.shape
    assert out.dtype == np.uint8
    assert not np.isnan(out.astype(np.float64)).any()
    assert int(out.min()) >= 0
    assert int(out.max()) <= 255


def test_tiling_overlap_is_blended_not_double_applied() -> None:
    height, width = 40, 56
    image = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.full((height, width), 255, dtype=np.uint8)
    engine = _ConstantEngine(value=100)
    out = TiledInpaint(tile_size=24, overlap=8).process(image, mask, engine)
    assert len(engine.calls) > 1
    assert np.unique(out).tolist() == [100], "overlap must blend, not accumulate"


def test_tiling_overlap_region_is_not_hard_cut() -> None:
    height, width = 32, 48
    image = _gradient_image(height, width)
    mask = np.full((height, width), 255, dtype=np.uint8)
    tile_size, overlap = 24, 8
    out = TiledInpaint(tile_size=tile_size, overlap=overlap).process(
        image, mask, _OriginPaintEngine()
    )
    step = tile_size - overlap
    band = out[4:12, step:tile_size, 0]
    unique_count = int(np.unique(band).size)
    assert unique_count >= 2


def test_tiling_identity_engine_preserves_unmasked_and_shape() -> None:
    image = _gradient_image(36, 36)
    mask = np.zeros((36, 36), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    out = TiledInpaint(tile_size=20, overlap=4).process(image, mask, _IdentityEngine())
    assert out.shape == image.shape
    assert np.array_equal(out[mask == 0], image[mask == 0])


def test_tiling_skips_empty_mask_tiles() -> None:
    image = _gradient_image(40, 40)
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[0:8, 0:8] = 255
    engine = _ConstantEngine(50)
    out = TiledInpaint(tile_size=16, overlap=4).process(image, mask, engine)
    assert engine.calls, "at least the masked tile must run"
    assert np.array_equal(out[mask == 0], image[mask == 0])


def test_tiling_rejects_overlap_not_smaller_than_tile() -> None:
    with pytest.raises(EngineError, match="overlap"):
        TiledInpaint(tile_size=16, overlap=16)


def test_single_tile_when_image_fits() -> None:
    image = _gradient_image(16, 20)
    mask = np.zeros((16, 20), dtype=np.uint8)
    mask[2:8, 2:8] = 255
    engine = _ConstantEngine(7)
    out = TiledInpaint(tile_size=32, overlap=4).process(image, mask, engine)
    assert engine.calls == [(16, 20)]
    assert out.shape == image.shape


def test_default_tile_size_512_splits_image_larger_than_tile() -> None:
    image = _gradient_image(520, 540)
    mask = np.full((520, 540), 255, dtype=np.uint8)
    engine = _ConstantEngine(80)
    tiled = TiledInpaint()
    assert tiled.tile_size == 512
    assert tiled.overlap == 32
    out = tiled.process(image, mask, engine)
    assert len(engine.calls) > 1
    assert all(height <= 512 and width <= 512 for height, width in engine.calls)
    assert out.shape == image.shape
    assert out.dtype == np.uint8
    assert not np.isnan(out.astype(np.float64)).any()
    assert int(out.min()) >= 0
    assert int(out.max()) <= 255


def test_tiling_rejects_invalid_tile_size_and_overlap() -> None:
    with pytest.raises(EngineError, match="tile_size"):
        TiledInpaint(tile_size=0, overlap=0)
    with pytest.raises(EngineError, match="overlap must be >= 0"):
        TiledInpaint(tile_size=16, overlap=-1)


def test_tiling_rejects_wrong_image_and_mask() -> None:
    tiled = TiledInpaint(tile_size=16, overlap=4)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1, 1] = 255
    with pytest.raises(EngineError, match="BGR uint8"):
        tiled.process(np.zeros((8, 8), dtype=np.uint8), mask, _IdentityEngine())
    with pytest.raises(MaskError, match="does not match"):
        tiled.process(
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.zeros((4, 4), dtype=np.uint8),
            _IdentityEngine(),
        )


def test_tiling_rejects_engine_output_with_wrong_shape() -> None:
    class _BadShape(InpaintEngine):
        def process(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
            del image, mask
            return np.zeros((2, 2, 3), dtype=np.uint8)

    image = _gradient_image(40, 40)
    mask = np.full((40, 40), 255, dtype=np.uint8)
    with pytest.raises(EngineError, match="same shape as the tile"):
        TiledInpaint(tile_size=16, overlap=4).process(image, mask, _BadShape())
