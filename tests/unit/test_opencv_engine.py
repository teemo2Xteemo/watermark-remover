from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from watermark_remover.config import Settings
from watermark_remover.engines.opencv_engine import OpenCVInpaintEngine
from watermark_remover.engines.registry import get_engine
from watermark_remover.exceptions import EngineError, MaskError
from watermark_remover.image_processor import ImageProcessor
from watermark_remover.io.image import read_image, write_image_atomic
from watermark_remover.masks.serialize import load_mask_png


def test_opencv_process_shape_and_dtype() -> None:
    image = np.full((32, 48, 3), (20, 40, 60), dtype=np.uint8)
    mask = np.zeros((32, 48), dtype=np.uint8)
    mask[8:16, 8:16] = 255
    out = OpenCVInpaintEngine(radius=3, method="telea").process(image, mask)
    assert out.shape == image.shape
    assert out.dtype == np.uint8


def test_opencv_does_not_change_unmasked_pixels() -> None:
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    image[:, :] = (15, 80, 160)
    image[4:8, 4:8] = (200, 10, 10)
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[4:8, 4:8] = 255
    out = OpenCVInpaintEngine(radius=3, method="telea").process(image, mask)
    assert np.array_equal(out[mask == 0], image[mask == 0])


def test_opencv_telea_is_deterministic() -> None:
    image = np.linspace(0, 255, 16 * 20 * 3, dtype=np.uint8).reshape(16, 20, 3)
    mask = np.zeros((16, 20), dtype=np.uint8)
    mask[3:8, 5:12] = 255
    engine = OpenCVInpaintEngine(radius=3, method="telea")
    first = engine.process(image, mask)
    second = engine.process(image, mask)
    assert np.array_equal(first, second)


def test_opencv_telea_byte_stable_png(fixtures_dir: Path, tmp_path: Path) -> None:
    image = read_image(fixtures_dir / "still_logo.png")
    mask = load_mask_png(fixtures_dir / "still_logo.mask.png")
    out = OpenCVInpaintEngine(radius=3, method="telea").process(image, mask)
    dest = tmp_path / "out.png"
    write_image_atomic(dest, out)
    baseline = (fixtures_dir / "still_logo_inpainted_opencv_telea.png").read_bytes()
    assert dest.read_bytes() == baseline


def test_get_engine_lama_raises() -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    with pytest.raises(EngineError, match="LaMa"):
        get_engine("lama", mask, Settings())


def test_get_engine_auto_selects_opencv_in_m1() -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    engine = get_engine("auto", mask, Settings())
    assert isinstance(engine, OpenCVInpaintEngine)


def test_get_engine_unknown() -> None:
    mask = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(EngineError, match="unknown engine"):
        get_engine("nope", mask, Settings())  # type: ignore[arg-type]


def test_image_processor_empty_mask_raises() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    with pytest.raises(MaskError, match="empty mask"):
        ImageProcessor().process(image, mask, "opencv", Settings())


def test_image_processor_full_mask_raises() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.full((8, 8), 255, dtype=np.uint8)
    with pytest.raises(MaskError, match="full-image mask"):
        ImageProcessor().process(image, mask, "opencv", Settings())
