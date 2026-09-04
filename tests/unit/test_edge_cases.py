"""Required edge-case table from 40-testing-quality.mdc."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from watermark_remover.config import Settings
from watermark_remover.engines.lama_engine import LaMaInpaintEngine
from watermark_remover.engines.opencv_engine import OpenCVInpaintEngine
from watermark_remover.engines.tiling import TiledInpaint
from watermark_remover.exceptions import EngineError, InputValidationError, MaskError
from watermark_remover.image_processor import ImageProcessor
from watermark_remover.io.image import read_image
from watermark_remover.io.validate import refuse_overwrite_unless_flag, validate_size_limits
from watermark_remover.masks.base import validate_mask_coverage


def test_empty_mask_raises_unless_allow_flag() -> None:
    empty = np.zeros((8, 8), dtype=np.uint8)
    with pytest.raises(MaskError, match="empty mask"):
        validate_mask_coverage(empty)
    validate_mask_coverage(empty, allow_empty_mask=True)


def test_full_image_mask_raises_unless_allow_flag() -> None:
    full = np.full((8, 8), 255, dtype=np.uint8)
    with pytest.raises(MaskError, match="full-image mask"):
        validate_mask_coverage(full)
    validate_mask_coverage(full, allow_full_mask=True)


def test_tiny_resolution_opencv_does_not_crash() -> None:
    image = np.array([[[10, 20, 30]]], dtype=np.uint8)
    mask = np.array([[255]], dtype=np.uint8)
    out = OpenCVInpaintEngine(radius=3, method="telea").process(image, mask)
    assert out.shape == (1, 1, 3)
    assert out.dtype == np.uint8


def test_huge_resolution_tiles_without_crash() -> None:
    image = np.full((40, 48, 3), 12, dtype=np.uint8)
    mask = np.zeros((40, 48), dtype=np.uint8)
    mask[4:20, 4:20] = 255
    engine = OpenCVInpaintEngine(radius=3, method="telea")
    out = TiledInpaint(tile_size=24, overlap=4).process(image, mask, engine)
    assert out.shape == image.shape
    assert out.dtype == np.uint8


def test_oversize_file_rejected_before_decode(tmp_path: Path) -> None:
    src = tmp_path / "big.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 64)
    with pytest.raises(InputValidationError, match="max_input_bytes"):
        validate_size_limits(src, max_input_bytes=8)
    with patch("watermark_remover.io.image.cv2.imdecode") as mocked:
        with pytest.raises(InputValidationError, match="max_input_bytes"):
            validate_size_limits(src, max_input_bytes=8)
        mocked.assert_not_called()
        with pytest.raises(InputValidationError):
            read_image(src)


def test_missing_lama_weights_engine_error_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("LaMa must not open the network")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    monkeypatch.setattr("urllib.request.urlretrieve", boom)
    with pytest.raises(EngineError, match="download_models"):
        LaMaInpaintEngine(tmp_path / "missing.onnx", "cpu")


def test_output_path_equals_input_refused_without_overwrite(tmp_path: Path) -> None:
    src = tmp_path / "same.png"
    src.write_bytes(b"png")
    with pytest.raises(InputValidationError, match="overwrite"):
        refuse_overwrite_unless_flag(src, src, overwrite=False)
    refuse_overwrite_unless_flag(src, src, overwrite=True)


def test_image_processor_empty_mask_unless_flag() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    with pytest.raises(MaskError, match="empty mask"):
        ImageProcessor().process(image, mask, "opencv", Settings())
    out = ImageProcessor().process(
        image, mask, "opencv", Settings(), allow_empty_mask=True
    )
    assert out.shape == image.shape
