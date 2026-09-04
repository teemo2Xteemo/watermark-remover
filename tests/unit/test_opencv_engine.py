from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from watermark_remover.config import Settings
from watermark_remover.engines.opencv_engine import OpenCVInpaintEngine
from watermark_remover.engines.registry import get_engine
from watermark_remover.exceptions import EngineError, MaskError
from watermark_remover.image_processor import ImageProcessor
from watermark_remover.io.image import read_image, write_image_atomic
from watermark_remover.masks.serialize import load_mask_png

SSIM_MIN_THRESHOLD = 0.95
PSNR_MIN_DB_THRESHOLD = 30.0


def _psnr(actual: np.ndarray, baseline: np.ndarray) -> float:
    mse = float(np.mean((actual.astype(np.float64) - baseline.astype(np.float64)) ** 2))
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10((255.0**2) / mse)


def _ssim(actual: np.ndarray, baseline: np.ndarray) -> float:
    gray_a = cv2.cvtColor(actual, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray_b = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY).astype(np.float64)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    ksize = (11, 11)
    sigma = 1.5
    mu_a = cv2.GaussianBlur(gray_a, ksize, sigma)
    mu_b = cv2.GaussianBlur(gray_b, ksize, sigma)
    sigma_a = cv2.GaussianBlur(gray_a**2, ksize, sigma) - mu_a**2
    sigma_b = cv2.GaussianBlur(gray_b**2, ksize, sigma) - mu_b**2
    sigma_ab = cv2.GaussianBlur(gray_a * gray_b, ksize, sigma) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (sigma_a + sigma_b + c2)
    return float(np.mean(num / den))


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
    """PNG container bytes vary by libpng; compare decoded pixels / SSIM vs baseline."""
    image = read_image(fixtures_dir / "still_logo.png")
    mask = load_mask_png(fixtures_dir / "still_logo.mask.png")
    out = OpenCVInpaintEngine(radius=3, method="telea").process(image, mask)
    dest = tmp_path / "out.png"
    write_image_atomic(dest, out)
    written = read_image(dest)
    baseline = read_image(fixtures_dir / "still_logo_inpainted_opencv_telea.png")
    assert written.shape == baseline.shape
    assert written.dtype == baseline.dtype
    if np.array_equal(written, baseline):
        return
    assert _ssim(written, baseline) >= SSIM_MIN_THRESHOLD
    assert _psnr(written, baseline) >= PSNR_MIN_DB_THRESHOLD


def test_get_engine_lama_missing_weights(tmp_path: Path) -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    with pytest.raises(EngineError, match="download_models"):
        get_engine("lama", mask, Settings(lama_weights=tmp_path / "missing.onnx"))


def test_get_engine_auto_selects_opencv_for_small_mask() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0, 0] = 255
    engine = get_engine("auto", mask, Settings(mask_area_threshold=0.03))
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


def test_opencv_ns_process_shape_and_unmasked_pixels() -> None:
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    image[:, :] = (15, 80, 160)
    image[4:8, 4:8] = (200, 10, 10)
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[4:8, 4:8] = 255
    out = OpenCVInpaintEngine(radius=3, method="ns").process(image, mask)
    assert out.shape == image.shape
    assert out.dtype == np.uint8
    assert np.array_equal(out[mask == 0], image[mask == 0])


def test_opencv_rejects_bad_radius_and_method() -> None:
    with pytest.raises(EngineError, match="radius"):
        OpenCVInpaintEngine(radius=0, method="telea")
    with pytest.raises(EngineError, match="unknown OpenCV inpaint method"):
        OpenCVInpaintEngine(radius=3, method="magic")  # type: ignore[arg-type]


def test_opencv_rejects_wrong_image_and_mask_shape() -> None:
    engine = OpenCVInpaintEngine(radius=3, method="telea")
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1, 1] = 255
    with pytest.raises(EngineError, match="BGR uint8"):
        engine.process(np.zeros((8, 8), dtype=np.uint8), mask)
    with pytest.raises(MaskError, match="does not match"):
        engine.process(np.zeros((8, 8, 3), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8))


def test_opencv_wraps_cv2_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import cv2

    def boom(*_args: object, **_kwargs: object) -> None:
        raise cv2.error("inpaint boom")

    monkeypatch.setattr("watermark_remover.engines.opencv_engine.cv2.inpaint", boom)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:4, 2:4] = 255
    with pytest.raises(EngineError, match="OpenCV inpaint failed"):
        OpenCVInpaintEngine(radius=3, method="telea").process(image, mask)


def test_opencv_casts_non_uint8_result(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_inpaint(
        image: np.ndarray, _mask: np.ndarray, _radius: float, _flags: int
    ) -> np.ndarray:
        return image.astype(np.float32)

    monkeypatch.setattr("watermark_remover.engines.opencv_engine.cv2.inpaint", fake_inpaint)
    image = np.full((8, 8, 3), 40, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1, 1] = 255
    out = OpenCVInpaintEngine(radius=3, method="telea").process(image, mask)
    assert out.dtype == np.uint8
    assert out.shape == image.shape
