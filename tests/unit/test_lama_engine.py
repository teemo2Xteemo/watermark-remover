from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from watermark_remover.config import Settings
from watermark_remover.engines.lama_engine import LaMaInpaintEngine
from watermark_remover.engines.opencv_engine import OpenCVInpaintEngine
from watermark_remover.engines.registry import get_engine
from watermark_remover.exceptions import EngineError
from watermark_remover.image_processor import ImageProcessor
from watermark_remover.io.image import read_image, write_image_atomic
from watermark_remover.masks.serialize import load_mask_png

SSIM_MIN_THRESHOLD = 0.95
PSNR_MIN_DB_THRESHOLD = 30.0


def _stub_weights(fixtures_dir: Path) -> Path:
    pytest.importorskip("onnxruntime")
    path = fixtures_dir / "tiny.onnx"
    if not path.is_file():
        pytest.skip("tests/fixtures/tiny.onnx is missing")
    return path


def _real_weights() -> Path | None:
    settings = Settings()
    candidates: list[Path] = []
    if settings.lama_weights is not None:
        candidates.append(Path(settings.lama_weights))
    candidates.append(Path("models") / "lama.onnx")
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve() if path.exists() else path
        if resolved in seen:
            continue
        seen.add(resolved)
        # Real Carve LaMa ONNX is ~208 MiB; ignore tiny placeholders.
        if path.is_file() and path.stat().st_size >= 1_000_000:
            return path
    return None


def _cuda_available() -> bool:
    runtime = pytest.importorskip("onnxruntime")
    return "CUDAExecutionProvider" in runtime.get_available_providers()


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


def test_lama_missing_weights_raises_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("LaMa must not open the network")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    monkeypatch.setattr("urllib.request.urlretrieve", boom)
    with pytest.raises(EngineError, match="download_models"):
        LaMaInpaintEngine(tmp_path / "missing.onnx", "cpu")


def test_lama_stub_process_shape_and_dtype(fixtures_dir: Path) -> None:
    image = np.linspace(0, 255, 16 * 24 * 3, dtype=np.uint8).reshape(16, 24, 3)
    mask = np.zeros((16, 24), dtype=np.uint8)
    mask[4:10, 6:14] = 255
    out = LaMaInpaintEngine(_stub_weights(fixtures_dir), "cpu").process(image, mask)
    assert out.shape == image.shape
    assert out.dtype == np.uint8
    assert np.array_equal(out[mask == 0], image[mask == 0])
    assert not np.isnan(out.astype(np.float64)).any()
    assert int(out.min()) >= 0
    assert int(out.max()) <= 255


def test_lama_stub_ssim_psnr_vs_committed_baseline(fixtures_dir: Path, tmp_path: Path) -> None:
    image = read_image(fixtures_dir / "still_logo.png")
    mask = load_mask_png(fixtures_dir / "still_logo.mask.png")
    out = LaMaInpaintEngine(_stub_weights(fixtures_dir), "cpu").process(image, mask)
    dest = tmp_path / "lama_stub.png"
    write_image_atomic(dest, out)
    baseline = read_image(fixtures_dir / "still_logo_inpainted_lama_stub.png")
    assert _ssim(out, baseline) >= SSIM_MIN_THRESHOLD
    assert _psnr(out, baseline) >= PSNR_MIN_DB_THRESHOLD


def test_lama_unknown_device(fixtures_dir: Path) -> None:
    with pytest.raises(EngineError, match="unknown LaMa device"):
        LaMaInpaintEngine(_stub_weights(fixtures_dir), "tpu")


def test_auto_small_mask_selects_opencv() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0, 0] = 255
    assert float(np.count_nonzero(mask)) / mask.size < 0.03
    engine = get_engine("auto", mask, Settings(mask_area_threshold=0.03))
    assert isinstance(engine, OpenCVInpaintEngine)


def test_auto_large_mask_selects_lama(fixtures_dir: Path) -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:4, :4] = 255
    assert float(np.count_nonzero(mask)) / mask.size >= 0.03
    settings = Settings(
        lama_weights=_stub_weights(fixtures_dir),
        mask_area_threshold=0.03,
    )
    engine = get_engine("auto", mask, settings)
    assert isinstance(engine, LaMaInpaintEngine)


def test_user_engine_lama_overrides_small_mask(fixtures_dir: Path) -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[0, 0] = 255
    settings = Settings(
        lama_weights=_stub_weights(fixtures_dir),
        mask_area_threshold=0.03,
    )
    engine = get_engine("lama", mask, settings)
    assert isinstance(engine, LaMaInpaintEngine)


def test_tiny_resolution_does_not_crash(fixtures_dir: Path) -> None:
    image = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
    mask = np.array([[255, 0]], dtype=np.uint8)
    out = LaMaInpaintEngine(_stub_weights(fixtures_dir), "cpu").process(image, mask)
    assert out.shape == (1, 2, 3)
    assert out.dtype == np.uint8


def test_huge_resolution_tiles_without_crash(fixtures_dir: Path) -> None:
    image = np.full((520, 540, 3), 40, dtype=np.uint8)
    mask = np.zeros((520, 540), dtype=np.uint8)
    mask[10:80, 10:80] = 255
    out = LaMaInpaintEngine(_stub_weights(fixtures_dir), "cpu").process(image, mask)
    assert out.shape == image.shape
    assert out.dtype == np.uint8
    assert np.array_equal(out[mask == 0], image[mask == 0])


def test_image_processor_tiles_huge_opencv() -> None:
    image = np.full((600, 580, 3), 15, dtype=np.uint8)
    mask = np.zeros((600, 580), dtype=np.uint8)
    mask[20:60, 20:60] = 255
    out = ImageProcessor().process(image, mask, "opencv", Settings())
    assert out.shape == image.shape
    assert out.dtype == np.uint8


@pytest.mark.slow
def test_lama_real_weights_ssim_psnr(fixtures_dir: Path) -> None:
    weights = _real_weights()
    if weights is None:
        pytest.skip("real LaMa weights not found; run python scripts/download_models.py")
    image = read_image(fixtures_dir / "still_logo.png")
    mask = load_mask_png(fixtures_dir / "still_logo.mask.png")
    out = LaMaInpaintEngine(weights, "cpu").process(image, mask)
    assert out.shape == image.shape
    assert out.dtype == np.uint8
    assert np.array_equal(out[mask == 0], image[mask == 0])
    baseline_path = fixtures_dir / "still_logo_inpainted_lama.png"
    if baseline_path.is_file():
        baseline = read_image(baseline_path)
        assert _ssim(out, baseline) >= SSIM_MIN_THRESHOLD
        assert _psnr(out, baseline) >= PSNR_MIN_DB_THRESHOLD


@pytest.mark.gpu
def test_lama_cuda_provider_or_skip(fixtures_dir: Path) -> None:
    if not _cuda_available():
        pytest.skip("CUDA execution provider not available")
    image = np.full((16, 16, 3), 30, dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:10, 4:10] = 255
    out = LaMaInpaintEngine(_stub_weights(fixtures_dir), "cuda").process(image, mask)
    assert out.shape == image.shape
    assert out.dtype == np.uint8
