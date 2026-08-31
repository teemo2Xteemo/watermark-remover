from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from watermark_remover.config import clear_settings_cache

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture(autouse=True)
def pin_seeds() -> None:
    np.random.seed(0)
    cv2.setRNGSeed(0)
    clear_settings_cache()
    yield
    clear_settings_cache()


@pytest.fixture
def has_ffmpeg() -> bool:
    from watermark_remover.video.encode import find_ffmpeg

    return find_ffmpeg() is not None
