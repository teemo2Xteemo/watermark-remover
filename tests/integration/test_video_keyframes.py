from __future__ import annotations

from pathlib import Path

import pytest

from watermark_remover.config import Settings
from watermark_remover.engines.opencv_engine import OpenCVInpaintEngine
from watermark_remover.masks.manual import KeyframeMaskProvider
from watermark_remover.masks.serialize import load_keyframes
from watermark_remover.video.encode import find_ffmpeg
from watermark_remover.video.processor import VideoProcessor

_HAS_FFMPEG = find_ffmpeg() is not None

pytestmark = pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not installed")


def test_keyframe_fixture_process_does_not_crash(
    fixtures_dir: Path, tmp_path: Path
) -> None:
    src = tmp_path / "clip_5s.mp4"
    src.write_bytes((fixtures_dir / "clip_5s.mp4").read_bytes())
    loaded = load_keyframes(fixtures_dir / "clip_5s.keyframes.json", (48, 64))
    assert len(loaded) >= 2
    assert loaded[0][0] != loaded[1][0]
    assert not (loaded[0][1] == loaded[1][1]).all()
    dest = tmp_path / "clip_5s_inpainted.mp4"
    provider = KeyframeMaskProvider(loaded, fps=10.0)
    result = VideoProcessor(
        Settings(max_workers=1, temporal_smoothing=False, keep_audio=True)
    ).process(
        src,
        provider,
        OpenCVInpaintEngine(radius=3, method="telea"),
        dest,
    )
    assert result == dest
    assert dest.is_file()
    assert dest.stat().st_size > 0
    assert src.read_bytes() == (fixtures_dir / "clip_5s.mp4").read_bytes()
