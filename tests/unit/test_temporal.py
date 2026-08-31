from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from watermark_remover.config import Settings
from watermark_remover.engines.base import InpaintEngine
from watermark_remover.masks.base import MaskProvider
from watermark_remover.video import temporal as temporal_mod
from watermark_remover.video.processor import VideoProcessor
from watermark_remover.video.temporal import TemporalSmoother


class _PassthroughEngine(InpaintEngine):
    def process(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        del mask
        return image


class _StaticMask(MaskProvider):
    def get_mask(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        del frame_idx
        return np.zeros(frame.shape[:2], dtype=np.uint8)


def _frames(size: int = 32) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current = np.arange(size * size * 3, dtype=np.uint8).reshape(size, size, 3)
    prev = current.copy()
    prev[8:24, 8:24] = 0
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    return prev, current, mask


def _write_tiny_video(path: Path, frame_count: int = 3) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    assert writer.isOpened(), "OpenCV could not open VideoWriter"
    try:
        for index in range(frame_count):
            frame = np.full((24, 32, 3), index * 3, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def _fake_encode(
    frames_dir: Path,
    audio_src: Path,
    output: Path,
    fps: float,
    crf: int,
    **_kwargs: object,
) -> None:
    del frames_dir, audio_src, fps, crf
    Path(output).write_bytes(b"ok")


def test_unmasked_pixels_are_byte_identical() -> None:
    prev, current, mask = _frames()
    out = TemporalSmoother(Settings(raft_enabled=False)).apply(prev, current, mask)
    assert np.array_equal(out[mask == 0], current[mask == 0])


def test_masked_region_changes_when_frames_differ() -> None:
    prev, current, mask = _frames()
    out = TemporalSmoother(Settings(raft_enabled=False)).apply(prev, current, mask)
    assert not np.array_equal(out[mask == 255], current[mask == 255])


def test_empty_mask_is_noop() -> None:
    prev, current, _mask = _frames()
    empty = np.zeros(current.shape[:2], dtype=np.uint8)
    out = TemporalSmoother(Settings(raft_enabled=False)).apply(prev, current, empty)
    assert np.array_equal(out, current)


def test_raft_not_imported_at_module_level() -> None:
    tree = ast.parse(Path(temporal_mod.__file__).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert "torch" not in names
    assert "torchvision" not in names
    assert "raft" not in names


def test_raft_not_imported_when_flag_off() -> None:
    before = set(sys.modules)
    prev, current, mask = _frames()
    TemporalSmoother(Settings(raft_enabled=False)).apply(prev, current, mask)
    new_modules = set(sys.modules) - before
    assert not any(
        "raft" in name.lower() or name == "torch" or name.startswith("torchvision")
        for name in new_modules
    )


def test_default_smoother_is_farneback_not_raft() -> None:
    assert Settings().raft_enabled is False
    assert Settings().temporal_smoothing is True
    module_src = inspect.getsource(temporal_mod)
    assert "calcOpticalFlowFarneback" in module_src
    assert "if self._raft_enabled" in inspect.getsource(TemporalSmoother.apply)


def test_processor_does_not_call_apply_when_temporal_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "tiny.mp4"
    _write_tiny_video(src, frame_count=3)
    dest = tmp_path / "out.mp4"
    monkeypatch.setattr("watermark_remover.video.processor.encode_video", _fake_encode)
    with patch("watermark_remover.video.processor.TemporalSmoother") as mocked:
        VideoProcessor(Settings(temporal_smoothing=False, max_workers=1)).process(
            src, _StaticMask(), _PassthroughEngine(), dest
        )
    mocked.assert_not_called()
    mocked.return_value.apply.assert_not_called()


def test_processor_calls_apply_after_first_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "tiny.mp4"
    _write_tiny_video(src, frame_count=3)
    dest = tmp_path / "out.mp4"
    monkeypatch.setattr("watermark_remover.video.processor.encode_video", _fake_encode)
    instance = MagicMock()
    instance.apply.side_effect = lambda prev, current, mask: current
    with patch("watermark_remover.video.processor.TemporalSmoother", return_value=instance):
        VideoProcessor(Settings(temporal_smoothing=True, max_workers=2)).process(
            src, _StaticMask(), _PassthroughEngine(), dest
        )
    assert instance.apply.call_count == 2


def test_processor_skips_apply_on_single_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "tiny.mp4"
    _write_tiny_video(src, frame_count=1)
    dest = tmp_path / "out.mp4"
    monkeypatch.setattr("watermark_remover.video.processor.encode_video", _fake_encode)
    instance = MagicMock()
    instance.apply.side_effect = lambda prev, current, mask: current
    with patch("watermark_remover.video.processor.TemporalSmoother", return_value=instance):
        VideoProcessor(Settings(temporal_smoothing=True, max_workers=1)).process(
            src, _StaticMask(), _PassthroughEngine(), dest
        )
    instance.apply.assert_not_called()


class _FillEngine(InpaintEngine):
    def process(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        del mask
        return np.full_like(image, 77)


def test_processor_apply_receives_inpainted_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "tiny.mp4"
    _write_tiny_video(src, frame_count=3)
    dest = tmp_path / "out.mp4"
    monkeypatch.setattr("watermark_remover.video.processor.encode_video", _fake_encode)
    seen: list[int] = []

    def fake_apply(prev: np.ndarray, current: np.ndarray, mask: np.ndarray) -> np.ndarray:
        del prev, mask
        seen.append(int(current[0, 0, 0]))
        return current

    instance = MagicMock()
    instance.apply.side_effect = fake_apply
    with patch("watermark_remover.video.processor.TemporalSmoother", return_value=instance):
        VideoProcessor(Settings(temporal_smoothing=True, max_workers=1)).process(
            src, _StaticMask(), _FillEngine(), dest
        )
    assert seen == [77, 77]

