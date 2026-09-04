from __future__ import annotations

import inspect
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from watermark_remover.config import Settings
from watermark_remover.engines.base import InpaintEngine
from watermark_remover.exceptions import ProcessingCancelled
from watermark_remover.masks.base import MaskProvider
from watermark_remover.video.processor import VideoProcessor

_FPS = 10.0
_SIZE = (32, 24)


class _StaticMask(MaskProvider):
    def get_mask(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        del frame_idx
        return np.zeros(frame.shape[:2], dtype=np.uint8)


class _PassthroughEngine(InpaintEngine):
    def process(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        del mask
        return image


def _write_tiny_video(path: Path, frame_count: int = 5) -> None:
    width, height = _SIZE
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        _FPS,
        (width, height),
    )
    assert writer.isOpened(), "OpenCV could not open VideoWriter"
    try:
        for index in range(frame_count):
            frame = np.full((height, width, 3), index * 3, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def _settings(**overrides: object) -> Settings:
    payload = Settings().model_dump()
    payload.update(overrides)
    return Settings.model_construct(**payload)


def _fake_encode_ok(
    frames_dir: Path,
    audio_src: Path,
    output: Path,
    fps: float,
    crf: int,
    **_kwargs: object,
) -> None:
    del frames_dir, audio_src, fps, crf
    Path(output).write_bytes(b"ok")


def test_process_signature_unchanged() -> None:
    params = list(inspect.signature(VideoProcessor.process).parameters)
    assert params == [
        "self",
        "input_path",
        "mask_provider",
        "engine",
        "output_path",
        "progress",
        "cancel_token",
    ]


class _IdxMask(MaskProvider):
    def get_mask(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask[0, 0] = frame_idx
        return mask


def test_parallel_writes_frames_in_index_order_when_later_frames_finish_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    n_frames = 5
    src = tmp_path / "tiny.mp4"
    _write_tiny_video(src, frame_count=n_frames)
    dest = tmp_path / "out.mp4"
    written: list[int] = []

    class _ReverseFinishEngine(InpaintEngine):
        def process(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
            frame_idx = int(mask[0, 0])
            time.sleep(0.04 * (n_frames - frame_idx))
            return np.full_like(image, frame_idx)

    def fake_encode(
        frames_dir: Path,
        audio_src: Path,
        output: Path,
        fps: float,
        crf: int,
        **_kwargs: object,
    ) -> None:
        del audio_src, fps, crf
        paths = sorted(frames_dir.glob("frame_*.png"))
        assert [p.name for p in paths] == [f"frame_{i:08d}.png" for i in range(len(paths))]
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            assert image is not None
            written.append(int(image[0, 0, 0]))
        Path(output).write_bytes(b"ok")

    monkeypatch.setattr("watermark_remover.video.processor.encode_video", fake_encode)
    VideoProcessor(_settings(temporal_smoothing=False, max_workers=4)).process(
        src, _IdxMask(), _ReverseFinishEngine(), dest
    )
    assert written == list(range(n_frames))


def test_temporal_smoothing_does_not_construct_thread_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "tiny.mp4"
    _write_tiny_video(src, frame_count=3)
    dest = tmp_path / "out.mp4"
    monkeypatch.setattr("watermark_remover.video.processor.encode_video", _fake_encode_ok)
    with patch("watermark_remover.video.processor.ThreadPoolExecutor") as mocked:
        VideoProcessor(_settings(temporal_smoothing=True, max_workers=8)).process(
            src, _StaticMask(), _PassthroughEngine(), dest
        )
    mocked.assert_not_called()


def test_parallel_cancel_does_not_wait_for_all_submitted_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    n_frames = 12
    workers = 2
    sleep_s = 0.2
    src = tmp_path / "tiny.mp4"
    _write_tiny_video(src, frame_count=n_frames)
    dest = tmp_path / "out.mp4"
    token = {"requested": False}
    calls = 0
    lock = threading.Lock()
    started = threading.Event()

    class _SlowEngine(InpaintEngine):
        def process(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
            nonlocal calls
            started.set()
            with lock:
                calls += 1
            time.sleep(sleep_s)
            return image

    def fake_encode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("encode should not run after cancel")

    monkeypatch.setattr("watermark_remover.video.processor.encode_video", fake_encode)

    def cancel_soon() -> None:
        started.wait(timeout=2.0)
        time.sleep(0.05)
        token["requested"] = True

    canceller = threading.Thread(target=cancel_soon, daemon=True)
    canceller.start()
    t0 = time.perf_counter()
    with pytest.raises(ProcessingCancelled):
        VideoProcessor(_settings(temporal_smoothing=False, max_workers=workers)).process(
            src, _StaticMask(), _SlowEngine(), dest, cancel_token=token
        )
    elapsed = time.perf_counter() - t0
    canceller.join(timeout=1.0)
    serial_s = sleep_s * n_frames
    assert elapsed < serial_s * 0.5
    assert calls <= workers + 1
    assert not dest.exists()
    assert not dest.with_name(f"{dest.stem}.tmp{dest.suffix}").exists()


def test_parallel_caps_thread_pool_at_cpu_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "tiny.mp4"
    _write_tiny_video(src, frame_count=2)
    dest = tmp_path / "out.mp4"
    monkeypatch.setattr("watermark_remover.video.processor.encode_video", _fake_encode_ok)
    monkeypatch.setattr("watermark_remover.video.processor.os.cpu_count", lambda: 2)
    seen: list[int] = []
    real = ThreadPoolExecutor

    def wrapping(
        *args: object, max_workers: int | None = None, **kwargs: object
    ) -> ThreadPoolExecutor:
        assert max_workers is not None
        seen.append(int(max_workers))
        return real(*args, max_workers=max_workers, **kwargs)

    monkeypatch.setattr("watermark_remover.video.processor.ThreadPoolExecutor", wrapping)
    VideoProcessor(_settings(temporal_smoothing=False, max_workers=99)).process(
        src, _StaticMask(), _PassthroughEngine(), dest
    )
    assert seen == [2]


def test_parallel_shares_one_engine_and_overlaps_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "tiny.mp4"
    _write_tiny_video(src, frame_count=4)
    dest = tmp_path / "out.mp4"
    monkeypatch.setattr("watermark_remover.video.processor.encode_video", _fake_encode_ok)
    engine_ids: set[int] = set()
    barrier = threading.Barrier(2, timeout=2.0)

    class _OverlapEngine(InpaintEngine):
        def process(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
            del mask
            engine_ids.add(id(self))
            barrier.wait()
            return image

    engine = _OverlapEngine()
    VideoProcessor(_settings(temporal_smoothing=False, max_workers=2)).process(
        src, _StaticMask(), engine, dest
    )
    assert engine_ids == {id(engine)}


def test_parallel_progress_tracks_completion_not_submit_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    n_frames = 4
    src = tmp_path / "tiny.mp4"
    _write_tiny_video(src, frame_count=n_frames)
    dest = tmp_path / "out.mp4"
    seen: list[int] = []
    started = threading.Barrier(n_frames, timeout=5.0)

    class _ReverseFinishEngine(InpaintEngine):
        def process(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
            frame_idx = int(mask[0, 0])
            started.wait()
            time.sleep(0.05 * (n_frames - frame_idx))
            return image

    monkeypatch.setattr("watermark_remover.video.processor.os.cpu_count", lambda: n_frames)
    monkeypatch.setattr("watermark_remover.video.processor.encode_video", _fake_encode_ok)
    VideoProcessor(_settings(temporal_smoothing=False, max_workers=n_frames)).process(
        src,
        _IdxMask(),
        _ReverseFinishEngine(),
        dest,
        progress=lambda **kwargs: seen.append(int(kwargs["frame_idx"])),
    )
    assert seen[0] == n_frames - 1
    assert sorted(seen) == list(range(n_frames))


def test_process_start_logs_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "tiny.mp4"
    _write_tiny_video(src, frame_count=1)
    dest = tmp_path / "out.mp4"
    monkeypatch.setattr("watermark_remover.video.processor.os.cpu_count", lambda: 8)
    monkeypatch.setattr("watermark_remover.video.processor.encode_video", _fake_encode_ok)
    events: list[tuple[str, dict[str, object]]] = []

    class _CaptureLogger:
        def info(self, event: str, **kwargs: object) -> None:
            events.append((event, dict(kwargs)))

        def debug(self, *_args: object, **_kwargs: object) -> None:
            return None

        def warning(self, *_args: object, **_kwargs: object) -> None:
            return None

        def error(self, *_args: object, **_kwargs: object) -> None:
            return None

    monkeypatch.setattr(
        "watermark_remover.video.processor.structlog.get_logger",
        lambda *_args, **_kwargs: _CaptureLogger(),
    )
    VideoProcessor(_settings(temporal_smoothing=False, max_workers=3)).process(
        src, _StaticMask(), _PassthroughEngine(), dest
    )
    starts = [kwargs for event, kwargs in events if event == "video_process_start"]
    assert starts
    assert starts[0]["mode"] == "parallel (3 workers, temporal smoothing disabled)"
    assert starts[0]["engine"] == "_PassthroughEngine"

    events.clear()
    dest2 = tmp_path / "out2.mp4"
    VideoProcessor(_settings(temporal_smoothing=True, max_workers=3)).process(
        src, _StaticMask(), _PassthroughEngine(), dest2
    )
    starts = [kwargs for event, kwargs in events if event == "video_process_start"]
    assert starts
    assert starts[0]["mode"] == "sequential (temporal smoothing enabled)"
