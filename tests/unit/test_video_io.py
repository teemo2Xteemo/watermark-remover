from __future__ import annotations

import inspect
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from watermark_remover.cli import app
from watermark_remover.config import Settings
from watermark_remover.engines.base import InpaintEngine
from watermark_remover.exceptions import EngineError, ProcessingCancelled, ResourceLimitError
from watermark_remover.io.video import VideoMetadata, probe_video
from watermark_remover.masks.base import MaskProvider
from watermark_remover.video.encode import encode_video, find_ffmpeg
from watermark_remover.video.extract import extract_frames, read_first_frame
from watermark_remover.video.processor import VideoProcessor, capped_max_workers, target_frame_size

_FPS = 10.0
_SIZE = (32, 24)  # width, height


class _PassthroughEngine(InpaintEngine):
    def process(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        del mask
        return image


class _StaticMask(MaskProvider):
    def get_mask(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        del frame_idx
        return np.zeros(frame.shape[:2], dtype=np.uint8)


class _BoomEngine(InpaintEngine):
    def process(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        del image, mask
        raise EngineError("boom")


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
            frame[2:10, 2:14] = (255, 255, 255)
            writer.write(frame)
    finally:
        writer.release()


def test_extract_frames_is_generator(tmp_path: Path) -> None:
    assert inspect.isgeneratorfunction(extract_frames)
    src = tmp_path / "tiny.mp4"
    _write_tiny_video(src, frame_count=4)
    iterator = extract_frames(src)
    assert inspect.isgenerator(iterator)
    frames = list(iterator)
    assert [idx for idx, _ in frames] == [0, 1, 2, 3]
    for _, frame in frames:
        assert frame.dtype == np.uint8
        assert frame.shape == (24, 32, 3)


def test_processor_does_not_materialize_all_frames() -> None:
    source = inspect.getsource(VideoProcessor)
    assert "list(extract_frames" not in source


def test_capped_max_workers_never_exceeds_cpu_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("watermark_remover.video.processor.os.cpu_count", lambda: 4)
    assert capped_max_workers(9999) == 4
    assert capped_max_workers(2) == 2
    settings = Settings.model_construct(**{**Settings().model_dump(), "max_workers": 10_000})
    assert capped_max_workers(settings.max_workers) == 4


def test_encode_video_stream_copies_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "frame_00000000.png").write_bytes(b"png")
    audio_src = tmp_path / "src.mp4"
    audio_src.write_bytes(b"src")
    output = tmp_path / "out.mp4"
    calls: list[list[str]] = []

    def fake_which(name: str) -> str | None:
        return name if name == "ffmpeg" else None

    def fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        Path(cmd[-1]).write_bytes(b"encoded")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("watermark_remover.video.encode.shutil.which", fake_which)
    monkeypatch.setattr("watermark_remover.video.encode.subprocess.run", fake_run)
    encode_video(frames_dir, audio_src, output, fps=10.0, crf=23)
    assert calls
    cmd = calls[0]
    assert "-c:a" in cmd
    assert cmd[cmd.index("-c:a") + 1] == "copy"
    assert "1:a:0?" in cmd
    assert "-crf" in cmd
    assert cmd[cmd.index("-crf") + 1] == "23"
    assert "-framerate" in cmd
    assert cmd[cmd.index("-framerate") + 1] == "10"
    assert output.read_bytes() == b"encoded"


def test_encode_video_retries_audio_reencode_when_copy_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "frame_00000000.png").write_bytes(b"png")
    audio_src = tmp_path / "src.mp4"
    audio_src.write_bytes(b"src")
    output = tmp_path / "out.mp4"

    def fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        audio_codec = cmd[cmd.index("-c:a") + 1]
        if audio_codec == "copy":
            return subprocess.CompletedProcess(cmd, 1, "", "codec not currently supported")
        Path(cmd[-1]).write_bytes(b"ok")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("watermark_remover.video.encode.shutil.which", lambda name: name)
    monkeypatch.setattr("watermark_remover.video.encode.subprocess.run", fake_run)
    encode_video(frames_dir, audio_src, output, fps=12.5, crf=18)
    assert output.read_bytes() == b"ok"


def test_encode_video_optional_audio_map_allows_source_without_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "frame_00000000.png").write_bytes(b"png")
    silent = tmp_path / "silent.mp4"
    _write_tiny_video(silent, frame_count=2)
    output = tmp_path / "out.mp4"
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        Path(cmd[-1]).write_bytes(b"video-only")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("watermark_remover.video.encode.shutil.which", lambda name: name)
    monkeypatch.setattr("watermark_remover.video.encode.subprocess.run", fake_run)
    encode_video(frames_dir, silent, output, fps=10.0, crf=23)
    assert "1:a:0?" in calls[0]
    assert calls[0][calls[0].index("-c:a") + 1] == "copy"
    assert output.read_bytes() == b"video-only"


def test_resource_limit_raised_when_max_ram_mb_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "tiny.mp4"
    _write_tiny_video(src)
    dest = tmp_path / "out.mp4"
    huge = VideoMetadata(
        fps=10.0,
        width=20000,
        height=20000,
        duration=1.0,
        codec="mp4v",
        frame_count=2,
        has_audio=False,
    )
    monkeypatch.setattr("watermark_remover.video.processor.probe_video", lambda _p: huge)
    processor = VideoProcessor(Settings(max_ram_mb=1, max_workers=8))
    with pytest.raises(ResourceLimitError, match="max_ram_mb"):
        processor.process(src, _StaticMask(), _PassthroughEngine(), dest)
    assert not dest.exists()


def test_resource_limit_skipped_when_max_ram_mb_unbounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "tiny.mp4"
    _write_tiny_video(src, frame_count=2)
    dest = tmp_path / "out.mp4"
    huge = VideoMetadata(
        fps=10.0,
        width=20000,
        height=20000,
        duration=1.0,
        codec="mp4v",
        frame_count=2,
        has_audio=False,
    )
    monkeypatch.setattr("watermark_remover.video.processor.probe_video", lambda _p: huge)

    def fake_encode(
        frames_dir: Path,
        audio_src: Path,
        output: Path,
        fps: float,
        crf: int,
        **_kwargs: object,
    ) -> None:
        del frames_dir, audio_src, fps, crf
        Path(output).write_bytes(b"ok")

    monkeypatch.setattr("watermark_remover.video.processor.encode_video", fake_encode)
    processor = VideoProcessor(Settings(max_ram_mb=None, max_workers=1))
    result = processor.process(src, _StaticMask(), _PassthroughEngine(), dest)
    assert result == dest
    assert dest.is_file()


def test_atomic_output_not_left_on_failure(tmp_path: Path) -> None:
    src = tmp_path / "tiny.mp4"
    _write_tiny_video(src, frame_count=3)
    dest = tmp_path / "final.mp4"
    processor = VideoProcessor(Settings(max_workers=1))
    with pytest.raises(EngineError, match="boom"):
        processor.process(src, _StaticMask(), _BoomEngine(), dest)
    assert not dest.exists()
    assert not dest.with_name(f"{dest.stem}.tmp{dest.suffix}").exists()


def test_frames_temp_dir_removed_when_encode_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "tiny.mp4"
    _write_tiny_video(src, frame_count=2)
    dest = tmp_path / "out.mp4"
    created: list[Path] = []
    original = tempfile.mkdtemp

    def fake_mkdtemp(*args: object, **kwargs: object) -> str:
        options = dict(kwargs)
        options["dir"] = str(tmp_path)
        path = original(*args, **options)
        created.append(Path(path))
        return path

    monkeypatch.setattr("watermark_remover.video.processor.tempfile.mkdtemp", fake_mkdtemp)

    def boom_encode(*_args: object, **_kwargs: object) -> None:
        raise EngineError("ffmpeg not found on PATH")

    monkeypatch.setattr("watermark_remover.video.processor.encode_video", boom_encode)
    with pytest.raises(EngineError, match="ffmpeg not found"):
        VideoProcessor(Settings(max_workers=1)).process(
            src, _StaticMask(), _PassthroughEngine(), dest
        )
    assert created
    assert all(not path.exists() for path in created)
    assert list(tmp_path.glob("watermark_remover_frames_*")) == []
    assert not dest.exists()


def test_frame_failed_log_does_not_hide_engine_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "tiny.mp4"
    _write_tiny_video(src, frame_count=2)
    dest = tmp_path / "out.mp4"

    class _Cp1252Logger:
        def info(self, *_args: object, **_kwargs: object) -> None:
            return None

        def debug(self, *_args: object, **_kwargs: object) -> None:
            return None

        def warning(self, *_args: object, **_kwargs: object) -> None:
            return None

        def error(self, *_args: object, **kwargs: object) -> None:
            raise UnicodeEncodeError("cp1252", "x", 0, 1, "console")

    monkeypatch.setattr(
        "watermark_remover.video.processor.structlog.get_logger",
        lambda *_args, **_kwargs: _Cp1252Logger(),
    )
    with pytest.raises(EngineError, match="boom"):
        VideoProcessor(Settings(max_workers=1)).process(src, _StaticMask(), _BoomEngine(), dest)
    assert not dest.exists()


def test_progress_callback_invoked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "tiny.mp4"
    _write_tiny_video(src, frame_count=3)
    dest = tmp_path / "out.mp4"
    seen: list[dict[str, object]] = []

    def fake_encode(
        frames_dir: Path,
        audio_src: Path,
        output: Path,
        fps: float,
        crf: int,
        **_kwargs: object,
    ) -> None:
        del frames_dir, audio_src, fps, crf
        Path(output).write_bytes(b"ok")

    monkeypatch.setattr("watermark_remover.video.processor.encode_video", fake_encode)
    VideoProcessor(Settings(max_workers=2)).process(
        src,
        _StaticMask(),
        _PassthroughEngine(),
        dest,
        progress=lambda **kwargs: seen.append(kwargs),
    )
    assert len(seen) == 3
    assert {item["frame_idx"] for item in seen} == {0, 1, 2}


def test_probe_video_reads_header_not_payload(tmp_path: Path) -> None:
    src = tmp_path / "tiny.mp4"
    _write_tiny_video(src, frame_count=5)
    meta = probe_video(src)
    assert meta.width == 32
    assert meta.height == 24
    assert meta.fps == pytest.approx(_FPS, abs=0.05)
    assert meta.frame_count in {0, 5} or meta.frame_count >= 1


def test_cli_video_uses_stem_inpainted_output(tmp_path: Path, fixtures_dir: Path) -> None:
    from typer.testing import CliRunner

    src = tmp_path / "scene.mp4"
    src.write_bytes((fixtures_dir / "clip_5s.mp4").read_bytes())
    with patch.object(
        VideoProcessor,
        "process",
        return_value=tmp_path / "scene_inpainted.mp4",
    ) as mocked:
        result = CliRunner().invoke(
            app,
            [
                "--input",
                str(src),
                "--mask",
                str(fixtures_dir / "clip_5s.mask.png"),
                "--engine",
                "opencv",
            ],
        )
    assert result.exit_code == 0, result.output
    assert mocked.call_args.args[3].name == "scene_inpainted.mp4"


def test_find_ffmpeg_prefers_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "watermark_remover.video.encode.shutil.which",
        lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None,
    )
    monkeypatch.setattr(
        "watermark_remover.video.encode._bundled_ffmpeg",
        lambda: "/bundled/ffmpeg",
    )
    assert find_ffmpeg() == "/usr/bin/ffmpeg"


def test_find_ffmpeg_falls_back_to_bundled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("watermark_remover.video.encode.shutil.which", lambda _name: None)
    monkeypatch.setattr(
        "watermark_remover.video.encode._bundled_ffmpeg",
        lambda: "/bundled/ffmpeg",
    )
    assert find_ffmpeg() == "/bundled/ffmpeg"


def test_find_ffmpeg_none_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("watermark_remover.video.encode.shutil.which", lambda _name: None)
    monkeypatch.setattr("watermark_remover.video.encode._bundled_ffmpeg", lambda: None)
    assert find_ffmpeg() is None


def test_encode_video_omits_audio_when_keep_audio_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "frame_00000000.png").write_bytes(b"png")
    audio_src = tmp_path / "src.mp4"
    audio_src.write_bytes(b"src")
    output = tmp_path / "out.mp4"
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        Path(cmd[-1]).write_bytes(b"silent")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("watermark_remover.video.encode.shutil.which", lambda name: name)
    monkeypatch.setattr("watermark_remover.video.encode.subprocess.run", fake_run)
    encode_video(frames_dir, audio_src, output, fps=10.0, crf=23, keep_audio=False)
    assert calls
    cmd = calls[0]
    assert "-an" in cmd
    assert "1:a:0?" not in cmd


def test_target_frame_size_never_upscales() -> None:
    settings = Settings(output_quality="720p")
    assert target_frame_size(640, 480, settings) == (640, 480)
    down = target_frame_size(1920, 1080, Settings(output_quality="720p"))
    assert down[1] == 720
    assert down[0] == 1280
    source = target_frame_size(1920, 1080, Settings(output_quality="source"))
    assert source == (1920, 1080)


def test_read_first_frame_does_not_yield_all(tmp_path: Path) -> None:
    src = tmp_path / "tiny.mp4"
    _write_tiny_video(src, frame_count=6)
    frame = read_first_frame(src)
    assert frame.shape == (24, 32, 3)
    assert frame.dtype == np.uint8


def test_processor_honors_frame_stride(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "tiny.mp4"
    _write_tiny_video(src, frame_count=6)
    dest = tmp_path / "out.mp4"
    seen: list[int] = []
    encode_fps: list[float] = []

    class _CountEngine(InpaintEngine):
        def process(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
            del mask
            return image

    class _IdxMask(MaskProvider):
        def get_mask(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
            seen.append(frame_idx)
            return np.zeros(frame.shape[:2], dtype=np.uint8)

    def fake_encode(
        frames_dir: Path,
        audio_src: Path,
        output: Path,
        fps: float,
        crf: int,
        **_kwargs: object,
    ) -> None:
        del frames_dir, audio_src, crf
        encode_fps.append(fps)
        Path(output).write_bytes(b"ok")

    monkeypatch.setattr("watermark_remover.video.processor.encode_video", fake_encode)
    VideoProcessor(Settings(max_workers=1, temporal_smoothing=False, frame_stride=2)).process(
        src, _IdxMask(), _CountEngine(), dest
    )
    assert seen == [0, 2, 4]
    assert encode_fps[0] == pytest.approx(5.0)


def test_processor_cancel_stops_and_cleans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "tiny.mp4"
    _write_tiny_video(src, frame_count=4)
    dest = tmp_path / "out.mp4"
    token = {"requested": False}

    class _FlipEngine(InpaintEngine):
        def process(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
            del mask
            token["requested"] = True
            return image

    def fake_encode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("encode should not run after cancel")

    monkeypatch.setattr("watermark_remover.video.processor.encode_video", fake_encode)
    with pytest.raises(ProcessingCancelled):
        VideoProcessor(Settings(max_workers=1, temporal_smoothing=False)).process(
            src, _StaticMask(), _FlipEngine(), dest, cancel_token=token
        )
    assert not dest.exists()
    assert not dest.with_name(f"{dest.stem}.tmp{dest.suffix}").exists()

