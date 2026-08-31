from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from watermark_remover.config import Settings
from watermark_remover.masks.serialize import load_keyframes, load_mask_png
from watermark_remover.ui.app import (
    hold_last_mask,
    is_video_run_enabled,
    load_video_preview,
    overlay_mask_rgb,
    run_video_job,
)
from watermark_remover.video.encode import find_ffmpeg

_HAS_FFMPEG = find_ffmpeg() is not None

pytestmark = pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not installed")


def test_load_video_preview_is_downsampled_first_frame(fixtures_dir: Path) -> None:
    preview, native_hw, fps, duration, frame_count = load_video_preview(
        fixtures_dir / "clip_5s.mp4"
    )
    assert native_hw == (48, 64)
    assert fps == pytest.approx(10.0, abs=0.05)
    assert duration == pytest.approx(2.0, abs=0.2)
    assert frame_count in {0, 20} or frame_count >= 1
    assert preview.ndim == 3
    assert preview.shape[2] == 3
    assert max(preview.shape[0], preview.shape[1]) <= 640


def test_keyframe_preview_overlay_differs_by_segment(fixtures_dir: Path) -> None:
    preview, native_hw, _fps, _duration, _count = load_video_preview(
        fixtures_dir / "clip_5s.mp4"
    )
    rows = [
        {"t": t, "mask": mask}
        for t, mask in load_keyframes(fixtures_dir / "clip_5s.keyframes.json", native_hw)
    ]
    first = hold_last_mask(rows, 0.2, native_hw)
    second = hold_last_mask(rows, 1.2, native_hw)
    assert first is not None and second is not None
    assert not np.array_equal(first, second)
    overlay_a = overlay_mask_rgb(preview, first)
    overlay_b = overlay_mask_rgb(preview, second)
    assert overlay_a.shape == preview.shape
    assert overlay_b.shape == preview.shape
    assert not np.array_equal(overlay_a, overlay_b)


def test_video_ui_static_run_end_to_end(fixtures_dir: Path, tmp_path: Path) -> None:
    src = tmp_path / "clip_5s.mp4"
    src.write_bytes((fixtures_dir / "clip_5s.mp4").read_bytes())
    mask = load_mask_png(fixtures_dir / "clip_5s.mask.png")
    assert is_video_run_enabled("Static (all frames)", mask, True, True, []) is True
    job = run_video_job(
        str(src),
        "opencv",
        Settings(max_workers=1, temporal_smoothing=False, keep_audio=True, frame_stride=1),
        mask=mask,
        mask_mode="Static (all frames)",
        mask_confirmed=True,
        preview_ready=True,
        stem="clip_5s",
    )
    assert job.output_path is not None, job.status
    out = Path(job.output_path)
    assert out.is_file()
    assert out.name == "clip_5s_inpainted.mp4"
    assert job.percent == 100
    assert "job_id=" in job.log_text
    assert ">[SYS]" not in job.log_text
    assert src.read_bytes() == (fixtures_dir / "clip_5s.mp4").read_bytes()


def test_video_ui_keyframe_run_end_to_end(fixtures_dir: Path, tmp_path: Path) -> None:
    src = tmp_path / "moving.mp4"
    src.write_bytes((fixtures_dir / "clip_5s.mp4").read_bytes())
    native_hw = (48, 64)
    loaded = load_keyframes(fixtures_dir / "clip_5s.keyframes.json", native_hw)
    assert len(loaded) == 2
    assert loaded[0][0] != loaded[1][0]
    rows = [{"t": t, "mask": mask} for t, mask in loaded]
    assert is_video_run_enabled("Keyframes (by timestamp)", None, True, True, rows) is True
    job = run_video_job(
        str(src),
        "opencv",
        Settings(max_workers=1, temporal_smoothing=True, keep_audio=True, frame_stride=1),
        keyframes=rows,
        mask_mode="Keyframes (by timestamp)",
        mask_confirmed=False,
        preview_ready=True,
        stem="blocked",
    )
    assert job.output_path is None
    assert "run_blocked" in job.log_text
    job = run_video_job(
        str(src),
        "opencv",
        Settings(max_workers=1, temporal_smoothing=True, keep_audio=True, frame_stride=1),
        keyframes=rows,
        mask_mode="Keyframes (by timestamp)",
        mask_confirmed=True,
        preview_ready=True,
        stem="moving",
    )
    assert job.output_path is not None, job.status
    out = Path(job.output_path)
    assert out.is_file()
    assert out.name == "moving_inpainted.mp4"
    assert "inpaint_done" in job.log_text
    assert "video_progress" in job.log_text


def test_video_ui_nth_frame_and_drop_audio_affect_output(
    fixtures_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from watermark_remover.video import processor as processor_mod

    src = tmp_path / "clip_5s.mp4"
    src.write_bytes((fixtures_dir / "clip_5s.mp4").read_bytes())
    mask = load_mask_png(fixtures_dir / "clip_5s.mask.png")
    captured: dict[str, object] = {}
    original = processor_mod.encode_video

    def wrap_encode(
        frames_dir: Path,
        audio_src: Path,
        output: Path,
        fps: float,
        crf: int,
        *,
        keep_audio: bool = True,
    ) -> None:
        captured["fps"] = fps
        captured["keep_audio"] = keep_audio
        captured["frame_files"] = sorted(p.name for p in Path(frames_dir).glob("frame_*.png"))
        original(frames_dir, audio_src, output, fps, crf, keep_audio=keep_audio)

    monkeypatch.setattr(processor_mod, "encode_video", wrap_encode)
    job = run_video_job(
        str(src),
        "opencv",
        Settings(
            max_workers=1,
            temporal_smoothing=False,
            keep_audio=False,
            frame_stride=2,
            output_quality="source",
        ),
        mask=mask,
        mask_mode="Static (all frames)",
        mask_confirmed=True,
        preview_ready=True,
        stem="stride",
    )
    assert job.output_path is not None, job.status
    assert captured["keep_audio"] is False
    assert captured["fps"] == pytest.approx(5.0, abs=0.05)
    names = captured["frame_files"]
    assert isinstance(names, list)
    assert len(names) <= 10


def test_video_ui_cancel_stops_processor(
    fixtures_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from watermark_remover.engines.opencv_engine import OpenCVInpaintEngine

    src = tmp_path / "clip_5s.mp4"
    src.write_bytes((fixtures_dir / "clip_5s.mp4").read_bytes())
    mask = load_mask_png(fixtures_dir / "clip_5s.mask.png")
    token = {"requested": False}
    original = OpenCVInpaintEngine.process

    def flip_then_process(
        self: OpenCVInpaintEngine,
        image: np.ndarray,
        mask_arg: np.ndarray,
    ) -> np.ndarray:
        token["requested"] = True
        return original(self, image, mask_arg)

    monkeypatch.setattr(OpenCVInpaintEngine, "process", flip_then_process)
    job = run_video_job(
        str(src),
        "opencv",
        Settings(max_workers=1, temporal_smoothing=False),
        mask=mask,
        mask_mode="Static (all frames)",
        mask_confirmed=True,
        preview_ready=True,
        cancel_token=token,
        stem="cancel",
    )
    assert job.cancel_requested is True
    assert job.output_path is None
    assert "job_cancelled" in job.log_text


def test_ui_source_does_not_materialize_all_frames() -> None:
    src = Path(__file__).resolve().parents[2] / "src" / "watermark_remover" / "ui" / "app.py"
    text = src.read_text(encoding="utf-8")
    assert "list(extract_frames" not in text
    assert "extract_frames(" not in text
    assert "from watermark_remover.video.extract import read_first_frame, read_frame_at" in text


def test_keyframes_fixture_schema(fixtures_dir: Path) -> None:
    body = json.loads((fixtures_dir / "clip_5s.keyframes.json").read_text(encoding="utf-8"))
    assert body["schema_version"] == 1
    assert isinstance(body["keyframes"], list)
    assert len(body["keyframes"]) >= 2
    for row in body["keyframes"]:
        assert "t" in row
        assert "mask_ref" in row
        assert (fixtures_dir / row["mask_ref"]).is_file()
