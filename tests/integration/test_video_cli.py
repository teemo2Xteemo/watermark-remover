from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from watermark_remover.cli import app
from watermark_remover.io.video import probe_video
from watermark_remover.video.encode import find_ffmpeg

runner = CliRunner()
_HAS_FFMPEG = find_ffmpeg() is not None
_FPS_EPS = 0.05

pytestmark = pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not installed")


def _run_ffprobe(args: list[str]) -> subprocess.CompletedProcess[str] | None:
    exe = shutil.which("ffprobe")
    if exe is None:
        ffmpeg = find_ffmpeg()
        if ffmpeg is not None:
            sibling = Path(ffmpeg).with_name(f"ffprobe{Path(ffmpeg).suffix}")
            if sibling.is_file():
                exe = str(sibling)
    if exe is None:
        return None
    return subprocess.run([exe, *args], check=False, capture_output=True, text=True)


def _ffmpeg_info(path: Path) -> str:
    ffmpeg = find_ffmpeg()
    assert ffmpeg is not None
    completed = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    return f"{completed.stderr}\n{completed.stdout}"


def _has_audio_stream(path: Path) -> bool:
    probed = _run_ffprobe(
        [
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
            str(path),
        ]
    )
    if probed is not None:
        assert probed.returncode == 0, probed.stderr
        payload = json.loads(probed.stdout or "{}")
        streams = payload.get("streams") or []
        return any(stream.get("codec_type") == "audio" for stream in streams)
    return bool(re.search(r"Audio:\s+\w+", _ffmpeg_info(path)))


def _fps_of(path: Path) -> float:
    probed = _run_ffprobe(
        [
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate",
            "-of",
            "json",
            str(path),
        ]
    )
    if probed is not None:
        assert probed.returncode == 0, probed.stderr
        payload = json.loads(probed.stdout or "{}")
        rate = payload["streams"][0]["r_frame_rate"]
        num_s, den_s = str(rate).split("/", 1)
        return float(num_s) / float(den_s)
    info = _ffmpeg_info(path)
    match = re.search(r"(\d+(?:\.\d+)?)\s+fps", info)
    assert match, f"could not parse fps from ffmpeg output:\n{info}"
    return float(match.group(1))


def _clip_with_audio(src: Path, tmp_path: Path) -> Path:
    dest = tmp_path / "clip_with_audio.mp4"
    if _has_audio_stream(src):
        dest.write_bytes(src.read_bytes())
        return dest
    meta = probe_video(src)
    duration = meta.duration if meta.duration > 0 else 2.0
    ffmpeg = find_ffmpeg()
    assert ffmpeg is not None
    completed = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(src),
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration}",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(dest),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return dest


def test_video_cli_preserves_audio_fps_and_exits_zero(fixtures_dir: Path, tmp_path: Path) -> None:
    src = _clip_with_audio(fixtures_dir / "clip_5s.mp4", tmp_path)
    output = tmp_path / "out.mp4"
    input_fps = _fps_of(src)
    result = runner.invoke(
        app,
        [
            "--input",
            str(src),
            "--mask",
            str(fixtures_dir / "clip_5s.mask.png"),
            "--engine",
            "opencv",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.is_file()
    assert _has_audio_stream(output)
    assert _fps_of(output) == pytest.approx(input_fps, abs=_FPS_EPS)


def test_video_cli_default_output_stem_inpainted(fixtures_dir: Path, tmp_path: Path) -> None:
    src = tmp_path / "clip_5s.mp4"
    prepared = _clip_with_audio(fixtures_dir / "clip_5s.mp4", tmp_path)
    src.write_bytes(prepared.read_bytes())
    result = runner.invoke(
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
    default_out = tmp_path / "clip_5s_inpainted.mp4"
    assert default_out.is_file()
    assert not src.samefile(default_out)
    assert src.is_file()
