from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import structlog

from watermark_remover.exceptions import EngineError, InputValidationError

_FRAME_GLOB = "frame_*.png"
_FRAME_PATTERN = "frame_%08d.png"


def encode_video(
    frames_dir: Path,
    audio_src: Path,
    output: Path,
    fps: float,
    crf: int,
) -> None:
    """Mux numbered PNG frames + original audio into `output`.

    Audio is stream-copied (`-c:a copy`). If the source has no audio track, the
    optional map is skipped and the file is video-only. If copy fails because the
    codec is incompatible with the output container, audio is re-encoded as a fallback.
    """
    if not _list_frame_files(Path(frames_dir)):
        raise EngineError(f"no frames to encode in {Path(frames_dir).name}")
    if fps <= 0:
        raise InputValidationError(f"invalid fps for encode: {fps}")
    dest = Path(output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffmpeg_bin()
    vcodec = _video_codec_for(dest.suffix)

    copy_cmd = _build_cmd(
        ffmpeg=ffmpeg,
        frames_dir=Path(frames_dir),
        audio_src=Path(audio_src),
        output=dest,
        fps=fps,
        crf=crf,
        vcodec=vcodec,
        audio_codec="copy",
    )
    completed = _run_ffmpeg(copy_cmd)
    if completed.returncode == 0:
        return

    log = structlog.get_logger("watermark_remover")
    log.info(
        "audio_copy_failed_retry_reencode",
        output_path=dest.name,
        error=(completed.stderr or "")[-500:],
    )
    retry_cmd = _build_cmd(
        ffmpeg=ffmpeg,
        frames_dir=Path(frames_dir),
        audio_src=Path(audio_src),
        output=dest,
        fps=fps,
        crf=crf,
        vcodec=vcodec,
        audio_codec=_audio_fallback_codec(dest.suffix),
    )
    retried = _run_ffmpeg(retry_cmd)
    if retried.returncode != 0:
        raise EngineError(_ffmpeg_error(retried))


def _list_frame_files(frames_dir: Path) -> list[Path]:
    return sorted(frames_dir.glob(_FRAME_GLOB))


def _ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise EngineError("ffmpeg not found on PATH")
    return exe


def _video_codec_for(suffix: str) -> str:
    kind = suffix.lower()
    if kind in {".mp4", ".mov"}:
        return "libx264"
    if kind == ".webm":
        return "libvpx-vp9"
    raise InputValidationError(
        f"unsupported output video format '{suffix}'; expected MP4, MOV, or WEBM"
    )


def _audio_fallback_codec(suffix: str) -> str:
    if suffix.lower() == ".webm":
        return "libopus"
    return "aac"


def _build_cmd(
    *,
    ffmpeg: str,
    frames_dir: Path,
    audio_src: Path,
    output: Path,
    fps: float,
    crf: int,
    vcodec: str,
    audio_codec: str,
) -> list[str]:
    fps_s = _format_fps(fps)
    args = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        fps_s,
        "-start_number",
        "0",
        "-i",
        str(frames_dir / _FRAME_PATTERN),
        "-i",
        str(audio_src),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        vcodec,
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        "yuv420p",
        "-r",
        fps_s,
        "-c:a",
        audio_codec,
    ]
    if output.suffix.lower() in {".mp4", ".mov"}:
        args.extend(["-movflags", "+faststart"])
    args.append(str(output))
    return args


def _format_fps(fps: float) -> str:
    if abs(fps - round(fps)) < 1e-6:
        return str(int(round(fps)))
    return f"{fps:.6f}".rstrip("0").rstrip(".")


def _run_ffmpeg(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def _ffmpeg_error(completed: subprocess.CompletedProcess[str]) -> str:
    err = (completed.stderr or completed.stdout or "").strip()
    if err:
        return f"ffmpeg encode failed: {err[-800:]}"
    return f"ffmpeg encode failed with exit code {completed.returncode}"
