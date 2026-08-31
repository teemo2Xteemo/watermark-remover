from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2

from watermark_remover.exceptions import InputValidationError
from watermark_remover.io.validate import is_video_path


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    width: int
    height: int
    duration: float
    codec: str
    frame_count: int
    has_audio: bool


def open_capture(path: Path) -> cv2.VideoCapture:
    """Open a streaming VideoCapture. Caller must release. Does not decode all frames."""
    src = Path(path)
    if not src.is_file():
        raise InputValidationError(f"input does not exist: {src.name}")
    if not is_video_path(src):
        raise InputValidationError(
            f"unsupported video format '{src.suffix}'; expected MP4, MOV, or WEBM"
        )
    capture = cv2.VideoCapture(str(src))
    if not capture.isOpened():
        raise InputValidationError(f"cannot open video: {src.name}")
    return capture


def probe_video(path: Path) -> VideoMetadata:
    """Return fps/resolution/duration/codec from headers — no full-video decode."""
    src = Path(path)
    capture = open_capture(src)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC) or 0)
        codec = _fourcc_to_str(fourcc)
    finally:
        capture.release()

    ffprobe = _ffprobe_metadata(src)
    has_audio = False
    if ffprobe is not None:
        fps = ffprobe.fps or fps
        width = ffprobe.width or width
        height = ffprobe.height or height
        frame_count = ffprobe.frame_count or frame_count
        if ffprobe.codec:
            codec = ffprobe.codec
        if ffprobe.duration > 0:
            duration = ffprobe.duration
        else:
            duration = (frame_count / fps) if fps > 0 else 0.0
        has_audio = ffprobe.has_audio
    else:
        duration = (frame_count / fps) if fps > 0 else 0.0

    if width < 1 or height < 1:
        raise InputValidationError(f"invalid video dimensions: {src.name}")
    if fps <= 0:
        raise InputValidationError(f"invalid video fps: {src.name}")
    return VideoMetadata(
        fps=float(fps),
        width=int(width),
        height=int(height),
        duration=float(duration),
        codec=codec or "unknown",
        frame_count=int(frame_count),
        has_audio=bool(has_audio),
    )


@dataclass(frozen=True)
class _FfprobeInfo:
    fps: float
    width: int
    height: int
    duration: float
    codec: str
    frame_count: int
    has_audio: bool


def _fourcc_to_str(fourcc: int) -> str:
    if fourcc <= 0:
        return "unknown"
    chars = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))
    cleaned = "".join(ch if ch.isprintable() else "" for ch in chars).strip()
    return cleaned.lower() or "unknown"


def _parse_frame_rate(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    if "/" in value:
        num_s, den_s = value.split("/", 1)
        try:
            num = float(num_s)
            den = float(den_s)
        except ValueError:
            return 0.0
        if den == 0:
            return 0.0
        return num / den
    try:
        return float(value)
    except ValueError:
        return 0.0


def _ffprobe_metadata(path: Path) -> _FfprobeInfo | None:
    exe = _find_ffprobe()
    if exe is None:
        return None
    completed = subprocess.run(
        [
            exe,
            "-v",
            "error",
            "-show_entries",
            "stream=index,codec_type,codec_name,width,height,r_frame_rate,nb_frames,duration",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None
    streams = payload.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = payload.get("format") or {}
    fps = 0.0
    width = 0
    height = 0
    codec = ""
    frame_count = 0
    duration = 0.0
    if video:
        fps = _parse_frame_rate(str(video.get("r_frame_rate") or ""))
        width = int(video.get("width") or 0)
        height = int(video.get("height") or 0)
        codec = str(video.get("codec_name") or "")
        try:
            frame_count = int(video.get("nb_frames") or 0)
        except (TypeError, ValueError):
            frame_count = 0
        try:
            duration = float(video.get("duration") or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
    if duration <= 0:
        try:
            duration = float(fmt.get("duration") or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
    return _FfprobeInfo(
        fps=fps,
        width=width,
        height=height,
        duration=duration,
        codec=codec,
        frame_count=frame_count,
        has_audio=audio is not None,
    )


def _find_ffprobe() -> str | None:
    on_path = shutil.which("ffprobe")
    if on_path:
        return on_path
    ffmpeg = shutil.which("ffmpeg")
    sibling = _ffprobe_sibling(ffmpeg)
    if sibling:
        return sibling
    try:
        from watermark_remover.video.encode import find_ffmpeg
    except ImportError:
        return None
    return _ffprobe_sibling(find_ffmpeg())


def _ffprobe_sibling(ffmpeg: str | None) -> str | None:
    if not ffmpeg:
        return None
    suffix = Path(ffmpeg).suffix
    sibling = Path(ffmpeg).with_name(f"ffprobe{suffix}")
    if sibling.is_file():
        return str(sibling)
    return None
