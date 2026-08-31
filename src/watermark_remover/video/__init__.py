from __future__ import annotations

from watermark_remover.video.encode import encode_video
from watermark_remover.video.extract import extract_frames
from watermark_remover.video.processor import VideoProcessor, capped_max_workers

__all__ = [
    "VideoProcessor",
    "capped_max_workers",
    "encode_video",
    "extract_frames",
]
