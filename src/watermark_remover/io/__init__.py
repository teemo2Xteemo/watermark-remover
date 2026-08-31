from __future__ import annotations

from watermark_remover.io.image import read_image, write_bytes_atomic, write_image_atomic
from watermark_remover.io.validate import (
    default_output_path,
    is_video_path,
    probe_image_dimensions,
    refuse_overwrite_unless_flag,
    validate_input_path,
    validate_resolution_limits,
    validate_size_limits,
)
from watermark_remover.io.video import VideoMetadata, open_capture, probe_video

__all__ = [
    "VideoMetadata",
    "default_output_path",
    "is_video_path",
    "open_capture",
    "probe_image_dimensions",
    "probe_video",
    "read_image",
    "refuse_overwrite_unless_flag",
    "validate_input_path",
    "validate_resolution_limits",
    "validate_size_limits",
    "write_bytes_atomic",
    "write_image_atomic",
]
