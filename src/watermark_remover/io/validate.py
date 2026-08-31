from __future__ import annotations

from pathlib import Path

from watermark_remover.config import Settings
from watermark_remover.exceptions import InputValidationError, ResourceLimitError

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".webm"})
MEDIA_SUFFIXES = IMAGE_SUFFIXES | VIDEO_SUFFIXES
_HEADER_BYTES = 65536
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"
_BYTES_PER_PIXEL_ESTIMATE = 8


def is_video_path(path: Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_SUFFIXES


def validate_input_path(path: Path) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise InputValidationError(f"input does not exist: {resolved.name}")
    if not resolved.is_file():
        raise InputValidationError(f"input is not a file: {resolved.name}")
    suffix = resolved.suffix.lower()
    if suffix not in MEDIA_SUFFIXES:
        raise InputValidationError(
            f"unsupported format '{suffix}'; expected JPG, PNG, WEBP, MP4, MOV, or WEBM"
        )
    return resolved


def validate_size_limits(path: Path, max_input_bytes: int) -> None:
    size = Path(path).stat().st_size
    if size > max_input_bytes:
        raise InputValidationError(f"input exceeds max_input_bytes ({size} > {max_input_bytes})")


def default_output_path(input_path: Path) -> Path:
    src = Path(input_path)
    return src.with_name(f"{src.stem}_inpainted{src.suffix}")


def refuse_overwrite_unless_flag(input_path: Path, output_path: Path, overwrite: bool) -> None:
    try:
        same = Path(input_path).resolve() == Path(output_path).resolve()
    except OSError:
        same = os_path_equal(input_path, output_path)
    if same and not overwrite:
        raise InputValidationError(
            "output path equals input path; pass --overwrite to replace the input"
        )


def os_path_equal(left: Path, right: Path) -> bool:
    return os_norm(left) == os_norm(right)


def os_norm(path: Path) -> str:
    return str(Path(path)).replace("\\", "/").lower()


def sniff_image_format(path: Path) -> str:
    with Path(path).open("rb") as handle:
        head = handle.read(16)
    if head.startswith(_PNG_MAGIC):
        return "png"
    if head.startswith(_JPEG_MAGIC):
        return "jpeg"
    if len(head) >= 12 and head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return "webp"
    raise InputValidationError(f"unrecognized image format: {Path(path).name}")


def probe_image_dimensions(path: Path) -> tuple[int, int]:
    """Return (width, height) from the container header — no full decode."""
    with Path(path).open("rb") as handle:
        data = handle.read(_HEADER_BYTES)
    kind = sniff_image_format(path)
    if kind == "png":
        return _png_dimensions(data)
    if kind == "jpeg":
        return _jpeg_dimensions(data)
    return _webp_dimensions(data)


def estimate_working_set_mb(width: int, height: int, copies: int = 1) -> float:
    """Rough RAM estimate for `copies` uncompressed HxW working buffers."""
    return (width * height * _BYTES_PER_PIXEL_ESTIMATE * max(1, copies)) / (1024 * 1024)


def validate_resolution_limits(path: Path, settings: Settings) -> None:
    if settings.max_ram_mb is None:
        return
    src = Path(path)
    if is_video_path(src):
        from watermark_remover.io.video import probe_video

        meta = probe_video(src)
        width, height = meta.width, meta.height
        copies = int(settings.max_workers)
    else:
        width, height = probe_image_dimensions(src)
        copies = 1
    estimate_mb = estimate_working_set_mb(width, height, copies)
    if estimate_mb > settings.max_ram_mb:
        raise ResourceLimitError(
            f"estimated working set {estimate_mb:.1f} MiB exceeds max_ram_mb={settings.max_ram_mb}"
        )


def _png_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or not data.startswith(_PNG_MAGIC):
        raise InputValidationError("truncated PNG header")
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    if width < 1 or height < 1:
        raise InputValidationError("invalid PNG dimensions")
    return width, height


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    index = 2
    length = len(data)
    while index < length - 8:
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker in {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }:
            height = int.from_bytes(data[index + 5 : index + 7], "big")
            width = int.from_bytes(data[index + 7 : index + 9], "big")
            if width < 1 or height < 1:
                raise InputValidationError("invalid JPEG dimensions")
            return width, height
        if marker in {0xD8, 0x01} or 0xD0 <= marker <= 0xD9:
            index += 2
            continue
        if index + 4 > length:
            break
        block = int.from_bytes(data[index + 2 : index + 4], "big")
        index += 2 + block
    raise InputValidationError("could not read JPEG dimensions")


def _webp_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 30 or data[0:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise InputValidationError("truncated WEBP header")
    kind = data[12:16]
    if kind == b"VP8X":
        width = int.from_bytes(data[24:27], "little") + 1
        height = int.from_bytes(data[27:30], "little") + 1
        return width, height
    if kind == b"VP8 " and len(data) >= 30:
        width = int.from_bytes(data[26:28], "little") & 0x3FFF
        height = int.from_bytes(data[28:30], "little") & 0x3FFF
        return width, height
    if kind == b"VP8L" and len(data) >= 25:
        bits = int.from_bytes(data[21:25], "little")
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return width, height
    raise InputValidationError("unsupported WEBP bitstream")
