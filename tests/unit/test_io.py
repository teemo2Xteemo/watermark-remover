from __future__ import annotations

import struct
import zlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from watermark_remover.config import Settings
from watermark_remover.exceptions import InputValidationError, ResourceLimitError
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
from watermark_remover.io.video import VideoMetadata


def _write_png(path: Path, image: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", image, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    assert ok
    path.write_bytes(buf.tobytes())


def _header_only_png(path: Path, width: int, height: int) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IEND", b""))


def test_read_image_returns_bgr_uint8(tmp_path: Path) -> None:
    src = tmp_path / "tiny.png"
    _write_png(src, np.full((12, 16, 3), (10, 20, 30), dtype=np.uint8))
    image = read_image(src)
    assert image.dtype == np.uint8
    assert image.shape == (12, 16, 3)
    assert tuple(image[0, 0]) == (10, 20, 30)


def test_write_image_atomic_replaces_and_removes_tmp(tmp_path: Path) -> None:
    dest = tmp_path / "out.png"
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    write_image_atomic(dest, image)
    assert dest.is_file()
    assert not dest.with_name(dest.name + ".tmp").exists()
    loaded = read_image(dest)
    assert loaded.shape == (8, 8, 3)


def test_write_bytes_atomic_is_replaced(tmp_path: Path) -> None:
    dest = tmp_path / "note.txt"
    write_bytes_atomic(dest, b"hello")
    assert dest.read_bytes() == b"hello"
    assert not dest.with_name(dest.name + ".tmp").exists()


def test_validate_input_path_missing(tmp_path: Path) -> None:
    with pytest.raises(InputValidationError, match="does not exist"):
        validate_input_path(tmp_path / "missing.png")


def test_validate_input_path_unsupported_suffix(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("nope", encoding="utf-8")
    with pytest.raises(InputValidationError, match="unsupported"):
        validate_input_path(path)


def test_validate_size_limits_rejects_before_decode(tmp_path: Path) -> None:
    src = tmp_path / "big.png"
    _write_png(src, np.zeros((20, 20, 3), dtype=np.uint8))
    with pytest.raises(InputValidationError, match="max_input_bytes"):
        validate_size_limits(src, max_input_bytes=10)


def test_refuse_overwrite_unless_flag(tmp_path: Path) -> None:
    src = tmp_path / "same.png"
    _write_png(src, np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises(InputValidationError, match="overwrite"):
        refuse_overwrite_unless_flag(src, src, overwrite=False)
    refuse_overwrite_unless_flag(src, src, overwrite=True)


def test_default_output_path_uses_stem_inpainted() -> None:
    path = Path("clip.png")
    assert default_output_path(path) == Path("clip_inpainted.png")


def test_default_output_path_preserves_video_suffix() -> None:
    assert default_output_path(Path("clip.mp4")) == Path("clip_inpainted.mp4")


def test_validate_input_path_accepts_video_suffix(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"not-decoded-here")
    assert validate_input_path(path) == path
    assert is_video_path(path)


def test_probe_png_dimensions_from_header(tmp_path: Path) -> None:
    src = tmp_path / "huge.png"
    _header_only_png(src, 12000, 8000)
    assert probe_image_dimensions(src) == (12000, 8000)


def test_huge_resolution_rejected_when_ram_capped(tmp_path: Path) -> None:
    src = tmp_path / "huge.png"
    _header_only_png(src, 20000, 20000)
    settings = Settings(max_ram_mb=1)
    with pytest.raises(ResourceLimitError):
        validate_resolution_limits(src, settings)


def test_huge_resolution_allowed_when_unbounded(tmp_path: Path) -> None:
    src = tmp_path / "huge.png"
    _header_only_png(src, 20000, 20000)
    validate_resolution_limits(src, Settings(max_ram_mb=None))


def test_video_working_set_rejected_when_ram_capped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"placeholder")
    monkeypatch.setattr(
        "watermark_remover.io.video.probe_video",
        lambda _path: VideoMetadata(
            fps=10.0,
            width=20000,
            height=20000,
            duration=1.0,
            codec="h264",
            frame_count=2,
            has_audio=False,
        ),
    )
    with pytest.raises(ResourceLimitError):
        validate_resolution_limits(src, Settings(max_ram_mb=1, max_workers=8))


def test_tiny_resolution_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "one.png"
    _write_png(src, np.array([[[1, 2, 3]]], dtype=np.uint8))
    image = read_image(src)
    assert image.shape == (1, 1, 3)
    dest = tmp_path / "one_out.png"
    write_image_atomic(dest, image)
    assert read_image(dest).shape == (1, 1, 3)


def test_validate_input_path_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(InputValidationError, match="not a file"):
        validate_input_path(tmp_path)


def test_read_image_jpeg_and_webp(tmp_path: Path) -> None:
    image = np.full((12, 16, 3), (10, 20, 30), dtype=np.uint8)
    jpg = tmp_path / "tiny.jpg"
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert ok
    jpg.write_bytes(buf.tobytes())
    loaded = read_image(jpg)
    assert loaded.shape == (12, 16, 3)
    assert loaded.dtype == np.uint8
    assert probe_image_dimensions(jpg) == (16, 12)

    webp = tmp_path / "tiny.webp"
    ok, buf = cv2.imencode(".webp", image)
    assert ok
    webp.write_bytes(buf.tobytes())
    loaded_webp = read_image(webp)
    assert loaded_webp.shape == (12, 16, 3)
    assert probe_image_dimensions(webp) == (16, 12)


def test_read_image_grayscale_png_becomes_bgr(tmp_path: Path) -> None:
    src = tmp_path / "gray.png"
    gray = np.full((8, 10), 40, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", gray, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    assert ok
    src.write_bytes(buf.tobytes())
    image = read_image(src)
    assert image.shape == (8, 10, 3)
    assert image.dtype == np.uint8
    assert tuple(image[0, 0]) == (40, 40, 40)


def test_read_image_rejects_undecodable_png(tmp_path: Path) -> None:
    src = tmp_path / "bad.png"
    src.write_bytes(b"\x89PNG\r\n\x1a\n" + b"not-a-png-payload")
    with pytest.raises(InputValidationError, match="failed to decode"):
        read_image(src)


def test_write_image_atomic_rejects_unsupported_suffix(tmp_path: Path) -> None:
    dest = tmp_path / "out.bmp"
    with pytest.raises(InputValidationError, match="unsupported output format"):
        write_image_atomic(dest, np.zeros((4, 4, 3), dtype=np.uint8))


def test_write_image_atomic_jpeg(tmp_path: Path) -> None:
    dest = tmp_path / "out.jpg"
    write_image_atomic(dest, np.full((8, 8, 3), 80, dtype=np.uint8))
    assert dest.is_file()
    loaded = read_image(dest)
    assert loaded.shape == (8, 8, 3)


def test_write_bytes_atomic_cleans_tmp_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "note.txt"

    def boom(_fd: int) -> None:
        raise OSError("fsync failed")

    monkeypatch.setattr("watermark_remover.io.image.os.fsync", boom)
    with pytest.raises(OSError, match="fsync failed"):
        write_bytes_atomic(dest, b"hello")
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".tmp").exists()


def test_refuse_overwrite_falls_back_when_resolve_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "same.png"
    _write_png(src, np.zeros((4, 4, 3), dtype=np.uint8))
    original_resolve = Path.resolve

    def failing_resolve(self: Path) -> Path:
        if self.name == "same.png":
            raise OSError("broken resolve")
        return original_resolve(self)

    monkeypatch.setattr(Path, "resolve", failing_resolve)
    with pytest.raises(InputValidationError, match="overwrite"):
        refuse_overwrite_unless_flag(src, src, overwrite=False)


def test_sniff_image_format_jpeg_webp_and_unknown(tmp_path: Path) -> None:
    from watermark_remover.io.validate import sniff_image_format

    jpeg = tmp_path / "x.jpg"
    jpeg.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 8)
    assert sniff_image_format(jpeg) == "jpeg"
    webp = tmp_path / "x.webp"
    webp.write_bytes(b"RIFF" + b"\x10\x00\x00\x00" + b"WEBPVP8X" + b"\x00" * 8)
    assert sniff_image_format(webp) == "webp"
    bogus = tmp_path / "x.png"
    bogus.write_bytes(b"not-an-image")
    with pytest.raises(InputValidationError, match="unrecognized image format"):
        sniff_image_format(bogus)


def test_probe_png_truncated_and_invalid_dimensions(tmp_path: Path) -> None:
    tiny = tmp_path / "short.png"
    tiny.write_bytes(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(InputValidationError, match="truncated PNG"):
        probe_image_dimensions(tiny)
    zero = tmp_path / "zero.png"
    _header_only_png(zero, 0, 10)
    with pytest.raises(InputValidationError, match="invalid PNG dimensions"):
        probe_image_dimensions(zero)


def test_probe_jpeg_dimensions_from_sof0(tmp_path: Path) -> None:
    src = tmp_path / "sof.jpg"
    # SOI + SOF0 (baseline) with height=20, width=30, 1 component.
    sof_body = struct.pack(">BHHB", 8, 20, 30, 1) + bytes([1, 0x11, 0])
    payload = (
        b"\xff\xd8"
        + b"\xff\xc0"
        + struct.pack(">H", len(sof_body) + 2)
        + sof_body
        + b"\xff\xd9"
    )
    src.write_bytes(payload)
    assert probe_image_dimensions(src) == (30, 20)


def test_probe_jpeg_skips_restart_and_app_markers(tmp_path: Path) -> None:
    src = tmp_path / "app.jpg"
    app = b"JFIF\x00"
    sof_body = struct.pack(">BHHB", 8, 12, 16, 1) + bytes([1, 0x11, 0])
    payload = (
        b"\xff\xd8"
        + b"\xff\xe0"
        + struct.pack(">H", len(app) + 2)
        + app
        + b"\xff\xd0"
        + b"\xff\xc2"
        + struct.pack(">H", len(sof_body) + 2)
        + sof_body
        + b"\xff\xd9"
    )
    src.write_bytes(payload)
    assert probe_image_dimensions(src) == (16, 12)


def test_probe_jpeg_missing_sof_raises(tmp_path: Path) -> None:
    src = tmp_path / "nosof.jpg"
    src.write_bytes(b"\xff\xd8\xff\xd9")
    with pytest.raises(InputValidationError, match="could not read JPEG"):
        probe_image_dimensions(src)


def test_probe_jpeg_invalid_dimensions(tmp_path: Path) -> None:
    src = tmp_path / "zero.jpg"
    sof_body = struct.pack(">BHHB", 8, 0, 30, 1) + bytes([1, 0x11, 0])
    payload = (
        b"\xff\xd8"
        + b"\xff\xc0"
        + struct.pack(">H", len(sof_body) + 2)
        + sof_body
        + b"\xff\xd9"
    )
    src.write_bytes(payload)
    with pytest.raises(InputValidationError, match="invalid JPEG dimensions"):
        probe_image_dimensions(src)


def test_probe_webp_vp8x_vp8_vp8l(tmp_path: Path) -> None:
    vp8x = tmp_path / "vp8x.webp"
    chunk = bytearray(30)
    chunk[0:4] = b"RIFF"
    chunk[8:12] = b"WEBP"
    chunk[12:16] = b"VP8X"
    chunk[24:27] = (63).to_bytes(3, "little")  # width - 1
    chunk[27:30] = (47).to_bytes(3, "little")  # height - 1
    vp8x.write_bytes(bytes(chunk))
    assert probe_image_dimensions(vp8x) == (64, 48)

    vp8 = tmp_path / "vp8.webp"
    chunk = bytearray(30)
    chunk[0:4] = b"RIFF"
    chunk[8:12] = b"WEBP"
    chunk[12:16] = b"VP8 "
    chunk[26:28] = (40).to_bytes(2, "little")
    chunk[28:30] = (24).to_bytes(2, "little")
    vp8.write_bytes(bytes(chunk))
    assert probe_image_dimensions(vp8) == (40, 24)

    vp8l = tmp_path / "vp8l.webp"
    chunk = bytearray(30)
    chunk[0:4] = b"RIFF"
    chunk[8:12] = b"WEBP"
    chunk[12:16] = b"VP8L"
    width, height = 32, 16
    bits = (width - 1) | ((height - 1) << 14)
    chunk[21:25] = bits.to_bytes(4, "little")
    vp8l.write_bytes(bytes(chunk))
    assert probe_image_dimensions(vp8l) == (32, 16)


def test_probe_webp_truncated_and_unsupported(tmp_path: Path) -> None:
    short = tmp_path / "short.webp"
    short.write_bytes(b"RIFF\x00\x00\x00\x00WEBP")
    with pytest.raises(InputValidationError, match="truncated WEBP"):
        probe_image_dimensions(short)
    unknown = tmp_path / "unk.webp"
    chunk = bytearray(30)
    chunk[0:4] = b"RIFF"
    chunk[8:12] = b"WEBP"
    chunk[12:16] = b"UNKN"
    unknown.write_bytes(bytes(chunk))
    with pytest.raises(InputValidationError, match="unsupported WEBP"):
        probe_image_dimensions(unknown)
