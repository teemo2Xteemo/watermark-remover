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
    probe_image_dimensions,
    refuse_overwrite_unless_flag,
    validate_input_path,
    validate_resolution_limits,
    validate_size_limits,
)


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


def test_tiny_resolution_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "one.png"
    _write_png(src, np.array([[[1, 2, 3]]], dtype=np.uint8))
    image = read_image(src)
    assert image.shape == (1, 1, 3)
    dest = tmp_path / "one_out.png"
    write_image_atomic(dest, image)
    assert read_image(dest).shape == (1, 1, 3)
