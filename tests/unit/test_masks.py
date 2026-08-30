from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from watermark_remover.exceptions import MaskError
from watermark_remover.masks.base import validate_mask_coverage
from watermark_remover.masks.manual import ManualMaskProvider
from watermark_remover.masks.serialize import (
    export_mask_json,
    export_mask_png,
    load_mask_json,
    load_mask_png,
)


def test_manual_mask_provider_contract() -> None:
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    mask = np.zeros((32, 48), dtype=np.uint8)
    mask[5:10, 5:10] = 255
    out = ManualMaskProvider(mask).get_mask(frame, 0)
    assert out.dtype == np.uint8
    assert set(np.unique(out).tolist()).issubset({0, 255})
    assert out.shape == frame.shape[:2]
    assert int(out[7, 7]) == 255
    assert int(out[0, 0]) == 0


def test_manual_mask_provider_shape_mismatch() -> None:
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    with pytest.raises(MaskError, match="does not match"):
        ManualMaskProvider(mask).get_mask(frame, 0)


def test_validate_mask_coverage_empty_and_full() -> None:
    empty = np.zeros((8, 8), dtype=np.uint8)
    full = np.full((8, 8), 255, dtype=np.uint8)
    with pytest.raises(MaskError, match="empty mask"):
        validate_mask_coverage(empty)
    with pytest.raises(MaskError, match="full-image mask"):
        validate_mask_coverage(full)
    validate_mask_coverage(empty, allow_empty_mask=True)
    validate_mask_coverage(full, allow_full_mask=True)


def test_load_and_export_mask_png_roundtrip(tmp_path: Path) -> None:
    mask = np.zeros((20, 24), dtype=np.uint8)
    mask[2:8, 4:10] = 255
    dest = tmp_path / "region.mask.png"
    export_mask_png(dest, mask)
    loaded = load_mask_png(dest)
    assert loaded.shape == mask.shape
    assert loaded.dtype == np.uint8
    assert np.array_equal(loaded, mask)


def test_load_mask_json_rasterizes_and_resizes(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "kind": "polygon",
        "size": [8, 6],
        "points": [[1, 1], [4, 1], [4, 4], [1, 4]],
        "closed": True,
    }
    src = tmp_path / "box.mask.json"
    src.write_text(json.dumps(payload), encoding="utf-8")
    mask = load_mask_json(src, (12, 16))
    assert mask.shape == (12, 16)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()).issubset({0, 255})
    assert int(mask.max()) == 255


def test_load_mask_json_bbox_matches_width_height(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "kind": "bbox",
        "size": [16, 12],
        "bbox": [3, 2, 5, 4],
    }
    src = tmp_path / "box.mask.json"
    src.write_text(json.dumps(payload), encoding="utf-8")
    mask = load_mask_json(src, (12, 16))
    expected = np.zeros((12, 16), dtype=np.uint8)
    expected[2:6, 3:8] = 255
    assert np.array_equal(mask, expected)
    assert int(mask[1, 3]) == 0
    assert int(mask[2, 8]) == 0
    assert int(mask[6, 3]) == 0
    assert int(np.count_nonzero(mask)) == 20


def test_load_mask_json_rejects_unknown_major(tmp_path: Path) -> None:
    src = tmp_path / "future.mask.json"
    src.write_text(json.dumps({"schema_version": 99, "kind": "polygon", "size": [2, 2]}))
    with pytest.raises(MaskError, match="schema_version"):
        load_mask_json(src, (2, 2))


def test_export_mask_json_writes_schema_version(tmp_path: Path) -> None:
    dest = tmp_path / "out.mask.json"
    export_mask_json(dest, {"kind": "polygon", "size": [4, 4], "points": [], "closed": True})
    body = json.loads(dest.read_text(encoding="utf-8"))
    assert body["schema_version"] == 1


def test_fixture_masks_are_binary(fixtures_dir: Path) -> None:
    for name in ("still_logo.mask.png", "still_empty.mask.png", "still_full.mask.png"):
        mask = load_mask_png(fixtures_dir / name)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask).tolist()).issubset({0, 255})
