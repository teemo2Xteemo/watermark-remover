from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from watermark_remover.exceptions import MaskError
from watermark_remover.masks.base import validate_mask_coverage
from watermark_remover.masks.manual import KeyframeMaskProvider, ManualMaskProvider
from watermark_remover.masks.serialize import (
    export_keyframes,
    export_mask_json,
    export_mask_png,
    load_keyframes,
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


def test_keyframe_mask_provider_hold_last() -> None:
    frame = np.zeros((8, 10, 3), dtype=np.uint8)
    first = np.zeros((8, 10), dtype=np.uint8)
    first[1:3, 1:3] = 255
    second = np.zeros((8, 10), dtype=np.uint8)
    second[4:6, 6:9] = 255
    provider = KeyframeMaskProvider([(0.0, first), (1.0, second)], fps=10.0)
    at_start = provider.get_mask(frame, 0)
    assert np.array_equal(at_start, first)
    before_switch = provider.get_mask(frame, 9)
    assert np.array_equal(before_switch, first)
    at_switch = provider.get_mask(frame, 10)
    assert np.array_equal(at_switch, second)
    after = provider.get_mask(frame, 15)
    assert np.array_equal(after, second)


def test_keyframe_mask_provider_requires_keyframes() -> None:
    with pytest.raises(MaskError, match="at least one keyframe"):
        KeyframeMaskProvider([], fps=10.0)


def test_keyframes_json_roundtrip(tmp_path: Path) -> None:
    first = np.zeros((6, 8), dtype=np.uint8)
    first[1:3, 1:4] = 255
    second = np.zeros((6, 8), dtype=np.uint8)
    second[2:5, 4:7] = 255
    dest = tmp_path / "clip.keyframes.json"
    export_keyframes(dest, [(0.0, first), (1.25, second)], "clip")
    body = json.loads(dest.read_text(encoding="utf-8"))
    assert body["schema_version"] == 1
    assert body["keyframes"][0]["t"] == 0.0
    assert body["keyframes"][0]["mask_ref"] == "clip.kf.0.png"
    assert body["keyframes"][1]["mask_ref"] == "clip.kf.1.png"
    loaded = load_keyframes(dest, (6, 8))
    assert len(loaded) == 2
    assert loaded[0][0] == 0.0
    assert np.array_equal(loaded[0][1], first)
    assert loaded[1][0] == 1.25
    assert np.array_equal(loaded[1][1], second)


def test_load_keyframes_rejects_unknown_schema(tmp_path: Path) -> None:
    src = tmp_path / "clip.keyframes.json"
    src.write_text(json.dumps({"schema_version": 99, "keyframes": [{"t": 0, "mask_ref": "x.png"}]}))
    with pytest.raises(MaskError, match="schema_version"):
        load_keyframes(src, (2, 2))


def test_fixture_keyframes_json(fixtures_dir: Path) -> None:
    loaded = load_keyframes(fixtures_dir / "clip_5s.keyframes.json", (48, 64))
    assert len(loaded) == 2
    assert loaded[0][0] == 0.0
    assert loaded[1][0] == 1.0
    assert loaded[0][1].shape == (48, 64)
    assert not np.array_equal(loaded[0][1], loaded[1][1])


def test_binarize_channel_masks_and_invalid_ndim() -> None:
    from watermark_remover.masks.base import binarize_mask, validate_mask_array

    single = np.zeros((4, 5, 1), dtype=np.uint8)
    single[1, 1, 0] = 200
    assert set(np.unique(binarize_mask(single)).tolist()) == {0, 255}
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    rgb[2, 2] = (10, 200, 30)
    out = binarize_mask(rgb)
    assert int(out[2, 2]) == 255
    with pytest.raises(MaskError, match="shape"):
        binarize_mask(np.zeros((2,), dtype=np.uint8))
    assert validate_mask_array(single).shape == (4, 5)


def test_validate_mask_coverage_zero_size() -> None:
    with pytest.raises(MaskError, match="zero size"):
        validate_mask_coverage(np.zeros((0, 4), dtype=np.uint8))


def test_manual_and_keyframe_reject_1d_frame() -> None:
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1, 1] = 255
    with pytest.raises(MaskError, match="at least 2 dimensions"):
        ManualMaskProvider(mask).get_mask(np.zeros((4,), dtype=np.uint8), 0)
    provider = KeyframeMaskProvider([(0.0, mask)], fps=10.0)
    with pytest.raises(MaskError, match="at least 2 dimensions"):
        provider.get_mask(np.zeros((4,), dtype=np.uint8), 0)
    with pytest.raises(MaskError, match="does not match"):
        provider.get_mask(np.zeros((8, 8, 3), dtype=np.uint8), 0)


def test_keyframe_mask_provider_rejects_non_positive_fps() -> None:
    mask = np.zeros((2, 2), dtype=np.uint8)
    mask[0, 0] = 255
    with pytest.raises(MaskError, match="fps must be positive"):
        KeyframeMaskProvider([(0.0, mask)], fps=0.0)


def test_load_mask_png_error_paths(tmp_path: Path) -> None:
    missing = tmp_path / "nope.mask.png"
    with pytest.raises(MaskError, match="does not exist"):
        load_mask_png(missing)
    txt = tmp_path / "mask.txt"
    txt.write_text("x", encoding="utf-8")
    with pytest.raises(MaskError, match="PNG mask expected"):
        load_mask_png(txt)
    bad = tmp_path / "bad.mask.png"
    bad.write_bytes(b"\x89PNG\r\n\x1a\nnot-png")
    with pytest.raises(MaskError, match="failed to decode"):
        load_mask_png(bad)


def test_load_mask_json_error_paths(tmp_path: Path) -> None:
    missing = tmp_path / "missing.mask.json"
    with pytest.raises(MaskError, match="does not exist"):
        load_mask_json(missing, (2, 2))
    bad_json = tmp_path / "bad.mask.json"
    bad_json.write_text("{", encoding="utf-8")
    with pytest.raises(MaskError, match="invalid mask JSON"):
        load_mask_json(bad_json, (2, 2))
    not_obj = tmp_path / "list.mask.json"
    not_obj.write_text("[1]", encoding="utf-8")
    with pytest.raises(MaskError, match="must be an object"):
        load_mask_json(not_obj, (2, 2))
    bad_size = tmp_path / "size.mask.json"
    bad_size.write_text(json.dumps({"schema_version": 1, "kind": "polygon", "size": [0, 2]}))
    with pytest.raises(MaskError, match="size must be"):
        load_mask_json(bad_size, (2, 2))
    unknown_kind = tmp_path / "kind.mask.json"
    unknown_kind.write_text(
        json.dumps({"schema_version": 1, "kind": "circle", "size": [4, 4], "points": []})
    )
    with pytest.raises(MaskError, match="unsupported mask kind"):
        load_mask_json(unknown_kind, (4, 4))


def test_load_mask_json_empty_polygon_and_bbox_points(tmp_path: Path) -> None:
    empty_poly = tmp_path / "empty.mask.json"
    empty_poly.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "polygon",
                "size": [8, 6],
                "points": [],
                "closed": True,
            }
        )
    )
    empty = load_mask_json(empty_poly, (6, 8))
    assert empty.shape == (6, 8)
    assert int(empty.max()) == 0

    two_points = tmp_path / "bbox_pts.mask.json"
    two_points.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "bbox",
                "size": [10, 8],
                "points": [[1, 2], [4, 5]],
            }
        )
    )
    box = load_mask_json(two_points, (8, 10))
    assert int(box[2, 1]) == 255
    assert int(box[5, 4]) == 255
    assert int(box[0, 0]) == 0

    zero_wh = tmp_path / "bbox_zero.mask.json"
    zero_wh.write_text(
        json.dumps({"schema_version": 1, "kind": "bbox", "size": [8, 6], "bbox": [1, 1, 0, 3]})
    )
    assert int(load_mask_json(zero_wh, (6, 8)).max()) == 0

    missing_bbox = tmp_path / "bbox_missing.mask.json"
    missing_bbox.write_text(
        json.dumps({"schema_version": 1, "kind": "bbox", "size": [8, 6], "points": [[1, 1]]})
    )
    with pytest.raises(MaskError, match="bbox"):
        load_mask_json(missing_bbox, (6, 8))


def test_load_mask_json_rejects_bad_points(tmp_path: Path) -> None:
    src = tmp_path / "pts.mask.json"
    src.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "polygon",
                "size": [8, 6],
                "points": "nope",
                "closed": True,
            }
        )
    )
    with pytest.raises(MaskError, match="points must be a list"):
        load_mask_json(src, (6, 8))
    src.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "polygon",
                "size": [8, 6],
                "points": [[1]],
                "closed": True,
            }
        )
    )
    with pytest.raises(MaskError, match="each point must be"):
        load_mask_json(src, (6, 8))
    src.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "polygon",
                "size": [8, 6],
                "points": [["a", 1]],
                "closed": True,
            }
        )
    )
    with pytest.raises(MaskError, match="numeric"):
        load_mask_json(src, (6, 8))


def test_load_keyframes_error_paths(tmp_path: Path) -> None:
    missing = tmp_path / "no.keyframes.json"
    with pytest.raises(MaskError, match="does not exist"):
        load_keyframes(missing, (2, 2))
    bad = tmp_path / "bad.keyframes.json"
    bad.write_text("{", encoding="utf-8")
    with pytest.raises(MaskError, match="invalid keyframes JSON"):
        load_keyframes(bad, (2, 2))
    not_obj = tmp_path / "list.keyframes.json"
    not_obj.write_text("[]", encoding="utf-8")
    with pytest.raises(MaskError, match="must be an object"):
        load_keyframes(not_obj, (2, 2))
    empty = tmp_path / "empty.keyframes.json"
    empty.write_text(json.dumps({"schema_version": 1, "keyframes": []}))
    with pytest.raises(MaskError, match="non-empty list"):
        load_keyframes(empty, (2, 2))
    bad_row = tmp_path / "row.keyframes.json"
    bad_row.write_text(json.dumps({"schema_version": 1, "keyframes": ["x"]}))
    with pytest.raises(MaskError, match="must be an object"):
        load_keyframes(bad_row, (2, 2))
    bad_t = tmp_path / "t.keyframes.json"
    bad_t.write_text(
        json.dumps({"schema_version": 1, "keyframes": [{"t": "0", "mask_ref": "a.png"}]})
    )
    with pytest.raises(MaskError, match="t must be a number"):
        load_keyframes(bad_t, (2, 2))
    bad_ref = tmp_path / "ref.keyframes.json"
    bad_ref.write_text(json.dumps({"schema_version": 1, "keyframes": [{"t": 0, "mask_ref": ""}]}))
    with pytest.raises(MaskError, match="mask_ref must be a filename"):
        load_keyframes(bad_ref, (2, 2))
    escape = tmp_path / "esc.keyframes.json"
    escape.write_text(
        json.dumps({"schema_version": 1, "keyframes": [{"t": 0, "mask_ref": ".."}]})
    )
    with pytest.raises(MaskError, match="sibling PNG"):
        load_keyframes(escape, (2, 2))
    not_png = tmp_path / "np.keyframes.json"
    not_png.write_text(
        json.dumps({"schema_version": 1, "keyframes": [{"t": 0, "mask_ref": "a.jpg"}]})
    )
    with pytest.raises(MaskError, match="sibling PNG"):
        load_keyframes(not_png, (2, 2))


def test_export_keyframes_requires_at_least_one(tmp_path: Path) -> None:
    with pytest.raises(MaskError, match="at least one keyframe"):
        export_keyframes(tmp_path / "clip.keyframes.json", [], "clip")


def test_mask_to_polygon_payload_empty_and_filled() -> None:
    from watermark_remover.masks.serialize import mask_to_polygon_payload

    empty = np.zeros((6, 8), dtype=np.uint8)
    payload = mask_to_polygon_payload(empty)
    assert payload["points"] == []
    filled = np.zeros((6, 8), dtype=np.uint8)
    filled[1:3, 2:5] = 255
    box = mask_to_polygon_payload(filled)
    assert box["schema_version"] == 1
    assert box["kind"] == "polygon"
    assert box["size"] == [8, 6]
    assert box["points"][0] == [2, 1]


def test_load_keyframes_resizes_to_frame(tmp_path: Path) -> None:
    mask = np.zeros((6, 8), dtype=np.uint8)
    mask[1:3, 1:4] = 255
    dest = tmp_path / "clip.keyframes.json"
    export_keyframes(dest, [(0.0, mask)], "clip")
    loaded = load_keyframes(dest, (12, 16))
    assert loaded[0][1].shape == (12, 16)
    assert set(np.unique(loaded[0][1]).tolist()).issubset({0, 255})
