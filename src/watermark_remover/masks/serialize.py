from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from watermark_remover.exceptions import MaskError
from watermark_remover.io.image import write_bytes_atomic
from watermark_remover.masks.base import validate_mask_array

_SUPPORTED_SCHEMA = 1
_PNG_PARAMS = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]


def load_mask_png(path: Path) -> np.ndarray:
    src = Path(path)
    if not src.is_file():
        raise MaskError(f"mask file does not exist: {src.name}")
    if src.suffix.lower() != ".png":
        raise MaskError(f"PNG mask expected, got '{src.suffix}'")
    payload = np.fromfile(str(src), dtype=np.uint8)
    decoded = cv2.imdecode(payload, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise MaskError(f"failed to decode mask PNG: {src.name}")
    return validate_mask_array(decoded)


def load_mask_json(path: Path, frame_hw: tuple[int, int]) -> np.ndarray:
    src = Path(path)
    if not src.is_file():
        raise MaskError(f"mask file does not exist: {src.name}")
    try:
        payload = json.loads(src.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MaskError(f"invalid mask JSON: {src.name}") from exc
    if not isinstance(payload, dict):
        raise MaskError("mask JSON must be an object")
    version = payload.get("schema_version")
    if not isinstance(version, int) or version != _SUPPORTED_SCHEMA:
        raise MaskError(f"unsupported mask schema_version: {version!r}")
    size = payload.get("size")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or not all(isinstance(v, int) and v > 0 for v in size)
    ):
        raise MaskError("mask JSON size must be [width, height]")
    width, height = int(size[0]), int(size[1])
    raster = _rasterize_schema(payload, width, height)
    frame_h, frame_w = int(frame_hw[0]), int(frame_hw[1])
    if raster.shape != (frame_h, frame_w):
        raster = cv2.resize(raster, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
    return validate_mask_array(raster)


def export_mask_png(path: Path, mask: np.ndarray) -> None:
    binary = validate_mask_array(mask)
    ok, buffer = cv2.imencode(".png", binary, _PNG_PARAMS)
    if not ok:
        raise MaskError(f"failed to encode mask PNG: {Path(path).name}")
    write_bytes_atomic(Path(path), buffer.tobytes())


def export_mask_json(path: Path, payload: dict[str, Any]) -> None:
    body = dict(payload)
    body.setdefault("schema_version", _SUPPORTED_SCHEMA)
    data = json.dumps(body, indent=2, sort_keys=True).encode("utf-8")
    write_bytes_atomic(Path(path), data)


def mask_to_polygon_payload(mask: np.ndarray) -> dict[str, Any]:
    binary = validate_mask_array(mask)
    height, width = binary.shape
    ys, xs = np.nonzero(binary)
    if xs.size == 0:
        points: list[list[int]] = []
    else:
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return {
        "schema_version": _SUPPORTED_SCHEMA,
        "kind": "polygon",
        "size": [width, height],
        "points": points,
        "closed": True,
    }


def _rasterize_schema(payload: dict[str, Any], width: int, height: int) -> np.ndarray:
    kind = payload.get("kind")
    canvas = np.zeros((height, width), dtype=np.uint8)
    if kind == "polygon":
        points = _as_points(payload.get("points"))
        if points.size == 0:
            return canvas
        cv2.fillPoly(canvas, [points], 255)
        return canvas
    if kind == "bbox":
        bbox = payload.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, int) for v in bbox):
            x, y, box_w, box_h = (int(v) for v in bbox)
            cv2.rectangle(canvas, (x, y), (x + box_w, y + box_h), 255, thickness=-1)
            return canvas
        points = _as_points(payload.get("points"))
        if points.shape[0] >= 2:
            x0, y0 = int(points[0, 0]), int(points[0, 1])
            x1, y1 = int(points[1, 0]), int(points[1, 1])
            cv2.rectangle(canvas, (x0, y0), (x1, y1), 255, thickness=-1)
            return canvas
        raise MaskError("bbox mask JSON needs bbox [x, y, w, h] or two points")
    raise MaskError(f"unsupported mask kind: {kind!r}")


def _as_points(raw: Any) -> np.ndarray:
    if raw is None:
        return np.zeros((0, 2), dtype=np.int32)
    if not isinstance(raw, list):
        raise MaskError("mask JSON points must be a list")
    points: list[list[int]] = []
    for item in raw:
        if not isinstance(item, list) or len(item) != 2:
            raise MaskError("each point must be [x, y]")
        x, y = item
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise MaskError("point coordinates must be numeric")
        points.append([int(x), int(y)])
    if not points:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray(points, dtype=np.int32)
