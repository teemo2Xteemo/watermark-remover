from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from watermark_remover.exceptions import MaskError
from watermark_remover.masks.base import MaskCandidate, MaskProvider, validate_mask_array

_MAX_CANDIDATES_PER_METHOD = 5
_ALPHA_MIN = 8
_DETECT_LONG_EDGE = 1280


class AutoDetectMaskProvider(MaskProvider):
    """Local overlay detector. Allowed methods only: matchTemplate, frame
    differencing, and edge/contrast heuristics.

    On a single still (no ``previous_frames``), static-region frame
    differencing is not available. Detection then degrades to template
    matching when a PNG(+alpha) template is set, otherwise local
    edge/contrast heuristics. When a template is set, only template
    matches are returned — heuristics are not mixed in (they latch onto
    photo details). ``get_mask`` never invents a mask from unconfirmed
    candidates — the user must ``confirm_candidate`` first.
    """

    def __init__(
        self,
        *,
        template: np.ndarray | None = None,
        sensitivity: float = 50.0,
        previous_frames: Sequence[np.ndarray] | None = None,
        confirmed_mask: np.ndarray | None = None,
        threshold_bias: float = 0.0,
    ) -> None:
        self._template = None if template is None else _as_template(template)
        self._sensitivity = _clip_sensitivity(sensitivity)
        self._previous_frames = (
            [_as_bgr(frame) for frame in previous_frames]
            if previous_frames
            else []
        )
        self._confirmed: np.ndarray | None = (
            None if confirmed_mask is None else validate_mask_array(confirmed_mask)
        )
        self._threshold_bias = float(threshold_bias)
        self.last_template_peak: float | None = None

    @property
    def threshold_bias(self) -> float:
        return self._threshold_bias

    def match_threshold(self) -> float:
        scale = self._sensitivity / 100.0
        base = 0.70 - 0.40 * scale
        return float(np.clip(base + self._threshold_bias, 0.28, 0.90))

    def detect_candidates(self, frame: np.ndarray, frame_idx: int) -> list[MaskCandidate]:
        """Return overlay candidates. Does not write a final mask."""
        del frame_idx
        bgr = _as_bgr(frame)
        self.last_template_peak = None
        if self._template is not None:
            return self._template_candidates(bgr)
        found: list[MaskCandidate] = []
        if len(self._previous_frames) >= 1:
            found.extend(self._frame_diff_candidates(bgr))
        found.extend(self._edge_contrast_candidates(bgr))
        return found

    def get_mask(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        del frame_idx
        if self._confirmed is None:
            raise MaskError("no confirmed mask; accept a candidate before inpaint")
        if frame.ndim < 2:
            raise MaskError("frame must have at least 2 dimensions")
        frame_hw = (int(frame.shape[0]), int(frame.shape[1]))
        if self._confirmed.shape != frame_hw:
            raise MaskError(
                f"mask shape {self._confirmed.shape} does not match frame {frame_hw}"
            )
        return self._confirmed

    def confirm_candidate(self, candidate: MaskCandidate) -> np.ndarray:
        self._confirmed = validate_mask_array(candidate.mask)
        self._threshold_bias = max(self._threshold_bias - 0.02, -0.2)
        return self._confirmed

    def reject_candidate(self, candidate: MaskCandidate) -> None:
        del candidate
        self._threshold_bias = min(self._threshold_bias + 0.05, 0.3)

    def _template_candidates(self, frame: np.ndarray) -> list[MaskCandidate]:
        templ_bgr, alpha = _split_template(self._template)
        height, width = frame.shape[:2]
        work, factor = _working_copy(frame)
        work_h, work_w = work.shape[:2]
        thresh = self.match_threshold()
        peaks: list[tuple[float, np.ndarray, tuple[int, int, int, int]]] = []
        for scale in _template_scales(
            templ_bgr.shape[0], templ_bgr.shape[1], work_h, work_w
        ):
            th = max(int(round(templ_bgr.shape[0] * scale)), 1)
            tw = max(int(round(templ_bgr.shape[1] * scale)), 1)
            if th >= work_h or tw >= work_w or th < 3 or tw < 3:
                continue
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            scaled = cv2.resize(templ_bgr, (tw, th), interpolation=interp)
            scaled_alpha: np.ndarray | None = None
            if alpha is not None:
                scaled_alpha = cv2.resize(
                    alpha, (tw, th), interpolation=cv2.INTER_NEAREST
                )
            score = _match_template(work, scaled, scaled_alpha)
            if score is None:
                continue
            _min_v, max_v, _min_loc, max_loc = cv2.minMaxLoc(score)
            peak = float(max_v)
            if self.last_template_peak is None or peak > self.last_template_peak:
                self.last_template_peak = peak
            ys, xs = np.where(score >= thresh)
            if xs.size == 0:
                if peak >= thresh:
                    xs = np.array([int(max_loc[0])], dtype=np.int32)
                    ys = np.array([int(max_loc[1])], dtype=np.int32)
                    values = np.array([peak], dtype=np.float32)
                else:
                    continue
            else:
                values = score[ys, xs]
            order = np.argsort(-values)[:_MAX_CANDIDATES_PER_METHOD]
            for idx in order:
                x = int(xs[idx])
                y = int(ys[idx])
                conf = float(values[idx])
                mask_work = np.zeros((work_h, work_w), dtype=np.uint8)
                if scaled_alpha is None:
                    mask_work[y : y + th, x : x + tw] = 255
                else:
                    mask_work[y : y + th, x : x + tw] = np.where(
                        scaled_alpha > _ALPHA_MIN, np.uint8(255), np.uint8(0)
                    )
                if int(np.count_nonzero(mask_work)) == 0:
                    continue
                mask, bbox = _mask_to_full_frame(
                    mask_work, (height, width), factor, (x, y, tw, th)
                )
                peaks.append((conf, mask, bbox))
        return _nms_candidates(peaks, method="template")

    def _edge_contrast_candidates(self, frame: np.ndarray) -> list[MaskCandidate]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = self._sensitivity / 100.0
        median = float(np.median(gray))
        canny_hi = int(np.clip(median * (1.2 - 0.3 * scale) + 24, 40, 200))
        canny_lo = max(canny_hi // 2, 10)
        edges = cv2.Canny(gray, canny_lo, canny_hi)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
        contour_cands = _contours_to_candidates(
            closed, gray=gray, max_keep=_MAX_CANDIDATES_PER_METHOD
        )
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        residual = cv2.max(cv2.subtract(gray, blur), cv2.subtract(blur, gray))
        resid_thresh = int(np.clip(14 + (1.0 - scale) * 16, 10, 40))
        _ok, residual_bin = cv2.threshold(residual, resid_thresh, 255, cv2.THRESH_BINARY)
        residual_bin = cv2.morphologyEx(
            residual_bin, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)
        )
        residual_cands = _components_to_candidates(
            residual_bin,
            score_map=residual,
            method="edge_contrast",
            max_keep=_MAX_CANDIDATES_PER_METHOD,
        )
        if not contour_cands:
            return residual_cands
        extras = [
            candidate
            for candidate in residual_cands
            if all(
                _bbox_iou(candidate.bbox, contour.bbox) < 0.3
                for contour in contour_cands
            )
        ]
        return (contour_cands + extras)[:_MAX_CANDIDATES_PER_METHOD]

    def _frame_diff_candidates(self, frame: np.ndarray) -> list[MaskCandidate]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        acc: np.ndarray | None = None
        for prev in self._previous_frames:
            prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
            if prev_gray.shape != gray.shape:
                prev_gray = cv2.resize(
                    prev_gray,
                    (int(gray.shape[1]), int(gray.shape[0])),
                    interpolation=cv2.INTER_AREA,
                )
            diff = cv2.absdiff(gray, prev_gray)
            acc = diff if acc is None else np.maximum(acc, diff)
        if acc is None:
            return []
        scale = self._sensitivity / 100.0
        static_thresh = int(np.clip(18 + (1.0 - scale) * 24, 8, 60))
        static = np.where(acc < static_thresh, np.uint8(255), np.uint8(0))
        static_ratio = float(np.count_nonzero(static)) / float(static.size)
        if static_ratio > 0.45:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            contrast = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            _ok, contrast_bin = cv2.threshold(contrast, 24, 255, cv2.THRESH_BINARY)
            static = cv2.bitwise_and(static, contrast_bin)
        static = cv2.morphologyEx(
            static, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)
        )
        static = cv2.morphologyEx(
            static, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8)
        )
        score_map = np.subtract(255, acc)
        return _components_to_candidates(
            static,
            score_map=score_map,
            method="frame_diff",
            max_keep=_MAX_CANDIDATES_PER_METHOD,
        )


def load_template(path: Path) -> np.ndarray:
    """Load a user-supplied template as BGR or BGRA uint8 (PNG+alpha allowed)."""
    src = Path(path)
    if not src.is_file():
        raise MaskError(f"template file does not exist: {src.name}")
    suffix = src.suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise MaskError(f"template must be PNG/JPG/WEBP, got '{src.suffix}'")
    payload = np.fromfile(str(src), dtype=np.uint8)
    image = cv2.imdecode(payload, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise MaskError(f"failed to decode template: {src.name}")
    return _as_template(image)


def template_to_display_rgb(template: np.ndarray) -> np.ndarray:
    """BGR/BGRA template → RGB preview (alpha composited on a checkerboard)."""
    arr = _as_template(template)
    rgb = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2RGB)
    if arr.shape[2] != 4:
        return rgb
    alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
    board = _checkerboard(int(rgb.shape[0]), int(rgb.shape[1]))
    blended = rgb.astype(np.float32) * alpha + board.astype(np.float32) * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _clip_sensitivity(value: float) -> float:
    return float(np.clip(value, 1.0, 100.0))


def _as_bgr(frame: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(frame), dtype=np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.ndim != 3 or arr.shape[2] not in {3, 4}:
        raise MaskError("frame must have shape (H, W, 3)")
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr


def _as_template(image: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(image), dtype=np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.ndim != 3 or arr.shape[2] not in {3, 4}:
        raise MaskError("template must have shape (H, W, 3) or (H, W, 4)")
    if arr.shape[2] == 4:
        arr = _trim_alpha(arr)
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        raise MaskError("template is too small")
    return arr


def _trim_alpha(template: np.ndarray) -> np.ndarray:
    alpha = template[:, :, 3]
    ys, xs = np.nonzero(alpha > _ALPHA_MIN)
    if xs.size == 0:
        raise MaskError("template alpha is empty")
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return template[y0:y1, x0:x1]


def _split_template(template: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    bgr = template[:, :, :3]
    if template.shape[2] == 4:
        return bgr, template[:, :, 3]
    return bgr, None


def _working_copy(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """Downscale large stills for matchTemplate; return (work_bgr, work/orig)."""
    height, width = frame.shape[:2]
    long_edge = max(height, width)
    if long_edge <= _DETECT_LONG_EDGE:
        return frame, 1.0
    factor = _DETECT_LONG_EDGE / float(long_edge)
    new_w = max(int(round(width * factor)), 1)
    new_h = max(int(round(height * factor)), 1)
    work = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return work, factor


def _mask_to_full_frame(
    mask_work: np.ndarray,
    frame_hw: tuple[int, int],
    factor: float,
    bbox_work: tuple[int, int, int, int],
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    frame_h, frame_w = frame_hw
    if factor == 1.0 and mask_work.shape == (frame_h, frame_w):
        return mask_work, bbox_work
    mask = cv2.resize(
        mask_work, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST
    )
    mask = validate_mask_array(mask)
    x, y, box_w, box_h = bbox_work
    inv = 1.0 / factor if factor else 1.0
    x0 = int(np.clip(round(x * inv), 0, frame_w - 1))
    y0 = int(np.clip(round(y * inv), 0, frame_h - 1))
    x1 = int(np.clip(round((x + box_w) * inv), x0 + 1, frame_w))
    y1 = int(np.clip(round((y + box_h) * inv), y0 + 1, frame_h))
    return mask, (x0, y0, x1 - x0, y1 - y0)


def _template_scales(
    templ_h: int, templ_w: int, frame_h: int, frame_w: int
) -> tuple[float, ...]:
    max_s = min((frame_h - 1) / float(templ_h), (frame_w - 1) / float(templ_w)) * 0.98
    if max_s <= 0:
        return ()
    min_s = min(max_s, max(8.0 / float(templ_h), 8.0 / float(templ_w), 0.12))
    if min_s >= max_s:
        return (float(max_s),)
    raw = np.geomspace(min_s, max_s, num=7)
    scales = {float(np.round(value, 4)) for value in raw.tolist()}
    if min_s <= 1.0 <= max_s:
        scales.add(1.0)
    return tuple(sorted(scales))


def _match_template(
    frame: np.ndarray,
    templ: np.ndarray,
    alpha: np.ndarray | None,
) -> np.ndarray | None:
    if templ.shape[0] >= frame.shape[0] or templ.shape[1] >= frame.shape[1]:
        return None
    mask: np.ndarray | None = None
    if alpha is not None:
        mask = np.where(alpha > _ALPHA_MIN, np.uint8(255), np.uint8(0))
        if int(np.count_nonzero(mask)) == 0:
            return None
    color = _match_one(frame, templ, mask)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_templ = cv2.cvtColor(templ, cv2.COLOR_BGR2GRAY)
    gray = _match_one(gray_frame, gray_templ, mask)
    if color is None:
        return gray
    if gray is None:
        return color
    return np.maximum(color, gray)


def _match_one(
    image: np.ndarray,
    templ: np.ndarray,
    mask: np.ndarray | None,
) -> np.ndarray | None:
    if mask is None:
        return cv2.matchTemplate(image, templ, cv2.TM_CCOEFF_NORMED)
    try:
        return cv2.matchTemplate(image, templ, cv2.TM_CCOEFF_NORMED, mask=mask)
    except cv2.error:
        return cv2.matchTemplate(image, templ, cv2.TM_CCORR_NORMED, mask=mask)


def _overlay_location_score(
    x: int, y: int, box_w: int, box_h: int, frame_w: int, frame_h: int
) -> float:
    cx = x + box_w / 2.0
    cy = y + box_h / 2.0
    dx = min(cx, frame_w - cx) / max(frame_w / 2.0, 1.0)
    dy = min(cy, frame_h - cy) / max(frame_h / 2.0, 1.0)
    border = 1.0 - min(dx, dy)
    area_frac = (box_w * box_h) / float(max(frame_w * frame_h, 1))
    if area_frac > 0.22:
        size = 0.25
    elif area_frac > 0.12:
        size = 0.6
    else:
        size = 1.0
    return float(np.clip(0.35 + 0.40 * border + 0.25 * size, 0.0, 1.0))


def _contours_to_candidates(
    edges: np.ndarray,
    *,
    gray: np.ndarray,
    max_keep: int,
) -> list[MaskCandidate]:
    height, width = edges.shape
    min_area = max(12, int(0.002 * height * width))
    max_area = int(0.22 * height * width)
    contours, _hierarchy = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    scored: list[tuple[float, np.ndarray, tuple[int, int, int, int]]] = []
    ring_kernel = np.ones((5, 5), dtype=np.uint8)
    for contour in contours:
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        area = int(np.count_nonzero(mask))
        if area < min_area or area > max_area:
            continue
        x, y, box_w, box_h = (int(v) for v in cv2.boundingRect(contour))
        if box_w >= width - 1 and box_h >= height - 1:
            continue
        interior = gray[mask > 0]
        dilated = cv2.dilate(mask, ring_kernel)
        ring = (dilated > 0) & (mask == 0)
        ring_vals = gray[ring]
        if interior.size == 0:
            continue
        uniformity = 1.0 - min(float(interior.std()) / 64.0, 1.0)
        if ring_vals.size:
            contrast = abs(float(interior.mean()) - float(ring_vals.mean())) / 255.0
        else:
            contrast = 0.0
        fill = area / float(max(box_w * box_h, 1))
        location = _overlay_location_score(x, y, box_w, box_h, width, height)
        confidence = float(
            np.clip(
                (0.35 * contrast + 0.4 * uniformity + 0.25 * fill) * location,
                0.0,
                1.0,
            )
        )
        scored.append((confidence, mask, (x, y, box_w, box_h)))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        _to_candidate(mask, confidence, "edge_contrast", bbox)
        for confidence, mask, bbox in scored[:max_keep]
    ]


def _components_to_candidates(
    binary: np.ndarray,
    *,
    score_map: np.ndarray,
    method: str,
    max_keep: int,
) -> list[MaskCandidate]:
    height, width = binary.shape
    min_area = max(12, int(0.002 * height * width))
    max_frac = 0.45 if method == "frame_diff" else 0.22
    max_area = int(max_frac * height * width)
    num, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    scored: list[tuple[float, np.ndarray, tuple[int, int, int, int]]] = []
    for index in range(1, num):
        x, y, box_w, box_h, area = (int(v) for v in stats[index])
        if area < min_area or area > max_area or box_w < 2 or box_h < 2:
            continue
        if box_w >= width - 1 and box_h >= height - 1:
            continue
        mask = np.where(labels == index, np.uint8(255), np.uint8(0))
        region = score_map[mask > 0]
        mean_score = float(region.mean()) / 255.0 if region.size else 0.0
        fill = area / float(max(box_w * box_h, 1))
        location = _overlay_location_score(x, y, box_w, box_h, width, height)
        confidence = float(
            np.clip((0.35 + 0.5 * mean_score + 0.15 * fill) * location, 0.0, 1.0)
        )
        scored.append((confidence, mask, (x, y, box_w, box_h)))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        _to_candidate(mask, confidence, method, bbox)
        for confidence, mask, bbox in scored[:max_keep]
    ]


def _nms_candidates(
    peaks: list[tuple[float, np.ndarray, tuple[int, int, int, int]]],
    *,
    method: str,
    iou_thresh: float = 0.3,
) -> list[MaskCandidate]:
    peaks = sorted(peaks, key=lambda item: item[0], reverse=True)
    kept: list[tuple[float, np.ndarray, tuple[int, int, int, int]]] = []
    for item in peaks:
        if any(_bbox_iou(item[2], other[2]) >= iou_thresh for other in kept):
            continue
        kept.append(item)
        if len(kept) >= _MAX_CANDIDATES_PER_METHOD:
            break
    return [
        _to_candidate(mask, confidence, method, bbox)
        for confidence, mask, bbox in kept
    ]


def _to_candidate(
    mask: np.ndarray,
    confidence: float,
    method: str,
    bbox: tuple[int, int, int, int],
) -> MaskCandidate:
    binary = validate_mask_array(mask)
    return MaskCandidate(
        mask=binary,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        method=method,
        bbox=bbox,
    )


def _bbox_iou(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    ax, ay, aw, ah = left
    bx, by, bw, bh = right
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / float(union)


def _checkerboard(height: int, width: int, cell: int = 8) -> np.ndarray:
    yy, xx = np.indices((height, width))
    flag = ((yy // cell) + (xx // cell)) % 2
    out = np.empty((height, width, 3), dtype=np.uint8)
    out[flag == 0] = (228, 228, 228)
    out[flag == 1] = (252, 252, 252)
    return out
