from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from watermark_remover.exceptions import MaskError
from watermark_remover.io.image import read_image
from watermark_remover.masks.auto_detect import AutoDetectMaskProvider, load_template
from watermark_remover.masks.base import MaskCandidate, MaskProvider
from watermark_remover.masks.serialize import load_mask_png

IOU_MIN = 0.5


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    a = left > 0
    b = right > 0
    intersection = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 0.0
    return intersection / union


def _best_iou(candidates: list[MaskCandidate], gt: np.ndarray) -> float:
    if not candidates:
        return 0.0
    return max(_iou(candidate.mask, gt) for candidate in candidates)


def _precision_recall(
    candidates: list[MaskCandidate],
    gt: np.ndarray,
    *,
    iou_min: float = IOU_MIN,
) -> tuple[float, float, int, int]:
    """Single-object precision/recall: TP if IoU(candidate, GT) >= iou_min."""
    true_positives = sum(1 for candidate in candidates if _iou(candidate.mask, gt) >= iou_min)
    false_positives = len(candidates) - true_positives
    false_negatives = 0 if true_positives else 1
    precision = (
        true_positives / float(true_positives + false_positives)
        if (true_positives + false_positives)
        else 0.0
    )
    recall = (
        true_positives / float(true_positives + false_negatives)
        if (true_positives + false_negatives)
        else 0.0
    )
    return precision, recall, true_positives, false_positives


def _assert_candidate_contract(candidate: MaskCandidate, frame: np.ndarray) -> None:
    assert candidate.mask.dtype == np.uint8
    assert set(np.unique(candidate.mask).tolist()).issubset({0, 255})
    assert candidate.mask.shape == frame.shape[:2]
    assert 0.0 <= float(candidate.confidence) <= 1.0
    assert isinstance(candidate.method, str) and candidate.method
    x, y, width, height = candidate.bbox
    assert width > 0 and height > 0
    assert x >= 0 and y >= 0
    assert x + width <= frame.shape[1]
    assert y + height <= frame.shape[0]


def test_auto_detect_is_mask_provider() -> None:
    provider = AutoDetectMaskProvider()
    assert isinstance(provider, MaskProvider)


def test_detect_candidates_contract_on_synthetic() -> None:
    frame = np.full((32, 48, 3), 40, dtype=np.uint8)
    frame[4:12, 6:20] = (0, 210, 255)
    provider = AutoDetectMaskProvider(sensitivity=50.0)
    candidates = provider.detect_candidates(frame, 0)
    assert isinstance(candidates, list)
    for candidate in candidates:
        _assert_candidate_contract(candidate, frame)


def test_get_mask_never_auto_applies(fixtures_dir: Path) -> None:
    frame = read_image(fixtures_dir / "detect_pos1.png")
    template = load_template(fixtures_dir / "template_logo.png")
    provider = AutoDetectMaskProvider(template=template, sensitivity=50.0)
    candidates = provider.detect_candidates(frame, 0)
    assert candidates
    with pytest.raises(MaskError, match="confirm"):
        provider.get_mask(frame, 0)


def test_get_mask_returns_confirmed_candidate(fixtures_dir: Path) -> None:
    frame = read_image(fixtures_dir / "detect_pos1.png")
    template = load_template(fixtures_dir / "template_logo.png")
    provider = AutoDetectMaskProvider(template=template, sensitivity=50.0)
    candidates = provider.detect_candidates(frame, 0)
    provider.confirm_candidate(candidates[0])
    out = provider.get_mask(frame, 0)
    assert out.dtype == np.uint8
    assert set(np.unique(out).tolist()).issubset({0, 255})
    assert out.shape == frame.shape[:2]
    assert np.array_equal(out, candidates[0].mask)


def test_detect_candidates_does_not_write_files(
    fixtures_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    frame = read_image(fixtures_dir / "detect_pos1.png")
    template = load_template(fixtures_dir / "template_logo.png")
    AutoDetectMaskProvider(template=template, sensitivity=50.0).detect_candidates(
        frame, 0
    )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "stem",
    ("detect_pos1", "detect_pos2", "detect_scale"),
)
def test_template_candidates_iou_vs_gt(fixtures_dir: Path, stem: str) -> None:
    frame = read_image(fixtures_dir / f"{stem}.png")
    gt = load_mask_png(fixtures_dir / f"{stem}.mask.png")
    template = load_template(fixtures_dir / "template_logo.png")
    provider = AutoDetectMaskProvider(template=template, sensitivity=50.0)
    candidates = provider.detect_candidates(frame, 0)
    template_hits = [c for c in candidates if c.method == "template"]
    assert template_hits
    precision, recall, true_positives, false_positives = _precision_recall(template_hits, gt)
    assert _best_iou(template_hits, gt) >= IOU_MIN
    assert true_positives >= 1
    assert recall == 1.0
    assert precision == 1.0
    assert false_positives == 0
    with pytest.raises(MaskError, match="confirm"):
        provider.get_mask(frame, 0)


def test_edge_contrast_on_still_logo_iou(fixtures_dir: Path) -> None:
    frame = read_image(fixtures_dir / "still_logo.png")
    gt = load_mask_png(fixtures_dir / "still_logo.mask.png")
    provider = AutoDetectMaskProvider(sensitivity=50.0)
    candidates = provider.detect_candidates(frame, 0)
    assert candidates
    assert all(c.method != "frame_diff" for c in candidates)
    assert any(c.method == "edge_contrast" for c in candidates)
    precision, recall, true_positives, false_positives = _precision_recall(candidates, gt)
    assert _best_iou(candidates, gt) >= IOU_MIN
    assert true_positives >= 1
    assert recall == 1.0
    assert precision >= 0.5
    assert false_positives <= true_positives


def test_single_still_skips_frame_differencing() -> None:
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    frame[2:10, 4:16] = 255
    provider = AutoDetectMaskProvider(sensitivity=80.0)
    candidates = provider.detect_candidates(frame, 0)
    assert candidates
    assert all(c.method != "frame_diff" for c in candidates)
    assert any(c.method == "edge_contrast" for c in candidates)


def test_frame_diff_static_logo_iou() -> None:
    h, w = 36, 48
    logo = np.zeros((10, 14, 3), dtype=np.uint8)
    logo[:, :] = (0, 200, 255)
    x, y = 8, 6

    def with_bg(color: tuple[int, int, int]) -> np.ndarray:
        frame = np.full((h, w, 3), color, dtype=np.uint8)
        frame[y : y + 10, x : x + 14] = logo
        return frame

    current = with_bg((20, 40, 80))
    previous = [with_bg((180, 30, 30)), with_bg((40, 160, 50))]
    gt = np.zeros((h, w), dtype=np.uint8)
    gt[y : y + 10, x : x + 14] = 255
    provider = AutoDetectMaskProvider(
        sensitivity=50.0,
        previous_frames=previous,
    )
    candidates = provider.detect_candidates(current, 1)
    diff_hits = [c for c in candidates if c.method == "frame_diff"]
    assert diff_hits
    precision, recall, true_positives, false_positives = _precision_recall(diff_hits, gt)
    assert _best_iou(diff_hits, gt) >= IOU_MIN
    assert true_positives >= 1
    assert recall == 1.0
    assert precision == 1.0
    assert false_positives == 0
    with pytest.raises(MaskError, match="confirm"):
        provider.get_mask(current, 1)


def test_reject_raises_session_threshold_without_applying() -> None:
    frame = np.full((32, 40, 3), 30, dtype=np.uint8)
    frame[4:12, 5:18] = (0, 210, 255)
    provider = AutoDetectMaskProvider(sensitivity=50.0)
    before = provider.detect_candidates(frame, 0)
    assert before
    provider.reject_candidate(before[0])
    with pytest.raises(MaskError, match="confirm"):
        provider.get_mask(frame, 0)
    assert provider.threshold_bias > 0.0


def test_load_template_reads_alpha(fixtures_dir: Path) -> None:
    template = load_template(fixtures_dir / "template_logo.png")
    assert template.ndim == 3
    assert template.shape[2] == 4
    assert template.dtype == np.uint8
    assert int(template[:, :, 3].min()) == 255


def test_template_mode_does_not_mix_edge_contrast(fixtures_dir: Path) -> None:
    frame = read_image(fixtures_dir / "detect_pos1.png")
    template = load_template(fixtures_dir / "template_logo.png")
    candidates = AutoDetectMaskProvider(
        template=template, sensitivity=50.0
    ).detect_candidates(frame, 0)
    assert candidates
    assert all(candidate.method == "template" for candidate in candidates)


def test_oversized_template_still_matches(fixtures_dir: Path) -> None:
    import cv2

    frame = read_image(fixtures_dir / "detect_pos1.png")
    gt = load_mask_png(fixtures_dir / "detect_pos1.mask.png")
    template = load_template(fixtures_dir / "template_logo.png")
    big = cv2.resize(
        template,
        (int(template.shape[1] * 5), int(template.shape[0] * 5)),
        interpolation=cv2.INTER_NEAREST,
    )
    provider = AutoDetectMaskProvider(template=big, sensitivity=50.0)
    candidates = provider.detect_candidates(frame, 0)
    assert candidates
    assert _best_iou(candidates, gt) >= IOU_MIN


def test_semitransparent_template_keeps_logo_pixels(fixtures_dir: Path) -> None:
    frame = read_image(fixtures_dir / "detect_pos1.png")
    gt = load_mask_png(fixtures_dir / "detect_pos1.mask.png")
    template = load_template(fixtures_dir / "template_logo.png").copy()
    template[:, :, 3] = 80
    provider = AutoDetectMaskProvider(template=template, sensitivity=80.0)
    candidates = provider.detect_candidates(frame, 0)
    assert candidates
    assert int(np.count_nonzero(candidates[0].mask)) > 0
    assert _best_iou(candidates, gt) >= IOU_MIN


def test_template_match_survives_work_image_downscale(
    fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("watermark_remover.masks.auto_detect._DETECT_LONG_EDGE", 40)
    frame = read_image(fixtures_dir / "detect_pos1.png")
    gt = load_mask_png(fixtures_dir / "detect_pos1.mask.png")
    template = load_template(fixtures_dir / "template_logo.png")
    provider = AutoDetectMaskProvider(template=template, sensitivity=50.0)
    candidates = provider.detect_candidates(frame, 0)
    assert candidates
    assert candidates[0].mask.shape == frame.shape[:2]
    assert _best_iou(candidates, gt) >= IOU_MIN


def test_load_template_missing_and_bad_suffix(tmp_path: Path) -> None:
    with pytest.raises(MaskError, match="does not exist"):
        load_template(tmp_path / "nope.png")
    txt = tmp_path / "logo.txt"
    txt.write_text("x", encoding="utf-8")
    with pytest.raises(MaskError, match="PNG/JPG/WEBP"):
        load_template(txt)
