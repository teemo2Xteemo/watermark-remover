from __future__ import annotations

import ast
import inspect
import uuid
from pathlib import Path

import numpy as np
import pytest

from watermark_remover.config import Settings
from watermark_remover.exceptions import InputValidationError, MaskError
from watermark_remover.image_processor import ImageProcessor
from watermark_remover.ui.app import (
    bgr_to_rgb,
    cuda_available,
    is_run_enabled,
    lama_cpu_warning_message,
    launch_kwargs,
    new_job_id,
    on_run,
    overlay_mask_rgb,
    rgb_to_bgr,
    run_image_job,
    ui_mask_to_uint8,
)


def test_rgb_to_bgr_swaps_channels() -> None:
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[:, :] = (255, 0, 0)
    bgr = rgb_to_bgr(rgb)
    assert bgr.dtype == np.uint8
    assert bgr.shape == rgb.shape
    assert int(bgr[0, 0, 0]) == 0
    assert int(bgr[0, 0, 2]) == 255
    roundtrip = bgr_to_rgb(bgr)
    assert np.array_equal(roundtrip, rgb)


def test_ui_mask_to_uint8_from_ndarray() -> None:
    raw = np.zeros((8, 10), dtype=np.uint8)
    raw[1:4, 2:6] = 200
    out = ui_mask_to_uint8(raw)
    assert out.shape == (8, 10)
    assert out.dtype == np.uint8
    assert set(np.unique(out).tolist()).issubset({0, 255})
    assert int(out[2, 3]) == 255
    assert int(out[0, 0]) == 0


def test_ui_mask_to_uint8_from_image_editor_layers() -> None:
    layer = np.zeros((6, 8, 4), dtype=np.uint8)
    layer[1:3, 2:5, :] = (255, 0, 0, 255)
    editor = {
        "background": np.zeros((6, 8, 3), dtype=np.uint8),
        "layers": [layer],
        "composite": None,
    }
    out = ui_mask_to_uint8(editor)
    assert out.shape == (6, 8)
    assert out.dtype == np.uint8
    assert int(out[1, 3]) == 255
    assert int(out[0, 0]) == 0


def test_ui_mask_to_uint8_none_raises() -> None:
    with pytest.raises(MaskError, match="no mask"):
        ui_mask_to_uint8(None)


def test_run_disabled_without_confirmed_mask_or_preview() -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    assert is_run_enabled(mask, mask_confirmed=False, preview_ready=True) is False
    assert is_run_enabled(mask, mask_confirmed=True, preview_ready=False) is False
    assert is_run_enabled(None, mask_confirmed=True, preview_ready=True) is False
    empty = np.zeros((8, 8), dtype=np.uint8)
    assert is_run_enabled(empty, mask_confirmed=True, preview_ready=True) is False
    assert is_run_enabled(mask, mask_confirmed=True, preview_ready=True) is True


def test_on_run_converts_rgb_to_bgr_before_processor(monkeypatch: pytest.MonkeyPatch) -> None:
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[:, :] = (255, 0, 0)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:5, 2:5] = 255
    captured: dict[str, np.ndarray] = {}

    def fake_process(
        self: object,
        image: np.ndarray,
        mask_arg: np.ndarray,
        engine_name: str,
        config: Settings,
        **kwargs: object,
    ) -> np.ndarray:
        del self, mask_arg, engine_name, config, kwargs
        captured["image"] = image.copy()
        return image

    monkeypatch.setattr(
        "watermark_remover.image_processor.ImageProcessor.process",
        fake_process,
    )
    out = on_run(rgb, mask, "opencv", Settings())
    assert captured["image"][0, 0, 0] == 0
    assert captured["image"][0, 0, 2] == 255
    assert out.dtype == np.uint8
    assert int(out[0, 0, 0]) == 255
    assert int(out[0, 0, 2]) == 0


def test_on_run_opencv_returns_rgb(fixtures_dir: Path) -> None:
    from watermark_remover.io.image import read_image
    from watermark_remover.masks.serialize import load_mask_png

    bgr = read_image(fixtures_dir / "still_logo.png")
    rgb = bgr_to_rgb(bgr)
    mask = load_mask_png(fixtures_dir / "still_logo.mask.png")
    out = on_run(rgb, mask, "opencv", Settings())
    assert out.shape == rgb.shape
    assert out.dtype == np.uint8


def test_on_run_without_image_raises() -> None:
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0, 0] = 255
    with pytest.raises(InputValidationError, match="no input"):
        on_run(None, mask, "opencv", Settings())


def test_on_run_does_not_write_input_path(fixtures_dir: Path, tmp_path: Path) -> None:
    from watermark_remover.io.image import read_image
    from watermark_remover.masks.serialize import load_mask_png

    src = tmp_path / "still_logo.png"
    src.write_bytes((fixtures_dir / "still_logo.png").read_bytes())
    before = src.read_bytes()
    bgr = read_image(src)
    rgb = bgr_to_rgb(bgr)
    mask = load_mask_png(fixtures_dir / "still_logo.mask.png")
    on_run(rgb, mask, "opencv", Settings())
    assert src.read_bytes() == before


def test_overlay_mask_rgb_keeps_shape() -> None:
    image = np.full((10, 12, 3), 40, dtype=np.uint8)
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[2:6, 3:8] = 255
    overlay = overlay_mask_rgb(image, mask)
    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8
    assert not np.array_equal(overlay[3, 4], image[3, 4])
    assert np.array_equal(overlay[0, 0], image[0, 0])


def test_new_job_id_is_uuid4() -> None:
    value = new_job_id()
    parsed = uuid.UUID(value)
    assert parsed.version == 4
    assert value != "WM-R-089A4"
    assert new_job_id() != value


def test_launch_binds_localhost_without_share() -> None:
    kwargs = launch_kwargs(Settings())
    assert kwargs["server_name"] == "127.0.0.1"
    assert kwargs["share"] is False


def test_launch_share_stays_false_even_if_settings_true() -> None:
    kwargs = launch_kwargs(Settings(gradio_share=True))
    assert kwargs["share"] is False


def test_on_run_mask_annotation_is_ndarray() -> None:
    annotation = str(inspect.signature(on_run).parameters["mask"].annotation)
    assert "dict" not in annotation


def test_run_image_job_log_is_structlog_stream(fixtures_dir: Path) -> None:
    from watermark_remover.io.image import read_image
    from watermark_remover.masks.serialize import load_mask_png

    rgb = bgr_to_rgb(read_image(fixtures_dir / "still_logo.png"))
    mask = load_mask_png(fixtures_dir / "still_logo.mask.png")
    job = run_image_job(rgb, mask, "opencv", Settings(), stem="still_logo")
    assert job.percent == 100
    assert job.output_path is not None
    assert Path(job.output_path).name == "still_logo_inpainted.png"
    assert ">[SYS]" not in job.log_text
    assert ">[ENG]" not in job.log_text
    assert "inpaint_done" in job.log_text
    assert f"job_id={job.job_id}" in job.log_text
    assert "engine=opencv" in job.log_text
    assert "frame_idx=0" in job.log_text
    assert "duration_ms=" in job.log_text
    assert uuid.UUID(job.job_id).version == 4


def test_run_image_job_blocked_without_confirm(fixtures_dir: Path) -> None:
    from watermark_remover.io.image import read_image
    from watermark_remover.masks.serialize import load_mask_png

    rgb = bgr_to_rgb(read_image(fixtures_dir / "still_logo.png"))
    mask = load_mask_png(fixtures_dir / "still_logo.mask.png")
    job = run_image_job(
        rgb,
        mask,
        "opencv",
        Settings(),
        mask_confirmed=False,
        preview_ready=True,
    )
    assert job.output_path is None
    assert job.image_rgb is None
    assert "run_blocked" in job.log_text
    assert "error=mask not confirmed" in job.log_text


def test_run_image_job_cancel_before_inpaint(fixtures_dir: Path) -> None:
    from watermark_remover.io.image import read_image
    from watermark_remover.masks.serialize import load_mask_png

    rgb = bgr_to_rgb(read_image(fixtures_dir / "still_logo.png"))
    mask = load_mask_png(fixtures_dir / "still_logo.mask.png")
    job = run_image_job(
        rgb,
        mask,
        "opencv",
        Settings(),
        cancel_token={"requested": True},
    )
    assert job.cancel_requested is True
    assert job.output_path is None
    assert "job_cancelled" in job.log_text
    assert f"job_id={job.job_id}" in job.log_text


def test_run_image_job_cancel_skips_write_after_inpaint(
    fixtures_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from watermark_remover.io.image import read_image
    from watermark_remover.masks.serialize import load_mask_png

    rgb = bgr_to_rgb(read_image(fixtures_dir / "still_logo.png"))
    mask = load_mask_png(fixtures_dir / "still_logo.mask.png")
    token = {"requested": False}
    original = ImageProcessor.process

    def flip_then_process(
        self: ImageProcessor,
        image: np.ndarray,
        mask_arg: np.ndarray,
        engine_name: str,
        config: Settings,
        **kwargs: object,
    ) -> np.ndarray:
        token["requested"] = True
        return original(self, image, mask_arg, engine_name, config, **kwargs)

    monkeypatch.setattr(ImageProcessor, "process", flip_then_process)
    job = run_image_job(rgb, mask, "opencv", Settings(), cancel_token=token)
    assert job.cancel_requested is True
    assert job.output_path is None
    assert "inpaint_done" in job.log_text
    assert "job_cancelled" in job.log_text


def test_lama_cpu_warning_when_cuda_mocked_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("watermark_remover.ui.app.cuda_available", lambda: False)
    message = lama_cpu_warning_message()
    assert "CPU" in message
    assert "CUDA" in message


def test_cuda_available_returns_bool() -> None:
    assert isinstance(cuda_available(), bool)


def test_engines_masks_io_do_not_import_gradio() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "watermark_remover"
    skip_parts = {"ui"}
    for path in root.rglob("*.py"):
        if skip_parts.intersection(path.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("gradio")
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("gradio")


def test_import_and_export_mask_roundtrip(tmp_path: Path) -> None:
    from watermark_remover.ui.app import export_session_masks, import_mask_from_path

    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[2:6, 3:9] = 255
    png_path, json_path = export_session_masks(mask, "photo", tmp_path)
    assert png_path.name == "photo.mask.png"
    assert json_path.name == "photo.mask.json"
    assert np.array_equal(import_mask_from_path(png_path, (12, 16)), mask)
    loaded_json = import_mask_from_path(json_path, (12, 16))
    assert loaded_json.shape == (12, 16)
    assert int(loaded_json.max()) == 255


def test_union_bbox_mask() -> None:
    from watermark_remover.ui.app import union_bbox_mask

    out = union_bbox_mask(None, (10, 12), 2, 3, 4, 2)
    assert out.shape == (10, 12)
    assert int(out[3, 3]) == 255
    assert int(out[0, 0]) == 0


def test_build_app_smoke() -> None:
    pytest.importorskip("gradio")
    from watermark_remover.ui.app import SECTION_TITLES, build_app

    demo = build_app(Settings())
    assert demo is not None
    assert getattr(demo, "analytics_enabled", False) is False
    assert SECTION_TITLES == ("Input", "Mask", "Preview", "Engine", "Run")


def test_ui_app_has_no_inpaint_math() -> None:
    src = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "watermark_remover"
        / "ui"
        / "app.py"
    )
    text = src.read_text(encoding="utf-8")
    assert "cv2.inpaint" not in text
    assert "INPAINT_TELEA" not in text
    assert "INPAINT_NS" not in text
