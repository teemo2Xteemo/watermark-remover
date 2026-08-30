from __future__ import annotations

import shutil
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import structlog

from watermark_remover.config import Settings, get_settings
from watermark_remover.exceptions import (
    EngineError,
    InputValidationError,
    MaskError,
    ResourceLimitError,
)
from watermark_remover.image_processor import ImageProcessor
from watermark_remover.io.image import read_image, write_image_atomic
from watermark_remover.io.validate import (
    refuse_overwrite_unless_flag,
    validate_input_path,
    validate_resolution_limits,
    validate_size_limits,
)
from watermark_remover.masks.base import validate_mask_array
from watermark_remover.masks.serialize import (
    export_mask_json,
    export_mask_png,
    load_mask_json,
    load_mask_png,
    mask_to_polygon_payload,
)

EngineName = Literal["opencv", "lama", "auto"]
SECTION_TITLES = ("Input", "Mask", "Preview", "Engine", "Run")

_OVERLAY_COLOR = np.array([15, 98, 254], dtype=np.float32)
_OVERLAY_ALPHA = 0.45
_MASK_LAYER_RGBA = (15, 98, 254, 180)
_STREAM_FIELDS = ("job_id", "engine", "frame_idx", "duration_ms", "error")


@dataclass(frozen=True)
class ImageJobResult:
    image_rgb: np.ndarray | None
    output_path: str | None
    temp_dir: str | None
    log_text: str
    percent: int
    job_id: str
    cancel_requested: bool
    status: str


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    rgb = _as_rgb(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    bgr = np.ascontiguousarray(np.asarray(image), dtype=np.uint8)
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise InputValidationError("image must have shape (H, W, 3)")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def ui_mask_to_uint8(mask: np.ndarray | dict[str, Any] | None) -> np.ndarray:
    if mask is None:
        raise MaskError("no mask")
    if isinstance(mask, dict):
        return _mask_from_editor_dict(mask)
    return validate_mask_array(np.asarray(mask))


def is_run_enabled(
    mask: np.ndarray | dict[str, Any] | None,
    mask_confirmed: bool,
    preview_ready: bool,
) -> bool:
    if not mask_confirmed or not preview_ready:
        return False
    try:
        binary = ui_mask_to_uint8(mask)
    except MaskError:
        return False
    return int(np.count_nonzero(binary)) > 0


def overlay_mask_rgb(image: np.ndarray, mask: np.ndarray | dict[str, Any] | None) -> np.ndarray:
    rgb = _as_rgb(image)
    binary = ui_mask_to_uint8(mask)
    if binary.shape != rgb.shape[:2]:
        binary = cv2.resize(
            binary,
            (int(rgb.shape[1]), int(rgb.shape[0])),
            interpolation=cv2.INTER_NEAREST,
        )
        binary = validate_mask_array(binary)
    overlay = rgb.astype(np.float32)
    region = binary > 0
    overlay[region] = overlay[region] * (1.0 - _OVERLAY_ALPHA) + _OVERLAY_COLOR * _OVERLAY_ALPHA
    return np.clip(overlay, 0, 255).astype(np.uint8)


def on_run(
    image: np.ndarray | None,
    mask: np.ndarray | None,
    engine_name: EngineName,
    config: Settings,
) -> np.ndarray:
    if image is None:
        raise InputValidationError("no input image")
    rgb = _as_rgb(image)
    mask_u8 = ui_mask_to_uint8(mask)
    if mask_u8.shape != rgb.shape[:2]:
        mask_u8 = cv2.resize(
            mask_u8,
            (int(rgb.shape[1]), int(rgb.shape[0])),
            interpolation=cv2.INTER_NEAREST,
        )
        mask_u8 = validate_mask_array(mask_u8)
    result_bgr = ImageProcessor().process(rgb_to_bgr(rgb), mask_u8, engine_name, config)
    return bgr_to_rgb(result_bgr)


def new_job_id() -> str:
    return str(uuid.uuid4())


def launch_kwargs(settings: Settings) -> dict[str, Any]:
    return {
        "server_name": settings.gradio_server_name,
        "share": False,
        "max_file_size": int(settings.max_input_bytes),
        "show_error": True,
        "footer_links": [],
    }


def cuda_available() -> bool:
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        pass
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def lama_cpu_warning_message() -> str:
    if cuda_available():
        return "GPU (CUDA): available"
    return (
        "LaMa CPU warning: CUDA is unavailable. "
        "LaMa will run on CPU (slow) when that engine is enabled."
    )


def import_mask_from_path(path: Path, frame_hw: tuple[int, int]) -> np.ndarray:
    src = Path(path)
    suffix = src.suffix.lower()
    if suffix == ".json":
        return load_mask_json(src, frame_hw)
    if suffix == ".png":
        mask = load_mask_png(src)
        if mask.shape != frame_hw:
            mask = cv2.resize(
                mask,
                (int(frame_hw[1]), int(frame_hw[0])),
                interpolation=cv2.INTER_NEAREST,
            )
            mask = validate_mask_array(mask)
        return mask
    raise MaskError(f"mask must be .png or .json, got '{src.suffix}'")


def export_session_masks(mask: np.ndarray, stem: str, dest_dir: Path) -> tuple[Path, Path]:
    binary = ui_mask_to_uint8(mask)
    dest_dir.mkdir(parents=True, exist_ok=True)
    png_path = dest_dir / f"{stem}.mask.png"
    json_path = dest_dir / f"{stem}.mask.json"
    export_mask_png(png_path, binary)
    export_mask_json(json_path, mask_to_polygon_payload(binary))
    return png_path, json_path


def union_bbox_mask(
    mask: np.ndarray | None,
    frame_hw: tuple[int, int],
    x: int,
    y: int,
    width: int,
    height: int,
) -> np.ndarray:
    if mask is None:
        canvas = np.zeros(frame_hw, dtype=np.uint8)
    else:
        canvas = ui_mask_to_uint8(mask)
        if canvas.shape != frame_hw:
            canvas = cv2.resize(
                canvas,
                (int(frame_hw[1]), int(frame_hw[0])),
                interpolation=cv2.INTER_NEAREST,
            )
            canvas = validate_mask_array(canvas)
    x0 = max(int(x), 0)
    y0 = max(int(y), 0)
    x1 = min(x0 + max(int(width), 0), int(frame_hw[1]))
    y1 = min(y0 + max(int(height), 0), int(frame_hw[0]))
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = 255
    return canvas


def cleanup_temp_dir(path: str | None) -> None:
    if not path:
        return
    dest = Path(path)
    if dest.is_dir():
        shutil.rmtree(dest, ignore_errors=True)


def format_structlog_event(event_dict: dict[str, Any]) -> str:
    event = event_dict.get("event")
    parts: list[str] = []
    if event:
        parts.append(str(event))
    for key in _STREAM_FIELDS:
        value = event_dict.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    return " ".join(parts)


@contextmanager
def capturing_structlog() -> Iterator[list[str]]:
    lines: list[str] = []

    def processor(
        logger: object, method_name: str, event_dict: dict[str, Any]
    ) -> dict[str, Any]:
        del logger, method_name
        formatted = format_structlog_event(event_dict)
        if formatted:
            lines.append(formatted)
        return event_dict

    previous = structlog.get_config()
    processors = list(previous.get("processors") or [])
    if processors:
        chained = [*processors[:-1], processor, processors[-1]]
    else:
        chained = [structlog.contextvars.merge_contextvars, processor]
    structlog.configure(
        processors=chained,
        wrapper_class=previous.get("wrapper_class"),
        context_class=previous.get("context_class", dict),
        logger_factory=previous.get("logger_factory"),
        cache_logger_on_first_use=False,
    )
    try:
        yield lines
    finally:
        structlog.configure(
            processors=previous.get("processors"),
            wrapper_class=previous.get("wrapper_class"),
            context_class=previous.get("context_class"),
            logger_factory=previous.get("logger_factory"),
            cache_logger_on_first_use=previous.get("cache_logger_on_first_use", True),
        )


def _cancel_requested(token: dict[str, Any] | bool | None) -> bool:
    if isinstance(token, dict):
        return bool(token.get("requested"))
    return bool(token)


def run_image_job(
    image: np.ndarray | None,
    mask: np.ndarray | None,
    engine_name: EngineName,
    config: Settings,
    *,
    stem: str = "image",
    input_path: str | None = None,
    mask_confirmed: bool = True,
    preview_ready: bool = True,
    cancel_token: dict[str, Any] | bool | None = None,
) -> ImageJobResult:
    job_id = new_job_id()

    def _done(
        lines: list[str],
        *,
        image_rgb: np.ndarray | None = None,
        output_path: str | None = None,
        temp_dir: str | None = None,
        percent: int = 0,
        status: str,
        cancelled: bool = False,
    ) -> ImageJobResult:
        return ImageJobResult(
            image_rgb=image_rgb,
            output_path=output_path,
            temp_dir=temp_dir,
            log_text="\n".join(lines),
            percent=percent,
            job_id=job_id,
            cancel_requested=cancelled,
            status=status,
        )

    with capturing_structlog() as lines:
        job_log = structlog.get_logger("watermark_remover.ui")
        job_log.info("job_start", job_id=job_id, engine=engine_name)
        if _cancel_requested(cancel_token):
            job_log.info("job_cancelled", job_id=job_id, engine=engine_name)
            return _done(lines, status="Status: cancelled", cancelled=True)
        if not is_run_enabled(mask, mask_confirmed, preview_ready):
            job_log.info(
                "run_blocked",
                job_id=job_id,
                engine=engine_name,
                error="mask not confirmed",
            )
            return _done(lines, status="Status: confirm the preview overlay before run")
        try:
            started = time.perf_counter()
            rgb = on_run(image, mask, engine_name, config)
            duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
            job_log.info(
                "inpaint_done",
                job_id=job_id,
                engine=engine_name,
                frame_idx=0,
                duration_ms=duration_ms,
            )
        except (InputValidationError, MaskError, EngineError, ResourceLimitError) as exc:
            job_log.error(
                "ui_run_failed",
                job_id=job_id,
                engine=engine_name,
                error=str(exc),
                exc_info=True,
            )
            return _done(lines, status=f"Status: {exc}")
        if _cancel_requested(cancel_token):
            job_log.info("job_cancelled", job_id=job_id, engine=engine_name)
            return _done(lines, status="Status: cancelled", cancelled=True)
        dest_dir = Path(tempfile.mkdtemp(prefix="watermark-remover-out-"))
        out_name = f"{stem}_inpainted.png"
        out_path = dest_dir / out_name
        try:
            if input_path is not None:
                refuse_overwrite_unless_flag(Path(input_path), out_path, overwrite=False)
            write_image_atomic(out_path, rgb_to_bgr(rgb))
        except InputValidationError as exc:
            cleanup_temp_dir(str(dest_dir))
            job_log.error(
                "ui_run_failed",
                job_id=job_id,
                engine=engine_name,
                error=str(exc),
                exc_info=True,
            )
            return _done(lines, status=f"Status: {exc}")
        return _done(
            lines,
            image_rgb=rgb,
            output_path=str(out_path),
            temp_dir=str(dest_dir),
            percent=100,
            status=f"Status: done ({out_name})",
        )


def format_byte_limit(max_input_bytes: int) -> str:
    gib = 1024**3
    mib = 1024**2
    if max_input_bytes >= gib and max_input_bytes % gib == 0:
        return f"{max_input_bytes // gib} GiB"
    if max_input_bytes >= mib:
        return f"{max_input_bytes / mib:.1f} MiB"
    return f"{max_input_bytes} bytes"


def _as_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(image), dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3 or arr.shape[2] not in {3, 4}:
        raise InputValidationError("image must be RGB uint8 with shape (H, W, 3)")
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr


def _layer_to_alpha(layer: np.ndarray) -> np.ndarray:
    arr = np.asarray(layer)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return arr[:, :, 3]
    if arr.ndim == 3:
        return np.max(arr[:, :, :3], axis=2)
    if arr.ndim == 2:
        return arr
    raise MaskError("unsupported mask layer shape")


def _mask_from_editor_dict(payload: dict[str, Any]) -> np.ndarray:
    nested = payload.get("mask")
    if nested is not None and not isinstance(nested, dict):
        return ui_mask_to_uint8(nested)
    layers = payload.get("layers") or []
    stacked: np.ndarray | None = None
    for layer in layers:
        if layer is None:
            continue
        alpha = _layer_to_alpha(np.asarray(layer))
        stacked = alpha if stacked is None else np.maximum(stacked, alpha)
    if stacked is not None:
        return validate_mask_array(stacked)
    composite = payload.get("composite")
    background = payload.get("background")
    if composite is not None and background is not None:
        comp = _as_rgb(np.asarray(composite))
        bg = _as_rgb(np.asarray(background))
        if comp.shape[:2] == bg.shape[:2]:
            diff = np.max(
                np.abs(comp.astype(np.int16) - bg.astype(np.int16)),
                axis=2,
            )
            return validate_mask_array(diff)
    if composite is not None:
        return validate_mask_array(np.asarray(composite))
    raise MaskError("no mask")


def _as_path(value: object) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value:
        return Path(value)
    if isinstance(value, dict):
        raw = value.get("path") or value.get("name")
        if isinstance(raw, str) and raw:
            return Path(raw)
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return Path(name)
    return None


def _mask_to_editor_layer(mask: np.ndarray) -> np.ndarray:
    binary = ui_mask_to_uint8(mask)
    layer = np.zeros((*binary.shape, 4), dtype=np.uint8)
    layer[binary > 0] = _MASK_LAYER_RGBA
    return layer


def _editor_value(rgb: np.ndarray, mask: np.ndarray | None) -> dict[str, Any]:
    layers = [] if mask is None else [_mask_to_editor_layer(mask)]
    return {"background": rgb, "layers": layers, "composite": rgb}


def _engine_name(raw: str | None) -> EngineName:
    if raw in {"opencv", "lama", "auto"}:
        return raw
    return "opencv"


def build_app(settings: Settings | None = None) -> Any:
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is required for the UI. Install with: pip install 'watermark-remover[ui]'"
        ) from exc

    settings = settings or get_settings()
    max_copy = format_byte_limit(settings.max_input_bytes)
    gpu_status = lama_cpu_warning_message()

    def _run_update(enabled: bool) -> Any:
        return gr.update(interactive=enabled)

    def on_open_file(
        file_value: object,
    ) -> tuple[Any, ...]:
        empty_editor = gr.update()
        disabled = _run_update(False)
        try:
            path = _as_path(file_value)
            if path is None:
                raise InputValidationError("no input image")
            validated = validate_input_path(path)
            validate_size_limits(validated, settings.max_input_bytes)
            validate_resolution_limits(validated, settings)
            bgr = read_image(validated)
            rgb = bgr_to_rgb(bgr)
        except (InputValidationError, ResourceLimitError, OSError) as exc:
            return (
                None,
                empty_editor,
                None,
                False,
                False,
                None,
                None,
                disabled,
                f"Status: {exc}",
                None,
            )
        editor = _editor_value(rgb, None)
        return (
            rgb,
            editor,
            None,
            False,
            False,
            validated.stem,
            str(validated),
            disabled,
            "Status: input loaded — draw or import a mask",
            rgb,
        )

    def on_editor_change(
        editor: dict[str, Any] | None,
        image_rgb: np.ndarray | None,
    ) -> tuple[Any, ...]:
        disabled = _run_update(False)
        if image_rgb is None:
            return None, False, False, None, disabled, "Status: waiting for input"
        try:
            mask = ui_mask_to_uint8(editor)
        except MaskError:
            return None, False, False, None, disabled, "Status: draw a mask on the editor"
        preview = overlay_mask_rgb(image_rgb, mask)
        return (
            mask,
            False,
            True,
            preview,
            disabled,
            "Status: preview ready — confirm the overlay before run",
        )

    def on_add_bbox(
        image_rgb: np.ndarray | None,
        mask: np.ndarray | None,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> tuple[Any, ...]:
        disabled = _run_update(False)
        if image_rgb is None:
            return None, gr.update(), None, False, False, disabled, "Status: waiting for input"
        frame_hw = (int(image_rgb.shape[0]), int(image_rgb.shape[1]))
        combined = union_bbox_mask(mask, frame_hw, int(x), int(y), int(width), int(height))
        preview = overlay_mask_rgb(image_rgb, combined)
        editor = _editor_value(_as_rgb(image_rgb), combined)
        return (
            combined,
            editor,
            preview,
            False,
            True,
            disabled,
            "Status: bbox added — confirm the overlay before run",
        )

    def on_import_mask(
        file_value: object,
        image_rgb: np.ndarray | None,
    ) -> tuple[Any, ...]:
        disabled = _run_update(False)
        if image_rgb is None:
            return None, gr.update(), None, False, False, disabled, "Status: waiting for input"
        try:
            path = _as_path(file_value)
            if path is None:
                raise MaskError("no mask file")
            frame_hw = (int(image_rgb.shape[0]), int(image_rgb.shape[1]))
            mask = import_mask_from_path(path, frame_hw)
            preview = overlay_mask_rgb(image_rgb, mask)
            editor = _editor_value(_as_rgb(image_rgb), mask)
        except (MaskError, OSError) as exc:
            return None, gr.update(), None, False, False, disabled, f"Status: {exc}"
        return (
            mask,
            editor,
            preview,
            False,
            True,
            disabled,
            "Status: mask imported — confirm the overlay before run",
        )

    def on_export_mask(
        mask: np.ndarray | None,
        stem: str | None,
    ) -> tuple[str | None, str | None, str]:
        if mask is None:
            return None, None, "Status: nothing to export"
        dest = Path(tempfile.mkdtemp(prefix="watermark-remover-mask-"))
        png_path, json_path = export_session_masks(mask, stem or "image", dest)
        return (
            str(png_path),
            str(json_path),
            f"Status: exported {png_path.name} and {json_path.name}",
        )

    def on_confirm_mask(
        mask: np.ndarray | None,
        preview_ready: bool,
    ) -> tuple[bool, Any, str]:
        enabled = is_run_enabled(mask, True, preview_ready)
        if not enabled:
            return False, _run_update(False), "Status: confirm requires a non-empty mask overlay"
        return True, _run_update(True), "Status: mask confirmed — Process All is enabled"

    def on_cancel(
        job_temp: str | None,
        cancel_token: dict[str, Any] | None,
        job_id: str | None,
    ) -> tuple[dict[str, bool], str, int, str]:
        token = dict(cancel_token) if isinstance(cancel_token, dict) else {}
        token["requested"] = True
        cleanup_temp_dir(job_temp)
        with capturing_structlog() as lines:
            job_log = structlog.get_logger("watermark_remover.ui")
            kwargs: dict[str, Any] = {}
            if job_id:
                kwargs["job_id"] = job_id
            job_log.info("job_cancelled", **kwargs)
        return token, "\n".join(lines), 0, "Status: cancelled"

    def on_process(
        image_rgb: np.ndarray | None,
        mask: np.ndarray | None,
        mask_confirmed: bool,
        preview_ready: bool,
        engine: str,
        radius: float,
        method: str,
        stem: str | None,
        input_path: str | None,
        job_temp: str | None,
        cancel_token: dict[str, Any] | bool | None,
    ) -> tuple[Any, ...]:
        cleanup_temp_dir(job_temp)
        token = cancel_token if isinstance(cancel_token, dict) else {"requested": False}
        cfg = settings.model_copy(
            update={
                "opencv_radius": max(int(radius), 1),
                "opencv_method": "ns" if method == "ns" else "telea",
            }
        )
        job = run_image_job(
            image_rgb,
            mask,
            _engine_name(engine),
            cfg,
            stem=stem or "image",
            input_path=input_path,
            mask_confirmed=mask_confirmed,
            preview_ready=preview_ready,
            cancel_token=token,
        )
        return (
            job.image_rgb,
            job.output_path,
            job.temp_dir,
            job.log_text,
            job.percent,
            job.job_id,
            {"requested": False},
            job.status,
        )

    with gr.Blocks(
        title="watermark-remover",
        analytics_enabled=False,
    ) as demo:
        image_state = gr.State(None)
        mask_state = gr.State(None)
        mask_confirmed = gr.State(False)
        preview_ready = gr.State(False)
        input_stem = gr.State("image")
        input_path_state = gr.State(None)
        sensitivity = gr.State(50)
        job_temp_state = gr.State(None)
        cancel_state = gr.State({"requested": False})

        gr.Markdown("# watermark-remover")
        gr.Markdown("Image mode — local inpainting. No cloud calls.")

        with gr.Group():
            gr.Markdown("## Input")
            gr.Markdown(
                f"Supports JPG, PNG, and WEBP. Maximum file size is {max_copy}."
            )
            input_file = gr.File(
                label="Open File",
                file_types=[".jpg", ".jpeg", ".png", ".webp"],
                file_count="single",
                type="filepath",
            )
            input_preview = gr.Image(
                label="Loaded image",
                type="numpy",
                interactive=False,
                buttons=["fullscreen"],
            )

        with gr.Group():
            gr.Markdown("## Mask")
            gr.Markdown(
                "Draw freehand or add a bounding box. "
                "Import/export uses `{stem}.mask.png` and `{stem}.mask.json`."
            )
            mask_editor = gr.ImageEditor(
                label="Mask editor (freehand)",
                type="numpy",
                image_mode="RGB",
                sources=(),
                transforms=(),
                layers=True,
                brush=gr.Brush(
                    default_size=20,
                    colors=["#ef4444"],
                    color_mode="fixed",
                    default_color="#ef4444",
                ),
                buttons=["fullscreen"],
            )
            with gr.Row():
                bbox_x = gr.Number(label="BBox x", value=0, precision=0)
                bbox_y = gr.Number(label="BBox y", value=0, precision=0)
                bbox_w = gr.Number(label="BBox width", value=32, precision=0)
                bbox_h = gr.Number(label="BBox height", value=32, precision=0)
            add_bbox_btn = gr.Button("Add bounding box")
            with gr.Row():
                mask_import = gr.File(
                    label="Import mask (.png / .json)",
                    file_types=[".png", ".json"],
                    file_count="single",
                    type="filepath",
                )
                export_png = gr.File(label="Export {stem}.mask.png")
                export_json = gr.File(label="Export {stem}.mask.json")
            export_btn = gr.Button("Export mask PNG + JSON")

        with gr.Group():
            gr.Markdown("## Preview")
            gr.Markdown("Overlay is required. Confirm it before Process All.")
            preview_image = gr.Image(
                label="Mask overlay",
                type="numpy",
                interactive=False,
                buttons=["fullscreen"],
            )
            confirm_btn = gr.Button("Confirm mask")

        with gr.Group():
            gr.Markdown("## Engine")
            engine = gr.Dropdown(
                label="Engine",
                choices=["opencv", "lama", "auto"],
                value="opencv",
            )
            gpu_md = gr.Markdown(gpu_status)
            with gr.Accordion("Advanced Settings", open=False):
                radius = gr.Number(label="radius", value=settings.opencv_radius, precision=0)
                method = gr.Radio(
                    label="method",
                    choices=["telea", "ns"],
                    value=settings.opencv_method,
                )
                gr.Markdown(f"Use GPU (CUDA): {'Active' if cuda_available() else 'Unavailable'}")

        with gr.Group():
            gr.Markdown("## Run")
            with gr.Row():
                run_btn = gr.Button("Process All", interactive=False, variant="primary")
                cancel_btn = gr.Button("Cancel")
            job_id_box = gr.Textbox(label="job_id", interactive=False)
            percent = gr.Slider(
                label="Progress %",
                minimum=0,
                maximum=100,
                value=0,
                interactive=False,
            )
            log_box = gr.Textbox(label="Log", lines=8, interactive=False)
            result_image = gr.Image(
                label="Result",
                type="numpy",
                interactive=False,
                buttons=["fullscreen"],
            )
            download = gr.File(label="Download result")
            status = gr.Markdown("Status: Waiting for input")

        _ = (sensitivity, gpu_md)

        input_file.upload(
            on_open_file,
            inputs=[input_file],
            outputs=[
                image_state,
                mask_editor,
                mask_state,
                mask_confirmed,
                preview_ready,
                input_stem,
                input_path_state,
                run_btn,
                status,
                input_preview,
            ],
        )
        mask_editor.change(
            on_editor_change,
            inputs=[mask_editor, image_state],
            outputs=[
                mask_state,
                mask_confirmed,
                preview_ready,
                preview_image,
                run_btn,
                status,
            ],
        )
        add_bbox_btn.click(
            on_add_bbox,
            inputs=[image_state, mask_state, bbox_x, bbox_y, bbox_w, bbox_h],
            outputs=[
                mask_state,
                mask_editor,
                preview_image,
                mask_confirmed,
                preview_ready,
                run_btn,
                status,
            ],
        )
        mask_import.upload(
            on_import_mask,
            inputs=[mask_import, image_state],
            outputs=[
                mask_state,
                mask_editor,
                preview_image,
                mask_confirmed,
                preview_ready,
                run_btn,
                status,
            ],
        )
        export_btn.click(
            on_export_mask,
            inputs=[mask_state, input_stem],
            outputs=[export_png, export_json, status],
        )
        confirm_btn.click(
            on_confirm_mask,
            inputs=[mask_state, preview_ready],
            outputs=[mask_confirmed, run_btn, status],
        )
        run_event = run_btn.click(
            on_process,
            inputs=[
                image_state,
                mask_state,
                mask_confirmed,
                preview_ready,
                engine,
                radius,
                method,
                input_stem,
                input_path_state,
                job_temp_state,
                cancel_state,
            ],
            outputs=[
                result_image,
                download,
                job_temp_state,
                log_box,
                percent,
                job_id_box,
                cancel_state,
                status,
            ],
        )
        cancel_btn.click(
            on_cancel,
            inputs=[job_temp_state, cancel_state, job_id_box],
            outputs=[cancel_state, log_box, percent, status],
            cancels=[run_event],
        )

    return demo


def launch(*, settings: Settings | None = None) -> None:
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is required for the UI. Install with: pip install 'watermark-remover[ui]'"
        ) from exc

    settings = settings or get_settings()
    demo = build_app(settings)
    kwargs = launch_kwargs(settings)
    kwargs["theme"] = gr.themes.Soft(primary_hue="blue")
    demo.launch(**kwargs)


if __name__ == "__main__":
    launch()
