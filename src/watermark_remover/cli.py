from __future__ import annotations

import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import structlog
import typer

from watermark_remover.config import Settings, get_settings
from watermark_remover.engines.registry import get_engine
from watermark_remover.exceptions import (
    EngineError,
    InputValidationError,
    MaskError,
    ResourceLimitError,
)
from watermark_remover.image_processor import ImageProcessor
from watermark_remover.io.image import read_image, write_image_atomic
from watermark_remover.io.validate import (
    default_output_path,
    is_video_path,
    refuse_overwrite_unless_flag,
    validate_input_path,
    validate_resolution_limits,
    validate_size_limits,
)
from watermark_remover.io.video import probe_video
from watermark_remover.masks.base import validate_mask_coverage
from watermark_remover.masks.manual import ManualMaskProvider
from watermark_remover.masks.serialize import (
    export_mask_json,
    export_mask_png,
    load_mask_json,
    load_mask_png,
    mask_to_polygon_payload,
)
from watermark_remover.video.processor import VideoProcessor

EngineName = Literal["opencv", "lama", "auto"]

app = typer.Typer(
    name="watermark-remover",
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


def _configure_logging(job_id: str, level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    shared = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer: structlog.types.Processor
    if sys.stderr.isatty():
        renderer = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()
    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(job_id=job_id)


def _load_mask(mask_path: Path, frame_hw: tuple[int, int]) -> np.ndarray:
    suffix = mask_path.suffix.lower()
    if suffix == ".json":
        return load_mask_json(mask_path, frame_hw)
    if suffix == ".png":
        return load_mask_png(mask_path)
    raise MaskError(f"mask must be .png or .json, got '{mask_path.suffix}'")


def _run(
    input_path: Path,
    mask_path: Path,
    engine: EngineName,
    output: Path | None,
    overwrite: bool,
    allow_empty_mask: bool,
    allow_full_mask: bool,
    export_mask: bool,
    settings: Settings,
) -> Path:
    job_id = str(uuid.uuid4())
    _configure_logging(job_id, settings.log_level)
    log = structlog.get_logger("watermark_remover")
    log.info("job_start", engine=engine, input_path=input_path.name)

    if str(mask_path) == "auto" or Path(mask_path).name == "auto":
        raise MaskError(
            "auto-detect never auto-applies; pass a mask file or accept a candidate in the UI"
        )

    validated_input = validate_input_path(input_path)
    validate_size_limits(validated_input, settings.max_input_bytes)
    validate_resolution_limits(validated_input, settings)

    output_path = output if output is not None else default_output_path(validated_input)
    refuse_overwrite_unless_flag(validated_input, output_path, overwrite)

    if is_video_path(validated_input):
        result_path = _run_video(
            validated_input=validated_input,
            mask_path=mask_path,
            engine=engine,
            output_path=output_path,
            allow_empty_mask=allow_empty_mask,
            allow_full_mask=allow_full_mask,
            export_mask=export_mask,
            settings=settings,
            log=log,
        )
        log.info("job_end", engine=engine, output_path=result_path.name)
        return result_path

    image = read_image(validated_input)
    mask = _load_mask(mask_path, (int(image.shape[0]), int(image.shape[1])))

    started = time.perf_counter()
    result = ImageProcessor().process(
        image,
        mask,
        engine,
        settings,
        allow_empty_mask=allow_empty_mask,
        allow_full_mask=allow_full_mask,
    )
    duration_ms = (time.perf_counter() - started) * 1000.0
    log.info(
        "inpaint_done",
        engine=engine,
        frame_idx=0,
        duration_ms=round(duration_ms, 2),
    )

    write_image_atomic(output_path, result)
    if export_mask:
        _export_mask_sidecars(validated_input, output_path, mask, log)

    log.info("job_end", engine=engine, output_path=output_path.name)
    return output_path


def _run_video(
    *,
    validated_input: Path,
    mask_path: Path,
    engine: EngineName,
    output_path: Path,
    allow_empty_mask: bool,
    allow_full_mask: bool,
    export_mask: bool,
    settings: Settings,
    log: Any,
) -> Path:
    meta = probe_video(validated_input)
    mask = _load_mask(mask_path, (meta.height, meta.width))
    validate_mask_coverage(
        mask,
        allow_empty_mask=allow_empty_mask,
        allow_full_mask=allow_full_mask,
    )
    provider = ManualMaskProvider(mask)
    inpaint_engine = get_engine(engine, mask, settings)

    def progress(**kwargs: object) -> None:
        log.debug("video_progress", **kwargs)

    started = time.perf_counter()
    result_path = VideoProcessor(settings).process(
        validated_input,
        provider,
        inpaint_engine,
        output_path,
        progress=progress,
    )
    duration_ms = (time.perf_counter() - started) * 1000.0
    log.info(
        "inpaint_done",
        engine=engine,
        frame_idx=None,
        duration_ms=round(duration_ms, 2),
    )
    if export_mask:
        _export_mask_sidecars(validated_input, output_path, mask, log)
    return result_path


def _export_mask_sidecars(
    validated_input: Path,
    output_path: Path,
    mask: np.ndarray,
    log: Any,
) -> None:
    stem = validated_input.stem
    export_dir = output_path.parent
    export_mask_png(export_dir / f"{stem}.mask.png", mask)
    export_mask_json(export_dir / f"{stem}.mask.json", mask_to_polygon_payload(mask))
    log.info("mask_exported", output_path=f"{stem}.mask.png")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    input_path: Annotated[
        Path | None,
        typer.Option("--input", help="Input image (JPG, PNG, WEBP) or video (MP4, MOV, WEBM)"),
    ] = None,
    mask: Annotated[
        Path | None,
        typer.Option("--mask", help="Mask PNG or schema_version 1 JSON"),
    ] = None,
    engine: Annotated[
        str,
        typer.Option("--engine", help="Inpaint engine: opencv, lama, or auto"),
    ] = "opencv",
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output path (default: {stem}_inpainted{suffix})"),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Allow writing onto the input path"),
    ] = False,
    allow_empty_mask: Annotated[
        bool,
        typer.Option("--allow-empty-mask", help="Permit an all-zero mask"),
    ] = False,
    allow_full_mask: Annotated[
        bool,
        typer.Option("--allow-full-mask", help="Permit an all-255 mask"),
    ] = False,
    export_mask: Annotated[
        bool,
        typer.Option("--export-mask", help="Write {stem}.mask.png and {stem}.mask.json"),
    ] = False,
) -> None:
    """Local watermark / object-removal inpainting."""
    if ctx.invoked_subcommand is not None:
        return
    try:
        if input_path is None or mask is None:
            raise InputValidationError("missing required options --input and --mask")
        if engine not in {"opencv", "lama", "auto"}:
            raise InputValidationError(
                f"unknown --engine {engine!r}; expected opencv, lama, or auto"
            )
        _run(
            input_path=input_path,
            mask_path=mask,
            engine=engine,  # type: ignore[arg-type]
            output=output,
            overwrite=overwrite,
            allow_empty_mask=allow_empty_mask,
            allow_full_mask=allow_full_mask,
            export_mask=export_mask,
            settings=get_settings(),
        )
    except KeyboardInterrupt as exc:
        raise typer.Exit(130) from exc
    except InputValidationError as exc:
        structlog.get_logger("watermark_remover").error(
            "validation_failed",
            error=str(exc),
            input_path=None if input_path is None else input_path.name,
            exc_info=True,
        )
        raise typer.Exit(1) from exc
    except MaskError as exc:
        structlog.get_logger("watermark_remover").error(
            "mask_failed",
            error=str(exc),
            input_path=None if input_path is None else input_path.name,
            exc_info=True,
        )
        raise typer.Exit(1) from exc
    except ResourceLimitError as exc:
        structlog.get_logger("watermark_remover").error(
            "resource_limit",
            error=str(exc),
            input_path=None if input_path is None else input_path.name,
            exc_info=True,
        )
        raise typer.Exit(3) from exc
    except EngineError as exc:
        structlog.get_logger("watermark_remover").error(
            "engine_failed",
            error=str(exc),
            engine=engine,
            exc_info=True,
        )
        raise typer.Exit(2) from exc
    except Exception as exc:
        structlog.get_logger("watermark_remover").error(
            "unhandled_error",
            error=str(exc),
            exc_info=True,
        )
        raise typer.Exit(2) from exc


@app.command("ui")
def start_ui() -> None:
    """Launch the local Gradio Image mode UI (127.0.0.1, share=False)."""
    from watermark_remover.ui.app import launch

    try:
        launch()
    except KeyboardInterrupt as exc:
        raise typer.Exit(130) from exc
