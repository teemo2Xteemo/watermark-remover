from __future__ import annotations

import contextlib
import os
import shutil
import stat
import tempfile
import time
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import structlog

from watermark_remover.config import Settings, get_settings
from watermark_remover.engines.base import InpaintEngine
from watermark_remover.exceptions import EngineError, ProcessingCancelled, ResourceLimitError
from watermark_remover.io.validate import estimate_working_set_mb
from watermark_remover.io.video import VideoMetadata, probe_video
from watermark_remover.masks.base import MaskProvider
from watermark_remover.video.encode import encode_video
from watermark_remover.video.extract import extract_frames
from watermark_remover.video.temporal import TemporalSmoother

_PNG_PARAMS = [int(cv2.IMWRITE_PNG_COMPRESSION), 1]
_FRAME_NAME = "frame_{idx:08d}.png"
_QUALITY_MAX_HEIGHT = {"1080p": 1080, "720p": 720}


def capped_max_workers(configured: int) -> int:
    """Never exceed `os.cpu_count()`, even if config asks for more."""
    cap = os.cpu_count() or 1
    return max(1, min(int(configured), cap))


def target_frame_size(width: int, height: int, settings: Settings) -> tuple[int, int]:
    """Return even WxH after output_quality cap. Never upscale."""
    w, h = int(width), int(height)
    cap_h = _QUALITY_MAX_HEIGHT.get(str(settings.output_quality))
    if cap_h is not None and h > cap_h:
        scale = cap_h / h
        w = int(round(w * scale))
        h = int(cap_h)
    return _even_dim(w), _even_dim(h)


class VideoProcessor:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings if settings is not None else get_settings()

    def process(
        self,
        input_path: Path,
        mask_provider: MaskProvider,
        engine: InpaintEngine,
        output_path: Path,
        progress: Callable[..., None] | None = None,
        cancel_token: dict[str, Any] | bool | None = None,
    ) -> Path:
        src = Path(input_path)
        dest = Path(output_path)
        meta = probe_video(src)
        workers = capped_max_workers(self._settings.max_workers)
        target_w, target_h = target_frame_size(meta.width, meta.height, self._settings)
        self._reject_if_over_ram(target_w, target_h, workers)

        log = structlog.get_logger("watermark_remover")
        if self._settings.temporal_smoothing:
            mode = "sequential (temporal smoothing enabled)"
        else:
            mode = f"parallel ({workers} workers, temporal smoothing disabled)"
        log.info(
            "video_process_start",
            input_path=src.name,
            fps=meta.fps,
            width=target_w,
            height=target_h,
            max_workers=workers,
            mode=mode,
            engine=type(engine).__name__,
        )
        _ensure_not_cancelled(cancel_token)

        tmp_out = dest.with_name(f"{dest.stem}.tmp{dest.suffix}")
        try:
            with _frames_temp_dir() as frames_dir:
                self._process_frames(
                    src=src,
                    meta=meta,
                    mask_provider=mask_provider,
                    engine=engine,
                    frames_dir=frames_dir,
                    workers=workers,
                    progress=progress,
                    cancel_token=cancel_token,
                    target_size=(target_w, target_h),
                )
                _ensure_not_cancelled(cancel_token)
                stride = max(1, int(self._settings.frame_stride))
                encode_fps = meta.fps / float(stride) if stride > 1 else meta.fps
                encode_video(
                    frames_dir,
                    src,
                    tmp_out,
                    fps=encode_fps,
                    crf=int(self._settings.crf),
                    keep_audio=bool(self._settings.keep_audio),
                )
                _fsync_file(tmp_out)
            _ensure_not_cancelled(cancel_token)
            os.replace(tmp_out, dest)
            log.info("video_process_end", output_path=dest.name)
            return dest
        finally:
            if tmp_out.exists():
                tmp_out.unlink(missing_ok=True)

    def _reject_if_over_ram(self, width: int, height: int, workers: int) -> None:
        if self._settings.max_ram_mb is None:
            return
        estimate_mb = estimate_working_set_mb(width, height, workers)
        if estimate_mb > self._settings.max_ram_mb:
            raise ResourceLimitError(
                f"estimated working set {estimate_mb:.1f} MiB exceeds "
                f"max_ram_mb={self._settings.max_ram_mb}"
            )

    def _process_frames(
        self,
        *,
        src: Path,
        meta: VideoMetadata,
        mask_provider: MaskProvider,
        engine: InpaintEngine,
        frames_dir: Path,
        workers: int,
        progress: Callable[..., None] | None,
        cancel_token: dict[str, Any] | bool | None,
        target_size: tuple[int, int],
    ) -> None:
        stride = max(1, int(self._settings.frame_stride))
        native_total = meta.frame_count if meta.frame_count > 0 else None
        if native_total is None:
            total = None
        else:
            total = (native_total + stride - 1) // stride
        started = time.perf_counter()
        completed = 0
        encode_idx = 0
        log = structlog.get_logger("watermark_remover")

        def _on_done(frame_idx: int) -> None:
            nonlocal completed
            completed += 1
            elapsed = time.perf_counter() - started
            throughput = completed / elapsed if elapsed > 0 else 0.0
            percent = (completed / total) * 100.0 if total else None
            if progress is not None:
                progress(
                    percent=percent,
                    frame_idx=frame_idx,
                    fps_throughput=throughput,
                )
            log.debug(
                "video_frame_done",
                frame_idx=frame_idx,
                percent=percent,
                fps_throughput=round(throughput, 3),
            )

        def _prepare(frame_idx: int, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            _ensure_not_cancelled(cancel_token)
            mask = mask_provider.get_mask(frame, frame_idx)
            tw, th = target_size
            if int(frame.shape[1]) != tw or int(frame.shape[0]) != th:
                frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
                mask = np.where(mask > 127, np.uint8(255), np.uint8(0))
            return frame, mask

        def _write_result(result: np.ndarray) -> None:
            nonlocal encode_idx
            _write_png(frames_dir / _FRAME_NAME.format(idx=encode_idx), result)
            encode_idx += 1

        if self._settings.temporal_smoothing:
            smoother = TemporalSmoother(self._settings)
            prev_result: np.ndarray | None = None
            for frame_idx, frame in extract_frames(src):
                _ensure_not_cancelled(cancel_token)
                if frame_idx % stride != 0:
                    continue
                prepared, mask = _prepare(frame_idx, frame)
                result = engine.process(prepared, mask)
                if prev_result is not None:
                    result = smoother.apply(prev_result, result, mask)
                prev_result = result
                _write_result(result)
                _on_done(frame_idx)
            if completed == 0:
                raise EngineError(f"no frames decoded from {src.name}")
            return

        def _inpaint_frame(frame_idx: int, frame: np.ndarray) -> tuple[int, np.ndarray]:
            _ensure_not_cancelled(cancel_token)
            prepared, mask = _prepare(frame_idx, frame)
            _ensure_not_cancelled(cancel_token)
            result = engine.process(prepared, mask)
            return frame_idx, result

        pending: dict[int, np.ndarray] = {}
        next_encode = 0
        in_flight: dict[Future[tuple[int, np.ndarray]], int] = {}

        def _flush_ready() -> None:
            nonlocal next_encode
            while next_encode in pending:
                _write_png(
                    frames_dir / _FRAME_NAME.format(idx=next_encode),
                    pending.pop(next_encode),
                )
                next_encode += 1

        def _take_completed() -> None:
            if not in_flight:
                return
            done, _ = wait(list(in_flight.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                frame_idx = in_flight.pop(future)
                try:
                    done_idx, result = future.result()
                except ProcessingCancelled:
                    raise
                except Exception as exc:
                    _log_frame_failed(frame_idx, exc)
                    raise
                pending[done_idx // stride] = result
                _flush_ready()
                _on_done(done_idx)

        # Threads, not processes: InpaintEngine (esp. ONNX) is not picklable on Windows spawn.
        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            for frame_idx, frame in extract_frames(src):
                _ensure_not_cancelled(cancel_token)
                if frame_idx % stride != 0:
                    continue
                while len(in_flight) + len(pending) >= workers:
                    _ensure_not_cancelled(cancel_token)
                    _take_completed()
                _ensure_not_cancelled(cancel_token)
                future = pool.submit(_inpaint_frame, frame_idx, frame)
                in_flight[future] = frame_idx
            while in_flight:
                _ensure_not_cancelled(cancel_token)
                _take_completed()
            _flush_ready()
        except Exception:
            for future in in_flight:
                future.cancel()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

        if completed == 0:
            raise EngineError(f"no frames decoded from {src.name}")


def _even_dim(value: int) -> int:
    n = max(2, int(value))
    return n if n % 2 == 0 else n - 1


def _cancel_requested(token: dict[str, Any] | bool | None) -> bool:
    if isinstance(token, dict):
        return bool(token.get("requested"))
    return bool(token)


def _ensure_not_cancelled(token: dict[str, Any] | bool | None) -> None:
    if _cancel_requested(token):
        raise ProcessingCancelled("job cancelled")


def _log_frame_failed(frame_idx: int, exc: BaseException) -> None:
    log = structlog.get_logger("watermark_remover")
    try:
        log.error("frame_failed", frame_idx=frame_idx, error=str(exc), exc_info=True)
    except Exception:
        try:
            log.error("frame_failed", frame_idx=frame_idx, error=str(exc), exc_info=False)
        except Exception:
            pass


def _fsync_file(path: Path) -> None:
    with Path(path).open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())


@contextlib.contextmanager
def _frames_temp_dir() -> Iterator[Path]:
    path = Path(tempfile.mkdtemp(prefix="watermark_remover_frames_"))
    try:
        yield path
    finally:
        _remove_tree(path)


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    delays = (0.0, 0.05, 0.15, 0.4)
    last_error: OSError | None = None
    for delay in delays:
        if delay:
            time.sleep(delay)
        try:
            shutil.rmtree(path)
            if not path.exists():
                return
        except OSError as exc:
            last_error = exc
            _chmod_writable(path)
    if path.exists():
        structlog.get_logger("watermark_remover").warning(
            "temp_cleanup_failed",
            path=path.name,
            error=str(last_error) if last_error else "directory still present",
        )


def _chmod_writable(path: Path) -> None:
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
    except OSError:
        pass
    if not path.is_dir():
        return
    for child in path.rglob("*"):
        try:
            os.chmod(child, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
        except OSError:
            continue


def _write_png(path: Path, image: np.ndarray) -> None:
    ok, buffer = cv2.imencode(".png", image, _PNG_PARAMS)
    if not ok:
        raise EngineError(f"failed to encode frame {path.name}")
    path.write_bytes(buffer.tobytes())
