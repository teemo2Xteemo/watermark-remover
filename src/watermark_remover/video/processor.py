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

import cv2
import numpy as np
import structlog

from watermark_remover.config import Settings, get_settings
from watermark_remover.engines.base import InpaintEngine
from watermark_remover.exceptions import EngineError, ResourceLimitError
from watermark_remover.io.validate import estimate_working_set_mb
from watermark_remover.io.video import VideoMetadata, probe_video
from watermark_remover.masks.base import MaskProvider
from watermark_remover.video.encode import encode_video
from watermark_remover.video.extract import extract_frames

_PNG_PARAMS = [int(cv2.IMWRITE_PNG_COMPRESSION), 1]
_FRAME_NAME = "frame_{idx:08d}.png"


def capped_max_workers(configured: int) -> int:
    """Never exceed `os.cpu_count()`, even if config asks for more."""
    cap = os.cpu_count() or 1
    return max(1, min(int(configured), cap))


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
    ) -> Path:
        src = Path(input_path)
        dest = Path(output_path)
        meta = probe_video(src)
        workers = capped_max_workers(self._settings.max_workers)
        self._reject_if_over_ram(meta, workers)

        log = structlog.get_logger("watermark_remover")
        log.info(
            "video_process_start",
            input_path=src.name,
            fps=meta.fps,
            width=meta.width,
            height=meta.height,
            max_workers=workers,
        )

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
                )
                encode_video(
                    frames_dir,
                    src,
                    tmp_out,
                    fps=meta.fps,
                    crf=int(self._settings.crf),
                )
                _fsync_file(tmp_out)
            os.replace(tmp_out, dest)
            log.info("video_process_end", output_path=dest.name)
            return dest
        finally:
            if tmp_out.exists():
                tmp_out.unlink(missing_ok=True)

    def _reject_if_over_ram(self, meta: VideoMetadata, workers: int) -> None:
        if self._settings.max_ram_mb is None:
            return
        estimate_mb = estimate_working_set_mb(meta.width, meta.height, workers)
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
    ) -> None:
        total = meta.frame_count if meta.frame_count > 0 else None
        started = time.perf_counter()
        completed = 0
        log = structlog.get_logger("watermark_remover")

        def _write_frame(frame_idx: int, frame: np.ndarray) -> int:
            mask = mask_provider.get_mask(frame, frame_idx)
            result = engine.process(frame, mask)
            _write_png(frames_dir / _FRAME_NAME.format(idx=frame_idx), result)
            return frame_idx

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

        in_flight: dict[Future[int], int] = {}
        # Threads, not processes: InpaintEngine (esp. ONNX) is not picklable on Windows spawn.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            try:
                for frame_idx, frame in extract_frames(src):
                    while len(in_flight) >= workers:
                        _collect_completed(in_flight, _on_done)
                    future = pool.submit(_write_frame, frame_idx, frame)
                    in_flight[future] = frame_idx
                while in_flight:
                    _collect_completed(in_flight, _on_done)
            except Exception:
                for future in in_flight:
                    future.cancel()
                raise

        if completed == 0:
            raise EngineError(f"no frames decoded from {src.name}")


def _collect_completed(
    in_flight: dict[Future[int], int],
    on_done: Callable[[int], None],
) -> None:
    if not in_flight:
        return
    done, _ = wait(list(in_flight.keys()), return_when=FIRST_COMPLETED)
    for future in done:
        frame_idx = in_flight.pop(future)
        try:
            future.result()
        except Exception as exc:
            _log_frame_failed(frame_idx, exc)
            raise
        on_done(frame_idx)


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
