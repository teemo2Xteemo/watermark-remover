"""One-time local download of LaMa ONNX weights. Not used by the processing pipeline."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import urllib.request
from pathlib import Path

from watermark_remover.config import get_settings

# Carve/LaMa-ONNX lama_fp32.onnx (recommended export, fixed 512x512, Apache-2.0).
LAMA_ONNX_URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx?download=true"
LAMA_ONNX_SHA256 = "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6"
LAMA_ONNX_BYTES = 208044816
_CHUNK_BYTES = 1024 * 1024
_TIMEOUT_SEC = 600
_USER_AGENT = "watermark-remover-setup/0.1"
_MAX_ATTEMPTS = 3

logger = logging.getLogger("download_models")


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_CHUNK_BYTES)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _download_and_verify(url: str, dest: Path, expected_sha256: str, expected_bytes: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept-Encoding": "identity",
        },
    )
    hasher = hashlib.sha256()
    total = 0
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SEC) as response:
            with tmp.open("wb") as handle:
                while True:
                    chunk = response.read(_CHUNK_BYTES)
                    if not chunk:
                        break
                    total += len(chunk)
                    hasher.update(chunk)
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        if total != expected_bytes:
            raise RuntimeError(
                f"incomplete download for {dest.name}: got {total} bytes, expected {expected_bytes}"
            )
        digest = hasher.hexdigest()
        if digest.lower() != expected_sha256.lower():
            raise RuntimeError(
                f"SHA256 mismatch for {dest.name}: expected {expected_sha256}, got {digest}"
            )
        os.replace(tmp, dest)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def download_models(*, force: bool) -> None:
    settings = get_settings()
    if settings.lama_weights is not None:
        dest = Path(settings.lama_weights)
    else:
        dest = Path(settings.model_dir) / "lama.onnx"
    if dest.exists() and not force:
        raise FileExistsError(
            f"{dest.name} already exists; pass --force to overwrite after re-download"
        )
    logger.info("downloading LaMa weights to %s", dest.name)
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            _download_and_verify(LAMA_ONNX_URL, dest, LAMA_ONNX_SHA256, LAMA_ONNX_BYTES)
            break
        except FileExistsError:
            raise
        except Exception as exc:
            logger.warning("download attempt %s/%s failed: %s", attempt, _MAX_ATTEMPTS, exc)
            if attempt == _MAX_ATTEMPTS:
                raise
    actual = _sha256_file(dest)
    if actual.lower() != LAMA_ONNX_SHA256.lower():
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA256 mismatch after write for {dest.name}: "
            f"expected {LAMA_ONNX_SHA256}, got {actual}"
        )
    logger.info("verified SHA256 for %s", dest.name)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Download local LaMa ONNX weights into MODEL_DIR")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing weights after a verified re-download",
    )
    args = parser.parse_args(argv)
    try:
        download_models(force=args.force)
    except FileExistsError as exc:
        logger.error("%s", exc)
        return 1
    except Exception:
        logger.error("download failed", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
