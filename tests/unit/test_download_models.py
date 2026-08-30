from __future__ import annotations

import hashlib
import io
from pathlib import Path
from urllib.request import Request

import download_models as download_mod
import pytest

from watermark_remover.config import Settings, clear_settings_cache


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._buffer = io.BytesIO(payload)

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _install_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    dest = model_dir / "lama.onnx"
    settings = Settings(model_dir=model_dir, lama_weights=dest)
    monkeypatch.setattr(download_mod, "get_settings", lambda: settings)
    clear_settings_cache()
    return dest


def test_download_models_verifies_sha256(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dest = _install_settings(tmp_path, monkeypatch)
    payload = b"fake-lama-onnx-bytes"
    monkeypatch.setattr(download_mod, "LAMA_ONNX_SHA256", _sha(payload))
    monkeypatch.setattr(download_mod, "LAMA_ONNX_BYTES", len(payload))

    def fake_urlopen(request: Request | str, timeout: float | None = None) -> _FakeResponse:
        url = request.full_url if isinstance(request, Request) else request
        assert "huggingface.co" in url
        assert timeout is not None
        return _FakeResponse(payload)

    monkeypatch.setattr(download_mod.urllib.request, "urlopen", fake_urlopen)
    download_mod.download_models(force=False)
    assert dest.is_file()
    assert dest.read_bytes() == payload
    assert not dest.with_name(dest.name + ".tmp").exists()


def test_download_models_refuses_overwrite_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = _install_settings(tmp_path, monkeypatch)
    dest.write_bytes(b"existing")

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network must not be used when refusing overwrite")

    monkeypatch.setattr(download_mod.urllib.request, "urlopen", boom)
    with pytest.raises(FileExistsError, match="--force"):
        download_mod.download_models(force=False)
    assert dest.read_bytes() == b"existing"


def test_download_models_force_overwrites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dest = _install_settings(tmp_path, monkeypatch)
    dest.write_bytes(b"stale")
    payload = b"replacement-weights"
    monkeypatch.setattr(download_mod, "LAMA_ONNX_SHA256", _sha(payload))
    monkeypatch.setattr(download_mod, "LAMA_ONNX_BYTES", len(payload))
    monkeypatch.setattr(
        download_mod.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeResponse(payload),
    )
    download_mod.download_models(force=True)
    assert dest.read_bytes() == payload


def test_download_models_sha256_mismatch_does_not_keep_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = _install_settings(tmp_path, monkeypatch)
    payload = b"tampered"
    monkeypatch.setattr(download_mod, "LAMA_ONNX_SHA256", "ab" * 32)
    monkeypatch.setattr(download_mod, "LAMA_ONNX_BYTES", len(payload))
    monkeypatch.setattr(
        download_mod.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeResponse(payload),
    )
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        download_mod.download_models(force=False)
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".tmp").exists()


def test_download_models_incomplete_size_does_not_keep_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = _install_settings(tmp_path, monkeypatch)
    payload = b"short"
    monkeypatch.setattr(download_mod, "LAMA_ONNX_SHA256", _sha(payload))
    monkeypatch.setattr(download_mod, "LAMA_ONNX_BYTES", 999)
    monkeypatch.setattr(download_mod, "_MAX_ATTEMPTS", 1)
    monkeypatch.setattr(
        download_mod.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeResponse(payload),
    )
    with pytest.raises(RuntimeError, match="incomplete download"):
        download_mod.download_models(force=False)
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".tmp").exists()
