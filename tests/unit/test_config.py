from __future__ import annotations

from pathlib import Path

import pytest

from watermark_remover.config import Settings, clear_settings_cache


def test_settings_ignores_empty_env_values(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "MAX_INPUT_BYTES=\nCRF=\nMAX_RAM_MB=\nLOG_LEVEL=\n",
        encoding="utf-8",
    )
    clear_settings_cache()
    settings = Settings(_env_file=env_file)
    assert settings.max_input_bytes == 2 * 1024**3
    assert settings.crf == 23
    assert settings.max_ram_mb is None
    assert settings.log_level == "INFO"


def test_settings_tile_size_from_env(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("TILE_SIZE=256\nTILE_OVERLAP=16\n", encoding="utf-8")
    clear_settings_cache()
    settings = Settings(_env_file=env_file)
    assert settings.tile_size == 256
    assert settings.tile_overlap == 16


def test_settings_loads_toml_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "config.toml"
    config.write_text("crf = 18\nmax_input_bytes = 4096\n", encoding="utf-8")
    monkeypatch.setenv("WATERMARK_REMOVER_CONFIG", str(config))
    monkeypatch.chdir(tmp_path)
    clear_settings_cache()
    settings = Settings(_env_file=None)
    assert settings.crf == 18
    assert settings.max_input_bytes == 4096


def test_max_workers_capped_at_cpu_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("watermark_remover.config._cpu_count", lambda: 2)
    settings = Settings(max_workers=128)
    assert settings.max_workers == 2


def test_video_settings_defaults() -> None:
    settings = Settings()
    assert settings.output_quality == "source"
    assert settings.keep_audio is True
    assert settings.frame_stride == 1
    assert settings.temporal_smoothing is True
    assert settings.gradio_server_name == "127.0.0.1"
    assert settings.gradio_share is False


def test_gradio_server_name_from_env(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("GRADIO_SERVER_NAME=0.0.0.0\n", encoding="utf-8")
    clear_settings_cache()
    settings = Settings(_env_file=env_file)
    assert settings.gradio_server_name == "0.0.0.0"
