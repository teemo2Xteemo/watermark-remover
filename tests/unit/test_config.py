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


def test_settings_loads_toml_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.toml"
    config.write_text("crf = 18\nmax_input_bytes = 4096\n", encoding="utf-8")
    monkeypatch.setenv("WATERMARK_REMOVER_CONFIG", str(config))
    monkeypatch.chdir(tmp_path)
    clear_settings_cache()
    settings = Settings(_env_file=None)
    assert settings.crf == 18
    assert settings.max_input_bytes == 4096
