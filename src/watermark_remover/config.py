from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from watermark_remover.exceptions import InputValidationError

_DEFAULT_MAX_INPUT_BYTES = 2 * 1024**3


def _cpu_count() -> int:
    return os.cpu_count() or 1


def resolve_config_toml_path() -> Path | None:
    """Q16: WATERMARK_REMOVER_CONFIG → ./config.toml → defaults."""
    env_path = os.environ.get("WATERMARK_REMOVER_CONFIG")
    if env_path:
        path = Path(env_path)
        if not path.is_file():
            raise InputValidationError(
                f"WATERMARK_REMOVER_CONFIG is not a file: {path.name}"
            )
        return path
    cwd_toml = Path.cwd() / "config.toml"
    if cwd_toml.is_file():
        return cwd_toml
    return None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
        protected_namespaces=(),
    )

    max_input_bytes: int = Field(
        default=_DEFAULT_MAX_INPUT_BYTES,
        ge=1,
        validation_alias=AliasChoices("MAX_INPUT_BYTES", "max_input_bytes"),
    )
    model_dir: Path = Field(
        default=Path("models"),
        validation_alias=AliasChoices("MODEL_DIR", "model_dir"),
    )
    lama_weights: Path | None = Field(
        default=None,
        validation_alias=AliasChoices("LAMA_WEIGHTS", "lama_weights"),
    )
    max_ram_mb: int | None = Field(
        default=None,
        validation_alias=AliasChoices("MAX_RAM_MB", "max_ram_mb"),
    )
    max_vram_mb: int | None = Field(
        default=None,
        validation_alias=AliasChoices("MAX_VRAM_MB", "max_vram_mb"),
    )
    crf: int = Field(
        default=23,
        ge=0,
        le=51,
        validation_alias=AliasChoices("CRF", "crf"),
    )
    mask_area_threshold: float = Field(default=0.03, ge=0.0, le=1.0)
    log_level: str = Field(
        default="INFO",
        validation_alias=AliasChoices("LOG_LEVEL", "log_level"),
    )
    opencv_radius: int = Field(default=3, ge=1)
    opencv_method: Literal["telea", "ns"] = "telea"
    max_workers: int = Field(default_factory=_cpu_count, ge=1)
    raft_enabled: bool = False
    tile_size: int = Field(default=512, ge=1)
    tile_overlap: int = Field(default=32, ge=0)
    output_quality: Literal["source", "1080p", "720p"] = "source"
    gradio_server_name: str = "127.0.0.1"
    gradio_share: bool = False

    @model_validator(mode="after")
    def _apply_derived_defaults(self) -> Settings:
        if self.lama_weights is None:
            self.lama_weights = Path(self.model_dir) / "lama.onnx"
        cap = _cpu_count()
        if self.max_workers > cap:
            self.max_workers = cap
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources: list[PydanticBaseSettingsSource] = [
            init_settings,
            env_settings,
            dotenv_settings,
        ]
        toml_path = resolve_config_toml_path()
        if toml_path is not None:
            sources.append(TomlConfigSettingsSource(settings_cls, toml_path))
        sources.append(file_secret_settings)
        return tuple(sources)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def clear_settings_cache() -> None:
    get_settings.cache_clear()
