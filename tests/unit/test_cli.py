from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from typer.testing import CliRunner

from watermark_remover.cli import app
from watermark_remover.config import Settings, clear_settings_cache
from watermark_remover.exceptions import ResourceLimitError
from watermark_remover.image_processor import ImageProcessor
from watermark_remover.io.image import read_image

runner = CliRunner()


def test_cli_happy_path_exit_zero(fixtures_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "result.png"
    result = runner.invoke(
        app,
        [
            "--input",
            str(fixtures_dir / "still_logo.png"),
            "--mask",
            str(fixtures_dir / "still_logo.mask.png"),
            "--engine",
            "opencv",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.is_file()
    image = read_image(output)
    assert image.shape[2] == 3
    assert image.dtype == np.uint8


def test_cli_default_output_stem_inpainted(fixtures_dir: Path, tmp_path: Path) -> None:
    src = tmp_path / "photo.png"
    src.write_bytes((fixtures_dir / "still_logo.png").read_bytes())
    result = runner.invoke(
        app,
        [
            "--input",
            str(src),
            "--mask",
            str(fixtures_dir / "still_logo.mask.png"),
            "--engine",
            "opencv",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "photo_inpainted.png").is_file()


def test_cli_empty_mask_exit_1(fixtures_dir: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "--input",
            str(fixtures_dir / "still_logo.png"),
            "--mask",
            str(fixtures_dir / "still_empty.mask.png"),
            "--output",
            str(tmp_path / "out.png"),
        ],
    )
    assert result.exit_code == 1


def test_cli_full_mask_exit_1(fixtures_dir: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "--input",
            str(fixtures_dir / "still_logo.png"),
            "--mask",
            str(fixtures_dir / "still_full.mask.png"),
            "--output",
            str(tmp_path / "out.png"),
        ],
    )
    assert result.exit_code == 1


def test_cli_allow_empty_mask_exit_0(fixtures_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "empty_ok.png"
    result = runner.invoke(
        app,
        [
            "--input",
            str(fixtures_dir / "still_logo.png"),
            "--mask",
            str(fixtures_dir / "still_empty.mask.png"),
            "--allow-empty-mask",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.is_file()


def test_cli_allow_full_mask_exit_0(fixtures_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "full_ok.png"
    result = runner.invoke(
        app,
        [
            "--input",
            str(fixtures_dir / "still_logo.png"),
            "--mask",
            str(fixtures_dir / "still_full.mask.png"),
            "--allow-full-mask",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.is_file()


def test_cli_overwrite_required(fixtures_dir: Path, tmp_path: Path) -> None:
    src = tmp_path / "same.png"
    src.write_bytes((fixtures_dir / "still_logo.png").read_bytes())
    original = src.read_bytes()
    denied = runner.invoke(
        app,
        [
            "--input",
            str(src),
            "--mask",
            str(fixtures_dir / "still_logo.mask.png"),
            "--output",
            str(src),
        ],
    )
    assert denied.exit_code == 1
    assert src.read_bytes() == original
    allowed = runner.invoke(
        app,
        [
            "--input",
            str(src),
            "--mask",
            str(fixtures_dir / "still_logo.mask.png"),
            "--output",
            str(src),
            "--overwrite",
        ],
    )
    assert allowed.exit_code == 0, allowed.output


def test_cli_lama_exit_2(
    fixtures_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LAMA_WEIGHTS", str(tmp_path / "missing.onnx"))
    clear_settings_cache()
    result = runner.invoke(
        app,
        [
            "--input",
            str(fixtures_dir / "still_logo.png"),
            "--mask",
            str(fixtures_dir / "still_logo.mask.png"),
            "--engine",
            "lama",
            "--output",
            str(tmp_path / "out.png"),
        ],
    )
    assert result.exit_code == 2


def test_cli_mask_auto_never_applies(fixtures_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "should_not_exist.png"
    result = runner.invoke(
        app,
        [
            "--input",
            str(fixtures_dir / "still_logo.png"),
            "--mask",
            "auto",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 1
    assert not output.exists()
    text = f"{result.output}\n{result.stderr}\n{result.exception}"
    assert "never auto-applies" in text


def test_cli_missing_input_exit_1() -> None:
    result = runner.invoke(app, ["--mask", "x.png"])
    assert result.exit_code == 1


def test_cli_keyboard_interrupt_exit_130(fixtures_dir: Path, tmp_path: Path) -> None:
    with patch.object(ImageProcessor, "process", side_effect=KeyboardInterrupt):
        result = runner.invoke(
            app,
            [
                "--input",
                str(fixtures_dir / "still_logo.png"),
                "--mask",
                str(fixtures_dir / "still_logo.mask.png"),
                "--output",
                str(tmp_path / "out.png"),
            ],
        )
    assert result.exit_code == 130


def test_cli_resource_limit_exit_3(fixtures_dir: Path, tmp_path: Path) -> None:
    with patch(
        "watermark_remover.cli.validate_resolution_limits",
        side_effect=ResourceLimitError("capped"),
    ):
        result = runner.invoke(
            app,
            [
                "--input",
                str(fixtures_dir / "still_logo.png"),
                "--mask",
                str(fixtures_dir / "still_logo.mask.png"),
                "--output",
                str(tmp_path / "out.png"),
            ],
        )
    assert result.exit_code == 3


def test_cli_json_mask(fixtures_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "from_json.png"
    result = runner.invoke(
        app,
        [
            "--input",
            str(fixtures_dir / "still_logo.png"),
            "--mask",
            str(fixtures_dir / "still_logo.mask.json"),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.is_file()


def test_cli_oversize_uses_settings(
    fixtures_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "watermark_remover.cli.get_settings",
        lambda: Settings(max_input_bytes=1),
    )
    result = runner.invoke(
        app,
        [
            "--input",
            str(fixtures_dir / "still_logo.png"),
            "--mask",
            str(fixtures_dir / "still_logo.mask.png"),
            "--output",
            str(tmp_path / "out.png"),
        ],
    )
    assert result.exit_code == 1


def test_cli_ui_invokes_launch() -> None:
    with patch("watermark_remover.ui.app.launch") as mocked:
        result = runner.invoke(app, ["ui"])
    assert result.exit_code == 0, result.output
    mocked.assert_called_once()
