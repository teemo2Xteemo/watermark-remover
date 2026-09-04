from __future__ import annotations

from datetime import date
from pathlib import Path

import prepare_release


def test_package_version_reads_pyproject(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "demo"\nversion = "1.2.3"\n', encoding="utf-8")
    assert prepare_release.package_version(pyproject) == "1.2.3"


def test_should_skip_merge_and_changelog_subjects() -> None:
    assert prepare_release.should_skip_subject("Merge branch 'main' into releases")
    assert prepare_release.should_skip_subject("chore: changelog for v1.2.3")
    assert not prepare_release.should_skip_subject("Close M9 with Docker images (#11)")


def test_render_and_upsert_changelog_prepends_new_version() -> None:
    first = prepare_release.render_section("0.1.0", date(2026, 9, 4), ["Initial release"])
    changelog = prepare_release.upsert_changelog("", first, "0.1.0")
    second = prepare_release.render_section("0.2.0", date(2026, 9, 5), ["Add release workflow"])
    changelog = prepare_release.upsert_changelog(changelog, second, "0.2.0")
    assert changelog.startswith("# Changelog\n")
    assert changelog.index("## [0.2.0]") < changelog.index("## [0.1.0]")
    notes = prepare_release.extract_notes(changelog, "0.2.0")
    assert notes.startswith("## [0.2.0] - 2026-09-05\n")
    assert "- Add release workflow" in notes


def test_upsert_replaces_existing_section_for_same_version() -> None:
    original = prepare_release.render_section("0.2.0", date(2026, 9, 5), ["old"])
    changelog = prepare_release.upsert_changelog("# Changelog\n", original, "0.2.0")
    updated = prepare_release.render_section("0.2.0", date(2026, 9, 5), ["new"])
    changelog = prepare_release.upsert_changelog(changelog, updated, "0.2.0")
    assert changelog.count("## [0.2.0]") == 1
    assert "- new" in changelog
    assert "- old" not in changelog


def test_write_changelog_file(tmp_path: Path) -> None:
    path = tmp_path / "CHANGELOG.md"
    prepare_release.write_changelog_file(
        path, "0.1.0", ["First tagged release"], today=date(2026, 9, 4)
    )
    text = path.read_text(encoding="utf-8")
    assert "## [0.1.0] - 2026-09-04" in text
    assert "- First tagged release" in text


def test_previous_tag_skips_current() -> None:
    assert prepare_release.tag_for_version("0.1.0") == "v0.1.0"
    assert prepare_release.previous_tag(["v0.2.0", "v0.1.0"], "v0.2.0") == "v0.1.0"
    assert prepare_release.previous_tag(["v0.1.0"], "v0.1.0") is None
