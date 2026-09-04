"""Build tag name and changelog notes from pyproject.toml + git history.

Used by .github/workflows/release.yml when commits land on the `releases` branch.
Does not call the processing pipeline or download weights.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

_VERSION_RE = re.compile(r'(?m)^version\s*=\s*"(?P<version>\d+\.\d+\.\d+)"\s*$')
_SECTION_RE = re.compile(
    r"(?ms)^## \[(?P<version>\d+\.\d+\.\d+)\] - (?P<day>\d{4}-\d{2}-\d{2})\n"
    r"(?P<body>.*?)(?=^## \[|\Z)"
)
_SKIP_SUBJECT = re.compile(
    r"^(Merge branch |Merge remote-tracking branch |chore: changelog for v\d+\.\d+\.\d+)",
)
CHANGELOG_HEADING = "# Changelog\n"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def package_version(pyproject: Path) -> str:
    text = pyproject.read_text(encoding="utf-8")
    match = _VERSION_RE.search(text)
    if match is None:
        raise SystemExit(f"no project version in {pyproject}")
    return match.group("version")


def tag_for_version(version: str) -> str:
    return f"v{version}"


def should_skip_subject(subject: str) -> bool:
    stripped = subject.strip()
    return not stripped or _SKIP_SUBJECT.match(stripped) is not None


def render_section(version: str, day: date, subjects: list[str]) -> str:
    lines = [f"## [{version}] - {day.isoformat()}", ""]
    if subjects:
        lines.append("### Changes")
        lines.append("")
        for subject in subjects:
            lines.append(f"- {subject}")
        lines.append("")
    else:
        lines.append("- No user-facing commit subjects since the previous tag.")
        lines.append("")
    return "\n".join(lines)


def upsert_changelog(existing: str, section: str, version: str) -> str:
    body = existing.strip() if existing.strip() else CHANGELOG_HEADING.strip()
    if not body.startswith("# Changelog"):
        body = f"{CHANGELOG_HEADING.strip()}\n\n{body}"
    marker = f"## [{version}]"
    if marker in body:
        replaced, count = _SECTION_RE.subn(
            lambda match: section if match.group("version") == version else match.group(0),
            body,
            count=0,
        )
        if count:
            return replaced.rstrip() + "\n"
        raise SystemExit(f"changelog already has {version} in an unexpected format")
    heading, _, rest = body.partition("\n")
    rest = rest.lstrip("\n")
    if rest:
        return f"{heading}\n\n{section}{rest}".rstrip() + "\n"
    return f"{heading}\n\n{section}".rstrip() + "\n"


def extract_notes(changelog: str, version: str) -> str:
    for match in _SECTION_RE.finditer(changelog):
        if match.group("version") == version:
            heading = f"## [{version}] - {match.group('day')}\n"
            return heading + match.group("body").rstrip() + "\n"
    raise SystemExit(f"no changelog section for {version}")


def _git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        err = result.stderr.strip() or result.stdout.strip()
        raise SystemExit(err or f"git {' '.join(args)} failed")
    return result.stdout


def list_version_tags(cwd: Path) -> list[str]:
    raw = _git(["tag", "--list", "v[0-9]*.[0-9]*.[0-9]*", "--sort=-v:refname"], cwd)
    return [line.strip() for line in raw.splitlines() if line.strip()]


def previous_tag(tags: list[str], current_tag: str) -> str | None:
    for tag in tags:
        if tag != current_tag:
            return tag
    return None


def commit_subjects_since(cwd: Path, since_tag: str | None) -> list[str]:
    rev_range = f"{since_tag}..HEAD" if since_tag else "HEAD"
    raw = _git(["log", rev_range, "--pretty=format:%s"], cwd)
    subjects: list[str] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        subject = line.strip()
        if should_skip_subject(subject) or subject in seen:
            continue
        seen.add(subject)
        subjects.append(subject)
    return subjects


def tag_exists(cwd: Path, tag: str) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}"],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def write_changelog_file(
    changelog_path: Path,
    version: str,
    subjects: list[str],
    today: date | None = None,
) -> str:
    section = render_section(version, today or datetime.now(timezone.utc).date(), subjects)
    existing = changelog_path.read_text(encoding="utf-8") if changelog_path.exists() else ""
    updated = upsert_changelog(existing, section, version)
    changelog_path.write_text(updated, encoding="utf-8")
    return updated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--print-version", action="store_true")
    group.add_argument("--print-tag", action="store_true")
    group.add_argument("--print-notes", action="store_true")
    group.add_argument("--write-changelog", action="store_true")
    group.add_argument(
        "--gate",
        action="store_true",
        help="print new|already_released for the current pyproject version tag",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = repo_root()
    version = package_version(root / "pyproject.toml")
    tag = tag_for_version(version)
    if args.print_version:
        print(version)
        return 0
    if args.print_tag:
        print(tag)
        return 0
    if args.gate:
        print("already_released" if tag_exists(root, tag) else "new")
        return 0
    tags = list_version_tags(root)
    prior = previous_tag(tags, tag)
    subjects = commit_subjects_since(root, prior)
    changelog_path = root / "CHANGELOG.md"
    if args.write_changelog:
        write_changelog_file(changelog_path, version, subjects)
        return 0
    if args.print_notes:
        text = changelog_path.read_text(encoding="utf-8") if changelog_path.exists() else ""
        if f"## [{version}]" not in text:
            text = upsert_changelog(
                text,
                render_section(version, datetime.now(timezone.utc).date(), subjects),
                version,
            )
        sys.stdout.write(extract_notes(text, version))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
