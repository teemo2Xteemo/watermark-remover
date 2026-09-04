from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "watermark_remover"
START_PACKAGES = ("engines", "masks", "video", "io")
FORBIDDEN_PREFIXES = (
    "requests",
    "httpx",
    "urllib.request",
    "urllib3",
    "aiohttp",
    "http.client",
)


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC_ROOT.parent)
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _iter_package_files(package: str) -> Iterator[Path]:
    root = SRC_ROOT / package
    yield from sorted(root.rglob("*.py"))


def _resolve_from_import(module_name: str, path: Path, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module
    parts = module_name.split(".")
    if path.name != "__init__.py":
        parts = parts[:-1]
    if node.level > len(parts):
        return None
    parent = parts[: len(parts) - node.level + 1]
    if node.module:
        parent.extend(node.module.split("."))
    return ".".join(parent) if parent else None


def _imported_names(path: Path) -> set[str]:
    module_name = _module_name(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            resolved = _resolve_from_import(module_name, path, node)
            if resolved:
                names.add(resolved)
            elif node.module:
                names.add(node.module)
            for alias in node.names:
                if resolved:
                    names.add(f"{resolved}.{alias.name}")
                elif node.level == 0:
                    names.add(alias.name)
    return names


def _is_forbidden(name: str) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in FORBIDDEN_PREFIXES)


def _our_module_path(module_name: str) -> Path | None:
    if not module_name.startswith("watermark_remover."):
        if module_name == "watermark_remover":
            init = SRC_ROOT / "__init__.py"
            return init if init.is_file() else None
        return None
    rest = module_name.removeprefix("watermark_remover.")
    file_path = SRC_ROOT / rest.replace(".", "/") 
    py_file = Path(str(file_path) + ".py")
    init_file = file_path / "__init__.py"
    if py_file.is_file():
        return py_file
    if init_file.is_file():
        return init_file
    return None


def _walk_import_graph() -> tuple[set[str], dict[str, Path]]:
    queue: list[Path] = []
    for package in START_PACKAGES:
        queue.extend(_iter_package_files(package))
    seen: dict[str, Path] = {}
    third_party: set[str] = set()
    while queue:
        path = queue.pop()
        module_name = _module_name(path)
        if module_name in seen:
            continue
        seen[module_name] = path
        for imported in _imported_names(path):
            our = _our_module_path(imported)
            if our is not None:
                if _module_name(our) not in seen:
                    queue.append(our)
                continue
            root = imported.split(".", 1)[0]
            if root not in {"watermark_remover"}:
                third_party.add(imported)
    return third_party, seen


def test_engines_masks_video_io_import_graph_has_no_http_clients() -> None:
    third_party, seen = _walk_import_graph()
    for package in START_PACKAGES:
        assert any(
            name == f"watermark_remover.{package}"
            or name.startswith(f"watermark_remover.{package}.")
            for name in seen
        ), f"import graph missed package {package}"
    hits = sorted(name for name in third_party if _is_forbidden(name))
    assert hits == [], f"HTTP clients reachable from engines/masks/video/io: {hits}"


def test_import_graph_starts_from_required_packages_only() -> None:
    _third_party, seen = _walk_import_graph()
    assert "watermark_remover.ui.app" not in seen
    assert "watermark_remover.cli" not in seen
