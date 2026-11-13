"""Gate: repository tooling must be path-portable and scan the tree it lives in.

Motivation (m17083 upstream-delta inventory): three ``tools/`` entry points had
absolute paths to *other agents' worktrees* baked in, e.g.
``tools/coverage_scan.py`` scanned ``/root/tp-agent/work/m16890`` no matter where
it was invoked from, so its report described a different checkout (that tree does
not even contain the modules added in this window).  These tests make the defect
class machine-detectable: code paths (not docstrings) must not reference an
absolute ``/root/tp-agent/work/m<N>`` location, and the coverage scan must report
the repository that contains it.
"""
from __future__ import annotations

import ast
import io
import re
from contextlib import redirect_stdout
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ABSOLUTE_WORKTREE = re.compile(r"/root/tp-agent/work/m\d+")


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """ids of Constant nodes that are module/class/function docstrings."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if not body:
                continue
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                ids.add(id(first.value))
    return ids


def _live_absolute_paths(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(), filename=str(path))
    doc_ids = _docstring_nodes(tree)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and id(node) not in doc_ids:
            if ABSOLUTE_WORKTREE.search(node.value):
                hits.append((node.lineno, node.value))
    return hits


@pytest.mark.parametrize("subdir", ["tools", "src", "scripts", "tests"])
def test_no_absolute_worktree_paths_in_live_code(subdir: str) -> None:
    offenders = {}
    for path in sorted((ROOT / subdir).rglob("*.py")):
        hits = _live_absolute_paths(path)
        if hits:
            offenders[str(path.relative_to(ROOT))] = hits
    assert not offenders, (
        "absolute paths to another agent worktree in live code (breaks portability "
        f"and silently reads a stale checkout): {offenders}"
    )


def test_coverage_scan_reports_this_repository() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "coverage_scan", ROOT / "tools" / "coverage_scan.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    assert Path(mod.DEFAULT_ROOT).resolve() == ROOT, \
        f"coverage_scan default root {mod.DEFAULT_ROOT} is not the repository root {ROOT}"

    rows = mod.scan(ROOT)
    modules = {name for _, _, name in rows}
    # modules added in this inventory window must be visible to the scan
    for expected in ("steady_kelvin", "shape_gradient", "neumann_kelvin"):
        assert expected in modules, f"{expected} missing from coverage scan of {ROOT}"

    buf = io.StringIO()
    with redirect_stdout(buf):
        mod.main(["--root", str(ROOT)])
    out = buf.getvalue().splitlines()
    assert out and Path(out[0].split("scan root:", 1)[1].strip()).resolve() == ROOT


def test_gen_irregular_frequency_figures_writes_inside_repository() -> None:
    src = (ROOT / "tools" / "gen_irregular_frequency_figures.py").read_text()
    assert "/root/tp-agent/work" not in src
    assert "parents[1]" in src  # resolves artifacts/ relative to the repository


def test_wageningen_anchor_requires_explicit_source() -> None:
    """No baked-in absolute reference path: the tool must fail closed instead."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_wageningen_anchor", ROOT / "tools" / "build_wageningen_anchor.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    with pytest.raises(SystemExit) as exc:
        mod.main([])
    assert "SHIP_PLATFORM_WAGENINGEN_SRC" in str(exc.value)