"""Liveness gate for artifact-backed regression guards.

Several test modules in this suite assert properties of a *committed study
artefact* and skip when that artefact is absent, e.g.::

    @pytest.mark.skipif(not ARTEFACT.exists(), reason="artifact not generated")

A skip is reported as a pass in most CI summaries, so an artefact that is
never committed silently converts the guard into dead weight: the suite stays
green while the physics it was supposed to police is unchecked.  That is
exactly what happened to two studies (``test_finite_depth_energy_convergence``
and ``test_ursell_reference_pinning``), whose artefacts lived under
``outputs/`` -- a directory with **zero** tracked files at the time of writing.

This module closes the class of defect rather than the two instances.  It walks
the AST of every test module, evaluates the module-level ``pathlib.Path``
constants that name repository paths, and requires each one that points at a
study artefact to

1. exist on disk, and
2. be tracked by git (i.e. survive a clean ``git archive HEAD`` checkout).

If git is unavailable the tracked-ness half is skipped, but the existence half
still runs.  Paths may be excluded through :data:`EXEMPT` with a reason; the
exemption list is deliberately tiny and reviewed.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent

#: Repository-relative prefixes whose module-level path constants are studied as
#: artefacts.  ``artifacts/`` is git-ignored by design, so it is not listed.
ARTEFACT_PREFIXES = ("outputs/", "tests/data/", "docs/", "examples/")

#: Paths that no test needs to guard, with the reason they are absent.
EXEMPT: dict[str, str] = {
    # generated on demand by the runner, never committed (see .gitignore)
    "artifacts/": "git-ignored build directory",
}


def _iter_test_modules() -> list[Path]:
    return sorted(
        path
        for path in TESTS_DIR.glob("test_*.py")
        if path.name != Path(__file__).name
    )


def _module_path_constants(
    tree: ast.Module, module_path: Path
) -> dict[str, Path]:
    """Evaluate ``Name = <Path chain>`` module-level assignments.

    Only expressions built from ``Path(__file__).resolve().parents[N]`` and ``/``
    with string literals are evaluated -- enough for every artefact constant in
    this suite, and it never executes the module.
    """
    env: dict[str, Path] = {"__file__": module_path}
    found: dict[str, Path] = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = _eval_chain(node.value, env)
        if value is None:
            continue
        env[target.id] = value
        found[target.id] = value
    return found


def _eval_chain(node: ast.AST, env: dict[str, Path]) -> Path | None:
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return Path(node.value)
    if isinstance(node, ast.Call):
        # Path(__file__).resolve().parents[N]  /  Path(__file__).resolve().parent
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in {"parents"}:
            # .parents[N] is a Subscript, handled below
            return None
        if isinstance(func, ast.Attribute) and func.attr == "resolve":
            return _eval_chain(func.value, env)
        if isinstance(func, ast.Name) and func.id == "Path":
            for arg in node.args:
                resolved = _eval_chain(arg, env)
                if resolved is not None:
                    return resolved
            return None
        return None
    if isinstance(node, ast.Subscript):
        base = node.value
        if isinstance(base, ast.Attribute) and base.attr == "parents":
            parent_dir = _eval_chain(base.value, env)
            if parent_dir is None:
                return None
            index = node.slice
            if isinstance(index, ast.Constant) and isinstance(index.value, int):
                # parents[0] of a file path is its directory
                parts = parent_dir.parts
                if index.value <= 0:
                    return Path(*parts[:-1]) if len(parts) > 1 else parent_dir
                return Path(*parts[: -1 - index.value])
            return None
        return None
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        base = _eval_chain(node.value, env)
        if base is None:
            return None
        return base.parent
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _eval_chain(node.left, env)
        right = _eval_chain(node.right, env)
        if left is None:
            return None
        if isinstance(node.right, ast.Constant) and isinstance(node.right.value, str):
            return left / node.right.value
        if right is not None:
            return left / right
        return None
    return None


def _artefact_paths() -> list[tuple[str, str, Path]]:
    """Collect ``(test module, constant name, repo-relative path)`` triples."""
    found: list[tuple[str, str, Path]] = []
    for module in _iter_test_modules():
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        for name, path in _module_path_constants(tree, module).items():
            if not path.is_absolute():
                continue
            try:
                rel = path.relative_to(REPO_ROOT)
            except ValueError:
                continue
            rel_posix = rel.as_posix() + ("/" if path.is_dir() else "")
            if not rel_posix.startswith(ARTEFACT_PREFIXES):
                continue
            if any(rel_posix.startswith(prefix) for prefix in EXEMPT):
                continue
            found.append((module.name, name, rel))
    return found


def _tracked_paths() -> set[str] | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        return None
    return {entry for entry in result.stdout.split("\0") if entry}


TRACKED = _tracked_paths()
ARTEFACTS = _artefact_paths()


def test_the_scan_finds_the_known_gates() -> None:
    """The AST scan must actually see gates -- an empty scan proves nothing."""
    names = {(module, constant) for module, constant, _ in ARTEFACTS}
    for expected in (
        ("test_finite_depth_energy_convergence.py", "ARTIFACT"),
        ("test_finite_depth_refinement.py", "ARTEFACT"),
        ("test_ursell_reference_pinning.py", "STUDY_JSON"),
        ("test_near_surface_band.py", "STUDY"),
    ):
        assert expected in names, f"gate not discovered: {expected}"


def test_every_gated_artefact_exists_on_disk() -> None:
    missing = [
        f"{module}:{constant} -> {path} (worktree checkout incomplete?)"
        for module, constant, path in ARTEFACTS
        if not (REPO_ROOT / path).exists()
    ]
    assert not missing, "artefacts referenced by tests but absent:\n" + "\n".join(missing)


@pytest.mark.skipif(TRACKED is None, reason="git not available")
def test_every_gated_artefact_is_tracked_by_git() -> None:
    """A clean ``git archive HEAD`` checkout must keep every guard alive."""
    assert TRACKED is not None
    untracked = [
        f"{module}:{constant} -> {path} (absent from git -> guard is permanently vacuous)"
        for module, constant, path in ARTEFACTS
        if path.as_posix() not in TRACKED
        and not any(
            tracked == path.as_posix() or tracked.startswith(path.as_posix() + "/")
            for tracked in TRACKED
        )
    ]
    assert not untracked, "artefact gates that would skip in a clean checkout:\n" + "\n".join(
        untracked
    )


def test_scan_actually_covered_a_nontrivial_number_of_paths() -> None:
    assert len(ARTEFACTS) >= 6, f"only {len(ARTEFACTS)} gated artefacts discovered"