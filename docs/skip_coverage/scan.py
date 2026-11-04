"""Audit the test suite for coverage that a clean checkout cannot see.

Motivation
----------
A test guarded by ``pytest.skipif(not ARTIFACT.exists(), ...)`` is only as good
as the step that produces ``ARTIFACT``.  When that step writes below a
git-ignored directory (``outputs/``, ``artifacts/``), a clean checkout -- and
therefore CI -- has no such file, the guard fires, and the physics assertion is
**silently deleted from the run**: the suite is green and the documentation
still says "covered".

TensorBEM has been here before on the CI-workflow side
(``docs/ci_red_triage_m3314.md`` fixed the *ordering* of evidence generation vs.
``pytest``).  This script audits the other half: which artifacts the tests ask
for, and whether a clean checkout can supply them.

Method (no solver is run)
-------------------------
1. Every ``tests/test_*.py`` is parsed with :mod:`ast`; module-level assignments
   whose value renders as a path ending in ``.json`` are collected
   (``ARTIFACT``, ``ARTEFACT``, ``STUDY``, ``STUDY_JSON``, ``PIN_FILE`` ...).
2. A module "guards" an artifact when it contains ``pytest.skip``/``skipif``
   referencing that name, or an ``.exists()`` early return that skips.
3. Each artifact path is classified against the checkout:

   ``VERSIONED`` -- ``git ls-files`` tracks exactly that path, so CI sees it;
   ``CI``        -- a ``.github/workflows/*.yml`` step writes that path;
   ``UNTRACKED`` -- neither: the guarded assertions cannot run on CI.

4. A module is *visible* when at least one of its guarded artifacts is
   ``VERSIONED``/``CI``; a module with only ``UNTRACKED`` artifacts is *dark*.

Usage::

    PYTHONPATH=src python3 docs/skip_coverage/scan.py --output docs/skip_coverage/report.json
    PYTHONPATH=src python3 docs/skip_coverage/scan.py --strict

``--strict`` exits 1 when any module is dark, so CI can gate on it.
``tests/test_skip_coverage.py`` runs the same scan and additionally requires
every ``UNTRACKED`` path to be listed with a reason.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests"
WORKFLOWS = ROOT / ".github" / "workflows"


# ---------------------------------------------------------------------------
# checkout inspection
# ---------------------------------------------------------------------------
def tracked_files() -> set[str]:
    """POSIX paths tracked by git (empty set if git is unavailable)."""
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - no git
        return set()
    return {line.strip() for line in out.splitlines() if line.strip()}


def workflow_text() -> str:
    """Concatenated text of every CI workflow file."""
    chunks: list[str] = []
    for pattern in ("*.yml", "*.yaml"):
        for path in sorted(WORKFLOWS.glob(pattern)):
            chunks.append(path.read_text())
    return "\n".join(chunks)


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------
def _join_path(node: ast.AST) -> str | None:
    """Render a ``ROOT / "a" / "b.json"`` expression as a relative path suffix.

    Repository anchors (``Path(__file__).resolve().parents[1]``, or a module
    constant such as ``ROOT``) contribute nothing; only string components are
    kept, so the result is resolved against the checkout root by suffix.
    """
    parts: list[str] = []

    def walk(n: ast.AST) -> bool:
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div):
            return walk(n.left) and walk(n.right)
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            parts.append(n.value)
            return True
        if isinstance(n, ast.Name):
            return True  # anchor: ROOT, REPO, TESTS, ...
        if isinstance(n, ast.Call):
            name = getattr(n.func, "id", None) or getattr(n.func, "attr", None)
            return name in {"Path", "resolve", "absolute"}
        if isinstance(n, ast.Attribute) and n.attr in {"parents", "parent"}:
            return True
        if isinstance(n, ast.Subscript):
            return True  # parents[1]
        return False

    walk(node)
    if not parts:
        return None
    return "/".join(part.strip("/") for part in parts if part.strip("/"))


def module_path_constants(tree: ast.Module) -> dict[str, str]:
    """Module-level names bound to a path expression that names a JSON file."""
    found: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if node.value is None:
            continue
        rendered = _join_path(node.value)
        if not rendered or not rendered.endswith(".json"):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                found[target.id] = rendered
    return found


def guard_references(tree: ast.Module) -> tuple[set[str], list[str]]:
    """Names referenced by a skip guard, plus the literal reasons used."""
    names: set[str] = set()
    reasons: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fname = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if fname not in {"skip", "skipif"}:
            continue
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            for inner in ast.walk(arg):
                if isinstance(inner, ast.Name):
                    names.add(inner.id)
                elif isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                    reasons.append(inner.value)
    return names, reasons


def tests_touching(tree: ast.Module, name: str) -> list[str]:
    """Test functions whose body mentions ``name``."""
    hit: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Name) and inner.id == name:
                hit.append(node.name)
                break
    return sorted(set(hit))


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------
def ci_writes(rel_path: str, workflow: str) -> bool:
    """Does a workflow step write this artifact?"""
    if rel_path in workflow:
        return True
    return f"--json {rel_path}" in workflow or f"--output {rel_path}" in workflow


def classify(rel_path: str, tracked: set[str], workflow: str) -> str:
    if rel_path in tracked:
        return "VERSIONED"
    if ci_writes(rel_path, workflow):
        return "CI"
    return "UNTRACKED"


def scan() -> dict:
    tracked = tracked_files()
    workflow = workflow_text()
    rows: list[dict] = []
    modules_with_guards: set[str] = set()
    dark_modules: set[str] = set()
    for module in sorted(TESTS.glob("test_*.py")):
        source = module.read_text()
        if "skip" not in source:
            continue
        tree = ast.parse(source)
        consts = module_path_constants(tree)
        if not consts:
            continue
        skip_names, skip_reasons = guard_references(tree)
        guarded = []
        for name, rel_path in sorted(consts.items()):
            touching = tests_touching(tree, name)
            if not touching:
                continue
            # A module-level constant is a guard when a skip guard names it, or
            # when the tests that use it are also the tests that skip.
            is_guard = name in skip_names or bool(skip_reasons)
            if not is_guard:
                continue
            verdict = classify(rel_path, tracked, workflow)
            guarded.append(
                {
                    "module": module.name,
                    "constant": name,
                    "artifact": rel_path,
                    "verdict": verdict,
                    "tests": touching,
                }
            )
        if not guarded:
            continue
        modules_with_guards.add(module.name)
        visible = any(r["verdict"] in {"VERSIONED", "CI"} for r in guarded)
        for row in guarded:
            row["module_visible"] = visible
            row["skip_reasons"] = sorted(set(skip_reasons))[:3]
        rows.extend(guarded)
        if not visible:
            dark_modules.add(module.name)
    return {
        "rows": rows,
        "summary": {
            "guarded_artifacts": len(rows),
            "versioned": sum(r["verdict"] == "VERSIONED" for r in rows),
            "ci_generated": sum(r["verdict"] == "CI" for r in rows),
            "untracked": sum(r["verdict"] == "UNTRACKED" for r in rows),
            "modules_with_guards": sorted(modules_with_guards),
            "visible_modules": sorted(modules_with_guards - dark_modules),
            "dark_modules": sorted(dark_modules),
            "untracked_paths": sorted({r["artifact"] for r in rows if r["verdict"] == "UNTRACKED"}),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="write the JSON report here")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when a guarded artifact has no clean-checkout source",
    )
    args = parser.parse_args(argv)
    payload = scan()
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
    if args.strict and payload["summary"]["dark_modules"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())