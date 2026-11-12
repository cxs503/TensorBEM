"""Adversarial audit of the capability ledger's "pinned by tests/..." debt notes.

The published baseline (``docs/audit_baseline.md``) lists modules that own no
capability row and defers each of them with the sentence

    "physics/geometry module <leaf> is pinned by tests/<file>.py ... and
     documented; it has no row of its own yet (debt registered in
     docs/m14869/report.md), so its level is NOT claimed here."

"Pinned by" is a strong word for a sentence nothing verifies: the registry's
own detached-test check only asks whether the test *file* mentions the module,
so a file that imports the module and asserts something unrelated would keep
the note alive.  This scanner re-derives the claim from the AST instead:

* **asserting nodes** -- test functions that contain an ``assert`` (or a
  ``pytest.approx``/``numpy.allclose``/``unittest`` comparison) whose line
  calls, or whose module body imports, a public symbol of the module.  A file
  that only imports the module yields *zero* asserting nodes.
* **structural nodes** -- test functions that call a public symbol without an
  assertion (smoke coverage only: proves the module runs, not that it is
  right).

Output is a JSON matrix used to decide, module by module, whether the debt
note's "pinned by" can honestly become a graded capability row, or whether the
note has to be downgraded.  Static only: no BEM solve, no import side effects
beyond ``ast`` parsing.

Usage::

    python scripts/audit_module_pins.py --json docs/capability_pin_audit.json
    python scripts/audit_module_pins.py --table
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"


def _public_symbols(module_path: pathlib.Path) -> list[str]:
    """Top-level public callables/classes defined by the module (AST only)."""

    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                names.append(node.name)
    return names


def _test_functions(path: pathlib.Path) -> list[tuple[str, ast.AST, set[str]]]:
    """``[(node_name, node, imported_names)]`` for every top-level test function."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported.add((alias.asname or alias.name).split(".")[0])
    out = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            out.append((node.name, node, imported))
    return out


def _called_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            fn = sub.func
            if isinstance(fn, ast.Name):
                names.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                names.add(fn.attr)
    return names


def _has_assertion(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Assert):
            return True
        if isinstance(sub, ast.Call):
            fn = sub.func
            label = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if label in {"raises", "approx", "allclose", "isclose", "assert_allclose", "assertAlmostEqual"}:
                return True
    return False


def _parse_test_file(path: pathlib.Path) -> dict | None:
    """Parse one test file once: text plus every test function (incl. methods)."""

    try:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
    except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - defensive
        return None
    records: list[tuple[str, set[str], bool]] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}::")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if child.name.startswith("test"):
                    records.append((f"{prefix}{child.name}", _called_names(child), _has_assertion(child)))
                walk(child, "")  # nested helpers are not test nodes

    walk(tree, "")
    return {"name": path.name, "text": text, "records": records}


def audit() -> dict:
    modules = sorted(p for p in (SRC / "tensorbem").glob("*.py") if p.name != "__init__.py")
    parsed = [rec for rec in (_parse_test_file(p) for p in sorted(TESTS.glob("test_*.py"))) if rec]
    result: dict[str, dict] = {}
    for module_path in modules:
        leaf = module_path.stem
        symbols = _public_symbols(module_path)
        if not symbols:
            continue
        symset = set(symbols)
        asserting: list[str] = []
        structural: list[str] = []
        files: set[str] = set()
        for rec in parsed:
            if leaf not in rec["text"]:
                continue
            for name, called, asserted in rec["records"]:
                if not (called & symset):
                    continue
                files.add(rec["name"])
                node_id = f"tests/{rec['name']}::{name}"
                (asserting if asserted else structural).append(node_id)
        result[leaf] = {
            "symbols": symbols,
            "asserting_nodes": sorted(asserting),
            "structural_nodes": sorted(structural),
            "files": sorted(files),
        }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=pathlib.Path, help="write the matrix here")
    parser.add_argument("--table", action="store_true", help="print a compact table")
    parser.add_argument(
        "--modules",
        nargs="*",
        default=None,
        help="restrict the table to these module leaves",
    )
    args = parser.parse_args(argv)

    matrix = audit()
    if args.modules:
        keep = set(args.modules)
        matrix = {k: v for k, v in matrix.items() if k in keep}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    if args.table or not args.json:
        width = max(len(k) for k in matrix)
        for leaf, rec in sorted(matrix.items()):
            print(
                f"{leaf:<{width}}  assert={len(rec['asserting_nodes']):>2}  "
                f"structural={len(rec['structural_nodes']):>2}  files={len(rec['files']):>2}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())