"""Inventory of public API never referenced by the test suite.

Why this exists
---------------
Every previous audit of this repo checked *whether the numbers are right*
(Ursell benchmark, hydrostatic convergence, doc-number re-computation, mutation
score).  None of them answered the cheaper, more embarrassing question: **which
public functions has no test ever called at all?**  A function that no test
mentions is not covered by any oracle by construction, so it is a verification
debt regardless of how correct it may be.

The scan is purely static (``ast`` + text search), so it costs milliseconds and
never executes the library.  It is deliberately conservative:

* only *public* top-level ``def``/``class`` names (no leading underscore) are
  reported, because private helpers are exercised through their callers;
* a name counts as "referenced" if it appears in **any** file under ``tests/``,
  including as a substring of a ``from tensorbem import ...`` line -- so an
  import-only test suppresses the item.  That makes the list a *lower bound* on
  the debt: still-unreferenced really means never touched;
* ``__init__.py`` re-exports are ignored as "usage" (an export is not a test).

Usage
-----
    PYTHONPATH=src python3 docs/m3108/tools/api_test_coverage.py \
        --json docs/m3108/outputs/api_test_coverage.json
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "src" / "tensorbem"
TESTS = REPO / "tests"


def _test_blob() -> str:
    parts = []
    for path in sorted(TESTS.rglob("*.py")):
        parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def _public_defs(path: Path) -> list[dict]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        name = node.name
        if name.startswith("_"):
            continue
        # docstring-only stubs would be a different kind of debt; flag them.
        doc = ast.get_docstring(node) or ""
        out.append(
            {
                "module": f"tensorbem.{path.stem}",
                "name": name,
                "kind": type(node).__name__,
                "lineno": node.lineno,
                "has_doc": bool(doc),
            }
        )
    return out


def scan() -> dict:
    blob = _test_blob()
    rows = []
    for path in sorted(SRC.glob("*.py")):
        if path.stem == "__init__":
            continue
        for item in _public_defs(path):
            item["referenced_in_tests"] = item["name"] in blob
            rows.append(item)
    untested = [r for r in rows if not r["referenced_in_tests"]]
    by_module: dict[str, dict] = {}
    for r in rows:
        agg = by_module.setdefault(r["module"], {"public": 0, "untested": 0, "names": []})
        agg["public"] += 1
        if not r["referenced_in_tests"]:
            agg["untested"] += 1
            agg["names"].append(r["name"])
    return {
        "public_defs": len(rows),
        "untested_defs": len(untested),
        "coverage_fraction": 1.0 - len(untested) / max(1, len(rows)),
        "by_module": dict(sorted(by_module.items(), key=lambda kv: -kv[1]["untested"])),
        "untested": untested,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    report = scan()
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"public defs: {report['public_defs']}  "
        f"never referenced by tests: {report['untested_defs']}  "
        f"coverage: {report['coverage_fraction']:.1%}"
    )
    for module, agg in report["by_module"].items():
        if agg["untested"]:
            print(f"  {module:44s} {agg['untested']:2d}/{agg['public']:2d}  {', '.join(agg['names'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())