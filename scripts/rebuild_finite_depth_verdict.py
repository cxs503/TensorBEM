#!/usr/bin/env python3
"""Recompute the finite-depth cross-validation verdict from frozen artifacts.

The study driver (:mod:`tensorbem.benchmarks.capytaine_finite_depth_crossval`)
rebuilds the TensorBEM ladder whenever it runs, which is right for a live study
but wrong for re-judging a frozen one: after a change to the *comparison* rules
(extrapolation gating, tolerance shares, gates) the numbers that changed are
derived from artifacts that did not.  This script re-derives the whole verdict
from

* the frozen independent reference ``capytaine_ref.json``,
* the frozen TensorBEM ladder ``tensor_ladder.json``,

and writes ``results.json``, the markdown page and the figures, without calling
a solver.  It regenerates nothing it cannot read: a missing artifact is an
error, not a zero.

Usage::

    PYTHONPATH=src python scripts/rebuild_finite_depth_verdict.py
    PYTHONPATH=src python scripts/rebuild_finite_depth_verdict.py --verify-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tensorbem.benchmarks import capytaine_finite_depth_crossval as study
from tensorbem.benchmarks import finite_depth_crossval_plan as plan


def rebuild(root: Path, *, write: bool = True,
            doc_path: Path = study.DOC_PATH) -> dict:
    """Re-derive comparison, gates, page and figures from the frozen artifacts."""
    ref_path = root / study.REFERENCE_FILE
    ladder_path = root / study.LADDER_FILE
    for path in (ref_path, ladder_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing: the verdict cannot be rebuilt from nothing"
            )
    reference = json.loads(ref_path.read_text())
    ladder = json.loads(ladder_path.read_text())
    comparison = study.compare(ladder, reference)
    gate_rows = study.gates(comparison, ladder, reference)
    result = {
        "case": {
            "body": "submerged sphere",
            "radius": plan.SPHERE_RADIUS,
            "submergence": plan.SPHERE_SUBMERGENCE,
            "depth_ratios": list(plan.DEPTH_RATIOS),
            "ka_grid": list(plan.KA_GRID),
            "density": plan.DENSITY,
            "gravity": plan.GRAVITY,
            "levels": [int(level) for level in plan.TENSORBEM_LEVELS],
            "execution_status": study.execution_status(
                ladder, plan.TENSORBEM_LEVELS, plan.EXTRA_TENSORBEM_LEVEL
            ),
            "verdict_rebuilt_from_frozen_artifacts": True,
        },
        "reference_provenance": reference.get("provenance", {}),
        "tensor_environment": ladder.get("environment", {}),
        "comparison": comparison,
        "gates": gate_rows,
        "reference": reference,
        "ladder": ladder,
    }
    if write:
        out = root / study.RESULT_FILE
        out.write_text(json.dumps(result, indent=1) + "\n")
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text(study.render_document(result))
        result["figures"] = study.write_figures(result, doc_path.parent)
        out.write_text(json.dumps(result, indent=1) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=study.DEFAULT_ROOT)
    parser.add_argument("--doc", type=Path, default=study.DOC_PATH)
    parser.add_argument("--verify-only", action="store_true",
                        help="rebuild in memory and compare against results.json")
    args = parser.parse_args(argv)
    result = rebuild(args.root, write=not args.verify_only, doc_path=args.doc)
    failing = [g["gate"] for g in result["gates"] if not g["passed"]]
    print(f"execution_status: {result['case']['execution_status']}")
    print(f"failing gates: {failing}")
    if args.verify_only:
        frozen = json.loads((args.root / study.RESULT_FILE).read_text())
        same = frozen["gates"] == result["gates"]
        print(f"frozen gates identical: {same}")
        return 0 if same else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())