#!/usr/bin/env python3
"""Regenerate case-pool artefacts that the registry promises but the tree lost.

The case-pool registry (:mod:`tensorbem.benchmarks.case_pool_registry`) points at
one committed artefact set per anchor, and ``tests/test_case_pool_registry.py``
turns red as soon as one of those paths stops existing.  On the merge that added
this script four anchors were red, all of them for *derived* files: the nine
figures of the annulus / cylinder / ellipse cases and the Havalock ``run.log``.
The numbers behind them were committed (``results.json`` / ``numbers.json``);
only the rendering step had been dropped from the merge, so the fix is to rerun
the *presentation* code on the committed data, not to re-solve the BEM problem
and not to delete the evidence pointer.

Every regeneration therefore calls the owning module's own figure routine with
the committed artefact as the only input -- no hand-copied numbers, no second
implementation of the plot.  Paths that have no such producer are reported as
unrepairable and the script exits non-zero, so a genuinely lost artefact can
never be silently papered over.

Usage::

    PYTHONPATH=src python3 scripts/repair_case_pool_evidence.py --check
    PYTHONPATH=src python3 scripts/repair_case_pool_evidence.py --repair

The audit trail (sha256 + byte size + producer of every regenerated file) is
written to ``docs/case_pool_records/evidence_repair.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorbem.benchmarks import case_pool_registry as registry  # noqa: E402

AUDIT_PATH = Path("docs/case_pool_records/evidence_repair.json")
HAVELOCK_NUMBERS = Path("docs/case_bem_submerged_sphere_havelock/numbers.json")
HAVELOCK_LOG = Path("docs/case_bem_submerged_sphere_havelock/run.log")
ANNULUS_ECCENTRIC = Path("benchmarks/verified/bem_annulus_laplace_eccentric")
CYLINDER = Path("benchmarks/verified/case_bem_laplace_cylinder")
ELLIPSE = Path("benchmarks/verified/case_bem_ellipse_2d_added_mass")


def _annulus_figures(_: Path) -> list[Path]:
    from tensorbem.benchmarks import eccentric_annulus

    result = json.loads((ANNULUS_ECCENTRIC / "results.json").read_text())
    return eccentric_annulus.write_figures(result, ANNULUS_ECCENTRIC / "figures")


def _cylinder_figures(_: Path) -> list[Path]:
    from tensorbem.benchmarks import case_cylinder_stream as case

    artifact = json.loads((CYLINDER / "results.json").read_text())
    figures = CYLINDER / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    return [
        case.plot_field(artifact, figures / "field.png"),
        case.plot_convergence(artifact, figures / "convergence.png"),
        case.plot_near_wall(artifact, figures / "near_wall.png"),
    ]


def _ellipse_figures(_: Path) -> list[Path]:
    from tensorbem.benchmarks import case_ellipse_2d_added_mass as case

    artifact = json.loads((ELLIPSE / "results.json").read_text())
    figures = ELLIPSE / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    return [figures / name for name in case._render_figures(artifact, figures)]


def _havelock_log(_: Path) -> list[Path]:
    """Materialise ``run.log`` from the transcript the driver stored in numbers.json."""
    numbers = json.loads(HAVELOCK_NUMBERS.read_text())
    lines = numbers.get("log")
    if not lines:
        raise RuntimeError(f"{HAVELOCK_NUMBERS} carries no 'log' transcript")
    HAVELOCK_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [HAVELOCK_LOG]


#: case code -> routine that rebuilds *all* of that case's derived artefacts
PRODUCERS: dict[str, Callable[[Path], list[Path]]] = {
    "CASE-BEM-ANNULUS-LAPLACE": _annulus_figures,
    "CASE-BEM-LAPLACE-CYLINDER": _cylinder_figures,
    "CASE-BEM-ELLIPSE-2D-ADDED-MASS": _ellipse_figures,
    "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK": _havelock_log,
}


def _digest(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="report missing paths only")
    group.add_argument("--repair", action="store_true", help="regenerate derived artefacts")
    args = parser.parse_args(argv)

    before = registry.missing_paths()
    print(json.dumps({"missing_before": before}, indent=2))
    if args.check:
        return 0 if not before else 1

    removed: list[dict[str, Any]] = []
    for case_code, paths in before.items():
        missing = {REPO_ROOT / rel for rel in paths if not (REPO_ROOT / rel).is_file()}
        if case_code not in PRODUCERS:
            print(f"UNREPAIRABLE {case_code}: {sorted(missing)}", file=sys.stderr)
            continue
        produced = [(REPO_ROOT / Path(p)).resolve() for p in PRODUCERS[case_code](REPO_ROOT)]
        produced = [p for p in produced if p.is_file()]
        if not produced:
            print(f"NO OUTPUT {case_code}", file=sys.stderr)
            continue
        missing_resolved = {p.resolve() for p in missing}
        hit = [p for p in produced if p in missing_resolved]
        if not hit:
            print(f"PRODUCER EMITTED NOTHING REGISTERED {case_code}: "
                  f"{[str(p) for p in produced]}", file=sys.stderr)
        removed.extend(_digest(p) for p in sorted(hit))

    after = registry.missing_paths()
    audit = {
        "schema": "tensorbem/case_pool_evidence_repair/1",
        "producers": {code: fn.__name__ for code, fn in PRODUCERS.items()},
        "missing_before": before,
        "regenerated": removed,
        "missing_after": after,
        "verdicts_after": {e.case_code: registry.artifact_verdict(e.case_code)
                           for e in registry.list_cases()},
    }
    out = REPO_ROOT / AUDIT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"regenerated": [r["path"] for r in removed],
                      "missing_after": after}, indent=2))
    return 0 if not after else 1


if __name__ == "__main__":  # pragma: no cover - script entry
    raise SystemExit(main())