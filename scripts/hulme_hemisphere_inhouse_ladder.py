#!/usr/bin/env python3
"""In-house refinement ladder of the anchor ``BEM-HULME-HEMISPHERE`` at one ka.

The anchor grades the panel solver at 1024 constant panels and quotes a worst
band error of 1.15 % against the Hulme multipole series.  Those two numbers --
1.15 % (solver, in-house) and 3.36 % (independent code vs reference) -- are
quoted side by side as if they were commensurable, but the solver's own
discretisation error has never been separated from its solution: the case rungs
are 256 and 1024 panels only, i.e. two points, and two points cannot carry an
observed convergence order.

This driver adds the missing rungs at the worst band point (ka = 2, where
B33 carries the largest error) so that ``tensorbem.benchmarks.
hemisphere_independent_convergence`` can fit the same three-parameter
convergence model to the in-house ladder that it fits to the independent code's
ladder, and compare continuum limits rather than discretised values.

The single-cell work is delegated to
``tensorbem.benchmarks.hulme_hemisphere_case.run_cell`` -- no solver code is
duplicated here, so the ladder cannot diverge from the graded case.

Cost (measured, 128-core host, one process per cell): 64 panels ~1 s,
256 ~40 s, 1024 ~200 s, 4096 ~1 h.  Run it through the compute gate, not the
control plane.

Usage
-----
    python scripts/hulme_hemisphere_inhouse_ladder.py --ka 2.0 \
        --subdivisions 2 3 4 5 \
        --out benchmarks/verified/casepool_bem_hulme_hemisphere/\
inhouse_ladder_ka2.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from tensorbem.benchmarks.hulme_hemisphere_case import run_cell

DEFAULT_OUT = Path(
    "benchmarks/verified/casepool_bem_hulme_hemisphere/inhouse_ladder_ka2.json"
)
QUANTITIES = (
    "A11_dimless",
    "B11_dimless",
    "A33_dimless",
    "B33_dimless",
    "Fx_dimless",
    "Fz_dimless",
)


def build(ka: float, subdivisions: list[int]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for sub in subdivisions:
        started = time.time()
        cell = run_cell(ka, sub)
        cell["wall_seconds"] = time.time() - started
        rows.append(cell)
        print(
            f"subdivisions {sub:2d} = {cell['panels']:5d} panels  "
            f"A33 {cell['A33_dimless']:.6f}  B33 {cell['B33_dimless']:.6f}  "
            f"{cell['wall_seconds']:.1f} s",
            flush=True,
        )
    return {
        "case_code": "BEM-HULME-HEMISPHERE",
        "purpose": (
            "in-house panel-solver refinement ladder at the worst band point, "
            "so that the solver's own convergence order and continuum limit can "
            "be extrapolated and compared with the reference series and with "
            "the independent code's extrapolated limit"
        ),
        "ka": ka,
        "quantities": list(QUANTITIES),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "host": platform.node(),
            "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
                timespec="seconds"
            ),
        },
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ka", type=float, default=2.0)
    parser.add_argument("--subdivisions", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    payload = build(args.ka, args.subdivisions)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "panels": [r["panels"] for r in payload["rows"]]}))
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())