#!/usr/bin/env python3
"""Probe one capytaine finite-depth cell at an arbitrary sphere resolution.

Why this exists
---------------
The frozen reference ladder stops at the 4608-panel rung, and its *measured*
order is often below the pre-registered floor
(:data:`tensorbem.benchmarks.finite_depth_crossval_plan.MIN_OBSERVED_ORDER`):
the reference is then a slowly converging series whose own truncation is of the
same size as the tolerance it is used to judge
(``reference_truncation_within_share`` in the study gates).  Whether adding a
rung helps, or whether this constant-panel family simply cannot reach the band,
is a measurement, not an opinion -- this script makes that measurement without
touching the frozen reference:

    PYTHONPATH=src python scripts/capytaine_finite_depth_resolution_probe.py \
        --cell 'h_over_a=3.0|ka=0.50' --resolution 96,192 --out /tmp/probe.json

The output is deliberately *not* written into ``benchmarks/verified``: a probe is
evidence about the reference, not a replacement for it, and promoting it would
silently change the study's ladder (which the plan fixes before the run).

Memory: capytaine factorises a dense ``n_panels^2`` complex matrix, so the
18432-panel rung needs ~5.4 GB for the matrix alone.  Peak RSS is recorded with
the result so a failed rung is attributable.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import capytaine_finite_depth_ref as producer  # noqa: E402

from tensorbem.benchmarks import finite_depth_crossval_plan as plan  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True,
                        help="cell id as produced by plan.cell_id, e.g. "
                             "'h_over_a=3.0|ka=0.50'")
    parser.add_argument("--resolution", required=True,
                        help="capytaine sphere resolution 'n_theta,n_phi'")
    parser.add_argument("--method", default=plan.CAPYTAINE_METHOD)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    resolution = tuple(int(v) for v in args.resolution.split(","))
    assert len(resolution) == 2, "--resolution takes two integers"
    cells = {
        plan.cell_id(cell["h_over_a"], cell["ka"]): cell for cell in plan.cells()
    }
    if args.cell not in cells:
        raise SystemExit(
            f"unknown cell {args.cell!r}; plan cells are {sorted(cells)}"
        )
    cell = cells[args.cell]

    import capytaine as cpt

    mesh = producer._mesh(cpt, resolution)
    t0 = time.perf_counter()
    values = producer._solve_cell(cpt, mesh, cell, args.method)
    seconds = time.perf_counter() - t0
    record = {
        "cell": args.cell,
        "h_over_a": cell["h_over_a"],
        "ka": cell["ka"],
        "k0h": cell["k0h"],
        "nu": cell["nu"],
        "omega": cell["omega"],
        "method": args.method,
        "resolution": list(resolution),
        "panels": int(mesh.nb_faces),
        "seconds": round(seconds, 3),
        "peak_rss_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1
        ),
        "values": values,
    }
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in
                      ("cell", "panels", "seconds", "peak_rss_mb")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())