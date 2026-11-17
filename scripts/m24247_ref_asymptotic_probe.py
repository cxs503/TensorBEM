#!/usr/bin/env python3
"""m24247: control-host re-run of the capytaine resolution probe (cj4526).

The m24149 closure report named two reference-side closure paths: (a) re-run
the cj4526 probe on the control host (the only host with capytaine), or (b)
build the finite-depth multipole analytic anchor.  This driver executes (a)
for the *whole* 12-cell grid at the (96,192) = 18432-panel rung, one rung
finer than the frozen reference ladder (288/1152/4608).

For every cell and quantity it reports:
  * ``order_orig``  -- observed order on the frozen 3-rung ladder
                       (288/1152/4608), i.e. what the study actually used;
  * ``order_fine``  -- observed order on the finest 3 rungs
                       (1152/4608/18432), i.e. what the order would be if the
                       ladder were extended one rung;
  * ``last_rung_change`` -- |v_18432 - v_4608| / |v_18432|, a conservative
                       bound on the reference's remaining truncation error at
                       the new rung.

The pre-registered floor is ``plan.MIN_OBSERVED_ORDER`` (0.3).  The output is
evidence about the reference, written to ``docs/`` -- it is deliberately NOT
promoted into ``benchmarks/verified`` (that would silently change the study's
pre-registered ladder).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorbem.benchmarks import finite_depth_crossval_plan as plan  # noqa: E402
from tensorbem.benchmarks.capytaine_finite_depth_crossval import (  # noqa: E402
    _order_from_three,
)

PROBE = ROOT / "scripts" / "capytaine_finite_depth_resolution_probe.py"
REF = ROOT / "benchmarks" / "verified" / "capytaine_finite_depth" / "capytaine_ref.json"
OUT = ROOT / "docs" / "m24247" / "ref_asymptotic_probe"
RESOLUTION = "96,192"
BATCH = 2  # ~50 cores each; 2 x 50 = 100 < 128


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ref = json.loads(REF.read_text())
    ref_cells = {c["cell"]: c for c in ref["cells"]}
    cells = list(plan.cells())

    # 1. run the probe for every cell (batches of BATCH)
    probe_files: dict[str, Path] = {}
    for i in range(0, len(cells), BATCH):
        batch = cells[i:i + BATCH]
        procs = []
        for cell in batch:
            cid = plan.cell_id(cell["h_over_a"], cell["ka"])
            out = OUT / f"probe_{cid.replace('|', '_').replace('=', '')}.json"
            probe_files[cid] = out
            procs.append(subprocess.Popen(
                [sys.executable, str(PROBE), "--cell", cid,
                 "--resolution", RESOLUTION, "--out", str(out)],
                cwd=str(ROOT),
                env={"PYTHONPATH": str(ROOT / "src"),
                     "PATH": "/usr/bin:/bin", "HOME": "/root"},
            ))
        for p in procs:
            p.wait()
        for cell in batch:
            cid = plan.cell_id(cell["h_over_a"], cell["ka"])
            if not probe_files[cid].exists():
                raise SystemExit(f"probe failed for {cid}")
            print(f"probed {cid}", flush=True)

    # 2. compute orders
    analysis = []
    for cell in cells:
        cid = plan.cell_id(cell["h_over_a"], cell["ka"])
        refc = ref_cells[cid]
        probe = json.loads(probe_files[cid].read_text())
        ladder = refc["levels"]
        panels = [r["panels"] for r in ladder]
        assert panels == [288, 1152, 4608], panels
        for dof in ("heave", "surge"):
            for qty in ("added_mass", "damping"):
                vals = [r["values"][dof][qty] for r in ladder]
                v_new = probe["values"][dof][qty]
                o_orig = _order_from_three(panels, vals)
                o_fine = _order_from_three(
                    [panels[1], panels[2], panels[2] * 4],
                    [vals[1], vals[2], v_new],
                )
                last_change = abs(v_new - vals[2]) / abs(v_new)
                analysis.append({
                    "cell": cid,
                    "h_over_a": cell["h_over_a"],
                    "ka": cell["ka"],
                    "k0h": cell["k0h"],
                    "quantity": f"{dof}.{qty}",
                    "v_288": vals[0],
                    "v_1152": vals[1],
                    "v_4608": vals[2],
                    "v_18432": v_new,
                    "order_orig_288_4608": o_orig,
                    "order_fine_1152_18432": o_fine,
                    "last_rung_change_4608_to_18432": last_change,
                    "order_fine_ge_floor": (
                        o_fine is not None
                        and o_fine >= plan.MIN_OBSERVED_ORDER
                    ),
                })

    o_orig = [a["order_orig_288_4608"] for a in analysis
              if a["order_orig_288_4608"] is not None]
    o_fine = [a["order_fine_1152_18432"] for a in analysis
              if a["order_fine_1152_18432"] is not None]
    summary = {
        "probe_resolution": [96, 192],
        "probe_panels": 18432,
        "frozen_ladder_panels": [288, 1152, 4608],
        "min_observed_order_floor": plan.MIN_OBSERVED_ORDER,
        "n_cells": len(cells),
        "n_quantities": len(analysis),
        "order_orig": {"min": min(o_orig), "max": max(o_orig),
                       "n_below_floor": sum(1 for o in o_orig
                                           if o < plan.MIN_OBSERVED_ORDER)},
        "order_fine": {"min": min(o_fine), "max": max(o_fine),
                       "n_below_floor": sum(1 for o in o_fine
                                            if o < plan.MIN_OBSERVED_ORDER)},
        "last_rung_change": {
            "max": max(a["last_rung_change_4608_to_18432"] for a in analysis),
            "per_quantity_max": {
                q: max(a["last_rung_change_4608_to_18432"] for a in analysis
                        if a["quantity"].split(".")[-1] == q)
                for q in ("added_mass", "damping")
            },
        },
        "analysis": analysis,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in
                      ("n_cells", "n_quantities", "order_orig", "order_fine",
                       "last_rung_change")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
