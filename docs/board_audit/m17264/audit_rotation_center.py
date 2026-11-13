"""Board audit m17264 / subject m17209 (CASE-BEM-CUMMINS-WIGLEY).

Root-cause probe for the R1 cross-solver gate of the audited case.

The case (``src/tensorbem/benchmarks/cummins_wigley_kernel_6dof.py``, function
``radiation_sweep``) calls ``solve_radiation_3d`` **without** the ``center``
keyword.  The solver default is ``(0, 0, 0)`` (waterline).  The frozen
independent reference ``benchmarks/verified/wigley3_recalled_ref/capytaine_ref.json``
was produced with rotation centre ``(0, 0, -T/2) = (0, 0, -0.09375)``, and the
repo's own earlier wigley3 study (``tensorbem/benchmarks/wigley3_recalled_ref.py``,
line ~166) does pass ``center=CENTER``.  This script re-solves the *same* meshes
with both conventions at a single frequency and prints the per-DOF diagonal
added mass / damping against the frozen reference, so the audit can state
whether the reported R1 roll failure (-55.5 %) is a solver-physics finding or a
rotation-centre convention mismatch.

The probe imports only modules that exist on ``origin/main`` (the audited case
module is *not* merged), i.e. it is runnable from any TensorBEM checkout.

    PYTHONPATH=src python3 docs/board_audit/m17264/audit_rotation_center.py \
        --levels 10x5 20x10 40x20 --omega 1.0 --out artifacts/audit_m17264/center_check.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from tensorbem.free_surface_solver3d import solve_radiation_3d
from tensorbem.wigley3_freq import (WIGLEY3_B, WIGLEY3_C, WIGLEY3_L, WIGLEY3_T,
                                    wigley3_wetted_hull_mesh)

LABELS = ("surge", "sway", "heave", "roll", "pitch", "yaw")
REF_DEFAULT = pathlib.Path("benchmarks/verified/wigley3_recalled_ref/capytaine_ref.json")


def load_reference(path=REF_DEFAULT, builder="wigley3_freq", level="n_sections=20,n_levels=10"):
    ref = json.loads(pathlib.Path(path).read_text())
    entry = next(e for e in ref["builders"][builder] if e["level"] == level)
    return ref["metadata"], entry


def solve(n_sections: int, n_levels: int, omega: float, center):
    mesh = wigley3_wetted_hull_mesh(L=WIGLEY3_L, B=WIGLEY3_B, T=WIGLEY3_T, c=WIGLEY3_C,
                                    n_sections=n_sections, n_levels=n_levels)
    kwargs = {} if center is None else {"center": center}
    t0 = time.time()
    rad = solve_radiation_3d(mesh, omega * omega / 9.80665, density=1025.0,
                             gravity=9.80665, chief_points="auto", **kwargs)
    added = np.diag(np.asarray(rad.added_mass, dtype=float))
    damping = np.diag(np.asarray(rad.damping, dtype=float))
    return {"added_mass": {l: float(added[i]) for i, l in enumerate(LABELS)},
            "damping": {l: float(damping[i]) for i, l in enumerate(LABELS)},
            "panels_quads": n_sections * n_levels * 2,
            "wall_seconds": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+", default=["20x10"], help="n_sectionsxn_levels")
    ap.add_argument("--omega", type=float, default=1.0)
    ap.add_argument("--ref-level", default="n_sections=20,n_levels=10")
    ap.add_argument("--out", default="artifacts/audit_m17264/center_check.json")
    args = ap.parse_args()

    meta, ref = load_reference(level=args.ref_level)
    ref_a = {l: ref["added_mass"][l.capitalize()] for l in LABELS}
    ref_b = {l: ref["damping"][l.capitalize()] for l in LABELS}

    runs = {}
    for spec in args.levels:
        ns, nl = (int(v) for v in spec.split("x"))
        for tag, center in (("default_center_z0", None),
                            ("center_zmT2", (0.0, 0.0, -WIGLEY3_T / 2.0))):
            res = solve(ns, nl, args.omega, center)
            res["added_mass_rel_err"] = {l: (res["added_mass"][l] - ref_a[l]) / ref_a[l] for l in LABELS}
            res["damping_rel_err"] = {l: (res["damping"][l] - ref_b[l]) / ref_b[l] for l in LABELS}
            res["worst_added_mass_rel_err"] = max(abs(v) for v in res["added_mass_rel_err"].values())
            runs[f"{ns}x{nl}/{tag}"] = res
            print(f"{spec:>7} {tag:<18} "
                  f"A_roll={res['added_mass']['roll']:.4f} (ref {ref_a['roll']:.4f}, "
                  f"{100 * res['added_mass_rel_err']['roll']:+.2f}%)  "
                  f"worst|A err|={100 * res['worst_added_mass_rel_err']:.2f}%  "
                  f"{res['wall_seconds']}s", flush=True)

    payload = {
        "probe": "rotation-centre convention check for CASE-BEM-CUMMINS-WIGLEY (m17209) R1 gate",
        "omega": args.omega,
        "reference": {"path": str(REF_DEFAULT), "level": args.ref_level,
                      "metadata": meta, "added_mass": ref_a, "damping": ref_b},
        "runs": runs,
        "case_tolerance": 0.10,
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())