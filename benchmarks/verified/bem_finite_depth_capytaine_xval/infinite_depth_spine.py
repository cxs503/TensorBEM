"""Infinite-depth mesh spine: separates discretisation from the finite-depth kernel.

The automatic-mode finite-depth cross-validation (``auto_mode_xval.py``) still
leaves a residual gap to capytaine.  This producer isolates *where* the gap
comes from by running TensorBEM's own infinite-depth Rankine solver on the
same geometry, same mesh ladder (sub 1 -> 2 -> 3) and same frequencies, and
comparing it with the cached capytaine infinite-depth levels
(``cases/cpt-*-hinf-*.json``).

If the infinite-depth gap shrinks with h-independent mesh refinement in the
same way as the finite-depth gap, the discrepancy is discretisation of the
shared constant-strength collocation scheme, not the John eigenseries kernel.

Usage
-----
    python3 infinite_depth_spine.py --shard 0 3
    python3 infinite_depth_spine.py --assemble
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = HERE / "cases_inf"
OUT = HERE / "infinite_depth_spine.json"

RADIUS = 1.0
SUBMERGENCE = 2.0
DENSITY = 1025.0
KA = (0.3, 1.0, 2.0)
SUBS = (1, 2, 3)
CPT_LEVELS = ("8x16", "12x24", "16x32", "24x48")


def case_id(sub: int, ka: float) -> str:
    return f"tbem-inf-sub{sub}-ka{ka:g}"


def solve(sub: int, ka: float) -> dict:
    from tensorbem.free_surface_solver3d import solve_radiation_3d
    from tensorbem.geometry3d import submerged_sphere_mesh

    mesh = submerged_sphere_mesh(sub, radius=RADIUS, submergence=SUBMERGENCE)
    t0 = time.time()
    res = solve_radiation_3d(mesh, ka / RADIUS, modes=[0, 1, 2], density=DENSITY)
    return {
        "tool": "TensorBEM",
        "solver": "free_surface_solver3d.solve_radiation_3d (infinite depth)",
        "ka": ka,
        "subdivisions": sub,
        "n_panels": int(mesh.panel_count),
        "added_mass_heave": float(res.added_mass[2, 2]),
        "damping_heave": float(res.damping[2, 2]),
        "seconds": round(time.time() - t0, 1),
    }


def run_shard(index: int, total: int) -> None:
    CASES.mkdir(parents=True, exist_ok=True)
    grid = [(sub, ka) for sub in SUBS for ka in KA]
    for i, (sub, ka) in enumerate(grid):
        if i % total != index:
            continue
        cid = case_id(sub, ka)
        path = CASES / f"{cid}.json"
        if path.exists():
            continue
        try:
            payload = solve(sub, ka)
            payload["status"] = "ok"
        except Exception:  # noqa: BLE001
            payload = {"status": "error", "error": traceback.format_exc()}
        payload["case_id"] = cid
        path.write_text(json.dumps(payload, indent=1, sort_keys=True))
        print(cid, payload["status"], payload.get("seconds"), flush=True)


def _rel(a: float, b: float) -> float:
    return abs(a / b - 1.0)


def cpt_inf(level: str, ka: float) -> dict | None:
    path = HERE / "cases" / f"cpt-{level}-hinf-ka{ka:g}.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    if d.get("status") != "ok" or d.get("added_mass_heave") is None:
        return None
    return {"level": level,
            "added_mass_heave": d.get("added_mass_heave"),
            "damping_heave": d.get("damping_heave")}


def assemble() -> dict:
    rows = {}
    if CASES.exists():
        for path in sorted(CASES.glob("*.json")):
            d = json.loads(path.read_text())
            if d.get("status") == "ok":
                rows[d["case_id"]] = d
    table = []
    for ka in KA:
        ref = cpt_inf("24x48", ka) or cpt_inf("16x32", ka)
        for sub in SUBS:
            row = rows.get(case_id(sub, ka))
            if not row:
                continue
            entry = {"ka": ka, "subdivisions": sub,
                     "n_panels": row["n_panels"],
                     "am": row["added_mass_heave"], "b": row["damping_heave"]}
            if ref:
                entry["cpt_level"] = ref["level"]
                entry["am_rel_error"] = _rel(row["added_mass_heave"],
                                             ref["added_mass_heave"])
                entry["b_rel_error"] = _rel(row["damping_heave"],
                                            ref["damping_heave"])
            table.append(entry)
    payload = {
        "study": "TensorBEM infinite-depth mesh spine vs capytaine",
        "anchor": "XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN",
        "purpose": "separate mesh discretisation from the finite-depth kernel",
        "table": table,
        "reference": "capytaine cached cases/cpt-24x48-hinf-ka*.json",
    }
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print("wrote", OUT, "rows", len(table))
    for e in table:
        print(e)
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", nargs=2, type=int, metavar=("I", "N"))
    ap.add_argument("--assemble", action="store_true")
    args = ap.parse_args()
    if args.shard:
        run_shard(args.shard[0], args.shard[1])
    if args.assemble:
        assemble()
    return 0


if __name__ == "__main__":
    sys.exit(main())