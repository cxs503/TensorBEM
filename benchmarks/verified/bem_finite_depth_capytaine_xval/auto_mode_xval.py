"""Automatic-mode cross-validation against capytaine (finite depth).

Why this script exists
----------------------
The frozen cross-validation study in ``cases/`` pinned ``mode_count`` to
48/64/96 while the eigenseries rows actually need 271-1791 modes at
subdivision 2 (4294 at h/R = 12 before the ``MODES_CAP = 4096`` ceiling).
Every one of those rows was therefore truncated, and the resulting cross-code
errors (A33 12.2 %, B33 34.0 %) were measuring the truncation artefact, not
the discretisation.

``finite_depth_solver3d`` now derives the mode count per row when
``mode_count=None`` (``_row_mode_requirements``); this producer re-runs the
same physical grid with the automatic count so gate G1/G2 can be evaluated on
numbers that are not cap-bound, and adds a mesh-refinement spine
(sub 1 -> 2 -> 3) so truncation error and discretisation error can be told
apart.

Grid
----
* ``h/R in {5, 12}`` x ``ka in {0.3, 0.6, 1.0, 1.5, 2.0}`` at sub 1 and 2
* the refinement spine ``ka = 1.0`` at sub 3 for both depths

Reference: cached capytaine runs in ``cases/`` (8x16, 12x24, 16x32, 24x48
panelled spheres), so the capytaine leg is not recomputed here.

Usage
-----
    python3 auto_mode_xval.py --shard 0 6      # worker
    python3 auto_mode_xval.py --requirements   # mesh-only mode budget
    python3 auto_mode_xval.py --assemble       # merge + gates -> json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CASES = HERE / "cases_auto"
OUT = HERE / "auto_mode_xval.json"
REQ_OUT = HERE / "mode_requirements.json"

RADIUS = 1.0
SUBMERGENCE = 2.0
DENSITY = 1025.0
KA = (0.3, 0.6, 1.0, 1.5, 2.0)
DEPTHS = (5.0, 12.0)

GRID: tuple[tuple[float, float, int], ...] = tuple(
    (h, ka, sub) for h in DEPTHS for ka in KA for sub in (1, 2)
) + ((5.0, 1.0, 3), (12.0, 1.0, 3))

#: capytaine levels, coarse -> fine; the finest cached level is the reference
CPT_LEVELS = ("8x16", "12x24", "16x32", "24x48")


def case_id(depth_ratio: float, ka: float, sub: int) -> str:
    return f"tbem-auto-sub{sub}-h{depth_ratio:g}-ka{ka:g}"


def solve(depth_ratio: float, ka: float, sub: int) -> dict:
    from tensorbem.finite_depth_solver3d import (
        _row_mode_requirements,
        solve_radiation_finite_depth_3d,
    )
    from tensorbem.geometry3d import submerged_sphere_mesh

    mesh = submerged_sphere_mesh(sub, radius=RADIUS, submergence=SUBMERGENCE)
    depth = depth_ratio * RADIUS
    need = _row_mode_requirements(mesh, depth)
    t0 = time.time()
    res = solve_radiation_finite_depth_3d(
        mesh, ka / RADIUS, depth, modes=[0, 1, 2], density=DENSITY
    )
    return {
        "tool": "TensorBEM",
        "depth_ratio": depth_ratio,
        "ka": ka,
        "subdivisions": sub,
        "n_panels": int(mesh.panel_count),
        "mode_count": int(res.mode_count),
        "mode_requirement_max": int(need.max()),
        "mode_cap_bound_rows": int(np.count_nonzero(need > res.mode_count)),
        "added_mass_heave": float(res.added_mass[2, 2]),
        "damping_heave": float(res.damping[2, 2]),
        "added_mass_surge": float(res.added_mass[0, 0]),
        "damping_surge": float(res.damping[0, 0]),
        "condition_number": float(res.condition_number),
        "relative_residual": float(res.relative_residual),
        "seconds": round(time.time() - t0, 1),
    }


def run_shard(index: int, total: int) -> None:
    CASES.mkdir(parents=True, exist_ok=True)
    for i, (h, ka, sub) in enumerate(GRID):
        if i % total != index:
            continue
        cid = case_id(h, ka, sub)
        path = CASES / f"{cid}.json"
        if path.exists():
            continue
        try:
            payload = solve(h, ka, sub)
            payload["status"] = "ok"
        except Exception:  # noqa: BLE001
            payload = {"status": "error", "error": traceback.format_exc()}
        payload["case_id"] = cid
        path.write_text(json.dumps(payload, indent=1, sort_keys=True))
        print(cid, payload["status"], payload.get("seconds"), flush=True)


def write_requirements() -> dict:
    """Mesh-only mode budget: how many rows are bound by MODES_CAP itself."""
    from tensorbem import finite_depth_solver3d as fd
    from tensorbem.geometry3d import submerged_sphere_mesh

    table = []
    for sub in (1, 2, 3):
        mesh = submerged_sphere_mesh(sub, radius=RADIUS, submergence=SUBMERGENCE)
        qpts, _, _ = fd.quadrature_points(mesh)
        horiz = np.hypot(
            mesh.centroids[:, None, 0] - qpts[None, :, 0],
            mesh.centroids[:, None, 1] - qpts[None, :, 1],
        )
        with np.errstate(invalid="ignore"):
            row_min = np.where(horiz > 0.0, horiz, np.inf).min(axis=1)
        for h in DEPTHS:
            raw = np.array([
                int(math.ceil(fd._MODE_DECAY * (h * RADIUS)
                              / (math.pi * max(float(r), 1.0e-9)))) + 2
                for r in row_min
            ], dtype=np.int64)
            table.append({
                "subdivisions": sub,
                "n_panels": int(mesh.panel_count),
                "depth_ratio": h,
                "raw_min": int(raw.min()),
                "raw_median": int(np.median(raw)),
                "raw_max": int(raw.max()),
                "modes_cap": int(fd._MODES_CAP),
                "ceiling_bound_rows": int(np.count_nonzero(raw >= fd._MODES_CAP)),
            })
    payload = {
        "study": "mode budget of the finite-depth eigenseries rows",
        "anchor": "XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN",
        "mode_decay": float(fd._MODE_DECAY),
        "table": table,
        "note": "raw = uncapped requirement; ceiling_bound_rows hit MODES_CAP",
    }
    REQ_OUT.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print("wrote", REQ_OUT)
    for row in table:
        print(row)
    return payload


def load_rows() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not CASES.exists():
        return rows
    for path in sorted(CASES.glob("*.json")):
        d = json.loads(path.read_text())
        if d.get("status") == "ok":
            rows[d["case_id"]] = d
    return rows


def cpt_reference(depth_ratio: float, ka: float, level: str = "24x48") -> dict | None:
    path = HERE / "cases" / f"cpt-{level}-h{depth_ratio:g}-ka{ka:g}.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    if d.get("status") != "ok" or d.get("added_mass_heave") is None:
        return None
    return {"level": level,
            "added_mass_heave": d.get("added_mass_heave"),
            "damping_heave": d.get("damping_heave")}


def _rel(a: float, b: float) -> float:
    return abs(a / b - 1.0)


def assemble() -> dict:
    rows = load_rows()

    # ---- capytaine reference ladder (is the reference itself converged?) ---
    cpt_ladder = []
    for h in DEPTHS:
        for ka in KA:
            levels = {lv: cpt_reference(h, ka, lv) for lv in CPT_LEVELS}
            levels = {k: v for k, v in levels.items() if v is not None}
            if not levels:
                continue
            fine_key = "24x48" if "24x48" in levels else sorted(levels)[-1]
            fine = levels[fine_key]
            prev = levels.get("16x32") or levels.get("12x24")
            cpt_ladder.append({
                "depth_ratio": h, "ka": ka,
                "levels": levels,
                "fine_level": fine_key,
                "self_conv_am": None if not prev else _rel(
                    prev["added_mass_heave"], fine["added_mass_heave"]),
                "self_conv_b": None if not prev else _rel(
                    prev["damping_heave"], fine["damping_heave"]),
            })

    # ---- cross-code comparison for every TensorBEM row -------------------
    comparison = []
    for cid in sorted(rows):
        row = rows[cid]
        h, ka, sub = row["depth_ratio"], row["ka"], row["subdivisions"]
        ref = cpt_reference(h, ka)
        entry = {
            "case_id": cid, "depth_ratio": h, "ka": ka, "subdivisions": sub,
            "n_panels": row["n_panels"], "mode_count": row["mode_count"],
            "mode_cap_bound_rows": row["mode_cap_bound_rows"],
            "am": row["added_mass_heave"], "b": row["damping_heave"],
            "cpt": ref,
        }
        if ref:
            entry["am_rel_error"] = _rel(row["added_mass_heave"],
                                        ref["added_mass_heave"])
            entry["b_rel_error"] = _rel(row["damping_heave"],
                                       ref["damping_heave"])
        comparison.append(entry)

    at_sub2 = [c for c in comparison if c["subdivisions"] == 2]
    am_errs = [c["am_rel_error"] for c in at_sub2 if "am_rel_error" in c]
    b_errs = [c["b_rel_error"] for c in at_sub2 if "b_rel_error" in c]

    # ---- mesh trend: same (h, ka) at increasing subdivision --------------
    trend = []
    for h in DEPTHS:
        for ka in KA:
            pts = [c for c in comparison
                   if c["depth_ratio"] == h and c["ka"] == ka]
            pts.sort(key=lambda c: c["subdivisions"])
            if len(pts) < 2:
                continue
            steps = []
            for a, b in zip(pts, pts[1:]):
                steps.append({
                    "from_sub": a["subdivisions"], "to_sub": b["subdivisions"],
                    "am_rel_change": _rel(a["am"], b["am"]),
                    "b_rel_change": _rel(a["b"], b["b"]),
                })
            trend.append({"depth_ratio": h, "ka": ka, "points": pts,
                          "steps": steps})

    ledger_tol = 0.15
    gates = {
        "G1_tbem_auto_vs_cpt_am_sub2": {
            "threshold": 0.10, "ledger_threshold": ledger_tol,
            "n": len(am_errs),
            "max_error": max(am_errs) if am_errs else None,
            "passed": bool(am_errs) and max(am_errs) <= 0.10,
        },
        "G2_tbem_auto_vs_cpt_b_sub2": {
            "threshold": 0.15, "ledger_threshold": ledger_tol,
            "n": len(b_errs),
            "max_error": max(b_errs) if b_errs else None,
            "passed": bool(b_errs) and max(b_errs) <= 0.15,
        },
        "G4_tbem_mesh_self_conv_am_sub1_to_sub2": {
            "threshold": 0.05,
            "n": sum(1 for t in trend
                     if t["steps"][0]["from_sub"] == 1),
            "max_error": max(
                (t["steps"][0]["am_rel_change"] for t in trend
                 if t["steps"][0]["from_sub"] == 1), default=None),
            "passed": bool(trend) and all(
                t["steps"][0]["am_rel_change"] <= 0.05 for t in trend
                if t["steps"][0]["from_sub"] == 1),
        },
    }
    ledger_verdict = {
        "A33_max_rel_error": max(am_errs) if am_errs else None,
        "B33_max_rel_error": max(b_errs) if b_errs else None,
        "tolerance": ledger_tol,
        "meets_ledger": bool(am_errs and b_errs)
        and max(am_errs) <= ledger_tol and max(b_errs) <= ledger_tol,
    }
    payload = {
        "study": "finite-depth 3-D cross-validation, automatic mode count",
        "anchor": "XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN",
        "geometry": {"type": "submerged_sphere", "radius_m": RADIUS,
                     "centre_z_m": -SUBMERGENCE, "density": DENSITY},
        "grid": {"depth_ratios": list(DEPTHS), "ka": list(KA),
                 "subdivisions": [1, 2, 3],
                 "note": "sub 3 only on the ka = 1.0 refinement spine"},
        "rows": [rows[k] for k in sorted(rows)],
        "comparison": comparison,
        "cpt_self_convergence": cpt_ladder,
        "mesh_trend": trend,
        "gates": gates,
        "ledger_verdict": ledger_verdict,
        "reference": "capytaine cached in cases/ (24x48 = 1152 faces is the fine level)",
    }
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print("wrote", OUT, "rows", len(rows))
    for c in comparison:
        print(c["case_id"], c["n_panels"], "M", c["mode_count"],
              "am_err %.4f" % c.get("am_rel_error", float("nan")),
              "b_err %.4f" % c.get("b_rel_error", float("nan")))
    print(json.dumps({k: {"passed": v["passed"],
                          "max_error": v["max_error"],
                          "n": v["n"]} for k, v in gates.items()}, indent=1))
    print(json.dumps(ledger_verdict, indent=1))
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", nargs=2, type=int, metavar=("I", "N"))
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--requirements", action="store_true")
    args = ap.parse_args()
    if args.shard:
        run_shard(args.shard[0], args.shard[1])
    if args.requirements:
        write_requirements()
    if args.assemble:
        assemble()
    return 0


if __name__ == "__main__":
    sys.exit(main())