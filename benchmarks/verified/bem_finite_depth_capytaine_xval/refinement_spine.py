#!/usr/bin/env python3
"""Mesh-refinement spine at the *automatic* mode count (XVAL anchor).

Why this exists
---------------
``auto_mode_xval.json`` compared TensorBEM against capytaine with
``mode_count=None`` (the per-row evanescent requirement) and found the residual
disagreement at icosphere sub=2 to be +4.1 % .. +5.0 % on A33 and +7.4 % ..
+8.6 % on B33 -- i.e. the mode cap explains *part* of the pinned cross-validation
bias but not all of it.  The open question is therefore:

    is the residual a *discretisation* error that vanishes under refinement,
    or a solver/kernel bias that does not?

A two-level comparison (sub 1 -> sub 2) cannot answer that: an error that halves
could be first-order or could be a bias with a large first-order term.  This
producer adds the **third level** on a refinement spine, so the observed order of
convergence ``p`` and the Richardson-extrapolated limit ``A_inf`` can be computed
and compared with the capytaine reference.  The verdict is arithmetic, not prose.

Levels
------
``sub`` 1/2/3 -> 80/320/1280 panels, panel edge halving exactly, so

    p  = log2(|A_2 - A_1| / |A_3 - A_2|)
    A∞ = A_3 + (A_3 - A_2) / (2**p - 1)

Usage
-----
    python3 refinement_spine.py --shard I/N    # one slice of the spine
    python3 refinement_spine.py --assemble     # -> refinement_spine.json

The sub 1 and sub 2 payloads are read from ``cases_auto/`` when they are already
on disk (identical solve call), so only the new sub 3 solves are submitted.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AUTO_CASES = HERE / "cases_auto"
CASES = HERE / "cases_refine"
CPT_CASES = HERE / "cases"
OUT = HERE / "refinement_spine.json"

RADIUS = 1.0
SUBMERGENCE = 2.0
DENSITY = 1025.0

#: (h/R, ka) stations of the refinement spine -- h/R = 5 is the station where the
#: sub 1 -> sub 2 step is already measured (11.5 % -> 5.0 % on A33), so the third
#: level is the informative one.  h/R = 12 sub 3 costs >2 h and is left out.
STATIONS: tuple[tuple[float, float], ...] = ((5.0, 0.3), (5.0, 1.5))
LEVELS: tuple[int, ...] = (1, 2, 3)
TASKS: tuple[tuple[float, float, int], ...] = tuple(
    (h, ka, sub) for (h, ka) in STATIONS for sub in LEVELS
)


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
        "case_id": case_id(depth_ratio, ka, sub),
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
    for i, (h, ka, sub) in enumerate(TASKS):
        if i % total != index:
            continue
        cid = case_id(h, ka, sub)
        path = CASES / f"{cid}.json"
        if path.exists():
            continue
        # sub 1 / sub 2 are already frozen in cases_auto/ (same solve call);
        # ``assemble`` falls back to them, so only the new levels are solved.
        if (AUTO_CASES / f"{cid}.json").exists():
            print(cid, "skip (cached in cases_auto)", flush=True)
            continue
        try:
            payload = solve(h, ka, sub)
            payload["status"] = "ok"
        except Exception:  # noqa: BLE001
            payload = {"case_id": cid, "status": "error",
                       "error": traceback.format_exc()}
        path.write_text(json.dumps(payload, indent=1, sort_keys=True))
        print(cid, payload["status"], payload.get("seconds"), flush=True)


def _cpt(depth_ratio: float, ka: float, level: str = "24x48") -> dict | None:
    path = CPT_CASES / f"cpt-{level}-h{depth_ratio:g}-ka{ka:g}.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    if d.get("status") != "ok":
        return None
    return {"level": level,
            "added_mass_heave": float(d["added_mass_heave"]),
            "damping_heave": float(d["damping_heave"])}


def _observed_order(a: float, b: float, c: float) -> float | None:
    """p for the triple (coarse, medium, fine) with a factor-2 edge halving."""
    d1 = abs(b - a)
    d2 = abs(c - b)
    if d2 <= 0.0:
        return None
    if d1 <= 0.0:
        return None
    return float(math.log2(d1 / d2))


def _richardson(b: float, c: float, p: float | None) -> float | None:
    if p is None or not math.isfinite(p) or 2.0 ** p - 1.0 == 0.0:
        return None
    return float(c + (c - b) / (2.0 ** p - 1.0))


def _rel(a: float, b: float) -> float:
    return float(abs(a / b - 1.0))


def _load(h: float, ka: float, sub: int) -> dict | None:
    cid = case_id(h, ka, sub)
    for base in (CASES, AUTO_CASES):
        path = base / f"{cid}.json"
        if path.exists():
            d = json.loads(path.read_text())
            if d.get("status") == "ok":
                return d
    return None


def assemble() -> dict:
    stations = []
    for (h, ka) in STATIONS:
        pts = [_load(h, ka, sub) for sub in LEVELS]
        have = [(sub, p) for sub, p in zip(LEVELS, pts) if p is not None]
        entry: dict = {"depth_ratio": h, "ka": ka,
                       "levels_present": [sub for sub, _ in have],
                       "points": [
                           {"subdivisions": sub,
                            "n_panels": p["n_panels"],
                            "mode_count": p["mode_count"],
                            "mode_cap_bound_rows": p["mode_cap_bound_rows"],
                            "added_mass_heave": p["added_mass_heave"],
                            "damping_heave": p["damping_heave"]}
                           for sub, p in have]}
        ref = _cpt(h, ka)
        entry["cpt"] = ref
        for sub, p in have:
            row = next(r for r in entry["points"] if r["subdivisions"] == sub)
            if ref:
                row["am_rel_error_vs_cpt"] = _rel(p["added_mass_heave"],
                                                  ref["added_mass_heave"])
                row["b_rel_error_vs_cpt"] = _rel(p["damping_heave"],
                                                 ref["damping_heave"])
        if len(have) == 3:
            (_, a), (_, b), (_, c) = have
            p_am = _observed_order(a["added_mass_heave"], b["added_mass_heave"],
                                   c["added_mass_heave"])
            p_b = _observed_order(a["damping_heave"], b["damping_heave"],
                                  c["damping_heave"])
            am_inf = _richardson(b["added_mass_heave"], c["added_mass_heave"], p_am)
            b_inf = _richardson(b["damping_heave"], c["damping_heave"], p_b)
            entry["observed_order"] = {"am": p_am, "b": p_b}
            entry["extrapolated"] = {"added_mass_heave": am_inf,
                                     "damping_heave": b_inf}
            if ref and am_inf is not None:
                entry["extrapolated_am_rel_error_vs_cpt"] = _rel(
                    am_inf, ref["added_mass_heave"])
            if ref and b_inf is not None:
                entry["extrapolated_b_rel_error_vs_cpt"] = _rel(
                    b_inf, ref["damping_heave"])
            entry["finest_am_rel_error_vs_cpt"] = _rel(
                c["added_mass_heave"], ref["added_mass_heave"]) if ref else None
            entry["finest_b_rel_error_vs_cpt"] = _rel(
                c["damping_heave"], ref["damping_heave"]) if ref else None
        stations.append(entry)

    # verdicts: is the residual discretisation (error shrinks with p ~ 1-2 and
    # the extrapolated limit lands on the reference) or a bias (extrapolated
    # limit stays off the reference by the sub-2 margin)?
    verdicts = []
    for st in stations:
        if "extrapolated_am_rel_error_vs_cpt" not in st:
            verdicts.append({
                "depth_ratio": st["depth_ratio"], "ka": st["ka"],
                "status": "incomplete",
                "levels_present": st["levels_present"]})
            continue
        verdicts.append({
            "depth_ratio": st["depth_ratio"], "ka": st["ka"],
            "status": "complete",
            "observed_order_am": st["observed_order"]["am"],
            "observed_order_b": st["observed_order"]["b"],
            "finest_am_rel_error": st["finest_am_rel_error_vs_cpt"],
            "extrapolated_am_rel_error": st["extrapolated_am_rel_error_vs_cpt"],
            "finest_b_rel_error": st["finest_b_rel_error_vs_cpt"],
            "extrapolated_b_rel_error": st["extrapolated_b_rel_error_vs_cpt"],
            "discretisation_explains_residual": bool(
                st["extrapolated_am_rel_error_vs_cpt"]
                < 0.5 * st["finest_am_rel_error_vs_cpt"]),
        })
    payload = {
        "study": "finite-depth 3-D mesh refinement spine at automatic mode count",
        "anchor": "XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN",
        "reference": "capytaine cases/cpt-24x48-h{h}-ka{ka}.json",
        "geometry": {"type": "submerged_sphere", "radius_m": RADIUS,
                     "centre_z_m": -SUBMERGENCE, "density": DENSITY},
        "levels": list(LEVELS),
        "stations": stations,
        "verdicts": verdicts,
    }
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print("wrote", OUT, "stations", len(stations),
          "complete", sum(v["status"] == "complete" for v in verdicts))
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
    raise SystemExit(main())