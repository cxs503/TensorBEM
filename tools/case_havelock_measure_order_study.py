"""Grid-convergence order study for CASE-BEM-SUBMERGED-SPHERE-HAVELOCK (round 4).

Why this study exists (the debt it pays)
----------------------------------------
Round 3 (``docs/case_submerged_sphere_havelock_round3.md`` §3.3/§3.4/§6) closed the
full ``ka`` band of the case pool entry under the *spherical-patch* area measure and
judged it PASS (320 panels, 28/28 items < 5 %, worst 2.253 %), but declared two gaps
in the same breath:

* the 1280-panel refinement **under the spherical-patch measure was never produced**
  (``refinement_status = "not_completed"``), so nothing showed that the 320-panel
  residual is discretisation error rather than a defect of the reference series;
* consequently the round-3 note "the patch is a first-order geometric correction, so
  the error still falls with h at first order" was an *assumption*, not a measurement.

An earlier V&V study (``benchmarks/verified/casepool_bem_submerged_sphere_havelock/
gci_continuum.py``) did fit an observed order and Richardson-extrapolate -- but on the
**flat-triangle** measure only, which is not the measure the round-3 verdict is scored
on.  Scoring a case on one measure and diagnosing convergence on another is exactly
the kind of evidence split this round removes.

Question (fixed before any number was produced)
-----------------------------------------------
On the measure actually used for the verdict, (a) does the BEM solution converge at a
measurable order, (b) does its Richardson continuum limit land on the Havelock
multipole reference (i.e. is the reference -- not the mesh -- right), and (c) does the
spherical-patch measure *raise the order* relative to the flat measure, or only
*reduce the error constant*?  (c) is a genuine either/or: the patch is an area-weight
correction, and a correction that only rescales areas cannot buy an order.

Predeclared experiment
----------------------
Grids (nested, refinement ratio r = 2 in h, panels x4 per rung)
    subdivision 0 / 1 / 2 / 3  =  20 / 80 / 320 / 1280 panels (production = 2);
    the order is fitted on the **finest three available rungs** (asymptotic range
    lives at the fine end), so an unavailable rung degrades coverage, not honesty.
Measures
    ``spherical_patch`` (verdict measure, round 3) and ``flat`` (round-1 control),
    both evaluated on the *same* grids so the comparison is paired.
Band
    ka in {0.1, 0.5, 1.0, 2.0} -- the low/mid band where the case's worst items live.
Quantities
    A33' = A33/(rho*nabla), B33' = B33/(rho*omega*nabla), A11', B11'
    (unit velocity amplitude; nabla = 4*pi*a^3/3 exact).
Reference
    tensorbem.analytic.submerged_sphere_wave_multipole (Havelock 1955 multipole;
    in-repo, truncation 20 terms; its own uncertainty was measured at < 1e-9 by
    submerged_sphere_havelock_integrity, so it does not mask mesh error).

Gates (predeclared)
    G1 order admissibility
        For every cell whose successive difference is resolvable
        (|f3-f2| > 1e-10 |f3|), the observed order p must satisfy 0.5 <= p <= 3.0.
        A p outside that band means the ladder is not in its asymptotic range and the
        Richardson estimate is not admissible for that cell (reported as such, never
        dropped).
    G2 continuum limit validates the reference
        |f_richardson - f_ref| / |f_ref| <= 2 % on the spherical-patch measure for
        every (ka, quantity).  This is the twin of the earlier flat-measure gate and
        is the direct answer to "who is wrong".
    G3 monotone refinement under the verdict measure
        error(sub3) < error(sub2) for every (ka, quantity) on the spherical-patch
        measure -- refinement must not be noise.
    H1 patch-vs-flat hypothesis (either/or, may be falsified)
        SUPPORTED if median(p_patch) >= median(p_flat) + 0.5 over resolved cells;
        otherwise FALSIFIED, and the honest reading is "the patch reduces the constant
        only".  Both outcomes are publishable; the gate is written so that a FAIL is
        not a crisis but a finding.

Exit codes / artefacts
    ``runs/m21520/ladder_cells.json``   raw per-cell numbers (zero hand-copying)
    ``runs/m21520/ladder_study.json``   orders, Richardson limits, GCI, gate verdicts

Usage
-----
    PYTHONPATH=src python3 tools/case_havelock_measure_order_study.py --stage solve
    PYTHONPATH=src python3 tools/case_havelock_measure_order_study.py --stage analyse
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from tensorbem.benchmarks.submerged_sphere_havelock import (
    AREA_MEASURE_FLAT,
    AREA_MEASURE_SPHERICAL,
    panel_area_measure,
    run_frequency_sweep,
    submerged_sphere_mesh,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "m21520"
CELLS = OUT / "ladder_cells.json"
STUDY = OUT / "ladder_study.json"

RADIUS, SUBMERGENCE, DENSITY, GRAVITY = 1.0, 2.0, 1025.0, 9.80665
SUBDIVISIONS = (0, 1, 2, 3)   # 20 / 80 / 320 / 1280 panels
KA_BAND = (0.1, 0.5, 1.0, 2.0)
REFINEMENT_RATIO = 2.0          # h halves between consecutive rungs (panels x4)
MEASURES = (AREA_MEASURE_FLAT, AREA_MEASURE_SPHERICAL)
QUANTITIES = ("a33", "b33", "a11", "b11")
SAFETY_FACTOR = 1.25            # Roache / Celik GCI safety factor (95 %)

G1_ORDER_BAND = (0.5, 3.0)
G2_CONTINUUM_TOL = 0.02         # 2 %
H1_ORDER_MARGIN = 0.5

_CALL = {"dt": 0.0}


def _solve_cell(args):
    """One (measure, subdivision, ka) cell -- identical arithmetic per cell."""
    measure, subdivision, ka = args
    start = time.time()
    row = run_frequency_sweep(
        ka_values=(ka,), radius=RADIUS, submergence=SUBMERGENCE,
        density=DENSITY, gravity=GRAVITY,
        subdivision=int(subdivision), area_measure=measure,
    )[0]
    row["wall_seconds"] = time.time() - start
    row["subdivision"] = int(subdivision)
    return row


def _mesh_area_diagnostic(subdivision: int) -> dict:
    """Total wetted-panel area under each measure vs the exact sphere area.

    The flat measure sums the areas of the *inscribed* triangles; a curved panel
    always loses area, so its total is short of 4*pi*a^2 by a factor that must
    shrink with h.  This is the mechanism the round-2/3 write-ups blamed for the
    flat-measure failures, so it is measured here rather than asserted.
    """
    mesh = submerged_sphere_mesh(subdivision, radius=RADIUS, submergence=SUBMERGENCE)
    exact = 4.0 * math.pi * RADIUS**2
    block = {"subdivision": int(subdivision), "n_panels": int(mesh.panel_count),
             "exact_area": exact,
             "h_mean": float(mesh.h_mean), "h_max": float(mesh.h_max)}
    for measure in MEASURES:
        areas = panel_area_measure(mesh, radius=RADIUS, submergence=SUBMERGENCE,
                                  area_measure=measure)
        # "flat" maps to None in the library (the mesh's own chord measure), so the
        # control is the sum of the inscribed triangle areas itself.
        total = float(np.sum(mesh.areas if areas is None else areas))
        block[measure] = {"total_area": total,
                          "deficit_ratio": 1.0 - total / exact}
    return block


def stage_solve(processes: int = 8, subdivisions=SUBDIVISIONS, merge: bool = True,
                serial: bool = False) -> None:
    """Solve the requested ladder rungs, one process per cell, store the raw table.

    ``subdivisions`` lets the ladder be produced in chunks (the cluster queue can be
    long, and a chunked run keeps every chunk inside the local short-test envelope);
    with ``merge`` an existing ``ladder_cells.json`` is extended instead of replaced,
    so chunks compose into one table and re-solving a cell overwrites it cleanly.
    """
    tasks = [(measure, sub, ka) for measure in MEASURES
             for sub in subdivisions for ka in KA_BAND]
    OUT.mkdir(parents=True, exist_ok=True)
    def _persist(rows_so_far):
        """Write the merged table now: an interrupted chunk must not lose solved cells."""
        prev = []
        if merge and CELLS.exists():
            prev = json.loads(CELLS.read_text()).get("rows", [])
        merged = {(r["area_measure"], int(r["subdivision"]), float(r["ka"])): r
                  for r in prev}
        for r in rows_so_far:
            merged[(r["area_measure"], int(r["subdivision"]), float(r["ka"]))] = r
        payload = {"case": "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK", "round": 4,
                   "subdivisions": list(SUBDIVISIONS), "ka_band": list(KA_BAND),
                   "measures": list(MEASURES),
                   "mesh_area_diagnostic": [_mesh_area_diagnostic(sd)
                                            for sd in sorted({int(r["subdivision"])
                                                              for r in merged.values()})],
                   "rows": [merged[k] for k in sorted(merged)]}
        CELLS.parent.mkdir(parents=True, exist_ok=True)
        CELLS.write_text(json.dumps(payload, indent=2, default=float))

    start = time.time()
    if serial or processes <= 1:
        # Serial path: used when the scheduler is congested and the ladder is solved in
        # small chunks inside the local short-test envelope.  Identical arithmetic.
        rows = []
        done = set()
        if CELLS.exists():
            done = {(r["area_measure"], int(r["subdivision"]), float(r["ka"]))
                    for r in json.loads(CELLS.read_text()).get("rows", [])}
        for task in tasks:
            if (task[0], int(task[1]), float(task[2])) in done:
                continue
            row = _solve_cell(task)
            rows.append(row)
            print("cell %-16s sub=%d ka=%.1f  panels=%d  %.1fs"
                  % (task[0], task[1], task[2], row["n_panels"], row["wall_seconds"]),
                  flush=True)
            _persist(rows)
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(processes, len(tasks))) as pool:
            rows = pool.map(_solve_cell, tasks)
    previous = []
    if merge and CELLS.exists():
        previous = json.loads(CELLS.read_text()).get("rows", [])
    fresh = {(row["area_measure"], int(row["subdivision"]), float(row["ka"])): row
             for row in previous}
    for row in rows:
        fresh[(row["area_measure"], int(row["subdivision"]), float(row["ka"]))] = row
    all_rows = [fresh[key] for key in sorted(fresh)]
    cells = {
        "case": "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
        "round": 4,
        "geometry": {"radius": RADIUS, "submergence": SUBMERGENCE,
                     "submergence_ratio": SUBMERGENCE / RADIUS,
                     "density": DENSITY, "gravity": GRAVITY},
        "subdivisions": list(SUBDIVISIONS), "ka_band": list(KA_BAND),
        "measures": list(MEASURES),
        "mesh_area_diagnostic": [_mesh_area_diagnostic(sub) for sub in SUBDIVISIONS],
        "wall_seconds_total": time.time() - start,
        "rows": all_rows,
        "rows_completed": len(all_rows),
        "expected_rows": len(MEASURES) * len(SUBDIVISIONS) * len(KA_BAND),
    }
    CELLS.write_text(json.dumps(cells, indent=2, default=float))
    print(json.dumps({"artifact": str(CELLS), "cells_this_chunk": len(rows),
                      "rows_completed": len(all_rows),
                      "expected_rows": cells["expected_rows"],
                      "wall_seconds": round(cells["wall_seconds_total"], 1),
                      "panels": sorted({row["n_panels"] for row in rows})}, indent=2))


# ---------------------------------------------------------------------------
# V&V kernels (pure; unit-tested)
# ---------------------------------------------------------------------------

def order_regime(coarse: float, medium: float, fine: float) -> str:
    """Classify a three-grid triple before any order is claimed.

    ``asymptotic``   successive differences share a sign and are resolvable -> p
                     is meaningful;
    ``non_monotone`` the approach oscillates, so no admissible order exists (the
                     convention already used by ``gci_continuum.observed_order``);
    ``at_roundoff``  the fine difference is at the level of the values themselves,
                     i.e. the finest rung has nothing left to say.
    """
    d1 = medium - coarse
    d2 = fine - medium
    scale = max(abs(coarse), abs(medium), abs(fine), 1e-300)
    if abs(d2) <= 1e-13 * scale or abs(d1) <= 1e-13 * scale:
        return "at_roundoff"
    if d1 * d2 <= 0.0:
        return "non_monotone"
    return "asymptotic"


def observed_order(coarse: float, medium: float, fine: float, r: float) -> float:
    """p from three successive grid values, f(h) = f* + C h^p (Roache eq. 5).

    p = ln(|f_m - f_c| / |f_f - f_m|) / ln r, with the in-repo admissibility rule:
    a non-monotone or round-off-limited ladder has **no** admissible order -> nan
    (this matches ``benchmarks/verified/.../gci_continuum.observed_order``, and the
    agreement is asserted in tests/test_case_havelock_measure_order.py).
    """
    if order_regime(coarse, medium, fine) != "asymptotic":
        return float("nan")
    ratio = (medium - coarse) / (fine - medium)
    return math.log(abs(ratio)) / math.log(r)


def richardson_continuum(medium: float, fine: float, p: float, r: float) -> float:
    """Richardson extrapolation of the fine grid to h -> 0."""
    if not math.isfinite(p) or p <= 0.0:
        return float("nan")
    return fine + (fine - medium) / (r**p - 1.0)


def gci_fine(medium: float, fine: float, p: float, r: float,
             safety: float = SAFETY_FACTOR) -> float:
    """Grid convergence index of the fine grid (Celik et al. 2008, Fs = 1.25)."""
    if not math.isfinite(p) or p <= 0.0:
        return float("nan")
    if abs(fine) == 0.0:
        return float("nan")
    return safety * abs((fine - medium) / fine) / (r**p - 1.0)


def _cells_by_key(rows):
    """(measure, ka) -> {subdivision: cell row}, the ladder rungs in order."""
    table: dict[tuple[str, float], dict[int, dict]] = {}
    for row in rows:
        key = (row["area_measure"], float(row["ka"]))
        table.setdefault(key, {})[int(row["subdivision"])] = row
    return table


def assemble() -> dict:
    payload = json.loads(CELLS.read_text())
    rows = payload["rows"]
    table = _cells_by_key(rows)
    available = {int(row["subdivision"]) for row in rows}
    declared = sorted(int(s) for s in payload["subdivisions"] if int(s) in available)
    if len(declared) < 3:
        raise SystemExit("a ladder needs >= 3 rungs; have %r -- solve more chunks"
                         % (declared,))
    subdivisions = declared[-3:]          # finest trio: asymptotic range lives there

    cells = []
    for (measure, ka), rung in sorted(table.items()):
        rungs = [rung.get(sub) for sub in subdivisions]
        assert all(rungs), "missing rung in ladder: %r" % ((measure, ka),)
        coarse, medium, fine = rungs
        for quantity in QUANTITIES:
            f_c = coarse["%s_bem" % quantity]
            f_m = medium["%s_bem" % quantity]
            f_f = fine["%s_bem" % quantity]
            ref = fine["%s_exact" % quantity]
            regime = order_regime(f_c, f_m, f_f)
            p = observed_order(f_c, f_m, f_f, REFINEMENT_RATIO)
            f_star = richardson_continuum(f_m, f_f, p, REFINEMENT_RATIO)
            gci = gci_fine(f_m, f_f, p, REFINEMENT_RATIO)
            err_fine = abs(f_f - ref) / abs(ref) if ref else float("nan")
            err_extrap = abs(f_star - ref) / abs(ref) if ref else float("nan")
            cells.append({
                "measure": measure, "ka": ka, "quantity": quantity,
                "panels": [coarse["n_panels"], medium["n_panels"], fine["n_panels"]],
                "values": [f_c, f_m, f_f],
                "reference": ref,
                "successive_differences": [abs(f_m - f_c), abs(f_f - f_m)],
                "order_regime": regime,
                "resolved": bool(regime == "asymptotic"),
                "observed_order": p,
                "richardson_continuum": f_star,
                "gci_fine": gci,
                "err_fine_vs_ref": err_fine,
                "err_continuum_vs_ref": err_extrap,
                "err_medium_vs_ref": (abs(f_m - ref) / abs(ref)) if ref else float("nan"),
                "err_coarse_vs_ref": (abs(f_c - ref) / abs(ref)) if ref else float("nan"),
            })

    gates = _gates(cells)
    study = {
        "case": "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
        "round": 4,
        "purpose": ("close round-3's declared gaps: 1280-panel refinement under the "
                    "verdict measure, and a measured (not assumed) observed order plus "
                    "a Richardson continuum limit checked against the multipole reference"),
        "geometry": payload["geometry"],
        "refinement_ratio_h": REFINEMENT_RATIO,
        "subdivisions_available": declared,
        "subdivisions_used": subdivisions,
        "panels_per_rung": [table[(MEASURES[0], KA_BAND[0])][sub]["n_panels"]
                            for sub in subdivisions],
        "ka_band": payload["ka_band"],
        "gates_predeclared": {
            "G1_order_band": list(G1_ORDER_BAND),
            "G2_continuum_tol": G2_CONTINUUM_TOL,
            "G3_monotone": "err(1280) < err(320) for every (ka, quantity), patch measure",
            "H1_order_margin": H1_ORDER_MARGIN,
        },
        "mesh_area_diagnostic": payload["mesh_area_diagnostic"],
        "cells": cells,
        "gates": gates,
        "wall_seconds_total": payload.get("wall_seconds_total"),
    }
    bad = _find_nan_except_orders(study)
    if bad:
        print("NOTE: non-finite values kept on purpose (unresolved orders): %d" % len(bad))
    OUT.mkdir(parents=True, exist_ok=True)
    STUDY.write_text(json.dumps(study, indent=2, default=float))
    print(json.dumps(gates, indent=2, default=float))
    return study


def _median(values):
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return float("nan")
    n = len(vals)
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def _gates(cells: list[dict]) -> dict:
    def subset(measure):
        return [c for c in cells if c["measure"] == measure]

    patch = subset(AREA_MEASURE_SPHERICAL)
    flat = subset(AREA_MEASURE_FLAT)

    g1_patch = [c for c in patch if c["resolved"]]
    g1_bad = [c for c in g1_patch
              if not (G1_ORDER_BAND[0] <= c["observed_order"] <= G1_ORDER_BAND[1])]
    g2_bad = [c for c in patch
              if not (math.isfinite(c["err_continuum_vs_ref"])
                      and c["err_continuum_vs_ref"] <= G2_CONTINUUM_TOL)]
    g3_bad = [c for c in patch if not (c["err_fine_vs_ref"] < c["err_medium_vs_ref"])
    ]
    p_patch = _median([c["observed_order"] for c in patch if c["resolved"]])
    p_flat = _median([c["observed_order"] for c in flat if c["resolved"]])
    h1_supported = bool(math.isfinite(p_patch) and math.isfinite(p_flat)
                        and p_patch >= p_flat + H1_ORDER_MARGIN)
    return {
        "G1_order_admissible": {
            "resolved_cells": len(g1_patch),
            "violations": len(g1_bad),
            "worst": max((abs(c["observed_order"]) for c in g1_patch), default=float("nan")),
            "verdict": "PASS" if g1_patch and not g1_bad else ("FAIL" if g1_bad else "NOT_RESOLVED"),
        },
        "G2_continuum_vs_reference": {
            "tolerance": G2_CONTINUUM_TOL, "cells": len(patch), "violations": len(g2_bad),
            "max_err_continuum": max((c["err_continuum_vs_ref"] for c in patch
                                      if math.isfinite(c["err_continuum_vs_ref"])),
                                     default=float("nan")),
            "max_err_fine": max(c["err_fine_vs_ref"] for c in patch),
            "verdict": "PASS" if not g2_bad else "FAIL",
            "violations_detail": [{"ka": c["ka"], "quantity": c["quantity"],
                                   "err": c["err_continuum_vs_ref"]} for c in g2_bad],
        },
        "G3_monotone_refinement": {
            "cells": len(patch), "violations": len(g3_bad),
            "verdict": "PASS" if not g3_bad else "FAIL",
            "violations_detail": [{"ka": c["ka"], "quantity": c["quantity"],
                                   "err_medium": c["err_medium_vs_ref"],
                                   "err_fine": c["err_fine_vs_ref"]} for c in g3_bad],
        },
        "H1_patch_raises_order": {
            "median_order_patch": p_patch, "median_order_flat": p_flat,
            "margin_required": H1_ORDER_MARGIN,
            "verdict": "SUPPORTED" if h1_supported else "FALSIFIED",
            "reading": ("patch also raises the observed order" if h1_supported
                        else "patch reduces the error constant only; order unchanged"),
        },
    }


def _find_nan_except_orders(payload, path="") -> list[str]:
    bad: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            bad += _find_nan_except_orders(value, "%s.%s" % (path, key))
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            bad += _find_nan_except_orders(value, "%s[%d]" % (path, index))
    elif isinstance(payload, float) and not math.isfinite(payload):
        bad.append(path or "<root>")
    return bad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("solve", "analyse", "all"), default="all")
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--subdivisions", type=int, nargs="+", default=list(SUBDIVISIONS),
                        help="ladder rungs to solve in this chunk (default: all)")
    parser.add_argument("--no-merge", action="store_true",
                        help="replace ladder_cells.json instead of extending it")
    args = parser.parse_args()
    if args.stage in ("solve", "all"):
        stage_solve(args.processes, tuple(args.subdivisions), merge=not args.no_merge,
                    serial=args.processes <= 1)
    if args.stage in ("analyse", "all"):
        assemble()


if __name__ == "__main__":
    main()