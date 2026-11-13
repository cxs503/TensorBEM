#!/usr/bin/env python3
"""m17243 — CASE-BEM-RIGID-LID-HEMISPHERE: rigid-lid hemisphere added mass (ka->0).

Benchmark-pool case (owner 2026-11-12 算例池令).  Reuses TensorBEM's existing
capabilities:

  * even-parity double-body Legendre series  -> exact ka->0 reference
  * 3-D Kelvin-Green-function radiation solver -> the BEM "got"
  * Hulme (1982) multipole series            -> independent cross-check

Produces (zero hand-copied numbers, all from the single payload):
  benchmarks/verified/bem_rigid_lid_hemisphere_ka0/{results.json,
  summary.md, figure.png}

Run:
    python3 benchmarks/verified/bem_rigid_lid_hemisphere_ka0/run_study.py
    python3 benchmarks/verified/bem_rigid_lid_hemisphere_ka0/make_figure.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _repo_root() -> Path:
    for parent in HERE.parents:
        if (parent / "src" / "tensorbem" / "__init__.py").exists():
            return parent
    raise RuntimeError("repo root not found above " + str(HERE))


sys.path.insert(0, str(_repo_root() / "src"))

from tensorbem.analytic import hulme_hemisphere_radiation  # noqa: E402
from tensorbem.benchmarks.rigid_lid_hemisphere_attribution import (  # noqa: E402
    even_mode_added_mass_dimless,
)
from tensorbem.free_surface_solver3d import solve_radiation_3d  # noqa: E402
from tensorbem.geometry3d import hemisphere_hull_mesh  # noqa: E402

# ---------------------------------------------------------------------------
# Physical parameters (SI)
# ---------------------------------------------------------------------------
RADIUS = 1.0
DENSITY = 1025.0
GRAVITY = 9.80665
VOLUME = (2.0 / 3.0) * math.pi * RADIUS**3          # displaced volume of hemisphere
EXACT_AREA = 2.0 * math.pi * RADIUS**2               # curved-surface area

# Mesh ladder (panels = 4 * 4**sub) and ka ladders.
MESH_LEVELS = (2, 3, 4)                              # 64, 256, 1024 panels
MESH_LADDER_KA = 1.0e-3                              # fixed small ka for mesh ladder
KA_LADDER = (1.0e-3, 1.0e-2, 3.0e-2, 0.1)           # ka ladder at 256 panels
KA_LADDER_SUB = 3

TOLERANCE = 0.05                                     # task tolerance: < 5 %


def _solve(sub: int, ka: float) -> dict:
    mesh = hemisphere_hull_mesh(sub, radius=RADIUS)
    t0 = time.time()
    res = solve_radiation_3d(mesh, float(ka), modes=[2],
                             density=DENSITY, gravity=GRAVITY)
    dt = time.time() - t0
    a33 = float(res.added_mass[0, 0])
    b33 = float(res.damping[0, 0])
    omega = math.sqrt(GRAVITY * ka)
    ratio = a33 / (DENSITY * VOLUME)
    hulme = hulme_hemisphere_radiation(float(ka), radius=RADIUS,
                                     density=DENSITY, gravity=GRAVITY)
    return {
        "subdivisions": int(sub),
        "panels": int(mesh.areas.size),
        "ka": float(ka),
        "omega": float(omega),
        "added_mass": a33,
        "added_mass_dimless": float(ratio),
        "damping_dimless": float(b33 / (omega * DENSITY * VOLUME)),
        "hulme_added_mass_dimless": float(hulme.heave_added_mass_dimless),
        "rel_error_vs_hulme": float(ratio / hulme.heave_added_mass_dimless - 1.0),
        "condition_number": float(res.condition_number),
        "relative_residual": float(res.relative_residual),
        "mesh_area_rel_error": float(np.sum(mesh.areas) / EXACT_AREA - 1.0),
        "solve_seconds": float(dt),
    }


def run_study() -> dict:
    t_start = time.time()

    # --- exact reference (even-parity double body) -------------------------
    excel = even_mode_added_mass_dimless(degree_max=60, quad_order=400)
    ref = excel["rigid_lid_added_mass_dimless"]
    series_tail = abs(excel["partial_sums_even"][-1] - excel["partial_sums_even"][-2])

    # --- mesh ladder at ka = 1e-3 -----------------------------------------
    mesh_rows = []
    for sub in MESH_LEVELS:
        row = _solve(sub, MESH_LADDER_KA)
        row["rel_error_vs_ref"] = float(row["added_mass_dimless"] / ref - 1.0)
        mesh_rows.append(row)

    # --- ka ladder at sub = 3 ---------------------------------------------
    ka_rows = []
    for ka in KA_LADDER:
        row = _solve(KA_LADDER_SUB, ka)
        row["rel_error_vs_ref"] = float(row["added_mass_dimless"] / ref - 1.0)
        ka_rows.append(row)

    # --- headline: finest mesh, smallest ka --------------------------------
    headline = mesh_rows[-1]                          # sub=4, ka=1e-3
    headline_err = headline["rel_error_vs_ref"]

    # --- observed convergence orders (mesh) --------------------------------
    orders = []
    for i in range(1, len(mesh_rows)):
        h0, h1 = mesh_rows[i - 1]["panels"], mesh_rows[i]["panels"]
        e0 = abs(mesh_rows[i - 1]["rel_error_vs_ref"])
        e1 = abs(mesh_rows[i]["rel_error_vs_ref"])
        orders.append(math.log(e0 / e1) / math.log(h1 / h0) if e1 > 0 else None)

    # --- criteria ----------------------------------------------------------
    criteria = {
        "C1_reference_below_rho_nabla": {
            "measured": ref,
            "criterion": "ref < 0.9 (refutes the 1.0 rho*nabla anchor)",
            "pass": bool(ref < 0.9),
        },
        "C2_headline_within_5pct": {
            "measured": 100.0 * headline_err,
            "criterion": "|rel err| < 5 % at ka=1e-3, finest mesh (sub=4)",
            "pass": bool(abs(headline_err) < TOLERANCE),
            "where": {"ka": headline["ka"], "panels": headline["panels"]},
        },
        "C3_mesh_convergence_monotone": {
            "measured": [abs(r["rel_error_vs_ref"]) for r in mesh_rows],
            "criterion": "error strictly decreases 64 -> 256 -> 1024 panels",
            "pass": bool(
                abs(mesh_rows[0]["rel_error_vs_ref"])
                > abs(mesh_rows[1]["rel_error_vs_ref"])
                > abs(mesh_rows[2]["rel_error_vs_ref"])
            ),
        },
        "C2b_256panel_within_5pct": {
            "measured": 100.0 * mesh_rows[1]["rel_error_vs_ref"],
            "criterion": "|rel err| < 5 % at ka=1e-3, 256 panels",
            "pass": bool(abs(mesh_rows[1]["rel_error_vs_ref"]) < TOLERANCE),
        },
        "C4_ka_ladder_bounded_5pct_at_ka_le_0.01": {
            "measured": max(
                abs(r["rel_error_vs_ref"])
                for r in ka_rows if r["ka"] <= 0.01
            ),
            "criterion": "max |rel err| < 5 % for ka <= 0.01 (256 panels)",
            "pass": bool(
                max(abs(r["rel_error_vs_ref"]) for r in ka_rows
                    if r["ka"] <= 0.01) < TOLERANCE
            ),
        },
        "C5_series_converged": {
            "measured": series_tail,
            "criterion": "last two even partial sums differ < 1e-6",
            "pass": bool(series_tail < 1e-6),
        },
    }

    payload = {
        "study": "m17243_rigid_lid_hemisphere_added_mass_ka0",
        "anchor": "CASE-BEM-RIGID-LID-HEMISPHERE",
        "physical_case": {
            "body": "hemisphere radius a, flat face on rigid lid (ka->0 limit "
                    "of the floating-hemisphere free-surface problem)",
            "radius": RADIUS,
            "density": DENSITY,
            "gravity": GRAVITY,
            "displaced_volume": VOLUME,
            "mode": "heave",
            "convention": "exp(-i omega t), unit velocity amplitude, "
                         "A33 = -rho Re(int phi n_z dS)",
        },
        "reference": {
            "method": "even-parity double-body Legendre series (rigid-lid mirror)",
            "rigid_lid_added_mass_dimless": ref,
            "pressure_release_added_mass_dimless":
                excel["pressure_release_added_mass_dimless"],
            "degree_max": excel["degree_max"],
            "n_terms_used": excel["n_terms_used"],
            "series_tail": series_tail,
            "note": "A33(0)/(rho*nabla); nabla = (2/3) pi a^3",
        },
        "mesh_ladder": mesh_rows,
        "ka_ladder": ka_rows,
        "headline": {
            "ka": headline["ka"],
            "panels": headline["panels"],
            "added_mass_dimless": headline["added_mass_dimless"],
            "rel_error_vs_ref": headline_err,
            "within_5pct": bool(abs(headline_err) < TOLERANCE),
        },
        "mesh_convergence_orders": orders,
        "criteria": criteria,
        "verdict": "ALL PASS" if all(c["pass"] for c in criteria.values()) else "FAIL",
        "wall_seconds": float(time.time() - t_start),
    }
    return payload


def render_summary(payload: dict) -> str:
    ref = payload["reference"]["rigid_lid_added_mass_dimless"]
    L = []
    L.append("# m17243 — Rigid-lid hemisphere added mass (ka -> 0)")
    L.append("")
    L.append(f"Reference (even-parity double-body series): "
             f"`A33(0)/(rho*nabla) = {ref:.9f}` "
             f"({payload['reference']['n_terms_used']} even modes, "
             f"tail = {payload['reference']['series_tail']:.2e}).")
    L.append("")
    L.append("## Mesh ladder (ka = 1e-3)")
    L.append("")
    L.append("| sub | panels | A33/(rho*nabla) | rel err vs ref | cond |")
    L.append("|---|---|---|---|---|")
    for r in payload["mesh_ladder"]:
        L.append(f"| {r['subdivisions']} | {r['panels']} | "
                 f"{r['added_mass_dimless']:.6f} | "
                 f"{100*r['rel_error_vs_ref']:+.3f} % | "
                 f"{r['condition_number']:.3g} |")
    L.append("")
    L.append("## ka ladder (256 panels)")
    L.append("")
    L.append("| ka | A33/(rho*nabla) | rel err vs ref | Hulme | rel err vs Hulme |")
    L.append("|---|---|---|---|---|")
    for r in payload["ka_ladder"]:
        L.append(f"| {r['ka']:.4g} | {r['added_mass_dimless']:.6f} | "
                 f"{100*r['rel_error_vs_ref']:+.3f} % | "
                 f"{r['hulme_added_mass_dimless']:.6f} | "
                 f"{100*r['rel_error_vs_hulme']:+.3f} % |")
    L.append("")
    L.append("## Criteria")
    L.append("")
    L.append("| id | measured | criterion | verdict |")
    L.append("|---|---|---|---|")
    for k, c in payload["criteria"].items():
        m = c["measured"]
        ms = f"{m:.6g}" if isinstance(m, float) else str(m)
        L.append(f"| {k} | {ms} | {c['criterion']} | "
                 f"{'PASS' if c['pass'] else 'FAIL'} |")
    L.append("")
    L.append(f"**Verdict: {payload['verdict']}**  "
             f"(wall {payload['wall_seconds']:.1f} s)")
    return "\n".join(L)


def main():
    payload = run_study()
    (HERE / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (HERE / "summary.md").write_text(render_summary(payload), encoding="utf-8")
    print(json.dumps({
        "verdict": payload["verdict"],
        "ref": payload["reference"]["rigid_lid_added_mass_dimless"],
        "headline": payload["headline"],
        "wall_seconds": payload["wall_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
