#!/usr/bin/env python3
"""Case-pool anchor ``BEM-HULME-HEMISPHERE``: Hulme (1982) floating hemisphere.

Why this driver exists
----------------------
The platform case pool (``tensor_case_pool.id = 138``, ``case_code =
BEM-HULME-HEMISPHERE``, ``tol_policy = <5%``) asks for one reproducible run of
the 3-D floating-hemisphere radiation/diffraction case against Hulme's (1982)
semi-analytical spherical wave-free multipole reference.  The repository has
the *solver* (``tensorbem.free_surface_solver3d``) and the *reference*
(``tensorbem.analytic.hulme_hemisphere_radiation``) already, but no artefact
that answers the case in one shot with a predeclared verdict.  This script is
that artefact: it states the decision, the metrics, the thresholds and the
controls *before* computing, then emits ``results.json`` (machine readable),
``report.md`` (tables) and ``figure.png`` (comparison plots).

Predeclared experiment (fixed before any number below was produced)
-------------------------------------------------------------------
Decision
    Does the 3-D deep-water Kelvin Green-function BEM in TensorBEM reproduce
    the Hulme (1982) floating-hemisphere radiation coefficients and
    wave-excitation forces to better than 5 % on a documented mesh, over a
    documented non-dimensional frequency band?
Alternatives considered
    (a) push the panel count until the band passes (what is tested);
    (b) declare the case unsupported for this solver (the honest-FAIL branch);
    (c) swap in a different formulation (Rankine + numerical beach) -- out of
        scope here, the case is defined on the Kelvin Green-function route.
Primary metrics (unit-velocity amplitude, ``exp(-i*omega*t)``)
    A11/(rho*nabla), B11/(rho*omega*nabla), A33/(rho*nabla),
    B33/(rho*omega*nabla), |Fx|/(rho*g*A*a^2), |Fz|/(rho*g*A*a^2) and the two
    force phases (deg).
    nabla = 2*pi*a^3/3 is the *exact* displaced volume, not the mesh volume:
    the coefficient normalisation must not inherit the mesh geometry error.
Reference (this is the quantity being trusted)
    Hulme (1982), J. Fluid Mech. 121:443-463, "The wave forces acting on a
    floating hemisphere undergoing forced periodic oscillations" -- wave-free
    multipole expansion in spherical harmonics about the origin, evaluated by
    ``tensorbem.analytic.hulme_hemisphere_radiation`` at 144 terms (the
    truncation ladder 48/96/144 with the geometric tail residual is part of
    the evidence, see ``ref_ladder`` in results.json).
Thresholds (predeclared)
    G1: |relative error| < 5 % for every quantity, every ka in the band, on the
        case mesh (1024 constant panels, subdivisions = 4).
    G2: geometry precondition -- |nabla_mesh/nabla - 1| < 2 % and
        |A_wp,mesh/(pi*a^2) - 1| < 2 % on the case mesh (a coefficient table is
        not readable if the body is mis-represented).
    G3: h-convergence -- the band-max error must not increase from 64 to 256
        panels.
    G4: reference self-convergence -- the 96/144-term reference difference must
        be at least 20x smaller than the 5 % gate at every ka.
    Stopping rule: the run is stopped (and reported as FAIL) if any G fails on
    the case mesh; no threshold is relaxed after inspecting results.
Controls / confounders
    * constant-panel resolution (``subdivisions``) -- primary confounder, both
      for the geometry and for the phase of the exciting force;
    * interior CHIEF null-field equations on/off -- confounder at the first
      irregular frequency (ka ~ 2.58 for this waterplane);
    * reference truncation ``n_terms`` -- must be shown negligible;
    * the whole band stays below ka = 2.0, i.e. away from the irregular
      frequency, so an irregular-frequency artefact cannot masquerade as
      discretisation error.
Repetitions
    Deterministic: no random seeds anywhere in the assembly or the solve, so
    each (ka, mesh) cell is computed once; the *sensitivity* of the verdict is
    established by the 3-level mesh ladder and the 3-level reference ladder
    rather than by repeated sampling.

Reproduce (single shard; the full run is the union over shards)
--------------------------------------------------------------
    python3 -m tensorbem.benchmarks.hulme_hemisphere_case \
        --shard 0 --shards 1 --out benchmarks/verified/casepool_bem_hulme_hemisphere
    python3 -m tensorbem.benchmarks.hulme_hemisphere_case \
        --merge --out benchmarks/verified/casepool_bem_hulme_hemisphere
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from tensorbem.analytic import hulme_hemisphere_radiation
from tensorbem.free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    solve_diffraction_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import hemisphere_hull_mesh

# --- case definition (fixed before running) --------------------------------
RADIUS = 1.0           # a [m]
DENSITY = 1025.0       # rho [kg/m^3] (sea water)
GRAVITY = 9.80665      # g [m/s^2]
AMPLITUDE = 1.0        # incident wave amplitude A [m] -> unit normalisation
BAND_KA = (0.2, 0.5, 0.8, 1.0, 1.5, 2.0)
CASE_SUBDIVISIONS = 4        # 4 * 4**4 = 1024 constant panels
COARSE_SUBDIVISIONS = 3      # 4 * 4**3 = 256 panels
LADDER_SUBDIVISIONS = (2, 3, 4)
LADDER_KA = (0.5, 1.0, 2.0)
REF_TERMS = (48, 96, 144)
GATE = 0.05                  # predeclared 5 % tolerance of the case pool
GEOM_GATE = 0.02             # 2 % geometry precondition
QUANTITIES = (
    "A11_dimless",
    "B11_dimless",
    "A33_dimless",
    "B33_dimless",
    "Fx_dimless",
    "Fz_dimless",
)
PHASE_KEYS = ("phase_Fx_deg", "phase_Fz_deg")

REPO_ROOT = Path(__file__).resolve().parents[3]


# --- reference -------------------------------------------------------------
def reference(ka: float, n_terms: int = REF_TERMS[-1]) -> dict[str, float]:
    """Hulme (1982) semi-analytical reference, non-dimensionalised.

    ``A`` is scaled by ``rho*nabla``, ``B`` by ``rho*omega*nabla``, forces by
    ``rho*g*A*a^2`` with ``nabla = 2*pi*a^3/3`` the exact displaced volume.
    """
    ref = hulme_hemisphere_radiation(
        ka,
        radius=RADIUS,
        amplitude=AMPLITUDE,
        density=DENSITY,
        gravity=GRAVITY,
        n_terms=n_terms,
        quad_order=max(2 * n_terms, 128),
    )
    scale_f = DENSITY * GRAVITY * AMPLITUDE * RADIUS * RADIUS
    fx = complex(ref.excitation_force_x) / scale_f
    fz = complex(ref.excitation_force_z) / scale_f
    return {
        "A11_dimless": ref.surge_added_mass_dimless,
        "B11_dimless": ref.surge_damping_dimless,
        "A33_dimless": ref.heave_added_mass_dimless,
        "B33_dimless": ref.heave_damping_dimless,
        "Fx_dimless": abs(fx),
        "Fz_dimless": abs(fz),
        "phase_Fx_deg": math.degrees(math.atan2(fx.imag, fx.real)),
        "phase_Fz_deg": math.degrees(math.atan2(fz.imag, fz.real)),
    }


def reference_ladder(ka: float) -> dict[str, Any]:
    """Truncation ladder: 48 -> 96 -> 144 terms (self-convergence of the ref)."""
    vals = {n: reference(ka, n) for n in REF_TERMS}
    top = vals[REF_TERMS[-1]]
    pairs = list(zip(REF_TERMS[:-1], REF_TERMS[1:]))
    deltas = {
        f"{a}->{b}": max(
            abs(vals[b][k] - vals[a][k]) / abs(top[k]) for k in QUANTITIES
        )
        for a, b in pairs
    }
    d1 = abs(vals[96]["A33_dimless"] - vals[48]["A33_dimless"])
    d2 = abs(vals[144]["A33_dimless"] - vals[96]["A33_dimless"])
    ratio = d2 / d1 if d1 > 0.0 else 0.0
    return {
        "ka": ka,
        "values": {str(n): {k: vals[n][k] for k in QUANTITIES} for n in REF_TERMS},
        "max_rel_delta": deltas,
        "tail_ratio_A33": ratio,
    }


# --- solver cells ----------------------------------------------------------
def _panel_cell(
    ka: float,
    subdivisions: int,
    *,
    chief: bool = True,
    with_diffraction: bool = True,
) -> dict[str, Any]:
    """One (ka, mesh) radiation+diffraction cell, fully documented."""
    mesh = hemisphere_hull_mesh(subdivisions, radius=RADIUS)
    nu = ka / RADIUS                       # nu = omega^2/g = k/a for ka = k*a
    t0 = time.time()
    ops = assemble_free_surface_operators_3d(
        mesh, nu, chief_points="auto" if chief else None
    )
    rad = solve_radiation_3d(
        mesh,
        nu,
        modes=(0, 2),
        density=DENSITY,
        gravity=GRAVITY,
        operators=ops,
    )
    cell: dict[str, Any] = {
        "ka": ka,
        "nu": nu,
        "subdivisions": subdivisions,
        "panels": mesh.panel_count,
        "chief": chief,
        "mesh_volume": float(mesh.volume),
        "mesh_waterplane_area": float(mesh.waterplane_area),
        "condition_number": float(ops.condition_number),
    }
    vol = (2.0 / 3.0) * math.pi * RADIUS**3
    norm_a = DENSITY * vol
    norm_b = DENSITY * math.sqrt(GRAVITY * nu) * vol
    cell["A11_dimless"] = float(rad.added_mass[0, 0]) / norm_a
    cell["A33_dimless"] = float(rad.added_mass[1, 1]) / norm_a
    cell["B11_dimless"] = float(rad.damping[0, 0]) / norm_b
    cell["B33_dimless"] = float(rad.damping[1, 1]) / norm_b
    cell["nabla_exact"] = vol
    if with_diffraction:
        diff = solve_diffraction_3d(
            mesh,
            nu,
            amplitude=AMPLITUDE,
            wave_angle=0.0,
            density=DENSITY,
            gravity=GRAVITY,
            operators=ops,
        )
        scale_f = DENSITY * GRAVITY * AMPLITUDE * RADIUS * RADIUS
        fx = complex(diff.excitation_force[0]) / scale_f
        fz = complex(diff.excitation_force[2]) / scale_f
        cell["Fx_dimless"] = abs(fx)
        cell["Fz_dimless"] = abs(fz)
        cell["phase_Fx_deg"] = math.degrees(math.atan2(fx.imag, fx.real))
        cell["phase_Fz_deg"] = math.degrees(math.atan2(fz.imag, fz.real))
        cell["Fx_real"], cell["Fx_imag"] = fx.real, fx.imag
        cell["Fz_real"], cell["Fz_imag"] = fz.real, fz.imag
    cell["wall_seconds"] = time.time() - t0
    ref = reference(ka)
    cell["reference"] = {k: ref[k] for k in QUANTITIES + PHASE_KEYS}
    for k in QUANTITIES:
        cell[f"rel_error_{k}"] = (cell[k] - ref[k]) / abs(ref[k])
    for k in PHASE_KEYS:
        cell[f"delta_{k}"] = cell[k] - ref[k]
    return cell


def run_cell(
    ka: float,
    subdivisions: int,
    *,
    chief: bool = True,
    with_diffraction: bool = True,
) -> dict[str, Any]:
    """Public single-cell entry: one ``(ka, mesh)`` BEM solve plus its errors.

    Thin, documented wrapper around :func:`_panel_cell` so callers (tests, the
    evidence bundle, downstream case-pool drivers) do not have to import a
    private name.
    """
    return _panel_cell(
        ka, subdivisions, chief=chief, with_diffraction=with_diffraction
    )


def build_tasks() -> list[dict[str, Any]]:
    """Deterministic task list: band x coarse/case mesh, mesh ladder, CHIEF control."""
    tasks: list[dict[str, Any]] = []
    for ka in BAND_KA:
        tasks.append(
            {"kind": "band", "ka": ka, "subdivisions": COARSE_SUBDIVISIONS}
        )
        tasks.append(
            {"kind": "band", "ka": ka, "subdivisions": CASE_SUBDIVISIONS}
        )
    for ka in LADDER_KA:
        tasks.append(
            {"kind": "ladder", "ka": ka, "subdivisions": LADDER_SUBDIVISIONS[-1]}
        )
    for ka in (1.0, 2.0):
        tasks.append(
            {
                "kind": "chief_off",
                "ka": ka,
                "subdivisions": CASE_SUBDIVISIONS,
            }
        )
    return tasks


# --- merge / verdict -------------------------------------------------------
def _rows(rows: Iterable[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    out = [r for r in rows if r["kind"] == kind]
    return sorted(out, key=lambda r: (r["ka"], r["subdivisions"]))


def evaluate(rows: list[dict[str, Any]], ref_ladders: dict[str, Any]) -> dict[str, Any]:
    """Apply the predeclared gates G1..G4 -- no relaxation after the fact."""
    case_rows = [
        r for r in _rows(rows, "band") if r["subdivisions"] == CASE_SUBDIVISIONS
    ]
    coarse_rows = [
        r for r in _rows(rows, "band") if r["subdivisions"] == COARSE_SUBDIVISIONS
    ]
    criteria: dict[str, Any] = {}

    worst = max(
        (abs(r[f"rel_error_{q}"]), q, r["ka"]) for r in case_rows for q in QUANTITIES
    )
    criteria["G1_case_mesh_band_error_below_5pct"] = {
        "criterion": f"|rel err| < {GATE:.0%} for every quantity and every ka on the case mesh",
        "measured": worst[0],
        "worst_quantity": worst[1],
        "worst_ka": worst[2],
        "pass": bool(worst[0] < GATE),
    }

    geom = max(
        (
            abs(r["mesh_volume"] / r["nabla_exact"] - 1.0),
            abs(r["mesh_waterplane_area"] / (math.pi * RADIUS**2) - 1.0),
            r["ka"],
        )
        for r in rows
        if r["subdivisions"] == CASE_SUBDIVISIONS
    )
    criteria["G2_geometry_precondition"] = {
        "criterion": f"|nabla_mesh/nabla-1| < {GEOM_GATE:.0%} and |A_wp/(pi a^2)-1| < {GEOM_GATE:.0%} on the case mesh",
        "measured_volume_error": geom[0],
        "measured_waterplane_error": geom[1],
        "pass": bool(max(geom[0], geom[1]) < GEOM_GATE),
    }

    worst_case = max(abs(r[f"rel_error_{q}"]) for r in case_rows for q in QUANTITIES)
    if coarse_rows:
        worst_coarse = max(
            abs(r[f"rel_error_{q}"]) for r in coarse_rows for q in QUANTITIES
        )
        g3 = {
            "criterion": (f"band-max error at {case_rows[0]['panels']} panels "
                          f"<= band-max error at {coarse_rows[0]['panels']} panels"),
            "measured_coarse": worst_coarse,
            "measured_case": worst_case,
            "pass": bool(worst_case <= worst_coarse),
        }
    else:
        # A run without the coarse rung cannot demonstrate that refinement
        # helps; that is a gate failure (insufficient evidence), not a pass.
        g3 = {
            "criterion": "band-max error at the case mesh <= band-max error at the "
                         "coarse mesh",
            "measured_coarse": None,
            "measured_case": worst_case,
            "pass": False,
            "note": "no coarse-mesh rows in this run: h-convergence not evidenced",
        }
    criteria["G3_refinement_does_not_worsen"] = g3

    ladder_margin = min(
        (0.2 * GATE) / max(v["max_rel_delta"]["96->144"], 1e-300)
        for v in ref_ladders.values()
    )
    worst_ladder = max(v["max_rel_delta"]["96->144"] for v in ref_ladders.values())
    criteria["G4_reference_self_converged"] = {
        "criterion": "reference 96->144 term drift at least 20x below the 5 % gate",
        "measured_worst_drift": worst_ladder,
        "margin_factor": ladder_margin,
        "pass": bool(worst_ladder * 20.0 < GATE),
    }

    return {
        "criteria": criteria,
        "verdict": "PASS" if all(c["pass"] for c in criteria.values()) else "FAIL",
    }


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def write_report(
    out_dir: Path,
    rows: list[dict[str, Any]],
    ref_ladders: dict[str, Any],
    verdict: dict[str, Any],
    meta: dict[str, Any],
) -> Path:
    band = _rows(rows, "band")
    case_rows = [r for r in band if r["subdivisions"] == CASE_SUBDIVISIONS]
    coarse_rows = [r for r in band if r["subdivisions"] == COARSE_SUBDIVISIONS]
    ladder = _rows(rows, "ladder")
    chief_off = _rows(rows, "chief_off")

    lines: list[str] = []
    lines.append("# CASE-BEM-HULME-HEMISPHERE -- floating hemisphere radiation/diffraction")
    lines.append("")
    lines.append(
        "Case pool id 138 (`BEM-HULME-HEMISPHERE`, repo TensorBEM, tol `<5%`). "
        "Generated by `benchmarks/verified/casepool_bem_hulme_hemisphere/run_case.py`."
    )
    lines.append("")
    lines.append("## 1. Case and reference")
    lines.append("")
    lines.append(
        "Floating hemisphere, flat face flush with the mean free surface, radius "
        f"a = {RADIUS} m, deep water, rho = {DENSITY} kg/m^3, g = {GRAVITY} m/s^2, "
        f"incident amplitude A = {AMPLITUDE} m. Nondimensional frequency "
        "ka = k*a with k = omega^2/g. Reference: Hulme (1982) *J. Fluid Mech.* "
        "**121**:443-463 (wave-free multipole expansion in spherical harmonics), "
        "evaluated by `tensorbem.analytic.hulme_hemisphere_radiation`."
    )
    lines.append("")
    lines.append(f"Environment: {meta['python']}, numpy {meta['numpy']}, host {meta['host']}, arch {meta['machine']}.")
    lines.append("")
    lines.append("## 2. Method")
    lines.append("")
    lines.append(
        "`tensorbem.free_surface_solver3d` assembles the Kelvin (Havelock) "
        "Green-function BIE on constant panels from `geometry3d.hemisphere_hull_mesh` "
        "(recursive octahedral subdivision of the lower half; 4*4^n panels), solves "
        "the radiation problem for surge (mode 0) and heave (mode 2) at unit "
        "velocity amplitude and the diffraction problem for head-on waves, and "
        "uses interior CHIEF null-field equations (`chief_points='auto'`)."
    )
    lines.append("")
    lines.append("Normalisation: A/(rho*nabla), B/(rho*omega*nabla) with "
                 "nabla = 2*pi*a^3/3 = %.6f m^3 (exact, not the mesh volume); "
                 "forces by rho*g*A*a^2 = %.1f N." % (2.0 / 3.0 * math.pi * RADIUS**3,
                                                      DENSITY * GRAVITY * AMPLITUDE * RADIUS**2))
    lines.append("")
    lines.append("## 3. Parameters")
    lines.append("")
    lines.append("| quantity | value |")
    lines.append("|---|---|")
    lines.append(f"| ka band | {', '.join(f'{ka:g}' for ka in BAND_KA)} |")
    lines.append(f"| case mesh | subdivisions {CASE_SUBDIVISIONS} = "
                 f"{4 * 4**CASE_SUBDIVISIONS} constant panels |")
    lines.append(f"| coarse mesh | subdivisions {COARSE_SUBDIVISIONS} = "
                 f"{4 * 4**COARSE_SUBDIVISIONS} panels |")
    lines.append(f"| mesh ladder | subdivisions {', '.join(map(str, LADDER_SUBDIVISIONS))} at ka = "
                 f"{', '.join(f'{ka:g}' for ka in LADDER_KA)} |")
    lines.append(f"| reference terms | {', '.join(map(str, REF_TERMS))} |")
    lines.append(f"| gate | {GATE:.0%} relative error, all quantities, all ka |")
    lines.append("")
    lines.append("## 4. Reference vs computed")
    lines.append("")
    for label, table in (
        (f"case mesh ({case_rows[0]['panels']} panels)", case_rows),
        (f"coarse mesh ({coarse_rows[0]['panels']} panels)", coarse_rows),
    ):
        lines.append(f"### {label}")
        lines.append("")
        lines.append(
            "| ka | A11 ref | A11 got | err | B11 ref | B11 got | err | A33 ref | A33 got | err "
            "| B33 ref | B33 got | err |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in table:
            ref = r["reference"]
            row = [f"{r['ka']:g}"]
            for key, err in (("A11_dimless", "rel_error_A11_dimless"),
                             ("B11_dimless", "rel_error_B11_dimless"),
                             ("A33_dimless", "rel_error_A33_dimless"),
                             ("B33_dimless", "rel_error_B33_dimless")):
                ok = "OK" if abs(r[err]) < GATE else "X"
                row += [_fmt(ref[key]), _fmt(r[key]), f"{r[err]:+.2%} {ok}"]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append(
            "| ka | |Fx| ref | |Fx| got | err | phase ref (deg) | phase got (deg) | delta "
            "| |Fz| ref | |Fz| got | err | phase ref (deg) | phase got (deg) | delta |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in table:
            ref = r["reference"]
            ok_x = "OK" if abs(r["rel_error_Fx_dimless"]) < GATE else "X"
            ok_z = "OK" if abs(r["rel_error_Fz_dimless"]) < GATE else "X"
            lines.append(
                "| " + " | ".join([
                    f"{r['ka']:g}",
                    _fmt(ref["Fx_dimless"]), _fmt(r["Fx_dimless"]),
                    f"{r['rel_error_Fx_dimless']:+.2%} {ok_x}",
                    _fmt(ref["phase_Fx_deg"], 2), _fmt(r["phase_Fx_deg"], 2),
                    f"{r['delta_phase_Fx_deg']:+.2f}",
                    _fmt(ref["Fz_dimless"]), _fmt(r["Fz_dimless"]),
                    f"{r['rel_error_Fz_dimless']:+.2%} {ok_z}",
                    _fmt(ref["phase_Fz_deg"], 2), _fmt(r["phase_Fz_deg"], 2),
                    f"{r['delta_phase_Fz_deg']:+.2f}",
                ]) + " |"
            )
        lines.append("")
    lines.append("## 5. Convergence and sensitivity")
    lines.append("")
    lines.append("### 5.1 Mesh ladder (subdivisions 2/3/4 = 64/256/1024 panels)")
    lines.append("")
    lines.append("| ka | panels | A33 err | B33 err | A11 err | B11 err | |Fx| err | |Fz| err | cond |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in ladder:
        lines.append(
            "| " + " | ".join([
                f"{r['ka']:g}", str(r["panels"]),
                f"{r['rel_error_A33_dimless']:+.2%}", f"{r['rel_error_B33_dimless']:+.2%}",
                f"{r['rel_error_A11_dimless']:+.2%}", f"{r['rel_error_B11_dimless']:+.2%}",
                f"{r['rel_error_Fx_dimless']:+.2%}", f"{r['rel_error_Fz_dimless']:+.2%}",
                _fmt(r["condition_number"], 2),
            ]) + " |"
        )
    lines.append("")
    lines.append("### 5.2 Reference truncation ladder")
    lines.append("")
    lines.append("| ka | 48->96 term drift | 96->144 term drift | tail ratio (A33) |")
    lines.append("|---|---|---|---|")
    for ka, lad in sorted(ref_ladders.items(), key=lambda kv: float(kv[0])):
        lines.append(
            f"| {float(ka):g} | {lad['max_rel_delta']['48->96']:.3e} | "
            f"{lad['max_rel_delta']['96->144']:.3e} | {lad['tail_ratio_A33']:.3e} |"
        )
    lines.append("")
    lines.append("### 5.3 CHIEF control (interior null-field equations off)")
    lines.append("")
    lines.append("| ka | chief | cond | A33 err | B33 err | B33 (rho*omega*nabla) |")
    lines.append("|---|---|---|---|---|---|")
    for r in case_rows + chief_off:
        if r["ka"] not in (1.0, 2.0):
            continue
        lines.append(
            f"| {r['ka']:g} | {'on' if r['chief'] else 'off'} | {r['condition_number']:.2f} | "
            f"{r['rel_error_A33_dimless']:+.2%} | {r['rel_error_B33_dimless']:+.2%} | "
            f"{r['B33_dimless']:.5f} |"
        )
    lines.append("")
    lines.append("## 6. Verdict")
    lines.append("")
    lines.append("| gate | criterion | measured | verdict |")
    lines.append("|---|---|---|---|")
    for key, crit in verdict["criteria"].items():
        if "measured_coarse" in crit and crit["measured_coarse"] is not None:
            measured = (f"coarse {crit['measured_coarse']:.4e} -> "
                        f"case {crit['measured_case']:.4e}")
        elif "measured_volume_error" in crit:
            measured = (f"volume {crit['measured_volume_error']:.4e}, "
                        f"waterplane {crit['measured_waterplane_error']:.4e}")
        else:
            measured = crit.get("measured")
            if measured is None:
                measured = crit.get("measured_worst_drift")
        if isinstance(measured, float):
            measured = f"{measured:.4e}"
        lines.append(f"| {key} | {crit['criterion']} | {measured} | {'PASS' if crit['pass'] else 'FAIL'} |")
    lines.append("")
    lines.append(f"**Verdict: {verdict['verdict']}**")
    lines.append("")
    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_figure(
    out_dir: Path, rows: list[dict[str, Any]], ref_ladders: dict[str, Any]
) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - plotting is optional
        return None

    band = [r for r in _rows(rows, "band") if r["subdivisions"] == CASE_SUBDIVISIONS]
    kas = np.array([r["ka"] for r in band])
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

    ax = axes[0]
    ax.plot(kas, [r["reference"]["A33_dimless"] for r in band], "k-", lw=2, label="Hulme 1982 ref")
    ax.plot(kas, [r["A33_dimless"] for r in band], "o", mfc="none", label="TensorBEM 256 panels")
    coarse = [r for r in _rows(rows, "band") if r["subdivisions"] == COARSE_SUBDIVISIONS]
    ax.plot([r["ka"] for r in coarse], [r["A33_dimless"] for r in coarse], "s", ms=4,
            mfc="none", label="TensorBEM 64 panels")
    ax.set_xlabel("ka = k a"); ax.set_ylabel("A33 / (rho nabla)")
    ax.set_title("Heave added mass"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(kas, [r["reference"]["B33_dimless"] for r in band], "k-", lw=2, label="Hulme 1982 ref")
    ax.plot(kas, [r["B33_dimless"] for r in band], "o", mfc="none", label="TensorBEM 256 panels")
    ax.plot([r["ka"] for r in coarse], [r["B33_dimless"] for r in coarse], "s", ms=4,
            mfc="none", label="TensorBEM 64 panels")
    ax.set_xlabel("ka = k a"); ax.set_ylabel("B33 / (rho omega nabla)")
    ax.set_title("Heave radiation damping"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(kas, [r["reference"]["Fx_dimless"] for r in band], "k-", lw=2, label="Hulme |Fx|")
    ax.plot(kas, [r["reference"]["Fz_dimless"] for r in band], "k--", lw=2, label="Hulme |Fz|")
    ax.plot(kas, [r["Fx_dimless"] for r in band], "o", mfc="none", label="TensorBEM |Fx|")
    ax.plot(kas, [r["Fz_dimless"] for r in band], "^", mfc="none", label="TensorBEM |Fz|")
    ax.set_xlabel("ka = k a"); ax.set_ylabel("|F| / (rho g A a^2)")
    ax.set_title("Wave excitation force magnitude"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle("CASE-BEM-HULME-HEMISPHERE: floating hemisphere (a = 1 m, deep water)")
    fig.tight_layout()
    path = out_dir / "figure.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_environment(out_dir: Path, extra: dict[str, Any] | None = None) -> Path:
    """Environment artefact: what the numbers below were produced with."""
    env = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "host": platform.node(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "argv": list(sys.argv[1:]),
        "case_code": "BEM-HULME-HEMISPHERE",
    }
    if extra:
        env.update(extra)
    path = out_dir / "environment.json"
    path.write_text(json.dumps(env, indent=2), encoding="utf-8")
    return path


def write_log(out_dir: Path, message: str) -> None:
    """Append one timestamped line to the run log (proof of a real run)."""
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    with (out_dir / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp} {message}\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/casepool_bem_hulme_hemisphere")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument(
        "--no-fine",
        action="store_true",
        help="drop the 1024-panel rung (cheap local reproduction of the band)",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks()
    if args.no_fine:
        tasks = [t for t in tasks if t["subdivisions"] != LADDER_SUBDIVISIONS[-1]]
    ref_ladders = {f"{ka:g}": reference_ladder(ka) for ka in BAND_KA}
    (out_dir / "reference.json").write_text(
        json.dumps(
            {
                "citation": "Hulme (1982) J. Fluid Mech. 121:443-463, wave-free multipole expansion",
                "implementation": "tensorbem.analytic.hulme_hemisphere_radiation",
                "terms": list(REF_TERMS),
                "terms_used": REF_TERMS[-1],
                "normalisation": {
                    "added_mass": "A / (rho * nabla), nabla = 2*pi*a^3/3 (exact)",
                    "damping": "B / (rho * omega * nabla)",
                    "force": "F / (rho * g * A * a^2)",
                    "time_convention": "exp(-i*omega*t)",
                },
                "reference": {f"{ka:g}": reference(ka) for ka in BAND_KA},
                "ladder": ref_ladders,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if not args.merge:
        if args.shards < 1 or not (0 <= args.shard < args.shards):
            raise SystemExit("invalid shard specification")
        mine = tasks[args.shard :: args.shards]
        if args.max_tasks:
            mine = mine[: args.max_tasks]
        rows: list[dict[str, Any]] = []
        for task in mine:
            started = time.time()
            cell = _panel_cell(
                task["ka"],
                task["subdivisions"],
                chief=task["kind"] != "chief_off",
                with_diffraction=True,
            )
            cell["kind"] = task["kind"]
            rows.append(cell)
            message = (
                f"[shard {args.shard}/{args.shards}] {task['kind']} ka={task['ka']:g} "
                f"panels={cell['panels']} worst={max(abs(cell[f'rel_error_{q}']) for q in QUANTITIES):.4%} "
                f"({time.time() - started:.0f}s)"
            )
            print(message, flush=True)
            write_log(out_dir, message)
        (out_dir / f"shard_{args.shard}.json").write_text(
            json.dumps({"rows": rows}, indent=2), encoding="utf-8"
        )
        return 0

    rows = []
    for path in sorted(out_dir.glob("shard_*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["rows"])
    if not rows:
        raise SystemExit(f"no shard results in {out_dir}")
    verdict = evaluate(rows, ref_ladders)
    meta = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "host": platform.node(),
        "machine": platform.machine(),
    }
    payload = {
        "case": {
            "case_code": "BEM-HULME-HEMISPHERE",
            "case_pool_id": 138,
            "mission_id": 17292,
            "origin_mission_id": 17181,
            "repo": "TensorBEM",
            "tol_policy": "<5%",
            "body": "floating hemisphere, flat face on the mean free surface",
            "radius_m": RADIUS,
            "density_kg_m3": DENSITY,
            "gravity_m_s2": GRAVITY,
            "amplitude_m": AMPLITUDE,
            "displaced_volume_m3": 2.0 / 3.0 * math.pi * RADIUS**3,
            "waterplane_area_m2": math.pi * RADIUS**2,
            "band_ka": list(BAND_KA),
            "case_subdivisions": CASE_SUBDIVISIONS,
            "coarse_subdivisions": COARSE_SUBDIVISIONS,
            "ladder_subdivisions": list(LADDER_SUBDIVISIONS),
            "ladder_ka": list(LADDER_KA),
            "reference_terms": list(REF_TERMS),
        },
        "reference_provenance": {
            "citation": "Hulme, A. (1982) The wave forces acting on a floating hemisphere undergoing forced periodic oscillations. J. Fluid Mech. 121:443-463.",
            "method": "spherical wave-free multipole expansion (Hulme's semi-analytical solution)",
            "implementation": "tensorbem.analytic.hulme_hemisphere_radiation",
            "terms_used": REF_TERMS[-1],
            "truncation_ladder": {k: v["max_rel_delta"] for k, v in ref_ladders.items()},
            "paper_digit_transcription": "NOT AVAILABLE (publisher paywall, no copy in the sandbox): no digit of Hulme's printed table is transcribed or hand-copied anywhere in this repository. Every reference number in results.json is computed at run time by the wave-free multipole series (recomputed and cross-checked by evidence.py), so 'ref' here means 'Hulme's semi-analytical method, re-derived and self-converged', not 'Hulme's printed digits' -- see docs/bem_hulme_hemisphere_case.md sections 1 and 6.",
            "shared_kernel_risk": "the reference series and the solver share tensorbem.free_surface3d.wave_function_3d, so agreement alone cannot catch a defect in that kernel; the independent axes are the third-party BEM crosscheck (capytaine, docs/capytaine_crosscheck.md) and the ka->0 rigid-lid even-mode series.",
            "independent_crosschecks": [
                "docs/capytaine_crosscheck.md (capytaine, a third-party BEM code, same body and frequencies)",
                "benchmarks/verified/bem_rigid_lid_hemisphere/summary.md (ka -> 0 rigid-lid even-mode series 0.830954)",
            ],
        },
        "environment": meta,
        "tasks": tasks,
        "reference": {f"{ka:g}": reference(ka) for ka in BAND_KA},
        "ref_ladder": ref_ladders,
        "rows": rows,
        "verdict": verdict,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_environment(out_dir, {"verdict": verdict["verdict"], "rows": len(rows),
                                "tasks": len(tasks)})
    report = write_report(out_dir, rows, ref_ladders, verdict, meta)
    figure = write_figure(out_dir, rows, ref_ladders)
    print(f"wrote {out_dir/'results.json'}, {report}, {figure}", flush=True)
    print(f"VERDICT: {verdict['verdict']}", flush=True)
    write_log(out_dir, f"merged {len(rows)} rows -> VERDICT {verdict['verdict']}")
    for key, crit in verdict["criteria"].items():
        write_log(out_dir, f"gate {key}: {'PASS' if crit['pass'] else 'FAIL'}")
    for key, crit in verdict["criteria"].items():
        print(f"  {key}: {'PASS' if crit['pass'] else 'FAIL'} measured={crit.get('measured', crit.get('measured_worst_drift'))}", flush=True)
    return 0 if verdict["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())