#!/usr/bin/env python3
"""Third-party (capytaine) mesh-refinement study for the Havelock submerged sphere.

Why this exists
---------------
``docs/case_bem_submerged_sphere_havelock/capytaine_crosscheck.json``
(``scripts/case_havelock_capytaine_crosscheck.py``) compares capytaine with the
multipole reference on two meshes (320 and 1280 panels) and reports a FAIL
(worst 6.76 %).  That verdict mixes two different things:

* capytaine's own discretisation error on a coarse mesh, and
* any genuine disagreement between capytaine and the multipole series.

A single-mesh comparison cannot separate them, so it cannot certify the
reference column the case pool gates against.  This script separates them with
a refinement ladder (80 / 320 / 1280 / 5120 panels) and, additionally, with a
Green-function singularity setting that removes a known accuracy floor of the
third-party code (capytaine's default ``gf_singularities='low_freq'`` drops the
high-frequency singular terms of the near-field integral, which leaves a
mesh-independent bias).

Two questions are answered, neither of which uses TensorBEM's solver:

A. Does an unrelated BEM code converge *to* the multipole series
   (``tensorbem.analytic.submerged_sphere_wave_multipole``)?  If the deviation
   decays with refinement and reaches well under the pool tolerance, the
   reference column is not an artefact of one implementation.
B. How accurate is the TensorBEM case-mesh result when it is measured against
   the capytaine fine-mesh value instead of against the reference series?

Output (machine-readable, committed):
    docs/case_bem_submerged_sphere_havelock/capytaine_refinement_ladder.json

Usage::

    PYTHONPATH=src python3 scripts/case_havelock_capytaine_refinement_ladder.py
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import capytaine as cpt

from tensorbem.analytic import submerged_sphere_wave_multipole
from tensorbem.geometry3d import submerged_sphere_mesh

COEFFICIENTS = ("a33", "b33", "a11", "b11")
BAND_KA = (0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
GF_SETTINGS = ("low_freq", "low_freq_with_rankine_part")

RADIUS = 1.0
SUBMERGENCE = 2.0
DENSITY = 1025.0
GRAVITY = 9.80665

# Pre-declared criteria (fixed before the numbers were looked at).
REFERENCE_TOL = 0.01        # |capytaine(finest) - ref| / |ref| < 1 %
DECAY_REQUIRED = 0.9        # random-access: |dev| must shrink on >= 90 % of items
DEGENERATE_SCALE = 0.1      # |ref| < 10 % of the band scale -> scale-normalised


def _capytaine_version() -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version("capytaine")
    except Exception:  # noqa: BLE001 - dev installs may not be registered
        return getattr(cpt, "__version__", "unknown")


def _solve_level(
    subdivision: int,
    ka_values: Sequence[float],
    gf_singularities: str,
    *,
    radius: float = RADIUS,
    submergence: float = SUBMERGENCE,
    density: float = DENSITY,
    gravity: float = GRAVITY,
) -> dict[str, Any]:
    """Solve heave + surge radiation on one mesh level for every ``ka``."""
    body_mesh = submerged_sphere_mesh(subdivision, radius=radius, submergence=submergence)
    mesh = cpt.Mesh(vertices=np.asarray(body_mesh.vertices, float),
                    faces=np.asarray(body_mesh.faces, int))
    body = cpt.FloatingBody(mesh=mesh, dofs=cpt.rigid_body_dofs())
    solver = cpt.BEMSolver(green_function=cpt.Delhommeau(gf_singularities=gf_singularities))
    exact_volume = 4.0 / 3.0 * math.pi * radius**3
    mesh_volume = float(mesh.volume)
    rows: list[dict[str, Any]] = []
    for ka in ka_values:
        nu = float(ka) / radius                       # deep water: omega^2 = g*k
        omega = math.sqrt(gravity * nu)
        started = time.perf_counter()
        heave = solver.solve(cpt.RadiationProblem(
            body=body, omega=omega, water_depth=np.inf, radiating_dof="Heave",
            rho=density, g=gravity,
        ))
        surge = solver.solve(cpt.RadiationProblem(
            body=body, omega=omega, water_depth=np.inf, radiating_dof="Surge",
            rho=density, g=gravity,
        ))
        seconds = time.perf_counter() - started
        got = {
            "a33": float(heave.added_mass["Heave"]) / (density * exact_volume),
            "b33": float(heave.radiation_damping["Heave"]) / (density * omega * exact_volume),
            "a11": float(surge.added_mass["Surge"]) / (density * exact_volume),
            "b11": float(surge.radiation_damping["Surge"]) / (density * omega * exact_volume),
        }
        ref = submerged_sphere_wave_multipole(
            ka, radius=radius, submergence=submergence, density=density, gravity=gravity,
        )
        exact = {
            "a33": ref.heave_added_mass_dimless,
            "b33": ref.heave_damping_dimless,
            "a11": ref.surge_added_mass_dimless,
            "b11": ref.surge_damping_dimless,
        }
        rows.append({
            "ka": float(ka),
            "subdivision": subdivision,
            "n_panels": int(body_mesh.panel_count),
            "omega": omega,
            "capytaine": got,
            "multipole_ref": exact,
            # signed: the sign is what tells a bias from a discretisation error
            "signed_rel_deviation": {
                key: (got[key] - exact[key]) / abs(exact[key]) for key in COEFFICIENTS
            },
            "abs_rel_deviation": {
                key: abs(got[key] - exact[key]) / abs(exact[key]) for key in COEFFICIENTS
            },
            "seconds": round(seconds, 3),
        })
    return {
        "subdivision": subdivision,
        "n_panels": int(body_mesh.panel_count),
        "mesh_volume": mesh_volume,
        "volume_deficit": abs(mesh_volume - exact_volume) / exact_volume,
        "rows": rows,
    }


def _analyse(ladder: Sequence[dict[str, Any]], ka_values: Sequence[float]) -> dict[str, Any]:
    """Deviation decay + observed order per (ka, quantity) across the ladder."""
    levels = sorted(ladder, key=lambda lv: lv["n_panels"])
    by_ka: dict[float, list[dict[str, Any]]] = {float(k): [] for k in ka_values}
    for level in levels:
        for row in level["rows"]:
            by_ka[float(row["ka"])].append(row)

    band_scale = 0.0
    for ka, rows in by_ka.items():
        for key in COEFFICIENTS:
            band_scale = max(band_scale, abs(rows[0]["multipole_ref"][key]))

    items: list[dict[str, Any]] = []
    for ka in ka_values:
        rows = sorted(by_ka[float(ka)], key=lambda r: r["n_panels"])
        for key in COEFFICIENTS:
            per_level = [
                {
                    "n_panels": r["n_panels"],
                    "subdivision": r["subdivision"],
                    "signed_rel_deviation": r["signed_rel_deviation"][key],
                    "abs_rel_deviation": r["abs_rel_deviation"][key],
                }
                for r in rows
            ]
            ref = rows[0]["multipole_ref"][key]
            fine = per_level[-1]
            mid = per_level[-2]
            r_ratio = math.sqrt(fine["n_panels"] / mid["n_panels"])
            d_fm = fine["signed_rel_deviation"] - mid["signed_rel_deviation"]
            d_mc = mid["signed_rel_deviation"] - per_level[-3]["signed_rel_deviation"] if len(
                per_level) >= 3 else float("nan")
            order = (
                math.log(abs(d_mc / d_fm)) / math.log(r_ratio)
                if (len(per_level) >= 3 and d_fm != 0.0 and d_mc * d_fm > 0.0) else None
            )
            degenerate = abs(ref) < DEGENERATE_SCALE * band_scale
            items.append({
                "ka": float(ka),
                "quantity": key,
                "n_panels": [p["n_panels"] for p in per_level],
                "per_level": per_level,
                "multipole_ref": ref,
                "finest_signed_deviation": fine["signed_rel_deviation"],
                "finest_abs_deviation": fine["abs_rel_deviation"],
                "mid_abs_deviation": mid["abs_rel_deviation"],
                "decay_ratio": fine["abs_rel_deviation"] / mid["abs_rel_deviation"],
                "observed_order": order,
                "degenerate": degenerate,
                "scale_normalised_deviation": abs(fine["signed_rel_deviation"]) * abs(ref) / band_scale,
                # degenerate rows (|ref| far below the band scale) are judged on the
                # scale-normalised deviation as well, so a 1e-6-sized reference is not
                # asked to carry a 1 % relative gate on its own
                "rel_pass": bool(fine["abs_rel_deviation"] < REFERENCE_TOL),
                "scaled_pass": bool(abs(fine["signed_rel_deviation"]) * abs(ref)
                                    / band_scale < REFERENCE_TOL),
                "pass": bool(fine["abs_rel_deviation"] < REFERENCE_TOL) if not degenerate
                else bool(fine["abs_rel_deviation"] < REFERENCE_TOL
                          or abs(fine["signed_rel_deviation"]) * abs(ref)
                          / band_scale < REFERENCE_TOL),
            })

    n_items = len(items)
    n_decayed = sum(1 for it in items if it["decay_ratio"] < 1.0)
    n_pass = sum(1 for it in items if it["pass"])
    worst = max(items, key=lambda it: it["finest_abs_deviation"])
    active = [it for it in items if not it["degenerate"]]
    worst_active = max(active, key=lambda it: it["finest_abs_deviation"]) if active else worst
    return {
        "items": items,
        "summary": {
            "n_items": n_items,
            "n_pass": n_pass,
            "n_decayed": n_decayed,
            "decay_fraction": n_decayed / n_items,
            "worst_finest_abs_deviation": worst["finest_abs_deviation"],
            "worst_item": {"ka": worst["ka"], "quantity": worst["quantity"]},
            "worst_active_abs_deviation": worst_active["finest_abs_deviation"],
            "worst_active_item": {"ka": worst_active["ka"],
                                  "quantity": worst_active["quantity"]},
            "max_decay_ratio": max(it["decay_ratio"] for it in items),
            "band_scale": band_scale,
        },
        "verdict": "PASS" if (worst_active["finest_abs_deviation"] < REFERENCE_TOL
                              and n_decayed / n_items >= DECAY_REQUIRED) else "FAIL",
    }


def _tensorbem_vs_capytaine(case_rows: Sequence[dict[str, Any]],
                            finest: dict[tuple[float, str], float],
                            band_scale: float) -> dict[str, Any] | None:
    """TensorBEM case-mesh result measured against the capytaine fine mesh."""
    items: list[dict[str, Any]] = []
    for row in case_rows:
        ka = float(row["ka"])
        for key in COEFFICIENTS:
            got = row.get(f"{key}_bem")
            lim = finest.get((ka, key))
            if got is None or lim is None:
                continue
            items.append({
                "ka": ka,
                "quantity": key,
                "tensorbem": float(got),
                "capytaine_finest": lim,
                "signed_rel_error": (float(got) - lim) / abs(lim),
                "abs_rel_error": abs(float(got) - lim) / abs(lim),
                "scale_normalised_error": abs(float(got) - lim) / band_scale,
            })
    if not items:
        return None
    worst = max(items, key=lambda r: r["abs_rel_error"])
    return {
        "n_items": len(items),
        "items": items,
        "worst": {"ka": worst["ka"], "quantity": worst["quantity"],
                  "abs_rel_error": worst["abs_rel_error"]},
        "max_abs_rel_error": worst["abs_rel_error"],
        "n_over_pool_tolerance": sum(1 for r in items if r["abs_rel_error"] >= 0.05),
    }


def run(
    subdivisions: Sequence[int] = (1, 2, 3, 4),
    ka_values: Sequence[float] = BAND_KA,
    gf_settings: Sequence[str] = GF_SETTINGS,
    case_numbers: Path | None = Path(
        "docs/case_bem_submerged_sphere_havelock/numbers.json"),
) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    for gf in gf_settings:
        ladder = [_solve_level(sub, ka_values, gf) for sub in subdivisions]
        analysis = _analyse(ladder, ka_values)
        settings[gf] = {"green_function": {"gf_singularities": gf},
                        "ladder": ladder, **analysis}

    primary = settings[gf_settings[-1]]
    band_scale = primary["summary"]["band_scale"]
    # values of the finest level of the primary setting
    finest_values: dict[tuple[float, str], float] = {}
    finest_level = max(primary["ladder"], key=lambda lv: lv["n_panels"])
    for row in finest_level["rows"]:
        for key in COEFFICIENTS:
            finest_values[(float(row["ka"]), key)] = row["capytaine"][key]

    payload: dict[str, Any] = {
        "case": "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
        "purpose": (
            "third-party (capytaine) mesh-refinement study: separates capytaine's own "
            "discretisation / near-field-integral error from any disagreement with the "
            "multipole reference that the case pool gates against"
        ),
        "capytaine_version": _capytaine_version(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "case_parameters": {
            "radius": RADIUS, "submergence": SUBMERGENCE,
            "submergence_ratio": SUBMERGENCE / RADIUS,
            "density": DENSITY, "gravity": GRAVITY,
            "water_depth": "infinite (deep water, omega^2 = g k)",
            "non_dimensionalisation": (
                "A/(rho*4/3*pi*a^3) and B/(rho*omega*4/3*pi*a^3): the *exact* sphere "
                "volume is used on both sides, so the polyhedral volume deficit of the "
                "mesh stays visible instead of being cancelled"),
        },
        "criteria": {
            "reference_tol": REFERENCE_TOL,
            "decay_required": DECAY_REQUIRED,
            "degenerate_scale": DEGENERATE_SCALE,
            "statement": (
                "on the finest mesh (5120 panels) the third-party code must agree with "
                "the multipole reference within 1 % for every non-degenerate item, and "
                "the deviation must shrink for at least 90 % of the items; absolute "
                "deviation of a *coarse* mesh is never used as a statement about the "
                "reference"),
        },
        "settings": settings,
        "primary_setting": gf_settings[-1],
        "verdicts": {gf: settings[gf]["verdict"] for gf in gf_settings},
    }
    payload["verdict"] = settings[gf_settings[-1]]["verdict"]

    if case_numbers is not None and case_numbers.exists():
        numbers = json.loads(case_numbers.read_text())
        block = _tensorbem_vs_capytaine(
            numbers.get("band_sweep_case", []), finest_values, band_scale)
        if block is not None:
            payload["tensorbem_vs_capytaine_finest"] = block
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subdivisions", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--ka", type=float, nargs="+", default=list(BAND_KA))
    parser.add_argument("--gf", nargs="+", default=list(GF_SETTINGS), choices=GF_SETTINGS)
    parser.add_argument("--outdocs", type=Path,
                        default=Path("docs/case_bem_submerged_sphere_havelock"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    payload = run(args.subdivisions, args.ka, args.gf)
    args.outdocs.mkdir(parents=True, exist_ok=True)
    out = args.out or (args.outdocs / "capytaine_refinement_ladder.json")
    out.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    for gf in args.gf:
        print(gf, json.dumps(payload["settings"][gf]["summary"]))
    print(f"verdict: {payload['verdict']} -> {out}")
    return 0 if payload["verdict"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())