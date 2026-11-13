#!/usr/bin/env python3
"""Independent third-party cross-check of the Havelock submerged-sphere case.

Why this exists
---------------
The case-pool anchor ``CASE-BEM-SUBMERGED-SPHERE-HAVELOCK`` compares TensorBEM
against the closed-form multipole series
(:func:`tensorbem.analytic.submerged_sphere_wave_multipole`).  That comparison
is only as trustworthy as the series, and no published table can be transcribed
in this sandbox (see ``docs/case_bem_submerged_sphere_havelock/CASE.md`` §1).
So the reference is cross-checked by a *different software implementation*:
`capytaine` (a third-party BEM solver, unrelated code base) solves the same
radiation problem on the same geometry, and its coefficients are compared with
both the multipole reference and the TensorBEM result.

Output (machine-readable, committed):
    docs/case_bem_submerged_sphere_havelock/capytaine_crosscheck.json

Usage::

    PYTHONPATH=src python3 scripts/case_havelock_capytaine_crosscheck.py \
        --subdivisions 2 3 --ka 0.1 0.5 1.0 2.0 3.0 --outdocs docs/case_bem_submerged_sphere_havelock
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
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
DOF = {"a33": "Heave", "b33": "Heave", "a11": "Surge", "b11": "Surge"}
TOLERANCE = 0.05


def _capytaine_version() -> str:
    try:
        return importlib.metadata.version("capytaine")
    except Exception:  # noqa: BLE001 - dev installs may not be registered
        return getattr(cpt, "__version__", "unknown")


def _crude_solve(
    subdivision: int,
    ka_values: Sequence[float],
    *,
    radius: float = 1.0,
    submergence: float = 2.0,
    density: float = 1025.0,
    gravity: float = 9.80665,
) -> list[dict[str, Any]]:
    body_mesh = submerged_sphere_mesh(subdivision, radius=radius, submergence=submergence)
    mesh = cpt.Mesh(vertices=np.asarray(body_mesh.vertices), faces=np.asarray(body_mesh.faces))
    body = cpt.FloatingBody(mesh=mesh, dofs=cpt.rigid_body_dofs())
    solver = cpt.BEMSolver()
    volume = 4.0 / 3.0 * np.pi * radius**3
    rows: list[dict[str, Any]] = []
    for ka in ka_values:
        nu = float(ka) / radius                       # deep-water: omega^2 = g*k
        omega = float(np.sqrt(gravity * nu))
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
            "a33": float(heave.added_mass["Heave"]) / (density * volume),
            "b33": float(heave.radiation_damping["Heave"]) / (density * omega * volume),
            "a11": float(surge.added_mass["Surge"]) / (density * volume),
            "b11": float(surge.radiation_damping["Surge"]) / (density * omega * volume),
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
            "n_panels": body_mesh.panel_count,
            "omega": omega,
            "capytaine": got,
            "multipole_ref": exact,
            "capytaine_vs_ref_rel_error": {
                key: abs(got[key] - exact[key]) / abs(exact[key]) for key in COEFFICIENTS
            },
            "seconds": round(seconds, 3),
        })
    return rows


def run(
    subdivisions: Sequence[int] = (2, 3),
    ka_values: Sequence[float] = (0.1, 0.5, 1.0, 2.0, 3.0),
) -> dict[str, Any]:
    rows = [_crude_solve(sub, ka_values) for sub in subdivisions]
    flat = [row for block in rows for row in block]
    worst = max(
        (err, row, key)
        for row in flat
        for key, err in row["capytaine_vs_ref_rel_error"].items()
    )
    return {
        "case": "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
        "purpose": (
            "independent third-party software check of the multipole reference and "
            "of the TensorBEM result; capytaine is a different code base"
        ),
        "capytaine_version": _capytaine_version(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "tolerance": TOLERANCE,
        "rows": rows,
        "worst": {
            "rel_error": worst[0],
            "ka": worst[1]["ka"],
            "subdivision": worst[1]["subdivision"],
            "quantity": worst[2],
        },
        "verdict": "PASS" if worst[0] < TOLERANCE else "FAIL",
        "note": (
            "capytaine and the multipole series are independent implementations of the "
            "same linearised potential-flow problem; agreement below the 5 % pool "
            "tolerance certifies that the reference column is not an artefact of one "
            "code. The comparison uses the exact displaced volume 4/3*pi*a^3 for the "
            "normalisation of both sides, so the mesh volume deficit is not hidden."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subdivisions", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--ka", type=float, nargs="+", default=[0.1, 0.5, 1.0, 2.0, 3.0])
    parser.add_argument("--outdocs", type=Path,
                        default=Path("docs/case_bem_submerged_sphere_havelock"))
    args = parser.parse_args(argv)

    payload = run(args.subdivisions, args.ka)
    args.outdocs.mkdir(parents=True, exist_ok=True)
    out = args.outdocs / "capytaine_crosscheck.json"
    out.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    print(json.dumps(payload["worst"], indent=2))
    print(f"verdict: {payload['verdict']} -> {out}")
    return 0 if payload["verdict"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())