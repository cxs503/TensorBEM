#!/usr/bin/env python3
"""Board-audit m17303: an *independent third-party* referee for the audited
submerged-sphere anchor ``CASE-BEM-SUBMERGED-SPHERE-HAVELOCK``.

Why this script exists
----------------------
The audited delivery (mission m17242, TensorBEM @ 1ae83e1) declares the
multipole solution ``tensorbem.analytic.submerged_sphere_wave_multipole`` as
its reference.  That reference is a different formulation from the panel BEM
(closed-form Thorne/Wang/Linton multipoles, no quadrature), so it is not the
same code -- but it *is* the same repository, and ``tensor_benchmark_cases``
holds **no registered referee row** for this anchor (0 rows at audit time).

This script supplies the missing external axis: capytaine (Nemoh successor),
an independently written Rankine panel code, solves the same linearised
free-surface radiation problem for the same geometry
(sphere a = 1 m, centre z = -2 m, deep water, rho = 1025, g = 9.80665).

capytaine is not a test dependency of TensorBEM; it happens to be installed in
this audit environment (3.1.dev0, built from source on loongarch64).  Only this
producer script imports it; the numbers are frozen into
``artifacts/audit_m17303/capytaine_independent.json`` with provenance.

Usage::

    python scripts/audit/m17303_capytaine_independent.py \
        --out artifacts/audit_m17303/capytaine_independent.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

DENSITY = 1025.0
GRAVITY = 9.80665
RADIUS = 1.0
SUBMERGENCE = 2.0
KA = 1.0
#: capytaine mesh resolutions (n_theta, n_phi) -- refinement ladder
RESOLUTIONS = ((8, 16), (12, 24), (16, 32), (24, 48))


def _solve_level(cpt: Any, solver: Any, resolution: Sequence[int]) -> dict[str, Any]:
    mesh = cpt.mesh_sphere(
        radius=RADIUS, center=(0.0, 0.0, -SUBMERGENCE), resolution=tuple(int(r) for r in resolution)
    )
    body = cpt.FloatingBody(mesh=mesh)
    body.add_all_rigid_body_dofs()
    k = KA / RADIUS
    omega = math.sqrt(GRAVITY * k)
    volume = (4.0 / 3.0) * math.pi * RADIUS**3
    out: dict[str, Any] = {"resolution": [int(r) for r in resolution],
                           "n_faces": int(mesh.nb_faces),
                           "mesh_volume": float(mesh.volume),
                           "omega": omega}
    for dof in ("Heave", "Surge"):
        res = solver.solve(
            cpt.RadiationProblem(
                body=body, omega=omega, radiating_dof=dof,
                water_depth=np.inf, rho=DENSITY, g=GRAVITY,
            )
        )
        added_mass = float(res.added_mass[dof])
        damping = float(res.radiation_damping[dof])
        out[dof.lower()] = {
            "added_mass": added_mass,
            "damping": damping,
            "a_dimless": added_mass / (DENSITY * volume),
            "b_dimless": damping / (DENSITY * omega * volume),
        }
    return out


def build(out_path: Path) -> dict[str, Any]:
    import capytaine as cpt

    solver = cpt.BEMSolver()
    levels = [_solve_level(cpt, solver, res) for res in RESOLUTIONS]
    payload = {
        "case": "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
        "role": "independent third-party referee for the board audit m17303",
        "geometry": {
            "body": "fully submerged sphere",
            "radius_m": RADIUS,
            "centre_z_m": -SUBMERGENCE,
            "submergence_ratio_f_over_a": SUBMERGENCE / RADIUS,
            "density": DENSITY,
            "gravity": GRAVITY,
            "water_depth": "infinite (np.inf)",
        },
        "ka": KA,
        "normalisation": "A' = A/(rho*V), B' = B/(rho*omega*V), V = 4/3*pi*a^3",
        "displaced_volume": (4.0 / 3.0) * math.pi * RADIUS**3,
        "levels": levels,
        "finest": levels[-1],
        "provenance": {
            "tool": "capytaine",
            "version": getattr(cpt, "__version__", "unknown"),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    return payload


def main() -> None:  # pragma: no cover - producer CLI
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/audit_m17303/capytaine_independent.json")
    a = ap.parse_args()
    payload = build(Path(a.out))
    fin = payload["finest"]
    print("capytaine finest level n_faces=%d" % fin["n_faces"])
    for dof in ("heave", "surge"):
        print("  %-5s A'=%.6f  B'=%.6f" % (dof, fin[dof]["a_dimless"], fin[dof]["b_dimless"]))


if __name__ == "__main__":
    main()