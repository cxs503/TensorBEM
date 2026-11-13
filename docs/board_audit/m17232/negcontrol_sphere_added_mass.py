#!/usr/bin/env python3
"""board-audit m17232 negative control for m17176 (CASE-BEM-SPHERE-ADDED-MASS).

Question: is the reported added mass actually *computed* from the boundary data,
or is ``0.5*rho*V`` baked into the driver?

Two probes, both at subdivision 1 (80 panels, fast):
  1. wrong angular mode: impose ``u = -0.5*U*a*cos(2*theta)`` instead of the
     translating-sphere trace ``cos(theta)``. A genuine solve must return an
     added mass far from ``0.5*rho*V``.
  2. geometric scaling: radius 2.0 must return 8x the radius-1.0 value
     (m_A ~ a^3) with the *same* relative error (the error is scale free).

Both probes call the same production entry points used by
``tensorbem.benchmarks.sphere_added_mass.run`` (icosphere + solve_laplace_surface),
so no private re-implementation is involved.

Run: PYTHONPATH=src python3 docs/board_audit/m17232/negcontrol_sphere_added_mass.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tensorbem.benchmarks.sphere_added_mass import run
from tensorbem.geometry3d import icosphere
from tensorbem.solver3d import solve_laplace_surface

HERE = Path(__file__).resolve().parent


def added_mass_for_trace(mode: int, subdivisions: int = 1, radius: float = 1.0,
                         speed: float = 1.0, density: float = 1025.0) -> float:
    mesh = icosphere(subdivisions, radius=radius)
    centroids = mesh.centroids
    radial = np.linalg.norm(centroids, axis=1)
    cos_theta = centroids[:, 0] / radial
    if mode == 1:
        potential = -0.5 * speed * radius * cos_theta
    elif mode == 2:
        potential = -0.5 * speed * radius * (2.0 * cos_theta**2 - 1.0)  # cos(2 theta)
    else:  # pragma: no cover - defensive
        raise ValueError(mode)
    result = solve_laplace_surface(mesh, potential, domain="exterior")
    return float(density / speed**2 * np.sum(potential * result.normal_derivative * mesh.areas))


def main() -> None:
    base = run(1, 1.0, 1.0, 1025.0)
    r2 = run(1, 2.0, 1.0, 1025.0)
    m_mode1 = added_mass_for_trace(1)
    m_mode2 = added_mass_for_trace(2)
    ref = 0.5 * 1025.0 * 4.0 / 3.0 * math.pi
    out = {
        "subdivisions": 1,
        "panels": base["panels"],
        "production_run_mode1_via_driver": m_mode1,
        "artifact_mode1": base["added_mass"],
        "driver_matches_direct_solve": abs(m_mode1 - base["added_mass"]) / base["added_mass"],
        "ref_0p5_rho_V": ref,
        "mode1_relative_error": base["added_mass_relative_error"],
        "wrong_mode2_cos2theta_added_mass": m_mode2,
        "wrong_mode2_relative_error_vs_ref": abs(m_mode2 - ref) / ref,
        "mode2_vs_mode1_ratio": m_mode2 / m_mode1,
        "radius2_added_mass": r2["added_mass"],
        "radius2_over_radius1": r2["added_mass"] / base["added_mass"],
        "radius2_relative_error": r2["added_mass_relative_error"],
        "radius1_relative_error": base["added_mass_relative_error"],
        "verdict": {
            "output_follows_input": abs(m_mode2 - ref) / ref > 0.10,
            "radius_scaling_close_to_8": abs(r2["added_mass"] / base["added_mass"] - 8.0) < 0.05,
            "relative_error_scale_invariant": abs(
                r2["added_mass_relative_error"] - base["added_mass_relative_error"]) < 1e-12,
        },
    }
    (HERE / "negcontrol_sphere_added_mass.json").write_text(
        json.dumps(out, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()