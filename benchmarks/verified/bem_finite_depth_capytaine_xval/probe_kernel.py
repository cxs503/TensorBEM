#!/usr/bin/env python3
"""Cheap diagnostics for the finite-depth 3-D kernel (no full study).

Two questions, each answered by a handful of 320-panel solves:

A. **Flat-panel area deficit.**  The inscribed icosphere underestimates the
   wetted surface (``spherical_patch_areas`` docstring: ~1.9 % at subdivision 2).
   The cross-validation showed a *frequency-independent* ~8 % low bias in A33 at
   320 panels, which smells like a geometry bias rather than a kernel error.
   Solve the same case with the exact surface measure and see how much of the
   bias disappears.

B. **Mode-count stability.**  A previous cluster round produced A33 values that
   exploded with the eigenseries truncation (M = 48 -> 9786, M = 64 -> 463427,
   M = 96 -> -2175 on the same physical case) and, for several truncations,
   ``numpy.linalg.LinAlgError: Eigenvalues did not converge``.  Reproduce the
   sweep on the *current* code and record the coefficient vs ``mode_count``.

Usage::

    PYTHONPATH=src python benchmarks/verified/bem_finite_depth_capytaine_xval/probe_kernel.py area
    PYTHONPATH=src python benchmarks/verified/bem_finite_depth_capytaine_xval/probe_kernel.py modes
"""

from __future__ import annotations

import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

RADIUS = 1.0
SUBMERGENCE = 2.0
DENSITY = 1025.0
GRAVITY = 9.80665
OUT = Path("benchmarks/verified/bem_finite_depth_capytaine_xval/cases")


def _solve(ka, depth_ratio, subdiv, mode_count=None, exact_areas=False):
    from tensorbem.geometry3d import submerged_sphere_mesh, spherical_patch_areas
    from tensorbem.finite_depth_solver3d import (
        assemble_finite_depth_operators_3d,
        solve_radiation_finite_depth_3d,
    )

    mesh = submerged_sphere_mesh(subdiv, radius=RADIUS, submergence=SUBMERGENCE)
    depth = float(depth_ratio) * RADIUS
    areas = None
    if exact_areas:
        areas = spherical_patch_areas(
            mesh.vertices, mesh.faces, center=(0.0, 0.0, -SUBMERGENCE), radius=RADIUS
        )
        assert abs(float(np.sum(areas)) / (4.0 * math.pi * RADIUS**2) - 1.0) < 1e-9
    t0 = time.time()
    ops = assemble_finite_depth_operators_3d(
        mesh, ka / RADIUS, depth, mode_count=mode_count, panel_areas=areas
    )
    res = solve_radiation_finite_depth_3d(
        mesh, ka / RADIUS, depth, modes=[0, 1, 2], density=DENSITY,
        gravity=GRAVITY, operators=ops, panel_areas=areas,
    )
    return {
        "ka": ka,
        "depth_ratio": float(depth_ratio),
        "subdivisions": int(subdiv),
        "n_panels": int(mesh.panel_count),
        "mode_count": int(ops.mode_count),
        "exact_areas": bool(exact_areas),
        "added_mass_heave": float(res.added_mass[2]),
        "damping_heave": float(res.damping[2]),
        "added_mass_surge": float(res.added_mass[0]),
        "condition_number": float(res.condition_number),
        "relative_residual": float(res.relative_residual),
        "seconds": round(time.time() - t0, 1),
    }


def run_area() -> list[dict]:
    rows = []
    for exact in (False, True):
        try:
            row = _solve(1.0, 12.0, 2, mode_count=48, exact_areas=exact)
        except Exception:  # pragma: no cover - diagnostics
            row = {"error": traceback.format_exc(), "exact_areas": exact}
        print(json.dumps(row, default=str), flush=True)
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probe_area.json").write_text(json.dumps(rows, indent=2, default=str))
    return rows


def run_modes() -> list[dict]:
    rows = []
    for modes in (8, 16, 24, 32, 48, 64):
        try:
            row = _solve(1.0, 5.0, 2, mode_count=modes)
            row["requested_mode_count"] = modes
        except Exception:
            row = {"requested_mode_count": modes, "error": traceback.format_exc()}
        print(json.dumps(row, default=str), flush=True)
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probe_modes.json").write_text(json.dumps(rows, indent=2, default=str))
    return rows


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "area"
    (run_area if which == "area" else run_modes)()