#!/usr/bin/env python3
"""Compute ka=4.0,5.0 on the case mesh (subdivision=3, 1280 panels) for Havelock submerged sphere.

Outputs JSON to stdout with BEM-computed values, condition numbers, and residuals.
"""
from __future__ import annotations
import json, sys
sys.path.insert(0, "src")

from tensorbem.analytic import submerged_sphere_wave_multipole
from tensorbem.benchmarks.submerged_sphere_havelock import (
    AREA_MEASURE_SPHERICAL, panel_area_measure, run_frequency_sweep
)

KA_VALUES = (4.0, 5.0)
RADIUS = 1.0
SUBMERGENCE = 2.0
DENSITY = 1025.0
GRAVITY = 9.80665
SUBDIVISION = 3  # 1280 panels

rows = run_frequency_sweep(
    ka_values=KA_VALUES,
    radius=RADIUS,
    submergence=SUBMERGENCE,
    subdivision=SUBDIVISION,
    density=DENSITY,
    gravity=GRAVITY,
    area_measure=AREA_MEASURE_SPHERICAL,
)

# Enrich with relative_residual from the solver
output = []
for row in rows:
    # Re-solve to get condition_number and relative_residual
    from tensorbem.geometry3d import submerged_sphere_mesh
    from tensorbem.free_surface_solver3d import assemble_free_surface_operators_3d, solve_radiation_3d
    
    ka = row["ka"]
    nu = ka / RADIUS
    mesh = submerged_sphere_mesh(SUBDIVISION, radius=RADIUS, submergence=SUBMERGENCE)
    areas = panel_area_measure(mesh, radius=RADIUS, submergence=SUBMERGENCE, area_measure=AREA_MEASURE_SPHERICAL)
    ops = assemble_free_surface_operators_3d(mesh, nu, panel_areas=areas)
    rad = solve_radiation_3d(
        mesh, nu, center=(0.0, 0.0, -SUBMERGENCE),
        density=DENSITY, gravity=GRAVITY, operators=ops, panel_areas=areas,
    )
    
    row["condition_number"] = rad.condition_number
    row["relative_residual"] = rad.relative_residual
    output.append(row)

print(json.dumps(output, indent=2))
