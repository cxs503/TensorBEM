#!/usr/bin/env python3
"""Compute ka=4.0,5.0 on 1280-panel (subdivision=3) case mesh for Havelock submerged sphere.

Outputs JSON with the same schema as band_sweep_case rows in numbers.json.
This replaces the GCI-estimated placeholder values (condition_number=2.05) with
actual BEM computation results.

Usage:
    PYTHONPATH=src python3 compute_ka45_case_mesh.py --outdir /tmp/ka45_results
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from tensorbem.analytic import submerged_sphere_wave_multipole
from tensorbem.benchmarks.case_havelock_submerged_sphere import (
    assemble_free_surface_operators_3d,
    panel_area_measure,
    solve_radiation_3d,
)
from tensorbem.geometry3d import submerged_sphere_mesh

RADIUS = 1.0
SUBMERGENCE = 2.0
DENSITY = 1025.0
GRAVITY = 9.80665
SUBDIVISION = 3  # 1280 panels
KA_VALUES = (4.0, 5.0)
COEFFICIENTS = ("a33", "b33", "a11", "b11")


def compute_single_ka(ka: float) -> dict:
    """Compute radiation coefficients for a single ka on the case mesh."""
    t0 = time.time()
    print(f"[compute_ka45] ka={ka}: building mesh (subdivision={SUBDIVISION})...", flush=True)
    mesh = submerged_sphere_mesh(SUBDIVISION, radius=RADIUS, submergence=SUBMERGENCE)
    print(f"[compute_ka45] ka={ka}: mesh has {mesh.panel_count} panels", flush=True)

    # Compute nu (wavenumber) from ka: ka = nu * a => nu = ka / a
    nu = ka / RADIUS

    # Reference (for volume and omega)
    ref = submerged_sphere_wave_multipole(
        ka, radius=RADIUS, submergence=SUBMERGENCE,
        density=DENSITY, gravity=GRAVITY,
    )
    volume = ref.displaced_volume
    omega = ref.omega

    # Assemble operators and solve
    print(f"[compute_ka45] ka={ka}: assembling operators (nu={nu:.4f})...", flush=True)
    areas = panel_area_measure(mesh, radius=RADIUS, submergence=SUBMERGENCE, area_measure="flat")
    ops = assemble_free_surface_operators_3d(mesh, nu, panel_areas=areas)

    print(f"[compute_ka45] ka={ka}: solving BEM...", flush=True)
    rad = solve_radiation_3d(
        mesh, nu, center=(0.0, 0.0, -SUBMERGENCE),
        density=DENSITY, gravity=GRAVITY, operators=ops, panel_areas=areas,
    )

    elapsed = time.time() - t0
    print(f"[compute_ka45] ka={ka}: solved in {elapsed:.1f}s, cond={rad.condition_number:.6f}", flush=True)

    # Extract BEM results (dimensionless)
    a33_bem = float(rad.added_mass[2, 2] / (DENSITY * volume))
    b33_bem = float(rad.damping[2, 2] / (DENSITY * omega * volume))
    a11_bem = float(rad.added_mass[0, 0] / (DENSITY * volume))
    b11_bem = float(rad.damping[0, 0] / (DENSITY * omega * volume))

    # Reference values (dimensionless)
    a33_exact = float(ref.heave_added_mass_dimless)
    b33_exact = float(ref.heave_damping_dimless)
    a11_exact = float(ref.surge_added_mass_dimless)
    b11_exact = float(ref.surge_damping_dimless)

    def rel_error(got, ref_val):
        if abs(ref_val) < 1e-30:
            return abs(got - ref_val)
        return abs(got - ref_val) / abs(ref_val)

    return {
        "ka": ka,
        "n_panels": mesh.panel_count,
        "a33_bem": a33_bem,
        "a33_exact": a33_exact,
        "a33_rel_error": rel_error(a33_bem, a33_exact),
        "b33_bem": b33_bem,
        "b33_exact": b33_exact,
        "b33_rel_error": rel_error(b33_bem, b33_exact),
        "a11_bem": a11_bem,
        "a11_exact": a11_exact,
        "a11_rel_error": rel_error(a11_bem, a11_exact),
        "b11_bem": b11_bem,
        "b11_exact": b11_exact,
        "b11_rel_error": rel_error(b11_bem, b11_exact),
        "condition_number": float(rad.condition_number),
        "elapsed_seconds": float(elapsed),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="/tmp/ka45_results")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    for ka in KA_VALUES:
        row = compute_single_ka(ka)
        results.append(row)

    outpath = outdir / "ka45_case_mesh.json"
    outpath.write_text(json.dumps(results, indent=2))
    print(f"[compute_ka45] Results written to {outpath}", flush=True)

    # Summary
    for r in results:
        print(f"  ka={r['ka']}: a33_err={r['a33_rel_error']:.4%}, b33_err={r['b33_rel_error']:.4%}, "
              f"a11_err={r['a11_rel_error']:.4%}, b11_err={r['b11_rel_error']:.4%}, "
              f"cond={r['condition_number']:.6f}", flush=True)


if __name__ == "__main__":
    main()
