#!/usr/bin/env python3
"""High-``ka`` near-field vs far-field damping cross-check (Havelock submerged sphere).

Why this study
--------------
The case pool gates 28 numbers (7 wavenumbers x a33/b33/a11/b11) against the
multipole reference on the 1280-panel case mesh.  At ``ka >= 4`` the damping
coefficients decay to ``B' ~ 1e-6 .. 1e-5``: the near-field value

    B_jj = -rho * Im[ int_{Gamma_B} phi_j n_j dS ] * omega

is then a *small difference of large complex numbers*, so a 5 % relative gate on
those rows says nothing about the discretisation error unless the integral is
shown to be free of a cancellation floor.  This script supplies exactly that
evidence, using a route that does not share the near-field cancellation:

    B_ij^far = (rho * omega * k / 4 pi) * int_0^{2pi} K_i(theta) conj(K_j(theta)) dtheta,

with ``K`` the Kochin function evaluated by
:func:`tensorbem.free_surface_solver3d.kochin_function_3d`.  The far-field route is
an integral over the same surface potentials but weighted by a *smooth* oscillatory
kernel, so its error is a quadrature bias rather than a difference of nearly equal
terms.

What is measured, per (subdivision, ka)
---------------------------------------
* ``b33_near`` / ``b11_near``  -- near-field (pressure) damping, the gated route;
* ``b33_far``  / ``b11_far``   -- far-field (Kochin/energy) damping;
* ``err_*_vs_ref``             -- both routes against the multipole reference;
* ``near_vs_far``              -- |near - far| / |far|: the route-independent
  consistency indicator (no reference used);
* ``azimuth_convergence``      -- far-field value at n_azimuth = 72 / 144 / 288,
  so the documented setting is justified rather than assumed;
* ``err_scaled``               -- |B - B_ref| / (rho*omega*nabla), the
  scale-free error, which stays meaningful when ``B_ref`` itself is tiny.

The output JSON is consumed by ``tests/test_casepool_havelock_far_field_high_ka.py``.

Run (cheap rung, local)::

    PYTHONPATH=src python3 \
        benchmarks/verified/casepool_bem_submerged_sphere_havelock/far_field_high_ka.py \
        --subdivisions 2 --kas 2.0,4.0 --n-azimuth 72,144 \
        --out /tmp/havelock_farfield_smoke.json

Run (case mesh, cluster)::

    PYTHONPATH=src python3 .../far_field_high_ka.py \
        --subdivisions 2,3 --kas 0.5,1.0,2.0,3.0,4.0,5.0 --n-azimuth 72,144,288 \
        --out docs/case_bem_submerged_sphere_havelock/far_field_high_ka.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorbem.analytic import submerged_sphere_wave_multipole
from tensorbem.free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    far_field_damping_matrix_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import submerged_sphere_mesh

CASE_CODE = "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK"
RADIUS = 1.0
SUBMERGENCE = 2.0
DENSITY = 1025.0
GRAVITY = 9.80665
DISPLACED_VOLUME_FACTOR = 4.0 / 3.0 * np.pi
MODES = (0, 2)  # surge, heave
SURGE, HEAVE = 0, 1  # indices into the (2, 2) matrices returned by the solver


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            args,
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except Exception:  # pragma: no cover - provenance is best effort
        return ""


def _rel(value: float, exact: float) -> float:
    denom = abs(exact)
    if denom == 0.0:
        return 0.0 if value == 0.0 else math.inf
    return abs(value - exact) / denom


def _scaled_error(value: float, exact: float, scale: float) -> float:
    return abs(value - exact) / scale


def run_point(
    subdivision: int,
    ka: float,
    *,
    azimuths: Sequence[int],
    radius: float = RADIUS,
    submergence: float = SUBMERGENCE,
    density: float = DENSITY,
    gravity: float = GRAVITY,
    n_terms: int = 20,
) -> dict[str, Any]:
    """One (mesh, frequency) working point."""
    nu = float(ka) / radius
    mesh = submerged_sphere_mesh(subdivision, radius=radius, submergence=submergence)
    ref = submerged_sphere_wave_multipole(
        float(ka),
        radius=radius,
        submergence=submergence,
        density=density,
        gravity=gravity,
        n_terms=n_terms,
    )
    centre = (0.0, 0.0, -float(submergence))

    t0 = time.perf_counter()
    ops = assemble_free_surface_operators_3d(mesh, nu)
    rad = solve_radiation_3d(
        mesh,
        nu,
        modes=MODES,
        center=centre,
        density=density,
        gravity=gravity,
        operators=ops,
    )
    t_solve = time.perf_counter() - t0

    vol_exact = float(ref.displaced_volume)
    scale_b = density * float(ref.omega) * vol_exact
    scale_a = density * vol_exact

    # Near-field (pressure) route: this is what the case pool gates.
    b_near = {
        "b33": float(rad.damping[HEAVE, HEAVE]) / scale_b,
        "b11": float(rad.damping[SURGE, SURGE]) / scale_b,
    }
    b_ref = {
        "b33": float(ref.heave_damping_dimless),
        "b11": float(ref.surge_damping_dimless),
    }
    a_near = {
        "a33": float(rad.added_mass[HEAVE, HEAVE]) / scale_a,
        "a11": float(rad.added_mass[SURGE, SURGE]) / scale_a,
    }
    a_ref = {
        "a33": float(ref.heave_added_mass_dimless),
        "a11": float(ref.surge_added_mass_dimless),
    }

    # Far-field (Kochin/energy) route, with an azimuth-resolution ladder.
    far_by_naz: dict[str, Any] = {}
    for n_az in azimuths:
        t1 = time.perf_counter()
        far = far_field_damping_matrix_3d(
            mesh, rad, density=density, gravity=gravity, n_azimuth=int(n_az)
        )
        far_by_naz[str(int(n_az))] = {
            "b33": float(far[HEAVE, HEAVE]) / scale_b,
            "b11": float(far[SURGE, SURGE]) / scale_b,
            "b31": float(far[HEAVE, SURGE]) / scale_b,
            "seconds": time.perf_counter() - t1,
        }

    finest = str(int(max(azimuths)))
    b_far = {key: far_by_naz[finest][key] for key in ("b33", "b11")}

    az_conv = {}
    for key in ("b33", "b11"):
        values = [far_by_naz[str(int(n))][key] for n in sorted(azimuths)]
        if len(values) > 1 and abs(values[-1]) > 0.0:
            az_conv[key] = abs(values[-2] - values[-1]) / abs(values[-1])
        else:
            az_conv[key] = float("nan")

    return {
        "subdivision": int(subdivision),
        "n_panels": int(mesh.panel_count),
        "h_mean": float(mesh.h_mean),
        "mesh_volume_ratio": float(mesh.volume) / vol_exact,
        "ka": float(ka),
        "omega": float(ref.omega),
        "wavenumber": nu,
        "condition_number": float(rad.condition_number),
        "relative_residual": float(rad.relative_residual),
        "solve_method": rad.solve_method,
        "seconds_solve": t_solve,
        "added_mass": {
            "near": a_near,
            "ref": a_ref,
            "err_rel": {k: _rel(a_near[k], a_ref[k]) for k in a_near},
            "err_scaled": {
                k: _scaled_error(a_near[k], a_ref[k], 0.5) for k in a_near
            },
        },
        "damping_near": {
            "value": b_near,
            "ref": b_ref,
            "err_rel": {k: _rel(b_near[k], b_ref[k]) for k in b_near},
            "err_scaled": {
                k: _scaled_error(b_near[k], b_ref[k], 1.0) for k in b_near
            },
        },
        "damping_far": {
            "value": b_far,
            "ref": b_ref,
            "err_rel": {k: _rel(b_far[k], b_ref[k]) for k in b_far},
            "err_scaled": {
                k: _scaled_error(b_far[k], b_ref[k], 1.0) for k in b_far
            },
            "by_azimuth": far_by_naz,
            "azimuth_convergence": az_conv,
            "n_azimuth_used": int(finest),
        },
        "near_vs_far": {
            k: (abs(b_near[k] - b_far[k]) / abs(b_far[k]) if abs(b_far[k]) > 0 else float("nan"))
            for k in b_near
        },
        "amplitude_ratio_a_over_b": {
            "a33_over_b33_ref": (
                abs(a_ref["a33"] / b_ref["b33"]) if b_ref["b33"] != 0.0 else float("inf")
            ),
            "a11_over_b11_ref": (
                abs(a_ref["a11"] / b_ref["b11"]) if b_ref["b11"] != 0.0 else float("inf")
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subdivisions", default="3")
    parser.add_argument("--kas", default="0.5,1.0,2.0,3.0,4.0,5.0")
    parser.add_argument("--n-azimuth", default="72,144,288")
    parser.add_argument("--n-terms", type=int, default=20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    subdivisions = [int(s) for s in args.subdivisions.split(",") if s.strip()]
    kas = [float(s) for s in args.kas.split(",") if s.strip()]
    azimuths = [int(s) for s in args.n_azimuth.split(",") if s.strip()]

    rows = []
    for subdivision in subdivisions:
        for ka in kas:
            t0 = time.perf_counter()
            row = run_point(subdivision, ka, azimuths=azimuths, n_terms=args.n_terms)
            row["seconds_total"] = time.perf_counter() - t0
            rows.append(row)
            print(
                f"[{subdivision}] ka={ka:<5} panels={row['n_panels']:<5} "
                f"b33 near={row['damping_near']['value']['b33']:.6e} "
                f"far={row['damping_far']['value']['b33']:.6e} "
                f"ref={row['damping_near']['ref']['b33']:.6e} "
                f"near_vs_far={row['near_vs_far']['b33']:.3e} "
                f"({row['seconds_total']:.1f}s)",
                flush=True,
            )

    payload = {
        "case": CASE_CODE,
        "purpose": (
            "high-ka near-field vs far-field (Kochin/energy) damping cross-check on the "
            "case mesh: separates a genuine quadrature bias from a near-field "
            "cancellation floor and justifies the Kochin azimuth resolution"
        ),
        "parameters": {
            "radius": RADIUS,
            "submergence": SUBMERGENCE,
            "submergence_ratio": SUBMERGENCE / RADIUS,
            "density": DENSITY,
            "gravity": GRAVITY,
            "modes": list(MODES),
            "mode_names": ["surge", "heave"],
            "kas": kas,
            "subdivisions": subdivisions,
            "n_azimuth": azimuths,
            "reference": "tensorbem.analytic.submerged_sphere_wave_multipole",
            "reference_terms": args.n_terms,
            "normalisation": (
                "A/(rho*nabla) and B/(rho*omega*nabla) with nabla = 4/3*pi*a^3 (exact "
                "sphere volume), i.e. the same convention as run_case.py and the "
                "case-pool numbers.json"
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "provenance": {
            "git_commit": _git("git", "rev-parse", "HEAD"),
            "git_branch": _git("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "script": "benchmarks/verified/casepool_bem_submerged_sphere_havelock/far_field_high_ka.py",
        },
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())