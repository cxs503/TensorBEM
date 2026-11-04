"""Accuracy envelope of the 3-D free-surface BEM over a (Ka, f/a) parameter map.

Why this exists
---------------
:mod:`tensorbem.benchmarks.submerged_sphere_wave` verifies the 3-D Kelvin
free-surface solver at **one** submergence ratio (``f/a = 1.5``) and sweeps only
the wavenumber.  The deeply-submerged analytic reference
(:func:`tensorbem.analytic.submerged_sphere_wave_multipole`, Wang 1986 / Linton
1991, zero numerical quadrature) is however valid for *every* ``(Ka, f/a)``, and
the whole point of a free-surface kernel is that it gets harder as the body
approaches ``z = 0`` (the mirror-image term of the double-layer kernel becomes
nearly singular).

This module therefore answers the question a single-point check cannot: **where
in the (Ka, f/a) plane does the merged solver actually hold the 3 % accuracy
standard, and what mesh budget buys it?**  It reports, per working point,

* added mass / damping of surge and heave normalised by the *exact displaced
  volume* ``rho*V`` (the convention of the published reference), with relative
  errors against the multipole solution;
* the same quantities re-normalised by the *polyhedral* mesh volume, which
  separates the leading geometric volume deficit of the triangulation from the
  hydrodynamic discretisation error;
* diffraction force errors, near-field vs far-field (Kochin) damping
  consistency, Haskind reciprocity residual and matrix condition number, i.e.
  reliability indicators that do not depend on the reference solution.

Run::

    python -m tensorbem.benchmarks.submerged_sphere_envelope --subdivision 2

The JSON payload is meant to be consumed by
``tests/test_submerged_sphere_wave_envelope.py`` and by the report in
``docs/m3134/``.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorbem.analytic import submerged_sphere_wave_multipole
from tensorbem.free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    far_field_damping_matrix_3d,
    haskind_excitation_force_3d,
    solve_diffraction_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import submerged_sphere_mesh

DEFAULT_KAS: tuple[float, ...] = (0.2, 0.5, 0.8, 1.2, 1.6, 2.0)
DEFAULT_RATIOS: tuple[float, ...] = (1.1, 1.25, 1.5, 2.0, 3.0)


def _relative_error(value: float, exact: float) -> float:
    denom = abs(exact)
    if denom == 0.0:
        return 0.0 if value == 0.0 else math.inf
    return abs(value - exact) / denom


def _guarded_relative(value: float, exact: float, scale: float) -> float:
    """Relative error, or NaN when the reference is small against ``scale``."""
    if abs(exact) < 0.1 * scale:
        return math.nan
    return abs(value - exact) / abs(exact)


def _complex_relative_error(value: complex, exact: complex) -> float:
    denom = abs(exact)
    if denom == 0.0:
        return 0.0 if abs(value) == 0.0 else math.inf
    return abs(value - exact) / denom


def evaluate_point(
    ka: float,
    submergence_ratio: float,
    *,
    subdivision: int = 2,
    radius: float = 1.0,
    amplitude: float = 1.0,
    density: float = 1025.0,
    gravity: float = 9.80665,
) -> dict[str, Any]:
    """Verify one ``(Ka, f/a)`` working point against the multipole reference."""
    submergence = float(submergence_ratio) * radius
    nu = float(ka) / radius
    mesh = submerged_sphere_mesh(int(subdivision), radius=radius, submergence=submergence)
    center = (0.0, 0.0, -submergence)

    ref = submerged_sphere_wave_multipole(
        float(ka),
        radius=radius,
        submergence=submergence,
        amplitude=amplitude,
        density=density,
        gravity=gravity,
    )
    ops = assemble_free_surface_operators_3d(mesh, nu)
    rad = solve_radiation_3d(
        mesh,
        nu,
        modes=(0, 2),
        center=center,
        density=density,
        gravity=gravity,
        operators=ops,
    )
    diff = solve_diffraction_3d(
        mesh,
        nu,
        amplitude=amplitude,
        center=center,
        density=density,
        gravity=gravity,
        operators=ops,
    )
    far = far_field_damping_matrix_3d(mesh, rad, density=density, gravity=gravity)
    hask = haskind_excitation_force_3d(
        mesh, rad, amplitude=amplitude, density=density, gravity=gravity
    )

    vol_exact = ref.displaced_volume
    vol_mesh = float(mesh.volume)
    exact_scale = density * vol_exact
    mesh_scale = density * vol_mesh

    bem_a11 = float(rad.added_mass[0, 0])
    bem_a33 = float(rad.added_mass[1, 1])
    bem_b11 = float(rad.damping[0, 0])
    bem_b33 = float(rad.damping[1, 1])
    exact_a11 = ref.surge_added_mass_dimless * exact_scale
    exact_a33 = ref.heave_added_mass_dimless * exact_scale
    exact_b11 = ref.surge_damping_dimless * density * ref.omega * vol_exact
    exact_b33 = ref.heave_damping_dimless * density * ref.omega * vol_exact

    # Re-normalisation by the polyhedral volume keeps the reference *dimensionless*
    # value; the ratio therefore isolates the part of the error that is not the
    # leading O(1 - V_mesh/V_sphere) volume scaling.
    bem_a11_mesh = bem_a11 / mesh_scale
    bem_a33_mesh = bem_a33 / mesh_scale
    exact_a11_mesh = exact_a11 / mesh_scale
    exact_a33_mesh = exact_a33 / mesh_scale
    bem_b11_mesh = bem_b11 / (density * ref.omega * vol_mesh)
    bem_b33_mesh = bem_b33 / (density * ref.omega * vol_mesh)
    exact_b11_mesh = exact_b11 / (density * ref.omega * vol_mesh)
    exact_b33_mesh = exact_b33 / (density * ref.omega * vol_mesh)

    near33 = float(rad.damping[1, 1])
    far33 = float(far[1, 1])
    return {
        "ka": float(ka),
        "f_over_a": float(submergence_ratio),
        "panels": int(mesh.panel_count),
        "h_max": float(mesh.h_max),
        "volume_ratio_mesh_over_exact": vol_mesh / vol_exact,
        "a11": bem_a11 / exact_scale,
        "a33": bem_a33 / exact_scale,
        "b11": bem_b11 / (density * ref.omega * vol_exact),
        "b33": bem_b33 / (density * ref.omega * vol_exact),
        "exact_a11": ref.surge_added_mass_dimless,
        "exact_a33": ref.heave_added_mass_dimless,
        "exact_b11": ref.surge_damping_dimless,
        "exact_b33": ref.heave_damping_dimless,
        "a11_mesh_norm": bem_a11_mesh,
        "a33_mesh_norm": bem_a33_mesh,
        "exact_a11_mesh_norm": exact_a11_mesh,
        "exact_a33_mesh_norm": exact_a33_mesh,
        # Primary metrics.  Near the free surface the heave added mass passes
        # through zero (see the f/a = 1.1 rows), so a *relative* error against A33
        # is meaningless there.  The scale-free, always-well-conditioned metric is
        # the absolute error measured in units of the unbounded-fluid added mass
        # 0.5*rho*V, and for damping in units of rho*omega*V.
        "err_a11_scaled": abs(bem_a11 - exact_a11) / (0.5 * exact_scale),
        "err_a33_scaled": abs(bem_a33 - exact_a33) / (0.5 * exact_scale),
        "err_b11_scaled": abs(bem_b11 - exact_b11) / (density * ref.omega * vol_exact),
        "err_b33_scaled": abs(bem_b33 - exact_b33) / (density * ref.omega * vol_exact),
        # Relative errors are reported only where the reference is not itself
        # small compared with 0.5*rho*V; NaN elsewhere, never a misleading zero.
        "err_a11_rel": _guarded_relative(bem_a11, exact_a11, 0.5 * exact_scale),
        "err_a33_rel": _guarded_relative(bem_a33, exact_a33, 0.5 * exact_scale),
        "err_b11_rel": _guarded_relative(bem_b11, exact_b11, density * ref.omega * vol_exact),
        "err_b33_rel": _guarded_relative(bem_b33, exact_b33, density * ref.omega * vol_exact),
        "err_a11_mesh_norm": abs(bem_a11_mesh - exact_a11_mesh) / (0.5 * mesh_scale),
        "err_a33_mesh_norm": abs(bem_a33_mesh - exact_a33_mesh) / (0.5 * mesh_scale),
        "force_x_rel_error": _complex_relative_error(
            complex(diff.excitation_force[0]), ref.excitation_force_x
        ),
        "force_z_rel_error": _complex_relative_error(
            complex(diff.excitation_force[2]), ref.excitation_force_z
        ),
        # Far-field Kochin reciprocity for heave: an identity the reference also
        # satisfies exactly, so it measures the solver's own consistency.
        "haskind_kochin_rel_error": _complex_relative_error(
            complex(hask["kochin_haskind"][1]),
            complex(hask["boundary_haskind"][1]),
        ),
        "near_far_damping_ratio_33": near33 / far33 if far33 != 0.0 else math.nan,
        "condition_number": float(rad.condition_number),
        "relative_residual": float(rad.relative_residual),
    }


def run_envelope(
    ka_values: Sequence[float] = DEFAULT_KAS,
    submergence_ratios: Sequence[float] = DEFAULT_RATIOS,
    *,
    subdivision: int = 2,
    radius: float = 1.0,
    density: float = 1025.0,
    gravity: float = 9.80665,
) -> dict[str, Any]:
    """Evaluate the whole ``(Ka, f/a)`` map and summarise the 3 % margins."""
    rows = [
        evaluate_point(
            float(ka),
            float(ratio),
            subdivision=subdivision,
            radius=radius,
            density=density,
            gravity=gravity,
        )
        for ratio in submergence_ratios
        for ka in ka_values
    ]
    errors = np.array([row["err_a33_scaled"] for row in rows], dtype=float).reshape(
        len(submergence_ratios), len(ka_values)
    )
    failures = [row for row in rows if row["err_a33_scaled"] > 0.03]
    worst = max(rows, key=lambda row: row["err_a33_scaled"])
    return {
        "benchmark": "submerged_sphere_3d_free_surface_accuracy_envelope",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "settings": {
            "subdivision": int(subdivision),
            "panels": rows[0]["panels"] if rows else 0,
            "ka_values": [float(ka) for ka in ka_values],
            "submergence_ratios": [float(r) for r in submergence_ratios],
            "radius": float(radius),
            "density": float(density),
            "gravity": float(gravity),
            "threshold": 0.03,
        },
        "rows": rows,
        "err_a33_scaled_map": errors.tolist(),
        "max_err_a33_scaled": float(errors.max()),
        "min_err_a33_scaled": float(errors.min()),
        "worst_point": {"ka": worst["ka"], "f_over_a": worst["f_over_a"]},
        "points_within_threshold": len(rows) - len(failures),
        "points_over_threshold": [
            {
                "ka": row["ka"],
                "f_over_a": row["f_over_a"],
                    "err_a33_scaled": row["err_a33_scaled"],
                "err_a33_rel": row["err_a33_rel"],
            }
            for row in failures
        ],
    }


def run_envelope_refinement(
    ka: float = 1.2,
    submergence_ratio: float = 1.1,
    subdivisions: Sequence[int] = (0, 1, 2),
    *,
    radius: float = 1.0,
    density: float = 1025.0,
    gravity: float = 9.80665,
) -> dict[str, Any]:
    """Mesh refinement at one envelope corner, with the observed convergence order.

    The order is measured, never assumed: it is what tells us whether a coarse
    row of the envelope is discretisation-limited or physics-limited.
    """
    levels: list[dict[str, Any]] = []
    for sub in subdivisions:
        row = evaluate_point(
            ka,
            submergence_ratio,
            subdivision=int(sub),
            radius=radius,
            density=density,
            gravity=gravity,
        )
        levels.append(row)

    orders: list[float | None] = [None]
    for idx in range(1, len(levels)):
        h0, h1 = levels[idx - 1]["h_max"], levels[idx]["h_max"]
        e0, e1 = levels[idx - 1]["err_a33_scaled"], levels[idx]["err_a33_scaled"]
        if e0 <= 0.0 or e1 <= 0.0 or h0 <= h1:
            orders.append(None)
        else:
            orders.append(math.log(e0 / e1) / math.log(h0 / h1))

    return {
        "ka": float(ka),
        "f_over_a": float(submergence_ratio),
        "subdivisions": [int(s) for s in subdivisions],
        "levels": levels,
        "observed_orders_a33": orders,
        "finest_err_a33_scaled": levels[-1]["err_a33_scaled"],
        "finest_err_a33_mesh_norm": levels[-1]["err_a33_mesh_norm"],
    }


def mesh_budget(
    ka_values: Sequence[float] = DEFAULT_KAS,
    submergence_ratios: Sequence[float] = DEFAULT_RATIOS,
    *,
    subdivisions: Sequence[int] = (0, 1, 2),
    density: float = 1025.0,
    gravity: float = 9.80665,
) -> list[dict[str, Any]]:
    """Smallest tested subdivision whose ``A33`` error stays below 3 % per point."""
    out: list[dict[str, Any]] = []
    for ratio in submergence_ratios:
        for ka in ka_values:
            errs: list[tuple[int, float]] = []
            for sub in subdivisions:
                row = evaluate_point(
                    ka,
                    ratio,
                    subdivision=int(sub),
                    density=density,
                    gravity=gravity,
                )
                errs.append((int(sub), row["err_a33_scaled"]))
            passing = [sub for sub, err in errs if err <= 0.03]
            out.append(
                {
                    "ka": float(ka),
                    "f_over_a": float(ratio),
                    "errors_by_subdivision": {str(sub): err for sub, err in errs},
                    "minimal_passing_subdivision": min(passing) if passing else None,
                }
            )
    return out


def run(
    ka_values: Sequence[float] = DEFAULT_KAS,
    submergence_ratios: Sequence[float] = DEFAULT_RATIOS,
    *,
    subdivision: int = 2,
    refinement_corners: Sequence[tuple[float, float]] = ((1.2, 1.1), (0.8, 1.5), (0.2, 3.0)),
    include_budget: bool = True,
    density: float = 1025.0,
    gravity: float = 9.80665,
) -> dict[str, Any]:
    payload = run_envelope(
        ka_values,
        submergence_ratios,
        subdivision=subdivision,
        density=density,
        gravity=gravity,
    )
    payload["refinement"] = [
        run_envelope_refinement(
            ka=ka, submergence_ratio=ratio, density=density, gravity=gravity
        )
        for ka, ratio in refinement_corners
    ]
    if include_budget:
        # A 3 x 3 corner subset is enough to state the budget law; running all
        # three subdivisions at every map cell would triple the cost for no extra
        # information (the refinement corners already fix the order).
        payload["mesh_budget"] = mesh_budget(
            ka_values=(0.2, 1.2, 2.0),
            submergence_ratios=(1.1, 1.5, 3.0),
            density=density,
            gravity=gravity,
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subdivision", type=int, default=2)
    parser.add_argument("--kas", type=float, nargs="+", default=list(DEFAULT_KAS))
    parser.add_argument(
        "--ratios", type=float, nargs="+", default=list(DEFAULT_RATIOS)
    )
    parser.add_argument("--no-budget", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    payload = run(
        ka_values=tuple(args.kas),
        submergence_ratios=tuple(args.ratios),
        subdivision=args.subdivision,
        include_budget=not args.no_budget,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output} ({len(text)} bytes)")


if __name__ == "__main__":
    main()