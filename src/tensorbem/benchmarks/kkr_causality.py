"""Kramers-Kronig causality audit of the Ursell cylinder solutions.

The radiation coefficients of a frequency-domain seakeeping solver are the
real and imaginary parts of one causal transfer function, so they must satisfy
the Kramers-Kronig relations: the added-mass curve predicts the damping curve
and vice versa (:mod:`tensorbem.causality`, eqs. 1 and 2). This benchmark
sweeps a frequency band, feeds the tabulated ``(A, B)`` pairs of several
solutions into that check, and reports the *causality residual* next to the
ordinary error against Ursell's analytical solution.

Three solutions are audited on one common frequency grid:

* the analytical multipole reference of :func:`tensorbem.analytic.
  ursell_cylinder_added_mass`;
* the indirect (source) panel solution of :func:`tensorbem.free_surface_solver.
  solve_radiation_indirect` on 32, 64 and 128 panels of the same circle.

The analytical reference is not a control experiment: it is run through the
*same* transform on the *same* grid, so its residual is the transform's own
floor (piecewise-linear interpolation of the sweep plus the two truncated
bands). A panel residual has to be read against that floor, and the report
records both.

The infinite-frequency added mass ``A_inf`` that eq. (1) requires is taken from
the analytical high-frequency limit for every solution (see
:func:`ursell_high_frequency_limit`), so the panel and reference residuals are
directly comparable; the data-driven fit is reported as well, because using it
is the only thing available for an arbitrary hull, and its bias is part of what
this benchmark measures.

Run: ``python -m tensorbem.benchmarks.kkr_causality --points 48``
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tensorbem.analytic import hulme_hemisphere_radiation, ursell_cylinder_added_mass
from tensorbem.causality import (
    AsymptoteFit,
    KKRCheck,
    fit_added_mass_asymptote,
    kkr_check,
)
from tensorbem.free_surface_solver import solve_radiation_indirect
from tensorbem.benchmarks.ursell_cylinder import submerged_circle_mesh

_GRAVITY = 9.80665
_DENSITY = 1025.0
_DEFAULT_PANELS = (32, 64, 128)
_HIGH_FREQUENCY_KA = (8.0, 12.0, 16.0, 20.0, 25.0, 30.0)


def ursell_high_frequency_limit(
    radius: float = 1.0,
    submergence: float = 2.0,
    density: float = _DENSITY,
    gravity: float = _GRAVITY,
    *,
    terms: int = 80,
) -> tuple[float, float]:
    """High-frequency added-mass limit of the submerged cylinder.

    As ``omega -> infinity`` the free-surface condition degenerates to
    ``phi = 0`` on ``z = 0`` and the added mass levels off. The multipole
    solution approaches that plateau slowly -- algebraically in ``k*a``, not
    exponentially -- so the limit is extrapolated from the analytic values at
    ``k*a = 8 ... 30`` with a ``1/(k*a)`` model, whose maximum residual is
    returned as the extrapolation uncertainty. (A ``1/(k*a)^2`` model does *not*
    describe the tail here; the two extrapolations differ by about 0.5 % of the
    limit, and the better-fitting one is used and reported.)

    Returns ``(added_mass_limit, extrapolation_max_residual)``.
    """
    ka = np.array(_HIGH_FREQUENCY_KA, dtype=float)
    omega = np.sqrt(ka * gravity / radius)
    values = np.array(
        [
            ursell_cylinder_added_mass(
                radius, submergence, w, density, gravity=gravity, terms=terms
            )[0]
            for w in omega
        ]
    )
    design = np.column_stack((np.ones_like(ka), 1.0 / ka))
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    residual = float(np.max(np.abs(design @ coefficients - values)))
    return float(coefficients[0]), residual


def _sweep(ka_min: float, ka_max: float, points: int, radius: float, gravity: float):
    ka = np.geomspace(ka_min, ka_max, points)
    return ka, np.sqrt(ka * gravity / radius)


def _kkr_payload(report: KKRCheck) -> dict[str, object]:
    return {
        "sample_omegas": report.sample_omegas.tolist(),
        "measured_damping": report.measured_damping.tolist(),
        "predicted_damping": report.predicted_damping.tolist(),
        "measured_added_mass": report.measured_added_mass.tolist(),
        "predicted_added_mass": report.predicted_added_mass.tolist(),
        "damping_relative_error": report.damping_relative_error.tolist(),
        "added_mass_relative_error": report.added_mass_relative_error.tolist(),
        "max_damping_relative_error": report.max_damping_relative_error,
        "max_added_mass_relative_error": report.max_added_mass_relative_error,
        "added_mass_inf": report.added_mass_inf,
        "tail_coefficient": report.tail_coefficient,
        "asymptote_relative_residual": report.asymptote_relative_residual,
        "damping_scale": report.damping_scale,
        "added_mass_scale": report.added_mass_scale,
    }


def _analytic_pair(
    omega,
    radius: float,
    submergence: float,
    density: float,
    gravity: float,
    *,
    terms: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    pairs = np.array(
        [
            ursell_cylinder_added_mass(
                radius, submergence, w, density, gravity=gravity, terms=terms
            )
            for w in omega
        ]
    )
    return pairs[:, 0], pairs[:, 1]


def run(
    panel_counts: tuple[int, ...] = _DEFAULT_PANELS,
    radius: float = 1.0,
    submergence: float = 2.0,
    ka_min: float = 0.05,
    ka_max: float = 3.0,
    points: int = 48,
    density: float = _DENSITY,
    gravity: float = _GRAVITY,
    panel_nodes: int = 24,
    include_3d: bool = True,
) -> dict[str, object]:
    """Run the causality audit and return a JSON-serialisable record."""
    if submergence <= radius:
        raise ValueError("the cylinder must be fully submerged: submergence > radius")
    if points < 12:
        raise ValueError("points must be at least 12 to resolve the sweep")

    ka, omega = _sweep(ka_min, ka_max, points, radius, gravity)
    added_mass_inf, limit_residual = ursell_high_frequency_limit(
        radius, submergence, density, gravity
    )
    reference_mass, reference_damping = _analytic_pair(
        omega, radius, submergence, density, gravity
    )
    scaled_mass = reference_mass / added_mass_inf

    cases: dict[str, object] = {}

    fit_wide = fit_added_mass_asymptote(omega, reference_mass)
    cases["analytic_multipole"] = {
        "description": "Ursell multipole closed form (analytical reference)",
        "added_mass": reference_mass.tolist(),
        "damping": reference_damping.tolist(),
        "kkr_with_limit_added_mass": _kkr_payload(
            kkr_check(omega, reference_mass, reference_damping, added_mass_inf=added_mass_inf)
        ),
        "kkr_with_fitted_added_mass": _kkr_payload(
            kkr_check(omega, reference_mass, reference_damping)
        ),
        "fitted_added_mass_inf": fit_wide.added_mass_inf,
        "fitted_asymptote_relative_residual": fit_wide.relative_residual,
    }

    for panel_count in panel_counts:
        mesh = submerged_circle_mesh(panel_count, radius, submergence)
        mass = np.empty(points)
        damper = np.empty(points)
        residuals = np.empty(points)
        for index, wavenumber in enumerate(ka / radius):
            result = solve_radiation_indirect(
                mesh, float(wavenumber), "heave", density=density, panel_nodes=panel_nodes
            )
            mass[index] = float(result.added_mass[1])
            damper[index] = float(result.damping[1])
            residuals[index] = result.relative_residual
        cases[f"panels_{panel_count}"] = {
            "description": f"indirect panel solution, {panel_count} panels",
            "panels": panel_count,
            "added_mass": mass.tolist(),
            "damping": damper.tolist(),
            "body_condition_relative_residual_max": float(np.max(residuals)),
            "added_mass_relative_error_max": float(
                np.max(np.abs(mass - reference_mass)) / np.max(np.abs(reference_mass))
            ),
            "damping_relative_error_max": float(
                np.max(np.abs(damper - reference_damping)) / np.max(np.abs(reference_damping))
            ),
            "kkr_with_limit_added_mass": _kkr_payload(
                kkr_check(omega, mass, damper, added_mass_inf=added_mass_inf)
            ),
            "kkr_with_fitted_added_mass": _kkr_payload(
                kkr_check(omega, mass, damper)
            ),
        }

    record: dict[str, object] = {
        "benchmark": "kramers_kronig_causality_audit",
        "case": "ursell_submerged_circular_cylinder_deep_water_heave",
        "radius": radius,
        "submergence": submergence,
        "density": density,
        "gravity": gravity,
        "ka_min": ka_min,
        "ka_max": ka_max,
        "points": points,
        "panel_nodes": panel_nodes,
        "ka": ka.tolist(),
        "omega": omega.tolist(),
        "added_mass_high_frequency_limit": added_mass_inf,
        "added_mass_limit_extrapolation_residual": limit_residual,
        "added_mass_limit_over_displaced_mass": added_mass_inf
        / (density * math.pi * radius**2),
        "displaced_mass_per_span": density * math.pi * radius**2,
        "cases": cases,
    }

    if include_3d:
        record["three_dimensional"] = _hulme_causality()

    return record


def _hulme_causality(
    radius: float = 1.0,
    density: float = _DENSITY,
    ka_min: float = 0.1,
    ka_max: float = 4.0,
    points: int = 40,
) -> dict[str, object]:
    """Causality residuals of the semi-analytical Hulme hemisphere solution."""
    ka = np.geomspace(ka_min, ka_max, points)
    results = [hulme_hemisphere_radiation(float(value), radius=radius, density=density) for value in ka]
    omega = np.array([item.omega for item in results])
    payload: dict[str, object] = {"ka_min": ka_min, "ka_max": ka_max, "points": points}
    for mode, mass_key, damper_key in (
        ("heave", "heave_added_mass", "heave_damping"),
        ("surge", "surge_added_mass", "surge_damping"),
    ):
        mass = np.array([getattr(item, mass_key) for item in results])
        damper = np.array([getattr(item, damper_key) for item in results])
        payload[mode] = {
            "added_mass": mass.tolist(),
            "damping": damper.tolist(),
            "kkr": _kkr_payload(kkr_check(omega, mass, damper)),
        }
    payload["omega"] = omega.tolist()
    payload["description"] = (
        "Hulme (1982) floating-hemisphere semi-analytical radiation coefficients"
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", type=int, nargs="+", default=list(_DEFAULT_PANELS))
    parser.add_argument("--ka-min", type=float, default=0.05)
    parser.add_argument("--ka-max", type=float, default=3.0)
    parser.add_argument("--points", type=int, default=48)
    parser.add_argument("--panel-nodes", type=int, default=24)
    parser.add_argument("--no-3d", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    record = run(
        panel_counts=tuple(args.panels),
        ka_min=args.ka_min,
        ka_max=args.ka_max,
        points=args.points,
        panel_nodes=args.panel_nodes,
        include_3d=not args.no_3d,
    )
    text = json.dumps(record, indent=2, sort_keys=False)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())