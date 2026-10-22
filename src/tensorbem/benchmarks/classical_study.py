"""Reproducible study of the classical Laplace benchmarks (no free surface).

Three families are reported, all with analytical references and all independent
of the boundary-element discretisation:

``ellipse_added_mass``
    Exterior 2-D Dirichlet problem on an elliptic cylinder; added mass per unit
    span for motion along ``x`` and ``y``. References ``rho*pi*b**2`` and
    ``rho*pi*a**2``. This is the first case in the repository that needs the
    *exterior* 2-D operator rather than the bounded-domain one.
``ellipsoid_added_mass``
    Exterior 3-D Dirichlet problem on a flat-faceted ellipsoid; added mass for
    translation along each principal axis. Reference
    ``rho*V*alpha_i/(2-alpha_i)`` with Lamb's shape factors, checked against
    ``alpha_1+alpha_2+alpha_3=2`` on every evaluation.
``interior_solid_harmonics``
    Interior 3-D Dirichlet problem with exact harmonic polynomials of degree 1,
    2 and 3, which are represented exactly on the polyhedron. This replaces the
    single linear field used by the earlier 3-D verification.

Every family reports the observed order against ``h_max``; nothing is hidden
when a trend is poor, and the aggregate ``checks`` say explicitly which trends
were met. The scope is Laplace verification: there is no free surface, no
frequency domain and no wave or ship physics here.

Run:
    python -m tensorbem.benchmarks.classical_study --output artifacts/classical-study.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np

from tensorbem.benchmarks.ellipse_added_mass import run as run_ellipse
from tensorbem.benchmarks.ellipsoid_added_mass import run as run_ellipsoid
from tensorbem.benchmarks.interior_harmonics import run as run_harmonics

ELLIPSE_AXES = ("x", "y")
ELLIPSOID_AXES = ("x", "y", "z")


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(math.isfinite(float(value)))
    return True


def _with_orders(rows: list[dict[str, Any]], metric: str) -> list[float | None]:
    """Attach observed orders against ``h_max`` and return the non-null ones."""
    orders: list[float | None] = []
    for index, row in enumerate(rows):
        order: float | None = None
        if index:
            previous = rows[index - 1]
            if previous[metric] > 0.0 and row[metric] > 0.0:
                order = math.log(previous[metric] / row[metric]) / math.log(
                    previous["h_max"] / row["h_max"]
                )
        row[metric + "_order"] = order
        orders.append(order)
    return orders


def _decreasing(rows: list[dict[str, Any]], metric: str) -> bool:
    return all(
        later[metric] < earlier[metric] for earlier, later in zip(rows, rows[1:])
    )


def run_study(
    panel_counts: tuple[int, ...] = (16, 32, 64, 128),
    subdivisions: tuple[int, ...] = (0, 1, 2),
    harmonic_levels: tuple[int, ...] = (1, 2),
    semi_major: float = 1.5,
    semi_minor: float = 0.75,
    semi_axes: tuple[float, float, float] = (1.5, 1.0, 0.75),
    density: float = 1025.0,
) -> dict[str, Any]:
    """Run the three classical benchmark families and return JSON evidence."""
    if len(panel_counts) < 2:
        raise ValueError("the ellipse study requires at least two panel counts")
    if len(set(panel_counts)) != len(panel_counts):
        raise ValueError("panel counts must be unique")
    if len(subdivisions) < 2:
        raise ValueError("the ellipsoid study requires at least two subdivision levels")
    if len(set(subdivisions)) != len(subdivisions):
        raise ValueError("subdivision levels must be unique")
    counts = sorted(int(value) for value in panel_counts)
    levels = sorted(int(value) for value in subdivisions)

    ellipse: dict[str, Any] = {
        "semi_major": semi_major,
        "semi_minor": semi_minor,
        "panel_counts": counts,
        "axes": {},
        "observed_orders": {},
        "checks": {},
    }
    for axis in ELLIPSE_AXES:
        rows = [
            run_ellipse(n, semi_major, semi_minor, 1.0, density, axis)
            for n in counts
        ]
        orders = [
            order
            for order in _with_orders(rows, "added_mass_relative_error")
            if order is not None
        ]
        flux_orders = [
            order
            for order in _with_orders(rows, "normal_derivative_relative_l2_error")
            if order is not None
        ]
        ellipse["axes"][axis] = rows
        ellipse["observed_orders"][axis] = {
            "added_mass": orders,
            "normal_derivative_relative_l2_error": flux_orders,
        }
        ellipse["checks"][f"{axis}_added_mass_error_decreases"] = _decreasing(
            rows, "added_mass_relative_error"
        )
        ellipse["checks"][f"{axis}_added_mass_order_near_two"] = bool(
            orders and min(orders) > 1.5
        )

    ellipsoid: dict[str, Any] = {
        "semi_axes": [float(value) for value in semi_axes],
        "levels": levels,
        "axes": {},
        "observed_orders": {},
        "checks": {},
    }
    for axis_name in ELLIPSOID_AXES:
        rows = []
        for level in levels:
            report = run_ellipsoid(level, semi_axes, 1.0, density)
            row = next(
                item for item in report["axes"] if item["motion_axis"] == axis_name
            )
            row = dict(row)
            row["h_max"] = report["h_max"]
            row["volume_ratio"] = report["volume_ratio"]
            rows.append(row)
        orders = [
            order
            for order in _with_orders(rows, "added_mass_relative_error")
            if order is not None
        ]
        ellipsoid["axes"][axis_name] = rows
        ellipsoid["observed_orders"][axis_name] = orders
        ellipsoid["checks"][f"{axis_name}_added_mass_error_decreases"] = _decreasing(
            rows, "added_mass_relative_error"
        )
        ellipsoid["checks"][f"{axis_name}_added_mass_order_near_two"] = bool(
            orders and min(orders) > 1.5
        )

    harmonics = run_harmonics(harmonic_levels, (1, 2, 3))

    checks = {
        "ellipse_added_mass_converges": all(
            value
            for key, value in ellipse["checks"].items()
            if key.endswith("error_decreases")
        ),
        "ellipsoid_added_mass_converges": all(
            value
            for key, value in ellipsoid["checks"].items()
            if key.endswith("error_decreases")
        ),
        "harmonic_flux_converges": all(
            case["checks"]["normal_derivative_relative_l2_error_decreases"]
            for case in harmonics["cases"].values()
        ),
        "harmonic_field_converges": all(
            case["checks"]["field_rmse_decreases"]
            for case in harmonics["cases"].values()
        ),
    }
    return {
        "schema_version": 1,
        "study": "tensorbem_classical_benchmarks",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "density": density,
        "ellipse": ellipse,
        "ellipsoid": ellipsoid,
        "interior_harmonics": harmonics,
        "checks": checks,
        "all_outputs_finite": _all_finite(
            {"ellipse": ellipse, "ellipsoid": ellipsoid, "harmonics": harmonics}
        ),
        "scope": (
            "Laplace verification without a free surface: a 2-D exterior elliptic "
            "cylinder, a 3-D exterior ellipsoid, and 3-D interior solid harmonics of "
            "degree 1-3. The references are classical closed forms, not another BEM "
            "solve. No free-surface, wave radiation/diffraction, viscosity, "
            "ship-hull or experimental validation is included."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--subdivisions", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--harmonic-levels", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--semi-major", type=float, default=1.5)
    parser.add_argument("--semi-minor", type=float, default=0.75)
    parser.add_argument("--semi-axes", nargs=3, type=float, default=[1.5, 1.0, 0.75])
    parser.add_argument("--density", type=float, default=1025.0)
    parser.add_argument("--output", type=Path, help="also write machine-readable JSON")
    args = parser.parse_args()
    try:
        study = run_study(
            tuple(args.panels),
            tuple(args.subdivisions),
            tuple(args.harmonic_levels),
            args.semi_major,
            args.semi_minor,
            tuple(args.semi_axes),
            args.density,
        )
    except ValueError as error:
        parser.error(str(error))
    rendered = json.dumps(study, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
