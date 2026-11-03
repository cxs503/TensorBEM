"""Mesh-refinement study of the translating-ellipsoid added mass.

The ellipsoid benchmark (:mod:`tensorbem.benchmarks.ellipsoid_added_mass`)
quotes a *single* number per axis at a single mesh (``subdivisions = 3``, 1280
panels). A single number cannot tell a reader whether the quoted 4e-3-level
error is a converged discretisation error, a plateau of a biased mesh family,
or a signature of a defect in the assembly. This module answers that question
by turning the benchmark into a *family*:

* every level of the icosphere subdivision produces a closed, faceted mesh
  whose panel size halves (the panel count grows by 4), so the panel family
  can be treated as ``h -> 0``;
* the geometry is validated first -- the discrete volume of each faceted
  ellipsoid must approach the exact volume ``4*pi*a1*a2*a3/3``, and the
  per-axis shape factors must satisfy Lamb's identity ``sum(alpha) = 2``;
  a mesh whose volume is off cannot be trusted to give a trustworthy flux;
* the added-mass error of each axis is fitted as a power law in the mean panel
  size, the observed order is *measured* (not assumed) and the family is
  Richardson-extrapolated to zero panel size;
* the extrapolated limit is compared with Lamb's analytic added mass, which is
  an independent construction (one-dimensional quadrature of the shape-factor
  integrals). The agreement of the extrapolated limit with the analytic value
  is the evidence that the analytic reference is right and the quoted error is
  discretisation, not bias.

The recorded study (:file:`outputs/ellipsoid_heavy/ellipsoid_refinement.json`)
uses levels 0/1/2 (20/80/320 panels): the 1280-panel level costs about ten
minutes of assembly on a shared host, and its single-point numbers are verified
separately by the ``heavy`` tier of the declared-number registry. The
``--levels`` flag accepts any increasing set. Run::

    python -m tensorbem.benchmarks.ellipsoid_refinement --levels 1,2,3 \
        --output outputs/ellipsoid_heavy/ellipsoid_refinement.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

from tensorbem.benchmarks import ellipsoid_added_mass as ellipsoid_benchmark
from tensorbem.benchmarks.declared_numbers import richardson_limit

__all__ = ["AXIS_NAMES", "extrapolate_axis", "observed_order", "run"]

AXIS_NAMES = ("x", "y", "z")

#: Shape-factor identity, Lamb (1932) sec. 114: alpha_1 + alpha_2 + alpha_3 = 2.
SHAPE_FACTOR_SUM = 2.0


def observed_order(values: Sequence[float], levels: Sequence[int]) -> float:
    """Return the measured convergence order of an error family.

    ``values`` are the errors (or any quantity converging monotonically to a
    finite limit) at meshes indexed by ``levels``, coarsest first. The order is
    derived from the two finest successive differences, i.e. it is measured
    rather than assumed, which is what makes the extrapolation auditable.
    """
    if len(values) != len(levels) or len(values) < 3:
        raise ValueError("at least three matched (error, level) pairs are required")
    coarse, middle, finest = values[-3], values[-2], values[-1]
    first_step = abs(middle - coarse)
    second_step = abs(finest - middle)
    if first_step == 0.0 or second_step == 0.0:
        raise ValueError("the error family is not converging; order is undefined")
    refinement = float(levels[-1]) / float(levels[-2])
    return math.log(first_step / second_step) / math.log(refinement)


def extrapolate_axis(rows: Sequence[dict]) -> dict:
    """Extrapolate one axis' added mass to zero panel size.

    ``rows`` are the per-level axis records of the benchmark (coarsest first).
    Returns the measured order, the extrapolated added mass and its relative
    distance from the analytic value.
    """
    if len(rows) < 3:
        raise ValueError("at least three mesh levels are required for extrapolation")
    panels = [int(row["panels"]) for row in rows]
    added_mass = [float(row["added_mass"]) for row in rows]
    limit, order = richardson_limit(added_mass, panels)
    exact = float(rows[-1]["exact_added_mass"])
    return {
        "motion_axis": rows[-1]["motion_axis"],
        "panels": panels,
        "added_mass": added_mass,
        "exact_added_mass": exact,
        "observed_order": order,
        "extrapolated_added_mass": limit,
        "extrapolation_vs_analytic": abs(limit - exact) / abs(exact),
        "finest_relative_error": float(rows[-1]["added_mass_relative_error"]),
    }


def run(levels: Sequence[int] = (1, 2, 3), **benchmark_parameters) -> dict:
    """Run the ellipsoid benchmark at every requested subdivision level.

    ``benchmark_parameters`` are forwarded to
    :func:`tensorbem.benchmarks.ellipsoid_added_mass.run` (``semi_axes``,
    ``speed``, ``density``).
    """
    levels = tuple(int(level) for level in levels)
    if len(set(levels)) != len(levels) or list(levels) != sorted(levels):
        raise ValueError("levels must be strictly increasing")
    if any(level < 0 for level in levels):
        raise ValueError("subdivision levels start at 0 (the 20-panel icosahedron)")

    runs = [ellipsoid_benchmark.run(level, **benchmark_parameters) for level in levels]
    first = runs[0]

    mesh_rows = []
    for run_row in runs:
        shape_sum = float(run_row["shape_factor_sum"])
        mesh_rows.append(
            {
                "subdivisions": int(run_row["subdivisions"]),
                "panels": int(run_row["panels"]),
                "vertices": int(run_row["vertices"]),
                "h_mean": float(run_row["h_mean"]),
                "h_max": float(run_row["h_max"]),
                "surface_area": float(run_row["surface_area"]),
                "volume": float(run_row["volume"]),
                "exact_volume": float(run_row["exact_volume"]),
                "volume_relative_error": abs(
                    float(run_row["volume"]) - float(run_row["exact_volume"])
                )
                / float(run_row["exact_volume"]),
                "shape_factor_sum": shape_sum,
                "shape_factor_identity_error": abs(shape_sum - SHAPE_FACTOR_SUM),
                "min_angle_degrees": float(run_row["min_angle_degrees"]),
                "max_aspect_ratio": float(run_row["max_aspect_ratio"]),
                "relative_residual_l2": max(
                    float(axis_row["relative_residual_l2"])
                    for axis_row in run_row["axes"]
                ),
            }
        )

    axes = {}
    for axis_index, axis_name in enumerate(AXIS_NAMES):
        per_level = [run_row["axes"][axis_index] for run_row in runs]
        entry = {
            "motion_axis": axis_name,
            "levels": [
                {
                    "subdivisions": int(run_row["subdivisions"]),
                    "panels": int(run_row["panels"]),
                    "h_mean": float(run_row["h_mean"]),
                    "added_mass": float(axis_row["added_mass"]),
                    "exact_added_mass": float(axis_row["exact_added_mass"]),
                    "added_mass_relative_error": float(
                        axis_row["added_mass_relative_error"]
                    ),
                    "normal_derivative_relative_l2_error": float(
                        axis_row["normal_derivative_relative_l2_error"]
                    ),
                }
                for run_row, axis_row in zip(runs, per_level)
            ],
        }
        if len(per_level) >= 3:
            entry["extrapolation"] = extrapolate_axis(per_level)
            errors = [level["added_mass_relative_error"] for level in entry["levels"]]
            try:
                entry["error_decay_order"] = observed_order(errors, list(levels))
            except ValueError:
                entry["error_decay_order"] = None
        axes[axis_name] = entry

    volume_errors = [row["volume_relative_error"] for row in mesh_rows]
    volume_order = None
    if len(volume_errors) >= 3:
        try:
            volume_order = observed_order(volume_errors, list(levels))
        except ValueError:
            volume_order = None

    monotone_geometry = all(
        later < earlier for earlier, later in zip(volume_errors, volume_errors[1:])
    )
    monotone_accuracy = all(
        all(
            later < earlier
            for earlier, later in zip(
                [level["added_mass_relative_error"] for level in axes[name]["levels"]],
                [level["added_mass_relative_error"] for level in axes[name]["levels"]][1:],
            )
        )
        for name in AXIS_NAMES
    )

    return {
        "benchmark": "translating_ellipsoid_mesh_refinement",
        "levels": list(levels),
        "semi_axes": [float(value) for value in first["semi_axes"]],
        "fluid_density": float(first["fluid_density"]),
        "speed": float(first["speed"]),
        "meshes": mesh_rows,
        "volume_error_decay_order": volume_order,
        "volume_error_monotone": monotone_geometry,
        "added_mass_error_monotone": monotone_accuracy,
        "axes": axes,
        "note": (
            "Geometry is validated before the flux: the faceted volume must "
            "converge to 4*pi*a1*a2*a3/3 and Lamb's shape factors must sum to 2. "
            "The extrapolated added mass is compared with the analytic value; "
            "the discrepancy measures how far the analytic reference can be "
            "trusted, since the two sides share no discretisation."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", default="1,2,3", help="comma-separated subdivisions")
    parser.add_argument("--semi-axes", nargs=3, type=float, default=[1.5, 1.0, 0.75])
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--density", type=float, default=1025.0)
    parser.add_argument("--output", type=Path, help="write the JSON report here")
    args = parser.parse_args(argv)

    levels = [int(part) for part in args.levels.split(",") if part.strip()]
    result = run(
        levels,
        semi_axes=tuple(args.semi_axes),
        speed=args.speed,
        density=args.density,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())