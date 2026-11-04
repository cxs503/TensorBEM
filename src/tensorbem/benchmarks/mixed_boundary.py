"""Analytic verification study for the mixed Dirichlet/Neumann 2-D panel solver.

Every case here has a closed-form harmonic solution, so the reference is a
formula rather than another discretisation:

``adjacent_mixed``
    ``u = exp(x) cos(y)`` on the unit square.  Dirichlet data on the two sides
    ``x = 0`` and ``y = 0``, Neumann data on ``x = 1`` and ``y = 1``; both data
    sets are non-constant.  The solver recovers, and this study measures, the
    trace it was *not* given on each side type.

``pure_neumann``
    ``u = x^2 - y^2`` with only normal derivatives prescribed.  Panel-constant
    data are exact here and the potential is determined only up to a constant,
    so the ``mean_zero`` gauge is exercised rather than bypassed.

Two controls ride along with the cases:

``dirichlet_regression``
    The same geometry and data with every panel Dirichlet, compared against
    :func:`tensorbem.solver.solve_laplace_dirichlet`.  The mixed assembly must
    reproduce the all-Dirichlet solve to roundoff, otherwise the block system
    is not a generalisation of the existing path.

``dirichlet_flux_baseline``
    The all-Dirichlet solve of the same analytic field, used to show that the
    first-order flux error on a square is a property of constant panels plus
    square corners, not of mixing boundary conditions.

Run::

    python -m tensorbem.benchmarks.mixed_boundary --output docs/mixed_boundary/outputs/study.json

The printed table, the JSON artifact and the report figure are all regenerated
from that artifact; no number in ``docs/mixed_boundary_conditions.md`` is typed
by hand.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..geometry import PanelMesh
from ..mixed import MixedBoundary, evaluate_mixed_interior_field, solve_laplace_mixed
from ..solver import solve_laplace_dirichlet

FloatArray = NDArray[np.float64]
_SIDE_TOL = 1.0e-12
_SIDE_NAMES = ("bottom", "right", "top", "left")
_SCHEMA = "tensorbem.mixed_boundary.study/1"


def square_mesh(panels_per_side: int) -> PanelMesh:
    """Counter-clockwise unit square with ``panels_per_side`` panels per side."""
    if panels_per_side < 1:
        raise ValueError("panels_per_side must be a positive integer")
    corners = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    points: list[tuple[float, float]] = []
    for k in range(4):
        start = np.array(corners[k], dtype=float)
        end = np.array(corners[(k + 1) % 4], dtype=float)
        for i in range(panels_per_side):
            point = start + (end - start) * (i / panels_per_side)
            points.append((float(point[0]), float(point[1])))
    return PanelMesh(vertices=np.array(points, dtype=float))


def side_masks(mesh: PanelMesh) -> dict[str, NDArray[np.bool_]]:
    """Boolean panel masks for each side of the unit square."""
    centers = mesh.centers
    masks = {
        "bottom": np.abs(centers[:, 1]) < _SIDE_TOL,
        "right": np.abs(centers[:, 0] - 1.0) < _SIDE_TOL,
        "top": np.abs(centers[:, 1] - 1.0) < _SIDE_TOL,
        "left": np.abs(centers[:, 0]) < _SIDE_TOL,
    }
    if sum(int(mask.sum()) for mask in masks.values()) != mesh.panel_count:
        raise ValueError("side detection failed: not every panel sits on one side")
    return masks


def exterior_normal(points: FloatArray, side: str) -> FloatArray:
    """Outward unit normal, tiled for each point on a unit-square side."""
    components = {
        "bottom": (0.0, -1.0),
        "right": (1.0, 0.0),
        "top": (0.0, 1.0),
        "left": (-1.0, 0.0),
    }[side]
    return np.tile(np.array(components, dtype=float), (points.shape[0], 1))


@dataclass(frozen=True)
class AnalyticCase:
    """A closed-form harmonic field plus how its boundary is split."""

    name: str
    dirichlet_sides: tuple[str, ...]
    neumann_sides: tuple[str, ...]
    description: str

    def value(self, points: FloatArray) -> FloatArray:
        raise NotImplementedError

    def normal_derivative(self, points: FloatArray, side: str) -> FloatArray:
        raise NotImplementedError


@dataclass(frozen=True)
class ExponentialCosine(AnalyticCase):
    """``u = exp(x) cos(y)``, harmonic with non-constant data of both types."""

    def value(self, points: FloatArray) -> FloatArray:
        return np.exp(points[:, 0]) * np.cos(points[:, 1])

    def normal_derivative(self, points: FloatArray, side: str) -> FloatArray:
        normal = exterior_normal(points, side)
        gradient_x = np.exp(points[:, 0]) * np.cos(points[:, 1])
        gradient_y = -np.exp(points[:, 0]) * np.sin(points[:, 1])
        return gradient_x * normal[:, 0] + gradient_y * normal[:, 1]


@dataclass(frozen=True)
class QuadraticSaddle(AnalyticCase):
    """``u = x^2 - y^2``; panel-constant data are exact, so pure Neumann is clean."""

    def value(self, points: FloatArray) -> FloatArray:
        return points[:, 0] ** 2 - points[:, 1] ** 2

    def normal_derivative(self, points: FloatArray, side: str) -> FloatArray:
        normal = exterior_normal(points, side)
        return 2.0 * points[:, 0] * normal[:, 0] - 2.0 * points[:, 1] * normal[:, 1]


ADJACENT_MIXED = ExponentialCosine(
    name="adjacent_mixed",
    dirichlet_sides=("left", "bottom"),
    neumann_sides=("right", "top"),
    description="u = exp(x)*cos(y); Dirichlet on x=0 and y=0, Neumann on x=1 and y=1",
)
PURE_NEUMANN = QuadraticSaddle(
    name="pure_neumann",
    dirichlet_sides=(),
    neumann_sides=_SIDE_NAMES,
    description="u = x^2 - y^2; pure Neumann, mean_zero gauge",
)
ALL_DIRICHLET = ExponentialCosine(
    name="all_dirichlet",
    dirichlet_sides=_SIDE_NAMES,
    neumann_sides=(),
    description="u = exp(x)*cos(y); all four sides Dirichlet",
)
DEFAULT_CASES = (ADJACENT_MIXED, PURE_NEUMANN)


def build_boundary(
    mesh: PanelMesh, case: AnalyticCase
) -> tuple[MixedBoundary, dict[str, NDArray[np.bool_]]]:
    """Assemble the panel-indexed boundary data of ``case`` on ``mesh``."""
    masks = side_masks(mesh)
    centers = mesh.centers
    dirichlet_panels: list[NDArray[np.int_]] = []
    dirichlet_values: list[FloatArray] = []
    neumann_panels: list[NDArray[np.int_]] = []
    neumann_values: list[FloatArray] = []
    for side in _SIDE_NAMES:
        index = np.flatnonzero(masks[side])
        if side in case.dirichlet_sides:
            dirichlet_panels.append(index)
            dirichlet_values.append(case.value(centers[index]))
        elif side in case.neumann_sides:
            neumann_panels.append(index)
            neumann_values.append(case.normal_derivative(centers[index], side))
        else:
            raise ValueError(f"side {side} has no boundary condition")
    boundary = MixedBoundary(
        dirichlet_panels=(
            np.concatenate(dirichlet_panels) if dirichlet_panels else np.empty(0, dtype=np.int_)
        ),
        dirichlet_values=(
            np.concatenate(dirichlet_values) if dirichlet_values else np.empty(0)
        ),
        neumann_panels=(
            np.concatenate(neumann_panels) if neumann_panels else np.empty(0, dtype=np.int_)
        ),
        neumann_values=(np.concatenate(neumann_values) if neumann_values else np.empty(0)),
    )
    return boundary, masks


def interior_probe_points(divisions: int = 6) -> FloatArray:
    """Fixed interior probe grid, kept away from the boundary."""
    axis = np.linspace(0.1, 0.9, divisions)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    return np.column_stack((xx.ravel(), yy.ravel()))


def exact_normal_derivative(
    mesh: PanelMesh, masks: dict[str, NDArray[np.bool_]], case: AnalyticCase
) -> FloatArray:
    """Exact normal derivative on every panel, along the outward normal."""
    centers = mesh.centers
    out = np.zeros(mesh.panel_count, dtype=float)
    for side in _SIDE_NAMES:
        index = np.flatnonzero(masks[side])
        if index.size:
            out[index] = case.normal_derivative(centers[index], side)
    return out


def relative_l2(computed: FloatArray, exact: FloatArray) -> float:
    """Relative discrete L2 error, safe for an all-zero reference."""
    denominator = float(np.linalg.norm(exact))
    numerator = float(np.linalg.norm(computed - exact))
    return numerator / denominator if denominator > 0.0 else numerator


def _zero_mean(values: FloatArray) -> FloatArray:
    return values - float(np.mean(values))


def run_case(case: AnalyticCase, panels_per_side: tuple[int, ...]) -> dict[str, object]:
    """Refinement study for one analytic case."""
    probes = interior_probe_points()
    exact_field = case.value(probes)
    rows: list[dict[str, object]] = []
    for n in panels_per_side:
        mesh = square_mesh(n)
        boundary, masks = build_boundary(mesh, case)
        result = solve_laplace_mixed(mesh, boundary)
        exact_q = exact_normal_derivative(mesh, masks, case)
        exact_u = case.value(mesh.centers)
        q_computed = result.normal_derivative[result.dirichlet_panels]
        q_exact = exact_q[result.dirichlet_panels]
        u_computed = result.boundary_values[result.neumann_panels]
        u_exact = exact_u[result.neumann_panels]
        field = evaluate_mixed_interior_field(mesh, result, probes)
        if result.is_pure_neumann:
            u_computed = _zero_mean(u_computed)
            u_exact = _zero_mean(u_exact)
            field = _zero_mean(field)
            exact_used = _zero_mean(exact_field)
        else:
            exact_used = exact_field
        rows.append(
            {
                "panels_per_side": n,
                "panel_count": mesh.panel_count,
                "h_max": float(np.max(mesh.lengths)),
                "condition_number": result.condition_number,
                "relative_residual_l2": result.relative_residual_l2,
                "q_unknown_side_rel_l2": relative_l2(q_computed, q_exact),
                "u_unknown_side_rel_l2": relative_l2(u_computed, u_exact),
                "interior_field_rel_l2": relative_l2(field, exact_used),
                "interior_field_max_abs": float(np.max(np.abs(field - exact_used))),
                "pure_neumann_gauge": result.pure_neumann_gauge,
            }
        )
    orders: dict[str, list[float]] = {}
    for key in (
        "q_unknown_side_rel_l2",
        "u_unknown_side_rel_l2",
        "interior_field_rel_l2",
    ):
        sequence = [float(row[key]) for row in rows]
        orders[key] = [
            math.log(sequence[i] / sequence[i + 1]) / math.log(2.0)
            for i in range(len(sequence) - 1)
            if sequence[i] > 0.0 and sequence[i + 1] > 0.0
        ]
    return {
        "name": case.name,
        "description": case.description,
        "dirichlet_sides": list(case.dirichlet_sides),
        "neumann_sides": list(case.neumann_sides),
        "rows": rows,
        "observed_order": orders,
    }


def run_dirichlet_regression(panels_per_side: tuple[int, ...]) -> dict[str, object]:
    """Mixed assembly with an empty Neumann set versus the all-Dirichlet solve."""
    rows: list[dict[str, object]] = []
    for n in panels_per_side:
        mesh = square_mesh(n)
        boundary, _ = build_boundary(mesh, ALL_DIRICHLET)
        mixed = solve_laplace_mixed(mesh, boundary)
        reference = solve_laplace_dirichlet(mesh, ALL_DIRICHLET.value(mesh.centers))
        rows.append(
            {
                "panels_per_side": n,
                "panel_count": mesh.panel_count,
                "q_max_abs_difference": float(
                    np.max(np.abs(mixed.normal_derivative - reference.normal_derivative))
                ),
                "q_rel_l2_difference": relative_l2(
                    mixed.normal_derivative, reference.normal_derivative
                ),
                "mixed_condition_number": mixed.condition_number,
                "dirichlet_condition_number": reference.condition_number,
                "condition_number_difference": abs(
                    mixed.condition_number - reference.condition_number
                ),
            }
        )
    return {
        "name": "dirichlet_regression",
        "description": "all-Dirichlet mixed assembly vs solve_laplace_dirichlet",
        "rows": rows,
    }


def run_dirichlet_flux_baseline(panels_per_side: tuple[int, ...]) -> dict[str, object]:
    """All-Dirichlet flux error for the mixed case's analytic field."""
    rows: list[dict[str, object]] = []
    for n in panels_per_side:
        mesh = square_mesh(n)
        masks = side_masks(mesh)
        result = solve_laplace_dirichlet(mesh, ADJACENT_MIXED.value(mesh.centers))
        exact_q = exact_normal_derivative(mesh, masks, ADJACENT_MIXED)
        rows.append(
            {
                "panels_per_side": n,
                "panel_count": mesh.panel_count,
                "q_rel_l2": relative_l2(result.normal_derivative, exact_q),
                "condition_number": result.condition_number,
            }
        )
    sequence = [float(row["q_rel_l2"]) for row in rows]
    return {
        "name": "dirichlet_flux_baseline",
        "description": "all four sides Dirichlet, same analytic field (corner-limited flux)",
        "rows": rows,
        "observed_order": [
            math.log(sequence[i] / sequence[i + 1]) / math.log(2.0)
            for i in range(len(sequence) - 1)
            if sequence[i] > 0.0 and sequence[i + 1] > 0.0
        ],
    }


def run_study(panels_per_side: tuple[int, ...]) -> dict[str, object]:
    """The full study: analytic cases plus both controls."""
    return {
        "schema": _SCHEMA,
        "panels_per_side": list(panels_per_side),
        "reference_length": 2.0,
        "cases": [run_case(case, panels_per_side) for case in DEFAULT_CASES],
        "dirichlet_regression": run_dirichlet_regression(panels_per_side),
        "dirichlet_flux_baseline": run_dirichlet_flux_baseline(panels_per_side),
    }


def _fmt(value: float) -> str:
    return f"{value:.3e}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panels",
        nargs="+",
        type=int,
        default=(2, 4, 8, 16),
        help="panels per side for the refinement study",
    )
    parser.add_argument("--output", type=Path, default=None, help="write the study JSON here")
    args = parser.parse_args(argv)
    study = run_study(tuple(args.panels))
    for case in study["cases"]:
        print(f"\n[{case['name']}] {case['description']}")
        print(
            f"{'panels':>7} {'cond':>10} {'q_err':>11} {'u_err':>11} "
            f"{'field_err':>11} {'field_max':>11}"
        )
        for row in case["rows"]:
            print(
                f"{row['panel_count']:>7} {row['condition_number']:>10.3f} "
                f"{_fmt(row['q_unknown_side_rel_l2']):>11} "
                f"{_fmt(row['u_unknown_side_rel_l2']):>11} "
                f"{_fmt(row['interior_field_rel_l2']):>11} "
                f"{_fmt(row['interior_field_max_abs']):>11}"
            )
        for key, orders in case["observed_order"].items():
            print(f"  observed order {key}: " + " ".join(f"{o:.3f}" for o in orders))
    regression = study["dirichlet_regression"]
    print("\n[dirichlet_regression] mixed (all Dirichlet) vs solve_laplace_dirichlet")
    for row in regression["rows"]:
        print(
            f"{row['panel_count']:>7} max|dq|={_fmt(row['q_max_abs_difference']):>11} "
            f"cond {row['mixed_condition_number']:.6f} vs "
            f"{row['dirichlet_condition_number']:.6f}"
        )
    baseline = study["dirichlet_flux_baseline"]
    print("\n[dirichlet_flux_baseline] all-Dirichlet flux error, same field")
    for row in baseline["rows"]:
        print(
            f"{row['panel_count']:>7} q_err={_fmt(row['q_rel_l2']):>11} "
            f"cond {row['condition_number']:.3f}"
        )
    print("  observed order: " + " ".join(f"{o:.3f}" for o in baseline["observed_order"]))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(study, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())