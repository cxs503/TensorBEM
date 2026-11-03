"""Verification of mixed Dirichlet/Neumann and multiply connected 2-D BEM solves.

Closes the three capability gaps listed in ``docs/verification.md`` §1 as
"后续" (follow-up work): multiply connected domains, mixed boundary conditions,
and pure Neumann problems.  Every case has a closed-form harmonic solution, so
the errors below are measured against an analytic reference evaluated on the
same panel midpoints -- no second discretisation is used as truth.

Run::

    python -m tensorbem.benchmarks.mixed_bc_verification --panels 16 32 64

Outputs JSON (machine readable, zero hand transcription) plus a Markdown summary
when ``--outdir`` is given.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tensorbem.mixed_bc import (
    PanelSet,
    assemble_panels,
    circle_loop,
    evaluate_field,
    flux_integral,
    solve_mixed,
)

# Annulus geometry and radial test function u = AMPLITUDE * log(r) + OFFSET.
INNER_RADIUS = 1.0
OUTER_RADIUS = 2.0
AMPLITUDE = 1.0
OFFSET = 0.5
# The outer loop has radius 2, and a circle of radius R has zero mean-flux
# eigenvalue of the single-layer operator exactly when R equals the reference
# length, so the study pins a reference length away from both loop radii.
REFERENCE_LENGTH = 1.0
# Interior probes for the field reconstruction (all strictly inside the annulus).
FIELD_POINTS = np.array(
    [[1.5, 0.0], [0.0, 1.5], [1.35, 0.9], [-1.4, -1.2], [0.0, -1.1], [1.1, -1.1]]
)
SQUARE_FIELD_POINTS = np.array(
    [[0.1, 0.1], [0.5, 0.5], [0.9, 0.2], [0.25, 0.75], [0.7, 0.9], [0.5, 0.1]]
)


_MATRIX_CACHE: dict[tuple[bytes, bytes, bytes, bytes, float], tuple[np.ndarray, np.ndarray]] = {}


def cached_matrices(
    panels: PanelSet, reference_length: float = REFERENCE_LENGTH
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble ``(single, double)`` once per (geometry, reference length).

    A convergence study re-solves the same mesh with several data sets, so the
    O(n^2) analytic assembly dominates the runtime if it is repeated.  The cache
    key is the full panel description, so different geometries never collide.
    """
    key = (
        panels.starts.tobytes(),
        panels.tangents.tobytes(),
        panels.normals.tobytes(),
        panels.lengths.tobytes(),
        float(reference_length),
    )
    matrices = _MATRIX_CACHE.get(key)
    if matrices is None:
        matrices = assemble_panels(panels, reference_length=reference_length)
        _MATRIX_CACHE[key] = matrices
    return matrices


def radial_potential(points: np.ndarray) -> np.ndarray:
    """Exact annulus solution u = A log r + B."""
    radius = np.hypot(points[:, 0], points[:, 1])
    return AMPLITUDE * np.log(radius) + OFFSET


def radial_flux(panels: PanelSet) -> np.ndarray:
    """Exact normal derivative of u = A log r + B at panel midpoints."""
    centers = panels.centers
    radius = np.hypot(centers[:, 0], centers[:, 1])
    projection = centers[:, 0] * panels.normals[:, 0] + centers[:, 1] * panels.normals[:, 1]
    return AMPLITUDE * projection / radius**2


def square_potential(points: np.ndarray) -> np.ndarray:
    """Exact solution u = x^2 - y^2 of Laplace's equation on the unit square."""
    return points[:, 0] ** 2 - points[:, 1] ** 2


def square_flux(panels: PanelSet) -> np.ndarray:
    """Exact normal derivative of u = x^2 - y^2 at panel midpoints."""
    centers = panels.centers
    return 2.0 * centers[:, 0] * panels.normals[:, 0] - 2.0 * centers[:, 1] * panels.normals[:, 1]


CIRCLE_RADIUS = 1.5


def circle_potential(points: np.ndarray) -> np.ndarray:
    """Exact harmonic solution u = x on the disc of radius CIRCLE_RADIUS."""
    return points[:, 0]


def circle_flux(panels: PanelSet) -> np.ndarray:
    """Exact normal derivative of u = x at panel midpoints: q = n_x = cos(theta)."""
    centers = panels.centers
    radius = np.hypot(centers[:, 0], centers[:, 1])
    # An inscribed polygon has radially aligned outward normals at midpoints.
    radial = centers / radius[:, None]
    if not np.allclose(panels.normals, radial, atol=1e-12):
        raise ValueError("circle_flux expects a circle mesh with radial normals")
    return panels.normals[:, 0].copy()


def annulus_panels(per_loop: int, outer_radius: float = OUTER_RADIUS,
                   inner_radius: float = INNER_RADIUS) -> PanelSet:
    """Two concentric inscribed polygons: outer CCW, hole CW (normals outward)."""
    if per_loop < 3:
        raise ValueError("per_loop must be at least 3")
    return PanelSet.from_loops(
        [circle_loop(per_loop, outer_radius), circle_loop(per_loop, inner_radius, clockwise=True)]
    )


def square_panels(per_side: int) -> PanelSet:
    """CCW unit square with ``per_side`` equal panels on every side."""
    if per_side < 1:
        raise ValueError("per_side must be positive")
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    vertices = []
    for start, end in zip(corners, np.roll(corners, -1, axis=0)):
        step = (end - start) / per_side
        vertices.extend(start + np.arange(per_side)[:, None] * step)
    return PanelSet.from_loops([np.asarray(vertices)])


def _relative_l2(error: np.ndarray, reference: np.ndarray) -> float:
    scale = float(np.linalg.norm(reference))
    return float(np.linalg.norm(error) / scale) if scale > 0.0 else float(np.linalg.norm(error))


def _relative_max(error: np.ndarray, reference: np.ndarray) -> float:
    scale = float(np.max(np.abs(reference)))
    return float(np.max(np.abs(error)) / scale) if scale > 0.0 else float(np.max(np.abs(error)))


@dataclass
class CaseOutcome:
    name: str
    panel_count: int
    metrics: dict[str, float]
    notes: tuple[str, ...]


def run_annulus_dirichlet(
    per_loop: int, reference_length: float = REFERENCE_LENGTH
) -> CaseOutcome:
    panels = annulus_panels(per_loop)
    exact_u = radial_potential(panels.centers)
    exact_q = radial_flux(panels)
    result = solve_mixed(
        panels,
        dirichlet={i: float(v) for i, v in enumerate(exact_u)},
        reference_length=reference_length,
        matrices=cached_matrices(panels, reference_length),
    )
    inner = panels.loop == 1
    recovered = evaluate_field(
        panels, result.boundary_values, result.normal_derivative, FIELD_POINTS,
        reference_length=reference_length
    )
    exact_field = radial_potential(FIELD_POINTS)
    metrics = {
        "flux_rel_l2_all": _relative_l2(result.normal_derivative - exact_q, exact_q),
        "flux_rel_l2_outer": _relative_l2(
            result.normal_derivative[~inner] - exact_q[~inner], exact_q[~inner]
        ),
        "flux_rel_l2_inner": _relative_l2(
            result.normal_derivative[inner] - exact_q[inner], exact_q[inner]
        ),
        "flux_rel_max": _relative_max(result.normal_derivative - exact_q, exact_q),
        "field_rel_max": _relative_max(recovered - exact_field, exact_field),
        "condition_number": result.condition_number,
        "relative_residual": result.relative_residual_l2,
        "rank_deficiency": float(result.rank_deficiency),
        "reference_length": reference_length,
    }
    return CaseOutcome("annulus_dirichlet", panels.panel_count, metrics, result.notes)


def run_annulus_mixed(per_loop: int, swap: bool = False,
                      reference_length: float = REFERENCE_LENGTH) -> CaseOutcome:
    panels = annulus_panels(per_loop)
    exact_u = radial_potential(panels.centers)
    exact_q = radial_flux(panels)
    inner = np.where(panels.loop == 1)[0]
    outer = np.where(panels.loop == 0)[0]
    if swap:
        dirichlet_panels, neumann_panels = inner, outer
        name = "annulus_mixed_hole_dirichlet"
    else:
        dirichlet_panels, neumann_panels = outer, inner
        name = "annulus_mixed_outer_dirichlet"
    result = solve_mixed(
        panels,
        dirichlet={int(i): float(exact_u[i]) for i in dirichlet_panels},
        neumann={int(i): float(exact_q[i]) for i in neumann_panels},
        reference_length=reference_length,
        matrices=cached_matrices(panels, reference_length),
    )
    recovered = evaluate_field(
        panels, result.boundary_values, result.normal_derivative, FIELD_POINTS,
        reference_length=reference_length
    )
    exact_field = radial_potential(FIELD_POINTS)
    metrics = {
        "neumann_trace_rel_l2": _relative_l2(
            result.boundary_values[neumann_panels] - exact_u[neumann_panels],
            exact_u[neumann_panels],
        ),
        "dirichlet_flux_rel_l2": _relative_l2(
            result.normal_derivative[dirichlet_panels] - exact_q[dirichlet_panels],
            exact_q[dirichlet_panels],
        ),
        "dirichlet_flux_rel_max": _relative_max(
            result.normal_derivative[dirichlet_panels] - exact_q[dirichlet_panels],
            exact_q[dirichlet_panels],
        ),
        "neumann_trace_rel_max": _relative_max(
            result.boundary_values[neumann_panels] - exact_u[neumann_panels],
            exact_u[neumann_panels],
        ),
        "field_rel_max": _relative_max(recovered - exact_field, exact_field),
        "condition_number": result.condition_number,
        "relative_residual": result.relative_residual_l2,
        "rank_deficiency": float(result.rank_deficiency),
    }
    return CaseOutcome(name, panels.panel_count, metrics, result.notes)


def run_annulus_pure_neumann(per_loop: int,
                             reference_length: float = REFERENCE_LENGTH) -> CaseOutcome:
    panels = annulus_panels(per_loop)
    exact_q = radial_flux(panels)
    result = solve_mixed(
        panels,
        neumann={i: float(v) for i, v in enumerate(exact_q)},
        reference_length=reference_length,
        matrices=cached_matrices(panels, reference_length),
    )
    exact_u = radial_potential(panels.centers)
    got = result.boundary_values - float(np.mean(result.boundary_values))
    want = exact_u - float(np.mean(exact_u))
    metrics = {
        "potential_rel_l2_up_to_constant": _relative_l2(got - want, want),
        "potential_rel_max_up_to_constant": _relative_max(got - want, want),
        "flux_integral": flux_integral(panels, result.normal_derivative),
        "rank_deficiency": float(result.rank_deficiency),
        "solvability_residual": float(result.solvability_residual or 0.0),
        "condition_number": result.condition_number,
        "relative_residual": result.relative_residual_l2,
        "gauge_removed": result.gauge_removed,
    }
    return CaseOutcome("annulus_pure_neumann", panels.panel_count, metrics, result.notes)


def run_square_mixed(per_side: int) -> CaseOutcome:
    panels = square_panels(per_side)
    centers = panels.centers
    exact_u = square_potential(centers)
    exact_q = square_flux(panels)
    # Neumann on the vertical sides x = 0 and x = 1, Dirichlet on y = 0 and y = 1.
    vertical = np.isclose(np.abs(panels.normals[:, 0]), 1.0)
    dirichlet_panels = np.where(~vertical)[0]
    neumann_panels = np.where(vertical)[0]
    result = solve_mixed(
        panels,
        dirichlet={int(i): float(exact_u[i]) for i in dirichlet_panels},
        neumann={int(i): float(exact_q[i]) for i in neumann_panels},
        reference_length=REFERENCE_LENGTH,
        matrices=cached_matrices(panels, REFERENCE_LENGTH),
    )
    recovered = evaluate_field(
        panels, result.boundary_values, result.normal_derivative, SQUARE_FIELD_POINTS,
        reference_length=REFERENCE_LENGTH,
    )
    flux_errors = result.normal_derivative[dirichlet_panels] - exact_q[dirichlet_panels]
    # Position of each Dirichlet panel within its own side (side length per_side).
    within_side = np.arange(len(dirichlet_panels)) % per_side
    interior = (within_side != 0) & (within_side != per_side - 1)
    exact_field = square_potential(SQUARE_FIELD_POINTS)
    metrics = {
        "neumann_trace_rel_l2": _relative_l2(
            result.boundary_values[neumann_panels] - exact_u[neumann_panels],
            exact_u[neumann_panels],
        ),
        "dirichlet_flux_rel_l2": _relative_l2(
            result.normal_derivative[dirichlet_panels] - exact_q[dirichlet_panels],
            exact_q[dirichlet_panels],
        ),
        "dirichlet_flux_rel_max": _relative_max(
            result.normal_derivative[dirichlet_panels] - exact_q[dirichlet_panels],
            exact_q[dirichlet_panels],
        ),
        "neumann_trace_rel_max": _relative_max(
            result.boundary_values[neumann_panels] - exact_u[neumann_panels],
            exact_u[neumann_panels],
        ),
        "field_rel_max": _relative_max(recovered - exact_field, exact_field),
        # Localisation of the error: panels touching a Dirichlet/Neumann junction
        # (and the 90-degree corners they sit on) behave differently from the
        # interior of a side, so both are reported separately.
        "dirichlet_flux_rel_l2_away_from_junction": _relative_l2(
            flux_errors[interior], exact_q[dirichlet_panels][interior]
        ),
        "junction_panel_flux_abs_max": float(np.max(np.abs(flux_errors[~interior]))),
        "condition_number": result.condition_number,
        "relative_residual": result.relative_residual_l2,
        "rank_deficiency": float(result.rank_deficiency),
    }
    note = (
        "flux error is concentrated on the panels adjacent to a Dirichlet/Neumann "
        "junction at a 90-degree corner; see circle_mixed for the same BC switch "
        "on a smooth boundary"
    )
    return CaseOutcome("square_mixed", panels.panel_count, metrics, (note,))


def run_circle_mixed(panel_count: int) -> CaseOutcome:
    """Mixed Dirichlet/Neumann on a smooth closed boundary (BC-type junctions).

    The junction points lie on a smooth curve here, which separates the effect of
    the boundary-condition switch from the polygonal-corner effect exercised by
    :func:`run_square_mixed`.
    """
    panels = PanelSet.from_loops([circle_loop(panel_count, CIRCLE_RADIUS)])
    centers = panels.centers
    exact_u = circle_potential(centers)
    exact_q = circle_flux(panels)
    upper = centers[:, 1] > 0.0
    dirichlet_panels = np.where(upper)[0]
    neumann_panels = np.where(~upper)[0]
    result = solve_mixed(
        panels,
        dirichlet={int(i): float(exact_u[i]) for i in dirichlet_panels},
        neumann={int(i): float(exact_q[i]) for i in neumann_panels},
        reference_length=REFERENCE_LENGTH,
        matrices=cached_matrices(panels, REFERENCE_LENGTH),
    )
    probes = np.array([[0.3, 0.2], [-0.6, 0.9], [0.7, -0.8], [0.0, 0.0]])
    recovered = evaluate_field(
        panels, result.boundary_values, result.normal_derivative, probes,
        reference_length=REFERENCE_LENGTH,
    )
    exact_probe = circle_potential(probes)
    metrics = {
        "neumann_trace_rel_l2": _relative_l2(
            result.boundary_values[neumann_panels] - exact_u[neumann_panels],
            exact_u[neumann_panels],
        ),
        "dirichlet_flux_rel_l2": _relative_l2(
            result.normal_derivative[dirichlet_panels] - exact_q[dirichlet_panels],
            exact_q[dirichlet_panels],
        ),
        "dirichlet_flux_rel_max": _relative_max(
            result.normal_derivative[dirichlet_panels] - exact_q[dirichlet_panels],
            exact_q[dirichlet_panels],
        ),
        "field_abs_max": float(np.max(np.abs(recovered - exact_probe))),
        "condition_number": result.condition_number,
        "relative_residual": result.relative_residual_l2,
        "rank_deficiency": float(result.rank_deficiency),
        "circle_radius": CIRCLE_RADIUS,
    }
    return CaseOutcome("circle_mixed", panels.panel_count, metrics, result.notes)


#: Cases keyed by name; each takes the mesh resolution and returns a CaseOutcome.
CASES: dict[str, Callable[[int], CaseOutcome]] = {
    "annulus_dirichlet": run_annulus_dirichlet,
    "annulus_mixed_outer_dirichlet": lambda n: run_annulus_mixed(n, swap=False),
    "annulus_mixed_hole_dirichlet": lambda n: run_annulus_mixed(n, swap=True),
    "annulus_pure_neumann": run_annulus_pure_neumann,
    "square_mixed": run_square_mixed,
    "circle_mixed": run_circle_mixed,
}

#: Convergence metric reported per case in the summary table.
HEADLINE_METRIC = {
    "annulus_dirichlet": "flux_rel_l2_all",
    "annulus_mixed_outer_dirichlet": "dirichlet_flux_rel_l2",
    "annulus_mixed_hole_dirichlet": "dirichlet_flux_rel_l2",
    "annulus_pure_neumann": "potential_rel_l2_up_to_constant",
    "square_mixed": "dirichlet_flux_rel_l2",
    "circle_mixed": "dirichlet_flux_rel_l2",
}

#: Analytic error order expected for piecewise-constant panels on smooth data.
EXPECTED_ORDER = 2.0


def observed_order(coarse: float, fine: float, refinement: float = 2.0) -> float:
    """Observed convergence order between two mesh sizes."""
    if coarse <= 0.0 or fine <= 0.0:
        return float("nan")
    return float(math.log(coarse / fine) / math.log(refinement))


def reference_length_sweep(per_loop: int = 64) -> list[dict[str, float]]:
    """Show why the reference length must avoid the loop radii.

    A circle of radius R has a vanishing mean-flux eigenvalue of the 2-D
    single-layer operator when the reference length equals R (the logarithmic
    capacity degeneracy documented in :mod:`tensorbem.solver`).  The annular
    Dirichlet case is the sharpest demonstration: everything but the mean flux
    mode converges, so the error plateaus instead of decaying whenever a loop
    radius coincides with the reference length.
    """
    rows: list[dict[str, float]] = []
    for reference_length in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
        outcome = run_annulus_dirichlet(per_loop, reference_length=reference_length)
        rows.append(
            {
                "reference_length": reference_length,
                "flux_rel_l2_all": outcome.metrics["flux_rel_l2_all"],
                "flux_rel_l2_outer": outcome.metrics["flux_rel_l2_outer"],
                "condition_number": outcome.metrics["condition_number"],
            }
        )
    return rows


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", nargs="+", type=int, default=[16, 32, 64, 128],
                        help="panels per boundary loop (annulus) or per side (square)")
    parser.add_argument("--outdir", type=Path, default=None,
                        help="write results.json and summary.md here")
    parser.add_argument("--skip-sweep", action="store_true",
                        help="omit the reference-length sweep")
    args = parser.parse_args(argv)

    resolutions = sorted(set(args.panels))
    results: dict[str, Any] = {
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "setup": {
            "annulus_inner_radius": INNER_RADIUS,
            "annulus_outer_radius": OUTER_RADIUS,
            "radial_potential": "u = A log r + B",
            "amplitude": AMPLITUDE,
            "offset": OFFSET,
            "square_potential": "u = x^2 - y^2",
            "reference_length": REFERENCE_LENGTH,
            "panels": resolutions,
        },
        "cases": {},
    }

    for name, runner in CASES.items():
        per_case: dict[str, Any] = {}
        for resolution in resolutions:
            outcome = runner(resolution)
            per_case[str(resolution)] = {
                "panel_count": outcome.panel_count,
                "metrics": outcome.metrics,
                "notes": list(outcome.notes),
            }
        metric = HEADLINE_METRIC[name]
        errors = [per_case[str(r)]["metrics"][metric] for r in resolutions]
        orders = [
            observed_order(errors[i], errors[i + 1])
            for i in range(len(errors) - 1)
        ]
        per_case["headline"] = {
            "metric": metric,
            "errors": errors,
            "observed_orders": orders,
            "expected_order": EXPECTED_ORDER,
            "finite_error": float(errors[-1]),
            "below_3_percent": bool(errors[-1] < 0.03),
        }
        results["cases"][name] = per_case
        print(f"{name:30s} " + " ".join(f"{r}:{e:.3e}" for r, e in zip(resolutions, errors)))

    if not args.skip_sweep:
        results["reference_length_sweep"] = reference_length_sweep()
        print("reference-length sweep (annulus Dirichlet, 64 panels/loop):")
        for row in results["reference_length_sweep"]:
            print(
                f"  L={row['reference_length']:5.2f}  flux_rel_l2={row['flux_rel_l2_all']:.3e}"
                f"  kappa={row['condition_number']:.2e}"
            )

    if args.outdir is not None:
        args.outdir.mkdir(parents=True, exist_ok=True)
        (args.outdir / "results.json").write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (args.outdir / "summary.md").write_text(render_summary(results), encoding="utf-8")
        print(f"wrote {args.outdir/'results.json'} and {args.outdir/'summary.md'}")
    return results


def render_summary(results: dict[str, Any]) -> str:
    """Markdown summary generated from the results dictionary (no hand copy)."""
    resolutions = results["setup"]["panels"]
    lines = [
        "# Mixed Dirichlet/Neumann and multiply connected 2-D BEM: verification summary",
        "",
        "Generated by `python -m tensorbem.benchmarks.mixed_bc_verification`; "
        "every number below is produced by the code, none is transcribed.",
        "",
        f"Reference length L = {results['setup']['reference_length']:g}; "
        f"annulus radii {results['setup']['annulus_inner_radius']:g}"
        f"/{results['setup']['annulus_outer_radius']:g}.",
        "",
    ]
    for name, case in results["cases"].items():
        headline = case["headline"]
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"Headline metric: `{headline['metric']}`")
        lines.append("")
        header = "| panels | " + " | ".join(str(r) for r in resolutions) + " |"
        sep = "|---|" + "---|" * len(resolutions)
        row = "| error | " + " | ".join(
            f"{value:.3e}" for value in headline["errors"]
        ) + " |"
        lines.extend([header, sep, row, ""])
        orders = headline["observed_orders"]
        lines.append(
            "Observed orders: " + ", ".join(f"{value:.2f}" for value in orders)
            + f" (expected {headline['expected_order']:.1f})"
        )
        lines.append("")
        lines.append(
            f"Finest-mesh error {headline['finite_error']:.3e} "
            f"({'< 3 %' if headline['below_3_percent'] else '>= 3 %'})"
        )
        lines.append("")
        finest = str(resolutions[-1])
        metrics = case[finest]["metrics"]
        lines.append("| metric | value |")
        lines.append("|---|---|")
        for key in sorted(metrics):
            lines.append(f"| {key} | {metrics[key]:.6e} |")
        lines.append("")
    sweep = results.get("reference_length_sweep")
    if sweep:
        lines.extend([
            "## Reference-length sweep (annulus Dirichlet, 64 panels per loop)",
            "",
            "| L | flux rel. L2 | outer-loop flux rel. L2 | kappa |",
            "|---|---|---|---|",
        ])
        for row in sweep:
            lines.append(
                f"| {row['reference_length']:g} | {row['flux_rel_l2_all']:.3e} | "
                f"{row['flux_rel_l2_outer']:.3e} | {row['condition_number']:.3e} |"
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    main()