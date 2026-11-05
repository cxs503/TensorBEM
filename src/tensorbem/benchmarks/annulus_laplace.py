"""Analytic verification of the 2-D Laplace solver on a multiply connected domain.

``docs/verification.md`` §1 lists the annulus ``u = A log r + B`` as
"后续：需多边界组件" (needs a multi-component boundary); :mod:`tensorbem.multiloop`
now supplies it.  This study turns that row into executable evidence.

Each case has a closed-form harmonic solution, so the reference is a formula,
never a second discretisation:

``radial_log``
    ``u = A log r + B`` on ``b < r < a``.  The gradient is radial and the two
    boundary values differ, so the field exercises the hole: an annular
    discretisation that silently drops the inner loop gets the outer trace right
    and the outer flux wrong by ``2`` (flux in = flux out fails).

``angular_cos``
    ``u = (r + c^2/r) cos(theta)`` with ``c != b``.  Harmonic, non-radial, and
    the inner normal derivative is *not* constant along the loop, so panel
    quadrature on both components is tested in a way the radial case cannot
    reach (on the radial case the exact ``q`` is constant along each loop).

Three checks ride along:

``flux_balance``
    ``phi q ds`` per loop, compared against the analytic ``2 pi A`` / ``-2 pi A``.
    This is the discrete form of the statement that a harmonic field has no net
    source in the annulus.

``orientation_control``
    The same meshes solved with the inner loop's normals *not* flipped.  The
    module docstring claims the convention matters; this control quantifies how
    much, so the claim is not a comment but a measured error.

``capacity_sweep``
    Condition number of the single-layer matrix as the outer radius approaches
    the package reference length ``L = 2``.  It documents the limit of validity
    (the logarithmic-capacity degeneracy already noted by ``solver._solve``)
    rather than hiding it.

Run::

    python -m tensorbem.benchmarks.annulus_laplace --output docs/annulus_laplace/study.json

Every number in ``docs/annulus_laplace.md`` is rendered from that artifact.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..multiloop import (
    MultiLoopMesh,
    annulus_mesh,
    boundary_flux,
    evaluate_multiloop_field,
    solve_laplace_multiloop,
)

FloatArray = NDArray[np.float64]

OUTER_RADIUS = 1.0
INNER_RADIUS = 0.5
RADIAL_A = 0.7
RADIAL_B = 1.3
ANGULAR_C = 0.4
LADDER = ((16, 8), (32, 16), (64, 32), (128, 64))
PROBE_RADII = (0.6, 0.75, 0.9)
PROBE_ANGLES = 16
CAPACITY_OUTER_RADII = (1.0, 1.5, 1.9, 1.99, 2.0, 2.01, 2.5)
ORIENTATION_PANELS = (64, 32)


# --------------------------------------------------------------------------
# analytic references
# --------------------------------------------------------------------------
def radial_log_potential(points: FloatArray) -> FloatArray:
    """``u = A log r + B`` (harmonic away from the origin)."""
    return RADIAL_A * np.log(np.linalg.norm(points, axis=1)) + RADIAL_B


def radial_log_normal_derivative(
    points: FloatArray, normals: FloatArray
) -> FloatArray:
    """``du/dn = A * (x . n) / r^2`` along a domain-outward normal."""
    radius = np.linalg.norm(points, axis=1)
    return RADIAL_A * np.sum(points * normals, axis=1) / radius**2


def angular_cos_potential(points: FloatArray) -> FloatArray:
    """``u = (r + c^2/r) cos(theta) = x * (1 + c^2/r^2)``."""
    x = points[:, 0]
    radius2 = np.sum(points * points, axis=1)
    return x * (1.0 + ANGULAR_C**2 / radius2)


def angular_cos_normal_derivative(
    points: FloatArray, normals: FloatArray
) -> FloatArray:
    """``du/dn`` of :func:`angular_cos_potential` along a domain-outward normal.

    With ``u = x (1 + c^2/r^2)``:
    ``du/dx = 1 + c^2/r^2 - 2 c^2 x^2 / r^4`` and
    ``du/dy = -2 c^2 x y / r^4``.
    Checked against central differences in ``tests/test_multiloop_annulus.py``.
    """
    x = points[:, 0]
    y = points[:, 1]
    radius2 = np.sum(points * points, axis=1)
    radius4 = radius2**2
    grad_x = 1.0 + ANGULAR_C**2 / radius2 - 2.0 * ANGULAR_C**2 * x**2 / radius4
    grad_y = -2.0 * ANGULAR_C**2 * x * y / radius4
    return grad_x * normals[:, 0] + grad_y * normals[:, 1]


@dataclass(frozen=True)
class Case:
    name: str
    doc: str
    potential: object
    normal_derivative: object


CASES = (
    Case(
        name="radial_log",
        doc="u = 0.7 log r + 1.3",
        potential=radial_log_potential,
        normal_derivative=radial_log_normal_derivative,
    ),
    Case(
        name="angular_cos",
        doc="u = (r + 0.16/r) cos(theta)",
        potential=angular_cos_potential,
        normal_derivative=angular_cos_normal_derivative,
    ),
)


# --------------------------------------------------------------------------
# measurement helpers
# --------------------------------------------------------------------------
def probe_points() -> FloatArray:
    """Fixed interior probe set: annular radii x uniformly spaced angles."""
    angles = 2.0 * np.pi * np.arange(PROBE_ANGLES) / PROBE_ANGLES
    points = [
        [radius * math.cos(angle), radius * math.sin(angle)]
        for radius in PROBE_RADII
        for angle in angles
    ]
    return np.asarray(points, dtype=float)


def _relative_l2(error: FloatArray, reference: FloatArray) -> float:
    denominator = float(np.linalg.norm(reference))
    numerator = float(np.linalg.norm(error))
    return numerator / denominator if denominator > 0.0 else numerator


def measure_level(case: Case, outer_panels: int, inner_panels: int) -> dict:
    """Solve one case at one mesh level and report every error metric."""
    mesh = annulus_mesh(outer_panels, inner_panels, OUTER_RADIUS, INNER_RADIUS)
    centers = mesh.centers
    normals = mesh.normals
    data = case.potential(centers)  # type: ignore[operator]
    result = solve_laplace_multiloop(mesh, data)
    exact_q = case.normal_derivative(centers, normals)  # type: ignore[operator]
    numeric_q = result.normal_derivative

    points = probe_points()
    numeric_u = evaluate_multiloop_field(mesh, result, points)
    exact_u = case.potential(points)  # type: ignore[operator]
    flux = boundary_flux(mesh, numeric_q)
    exact_flux = boundary_flux(mesh, exact_q)

    return {
        "outer_panels": outer_panels,
        "inner_panels": inner_panels,
        "panel_count": mesh.panel_count,
        "condition_number": result.condition_number,
        "relative_residual": result.relative_residual_l2,
        "flux_outer": flux[0],
        "flux_inner": flux[1],
        "exact_flux_outer": float(exact_flux[0]),
        "exact_flux_inner": float(exact_flux[1]),
        "flux_imbalance": float(abs(flux[0] + flux[1])),
        "flux_scale": float(np.sum(np.abs(exact_q) * mesh.lengths)),
        # normalised by the integral of |q| over the boundary, not by the exact
        # flux: the exact flux of a closed loop vanishes for angular_cos, so
        # dividing by it would report an arbitrary ratio.
        "flux_relative_error": float(
            np.linalg.norm(np.asarray(flux) - np.asarray(exact_flux))
            / float(np.sum(np.abs(exact_q) * mesh.lengths))
        ),
        "q_relative_l2_error": _relative_l2(numeric_q - exact_q, exact_q),
        "q_relative_linf_error": float(
            np.max(np.abs(numeric_q - exact_q)) / np.max(np.abs(exact_q))
        ),
        "field_rmse": float(np.sqrt(np.mean((numeric_u - exact_u) ** 2))),
        "field_max_abs_error": float(np.max(np.abs(numeric_u - exact_u))),
        "field_scale": float(np.max(np.abs(exact_u))),
    }


def observed_orders(levels: list[dict], key: str) -> list[float]:
    """log2 error ratio between consecutive refinement levels."""
    orders = []
    for coarse, fine in zip(levels, levels[1:]):
        numerator = coarse[key]
        denominator = fine[key]
        if numerator <= 0.0 or denominator <= 0.0:
            orders.append(float("nan"))
        else:
            orders.append(float(math.log2(numerator / denominator)))
    return orders


def orientation_control(case: Case, outer_panels: int, inner_panels: int) -> dict:
    """Same mesh and data, inner loop normals left as ``PanelMesh`` reports them."""
    flipped = annulus_mesh(outer_panels, inner_panels, OUTER_RADIUS, INNER_RADIUS)
    unflipped = MultiLoopMesh(loops=flipped.loops, hole_loops=())
    data_flipped = case.potential(flipped.centers)  # type: ignore[operator]
    data_unflipped = case.potential(unflipped.centers)  # type: ignore[operator]
    exact_q = case.normal_derivative(flipped.centers, flipped.normals)  # type: ignore[operator]

    good = solve_laplace_multiloop(flipped, data_flipped)
    bad = solve_laplace_multiloop(unflipped, data_unflipped)
    return {
        "case": case.name,
        "outer_panels": outer_panels,
        "inner_panels": inner_panels,
        "correct_convention_q_relative_l2_error": _relative_l2(
            good.normal_derivative - exact_q, exact_q
        ),
        "unflipped_inner_loop_q_relative_l2_error": _relative_l2(
            bad.normal_derivative - exact_q, exact_q
        ),
    }


def capacity_sweep() -> list[dict]:
    """Condition number vs outer radius at fixed panel counts."""
    rows = []
    for outer_radius in CAPACITY_OUTER_RADII:
        mesh = annulus_mesh(32, 16, outer_radius, INNER_RADIUS)
        result = solve_laplace_multiloop(mesh, radial_log_potential(mesh.centers))
        exact_q = radial_log_normal_derivative(mesh.centers, mesh.normals)
        rows.append(
            {
                "outer_radius": float(outer_radius),
                "inner_radius": INNER_RADIUS,
                "reference_length": result.reference_length,
                "condition_number": result.condition_number,
                "q_relative_l2_error": _relative_l2(
                    result.normal_derivative - exact_q, exact_q
                ),
            }
        )
    return rows


def run_study(ladder: tuple[tuple[int, int], ...] | None = None) -> dict:
    """Run every case, control and sweep; return the strict JSON payload."""
    ladder = LADDER if ladder is None else tuple(ladder)
    cases = []
    for case in CASES:
        levels = [measure_level(case, no, ni) for no, ni in ladder]
        cases.append(
            {
                "name": case.name,
                "declared_solution": case.doc,
                "outer_radius": OUTER_RADIUS,
                "inner_radius": INNER_RADIUS,
                "levels": levels,
                "observed_order_q_relative_l2": observed_orders(
                    levels, "q_relative_l2_error"
                ),
                "observed_order_field_rmse": observed_orders(levels, "field_rmse"),
                "fixture": case is CASES[0],
            }
        )

    finest = {case["name"]: case["levels"][-1] for case in cases}
    controls = [
        orientation_control(CASES[0], *ORIENTATION_PANELS),
        orientation_control(CASES[1], *ORIENTATION_PANELS),
    ]
    return {
        "schema": "tensorbem-annulus-laplace/1",
        "summary": "Analytic verification of 2-D Laplace BEM on a multiply connected (annular) domain",
        "conventions": {
            "reference_length": 2.0,
            "panels": "constant strength, midpoint collocation, analytic straight-panel kernels",
            "normal_derivative": "du/dn along the domain-outward normal; on the inner loop that points into the hole",
            "ladder": [f"{no}/{ni}" for no, ni in (LADDER if ladder is None else ladder)],
        },
        "parameters": {
            "outer_radius": OUTER_RADIUS,
            "inner_radius": INNER_RADIUS,
            "radial_A": RADIAL_A,
            "radial_B": RADIAL_B,
            "angular_c": ANGULAR_C,
            "probe_radii": list(PROBE_RADII),
            "probe_angles": PROBE_ANGLES,
        },
        "cases": cases,
        "orientation_control": controls,
        "capacity_sweep": capacity_sweep(),
        "finest_level": {
            name: {
                key: level[key]
                for key in (
                    "panel_count",
                    "q_relative_l2_error",
                    "field_rmse",
                    "flux_relative_error",
                    "condition_number",
                )
            }
            for name, level in finest.items()
        },
        "acceptance": {
            "q_relative_l2_error_lt": 0.03,
            "field_rmse_lt": 1.0e-3,
            "flux_relative_error_lt": 0.03,
            "min_observed_order": 1.5,
        },
    }


def verdict(study: dict) -> tuple[bool, list[str]]:
    """Apply the acceptance thresholds declared inside the artifact."""
    accept = study["acceptance"]
    messages: list[str] = []
    ok = True
    for case in study["cases"]:
        finest = case["levels"][-1]
        name = case["name"]
        q_error = finest["q_relative_l2_error"]
        if q_error < accept["q_relative_l2_error_lt"]:
            messages.append(f"PASS {name}: q relative L2 error {q_error:.3e} < 3%")
        else:
            ok = False
            messages.append(f"FAIL {name}: q relative L2 error {q_error:.3e} >= 3%")
        rmse = finest["field_rmse"]
        if rmse < accept["field_rmse_lt"]:
            messages.append(f"PASS {name}: field RMSE {rmse:.3e} < 1.0e-3")
        else:
            ok = False
            messages.append(f"FAIL {name}: field RMSE {rmse:.3e} >= 1.0e-3")
        flux = finest["flux_relative_error"]
        if flux < accept["flux_relative_error_lt"]:
            messages.append(f"PASS {name}: flux relative error {flux:.3e} < 3%")
        else:
            ok = False
            messages.append(f"FAIL {name}: flux relative error {flux:.3e} >= 3%")
        orders = [order for order in case["observed_order_q_relative_l2"] if math.isfinite(order)]
        worst = min(orders) if orders else float("nan")
        if orders and worst >= accept["min_observed_order"]:
            messages.append(f"PASS {name}: worst observed order {worst:.3f} >= 1.5")
        else:
            ok = False
            messages.append(f"FAIL {name}: worst observed order {worst:.3f} < 1.5")
    sweep = {row["outer_radius"]: row for row in study["capacity_sweep"]}
    base = sweep[1.0]["condition_number"]
    near = sweep[2.01]["condition_number"]
    if near > 5.0 * base:
        messages.append(
            f"PASS capacity sweep: cond {base:.2f} at R=1 vs {near:.2f} at "
            "R=2.01 (reference length) -- the annulus inherits the logarithmic-"
            "capacity cliff, it does not remove it"
        )
    else:
        ok = False
        messages.append("FAIL capacity sweep: no degeneracy signal near reference length")
    for control in study["orientation_control"]:
        good = control["correct_convention_q_relative_l2_error"]
        bad = control["unflipped_inner_loop_q_relative_l2_error"]
        if bad > 10.0 * good:
            messages.append(
                f"PASS orientation control {control['case']}: flipping the inner "
                f"normals costs {good:.3e} -> {bad:.3e}"
            )
        else:
            ok = False
            messages.append(
                f"FAIL orientation control {control['case']}: {good:.3e} -> {bad:.3e} "
                "is not a distinguishable difference"
            )
    return ok, messages



def render_markdown(study: dict) -> str:
    """Render the result tables of the study as markdown (no hand-typed numbers)."""
    lines: list[str] = []
    for case in study["cases"]:
        lines.append(f"### `{case['name']}` — {case['declared_solution']}")
        lines.append("")
        lines.append("| outer/inner panels | N | cond(S) | ‖q_h−q‖₂/‖q‖₂ | ‖q_h−q‖∞/‖q‖∞ | field RMSE | flux err |")
        lines.append("|---|---|---|---|---|---|---|")
        for level in case["levels"]:
            lines.append(
                "| {}/{} | {} | {:.3e} | {:.3e} | {:.3e} | {:.3e} | {:.3e} |".format(
                    level["outer_panels"],
                    level["inner_panels"],
                    level["panel_count"],
                    level["condition_number"],
                    level["q_relative_l2_error"],
                    level["q_relative_linf_error"],
                    level["field_rmse"],
                    level["flux_relative_error"],
                )
            )
        orders = ", ".join(f"{value:.3f}" for value in case["observed_order_q_relative_l2"])
        field_orders = ", ".join(f"{value:.3f}" for value in case["observed_order_field_rmse"])
        lines.append("")
        lines.append(f"Observed order of the `q` error: **{orders}**; of the field RMSE: **{field_orders}**.")
        lines.append("")

    lines.append("### Flux per loop (finest level of each case)")
    lines.append("")
    lines.append("| case | ∮q ds outer | exact | ∮q ds inner | exact | imbalance |")
    lines.append("|---|---|---|---|---|---|")
    for case in study["cases"]:
        level = case["levels"][-1]
        lines.append(
            "| {} | {:+.6f} | {:+.6f} | {:+.6f} | {:+.6f} | {:.3e} |".format(
                case["name"],
                level["flux_outer"],
                level["exact_flux_outer"],
                level["flux_inner"],
                level["exact_flux_inner"],
                level["flux_imbalance"],
            )
        )
    lines.append("")
    lines.append("### Orientation control (inner loop normals not flipped)")
    lines.append("")
    lines.append("| case | panels | correct ‖q_h−q‖/‖q‖ | unflipped ‖q_h−q‖/‖q‖ | ratio |")
    lines.append("|---|---|---|---|---|")
    for control in study["orientation_control"]:
        good = control["correct_convention_q_relative_l2_error"]
        bad = control["unflipped_inner_loop_q_relative_l2_error"]
        lines.append(
            "| {} | {}/{} | {:.3e} | {:.3e} | {:.1f}× |".format(
                control["case"],
                control["outer_panels"],
                control["inner_panels"],
                good,
                bad,
                bad / good,
            )
        )
    lines.append("")
    lines.append("### Condition number vs outer radius (reference length L = 2)")
    lines.append("")
    lines.append("| outer radius | inner radius | L | cond(S) | ‖q_h−q‖/‖q‖ |")
    lines.append("|---|---|---|---|---|")
    for row in study["capacity_sweep"]:
        lines.append(
            "| {:.3f} | {:.3f} | {:.1f} | {:.3e} | {:.3e} |".format(
                row["outer_radius"],
                row["inner_radius"],
                row["reference_length"],
                row["condition_number"],
                row["q_relative_l2_error"],
            )
        )
    lines.append("")
    return "\n".join(lines)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the strict JSON artifact here (parent directories are created)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if any acceptance threshold in the artifact is violated",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use the three coarsest ladder levels only (test-suite cost)",
    )
    parser.add_argument(
        "--write-doc",
        type=Path,
        default=None,
        help="render the result tables into the marked region of this markdown file",
    )
    args = parser.parse_args(argv)

    study = run_study(ladder=LADDER[:3] if args.quick else None)
    if args.write_doc is not None:
        text = args.write_doc.read_text(encoding="utf-8")
        start = "<!-- tables:start -->"
        end = "<!-- tables:end -->"
        if start not in text or end not in text:
            raise SystemExit(f"{args.write_doc} has no {start} / {end} markers")
        head, rest = text.split(start, 1)
        _, tail = rest.split(end, 1)
        args.write_doc.write_text(
            head + start + "\n\n" + render_markdown(study) + "\n" + end + tail,
            encoding="utf-8",
        )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(study, indent=2) + "\n", encoding="utf-8")
    ok, messages = verdict(study)
    print(json.dumps({"verdict": "PASS" if ok else "FAIL", "checks": messages}, indent=2))
    return 0 if (ok or not args.check) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())