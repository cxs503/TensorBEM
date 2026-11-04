"""Multi-connected (annulus) verification of the 2-D Laplace Dirichlet solver.

Why this module exists
----------------------
``docs/verification.md`` §1 registers the annulus radial-conduction case
``u = A log r + B`` as *not verified*, with the reason spelled out: the case
needs multi-component boundaries and in/out normal handling, which the
single-polygon mesh of :mod:`tensorbem.geometry` cannot express
("后续：需多边界组件").  Every other Dirichlet case in that table is a
*simply connected* domain, so the multi-connected topology -- two loops, one of
them with a normal pointing into the hole -- had no analytic check at all.

This module closes that cell with three analytic families on the same annulus
``0.5 <= r <= 1``:

``log``
    ``u = A log r + B``, the registered case.  Its flux is not zero per loop:
    ``oint_{outer} q ds = 2 pi A`` and ``oint_{hole} q ds = -2 pi A``, while the
    *total* flux vanishes because no source sits in the fluid.  That per-loop
    circulation is a property a single-loop mesh cannot even express, so it is
    reported separately from the flux error.
``power1``, ``power2``
    ``u = Re(z^n + z^-n)``, ``n = 1, 2``.  ``z^-n`` is harmonic and single
    valued in the annulus but singular at the origin, which lies *outside* the
    fluid: the pair loads the hole boundary as strongly as the outer one.

Every number in the accompanying document is produced here.  Three independent
checks are reported per (case, mesh):

* flux error against the closed-form ``grad(u).n``;
* interior probe error against the closed-form potential;
* the per-loop flux and the total-flux identity, both computed from the panel
  solution alone, i.e. without the analytic field entering the check (the
  analytic per-loop value is quoted alongside for comparison).

The negative control is the same annulus with the hole *not* reversed, so its
normal points into the fluid.  If orientation were cosmetic that run would
reproduce the correct numbers; it does not.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from ..geometry import regular_polygon
from ..solver import solve_laplace_dirichlet
from ..multicomponent import (
    annulus_mesh,
    evaluate_fluid_field,
    log_field,
    observed_order,
    power_field,
    run_annulus_case,
    run_wrong_orientation_control,
    solve_multicomponent_dirichlet,
)

INNER_RADIUS = 0.5
OUTER_RADIUS = 1.0
DEFAULT_LADDER = (16, 32, 64, 128)


def _case_specs(degrees: Sequence[int] = (1, 2)) -> list[tuple[str, callable]]:
    cases: list[tuple[str, callable]] = [
        ("log", lambda: log_field(amplitude=1.0, offset=0.5))
    ]
    for n in degrees:
        cases.append((f"power{int(n)}", (lambda n=int(n): power_field(n))))
    return cases


def study(
    ladder: Sequence[int] = DEFAULT_LADDER,
    degrees: Sequence[int] = (1, 2),
    inner_radius: float = INNER_RADIUS,
    outer_radius: float = OUTER_RADIUS,
) -> dict[str, object]:
    """Panel-refinement study of the annulus Dirichlet problem."""
    cases = _case_specs(degrees)
    rows: list[dict[str, object]] = []
    for case_name, spec in cases:
        for panels in ladder:
            row = run_annulus_case(
                spec,
                panels_per_loop=int(panels),
                inner_radius=inner_radius,
                outer_radius=outer_radius,
            )
            row["case_key"] = case_name
            rows.append(row)

    orders: dict[str, dict[str, float]] = {}
    summary: dict[str, dict[str, object]] = {}
    for case_name, _ in cases:
        case_rows = [r for r in rows if r["case_key"] == case_name]
        orders[case_name] = {
            quantity: observed_order(
                float(case_rows[-2][quantity]), float(case_rows[-1][quantity])
            )
            for quantity in ("flux_rel_l2", "flux_max_rel", "field_rmse")
        }
        finest = case_rows[-1]
        worst = max(case_rows, key=lambda r: float(r["flux_rel_l2"]))
        # Per-loop flux: compare the panel solution with the analytic per-loop
        # value, relative to 2 pi |A| (the magnitude of the loop circulation).
        circulation = 2.0 * math.pi * abs(1.0)
        per_loop_error = [
            abs(float(q) - float(q_ref)) / circulation
            for q, q_ref in zip(finest["per_loop_flux"], finest["per_loop_flux_reference"])
        ]
        summary[case_name] = {
            "finest_panels_per_loop": int(finest["panels_per_loop"]),
            "finest_flux_rel_l2": float(finest["flux_rel_l2"]),
            "finest_flux_max_rel": float(finest["flux_max_rel"]),
            "finest_field_rmse": float(finest["field_rmse"]),
            "finest_field_max_rel": float(finest["field_max_rel"]),
            "worst_flux_rel_l2": float(worst["flux_rel_l2"]),
            "worst_panels_per_loop": int(worst["panels_per_loop"]),
            "finest_condition_number": float(finest["condition_number"]),
            "finest_relative_residual_l2": float(finest["relative_residual_l2"]),
            "finest_net_flux_relative": float(finest["net_flux_relative"]),
            "finest_per_loop_flux": [float(v) for v in finest["per_loop_flux"]],
            "finest_per_loop_flux_reference": [
                float(v) for v in finest["per_loop_flux_reference"]
            ],
            "finest_per_loop_flux_relative_error": per_loop_error,
            "monotone_flux": all(
                float(b["flux_rel_l2"]) < float(a["flux_rel_l2"])
                for a, b in zip(case_rows, case_rows[1:])
            ),
            "orders": orders[case_name],
        }

    conditioning = conditioning_comparison(
        inner_radius=inner_radius, outer_radius=outer_radius, panels_per_loop=(64, 128)
    )
    control = run_wrong_orientation_control(
        inner_radius=inner_radius, outer_radius=outer_radius, panels_per_loop=64
    )
    control_row = next(
        r
        for r in rows
        if r["case_key"] == "log" and int(r["panels_per_loop"]) == int(control["panels_per_loop"])
    )
    correct_at_control = float(control_row["flux_rel_l2"])

    return {
        "domain": {
            "shape": "annulus",
            "inner_radius": inner_radius,
            "outer_radius": outer_radius,
            "fluid_area_polygon": rows[-1]["flow_area"],
            "fluid_area_exact": rows[-1]["polygon_area"],
            "ladder_panels_per_loop": [int(n) for n in ladder],
            "probes_per_case": 18,
            "panels_total": [int(r["panels_total"]) for r in rows[: len(ladder)]],
        },
        "cases": [name for name, _ in cases],
        "rows": rows,
        "orders": orders,
        "summary": summary,
        "conditioning_comparison": conditioning,
        "orientation_control": {
            **control,
            "correct_flux_rel_l2": correct_at_control,
            "correct_condition_number": float(control_row["condition_number"]),
            "correct_relative_residual_l2": float(control_row["relative_residual_l2"]),
            "error_ratio": float(control["flux_rel_l2"]) / correct_at_control,
        },
    }


def conditioning_comparison(
    inner_radius: float = INNER_RADIUS,
    outer_radius: float = OUTER_RADIUS,
    panels_per_loop: Sequence[int] = (64, 128),
) -> list[dict[str, float]]:
    """Condition number of the first-kind operator: one loop versus two.

    The single-layer matrix ``S`` does not depend on the normal orientation, so
    the only difference measured here is the number of boundary components at
    the *same total panel count*: a single circle of radius ``outer_radius`` with
    ``2 N`` panels versus the annulus with ``N`` panels per loop.
    """
    rows: list[dict[str, float]] = []
    field = log_field(amplitude=1.0, offset=0.5)
    for panels in panels_per_loop:
        circle = regular_polygon(2 * int(panels), outer_radius)
        circle_result = solve_laplace_dirichlet(circle, np.log(np.hypot(*circle.centers.T)))
        annulus = annulus_mesh(inner_radius, outer_radius, int(panels))
        annulus_result = solve_multicomponent_dirichlet(annulus, field.values(annulus.centers))
        rows.append(
            {
                "panels_per_loop": int(panels),
                "total_panels": int(2 * panels),
                "single_loop_condition_number": float(circle_result.condition_number),
                "two_loop_condition_number": float(annulus_result.condition_number),
                "ratio": float(annulus_result.condition_number)
                / float(circle_result.condition_number),
            }
        )
    return rows


def field_map(
    panels_per_loop: int = 64,
    grid: int = 41,
    inner_radius: float = INNER_RADIUS,
    outer_radius: float = OUTER_RADIUS,
) -> dict[str, object]:
    """Panel solution and pointwise error of the ``log`` case on a Cartesian grid."""
    field = log_field(amplitude=1.0, offset=0.5)
    mesh = annulus_mesh(inner_radius, outer_radius, panels_per_loop)
    values = field.values(mesh.centers)
    result = solve_multicomponent_dirichlet(mesh, values)

    axis = np.linspace(-outer_radius * 1.02, outer_radius * 1.02, grid)
    xg, yg = np.meshgrid(axis, axis)
    flat = np.column_stack((xg.ravel(), yg.ravel()))
    inside = mesh.fluid_contains(flat)
    points = flat[inside]
    u_bem = evaluate_fluid_field(mesh, result, points)
    u_ref = field.values(points)
    error = u_bem - u_ref

    def grid_of(values_flat: np.ndarray, fill: float = float("nan")) -> np.ndarray:
        out = np.full(len(flat), fill)
        out[inside] = values_flat
        return out.reshape(xg.shape)

    return {
        "panels_per_loop": int(panels_per_loop),
        "grid": int(grid),
        "x": axis.tolist(),
        "y": axis.tolist(),
        "u_bem": grid_of(u_bem).tolist(),
        "u_reference": grid_of(u_ref).tolist(),
        "error": grid_of(error).tolist(),
        "max_abs_error": float(np.max(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "points": int(len(points)),
    }


def plot(study_result: dict[str, object], field: dict[str, object], path: Path) -> None:
    """Two-panel figure: field/error map, and convergence with the control bar."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    x = np.asarray(field["x"])
    y = np.asarray(field["y"])
    u_bem = np.asarray(field["u_bem"])
    error = np.asarray(field["error"])

    levels = np.linspace(np.nanmin(u_bem), np.nanmax(u_bem), 21)
    contour = axes[0].contourf(x, y, u_bem, levels=levels, cmap="viridis")
    axes[0].set_aspect("equal")
    axes[0].set_title(
        f"panel solution u = A log r + B\n(reference radius drawn hollow, "
        f"{field['panels_per_loop']} panels/loop)"
    )
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    figure.colorbar(contour, ax=axes[0], shrink=0.85, label="u")

    with np.errstate(invalid="ignore"):
        abs_error = np.abs(error)
    finite = abs_error[np.isfinite(abs_error)]
    floor = max(float(np.min(finite)), 1e-12)
    contour2 = axes[1].contourf(
        x, y, abs_error, levels=np.logspace(np.log10(floor), np.log10(np.max(finite)), 21),
        cmap="magma",
    )
    axes[1].set_aspect("equal")
    axes[1].set_title(
        f"|u_panel - u_exact|  (max {field['max_abs_error']:.2e})"
    )
    axes[1].set_xlabel("x")
    figure.colorbar(contour2, ax=axes[1], shrink=0.85, label="|error| (log)")

    ladder = [int(v) for v in study_result["domain"]["ladder_panels_per_loop"]]
    marker = {"log": "o", "power1": "s", "power2": "^"}
    for case_name in study_result["cases"]:
        case_rows = [r for r in study_result["rows"] if r["case_key"] == case_name]
        panels = [int(r["panels_per_loop"]) for r in case_rows]
        flux = [float(r["flux_rel_l2"]) for r in case_rows]
        axes[2].loglog(panels, flux, marker=marker.get(case_name, "o"), label=f"{case_name} flux L2")
        axes[2].loglog(
            panels,
            [float(r["field_rmse"]) for r in case_rows],
            marker=marker.get(case_name, "o"),
            linestyle="--",
            alpha=0.55,
            label=f"{case_name} probe RMSE",
        )
    reference_panels = np.asarray([ladder[0], ladder[-1]], dtype=float)
    anchor = float(
        next(
            r["flux_rel_l2"]
            for r in study_result["rows"]
            if r["case_key"] == "log" and int(r["panels_per_loop"]) == 16
        )
    )
    axes[2].loglog(reference_panels, anchor * (reference_panels / 16.0) ** (-2.0), "k:", label="slope -2")
    control = study_result["orientation_control"]
    axes[2].axhline(
        float(control["flux_rel_l2"]),
        color="crimson",
        linestyle="-.",
        label=f"hole orientation reversed (control)",
    )
    axes[2].set_xlabel("panels per loop")
    axes[2].set_ylabel("relative error")
    axes[2].set_title("refinement ladder and the negative control")
    axes[2].grid(True, which="both", alpha=0.3)
    axes[2].legend(fontsize=7.5)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def render_markdown(result: dict[str, object], field: dict[str, object]) -> str:
    """Programmatic document body: every number below comes from ``result``."""
    domain = result["domain"]
    summary = result["summary"]
    control = result["orientation_control"]
    rows = result["rows"]
    ladder = domain["ladder_panels_per_loop"]
    cases = result["cases"]

    lines: list[str] = []
    lines.append("### Generated tables\n")
    lines.append(
        "Domain: annulus "
        f"$r_{{\\mathrm{{in}}}} = {domain['inner_radius']:g}$, "
        f"$r_{{\\mathrm{{out}}}} = {domain['outer_radius']:g}$; "
        f"ladder $N = {', '.join(str(n) for n in ladder)}$ panels per loop; "
        f"{domain['probes_per_case']} interior probes per case. "
        "All values written by `python -m tensorbem.benchmarks.multicomponent_annulus`.\n"
    )
    lines.append("### Panel refinement\n")
    header = "| case | " + " | ".join(f"N={n}" for n in ladder) + " | order (last pair) |"
    lines.append(header)
    lines.append("|---|" + "---|" * len(ladder) + "---|")
    for case_name in cases:
        case_rows = [r for r in rows if r["case_key"] == case_name]
        cells = " | ".join(f"{float(r['flux_rel_l2']):.3e}" for r in case_rows)
        lines.append(
            f"| {case_name} flux $L_2$ | {cells} | "
            f"{summary[case_name]['orders']['flux_rel_l2']:.2f} |"
        )
        cells = " | ".join(f"{float(r['field_rmse']):.3e}" for r in case_rows)
        lines.append(
            f"| {case_name} probe RMSE | {cells} | "
            f"{summary[case_name]['orders']['field_rmse']:.2f} |"
        )
    lines.append("")

    lines.append("### Per-loop flux identity (log case, finest mesh)\n")
    finest = summary["log"]
    lines.append("| loop | panel solution $\\oint q\\,ds$ | analytic | relative to $2\\pi A$ |")
    lines.append("|---|---:|---:|---:|")
    for index, (value, reference, error) in enumerate(
        zip(
            finest["finest_per_loop_flux"],
            finest["finest_per_loop_flux_reference"],
            finest["finest_per_loop_flux_relative_error"],
        )
    ):
        label = "outer" if index == 0 else "hole"
        lines.append(f"| {label} | {value:.6f} | {reference:.6f} | {error:.3e} |")
    lines.append(
        f"| total | {sum(finest['finest_per_loop_flux']):.3e} | "
        f"{sum(finest['finest_per_loop_flux_reference']):.3e} | "
        f"net flux / $\\oint|q|\\,ds$ = {finest['finest_net_flux_relative']:.2e} |"
    )
    lines.append("")

    lines.append("### Mesh conditioning and residuals (finest mesh per case)\n")
    lines.append("| case | condition number | relative residual $L_2$ |")
    lines.append("|---|---:|---:|")
    for case_name in cases:
        entry = summary[case_name]
        lines.append(
            f"| {case_name} | {entry['finest_condition_number']:.2f} | "
            f"{entry['finest_relative_residual_l2']:.2e} |"
        )
    lines.append("")

    lines.append("### Negative control: hole orientation\n")
    lines.append(
        f"| configuration | flux $L_2$ error | condition number |\n|---|---:|---:|\n"
        f"| hole reversed (correct) | {control['correct_flux_rel_l2']:.3e} | "
        f"{control['correct_condition_number']:.2f} |\n"
        f"| hole not reversed (control) | {control['flux_rel_l2']:.3e} | "
        f"{control['condition_number']:.2f} |\n"
        f"\nError ratio control/correct = **{control['error_ratio']:.0f}x** at "
        f"N = {control['panels_per_loop']} panels per loop. The reversed mesh still "
        "solves its own system to round-off "
        f"(relative residual {control['relative_residual_l2']:.1e}), so the wrong answer "
        "is not detectable from the residual alone. The conditioning does not flag it "
        "either: the single-layer matrix $S$ is orientation independent, only the "
        "double-layer $H$ changes sign on the hole, so "
        f"$\\kappa = {control['condition_number']:.2f}$ in both cases. That is why the "
        "orientation check is enforced in `MultiComponentMesh` instead of being left "
        "to the caller.\n"
    )

    lines.append("### Pointwise field error map (log case)\n")
    lines.append(
        f"$N$ = {field['panels_per_loop']} panels per loop, {field['points']} grid points "
        f"inside the fluid, max $|u_{{panel}} - u_{{exact}}|$ = {field['max_abs_error']:.3e}, "
        f"RMSE = {field['rmse']:.3e}.\n"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="write the study as JSON here")
    parser.add_argument("--field-json", type=Path, help="write the field map as JSON here")
    parser.add_argument("--figure", type=Path, help="write the figure here")
    parser.add_argument("--markdown", type=Path, help="write the generated markdown here")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="short ladder and coarse grid, for smoke tests",
    )
    arguments = parser.parse_args()
    ladder = (16, 32, 64) if arguments.quick else DEFAULT_LADDER
    grid = 17 if arguments.quick else 41
    panels_for_map = 32 if arguments.quick else 64
    result = study(ladder=ladder)
    field = field_map(panels_per_loop=panels_for_map, grid=grid)
    if arguments.json:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(json.dumps(result, indent=2) + "\n")
    if arguments.field_json:
        arguments.field_json.parent.mkdir(parents=True, exist_ok=True)
        arguments.field_json.write_text(json.dumps(field, indent=2) + "\n")
    if arguments.figure:
        plot(result, field, arguments.figure)
    if arguments.markdown:
        arguments.markdown.parent.mkdir(parents=True, exist_ok=True)
        arguments.markdown.write_text(render_markdown(result, field) + "\n")
    print(json.dumps({k: v["finest_flux_rel_l2"] for k, v in result["summary"].items()}, indent=2))


if __name__ == "__main__":
    main()