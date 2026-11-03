"""Near-surface field evaluation study: fixed-order Duffy versus the exact edge rule.

Reproduces the numbers in ``docs/near_singular_field_3d.md``:

* panel level: relative error of the single-layer potential of one flat triangle
  seen from a nearby target, for the legacy 8x8 Duffy rule and for the exact
  edge rule, against an independent polar/analytic-radial reference;
* solver level: relative error of the potential of a sphere with Dirichlet data
  ``u = 1`` (exact ``phi = 1/r``) at clearances down to ``5e-4`` radii, for the
  legacy rule (which the built-in ``evaluate_surface_field`` refuses below
  ``0.05 h_max``) and for the representation assembled with the exact edge-rule
  single-layer rows of :mod:`tensorbem.near_field3d`.

Usage::

    python -m tensorbem.benchmarks.near_field_study [--json path] [--figure path]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..geometry3d import icosphere
from ..kernels3d import duffy_quadrature_batch, solid_angle_triangles_batch
from ..near_field3d import (
    integrate_single_layer_flat_triangles,
    integrate_single_layer_flat_triangles_batch,
    polar_reference_integrals,
)
from ..solver3d import solve_laplace_surface

TRIANGLE = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.3, 0.0]])
CLEARANCES = (1.0, 0.25, 0.05, 0.02, 5.0e-3, 1.0e-3)
SPHERE_CLEARANCES = (0.3, 0.05, 0.01, 2.0e-3, 5.0e-4)


def _legacy_single_layer_row(target, mesh, order: int = 8):
    points, weights = duffy_quadrature_batch(mesh.v0, mesh.v1, mesh.v2, order=order)
    points = points.reshape(-1, 3)
    weights = weights.reshape(-1)
    distance = np.linalg.norm(target[None, :] - points, axis=1)
    per_panel = (weights / (4.0 * np.pi * distance)).reshape(mesh.panel_count, -1)
    return per_panel.sum(axis=1)


def panel_level_table() -> list[dict[str, float]]:
    v0 = TRIANGLE[0:1]
    v1 = TRIANGLE[1:2]
    v2 = TRIANGLE[2:3]
    rows = []
    for clearance in CLEARANCES:
        targets = [
            np.array([0.1, 0.1, clearance]),
            np.array([0.33, 0.06, clearance]),
            np.array([-0.2, 0.4, clearance]),
        ]
        exact_errors = []
        legacy_errors = []
        for target in targets:
            reference, _ = polar_reference_integrals(target, v0, v1, v2, angular_order=240)
            exact, _ = integrate_single_layer_flat_triangles(target, v0, v1, v2, order=24)
            points, weights = duffy_quadrature_batch(v0, v1, v2, order=8)
            points = points.reshape(-1, 3)
            weights = weights.reshape(-1)
            distance = np.linalg.norm(target[None, :] - points, axis=1)
            legacy = float(np.sum(weights / (4.0 * np.pi * distance)))
            exact_errors.append(abs(exact[0] - reference[0]) / abs(reference[0]))
            legacy_errors.append(abs(legacy - reference[0]) / abs(reference[0]))
        rows.append(
            {
                "clearance": clearance,
                "exact_max_rel_error": float(max(exact_errors)),
                "legacy_max_rel_error": float(max(legacy_errors)),
            }
        )
    return rows


def sphere_level_table(subdivisions: int) -> dict[str, object]:
    mesh = icosphere(subdivisions, radius=1.0)
    result = solve_laplace_surface(mesh, np.ones(mesh.panel_count), domain="exterior")
    clearances = np.array(SPHERE_CLEARANCES)
    targets = np.stack(
        [np.zeros_like(clearances), np.zeros_like(clearances), 1.0 + clearances], axis=1
    )
    exact = 1.0 / np.linalg.norm(targets, axis=1)
    single_rows, _ = integrate_single_layer_flat_triangles_batch(
        targets, mesh.v0, mesh.v1, mesh.v2, order=24
    )
    rows = []
    for index, clearance in enumerate(clearances):
        target = targets[index]
        single_row = _legacy_single_layer_row(target, mesh)
        double_row = -solid_angle_triangles_batch(target, mesh.v0, mesh.v1, mesh.v2) / (
            4.0 * np.pi
        )
        legacy = single_row @ result.normal_derivative + double_row @ result.boundary_values
        exact_value = (
            single_rows[index] @ result.normal_derivative
            + double_row @ result.boundary_values
        )
        rows.append(
            {
                "clearance": float(clearance),
                "clearance_over_hmax": float(clearance / mesh.h_max),
                "exact_rule": float(abs(exact_value - exact[index]) / exact[index]),
                "legacy_duffy": float(abs(legacy - exact[index]) / exact[index]),
                "legacy_would_refuse": bool(clearance < 0.05 * mesh.h_max),
            }
        )
    return {
        "subdivisions": subdivisions,
        "panels": mesh.panel_count,
        "h_max": float(mesh.h_max),
        "h_mean": float(mesh.h_mean),
        "condition_number": float(result.condition_number),
        "relative_residual": float(result.relative_residual_l2),
        "rows": rows,
    }


def build_report() -> dict[str, object]:
    return {
        "panel_level": panel_level_table(),
        "spheres": [sphere_level_table(2), sphere_level_table(3)],
        "settings": {
            "panel_triangle": TRIANGLE.tolist(),
            "panel_area": float(
                0.5
                * np.linalg.norm(np.cross(TRIANGLE[1] - TRIANGLE[0], TRIANGLE[2] - TRIANGLE[0]))
            ),
            "edge_order": 24,
            "legacy_duffy_order": 8,
            "reference": "polar/analytic-radial (independent scheme)",
        },
    }


def write_figure(report: dict[str, object], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    panel = report["panel_level"]
    clearances = np.array([row["clearance"] for row in panel])
    axes[0].loglog(
        clearances,
        [row["legacy_max_rel_error"] for row in panel],
        "o-",
        label="legacy 8x8 Duffy",
    )
    axes[0].loglog(
        clearances,
        [row["exact_max_rel_error"] for row in panel],
        "s-",
        label="exact edge rule (order 24)",
    )
    axes[0].set_xlabel("clearance / panel edge length")
    axes[0].set_ylabel("relative error of $\\int_T G\\,dS$")
    axes[0].set_title("one flat panel: single-layer potential")
    axes[0].invert_xaxis()
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    for sphere, style in zip(report["spheres"], ("o-", "s--")):
        rows = sphere["rows"]
        clearance = np.array([row["clearance"] for row in rows])
        axes[1].loglog(clearance, [row["legacy_duffy"] for row in rows], style,
                       label=f"legacy Duffy, {sphere['panels']} panels")
        axes[1].loglog(clearance, [row["exact_rule"] for row in rows], style,
                       alpha=0.45, label=f"exact rule, {sphere['panels']} panels")
    axes[1].set_xlabel("clearance from the sphere surface (radii)")
    axes[1].set_ylabel("relative error of $\\phi$ vs $1/r$")
    axes[1].set_title("sphere Dirichlet problem, $u=1$")
    axes[1].invert_xaxis()
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default="docs/m2489/near_field_study.json")
    parser.add_argument("--figure", default="docs/figures/near_singular_field_3d.png")
    arguments = parser.parse_args()
    report = build_report()
    json_path = Path(arguments.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    figure_path = Path(arguments.figure)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    write_figure(report, figure_path)
    print(f"wrote {json_path} and {figure_path}")
    for row in report["panel_level"]:
        print(
            "panel clearance=%.4g legacy=%.3e exact=%.3e"
            % (row["clearance"], row["legacy_max_rel_error"], row["exact_max_rel_error"])
        )
    for sphere in report["spheres"]:
        print(f"sphere panels={sphere['panels']} h_max={sphere['h_max']:.4f}")
        for row in sphere["rows"]:
            print(
                "  clearance=%.4g exact=%.3e legacy=%.3e refused_before=%s"
                % (
                    row["clearance"],
                    row["exact_rule"],
                    row["legacy_duffy"],
                    row["legacy_would_refuse"],
                )
            )


if __name__ == "__main__":
    main()