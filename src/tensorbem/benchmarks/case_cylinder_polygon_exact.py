"""Split the anchor case error: panel discretisation vs polygon faceting.

``CASE-BEM-LAPLACE-CYLINDER`` reports the BEM field of an inscribed ``n``-gon
against the *circular* cylinder potential, so its error mixes the boundary
element discretisation of the polygon with the geometric defect of the polygon
itself.  This study re-runs the same solver on the same meshes with the exact
**polygon** solution of :mod:`tensorbem.benchmarks.polygon_stream_exact`:

* ``faceting``      -- ``|v_polygon - v_cylinder|`` (two exact fields, no BEM);
* ``panel_error``   -- ``|v_BEM - v_polygon|`` with *consistently* imposed
  polygon data (Dirichlet: exact perturbation potential at the panel midpoints;
  Neumann: exact fluid-side normal derivative ``U cos(theta_k)``, i.e. no flux
  through the polygon);
* ``case_error``    -- the anchor case's own metric ``|v_BEM - v_cylinder|``,
  together with the data actually imposed there (the cylinder's own boundary
  data sampled at the polygon midpoints).

Wall velocity is recovered by differentiating the boundary integral along
each face. The triangle inequality is checked for this recovered wall field
and the two exact references using one common norm. The original cylinder
anchor's metrics are also reported for comparison.

Usage::

    python -m tensorbem.benchmarks.case_cylinder_polygon_exact --output DIR
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..geometry import regular_polygon
from ..solver import (
    evaluate_exterior_field,
    evaluate_exterior_surface_velocity,
    solve_laplace_exterior_dirichlet,
    solve_laplace_exterior_neumann,
)
from . import case_cylinder_stream as anchor
from . import polygon_stream_exact as pe

STUDY_ID = "BEM-LAPLACE-CYLINDER-POLYGON-EXACT"
SCHEMA = "tensorbem/case_cylinder_polygon_exact/1"
RADIUS = 1.0
SPEED = 1.0
LADDER: tuple[int, ...] = (8, 16, 32, 64, 128)
BUDGET_PANELS = 128
TOLERANCE = 0.03
ORDER_MIN = 1.5


def _relative_l2(computed: np.ndarray, exact: np.ndarray) -> float:
    denominator = float(np.linalg.norm(exact))
    if denominator <= 0.0:
        return float("nan")
    return float(np.linalg.norm(np.asarray(computed) - np.asarray(exact)) / denominator)


def polygon_data(panels: int, radius: float = RADIUS, speed: float = SPEED) -> dict[str, Any]:
    """Exact boundary data of the polygon flow at the panel midpoints."""
    theta = pe.surface_midpoint_angles(panels)
    c = pe.scale_factor(panels, radius)
    mids = pe.surface_midpoint_positions(panels, radius)
    total_potential = speed * c * 2.0 * np.cos(theta)  # Re[U c (w + 1/w)] on |w|=1
    perturbation = total_potential - speed * mids[:, 0]
    # fluid-side normal derivative of the perturbation: q_e = U cos(theta)
    normal_derivative = speed * np.cos(theta)
    velocity, speed_wall = pe.surface_midpoint_velocity(panels, radius, speed)
    return {
        "theta": theta,
        "midpoints": mids,
        "dirichlet_perturbation": perturbation,
        "neumann_fluid_normal": normal_derivative,
        "surface_velocity": velocity,
        "surface_speed": speed_wall,
        "total_potential": total_potential,
    }


def run_level(panels: int, radius: float = RADIUS, speed: float = SPEED) -> dict[str, Any]:
    """Solve both polygon-consistent boundary value problems at one panel count."""
    mesh = regular_polygon(panels, radius)
    data = polygon_data(panels, radius, speed)
    probes = anchor.probe_points()
    exact_probe_velocity = pe.exact_velocity(probes, panels, radius, speed)
    exact_probe_potential = pe.exact_potential(probes, panels, radius, speed) - speed * probes[:, 0]
    cylinder_probe_velocity = anchor.exact_velocity(probes, radius, speed)

    # --- Neumann route with the polygon's own no-flux data ------------------
    neumann = solve_laplace_exterior_neumann(mesh, data["neumann_fluid_normal"])
    surface_velocity = evaluate_exterior_surface_velocity(mesh, neumann, method="boundary_integral") + np.array(
        [speed, 0.0]
    )
    field = evaluate_exterior_field(mesh, neumann, probes)
    neumann_velocity = field.gradient + np.array([speed, 0.0])
    offset = float(np.mean(field.potential - exact_probe_potential))
    cp_numerical = 1.0 - np.sum(surface_velocity**2, axis=1) / speed**2
    cp_exact = 1.0 - data["surface_speed"] ** 2 / speed**2
    neumann_metrics = {
        "surface_velocity_relative_l2": _relative_l2(
            surface_velocity, data["surface_velocity"]
        ),
        "surface_velocity_max_error_over_speed": float(
            np.max(np.linalg.norm(surface_velocity - data["surface_velocity"], axis=1))
        )
        / speed,
        "surface_speed_relative_l2": _relative_l2(
            np.hypot(*surface_velocity.T), data["surface_speed"]
        ),
        "cp_relative_l2": _relative_l2(cp_numerical, cp_exact),
        "surface_velocity_method": "boundary_integral",
        "cp_min": float(np.min(cp_numerical)),
        "cp_min_exact": float(np.min(cp_exact)),
        "normal_velocity_residual_over_speed": float(
            np.max(np.abs(np.sum(surface_velocity * mesh.normals, axis=1)))
        )
        / speed,
        "velocity_relative_l2": _relative_l2(neumann_velocity, exact_probe_velocity),
        "velocity_max_error_over_speed": float(
            np.max(np.linalg.norm(neumann_velocity - exact_probe_velocity, axis=1))
        )
        / speed,
        "potential_relative_l2_up_to_constant": _relative_l2(
            field.potential - offset, exact_probe_potential
        ),
        "potential_offset": offset,
        "condition_number": neumann.condition_number,
        "relative_residual_l2": neumann.relative_residual_l2,
        "case_metric_velocity_relative_l2": _relative_l2(
            neumann_velocity, cylinder_probe_velocity
        ),
    }

    # --- Dirichlet route with the polygon's own potential data --------------
    dirichlet = solve_laplace_exterior_dirichlet(mesh, data["dirichlet_perturbation"])
    dirichlet_field = evaluate_exterior_field(mesh, dirichlet, probes)
    dirichlet_velocity = dirichlet_field.gradient + np.array([speed, 0.0])
    dirichlet_metrics = {
        "potential_relative_l2": _relative_l2(
            dirichlet_field.potential, exact_probe_potential
        ),
        "potential_max_error": float(
            np.max(np.abs(dirichlet_field.potential - exact_probe_potential))
        ),
        "velocity_relative_l2": _relative_l2(dirichlet_velocity, exact_probe_velocity),
        "velocity_max_error_over_speed": float(
            np.max(np.linalg.norm(dirichlet_velocity - exact_probe_velocity, axis=1))
        )
        / speed,
        "condition_number": dirichlet.condition_number,
        "relative_residual_l2": dirichlet.relative_residual_l2,
        "normal_derivative_net_flux_over_speed_radius": float(
            np.sum(dirichlet.normal_derivative * mesh.lengths) / (speed * radius)
        ),
        "case_metric_velocity_relative_l2": _relative_l2(
            dirichlet_velocity, cylinder_probe_velocity
        ),
    }

    # --- faceting: two exact fields on the same points ----------------------
    faceting = {
        "surface_velocity_relative_l2": _relative_l2(
            data["surface_velocity"], anchor.exact_velocity(mesh.centers, radius, speed)
        ),
        "probe_velocity_relative_l2": _relative_l2(
            exact_probe_velocity, cylinder_probe_velocity
        ),
        "probe_potential_relative_l2": _relative_l2(
            exact_probe_potential, anchor.exact_perturbation_potential(probes, radius, speed)
        ),
    }

    # --- the anchor case's own numbers at this panel count -----------------
    case_neumann = anchor.route_neumann(panels, radius, speed)
    case_dirichlet = anchor.route_dirichlet(panels, radius, speed)

    return {
        "panels": int(panels),
        "radius": radius,
        "speed": speed,
        "h_mean": float(np.mean(mesh.lengths)),
        "h_over_radius": float(np.mean(mesh.lengths) / radius),
        "apothem": pe.apothem(panels, radius),
        "geometry_gate": pe.geometry_gate(panels, radius),
        "neumann": neumann_metrics,
        "dirichlet": dirichlet_metrics,
        "faceting": faceting,
        "case_route_neumann": {
            key: case_neumann[key]
            for key in (
                "surface_velocity_relative_l2",
                "velocity_relative_l2",
                "cp_relative_l2",
                "potential_relative_l2_up_to_constant",
                "normal_velocity_residual_over_speed",
            )
        },
        "case_route_dirichlet": {
            key: case_dirichlet[key]
            for key in (
                "velocity_relative_l2",
                "potential_relative_l2",
                "normal_derivative_net_flux_over_speed_radius",
            )
        },
        "decomposition": {
            "case_error": neumann_metrics["velocity_relative_l2"]
            + 0.0,
            "panel_error": neumann_metrics["surface_velocity_relative_l2"],
            "recovered_surface_vs_circle": _relative_l2(
                surface_velocity, anchor.exact_velocity(mesh.centers, radius, speed)
            ),
            "polygon_to_circle_velocity_norm_ratio": float(
                np.linalg.norm(data["surface_velocity"])
                / np.linalg.norm(anchor.exact_velocity(mesh.centers, radius, speed))
            ),
            "faceting_surface": faceting["surface_velocity_relative_l2"],
            "faceting_share_of_case_surface_error": faceting["surface_velocity_relative_l2"]
            / max(
                case_neumann["surface_velocity_relative_l2"],
                1e-300,
            ),
        },
    }


def _orders(levels: list[dict[str, Any]], *path: str) -> list[float]:
    """Observed order of a metric along the ladder, using ``h_mean``."""
    values = []
    for coarse, fine in zip(levels, levels[1:]):
        error_coarse = _dig(coarse, path)
        error_fine = _dig(fine, path)
        h_ratio = coarse["h_mean"] / fine["h_mean"]
        if error_fine <= 0.0 or error_coarse <= 0.0:
            values.append(float("nan"))
        else:
            values.append(math.log(error_coarse / error_fine) / math.log(h_ratio))
    return values


def _dig(level: dict[str, Any], path: Sequence[str]) -> float:
    node: Any = level
    for key in path:
        node = node[key]
    return float(node)


def exact_solution_gates(panels: int, radius: float = RADIUS, speed: float = SPEED) -> dict[str, Any]:
    """Checks that the closed-form polygon solution is a solution of the problem."""
    mesh = regular_polygon(panels, radius)
    data = polygon_data(panels, radius, speed)
    # no flux through every edge, evaluated from the closed form alone
    normal_component = np.sum(
        data["surface_velocity"] * mesh.normals, axis=1
    ) / speed
    # the flow must be symmetric under y -> -y (no lift)
    probes = anchor.probe_points()
    reflected = pe.exact_velocity(
        np.column_stack((probes[:, 0], -probes[:, 1])), panels, radius, speed
    )
    original = pe.exact_velocity(probes, panels, radius, speed)
    symmetry = float(
        np.max(np.abs(reflected[:, 0] - original[:, 0]))
        + np.max(np.abs(reflected[:, 1] + original[:, 1]))
    ) / speed
    far = 1.0e3 * radius
    far_points = np.array([[far, 0.0], [0.0, far], [-0.7 * far, 0.5 * far]])
    far_velocity = pe.exact_velocity(far_points, panels, radius, speed)
    far_error = float(np.max(np.abs(far_velocity - np.array([speed, 0.0])))) / speed
    # the imposed Neumann data must equal the normal derivative of the
    # perturbation potential taken from the closed-form complex velocity
    perturbation_velocity = data["surface_velocity"] - np.array([speed, 0.0])
    fluid_normal = -mesh.normals
    data_check = np.sum(perturbation_velocity * fluid_normal, axis=1) / speed
    data_error = float(np.max(np.abs(data_check - data["neumann_fluid_normal"]))) / speed
    return {
        "panels": panels,
        "no_flux_residual_over_speed": float(np.max(np.abs(normal_component))),
        "mirror_symmetry_residual_over_speed": symmetry,
        "far_field_velocity_error_over_speed": far_error,
        "far_field_radius_over_radius": far / radius,
        "neumann_data_consistency_error_over_speed": data_error,
    }


def criteria(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(name: str, value: Any, threshold: str, ok: bool, note: str) -> None:
        rows.append(
            {"name": name, "value": value, "threshold": threshold, "ok": bool(ok), "note": note}
        )

    levels = artifact["levels"]
    budget = levels[-1]
    gates = artifact["exact_solution_gates"]

    add(
        "geometry_gate_all_levels",
        max(
            level["geometry_gate"]["edge_length_relative_error"]
            + level["geometry_gate"]["midpoint_relative_error"]
            + level["geometry_gate"]["vertex_radius_relative_error"]
            for level in levels
        ),
        "<= 1e-6",
        all(
            level["geometry_gate"]["edge_length_relative_error"] <= 1.0e-6
            and level["geometry_gate"]["midpoint_relative_error"] <= 1.0e-6
            and level["geometry_gate"]["vertex_radius_relative_error"] <= 1.0e-6
            and level["geometry_gate"]["rotation_symmetry_residual"] <= 1.0e-12
            for level in levels
        ),
        "the Schwarz-Christoffel map must reproduce the mesh geometry",
    )
    add(
        "exact_solution_no_flux",
        gates["no_flux_residual_over_speed"],
        "<= 1e-12 U",
        gates["no_flux_residual_over_speed"] <= 1.0e-12,
        "closed-form wall velocity has zero normal component on every edge",
    )
    add(
        "exact_solution_mirror_symmetry",
        gates["mirror_symmetry_residual_over_speed"],
        "<= 1e-12 U",
        gates["mirror_symmetry_residual_over_speed"] <= 1.0e-12,
        "lift-free flow: v_y must change sign under y -> -y",
    )
    add(
        "exact_solution_far_field",
        gates["far_field_velocity_error_over_speed"],
        "<= 1e-5 U",
        gates["far_field_velocity_error_over_speed"] <= 1.0e-5,
        "uniform stream recovered at 1e3 radii",
    )
    add(
        "exact_neumann_data_matches_closed_form_velocity",
        gates["neumann_data_consistency_error_over_speed"],
        "<= 1e-5 U",
        gates["neumann_data_consistency_error_over_speed"] <= 1.0e-5,
        "U cos(theta) re-derived from the closed-form complex velocity",
    )

    for key, label in (
        ("surface_velocity_relative_l2", "polygon-exact wall velocity"),
        ("cp_relative_l2", "polygon-exact Cp"),
        ("velocity_relative_l2", "polygon-exact exterior velocity"),
        ("potential_relative_l2_up_to_constant", "polygon-exact perturbation potential"),
    ):
        add(
            "neumann_" + key + "_budget",
            budget["neumann"][key],
            "< 0.03",
            budget["neumann"][key] < TOLERANCE,
            f"route N vs {label} at {BUDGET_PANELS} panels",
        )
    add(
        "dirichlet_velocity_relative_l2_budget",
        budget["dirichlet"]["velocity_relative_l2"],
        "< 0.03",
        budget["dirichlet"]["velocity_relative_l2"] < TOLERANCE,
        "route D vs polygon-exact exterior velocity at the budget mesh",
    )
    add(
        "dirichlet_potential_relative_l2_budget",
        budget["dirichlet"]["potential_relative_l2"],
        "< 0.03",
        budget["dirichlet"]["potential_relative_l2"] < TOLERANCE,
        "route D vs polygon-exact perturbation potential at the budget mesh",
    )
    add(
        "neumann_no_flux_at_budget",
        budget["neumann"]["normal_velocity_residual_over_speed"],
        "<= 0.01 U",
        budget["neumann"]["normal_velocity_residual_over_speed"] <= 0.01,
        "the polygon must be a streamline for the panel solution",
    )

    surface_orders = _orders(levels, "neumann", "surface_velocity_relative_l2")
    add(
        "neumann_surface_velocity_observed_order",
        surface_orders,
        ">= 1.5",
        bool(surface_orders) and all(order >= ORDER_MIN for order in surface_orders[-2:]),
        "order of the panel solution against the exact polygon field",
    )
    dirichlet_orders = _orders(levels, "dirichlet", "velocity_relative_l2")
    add(
        "dirichlet_velocity_observed_order",
        dirichlet_orders,
        ">= 1.5",
        bool(dirichlet_orders) and all(order >= ORDER_MIN for order in dirichlet_orders[-2:]),
        "order of the exterior field against the exact polygon field",
    )

    minimum_gap = min(
        level["decomposition"]["faceting_share_of_case_surface_error"] for level in levels
    )
    add(
        "faceting_dominates_anchor_surface_metric",
        minimum_gap,
        "> 1 on every ladder level",
        minimum_gap > 1.0,
        "the anchor's wall-velocity error is larger than the panel error because the "
        "circle is the wrong reference for a polygon body",
    )
    # Compare the same reconstructed wall field to both reference fields
    # and express all three distances in units of the circle-reference norm.
    margins = [
        row["faceting"]["surface_velocity_relative_l2"]
        + row["neumann"]["surface_velocity_relative_l2"]
        * row["decomposition"]["polygon_to_circle_velocity_norm_ratio"]
        - row["decomposition"]["recovered_surface_vs_circle"]
        for row in levels
    ]
    add(
        "triangle_inequality_consistent", min(margins), ">= 0",
        min(margins) >= -1e-12,
        "recovered wall field, polygon and cylinder obey the triangle inequality",
    )
    # wrong-reference control: a polygon rotated by pi/n is a different body
    rotated_error = rotated_reference_control(artifact["panels_budget"])
    add(
        "wrong_reference_control",
        rotated_error["relative_l2"],
        ">= 0.03",
        rotated_error["relative_l2"] >= TOLERANCE,
        "the same numbers against the polygon rotated by pi/n must fail the case "
        "tolerance, so the reference is not interchangeable with its symmetric twin",
    )
    return rows


def rotated_reference_control(panels: int, radius: float = RADIUS, speed: float = SPEED) -> dict[str, Any]:
    """Compare the solved wall velocity with the exact flow past a rotated polygon."""
    mesh = regular_polygon(panels, radius)
    data = polygon_data(panels, radius, speed)
    result = solve_laplace_exterior_neumann(mesh, data["neumann_fluid_normal"])
    surface_velocity = evaluate_exterior_surface_velocity(mesh, result, method="boundary_integral") + np.array(
        [speed, 0.0]
    )
    theta = data["theta"] + math.pi / panels
    alpha = 2.0 / panels
    w = np.exp(1j * theta)
    complex_velocity = speed * (1.0 - w**-2) / 2.0**alpha
    wrong = np.column_stack((complex_velocity.real, -complex_velocity.imag))
    return {
        "panels": panels,
        "relative_l2": _relative_l2(surface_velocity, wrong),
        "correct_reference_relative_l2": _relative_l2(
            surface_velocity, data["surface_velocity"]
        ),
        "discrimination_ratio": _relative_l2(surface_velocity, wrong)
        / max(_relative_l2(surface_velocity, data["surface_velocity"]), 1e-300),
    }


def run_study(
    ladder: Sequence[int] = LADDER, radius: float = RADIUS, speed: float = SPEED
) -> dict[str, Any]:
    started = time.perf_counter()
    levels = [run_level(panels, radius, speed) for panels in ladder]
    gates = exact_solution_gates(int(ladder[-1]), radius, speed)
    artifact: dict[str, Any] = {
        "study_id": STUDY_ID,
        "schema": SCHEMA,
        "repo": "TensorBEM",
        "anchor_case": anchor.CASE_ID,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
        },
        "parameters": {
            "radius": radius,
            "speed": speed,
            "panel_ladder": list(ladder),
            "budget_panels": BUDGET_PANELS,
            "probe_rings": list(anchor.PROBE_RINGS),
            "probe_angles": anchor.PROBE_ANGLES,
            "tolerance": TOLERANCE,
        },
        "exact_solution": {
            "map": "z(w) = c int^w (1 - t^{-n})^{2/n} dt, c = a sin(pi/n)/(alpha 2^alpha I(alpha))",
            "complex_potential": "W = U c (w + 1/w)",
            "complex_velocity": "dW/dz = U (1 - w^{-2}) / (1 - w^{-n})^{2/n}",
            "wall_speed": "|v| = 2 U |sin(theta_w)| / 2^{2/n}",
            "reference_form": "closed_form",
            "hand_typed_numbers": 0,
        },
        "levels": levels,
        "exact_solution_gates": gates,
        "panels_budget": int(BUDGET_PANELS),
        "repro_cmd": (
            "python -m tensorbem.benchmarks.case_cylinder_polygon_exact --output "
            "benchmarks/verified/case_bem_laplace_cylinder_polygon_exact"
        ),
    }
    artifact["criteria"] = criteria(artifact)
    artifact["overall"] = "PASS" if all(row["ok"] for row in artifact["criteria"]) else "FAIL"
    artifact["wall_seconds"] = time.perf_counter() - started
    return artifact


def plot_convergence(artifact: dict[str, Any], output: Path) -> Path | None:
    """Convergence of the panel error against the exact polygon flow."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - matplotlib is optional
        return None
    levels = artifact["levels"]
    h = np.array([level["h_mean"] for level in levels])
    panel = np.array([level["neumann"]["surface_velocity_relative_l2"] for level in levels])
    case = np.array(
        [level["case_route_neumann"]["surface_velocity_relative_l2"] for level in levels]
    )
    facet = np.array([level["faceting"]["surface_velocity_relative_l2"] for level in levels])
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    axes[0].loglog(h, panel, "o-", label="BEM vs exact polygon flow (wall velocity)")
    axes[0].loglog(h, case, "s--", label="anchor metric: BEM vs circle")
    axes[0].loglog(h, facet, "^:", label="faceting: exact polygon vs exact circle")
    reference = panel[-1] * (h / h[-1]) ** 2.0
    axes[0].loglog(h, reference, "k-", lw=1.0, alpha=0.6, label="$h^2$ slope")
    axes[0].set_xlabel("mean panel length $h/a$")
    axes[0].set_ylabel("relative $L_2$ velocity error")
    axes[0].set_title("wall velocity: panel error vs shape error")
    axes[0].legend(fontsize=7.5)
    axes[0].grid(True, which="both", alpha=0.3)
    ratio = facet / panel
    axes[1].semilogx(h, ratio, "d-", color="tab:red")
    axes[1].axhline(1.0, color="k", lw=1.0)
    axes[1].set_xlabel("mean panel length $h/a$")
    axes[1].set_ylabel("faceting / panel error")
    axes[1].set_title("how much of the anchor metric is geometry, not the solver")
    axes[1].grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    path = output / "convergence.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_field(artifact: dict[str, Any], output: Path, panels: int = 64) -> Path | None:
    """Analytic polygon field with the panel solution's velocity arrows."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - matplotlib is optional
        return None
    radius = artifact["parameters"]["radius"]
    speed = artifact["parameters"]["speed"]
    grid = np.linspace(-2.2, 2.2, 161) * radius
    xx, yy = np.meshgrid(grid, grid)
    inside = np.abs(xx) + 0.0 * yy
    points = np.column_stack((xx.ravel(), yy.ravel()))
    apothem = pe.apothem(panels, radius)
    limit = apothem / np.maximum(
        np.abs(np.cos(np.arctan2(points[:, 1], points[:, 0]) * panels / 2.0)), 1e-12
    )
    ok = np.hypot(points[:, 0], points[:, 1]) > limit.view() if False else None
    radial = np.hypot(points[:, 0], points[:, 1])
    angular = np.arctan2(points[:, 1], points[:, 0])
    boundary_radius = apothem / np.cos(
        np.mod(angular + np.pi / panels, 2.0 * np.pi / panels) - np.pi / panels
    )
    outside = radial > boundary_radius + 1e-3 * radius
    magnitude = np.full(points.shape[0], np.nan)
    if np.any(outside):
        velocity = pe.exact_velocity(points[outside], panels, radius, speed)
        magnitude[outside] = np.hypot(velocity[:, 0], velocity[:, 1]) / speed
    figure, axes = plt.subplots(figsize=(6.4, 5.6))
    image = axes.pcolormesh(
        xx, yy, magnitude.reshape(xx.shape), shading="auto", cmap="viridis"
    )
    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    boundary = apothem / np.cos(np.mod(theta + np.pi / panels, 2.0 * np.pi / panels) - np.pi / panels)
    axes.fill(
        boundary * np.cos(theta),
        boundary * np.sin(theta),
        color="0.85",
        edgecolor="k",
        linewidth=1.0,
        zorder=3,
    )
    probes = anchor.probe_points()
    exact_probe = pe.exact_velocity(probes, panels, radius, speed) / speed
    axes.quiver(
        probes[:, 0], probes[:, 1], exact_probe[:, 0], exact_probe[:, 1],
        color="w", scale=18, width=0.004, zorder=4,
    )
    figure.colorbar(image, ax=axes, label="$|v|/U$ (exact polygon flow)")
    axes.set_aspect("equal")
    axes.set_title(f"exact potential flow past a regular {panels}-gon")
    axes.set_xlabel("$x/a$")
    axes.set_ylabel("$y/a$")
    figure.tight_layout()
    path = output / "field.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def render_markdown(artifact: dict[str, Any]) -> str:
    """Report body generated from the artifact (no hand-typed numbers)."""
    lines: list[str] = []
    parameters = artifact["parameters"]
    lines.append("# 圆柱/正多边形绕流：把算例误差拆成「面板离散」与「几何多面体」两部分")
    lines.append("")
    lines.append(f"- 研究编号：`{artifact['study_id']}`，锚点算例 `{artifact['anchor_case']}`")
    lines.append(
        f"- 参数：$a={parameters['radius']}$，$U={parameters['speed']}$，"
        f"面板梯 ${parameters['panel_ladder']}$，预算 $N={parameters['budget_panels']}$，"
        f"容差 {parameters['tolerance']}"
    )
    lines.append("")
    lines.append("## 1. 方法")
    lines.append(
        "对半径 $a$ 的正 $n$ 边形，单位圆外域 Schwarz-Christoffel 映射 "
        "$z(w)=c\\int^w(1-t^{-n})^{2/n}dt$，$c=a\\sin(\\pi/n)/(\\alpha 2^{\\alpha}I(\\alpha))$，"
        "$\\alpha=2/n$，复势 $W=Uc(w+1/w)$，复速度 "
        "$dW/dz=U(1-w^{-2})/(1-w^{-n})^{2/n}$；边中点处 $w^{-n}=-1$，故壁面速度有闭式 "
        "$|v|=2U|\\sin\\theta_w|/2^{2/n}$，无需数值积分。"
    )
    lines.append("")
    lines.append("对边界积分方程沿直面元求切向导数，联立求解各面元中点的壁面势斜率。法向速度使用原 Neumann 数据。外场扰动势对照解析总势减去均匀来流势 Ux。")
    lines.append("")
    lines.append("## 2. 结果")
    header = (
        "| $n$ | $h_{mean}/a$ | 面板误差(壁面速度) | 面板误差(Cp) | 面板误差(外场速度) "
        "| 锚点指标(BEM vs 圆) | 多面体几何误差 | 几何误差/锚点指标 |"
    )
    lines.append(header)
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for level in artifact["levels"]:
        lines.append(
            "| {n} | {h:.4e} | {p:.3e} | {cp:.3e} | {v:.3e} | {c:.3e} | {g:.3e} | {r:.1f} |".format(
                n=level["panels"],
                h=level["h_over_radius"],
                p=level["neumann"]["surface_velocity_relative_l2"],
                cp=level["neumann"]["cp_relative_l2"],
                v=level["neumann"]["velocity_relative_l2"],
                c=level["case_route_neumann"]["surface_velocity_relative_l2"],
                g=level["faceting"]["surface_velocity_relative_l2"],
                r=level["decomposition"]["faceting_share_of_case_surface_error"],
            )
        )
    lines.append("")
    lines.append("## 3. 判据")
    lines.append("| 判据 | 值 | 阈值 | 结果 |")
    lines.append("| --- | --- | --- | --- |")
    for row in artifact["criteria"]:
        value = row["value"]
        if isinstance(value, float):
            shown = f"{value:.4e}"
        else:
            shown = str(value)
        lines.append(
            f"| {row['name']} | {shown} | {row['threshold']} | {'PASS' if row['ok'] else 'FAIL'} |"
        )
    lines.append("")
    lines.append(f"总判定：**{artifact['overall']}**")
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--panels", type=int, default=None, help="single panel count")
    args = parser.parse_args(argv)
    ladder: Sequence[int] = (args.panels,) if args.panels else LADDER
    artifact = run_study(ladder)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / "results.json"
        path.write_text(json.dumps(artifact, indent=2) + "\n")
        figures = []
        for maker in (plot_convergence, plot_field):
            produced = maker(artifact, args.output)
            if produced is not None:
                figures.append(produced.name)
        artifact["figures"] = figures
        (args.output / "report.md").write_text(render_markdown(artifact))
        print(f"wrote {path} figures={figures}")
    print(artifact["overall"])
    for row in artifact["criteria"]:
        print(("PASS " if row["ok"] else "FAIL ") + row["name"], row["value"])
    return 0 if artifact["overall"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())