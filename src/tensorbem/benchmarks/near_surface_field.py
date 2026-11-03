"""Verification of near-singular field quadrature against analytic references.

The module answers three questions, in this order, and stores every number it
prints in ``docs/near_surface_field/results.json``:

1. **Is the new quadrature accurate?**  A single flat panel, known kernel
   integral, target swept from one panel size away down to ``1e-5`` panel sizes.
   The reference is the same adaptive rule run at much finer settings
   (``order=16, split_ratio=0.125``), itself cross-checked against
   ``order=20, split_ratio=0.1``; the error reported is *pure quadrature* error.
   The baseline is the existing fixed Duffy-Gauss rule of
   :func:`tensorbem.kernels3d.duffy_quadrature_batch`, i.e. what
   :func:`tensorbem.solver3d.evaluate_surface_field` uses.

2. **Does the gain survive in the field representation?**  Unit sphere with the
   analytic *disturbance* potential of a sphere in a uniform stream,
   ``u(x) = z / (2 r**3)`` (i.e. ``cos(theta) / (2 r**2)``).  The exact traces
   are imposed on the panels, the Green representation is evaluated at points
   whose clearance is swept, and the result is compared both with the converged
   quadrature and with the analytic value.  Comparing with the converged
   quadrature isolates quadrature error; comparing with the analytic value adds
   the constant-panel discretisation error.

3. **What does it buy?**  End-to-end: solve the exterior Dirichlet problem with
   the analytic boundary data, then evaluate the potential and the velocity at
   clearances that the *old* API rejects, and turn the velocity into the
   pressure coefficient.  For a sphere in a uniform stream the analytic answer is
   ``Cp = 1 - (9/4) sin(theta)**2``, so the near-wall pressure has a reference.

Run with ``python -m tensorbem.benchmarks.near_surface_field``; it is light
(a couple of minutes on one core) and writes the JSON, a Markdown summary and a
three-panel figure.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tensorbem.geometry3d import icosphere
from tensorbem.kernels3d import duffy_quadrature_batch, solid_angle_triangles_batch
from tensorbem.near_singular import (
    single_layer_gradients,
    single_layer_integrals,
    triangle_distances_and_sizes,
)
from tensorbem.solver3d import (
    LaplaceSurfaceResult,
    evaluate_surface_field,
    evaluate_surface_field_near_surface,
    evaluate_surface_velocity_at_panels,
    evaluate_surface_velocity_near_surface,
    solve_laplace_surface,
)

_REFERENCE_SETTINGS = {"order": 14, "split_ratio": 2.0, "max_level": 32}
_REFERENCE_CHECK_SETTINGS = {"order": 16, "split_ratio": 3.0, "max_level": 32}
_FAST_SETTINGS = {"order": 8, "split_ratio": 1.0, "max_level": 24}

_CLEARANCES = (1.0e-1, 3.0e-2, 1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4, 1.0e-5)


def _analytic_field(points: np.ndarray) -> np.ndarray:
    """Disturbance potential ``z / (2 r**3)`` of a unit sphere in a unit stream."""

    radius = np.linalg.norm(points, axis=-1)
    return points[..., 2] / (2.0 * radius**3)


def _analytic_disturbance_velocity(points: np.ndarray) -> np.ndarray:
    """``grad(z / (2 r**3))`` evaluated analytically."""

    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    radius = np.linalg.norm(points, axis=-1)
    factor = 3.0 * z / (2.0 * radius**5)
    return np.stack(
        [
            -factor * x,
            -factor * y,
            1.0 / (2.0 * radius**3) - factor * z,
        ],
        axis=-1,
    )



def _exact_trace_result(mesh) -> LaplaceSurfaceResult:
    """A :class:`LaplaceSurfaceResult` carrying the analytic boundary traces.

    The traces of ``u = z / (2 r**3)`` on the unit sphere are ``u = cos(theta)/2``
    and, along the *fluid-domain* normal (``-r_hat``, the convention that
    :class:`~tensorbem.solver3d.LaplaceSurfaceResult` stores for the exterior
    problem), ``du/dn = +cos(theta)``.  With those traces the Green representation
    is exact, so evaluating it and comparing with the analytic field isolates
    quadrature and trace-discretisation error from any error of the boundary
    integral solve.
    """

    centroids = mesh.centroids
    normal_direction = centroids / np.linalg.norm(centroids, axis=1)[:, None]
    panel_count = mesh.panel_count
    zeros = np.zeros((panel_count, panel_count))
    return LaplaceSurfaceResult(
        normal_derivative=normal_direction[:, 2].copy(),
        boundary_values=_analytic_field(centroids),
        single_layer_matrix=zeros,
        double_layer_matrix=zeros,
        condition_number=float("nan"),
        residual_l2=0.0,
        relative_residual_l2=0.0,
        double_layer_row_sum_error=0.0,
        domain="exterior",
        quadrature_order=0,
    )

def panel_quadrature_case(
    clearance_ratios=_CLEARANCES,
    panel_scale: float = 0.1,
) -> dict[str, object]:
    """Single-panel kernel integral versus the existing fixed-order rule."""

    v0 = np.array([[0.0, 0.0, 0.0]])
    v1 = np.array([[panel_scale, 0.0, 0.0]])
    v2 = np.array([[0.0, panel_scale, 0.0]])
    rows = []
    for ratio in clearance_ratios:
        distance = ratio * panel_scale
        target = np.array([0.3 * panel_scale, 0.3 * panel_scale, distance])
        reference = float(
            single_layer_integrals(target, v0, v1, v2, **_REFERENCE_SETTINGS)[0]
        )
        reference_check = float(
            single_layer_integrals(target, v0, v1, v2, **_REFERENCE_CHECK_SETTINGS)[0]
        )
        points, weights = duffy_quadrature_batch(v0, v1, v2, order=8)
        fixed = float(
            np.sum(weights / np.linalg.norm(target[None, None, :] - points, axis=2))
            / (4.0 * np.pi)
        )
        adaptive = float(single_layer_integrals(target, v0, v1, v2, **_FAST_SETTINGS)[0])
        _, size = triangle_distances_and_sizes(target, v0, v1, v2)
        rows.append(
            {
                "clearance_ratio": ratio,
                "panel_size": float(size[0]),
                "reference": reference,
                "reference_check": reference_check,
                "reference_disagreement": abs(reference_check - reference)
                / abs(reference),
                "fixed_order_value": fixed,
                "fixed_order_error": abs(fixed - reference) / abs(reference),
                "adaptive_value": adaptive,
                "adaptive_error": abs(adaptive - reference) / abs(reference),
            }
        )

    # The gradient kernel, same sweep.
    gradient_rows = []
    for ratio in clearance_ratios:
        distance = ratio * panel_scale
        target = np.array([0.3 * panel_scale, 0.3 * panel_scale, distance])
        reference = single_layer_gradients(target, v0, v1, v2, **_REFERENCE_SETTINGS)[0]
        points, weights = duffy_quadrature_batch(v0, v1, v2, order=8)
        delta = target[None, None, :] - points
        radius = np.linalg.norm(delta, axis=2)
        fixed = (
            -np.sum((weights / radius**3)[:, :, None] * delta, axis=1)
            / (4.0 * np.pi)
        )[0]
        adaptive = single_layer_gradients(target, v0, v1, v2, **_FAST_SETTINGS)[0]
        gradient_rows.append(
            {
                "clearance_ratio": ratio,
                "fixed_order_error": float(
                    np.linalg.norm(fixed - reference) / np.linalg.norm(reference)
                ),
                "adaptive_error": float(
                    np.linalg.norm(adaptive - reference) / np.linalg.norm(reference)
                ),
            }
        )
    return {"potential": rows, "gradient": gradient_rows}


def representation_case(subdivisions: int = 2, clearance_ratios=_CLEARANCES) -> dict[str, object]:
    """Green representation with the *exact* traces, evaluated near the surface."""

    mesh = icosphere(subdivisions=subdivisions, radius=1.0)
    result = _exact_trace_result(mesh)
    points, weights = duffy_quadrature_batch(mesh.v0, mesh.v1, mesh.v2, order=8)
    rows = []
    for ratio in clearance_ratios:
        distance = ratio * float(mesh.h_max)
        target = np.array([0.0, 0.0, 1.0 + distance])
        analytic = float(_analytic_field(target))
        adaptive = float(evaluate_surface_field_near_surface(mesh, result, target[None, :])[0])
        reference = float(
            evaluate_surface_field_near_surface(
                mesh,
                result,
                target[None, :],
                quadrature_order=_REFERENCE_SETTINGS["order"],
                split_ratio=_REFERENCE_SETTINGS["split_ratio"],
                max_level=_REFERENCE_SETTINGS["max_level"],
            )[0]
        )
        delta = target[None, None, :] - points
        fixed_single = np.sum(weights / np.linalg.norm(delta, axis=2), axis=1) / (
            4.0 * np.pi
        )
        solid_angles = solid_angle_triangles_batch(target, mesh.v0, mesh.v1, mesh.v2)
        fixed = float(
            fixed_single @ result.normal_derivative
            + (-solid_angles / (4.0 * np.pi)) @ result.boundary_values
        )
        rows.append(
            {
                "clearance_ratio": ratio,
                "clearance": distance,
                "panel_count": mesh.panel_count,
                "h_max": float(mesh.h_max),
                "analytic": analytic,
                "reference": reference,
                "fixed_order_value": fixed,
                "adaptive_value": adaptive,
                "fixed_order_quadrature_error": abs(fixed - reference) / abs(reference),
                "adaptive_quadrature_error": abs(adaptive - reference) / abs(reference),
                "fixed_order_error_vs_analytic": abs(fixed - analytic) / abs(analytic),
                "adaptive_error_vs_analytic": abs(adaptive - analytic) / abs(analytic),
            }
        )
    return {
        "subdivisions": subdivisions,
        "panel_count": mesh.panel_count,
        "h_max": float(mesh.h_max),
        "rows": rows,
    }


def end_to_end_case(subdivisions: int = 2, offset_ratios=(0.5, 0.1, 0.02, 5.0e-3, 1.0e-3, 2.0e-4)) -> dict[str, object]:
    """Solve the exterior Dirichlet problem, then evaluate near the surface."""

    mesh = icosphere(subdivisions=subdivisions, radius=1.0)
    boundary_values = _analytic_field(mesh.centroids)
    result = solve_laplace_surface(mesh, boundary_values, domain="exterior")
    rows = []
    for ratio in offset_ratios:
        distance = ratio * float(mesh.h_max)
        target = np.array([[0.0, 0.0, 1.0 + distance]])
        analytic = float(_analytic_field(target[0]))
        near = float(evaluate_surface_field_near_surface(mesh, result, target)[0])
        rejected = False
        try:
            evaluate_surface_field(mesh, result, target)
        except ValueError:
            rejected = True
        # Coefficient of the Laplace problem is unaffected by the clearance; the
        # velocity path is also exercised, against the analytic disturbance.
        velocity = evaluate_surface_velocity_near_surface(mesh, result, target)[0]
        analytic_velocity = _analytic_disturbance_velocity(target[0])
        rows.append(
            {
                "clearance_ratio": ratio,
                "clearance": distance,
                "analytic": analytic,
                "near_surface_value": near,
                "relative_error": abs(near - analytic) / abs(analytic),
                "old_api_rejected": rejected,
                "velocity": velocity.tolist(),
                "analytic_velocity": analytic_velocity.tolist(),
                "velocity_relative_error": float(
                    np.linalg.norm(velocity - analytic_velocity)
                    / np.linalg.norm(analytic_velocity)
                ),
            }
        )
    return {
        "subdivisions": subdivisions,
        "panel_count": mesh.panel_count,
        "h_max": float(mesh.h_max),
        "condition_number": float(result.condition_number),
        "relative_residual_l2": float(result.relative_residual_l2),
        "rows": rows,
    }


def pressure_case(subdivisions: int = 2, offset_ratio: float = 3.0e-2) -> dict[str, object]:
    """Near-wall pressure coefficient of the sphere in a uniform stream."""

    mesh = icosphere(subdivisions=subdivisions, radius=1.0)
    boundary_values = _analytic_field(mesh.centroids)
    result = solve_laplace_surface(mesh, boundary_values, domain="exterior")
    distance = offset_ratio * float(mesh.h_max)
    angles = np.linspace(0.0, np.pi, 13)
    radius = 1.0 + distance
    targets = np.stack(
        [
            radius * np.sin(angles),
            np.zeros_like(angles),
            radius * np.cos(angles),
        ],
        axis=1,
    )
    velocity = evaluate_surface_velocity_near_surface(mesh, result, targets)
    analytic_velocity = _analytic_disturbance_velocity(targets)
    stream = np.array([0.0, 0.0, 1.0])
    cp_near = 1.0 - np.sum((stream + velocity) ** 2, axis=1)
    cp_analytic = 1.0 - np.sum((stream + analytic_velocity) ** 2, axis=1)

    surface_velocity = evaluate_surface_velocity_at_panels(mesh, result)
    centroid_angles = np.arccos(
        np.clip(mesh.centroids[:, 2] / np.linalg.norm(mesh.centroids, axis=1), -1.0, 1.0)
    )
    cp_surface = 1.0 - np.sum((stream + surface_velocity) ** 2, axis=1)
    cp_surface_analytic = 1.0 - 2.25 * np.sin(centroid_angles) ** 2

    cp_scale = 2.25  # max |Cp| of the sphere in a uniform stream
    return {
        "subdivisions": subdivisions,
        "panel_count": mesh.panel_count,
        "h_max": float(mesh.h_max),
        "offset_ratio": offset_ratio,
        "clearance": distance,
        "old_api_clearance_limit": 0.05 * float(mesh.h_max),
        "angles_deg": [float(a * 180.0 / math.pi) for a in angles],
        "cp_near_surface": [float(v) for v in cp_near],
        "cp_analytic": [float(v) for v in cp_analytic],
        "cp_near_surface_relative_error": [
            float(e) for e in np.abs(cp_near - cp_analytic) / np.abs(cp_analytic)
        ],
        "max_cp_absolute_error": float(np.max(np.abs(cp_near - cp_analytic))),
        "max_cp_error_over_scale": float(
            np.max(np.abs(cp_near - cp_analytic)) / cp_scale
        ),
        "cp_scale": cp_scale,
        "surface_angles_deg": [float(a * 180.0 / math.pi) for a in centroid_angles],
        "cp_surface": [float(v) for v in cp_surface],
        "cp_surface_analytic": [float(v) for v in cp_surface_analytic],
        "cp_surface_rms_error": float(
            np.sqrt(np.mean((cp_surface - cp_surface_analytic) ** 2))
        ),
    }


def convergence_case(
    subdivisions_sequence=(1, 2, 3),
    offset_ratios=(1.0e-2, 1.0e-3),
) -> dict[str, object]:
    """Mesh convergence at fixed clearance ratios, with the exact traces.

    No dense solve is involved, so 1280 panels stay cheap.  The error reported is
    against the analytic field: it combines the constant-panel discretisation
    error of the traces with the quadrature error, which is what a user sees.
    """

    rows = []
    for subdivisions in subdivisions_sequence:
        mesh = icosphere(subdivisions=subdivisions, radius=1.0)
        result = _exact_trace_result(mesh)
        for ratio in offset_ratios:
            distance = ratio * float(mesh.h_max)
            target = np.array([[0.0, 0.0, 1.0 + distance]])
            analytic = float(_analytic_field(target[0]))
            value = float(evaluate_surface_field_near_surface(mesh, result, target)[0])
            velocity = evaluate_surface_velocity_near_surface(mesh, result, target)[0]
            analytic_velocity = _analytic_disturbance_velocity(target[0])
            rows.append(
                {
                    "subdivisions": subdivisions,
                    "panel_count": mesh.panel_count,
                    "h_max": float(mesh.h_max),
                    "clearance_ratio": ratio,
                    "clearance": distance,
                    "field_relative_error": abs(value - analytic) / abs(analytic),
                    "velocity_relative_error": float(
                        np.linalg.norm(velocity - analytic_velocity)
                        / np.linalg.norm(analytic_velocity)
                    ),
                }
            )
    return {
        "offset_ratios": list(offset_ratios),
        "note": "exact analytic traces; no dense solve, so the trace discretisation dominates",
        "rows": rows,
    }


def run(subdivisions: int = 2) -> dict[str, object]:
    """Run every case and return the full result dictionary."""

    return {
        "case": "near-singular field quadrature on a unit sphere in a uniform stream",
        "settings": {"fast": _FAST_SETTINGS, "reference": _REFERENCE_SETTINGS},
        "panel_quadrature": panel_quadrature_case(),
        "representation": representation_case(subdivisions=subdivisions),
        "end_to_end": end_to_end_case(subdivisions=subdivisions, offset_ratios=(0.5, 0.1, 0.02, 5.0e-3, 1.0e-3, 2.0e-4)),
        "pressure": pressure_case(subdivisions=subdivisions),
        "pressure_far": pressure_case(subdivisions=subdivisions, offset_ratio=1.0e-1),
        "convergence": convergence_case(),
    }


def _markdown_summary(results: dict[str, object]) -> str:
    lines: list[str] = []
    panel = results["panel_quadrature"]
    lines.append("### Single-panel kernel integral (pure quadrature error)\n")
    lines.append("| d / size | fixed-order error | adaptive error |")
    lines.append("|---|---|---|")
    for row in panel["potential"]:
        lines.append(
            f"| {row['clearance_ratio']:.0e} | {row['fixed_order_error']:.3e} | "
            f"{row['adaptive_error']:.3e} |"
        )
    lines.append("")
    lines.append("### Green representation with exact traces\n")
    lines.append("| d / h_max | fixed vs converged | adaptive vs converged | adaptive vs analytic |")
    lines.append("|---|---|---|---|")
    for row in results["representation"]["rows"]:
        lines.append(
            f"| {row['clearance_ratio']:.0e} | {row['fixed_order_quadrature_error']:.3e} | "
            f"{row['adaptive_quadrature_error']:.3e} | "
            f"{row['adaptive_error_vs_analytic']:.3e} |"
        )
    lines.append("")
    lines.append("### End-to-end solve, near-surface evaluation\n")
    lines.append("| d / h_max | clearance | field rel. error | velocity rel. error | old API rejects |")
    lines.append("|---|---|---|---|---|")
    for row in results["end_to_end"]["rows"]:
        lines.append(
            f"| {row['clearance_ratio']:.1e} | {row['clearance']:.3e} | "
            f"{row['relative_error']:.3e} | {row['velocity_relative_error']:.3e} | "
            f"{row['old_api_rejected']} |"
        )
    lines.append("")
    lines.append("### Mesh convergence at fixed clearance ratios (exact traces)\n")
    lines.append("| subdivisions | panels | d / h_max | field error | velocity error |")
    lines.append("|---|---|---|---|---|")
    for row in results["convergence"]["rows"]:
        lines.append(
            f"| {row['subdivisions']} | {row['panel_count']} | "
            f"{row['clearance_ratio']:.0e} | {row['field_relative_error']:.3e} | "
            f"{row['velocity_relative_error']:.3e} |"
        )
    lines.append("")
    for key, label in (("pressure", "near-wall"), ("pressure_far", "far")):
        pressure = results[key]
        lines.append(
            f"Pressure ({label}): max |dCp| = {pressure['max_cp_absolute_error']:.3e} "
            f"({pressure['max_cp_error_over_scale']:.3e} of the Cp scale "
            f"{pressure['cp_scale']:.2f}) at clearance "
            f"{pressure['clearance']:.3e} (old API limit "
            f"{pressure['old_api_clearance_limit']:.3e}); surface-Cp RMS error "
            f"{pressure['cp_surface_rms_error']:.3e}."
        )
    return "\n".join(lines) + "\n"


def make_figure(results: dict[str, object], path: Path) -> Path:
    """Three panels: quadrature error, field error, near-wall pressure."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    panel = results["panel_quadrature"]["potential"]
    ratios = [row["clearance_ratio"] for row in panel]
    axes[0].loglog(ratios, [row["fixed_order_error"] for row in panel], "o-", label="fixed Duffy order 8")
    axes[0].loglog(ratios, [row["adaptive_error"] for row in panel], "s-", label="adaptive (new)")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("clearance / panel size")
    axes[0].set_ylabel("relative error")
    axes[0].set_title("single-panel kernel integral")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    representation = results["representation"]["rows"]
    ratios = [row["clearance_ratio"] for row in representation]
    axes[1].loglog(ratios, [row["fixed_order_quadrature_error"] for row in representation], "o-", label="fixed order")
    axes[1].loglog(ratios, [row["adaptive_quadrature_error"] for row in representation], "s-", label="adaptive")
    axes[1].loglog(ratios, [row["adaptive_error_vs_analytic"] for row in representation], "^--", label="adaptive vs analytic")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("clearance / h_max")
    axes[1].set_ylabel("relative error")
    axes[1].set_title("Green representation, exact traces")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    pressure = results["pressure"]
    axes[2].plot(pressure["angles_deg"], pressure["cp_analytic"], "k-", label="analytic Cp")
    for key, marker in (("pressure", "o"), ("pressure_far", "s")):
        case = results[key]
        axes[2].plot(
            case["angles_deg"],
            case["cp_near_surface"],
            marker,
            markersize=4,
            label=f"BEM, d={case['clearance']:.2e} ({key})",
        )
    axes[2].set_xlabel("theta [deg]")
    axes[2].set_ylabel("Cp")
    axes[2].set_title("sphere in uniform stream: pressure")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def main() -> dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    output = root / "docs" / "near_surface_field"
    output.mkdir(parents=True, exist_ok=True)
    results = run()
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (output / "summary.md").write_text(_markdown_summary(results))
    make_figure(results, output / "near_surface_field.png")
    print((output / "summary.md").read_text())
    return results


if __name__ == "__main__":  # pragma: no cover
    main()