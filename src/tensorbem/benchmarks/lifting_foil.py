"""Benchmark: analytic Joukowski lifting reference vs the exterior-Neumann solver.

Outputs a JSON summary and two figures under ``docs/m2480/outputs/lifting_foil``:

* ``benchmark.json`` -- every number quoted in the report, computed here;
* ``nonlifting_agreement.png`` -- boundary-element disturbance field and surface
  speed against the analytic reference, plus the panel-refinement trend;
* ``lifting_reference.png`` -- analytic lift slope, circulation and the surface
  pressure distribution of the Kutta solution.

Run as ``python -m tensorbem.benchmarks.lifting_foil``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tensorbem.lifting import (
    JoukowskiAirfoil,
    compare_nonlifting_field,
    panel_tangents,
    stream_tangential_velocity,
    surface_arc_coordinates,
)

OUTPUT = Path("docs/m2480/outputs/lifting_foil")
PANEL_COUNTS = (32, 64, 128, 256)
ANGLES = (0.0, 2.0, 4.0, 6.0)


def analytic_reference(foil: JoukowskiAirfoil, speed: float = 1.0) -> dict:
    """Closed-form lift of the section plus the pressure integral that confirms it."""
    mesh = foil.mesh(256)
    cp = foil.exact_pressure_coefficient(mesh.centers, speed)
    lift_from_pressure = -float(np.sum(cp * mesh.normals[:, 1] * mesh.lengths))
    circulation = foil.exact_circulation(speed)
    return {
        "alpha_deg": foil.alpha_deg,
        "circulation": circulation,
        "lift_coefficient_from_circulation": 2.0 * circulation / (speed * foil.chord),
        # L' = -\oint p n_y ds with p = 0.5 rho U^2 C_p and rho = 1, so the
        # pressure coefficient integral carries a factor 0.5 that the
        # circulation form does not.
        "lift_coefficient_from_pressure": 0.5
        * lift_from_pressure
        / (0.5 * speed**2 * foil.chord),
        "thin_aerofoil_lift_coefficient": foil.exact_thin_aerofoil_lift_coefficient(),
    }


def main() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    foil = JoukowskiAirfoil(thickness=0.1, transform_radius=1.0, alpha_deg=4.0)

    lift = [analytic_reference(JoukowskiAirfoil(0.1, 1.0, alpha)) for alpha in ANGLES]
    for row in lift:
        reference = abs(row["lift_coefficient_from_circulation"])
        row["pressure_vs_circulation_gap"] = (
            0.0
            if reference == 0.0
            else abs(
                row["lift_coefficient_from_pressure"]
                - row["lift_coefficient_from_circulation"]
            )
            / reference
        )

    refinement = []
    for count in PANEL_COUNTS:
        comparison = compare_nonlifting_field(foil, count)
        mesh = foil.mesh(count)
        refinement.append(
            {
                "panels": count,
                "panel_length_min": float(mesh.lengths.min()),
                "panel_length_max": float(mesh.lengths.max()),
                "relative_potential_error": comparison.relative_potential_error,
                "potential_rms_error": comparison.potential_rms_error,
                "relative_speed_error": comparison.relative_speed_error,
                "speed_max_abs_error": comparison.speed_max_abs_error,
                "condition_number": comparison.condition_number,
                "relative_residual_l2": comparison.relative_residual_l2,
            }
        )

    summary = {
        "geometry": {
            "thickness": foil.thickness,
            "transform_radius": foil.transform_radius,
            "chord": foil.chord,
            "thickness_ratio": foil.thickness_ratio(),
            "trailing_edge": list(foil.trailing_edge),
            "leading_edge": list(foil.leading_edge),
            "alpha_deg": foil.alpha_deg,
        },
        "analytic_lift": lift,
        "nonlifting_refinement": refinement,
        "decay_offset": foil.decay_offset(),
        "verdict": {
            "analytic_pressure_vs_circulation_max_gap": max(
                row["pressure_vs_circulation_gap"] for row in lift
            ),
            "analytic_pressure_vs_circulation_gap_at_4deg": lift[2][
                "pressure_vs_circulation_gap"
            ],
            "best_relative_potential_error": min(
                row["relative_potential_error"] for row in refinement
            ),
            "lifting_bem_shipped": False,
        },
    }
    (OUTPUT / "benchmark.json").write_text(json.dumps(summary, indent=2) + "\n")

    _plot_nonlifting(foil)
    _plot_lifting_reference(foil)
    return summary


def _plot_nonlifting(foil: JoukowskiAirfoil) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    counts = (32, 64, 128, 256)
    errors = []
    for count in counts:
        errors.append(compare_nonlifting_field(foil, count).relative_potential_error)

    comparison = compare_nonlifting_field(foil, 128)
    mesh = foil.mesh(128)
    arc = surface_arc_coordinates(mesh)
    axes[0].plot(arc, comparison.reference_potential, "k-", lw=1.6, label="analytic $\\phi_d$")
    axes[0].plot(
        arc,
        comparison.result.boundary_values,
        "o",
        ms=2.6,
        color="tab:red",
        label="BEM, 128 panels",
    )
    axes[0].set_xlabel("arc length along the contour, from the trailing edge")
    axes[0].set_ylabel("disturbance potential $\\phi_d$  (U=1)")
    axes[0].set_title("Non-lifting field: BEM vs analytic reference")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].loglog(counts, errors, "s-", color="tab:blue", label="max |$\\Delta\\phi$| / max|$\\phi_d$|")
    axes[1].loglog(
        counts,
        [2.0 * errors[0] * counts[0] / c for c in counts],
        "k--",
        lw=1.0,
        label="$O(h)$ reference",
    )
    axes[1].set_xlabel("panels")
    axes[1].set_ylabel("relative potential error")
    axes[1].set_title("Refinement of the non-lifting agreement")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")
    figure.tight_layout()
    figure.savefig(OUTPUT / "nonlifting_agreement.png", dpi=140)
    plt.close(figure)


def _plot_lifting_reference(foil: JoukowskiAirfoil) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    alphas = np.linspace(0.0, 8.0, 17)
    exact = [
        JoukowskiAirfoil(foil.thickness, foil.transform_radius, a).exact_lift_coefficient()
        for a in alphas
    ]
    thin = [
        JoukowskiAirfoil(
            foil.thickness, foil.transform_radius, a
        ).exact_thin_aerofoil_lift_coefficient()
        for a in alphas
    ]
    axes[0].plot(alphas, exact, "k-", lw=1.8, label="$2\\Gamma/(Uc)$, $\\Gamma=4\\pi aU\\sin\\alpha$")
    axes[0].plot(alphas, thin, "b--", lw=1.2, label="$2\\pi(a/b)\\sin\\alpha$ (thin aerofoil)")
    axes[0].set_xlabel("angle of attack $\\alpha$ [deg]")
    axes[0].set_ylabel("$C_L$")
    axes[0].set_title("Analytic lift of the Joukowski section")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    mesh = foil.mesh(256)
    cp = foil.exact_pressure_coefficient(mesh.centers)
    arc = surface_arc_coordinates(mesh)
    half = len(arc) // 2
    axes[1].plot(arc[: half + 1] / foil.chord, cp[: half + 1], "tab:red", lw=1.5, label="upper surface")
    axes[1].plot(arc[half:] / foil.chord, cp[half:], "tab:blue", lw=1.5, label="lower surface")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("arc length / chord, from the trailing edge")
    axes[1].set_ylabel("$C_p$")
    axes[1].set_title("Exact pressure distribution, $\\alpha=4^\\circ$")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(OUTPUT / "lifting_reference.png", dpi=140)
    plt.close(figure)


if __name__ == "__main__":
    result = main()
    print(json.dumps(result["verdict"], indent=2))
    print(json.dumps(result["nonlifting_refinement"], indent=2))
    print(json.dumps(result["analytic_lift"], indent=2))
