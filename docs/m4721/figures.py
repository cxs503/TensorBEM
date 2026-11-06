"""Report figures for the two-cylinder mutual added-mass benchmark (m4721).

Produces (into ``artifacts/two_cylinder_added_mass/``):

1. ``field_and_boundary.png`` -- potential cloud plot of the multipole series
   for the ``body1_surge`` mode, plus the boundary-potential comparison
   (BEM markers versus series line on both bodies);
2. ``refinement.png`` -- mesh-refinement of the BEM coefficients against the
   series reference at ``d/a = 3``;
3. ``interaction.png`` -- normalised mutual coefficient versus separation on a
   log-log scale, with the ``2*(a/d)**2`` asymptote and the BEM points.

Every number plotted here is computed in this script from
``tensorbem.analytic`` and ``tensorbem.benchmarks.two_cylinder_added_mass``;
nothing is transcribed by hand. Outputs are written next to a JSON digest so
the report can quote them without copy-paste.

Run::

    PYTHONPATH=src python3 docs/m4721/figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorbem.analytic import two_cylinder_added_mass, two_cylinder_mode_potential
from tensorbem.benchmarks.two_cylinder_added_mass import (
    DENSITY,
    interaction_decay,
    refinement_study,
    run,
    solve_added_mass_potentials,
    two_circle_mesh,
)

RADIUS = 1.0
ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts" / "two_cylinder_added_mass"
MODES = ("body0_surge", "body0_heave", "body1_surge", "body1_heave")


def field_figure(path: Path) -> dict[str, float]:
    separation = 3.0
    half = 0.5 * separation
    grid = np.linspace(-6.0, 6.0, 401)
    x, y = np.meshgrid(grid, grid)
    points = np.column_stack((x.ravel(), y.ravel()))
    potential = two_cylinder_mode_potential(
        RADIUS, separation, "body1_surge", points
    ).reshape(x.shape)
    mask = (np.hypot(x + half, y) < RADIUS) | (np.hypot(x - half, y) < RADIUS)
    plotted = np.ma.array(potential, mask=mask)

    mesh = two_circle_mesh(RADIUS, separation, 64)
    _, _, potentials = solve_added_mass_potentials(mesh, DENSITY)
    column = MODES.index("body1_surge")
    centres = mesh.centers
    exact = two_cylinder_mode_potential(RADIUS, separation, "body1_surge", centres)

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    limit = float(np.max(np.abs(potential[~mask])))
    contour = axes[0].pcolormesh(
        x, y, plotted, cmap="RdBu_r", vmin=-limit, vmax=limit, shading="auto"
    )
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    for centre in (-half, half):
        axes[0].plot(
            centre + RADIUS * np.cos(theta), RADIUS * np.sin(theta), color="k", lw=1.2
        )
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x / a")
    axes[0].set_ylabel("y / a")
    axes[0].set_title("multipole series: phi for body 1 surge (unit velocity)")
    figure.colorbar(contour, ax=axes[0], shrink=0.85, label="phi / (U a)")

    for body, centre in enumerate((-half, half)):
        sl = mesh.loop_slices[body]
        angle = np.arctan2(centres[sl, 1], centres[sl, 0] - centre)
        order = np.argsort(angle)
        axes[1].plot(
            np.degrees(angle[order]),
            exact[sl][order],
            "-",
            color="C0" if body == 0 else "C1",
            label=f"series, body {body}",
        )
        axes[1].plot(
            np.degrees(angle[order]),
            potentials[sl, column][order],
            "o",
            ms=2.5,
            color="C0" if body == 0 else "C1",
            alpha=0.6,
            label=f"BEM, body {body}",
        )
    axes[1].set_xlabel("angle around body centre / deg")
    axes[1].set_ylabel("phi / (U a)")
    axes[1].set_title("boundary potential, body 1 surge (64 panels/body)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return {
        "boundary_potential_relative_l2_body0": float(
            np.linalg.norm(
                potentials[mesh.loop_slices[0], column] - exact[mesh.loop_slices[0]]
            )
            / np.linalg.norm(exact[mesh.loop_slices[0]])
        ),
        "boundary_potential_relative_l2_body1": float(
            np.linalg.norm(
                potentials[mesh.loop_slices[1], column] - exact[mesh.loop_slices[1]]
            )
            / np.linalg.norm(exact[mesh.loop_slices[1]])
        ),
    }


def interaction_figure(path: Path) -> dict[str, float]:
    rows = interaction_decay(separation_ratios=(2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 20.0, 40.0))
    ratios = np.array([row["separation_ratio"] for row in rows])
    mutual = np.array([row["mutual_transverse_over_displaced"] for row in rows])
    bem_points = []
    for ratio, panels in ((3.0, 48), (3.0, 96), (5.0, 64), (10.0, 64)):
        result = run(panels=panels, separation_ratio=ratio)
        bem_points.append(
            (
                ratio,
                result["bem"]["mutual_transverse"] / result["displaced_mass_per_span"],
                panels,
            )
        )
    figure, axes = plt.subplots(figsize=(6.4, 4.4))
    axes.loglog(ratios, mutual, "k-", label="multipole series")
    axes.loglog(
        ratios,
        2.0 / ratios**2,
        "k--",
        lw=1.0,
        label=r"asymptote $2\,(a/d)^2$",
    )
    for ratio, value, panels in bem_points:
        axes.loglog(ratio, value, "o", ms=7, mfc="none", label=f"BEM, {panels} panels/body")
    axes.set_xlabel("centre separation d / a")
    axes.set_ylabel(r"mutual  $A^{(12)}_{yy}/(\rho\pi a^2)$")
    axes.set_title("transverse interaction coefficient (in phase sway)")
    axes.grid(which="both", alpha=0.3)
    axes.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return {
        "asymptote_ratio_at_40": float(mutual[-1] * ratios[-1] ** 2 / 2.0),
        "bem_points": [[float(r), float(v), int(p)] for r, v, p in bem_points],
    }


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    from tensorbem.benchmarks.two_cylinder_added_mass import plot_refinement

    study = refinement_study(panels=(48, 64, 96), separation_ratio=3.0)
    plot_refinement(study, ARTIFACTS / "refinement.png")
    digest = {
        "refinement": {
            "panels_per_body": study["panels_per_body"],
            "max_relative_errors": study["max_relative_errors"],
            "monotone_decrease": study["monotone_decrease"],
            "reference": study["rows"][-1]["reference"],
            "bem_finest": study["rows"][-1]["bem"],
            "relative_errors_finest": study["rows"][-1]["relative_errors"],
            "condition_number": study["rows"][-1]["single_layer_condition_number"],
            "reference_boundary_residual": study["rows"][-1]["reference_boundary_residual"],
        },
        "field": field_figure(ARTIFACTS / "field_and_boundary.png"),
        "interaction": interaction_figure(ARTIFACTS / "interaction.png"),
        "injected_layer_sign": run(panels=48, separation_ratio=3.0, layer_sign=-1.0)[
            "relative_errors"
        ],
        "reference_matrix_sep3": two_cylinder_added_mass(RADIUS, 3.0, DENSITY).added_mass_matrix.tolist(),
    }
    (ARTIFACTS / "figures_digest.json").write_text(json.dumps(digest, indent=2))
    print(json.dumps(digest, indent=2))


if __name__ == "__main__":
    main()