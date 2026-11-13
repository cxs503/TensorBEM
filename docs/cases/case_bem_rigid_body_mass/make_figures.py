"""Figures for CASE-BEM-RIGID-BODY-MASS (run.json is the source of truth).

Three plots, all built only from the numbers in ``run.json`` so nothing is
hand-transcribed:

1. ``fig_convergence.png``  -- relative error of volume, CG inertia and the
   worst 6x6 entry vs panel count, with the 3 % criterion as a horizontal line.
2. ``fig_block_matrix.png`` -- the reference block matrix, the mesh-measured one
   at 1280 panels and their relative deviation (log scale).
3. ``fig_energy_identity.png`` -- the kinetic-energy identity residual for the
   correct assembly, the platform assembly (``hydrostatic.compute_mass_matrix``)
   and the two controlled perturbations, on a log axis: this is what makes the
   criterion sensitive to a sign slip that symmetry cannot see.

Usage::

    PYTHONPATH=src python3 docs/cases/case_bem_rigid_body_mass/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE / "run.json"
MODES = ["surge", "sway", "heave", "roll", "pitch", "yaw"]


def _load() -> dict:
    return json.loads(RUN.read_text())


def fig_convergence(report: dict) -> Path:
    levels = report["levels"]
    panels = np.array([row["panels"] for row in levels], dtype=float)
    figure, axes = plt.subplots(figsize=(7.0, 4.6))
    series = {
        "displaced volume": [row["volume_rel_error"] for row in levels],
        "CG inertia (worst principal)": [row["inertia_rel_error"] for row in levels],
        "6x6 matrix (worst entry)": [row["max_entry_rel_error"] for row in levels],
    }
    markers = {"displaced volume": "o", "CG inertia (worst principal)": "s", "6x6 matrix (worst entry)": "^"}
    for label, values in series.items():
        axes.loglog(panels, values, marker=markers[label], label=label)
    tolerance = report["parameters"]["tolerance"]
    axes.axhline(tolerance, color="crimson", linestyle="--", label=f"criterion {tolerance:.0%}")
    accepted = report["parameters"]["accepted_from_subdivision"]
    for row in levels:
        if row["subdivisions"] == accepted:
            axes.axvline(row["panels"], color="grey", linestyle=":", label=f"accepted from {row['panels']} panels")
    axes.set_xlabel("panels (icosphere subdivisions 0-4)")
    axes.set_ylabel("relative error vs closed form")
    axes.set_title("Rigid-body mass matrix: panel convergence (ellipsoid 120x24x16 m)")
    axes.grid(True, which="both", alpha=0.3)
    axes.legend(fontsize=8)
    figure.tight_layout()
    out = HERE / "fig_convergence.png"
    figure.savefig(out, dpi=150)
    plt.close(figure)
    return out


def fig_block_matrix(report: dict) -> Path:
    reference = np.asarray(report["reference"]["matrix"], dtype=float)
    level = max(report["levels"], key=lambda row: row["panels"])
    measured = np.asarray(level["matrix"], dtype=float)
    deviation = np.abs(measured - reference) / np.maximum(np.abs(reference), 1e-300)
    scale = np.max(np.abs(reference))

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    image = axes[0].imshow(reference / scale, cmap="viridis")
    axes[0].set_title("closed form M_RB / max|M|")
    figure.colorbar(image, ax=axes[0], fraction=0.046)
    image = axes[1].imshow(measured / scale, cmap="viridis")
    axes[1].set_title(f"mesh route, {level['panels']} panels / max|M|")
    figure.colorbar(image, ax=axes[1], fraction=0.046)
    image = axes[2].imshow(np.maximum(deviation, 1e-16), cmap="magma")
    axes[2].set_title("relative deviation (log colour scale)")
    figure.colorbar(image, ax=axes[2], fraction=0.046)
    for axis in axes:
        axis.set_xticks(range(6), MODES, rotation=45, fontsize=8)
        axis.set_yticks(range(6), MODES, fontsize=8)
    figure.suptitle(
        "CASE-BEM-RIGID-BODY-MASS: 6x6 matrix, r_g = (5, 0, -3) m, rho = 1025 kg/m^3"
    )
    figure.tight_layout()
    out = HERE / "fig_block_matrix.png"
    figure.savefig(out, dpi=150)
    plt.close(figure)
    return out


def fig_energy_identity(report: dict) -> Path:
    control = report["negative_control"]
    labels = [
        "closed-form assembly",
        "platform assembly\n(hydrostatic.compute_mass_matrix)",
        "control: sign flip\n(coupling blocks negated)",
        "control: pre-m17324\nhard-coded formula",
    ]
    values = [
        report["assembly"]["energy_identity_error"],
        report["platform_assembly"]["energy_identity_error"],
        control["sign_flip"]["energy_identity_error"],
        control["legacy_assembly"]["energy_identity_error"],
    ]
    figure, axes = plt.subplots(figsize=(8.0, 4.4))
    colours = ["#2b7bba", "#2b7bba", "#c0392b", "#c0392b"]
    bars = axes.bar(range(len(labels)), values, color=colours)
    axes.set_yscale("log")
    axes.axhline(1e-12, color="green", linestyle="--", label="round-off reference 1e-12")
    axes.axhline(0.1, color="crimson", linestyle=":", label="detection threshold 0.1")
    for bar, value in zip(bars, values):
        axes.text(bar.get_x() + bar.get_width() / 2, value * 1.4, f"{value:.2e}", ha="center", fontsize=8)
    axes.set_xticks(range(len(labels)), labels, fontsize=8)
    axes.set_ylabel("max relative kinetic-energy error")
    axes.set_title("Kinetic-energy identity: the arbiter that catches the sign slip")
    axes.legend(fontsize=8)
    axes.set_ylim(min(values) / 50.0, 10.0)
    figure.tight_layout()
    out = HERE / "fig_energy_identity.png"
    figure.savefig(out, dpi=150)
    plt.close(figure)
    return out


def main() -> None:
    report = _load()
    for path in (
        fig_convergence(report),
        fig_block_matrix(report),
        fig_energy_identity(report),
    ):
        print(path)


if __name__ == "__main__":
    main()