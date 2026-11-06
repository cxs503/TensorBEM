"""Regenerate the m4750 curved-waterplane audit artifacts (numbers + figures).

Every number in ``dev_report_m4750_curved_waterplane_restoring.md`` is copied
from ``curved_waterplane_audit.json`` by the report generator, so there is no
hand transcription:

    PYTHONPATH=src python3 outputs/verif_m4750/make_artifacts.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from tensorbem.benchmarks import hydrostatic_curved_waterplane as curved

OUT = Path(__file__).resolve().parent
REPORT_FIG = OUT.parent.parent / "docs" / "figures"

CASE_LABEL = {
    "eccentric_sphere": "eccentric sphere\n(curved wall, circle)",
    "yawed_spheroid": "yawed spheroid\n(curved wall, ellipse)",
    "eccentric_yawed_spheroid": "eccentric yawed spheroid\n(all six entries)",
}


def build_record() -> dict:
    return curved.run(subdivisions=(2, 3, 4))


def figure_geometry(record: dict) -> Path:
    """Left: waterplane outlines. Right: per-entry deviations, both references."""
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))

    for case in curved.build_cases(3):
        vertices = case.hull.vertices
        edge = case.hull.waterline_edges
        loops = {}
        for a, b in edge:
            loops.setdefault(round(float(vertices[a, 2]), 12), []).append((a, b))
        # Plot every waterline vertex; dense enough to show the curved outline.
        mask = np.abs(vertices[:, 2]) < 1.0e-12
        order = np.argsort(np.arctan2(vertices[mask, 1] - vertices[mask, 1].mean(),
                                      vertices[mask, 0] - vertices[mask, 0].mean()))
        ring = vertices[mask][order][:, :2]
        ring = np.vstack([ring, ring[0]])
        axes[0].plot(ring[:, 0], ring[:, 1], marker=".", label=case.key)

    axes[0].axhline(0.0, color="0.7", lw=0.8)
    axes[0].axvline(0.0, color="0.7", lw=0.8)
    axes[0].plot([0.0], [0.0], "k+", ms=10, label="origin (reference point)")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].set_title("Waterplane outlines (z = 0), subdivisions = 3")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.2)

    case = next(c for c in record["cases"] if c["key"] == "eccentric_yawed_spheroid")
    symbols = [row["symbol"] for row in case["rows"]]
    positions = np.arange(len(symbols))
    axes[1].bar(positions - 0.2, [row["relerr_code_vs_energy"] for row in case["rows"]],
                width=0.4, label="vs energy Hessian (formula)")
    axes[1].bar(positions + 0.2, [row["relerr_code_vs_closed_form"] for row in case["rows"]],
                width=0.4, label="vs exact geometry (faceting)")
    axes[1].axhline(1.0e-6, color="C3", ls="--", lw=1.0, label="1e-6 contract")
    axes[1].axhline(3.0e-2, color="C2", ls=":", lw=1.0, label="3 % contract")
    axes[1].set_yscale("log")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(symbols, rotation=20)
    axes[1].set_ylabel("relative deviation")
    axes[1].set_title("Eccentric yawed spheroid: two independent references")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.2, axis="y")

    path = OUT / "fig_curved_waterplane.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def figure_convergence(record: dict) -> Path:
    """Left: FD step study (order ~1, hence the Richardson exponent). Right: faceting."""
    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))

    for key, series in record["fd_steps"]["series"].items():
        steps = [point["step"] for point in series]
        worst = [point["worst_relative"] for point in series]
        axes[0].loglog(steps, worst, marker="o", label=f"{key} (order {record['fd_steps']['fitted_orders'][key]:.2f})")
    guide = np.array([2.5e-5, 4.0e-4])
    axes[0].loglog(guide, 1.9 * guide, "k--", lw=1.0, label="slope 1 guide")
    axes[0].set_xlabel("finite-difference step (m, rad)")
    axes[0].set_ylabel("worst relative residual vs code")
    axes[0].set_title("Energy-Hessian residual is O(step) truncation")
    axes[0].grid(alpha=0.2, which="both")
    axes[0].legend(fontsize=7)

    for key, series in record["convergence"]["levels"].items():
        panels = [point["panels"] for point in series]
        worst = [point["worst_relerr_vs_closed_form"] for point in series]
        axes[1].loglog(panels, worst, marker="s", label=key)
    axes[1].axhline(3.0e-2, color="C3", ls="--", lw=1.0, label="3 % contract")
    axes[1].set_xlabel("panels")
    axes[1].set_ylabel("worst relative deviation vs exact geometry")
    axes[1].set_title("Faceting error of the curved hulls")
    axes[1].grid(alpha=0.2, which="both")
    axes[1].legend(fontsize=7)

    path = OUT / "fig_curved_convergence.png"
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def main() -> int:
    record = build_record()
    (OUT / "curved_waterplane_audit.json").write_text(json.dumps(record, indent=2))
    first = figure_geometry(record)
    second = figure_convergence(record)
    REPORT_FIG.mkdir(parents=True, exist_ok=True)
    for path in (first, second):
        (REPORT_FIG / path.name).write_bytes(path.read_bytes())
    print("wrote", first, second)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())