"""Figures for the annular (multiply connected) Laplace verification study.

Reads ``docs/annulus_laplace/study.json`` -- the artifact written by
``python -m tensorbem.benchmarks.annulus_laplace`` -- and writes

* ``convergence.png``: ``q`` error and interior-field RMSE against panel count,
  with the ``O(1/N)``/``O(1/N^2)`` reference slopes;
* ``field.png``: analytic potential on the annulus with the two boundary loops
  and the interior probe points used by the study.

No number is typed here; both panels are drawn from the artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..multiloop import annulus_mesh, evaluate_multiloop_field, solve_laplace_multiloop  # noqa: E402
from . import annulus_laplace as study_mod  # noqa: E402

DEFAULT_STUDY = Path("docs/annulus_laplace/study.json")
DEFAULT_OUTDIR = Path("docs/annulus_laplace")


def plot_convergence(study: dict, outdir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    colors = {"radial_log": "C0", "angular_cos": "C3"}
    for case in study["cases"]:
        counts = np.array([level["panel_count"] for level in case["levels"]], dtype=float)
        q_error = np.array([level["q_relative_l2_error"] for level in case["levels"]])
        rmse = np.array([level["field_rmse"] for level in case["levels"]])
        color = colors.get(case["name"], None)
        axes[0].loglog(counts, q_error, "o-", color=color, label=case["declared_solution"])
        axes[1].loglog(counts, rmse, "o-", color=color, label=case["declared_solution"])
    counts = np.array(
        [level["panel_count"] for level in study["cases"][0]["levels"]], dtype=float
    )
    anchor = study["cases"][0]["levels"][-1]["q_relative_l2_error"]
    axes[0].loglog(counts, anchor * (counts[-1] / counts) ** 2, "k--", lw=1.0, label=r"$O(N^{-2})$")
    anchor_r = study["cases"][0]["levels"][-1]["field_rmse"]
    axes[1].loglog(counts, anchor_r * (counts[-1] / counts) ** 2, "k--", lw=1.0, label=r"$O(N^{-2})$")
    for axis, title, ylabel in (
        (axes[0], "normal derivative error", r"$\|q_h-q\|_2/\|q\|_2$"),
        (axes[1], "interior field error", "RMSE of $u_h-u$ at probe points"),
    ):
        axis.set_xlabel("panels $N$ (outer + inner)")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)
    fig.suptitle("2-D Laplace BEM on an annulus: analytic convergence", fontsize=11)
    fig.tight_layout()
    path = outdir / "convergence.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_field(study: dict, outdir: Path) -> Path:
    params = study["parameters"]
    outer = params["outer_radius"]
    inner = params["inner_radius"]
    grid = np.linspace(-1.15 * outer, 1.15 * outer, 121)
    xx, yy = np.meshgrid(grid, grid)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    radius = np.linalg.norm(points, axis=1)

    mesh = annulus_mesh(64, 32, outer, inner)
    data = study_mod.radial_log_potential(mesh.centers)
    result = solve_laplace_multiloop(mesh, data)
    keep = (radius > inner) & (radius < outer)
    numeric = np.full(points.shape[0], np.nan)
    numeric[keep] = evaluate_multiloop_field(mesh, result, points[keep])
    exact = study_mod.radial_log_potential(points)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4))
    levels = np.linspace(exact[keep].min(), exact[keep].max(), 15)
    contour = axes[0].contourf(
        xx, yy, numeric.reshape(xx.shape), levels=levels, cmap="viridis"
    )
    fig.colorbar(contour, ax=axes[0], label="$u_h$")
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    for radius_loop in (outer, inner):
        axes[0].plot(radius_loop * np.cos(theta), radius_loop * np.sin(theta), "w-", lw=1.0)
    probes = study_mod.probe_points()
    axes[0].plot(probes[:, 0], probes[:, 1], "r.", ms=3, label="probe points")
    axes[0].set_title("BEM potential on the annulus (64/32 panels)")
    axes[0].set_aspect("equal")
    axes[0].legend(fontsize=8, loc="upper right")

    error = np.abs(numeric - exact)
    scale = 10.0 ** np.floor(np.log10(np.nanmax(error)) if np.nanmax(error) > 0 else 0.0)
    contour2 = axes[1].contourf(
        xx,
        yy,
        error.reshape(xx.shape),
        levels=np.linspace(0.0, max(scale, np.nanmax(error)), 15),
        cmap="magma",
    )
    fig.colorbar(contour2, ax=axes[1], label="$|u_h-u|$")
    axes[1].set_title("absolute error against the closed form")
    axes[1].set_aspect("equal")
    for axis in axes:
        axis.set_xlabel("$x$ (m)")
        axis.set_ylabel("$y$ (m)")
    fig.tight_layout()
    path = outdir / "field.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args(argv)
    study = json.loads(args.study.read_text(encoding="utf-8"))
    args.outdir.mkdir(parents=True, exist_ok=True)
    for path in (plot_convergence(study, args.outdir), plot_field(study, args.outdir)):
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())