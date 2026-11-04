"""Reproducible study of the multiply-connected (annulus) 2-D Laplace solve.

``docs/verification.md``'s case-selection table registers the annulus case
(``u = A log r + B``, two boundary components with opposite normals) as
*"后续：需多边界组件"*. :mod:`tensorbem.multicomponent` supplies the missing
pieces (annulus mesh, orientation guard, closed-form reference, loop-flux
bookkeeping); this driver measures what the existing collocation solver does
with them, so the status of the case is a number rather than a guess.

Every number printed in ``docs/multicomponent_domain.md`` comes from here::

    python -m tensorbem.benchmarks.annulus_log_potential \
        --panels 8 16 32 64 --output docs/multicomponent/outputs/study.json

The reported quantities are, per refinement level:

* ``flux_outer_rel_error`` / ``flux_inner_rel_error`` -- the panel-constant
  flux ``sum q_j length_j`` on each loop against the exact ``+/- 2 pi A``. This
  is discretisation-free in the sense that even a *polygonal* loop carries
  ``2 pi A`` exactly (the chords telescope), so the only error source is ``q``;
* ``flux_net_over_exact`` -- ``|flux_outer + flux_inner| / (2 pi A)``, which
  must vanish for a source-free field and therefore tests that the two
  components were assembled with consistent normal senses;
* ``q_relative_l2_error`` -- the unknown trace against ``+/- A/r`` at the panel
  midpoints;
* ``interior_field_relative_l2_error`` -- the reconstructed potential against
  ``A log r + B`` on a probe set inside the annulus.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import numpy as np

from ..multicomponent import (
    annulus_mesh,
    log_potential,
    log_potential_boundary_data,
    log_potential_normal_derivative,
    loop_fluxes,
    solve_annulus_dirichlet,
)
from ..solver import evaluate_interior_field

SCHEMA = "tensorbem.annulus_log_potential.study/1"


def probe_points(inner_radius: float, outer_radius: float, count: int = 8):
    """Interior probe ring set, kept clear of both components."""
    radii = np.linspace(
        inner_radius + 0.15 * (outer_radius - inner_radius),
        outer_radius - 0.15 * (outer_radius - inner_radius),
        3,
    )
    angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    points = [
        (r * math.cos(t), r * math.sin(t)) for r in radii for t in angles
    ]
    return np.asarray(points, dtype=float)


def run_level(
    panels_per_loop: int,
    outer_radius: float = 1.0,
    inner_radius: float = 0.5,
    reference_length: float = 0.5,
) -> dict:
    """Solve one refinement level and return every judged quantity."""
    annulus = annulus_mesh(
        2 * panels_per_loop,
        panels_per_loop,
        outer_radius=outer_radius,
        inner_radius=inner_radius,
    )
    result = solve_annulus_dirichlet(annulus, reference_length=reference_length)
    exact_q = log_potential_normal_derivative(annulus)
    measured_q = result.normal_derivative

    fluxes = loop_fluxes(annulus, measured_q)
    exact_loop_flux = 2.0 * math.pi  # amplitude A = 1

    probes = probe_points(inner_radius, outer_radius)
    exact_field = log_potential(np.linalg.norm(probes, axis=1))
    measured_field = evaluate_interior_field(annulus.mesh, result, probes)

    return {
        "panels_per_loop": int(panels_per_loop),
        "panel_count": annulus.panel_count,
        "h_max": float(np.max(annulus.mesh.lengths)),
        "condition_number": float(result.condition_number),
        "relative_residual_l2": float(result.relative_residual_l2),
        "q_relative_l2_error": float(
            np.linalg.norm(measured_q - exact_q) / np.linalg.norm(exact_q)
        ),
        "q_max_abs_error": float(np.max(np.abs(measured_q - exact_q))),
        "flux_outer": fluxes["outer"],
        "flux_inner": fluxes["inner"],
        "flux_net": fluxes["net"],
        "flux_outer_rel_error": float(fluxes["outer"] / exact_loop_flux - 1.0),
        "flux_inner_rel_error": float(fluxes["inner"] / -exact_loop_flux - 1.0),
        "flux_net_over_exact": float(abs(fluxes["net"]) / exact_loop_flux),
        "interior_field_relative_l2_error": float(
            np.linalg.norm(measured_field - exact_field)
            / np.linalg.norm(exact_field)
        ),
        "interior_field_max_abs_error": float(
            np.max(np.abs(measured_field - exact_field))
        ),
    }


def wrong_winding_control(
    panels_per_loop: int,
    outer_radius: float = 1.0,
    inner_radius: float = 0.5,
    reference_length: float = 0.5,
) -> dict:
    """Solve the same annulus with the inner loop wound the wrong way.

    The concatenated mesh is the same point set: flipping the traversal of the
    inner loop only flips the sign of its normals. The linear system therefore
    stays square and well-posed, its residual stays at rounding level, and the
    returned ``q`` is simply wrong — the silent failure that
    :func:`tensorbem.multicomponent.check_loop_orientation` exists to prevent.
    """
    annulus = annulus_mesh(
        2 * panels_per_loop,
        panels_per_loop,
        outer_radius=outer_radius,
        inner_radius=inner_radius,
    )
    flipped = annulus.mesh.vertices.copy()
    inner = np.arange(annulus.outer_panels, annulus.panel_count)
    flipped[inner] = flipped[inner][::-1]
    from ..geometry import PanelMesh
    from ..solver import solve_laplace_dirichlet

    values = log_potential_boundary_data(annulus)
    result = solve_laplace_dirichlet(
        PanelMesh(flipped), values, reference_length=reference_length
    )
    exact_q = log_potential_normal_derivative(annulus)
    return {
        "panels_per_loop": int(panels_per_loop),
        "relative_residual_l2": float(result.relative_residual_l2),
        "q_relative_l2_error": float(
            np.linalg.norm(result.normal_derivative - exact_q)
            / np.linalg.norm(exact_q)
        ),
    }


def run_study(
    panels: tuple[int, ...] = (8, 16, 32, 64),
    reference_length: float = 0.5,
) -> dict:
    rows = [
        run_level(int(n), reference_length=reference_length) for n in panels
    ]
    orders = {}
    for key in (
        "q_relative_l2_error",
        "interior_field_relative_l2_error",
        "flux_outer_rel_error",
        "flux_net_over_exact",
    ):
        values = [abs(row[key]) for row in rows]
        orders[key] = [
            float(math.log2(values[i] / values[i + 1]))
            if values[i + 1] > 0.0
            else float("nan")
            for i in range(len(values) - 1)
        ]
    return {
        "schema": SCHEMA,
        "reference_solution": "u = A log r + B, A = 1, B = 0",
        "geometry": {
            "outer_radius": 1.0,
            "inner_radius": 0.5,
            "outer_panels": "2 x panels_per_loop",
            "inner_panels": "panels_per_loop (wound clockwise)",
        },
        "reference_length": float(reference_length),
        "rows": rows,
        "observed_orders": orders,
        "wrong_winding_control": wrong_winding_control(
            int(panels[-1]), reference_length=reference_length
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }


def markdown_summary(study: dict) -> str:
    lines = [
        "| panels/loop | N | kappa | q rel L2 | inner field rel L2 "
        "| outer flux err | inner flux err | net flux |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in study["rows"]:
        lines.append(
            "| {p} | {n} | {k:.1f} | {q:.3e} | {f:.3e} | {fo:+.3%} | {fi:+.3%} | {fn:.3e} |".format(
                p=row["panels_per_loop"],
                n=row["panel_count"],
                k=row["condition_number"],
                q=row["q_relative_l2_error"],
                f=row["interior_field_relative_l2_error"],
                fo=row["flux_outer_rel_error"],
                fi=row["flux_inner_rel_error"],
                fn=row["flux_net_over_exact"],
            )
        )
    control = study["wrong_winding_control"]
    lines.append("")
    lines.append(
        "wrong-winding control (N={}): residual {:.2e}, q rel L2 {:.3f}".format(
            control["panels_per_loop"] * 3,
            control["relative_residual_l2"],
            control["q_relative_l2_error"],
        )
    )
    return "\n".join(lines)


def make_figure(study: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = study["rows"]
    panels = [row["panel_count"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    ax = axes[0]
    ax.loglog(panels, [abs(r["flux_outer_rel_error"]) for r in rows], "o-",
              label="outer loop flux")
    ax.loglog(panels, [abs(r["flux_inner_rel_error"]) for r in rows], "s-",
              label="inner loop flux")
    ax.loglog(panels, [abs(r["flux_net_over_exact"]) for r in rows], "^-",
              label="net flux")
    ax.loglog(panels, [r["q_relative_l2_error"] for r in rows], "d--",
              label="unknown trace q")
    ax.axhline(0.03, color="k", ls=":", lw=1.0, label="3% acceptance")
    ax.set_xlabel("panels on the boundary (both loops)")
    ax.set_ylabel("relative error vs closed form")
    ax.set_title("Each loop's flux stays ~13% high; refinement does not remove it")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.semilogx(panels, [r["condition_number"] for r in rows], "o-", color="C4")
    ax.set_xlabel("panels on the boundary (both loops)")
    ax.set_ylabel(r"2-norm condition number of $S$")
    ax.set_title(r"Conditioning grows $\sim N^{1.5}$ (near-capacity mode)")
    ax.grid(True, which="both", alpha=0.3)
    fig.suptitle(
        "TensorBEM: multiply-connected annulus, u = log r reference "
        "(reference_length = {:.1f})".format(study["reference_length"]),
        fontsize=9,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--reference-length", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--figure", type=Path, default=None)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args(argv)

    study = run_study(
        tuple(int(n) for n in args.panels),
        reference_length=args.reference_length,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(study, indent=1) + "\n")
    if args.figure is not None:
        make_figure(study, args.figure)
    if args.markdown:
        print(markdown_summary(study))
    else:
        print(json.dumps(study, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())