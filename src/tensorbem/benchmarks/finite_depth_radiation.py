"""Finite-depth radiation benchmark for a submerged circular section.

Computes the added mass and damping of a circular cylinder (radius ``a``, axis
at depth ``f``) oscillating in heave and sway in water of constant depth ``h``,
with the finite-depth indirect solver of
:mod:`tensorbem.finite_depth_solver`.

Three checks are reported for every case in addition to the coefficients:

* **deep-water collapse** - at large ``h`` the finite-depth coefficients must
  approach Ursell's analytic deep-water values
  (:func:`tensorbem.analytic.ursell_cylinder_added_mass`), which are an
  independent closed-form construction;
* **energy identity** - the damping read off the radiated wave amplitude must
  equal the damping obtained from the surface pressure,
  ``omega*rho*Im(force)``;
* **method of fundamental solutions** - a second discretisation of the same
  kernel (point sources, no panels) must reproduce the panel coefficients.

Usage::

    PYTHONPATH=src python3 -m tensorbem.benchmarks.finite_depth_radiation \
        --json outputs/finite_depth_radiation/benchmark.json \
        --figures outputs/finite_depth_radiation
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from tensorbem.analytic import ursell_cylinder_added_mass
from tensorbem.finite_depth_solver import (
    radiation_reference,
    solve_radiation,
)
from tensorbem.geometry import PanelMesh
from tensorbem.free_surface_solver import solve_radiation_indirect

GRAVITY = 9.80665
MOTIONS = {"heave": 1, "sway": 0}


def submerged_circle_mesh(
    panel_count: int, radius: float = 1.0, submergence: float = 2.0
) -> PanelMesh:
    """Equally spaced panels on a circle of the given radius, centred at ``-f``."""
    if panel_count < 8:
        raise ValueError("panel_count must be at least 8")
    angles = 2.0 * math.pi * np.arange(panel_count) / panel_count
    return PanelMesh(
        np.column_stack(
            (radius * np.cos(angles), radius * np.sin(angles) - submergence)
        )
    )


def _case(
    depth: float,
    motion: str,
    panel_count: int,
    ka: float,
    radius: float,
    submergence: float,
    density: float,
    source_count: int,
    analytic: tuple[float, float],
) -> dict:
    nu = ka / radius
    mesh = submerged_circle_mesh(panel_count, radius, submergence)
    index = MOTIONS[motion]
    result, far = solve_radiation(depth=depth, mesh=mesh, nu=nu, direction=motion,
                                  density=density, gravity=GRAVITY)
    reference = radiation_reference(
        mesh, nu, depth, motion, source_count=source_count, density=density,
        gravity=GRAVITY,
    )
    analytic_mass, analytic_damping = analytic
    added_mass = float(result.added_mass[index])
    damping = float(result.damping[index])
    return {
        "motion": motion,
        "depth": float(depth),
        "nu_h": float(nu * depth),
        "clearance": float(depth - submergence - radius),
        "added_mass": added_mass,
        "damping": damping,
        "far_field_damping": float(far),
        "far_over_near": float(far / damping) if damping else float("nan"),
        "mfs_added_mass": float(reference.added_mass[index]),
        "mfs_damping": float(reference.damping[index]),
        "mfs_mass_rel_diff": float(
            abs(reference.added_mass[index] - added_mass) / abs(added_mass)
        ),
        "mfs_damping_rel_diff": float(
            abs(reference.damping[index] - damping) / abs(damping)
        ),
        "deep_added_mass": float(analytic_mass),
        "deep_damping": float(analytic_damping),
        "mass_dev_from_deep": float(added_mass / analytic_mass - 1.0),
        "damping_dev_from_deep": float(damping / analytic_damping - 1.0),
        "condition_number": float(result.condition_number),
        "relative_residual": float(result.relative_residual),
        "panels": int(panel_count),
    }


def run(
    depths: tuple[float, ...] = (3.2, 4.0, 5.0, 6.0, 8.0, 10.0, 14.0, 20.0, 50.0),
    panel_count: int = 64,
    ka: float = 0.6,
    radius: float = 1.0,
    submergence: float = 2.0,
    density: float = 1025.0,
    source_count: int = 24,
    refinement_depths: tuple[float, ...] = (),
    refinement_panels: tuple[int, ...] = (32, 64, 128),
) -> dict:
    """Run the depth sweep, the panel refinement and the deep-water comparison."""
    cases = []
    started = time.time()
    analytic = ursell_cylinder_added_mass(
        radius, submergence, math.sqrt(ka / radius * GRAVITY), density,
        gravity=GRAVITY,
    )
    for depth in depths:
        for motion in MOTIONS:
            cases.append(
                _case(
                    depth, motion, panel_count, ka, radius, submergence,
                    density, source_count, analytic,
                )
            )
    refinement = []
    for depth in refinement_depths:
        for panels in refinement_panels:
            for motion in MOTIONS:
                refinement.append(
                    _case(
                        depth, motion, panels, ka, radius, submergence,
                        density, source_count, analytic,
                    )
                )

    masses = [c["added_mass"] for c in cases if c["motion"] == "heave"]
    analytic_mass, analytic_damping = analytic
    return {
        "benchmark": "finite_depth_radiation_submerged_circle",
        "radius": radius,
        "submergence": submergence,
        "ka": ka,
        "nu": ka / radius,
        "density": density,
        "gravity": GRAVITY,
        "panels": panel_count,
        "elapsed_seconds": round(time.time() - started, 2),
        "cases": cases,
        "refinement": refinement,
        "deep_analytic": {
            "added_mass": float(analytic_mass),
            "damping": float(analytic_damping),
        },
        "max_mass_deviation_from_deep": max(abs(m / analytic_mass - 1.0) for m in masses),
    }


def _write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def plot(payload: dict, directory: Path) -> list[Path]:
    """Depth sweep, deep-limit convergence, energy identity and a field map."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    written = []
    cases = payload["cases"]
    analytic = payload["deep_analytic"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for motion, colour in (("heave", "#1f77b4"), ("sway", "#d62728")):
        subset = sorted(
            (c for c in cases if c["motion"] == motion), key=lambda c: c["depth"]
        )
        depth = [c["depth"] for c in subset]
        axes[0].semilogx(
            depth, [c["added_mass"] for c in subset], "o-", color=colour,
            label=f"{motion} (finite depth)",
        )
        axes[1].semilogx(
            depth, [c["damping"] for c in subset], "o-", color=colour,
            label=f"{motion} (finite depth)",
        )
    for axis, key, ylabel in (
        (axes[0], "added_mass", r"added mass $A$ [kg/m]"),
        (axes[1], "damping", r"damping $B$ [kg/(m s)]"),
    ):
        axis.axhline(analytic[key], color="k", ls="--", lw=1.2,
                     label="Ursell analytic (deep water)")
        axis.set_xlabel(r"depth $h/a$")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    axes[0].set_title("Depth sweep, $ka=0.6$, $f/a=2$")
    fig.tight_layout()
    path = directory / "depth_sweep.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for motion, colour in (("heave", "#1f77b4"), ("sway", "#d62728")):
        subset = sorted(
            (c for c in cases if c["motion"] == motion), key=lambda c: c["depth"]
        )
        axes[0].loglog(
            [c["depth"] for c in subset],
            [abs(c["mass_dev_from_deep"]) for c in subset],
            "o-", color=colour, label=motion,
        )
        axes[1].semilogx(
            [c["depth"] for c in subset],
            [c["far_over_near"] for c in subset],
            "o-", color=colour, label=motion,
        )
    axes[0].set_xlabel(r"depth $h/a$")
    axes[0].set_ylabel(r"$|A(h)/A_{\infty} - 1|$")
    axes[0].set_title("Collapse onto the deep-water analytic solution")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)
    axes[1].axhline(1.0, color="k", ls="--", lw=1.2)
    axes[1].set_xlabel(r"depth $h/a$")
    axes[1].set_ylabel(r"$B_{\rm far}/B_{\rm near}$")
    axes[1].set_title("Energy identity (64 panels)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = directory / "verification.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    # Radiated potential in the fluid strip: a genuine field quantity, not a
    # curve fit.  The panels carry a source density, so phi is the kernel
    # integral evaluated on a grid.
    try:
        from tensorbem.finite_depth_solver import _solve_sources
        from tensorbem.geometry import PanelMesh as _PanelMesh

        depth = min(c["depth"] for c in cases)
        mesh = submerged_circle_mesh(64, payload["radius"], payload["submergence"])
        sources, _, _, _ = _solve_sources(
            mesh, payload["nu"], depth, "heave", 8, None
        )
        x = np.linspace(-3.5, 3.5, 260)
        z = np.linspace(-depth + 1e-3, -1e-3, 150)
        grid_x, grid_z = np.meshgrid(x, z)
        # Reuse the modal part of the kernel plus the Rankine logarithm; the
        # grid is coarse enough that the self-panel singularity is irrelevant.
        from tensorbem.finite_depth import green_function as fd_green

        field = np.zeros_like(grid_x, dtype=complex)
        for row, z_value in enumerate(z):
            field[row] = fd_green(
                x, np.full_like(x, z_value),
                mesh.centers[:, 0][:, None],
                mesh.centers[:, 1][:, None],
                payload["nu"],
                depth,
                modes=512,
            ) @ sources
        fig, axis = plt.subplots(figsize=(7.2, 4.0))
        contour = axis.contourf(x, z, field.real, levels=40, cmap="RdBu_r")
        axis.plot(mesh.centers[:, 0], mesh.centers[:, 1], "k-", lw=1.0)
        axis.set_xlabel(r"$x/a$")
        axis.set_ylabel(r"$z/a$")
        axis.set_title(
            rf"Radiated potential $\Re\phi$ (heave, $h/a={depth:.1f}$, "
            rf"$ka={payload['ka']}$)"
        )
        fig.colorbar(contour, ax=axis, label=r"$\Re\phi$ [m$^2$/s]")
        fig.tight_layout()
        path = directory / "radiated_potential.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(path)
    except Exception as error:  # pragma: no cover - figure is best effort
        print(f"field figure skipped: {error}")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--figures", type=Path, default=None)
    parser.add_argument("--panels", type=int, default=64)
    parser.add_argument("--ka", type=float, default=0.6)
    args = parser.parse_args(argv)
    payload = run(panel_count=args.panels, ka=args.ka)
    if args.json is not None:
        _write_json(payload, args.json)
    if args.figures is not None:
        for path in plot(payload, args.figures):
            print(f"wrote {path}")
    for case in payload["cases"]:
        print(
            f"h/a={case['depth']:5.1f} {case['motion']:5s} "
            f"A={case['added_mass']:9.2f} B={case['damping']:9.2f} "
            f"Bfar/Bnear={case['far_over_near']:7.4f} "
            f"MFS dA={case['mfs_mass_rel_diff']:.2e} "
            f"dev_deep={case['mass_dev_from_deep']:+.2e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())