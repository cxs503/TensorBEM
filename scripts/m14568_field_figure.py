#!/usr/bin/env python
"""Field picture for m14568: surface potential from the quadrature-kernel solve.

The cross-check figure (``fig_m14568_quadrature_kernel_solver.png``) carries the
comparison curves; this script carries the field quantity.  It solves the
20-panel submerged geodesic sphere on the *quadrature* kernel with the heave
mode and writes:

* ``outputs/m14568/field_potential.json`` -- per-panel ``|phi|`` and phase,
* ``docs/figures/fig_m14568_quadrature_kernel_field.png`` -- surface potential
  map next to the eigenseries solve of the same case (both kernels, same mesh).

Usage::

    python scripts/m14568_field_figure.py [--subdivision 0] [--nodes 8]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorbem.finite_depth_solver3d import solve_radiation_finite_depth_3d  # noqa: E402
from tensorbem.geometry3d import submerged_sphere_mesh  # noqa: E402

OUT = ROOT / "outputs" / "m14568"
FIG = ROOT / "docs" / "figures" / "fig_m14568_quadrature_kernel_field.png"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subdivision", type=int, default=0)
    parser.add_argument("--nodes", type=int, default=8)
    parser.add_argument("--nu", type=float, default=0.6)
    parser.add_argument("--depth", type=float, default=4.0)
    args = parser.parse_args()

    mesh = submerged_sphere_mesh(args.subdivision, radius=1.0, submergence=1.5)
    results = {}
    for kernel in ("eigenseries", "quadrature"):
        res = solve_radiation_finite_depth_3d(
            mesh, args.nu, args.depth, modes=(2,),
            kernel=kernel, kernel_nodes=args.nodes,
        )
        results[kernel] = {
            "abs_potential": [float(abs(v)) for v in res.potentials[:, 0]],
            "phase": [float(np.angle(v)) for v in res.potentials[:, 0]],
            "added_mass_heave": float(res.added_mass[0, 0]),
            "damping_heave": float(res.damping[0, 0]),
            "condition_number": float(res.condition_number),
            "relative_residual": float(res.relative_residual),
        }
        print(f"{kernel}: A33={res.added_mass[0, 0]:.6f} B33={res.damping[0, 0]:.6f}", flush=True)

    payload = {
        "geometry": {
            "body": "submerged geodesic sphere",
            "panels": int(mesh.panel_count),
            "radius": 1.0,
            "submergence": 1.5,
            "nu": args.nu,
            "depth": args.depth,
            "mode": "heave",
            "quadrature_nodes": args.nodes,
        },
        "kernels": results,
        "rel_err_abs_potential": float(
            np.max(
                np.abs(np.array(results["quadrature"]["abs_potential"])
                       - np.array(results["eigenseries"]["abs_potential"]))
                / np.maximum(np.abs(np.array(results["eigenseries"]["abs_potential"])), 1e-12)
            )
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "field_potential.json").write_text(json.dumps(payload, indent=1))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    vertices = mesh.vertices
    faces = mesh.faces
    centred = vertices - np.array([0.0, 0.0, -1.5])
    fig = plt.figure(figsize=(11.0, 4.6))
    vmax = max(
        max(results["eigenseries"]["abs_potential"]),
        max(results["quadrature"]["abs_potential"]),
    )
    for index, kernel in enumerate(("eigenseries", "quadrature")):
        ax = fig.add_subplot(1, 2, index + 1, projection="3d")
        values = np.array(results[kernel]["abs_potential"])
        face_values = values[faces].mean(axis=1)
        norm = plt.Normalize(0.0, vmax)
        colours = plt.cm.viridis(norm(face_values))
        poly = Poly3DCollection(
            centred[faces], facecolors=colours, edgecolors=(0, 0, 0, 0.15), linewidths=0.4
        )
        ax.add_collection3d(poly)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        ax.set_title(
            f"{kernel} kernel: |phi| (heave)\n"
            f"A33={results[kernel]['added_mass_heave']:.4f}, "
            f"B33={results[kernel]['damping_heave']:.4f}",
            fontsize=9,
        )
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0.0, vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=fig.axes, shrink=0.7, label="|phi| [m^2/s] (unit velocity amplitude)")
    fig.suptitle(
        "m14568  submerged sphere, 20 panels, nu=0.6, h=4 m: same solver on two kernels",
        fontsize=10,
    )
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIG}", flush=True)
    print(f"max rel err |phi| between kernels: {payload['rel_err_abs_potential']:.3e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())