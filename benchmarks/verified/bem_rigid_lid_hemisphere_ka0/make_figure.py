#!/usr/bin/env python3
"""m17243 — figure for the rigid-lid hemisphere added-mass study.

Three panels:
  (a) A33/(rho*nabla) vs ka (log) — solver mesh ladder, Hulme (1982),
      rigid-lid reference (0.830954) and pressure-release line (0.5).
  (b) relative error vs ka — vs rigid-lid reference and vs Hulme.
  (c) Re(phi) on the hemisphere surface (ka=1e-3, 256 panels) — the
      even-parity (rigid-lid) potential field.

Run:  python3 runs/m17243_rigid_lid_hemisphere/make_figure.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _repo_root() -> Path:
    for parent in HERE.parents:
        if (parent / "src" / "tensorbem" / "__init__.py").exists():
            return parent
    raise RuntimeError("repo root not found")


sys.path.insert(0, str(_repo_root() / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tensorbem.free_surface_solver3d import solve_radiation_3d
from tensorbem.geometry3d import hemisphere_hull_mesh

RADIUS = 1.0
DENSITY = 1025.0
GRAVITY = 9.80665


def main():
    out = HERE
    payload = json.loads((out / "results.json").read_text())
    ref = payload["reference"]["rigid_lid_added_mass_dimless"]
    mesh_rows = payload["mesh_ladder"]
    ka_rows = payload["ka_ladder"]

    # solve once more for the field (sub=3, ka=1e-3)
    mesh = hemisphere_hull_mesh(3, radius=RADIUS)
    res = solve_radiation_3d(mesh, 1.0e-3, modes=[2],
                             density=DENSITY, gravity=GRAVITY)
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    phi = np.asarray(res.potentials, dtype=complex)
    phi = phi[:, 0].real if phi.ndim == 2 else phi.real

    fig = plt.figure(figsize=(15.5, 4.8))

    # (a) A33 vs ka
    ax = fig.add_subplot(1, 3, 1)
    for sub, color, ls in [(2, "tab:blue", ":"), (3, "tab:orange", "--"),
                           (4, "tab:green", "-")]:
        rows = [r for r in mesh_rows if r["subdivisions"] == sub]
        ax.plot([r["ka"] for r in rows], [r["added_mass_dimless"] for r in rows],
                marker="o", ms=4, ls=ls, color=color,
                label=f"BEM {rows[0]['panels']} panels")
    ax.plot([r["ka"] for r in ka_rows],
            [r["hulme_added_mass_dimless"] for r in ka_rows],
            marker="s", ms=4, ls="-", color="tab:red", label="Hulme (1982)")
    ax.axhline(ref, color="k", lw=1.2, label=f"rigid-lid ref {ref:.4f}")
    ax.axhline(0.5, color="gray", lw=1.0, ls=":", label="pressure release 0.5")
    ax.set_xscale("log")
    ax.set_xlabel("ka")
    ax.set_ylabel(r"$A_{33}/(\rho\nabla)$")
    ax.set_title("Heave added mass, floating hemisphere")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)

    # (b) relative error vs ka
    ax = fig.add_subplot(1, 3, 2)
    ax.plot([r["ka"] for r in ka_rows],
            [100 * r["rel_error_vs_ref"] for r in ka_rows],
            marker="o", ms=4, color="tab:blue", label="vs rigid-lid ref")
    ax.plot([r["ka"] for r in ka_rows],
            [100 * r["rel_error_vs_hulme"] for r in ka_rows],
            marker="s", ms=4, color="tab:red", label="vs Hulme (1982)")
    ax.axhline(5.0, color="gray", lw=0.8, ls=":")
    ax.axhline(-5.0, color="gray", lw=0.8, ls=":")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("ka")
    ax.set_ylabel("rel. error (%)")
    ax.set_title("BEM error (256 panels)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (c) Re(phi) on the hemisphere surface
    ax = fig.add_subplot(1, 3, 3, projection="3d")
    acc = np.zeros(verts.shape[0])
    cnt = np.zeros(verts.shape[0])
    for f, pv in zip(faces, phi):
        for v in f:
            acc[v] += pv
            cnt[v] += 1
    vphi = acc / np.maximum(cnt, 1)
    tri = verts[faces]
    cmin, cmax = vphi.min(), vphi.max()
    for i in range(tri.shape[0]):
        c = 0.5 * (vphi[faces[i, 0]] + vphi[faces[i, 1]] + vphi[faces[i, 2]])
        ax.add_collection3d(Poly3DCollection(
            [tri[i]],
            facecolors=plt.cm.coolwarm((c - cmin) / (cmax - cmin + 1e-30)),
            edgecolors="none",
        ))
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 0.1)
    ax.set_box_aspect((1, 1, 0.6))
    ax.set_title(r"$\mathrm{Re}(\phi)$ on hull (ka=1e-3, 256 panels)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    fig.tight_layout()
    out_path = out / "figure.png"
    fig.savefig(out_path, dpi=150)
    print(f"figure written: {out_path}")


if __name__ == "__main__":
    main()
