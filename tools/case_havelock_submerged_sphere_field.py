"""Heave-radiation surface field figure for CASE-BEM-SUBMERGED-SPHERE-HAVELOCK.

Solves the 320-panel BEM radiation problem once at ``ka = 1, f/a = 2`` and
plots the heave radiation potential on the wetted surface (panel centroids,
colour = Re / |phi_3|, normalised by ``a * U``).  The evaluated field is also
written to ``runs/m17242/field_heave_ka1.json`` so the figure is reproducible
from recorded numbers.

Usage::

    PYTHONPATH=src python3 tools/case_havelock_submerged_sphere_field.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorbem.free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import submerged_sphere_mesh

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "m17242"
FIGDIR = ROOT / "docs" / "figures"

RADIUS = 1.0
SUBMERGENCE = 2.0
KA = 1.0
SUBDIVISION = 2
RHO = 1025.0
G = 9.80665


def main() -> None:
    mesh = submerged_sphere_mesh(SUBDIVISION, radius=RADIUS, submergence=SUBMERGENCE)
    nu = KA / RADIUS
    ops = assemble_free_surface_operators_3d(mesh, nu)
    rad = solve_radiation_3d(
        mesh, nu, modes=(0, 2), center=(0.0, 0.0, -SUBMERGENCE),
        density=RHO, gravity=G, operators=ops,
    )
    verts = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    centers = verts[faces].mean(axis=1)
    # modes are (surge, heave) in the order passed above -> heave is column 1
    phi = np.asarray(rad.potentials, dtype=complex)[:, 1]
    # normalised by a * U (unit velocity amplitude, radius 1 m) -> dimensionless
    phi_dimless = phi / RADIUS

    RUNS.mkdir(parents=True, exist_ok=True)
    payload = {
        "case": "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
        "field": "heave radiation potential phi_3 on the wetted sphere surface",
        "parameters": {
            "radius": RADIUS, "submergence": SUBMERGENCE, "ka": KA,
            "subdivision": SUBDIVISION, "n_panels": int(mesh.panel_count),
            "density": RHO, "gravity": G, "normalisation": "phi / (a * U)",
        },
        "panel_centers": centers.tolist(),
        "phi_real": phi_dimless.real.tolist(),
        "phi_imag": phi_dimless.imag.tolist(),
        "phi_abs": np.abs(phi_dimless).tolist(),
        "condition_number": rad.condition_number,
        "relative_residual": rad.relative_residual,
        "added_mass_33": float(rad.added_mass[1, 1]),
        "damping_33": float(rad.damping[1, 1]),
    }
    (RUNS / "field_heave_ka1.json").write_text(json.dumps(payload))

    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11.0, 4.2))
    for i, (values, title) in enumerate(
        ((phi_dimless.real, r"$\mathrm{Re}\,\phi_3/(aU)$"),
         (np.abs(phi_dimless), r"$|\phi_3|/(aU)$"))
    ):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        sc = ax.scatter(
            centers[:, 0], centers[:, 1], centers[:, 2],
            c=values, cmap="viridis" if i else "coolwarm", s=9,
        )
        fig.colorbar(sc, ax=ax, shrink=0.72, label=title)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
        ax.set_box_aspect((1, 1, 1))
        ax.set_title("%s, $ka=%.1f$, $f/a=%.1f$, %d panels"
                     % ("Re" if i == 0 else "|.|", KA, SUBMERGENCE / RADIUS, mesh.panel_count),
                     fontsize=9)
    fig.suptitle("Heave radiation potential on the sphere surface (BEM)")
    fig.tight_layout()
    out = FIGDIR / "case_havelock_field.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out, "cond=%.3f res=%.1e" % (rad.condition_number, rad.relative_residual))
    print("phi_abs range %.4f .. %.4f" % (np.abs(phi_dimless).min(), np.abs(phi_dimless).max()))


if __name__ == "__main__":
    main()