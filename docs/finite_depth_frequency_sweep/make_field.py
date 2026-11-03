"""Radiated-potential field map for the short-wave end of the ``ka`` sweep.

Companion to :mod:`tensorbem.benchmarks.finite_depth_frequency_sweep`: the
sweep reports coefficients only, so this script draws the field quantity those
coefficients come from.  The case is the highest ``ka`` in the sweep
(``ka = 1.5``, ``lambda/a = 4.19``) with the section at ``f/a = 2``.  The map is
the real part of the unit-velocity heave potential, i.e. the quantity that is
integrated over the body to obtain the added mass.

Because the free-surface potential carries the factor ``exp(k0*z)``, the map is
drawn on a window that starts just below the free surface; the outgoing wave
train is visible along both directions, which is the far-field condition the
energy identity is built on.

    PYTHONPATH=src python3 docs/finite_depth_frequency_sweep/make_field.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tensorbem.benchmarks.finite_depth_radiation import submerged_circle_mesh
from tensorbem.finite_depth import green_function
from tensorbem.finite_depth_solver import _solve_sources

GRAVITY = 9.80665
DENSITY = 1025.0
KA = 1.5
RATIO = 2.0
DEPTH_RATIO = 50.0
PANELS = 64

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "finite_depth_frequency_sweep"


def build() -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nu = KA
    depth = DEPTH_RATIO
    mesh = submerged_circle_mesh(PANELS, 1.0, RATIO)
    sources, _, _, _ = _solve_sources(mesh, nu, depth, "heave", 8, None)

    x = np.linspace(-4.0, 4.0, 320)
    z = np.linspace(-3.0, -0.02, 220)
    grid_x, grid_z = np.meshgrid(x, z)
    field = np.zeros_like(grid_x, dtype=complex)
    for row, z_value in enumerate(z):
        field[row] = green_function(
            x,
            np.full_like(x, z_value),
            mesh.centers[:, 0][:, None],
            mesh.centers[:, 1][:, None],
            nu,
            depth,
            modes=2048,
        ) @ sources

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    contour = axes[0].contourf(x, z, field.real, levels=48, cmap="RdBu_r")
    axes[0].plot(mesh.centers[:, 0], mesh.centers[:, 1], "k-", lw=1.2)
    axes[0].set_xlabel(r"$x/a$")
    axes[0].set_ylabel(r"$z/a$")
    axes[0].set_title(rf"$\Re\phi$ for unit heave velocity, $ka={KA}$, $f/a={RATIO}$")
    fig.colorbar(contour, ax=axes[0], label=r"$\Re\phi$ [m$^2$/s]")

    amplitude = np.abs(field)
    contour = axes[1].contourf(x, z, amplitude, levels=48, cmap="viridis")
    axes[1].plot(mesh.centers[:, 0], mesh.centers[:, 1], "w-", lw=1.2)
    axes[1].set_xlabel(r"$x/a$")
    axes[1].set_ylabel(r"$z/a$")
    axes[1].set_title(r"$|\phi|$: surface wave train at $ka=1.5$")
    fig.colorbar(contour, ax=axes[1], label=r"$|\phi|$ [m$^2$/s]")
    fig.tight_layout()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "radiated_potential_ka1.5.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)

    # Sample the free-surface elevation proxy on the surface line to check the
    # radiated wavelength directly against 2*pi/k0 = lambda.
    surface_z = -1.0e-3
    surface = green_function(
        x, np.full_like(x, surface_z),
        mesh.centers[:, 0][:, None], mesh.centers[:, 1][:, None],
        nu, depth, modes=2048,
    ) @ sources
    crossings = np.sum(np.diff(np.sign(np.real(surface))) != 0)
    return {
        "figure": str(path.relative_to(ROOT)),
        "ka": KA,
        "submergence_ratio": RATIO,
        "depth_ratio": DEPTH_RATIO,
        "density": DENSITY,
        "gravity": GRAVITY,
        "modes": 2048,
        "x_window": [float(x[0]), float(x[-1])],
        "z_window": [float(z[0]), float(z[-1])],
        "free_surface_sign_changes": int(crossings),
        "lambda_over_a": 2.0 * math.pi / KA,
    }


def main() -> int:
    payload = build()
    path = OUTPUT / "field_summary.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())