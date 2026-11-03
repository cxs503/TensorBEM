"""Numerical evidence for the finite-depth free-surface kernel.

Prints a machine-generated record of:

1. wave kinematics - ``k0``, phase and group velocity against ``k0*h``, with the
   shallow- and deep-water limits;
2. the ``depth -> infinity`` limit against the deep-water kernel of
   :mod:`tensorbem.free_surface` (independent code path, no fitted constant);
3. convergence of the Kummer-accelerated modal sum in the mode count;
4. the residual of the free-surface and bottom conditions, i.e. evidence that
   the series really solves the strip problem rather than merely matching a
   reference number.

Every number in the accompanying report is produced here; nothing is copied by
hand. ``--json PATH`` writes the same record as JSON.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

from tensorbem import finite_depth, free_surface

NU = 0.5
GRAVITY = 9.80665
POINTS = [(0.3, -1.0, -1.5), (2.0, -0.5, -0.5), (5.0, -2.0, -0.3), (-1.7, -3.0, -0.9)]
GRADIENT_POINTS = [(0.3, -1.0, -1.5), (2.0, -0.5, -0.5), (5.0, -2.0, -0.3)]


def kinematics(nu: float) -> list[dict]:
    rows = []
    for depth in (0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 50.0, 400.0):
        k0 = finite_depth.finite_depth_wavenumber(nu, depth)
        c = finite_depth.phase_velocity(nu, depth, gravity=GRAVITY)
        cg = finite_depth.group_velocity(nu, depth, gravity=GRAVITY)
        rows.append(
            {
                "depth": depth,
                "k0": k0,
                "k0_h": k0 * depth,
                "phase_velocity": c,
                "group_velocity": cg,
                "group_over_phase": cg / c,
            }
        )
    return rows


def shallow_water_error(nu: float, depth: float) -> float:
    k0 = finite_depth.finite_depth_wavenumber(nu, depth)
    return abs(k0 - math.sqrt(nu / depth)) / k0


def deep_water_limit(nu: float) -> list[dict]:
    rows = []
    for depth in (3.0, 10.0, 50.0, 200.0, 800.0):
        value_error = 0.0
        gradient_error = 0.0
        for x, z, source_z in POINTS:
            finite = finite_depth.green_function(x, z, 0.0, source_z, nu, depth)
            deep = free_surface.green_function(x, z, 0.0, source_z, nu)
            value_error = max(value_error, abs(finite - deep) / abs(deep))
        for x, z, source_z in GRADIENT_POINTS:
            step = 1e-6
            deep_x = (
                free_surface.green_function(x + step, z, 0.0, source_z, nu)
                - free_surface.green_function(x - step, z, 0.0, source_z, nu)
            ) / (2.0 * step)
            deep_z = (
                free_surface.green_function(x, z + step, 0.0, source_z, nu)
                - free_surface.green_function(x, z - step, 0.0, source_z, nu)
            ) / (2.0 * step)
            gx, gz = finite_depth.green_gradient(
                x, z, 0.0, source_z, nu, depth
            )
            scale = max(abs(deep_x), abs(deep_z))
            gradient_error = max(
                gradient_error, abs(gx - deep_x) / scale, abs(gz - deep_z) / scale
            )
        rows.append(
            {
                "depth": depth,
                "nu_depth": nu * depth,
                "max_relative_value_error": float(value_error),
                "max_relative_gradient_error": float(gradient_error),
            }
        )
    return rows


def mode_convergence(nu: float) -> list[dict]:
    rows = []
    x, z, source_z = 0.05, -0.7, -1.4
    for depth in (3.0, 16.0):
        reference = finite_depth.green_function(
            x, z, 0.0, source_z, nu, depth, modes=16384
        )
        row = {"depth": depth, "reference": [float(reference.real), float(reference.imag)]}
        for modes in (64, 256, 1024, 4096):
            value = finite_depth.green_function(
                x, z, 0.0, source_z, nu, depth, modes=modes
            )
            row["error_modes_%d" % modes] = float(abs(value - reference))
        rows.append(row)
    return rows


def boundary_residuals(nu: float) -> list[dict]:
    rows = []
    step = 1e-6
    x = np.array([0.37, -1.4, 2.9])
    source_x = np.array([0.11, 0.11, 0.11])
    for depth in (1.0, 3.0, 16.0):
        for source_z in (-0.8,):
            source = np.full_like(x, source_z)

            def kernel(z_value):
                return finite_depth.green_function(
                    x, np.full_like(x, z_value), source_x, source, nu, depth
                )

            surface = (kernel(0.0) - kernel(-2.0 * step)) / (2.0 * step) - nu * kernel(0.0)
            bottom = (kernel(-depth + step) - kernel(-depth)) / step
            rows.append(
                {
                    "depth": depth,
                    "source_depth": source_z,
                    "surface_relative_residual": float(
                        np.abs(surface).max() / np.abs(nu * kernel(0.0)).max()
                    ),
                    "bottom_relative_residual": float(
                        np.abs(bottom).max() / np.abs(kernel(-depth + step)).max()
                    ),
                }
            )
    return rows


def group_velocity_error(nu: float) -> float:
    worst = 0.0
    for depth in (0.5, 3.0, 40.0):
        k0 = finite_depth.finite_depth_wavenumber(nu, depth)
        analytic = finite_depth.group_velocity(nu, depth, gravity=GRAVITY)

        def omega_of_k(k):
            return math.sqrt(GRAVITY * k * math.tanh(k * depth))

        step = 1e-6 * k0
        numeric = (omega_of_k(k0 + step) - omega_of_k(k0 - step)) / (2.0 * step)
        worst = max(worst, abs(analytic - numeric) / abs(numeric))
    return float(worst)


def build_record(nu: float = NU) -> dict:
    return {
        "nu": nu,
        "gravity": GRAVITY,
        "kinematics": kinematics(nu),
        "shallow_water_relative_error": {
            "nu_0.5_h_0.001": shallow_water_error(nu, 1e-3),
            "nu_0.5_h_0.01": shallow_water_error(nu, 1e-2),
        },
        "deep_water_limit": deep_water_limit(nu),
        "mode_convergence": mode_convergence(nu),
        "boundary_residuals": boundary_residuals(nu),
        "group_velocity_relative_error": group_velocity_error(nu),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=None, help="write the record as JSON here")
    parser.add_argument("--nu", type=float, default=NU)
    arguments = parser.parse_args()
    record = build_record(arguments.nu)

    print("Wave kinematics, nu = %.3f (deep water: k0 -> nu, c_g/c -> 0.5)" % record["nu"])
    print("%8s %10s %10s %12s %12s %10s" % ("h", "k0", "k0*h", "c", "c_g", "c_g/c"))
    for row in record["kinematics"]:
        print(
            "%8.3f %10.6f %10.4f %12.6f %12.6f %10.6f"
            % (
                row["depth"],
                row["k0"],
                row["k0_h"],
                row["phase_velocity"],
                row["group_velocity"],
                row["group_over_phase"],
            )
        )
    print()
    print("Shallow-water limit, |k0 - sqrt(nu/h)|/k0:")
    for key, value in record["shallow_water_relative_error"].items():
        print("  %-14s %.3e" % (key, value))
    print()
    print("depth -> infinity limit against tensorbem.free_surface:")
    print("%8s %10s %18s %20s" % ("h", "nu*h", "value rel. err", "gradient rel. err"))
    for row in record["deep_water_limit"]:
        print(
            "%8.1f %10.1f %18.3e %20.3e"
            % (
                row["depth"],
                row["nu_depth"],
                row["max_relative_value_error"],
                row["max_relative_gradient_error"],
            )
        )
    print()
    print("Kummer convergence at (x, z, z0) = (0.05, -0.7, -1.4):")
    for row in record["mode_convergence"]:
        print(
            "  h=%.1f  " % row["depth"]
            + "  ".join(
                "%d:%.2e" % (mode, row["error_modes_%d" % mode])
                for mode in (64, 256, 1024, 4096)
            )
        )
    print()
    print("Boundary-condition residuals (relative):")
    for row in record["boundary_residuals"]:
        print(
            "  h=%.1f source_z=%.2f  surface=%.2e  bottom=%.2e"
            % (
                row["depth"],
                row["source_depth"],
                row["surface_relative_residual"],
                row["bottom_relative_residual"],
            )
        )
    print()
    print("Group velocity vs numerical d(omega)/dk: %.3e" % record["group_velocity_relative_error"])

    if arguments.json:
        with open(arguments.json, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
        print("wrote %s" % arguments.json)


if __name__ == "__main__":
    main()