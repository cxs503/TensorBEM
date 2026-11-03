"""Evidence generator for the 3-D finite-depth free-surface Green function.

Running this module produces the numbers quoted in ``docs/finite_depth3d_kernel.md``:

* **deep-water limit** -- ``G_finite(h)`` against the validated Kelvin kernel
  :func:`tensorbem.free_surface3d.green_function_3d` as ``k0*h`` grows.  The
  residual is the bottom-image term, so it must fall like ``1/h``;
* **modal truncation** -- the error of the modal sum at fixed geometry versus the
  number of evanescent modes, which sets the ``modes`` rule;
* **boundary conditions** -- ``dG/dz - nu*G`` at ``z = 0`` and ``dG/dz`` at
  ``z = -h``, the two conditions the kernel is built to satisfy;
* **gradient** -- analytic derivatives against central differences;
* **dimension reduction** -- the classical identity that maps the evanescent
  sector onto the 2-D kernel :mod:`tensorbem.finite_depth`.

Usage::

    python -m tensorbem.benchmarks.finite_depth3d_kernel --report finite_depth3d.json \
        --figure finite_depth3d.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ..finite_depth import evanescent_wavenumbers, finite_depth_wavenumber
from ..finite_depth3d import (
    convergence_history,
    green_function,
    green_gradient_field,
    modified_bessel_k01,
    suggested_mode_count,
)
from ..free_surface3d import green_function_3d

DEPTHS = (30.0, 60.0, 120.0, 240.0, 480.0)
PROBES = ((1.0, -0.5, -0.5), (2.0, 0.0, -1.0), (0.7, -0.2, -2.0))


def _pair(radius: float, z: float, z0: float):
    return (
        np.array([[radius, 0.0, z]], dtype=float),
        np.array([[0.0, 0.0, z0]], dtype=float),
    )


def deep_water_table() -> list[dict[str, float]]:
    nu = 1.0
    rows: list[dict[str, float]] = []
    for depth in DEPTHS:
        for radius, z, z0 in PROBES:
            target, source = _pair(radius, z, z0)
            reference = complex(green_function_3d(target, source, nu).ravel()[0])
            value = complex(green_function(target, source, nu, depth).ravel()[0])
            rows.append(
                {
                    "k0_h": nu * depth,
                    "radius": radius,
                    "z": z,
                    "z0": z0,
                    "value_real": value.real,
                    "value_imag": value.imag,
                    "kelvin_real": reference.real,
                    "kelvin_imag": reference.imag,
                    "relative_error": abs(value - reference) / abs(reference),
                }
            )
    return rows


def truncation_table() -> list[dict[str, float]]:
    nu, depth = 1.0, 20.0
    target, source = _pair(0.9, -1.0, -0.7)
    history = convergence_history(
        target, source, nu, depth, (8, 16, 32, 64, 128, 256, 512, 1024)
    )
    k0 = finite_depth_wavenumber(nu, depth)
    mu = evanescent_wavenumbers(nu, depth, 1)[0]
    for record in history:
        record["mu_1_times_R"] = float(mu * 0.9)
        record["k0_times_depth"] = float(k0 * depth)
    return history


def boundary_tables() -> dict[str, list[dict[str, float]]]:
    rng = np.random.default_rng(20261103)
    cases = [(0.05, 2.0), (0.35, 12.0), (1.0, 20.0), (2.0, 60.0), (3.0, 150.0)]
    free_surface: list[dict[str, float]] = []
    bottom: list[dict[str, float]] = []
    count = 8
    for nu, depth in cases:
        radius = rng.uniform(0.1, 5.0, count)
        z_source = rng.uniform(-0.95 * depth, -0.02 * depth, count)
        source = np.stack([np.zeros(count), np.zeros(count), z_source], axis=1)
        target = np.stack([radius, rng.uniform(-0.5, 0.5, count), np.zeros(count)], axis=1)
        value = green_function(target, source, nu, depth)
        _, _, dg_dz = green_gradient_field(target, source, nu, depth)
        free_surface.append(
            {
                "nu": nu,
                "depth": depth,
                "max_abs_g": float(np.max(np.abs(value))),
                "max_residual": float(np.max(np.abs(dg_dz - nu * value))),
                "max_relative_residual": float(
                    np.max(np.abs(dg_dz - nu * value)) / np.max(np.abs(value))
                ),
            }
        )
        target_bottom = np.stack(
            [radius, rng.uniform(-0.5, 0.5, count), np.full(count, -depth)], axis=1
        )
        value_bottom = green_function(target_bottom, source, nu, depth)
        _, _, dg_dz_bottom = green_gradient_field(target_bottom, source, nu, depth)
        bottom.append(
            {
                "nu": nu,
                "depth": depth,
                "max_abs_g": float(np.max(np.abs(value_bottom))),
                "max_residual": float(np.max(np.abs(dg_dz_bottom))),
                "max_relative_residual": float(
                    np.max(np.abs(dg_dz_bottom)) / np.max(np.abs(value_bottom))
                ),
            }
        )
    return {"free_surface": free_surface, "bottom": bottom}


def gradient_table() -> list[dict[str, float]]:
    nu, depth = 0.35, 12.0
    rng = np.random.default_rng(41)
    count = 6
    target = np.stack(
        [
            rng.uniform(0.2, 4.0, count),
            rng.uniform(-1.0, 1.0, count),
            rng.uniform(-0.95 * depth, -0.1 * depth, count),
        ],
        axis=1,
    )
    source = np.stack(
        [np.zeros(count), np.zeros(count), rng.uniform(-0.9 * depth, -0.2 * depth, count)],
        axis=1,
    )
    analytic = green_gradient_field(target, source, nu, depth)
    step = 1e-6
    rows: list[dict[str, float]] = []
    for axis, name, component in ((0, "x", analytic[0]), (1, "y", analytic[1]), (2, "z", analytic[2])):
        plus = target.copy()
        minus = target.copy()
        plus[:, axis] += step
        minus[:, axis] -= step
        numeric = (
            green_function(plus, source, nu, depth) - green_function(minus, source, nu, depth)
        ) / (2.0 * step)
        rows.append(
            {
                "axis": name,
                "max_abs_difference": float(np.max(np.abs(component - numeric))),
                "max_relative_difference": float(
                    np.max(np.abs(component - numeric)) / np.max(np.abs(numeric))
                ),
            }
        )
    return rows


def reduction_table() -> dict[str, float]:
    mu, x = 1.3, 0.8
    y = np.linspace(0.0, 60.0, 600001)
    integrand = modified_bessel_k01(mu * np.hypot(x, y))[0]
    integral = 2.0 * np.trapezoid(integrand, y)
    expected = math.pi / mu * math.exp(-mu * x)
    return {
        "mu": mu,
        "X": x,
        "numeric_integral": float(integral),
        "analytic": float(expected),
        "relative_error": float(abs(integral - expected) / expected),
    }


def mode_rule_table() -> list[dict[str, float]]:
    nu, depth = 1.0, 20.0
    rows: list[dict[str, float]] = []
    for separation in (0.2, 0.5, 1.0, 2.0):
        count = suggested_mode_count(nu, depth, separation, tolerance=1e-10)
        target, source = _pair(separation, -1.0, -0.7)
        rule = complex(green_function(target, source, nu, depth, modes=count).ravel()[0])
        reference = complex(
            green_function(target, source, nu, depth, modes=min(4 * count, 4096)).ravel()[0]
        )
        rows.append(
            {
                "R": separation,
                "modes": float(count),
                "relative_to_reference": float(abs(rule - reference) / abs(reference)),
            }
        )
    return rows


def _figure(report: dict, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - plotting is optional
        return False

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    deep = report["deep_water"]
    for radius, z, z0 in PROBES:
        subset = [
            row
            for row in deep
            if row["radius"] == radius and row["z"] == z and row["z0"] == z0
        ]
        axes[0].loglog(
            [row["k0_h"] for row in subset],
            [max(row["relative_error"], 1e-14) for row in subset],
            "o-",
            label=f"R={radius:g}, z={z:g}, z0={z0:g}",
        )
    axes[0].set_xlabel(r"$k_0 h$")
    axes[0].set_ylabel("relative error vs Kelvin kernel")
    axes[0].set_title("Deep-water limit (bottom image decays as 1/h)")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7)

    trunc = report["truncation"]
    axes[1].semilogy(
        [row["modes"] for row in trunc],
        [max(row["relative_to_finest"], 1e-16) for row in trunc],
        "s-",
    )
    axes[1].set_xlabel("evanescent modes in the sum")
    axes[1].set_ylabel("relative error vs 1024-mode reference")
    axes[1].set_title(r"Modal truncation at $k_0h=20$, $R=0.9$")
    axes[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--figure", type=Path, default=None)
    args = parser.parse_args(argv)

    report = {
        "tool": "tensorbem.benchmarks.finite_depth3d_kernel",
        "convention": "z up, free surface z=0, rigid bottom z=-h, -nabla^2 G = delta, exp(-i omega t)",
        "deep_water": deep_water_table(),
        "truncation": truncation_table(),
        "boundaries": boundary_tables(),
        "gradient": gradient_table(),
        "reduction": reduction_table(),
        "mode_rule": mode_rule_table(),
    }

    if args.figure is not None:
        report["figure_written"] = _figure(report, args.figure)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))

    print(json.dumps({key: report[key] for key in ("reduction", "mode_rule")}, indent=2))
    worst = max(row["relative_error"] for row in report["deep_water"])
    print(f"worst deep-water relative error: {worst:.3e}")
    print(
        "worst free-surface residual: "
        f"{max(row['max_relative_residual'] for row in report['boundaries']['free_surface']):.3e}"
    )
    print(
        "worst bottom residual: "
        f"{max(row['max_relative_residual'] for row in report['boundaries']['bottom']):.3e}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())