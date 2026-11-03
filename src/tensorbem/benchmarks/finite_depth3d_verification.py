"""Verification study for the 3-D finite-depth free-surface Green function.

Generates the numbers quoted in ``docs/finite_depth3d.md`` and the report
figures, with zero hand-transcription: every table entry is computed here.

Run as ``python -m tensorbem.benchmarks.finite_depth3d_verification`` from a
tree with ``src`` on ``PYTHONPATH``.
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np

from tensorbem.finite_depth import evanescent_wavenumbers, finite_depth_wavenumber
from tensorbem.finite_depth3d import (
    bessel_k0,
    free_surface_residual,
    green_function_3d_finite_depth,
    hankel_h0,
    wave_part_3d_finite_depth,
)
from tensorbem.free_surface3d import green_function_3d

NU = 1.0
SOURCE = np.array([0.0, 0.0, -1.5])
FIELD = np.array(
    [
        [0.5, 0.0, 0.0],
        [1.0, 0.7, -0.4],
        [2.5, -1.0, -1.0],
        [-3.0, 2.0, -0.9],
        [6.0, 0.5, -2.0],
    ]
)


def deep_water_limit_table(depths):
    rows = []
    deep = green_function_3d(FIELD, SOURCE, NU)
    for depth in depths:
        finite = green_function_3d_finite_depth(FIELD, SOURCE, NU, depth)
        rows.append(
            {
                "nu_h": depth * NU,
                "k0": finite_depth_wavenumber(NU, depth),
                "k0_over_nu": finite_depth_wavenumber(NU, depth) / NU,
                "rel_err_total": float(np.max(np.abs(finite - deep)) / np.max(np.abs(deep))),
                "rel_err_real": float(
                    np.max(np.abs(finite.real - deep.real)) / np.max(np.abs(deep.real))
                ),
                "rel_err_imag": float(
                    np.max(np.abs(finite.imag - deep.imag)) / np.max(np.abs(deep.imag))
                ),
            }
        )
    for i in range(len(rows) - 1):
        ratio = rows[i]["rel_err_total"] / rows[i + 1]["rel_err_total"]
        rows[i]["observed_order"] = float(math.log(ratio) / math.log(rows[i + 1]["nu_h"] / rows[i]["nu_h"]))
    return rows


def practical_deviation_table(depths):
    """How wrong is the deep-water kernel at finite depth?"""
    rows = []
    for depth in depths:
        source = np.array([0.0, 0.0, -0.4 * depth])
        points = FIELD.copy()
        points[:, 2] = np.clip(points[:, 2], -0.85 * depth, 0.0)
        deep = green_function_3d(points, source, NU)
        finite = green_function_3d_finite_depth(points, source, NU, depth)
        rows.append(
            {
                "nu_h": depth * NU,
                "k0": finite_depth_wavenumber(NU, depth),
                "phase_speed_ratio": finite_depth_wavenumber(NU, depth) / NU,
                "max_rel_deviation_of_deep_water": float(
                    np.max(np.abs(finite - deep)) / np.max(np.abs(finite))
                ),
            }
        )
    return rows


def boundary_condition_table(depths):
    xy = np.array([[0.5, 0.0], [1.3, -0.6], [4.0, 1.0]])
    rows = []
    for depth in depths:
        entry = {"nu_h": depth * NU}
        for factor in (1e-3, 1e-4):
            entry[f"free_surface_residual_step_{factor:g}h"] = float(
                np.max(free_surface_residual(xy, -0.4 * depth, NU, depth, step=factor * depth))
            )
        step = 1e-6 * depth
        target = np.array([[0.9, 0.3, -depth]])
        source = np.array([0.4, 0.1, -0.5 * depth])
        g0 = green_function_3d_finite_depth(target, source, NU, depth)[0]
        g1 = green_function_3d_finite_depth(target + np.array([[0.0, 0.0, step]]), source, NU, depth)[0]
        entry["bottom_normal_derivative_times_h_over_g"] = float(abs(g0 - g1) / step * depth / abs(g0))
        rows.append(entry)
    return rows


def bessel_table():
    try:
        from scipy import special
    except ImportError:  # pragma: no cover - scipy is only a verification aid
        return {"scipy": False}
    x = np.linspace(1e-3, 200.0, 800)
    k0 = bessel_k0(x)
    h = hankel_h0(x)
    small = x < 2.0
    return {
        "scipy": True,
        "k0_max_rel_err": float(np.max(np.abs(k0 - special.k0(x))) / np.max(np.abs(special.k0(x)))),
        "k0_max_rel_err_small_argument": float(
            np.max(np.abs(k0[small] - special.k0(x[small]))) / np.max(np.abs(special.k0(x[small])))
        ),
        "j0_max_rel_err": float(np.max(np.abs(h.real - special.j0(x))) / np.max(np.abs(special.j0(x)))),
        "y0_max_rel_err": float(np.max(np.abs(h.imag - special.y0(x))) / np.max(np.abs(special.y0(x)))),
    }


def mode_sensitivity_table():
    rows = []
    for depth in (1.0, 5.0, 20.0):
        source = np.array([0.0, 0.0, -0.4 * depth])
        points = np.array([[0.4 * depth, 0.0, 0.0], [1.2 * depth, 0.6 * depth, -0.5 * depth]])
        auto = green_function_3d_finite_depth(points, source, NU, depth)
        rows.append(
            {
                "nu_h": depth * NU,
                "rel_change_with_10x_modes": float(
                    np.max(np.abs(auto - green_function_3d_finite_depth(points, source, NU, depth, modes=20000)))
                    / np.max(np.abs(auto))
                ),
                "evanescent_mode_residual": float(
                    np.max(
                        np.abs(
                            evanescent_wavenumbers(NU, depth, 8) * np.tan(evanescent_wavenumbers(NU, depth, 8) * depth)
                            + NU
                        )
                    )
                ),
            }
        )
    return rows


def far_field_phase_table(depth=3.0):
    k0 = finite_depth_wavenumber(NU, depth)
    source = np.array([0.0, 0.0, -1.0])
    radii = np.linspace(20.0, 23.0, 7)
    points = np.column_stack([radii, np.zeros_like(radii), np.zeros_like(radii)])
    wave = wave_part_3d_finite_depth(points, source, NU, depth)
    steps = np.angle(wave[1:] / wave[:-1])
    expected = k0 * np.diff(radii)
    return {
        "nu_h": depth * NU,
        "k0": k0,
        "max_phase_step_error": float(np.max(np.abs(steps - expected))),
        "mean_phase_wavenumber": float(np.mean(steps) / np.mean(np.diff(radii))),
    }


def figures(outdir: pathlib.Path, limit_rows, dev_rows):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return []

    written = []
    depths = [1.0, 2.0, 5.0, math.inf]
    radii = np.linspace(0.2, 40.0, 600)
    points = np.column_stack([radii, np.zeros_like(radii), np.zeros_like(radii)])
    source = np.array([0.0, 0.0, -0.5])

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for depth in depths:
        if math.isinf(depth):
            values = green_function_3d(points, source, NU)
            label = r"$h=\infty$"
        else:
            values = green_function_3d_finite_depth(points, source, NU, depth)
            label = rf"$\nu h={depth:g}$"
        axes[0].plot(NU * radii, np.abs(values), lw=1.4, label=label)
        axes[1].plot(NU * radii, values.real, lw=1.1, label=label)
    axes[0].set_xlabel(r"$\nu R$")
    axes[0].set_ylabel(r"$|G|\,\nu$")
    axes[0].set_title("free-surface kernel magnitude")
    axes[1].set_xlabel(r"$\nu R$")
    axes[1].set_ylabel(r"$\mathrm{Re}(G)\,\nu$")
    axes[1].set_title("wave pattern (surface elevation)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "fig_freesurface_curves.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    written.append(str(path.name))

    xs = np.linspace(-12.0, 12.0, 260)
    zs = np.linspace(-2.0, 0.0, 120)
    grid = np.array([[x, 0.0, z] for z in zs for x in xs])
    depth = 2.0
    finite = green_function_3d_finite_depth(grid, source, NU, depth).imag.reshape(zs.size, xs.size)
    deep = green_function_3d(grid, source, NU).imag.reshape(zs.size, xs.size)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), sharey=True)
    vmax = float(np.max(np.abs(finite)))
    for ax, data, title in (
        (axes[0], finite, rf"finite depth $\nu h=2$"),
        (axes[1], deep, r"deep water"),
    ):
        mesh = ax.pcolormesh(xs, zs, data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        ax.set_title(title)
        ax.set_xlabel(r"$\nu x$")
    axes[0].set_ylabel(r"$\nu z$")
    fig.colorbar(mesh, ax=axes, shrink=0.85, label=r"$\mathrm{Im}(G)\,\nu$")
    fig.savefig(outdir / "fig_field_map.png", dpi=110)
    plt.close(fig)
    written.append("fig_field_map.png")

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    nh = np.array([row["nu_h"] for row in limit_rows])
    err = np.array([row["rel_err_total"] for row in limit_rows])
    ax.loglog(nh, err, "o-", label="max relative difference")
    ax.loglog(nh, err[0] * (nh[0] / nh) ** 3, "k--", lw=1.0, label=r"$\propto (\nu h)^{-3}$")
    ax.set_xlabel(r"$\nu h$")
    ax.set_ylabel("deep-water limit residual")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig_limit_convergence.png", dpi=110)
    plt.close(fig)
    written.append("fig_limit_convergence.png")

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    nh = np.array([row["nu_h"] for row in dev_rows])
    dev = np.array([row["max_rel_deviation_of_deep_water"] for row in dev_rows])
    ax.loglog(nh, dev, "s-", color="crimson")
    ax.axhline(0.03, color="k", ls=":", lw=1.0, label="3% tolerance")
    ax.set_xlabel(r"$\nu h$")
    ax.set_ylabel("deep-water kernel error")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig_practical_deviation.png", dpi=110)
    plt.close(fig)
    written.append("fig_practical_deviation.png")
    return written


def main() -> dict:
    outdir = pathlib.Path("outputs/finite_depth3d")
    outdir.mkdir(parents=True, exist_ok=True)
    limit_rows = deep_water_limit_table([25.0, 50.0, 100.0, 200.0, 400.0])
    dev_rows = practical_deviation_table([0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0])
    payload = {
        "nu": NU,
        "source": SOURCE.tolist(),
        "deep_water_limit": limit_rows,
        "practical_deviation": dev_rows,
        "boundary_conditions": boundary_condition_table([1.0, 5.0, 20.0]),
        "bessel": bessel_table(),
        "mode_sensitivity": mode_sensitivity_table(),
        "far_field_phase": far_field_phase_table(),
    }
    payload["figures"] = figures(outdir, limit_rows, dev_rows)
    (outdir / "metrics.json").write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    data = main()
    print(json.dumps({k: v for k, v in data.items() if k != "figures"}, indent=2)[:4000])
    print("figures:", data["figures"])
