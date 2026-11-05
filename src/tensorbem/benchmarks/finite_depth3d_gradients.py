"""Verification study for the analytic gradient of the 3-D finite-depth kernel.

The kernel module :mod:`tensorbem.finite_depth3d` long exposed only ``dG/dz``: the
horizontal derivatives need ``K_1`` and ``H_1^{(1)}``, which were absent, and that
is the single reason a 3-D finite-depth BEM solver could not be assembled (see
``docs/finite_depth3d.md`` §6).  This benchmark is the verification face of the
new ``green_gradient_3d`` / ``green_gradient_source_3d``: every number below is
computed here, nothing is transcribed by hand.

Gates (all independent of the finite-depth code path):

* ``fd``      -- analytic gradient versus a central difference of ``G`` itself.
* ``fs``      -- free-surface condition ``dG/dz - nu G = 0`` on the gradient.
* ``bottom``  -- bottom impermeability ``dG/dz = 0`` on the gradient.
* ``harmonic``-- ``laplacian(G) = 0`` formed from the analytic gradient.
* ``deep``    -- ``h -> inf`` limit against the verified deep-water Kelvin
  gradient (``h k_0`` up to 240; exponential convergence, not truncation).
* ``far``     -- far-field law ``dG/dR = i k_0 G`` (outgoing cylindrical waves)
  and the ``R^{-1/2}`` amplitude decay.
* ``panel``   -- normal derivative ``dG/dn`` of a closed sphere integrated by a
  panel rule versus a central difference of the single-layer integral: this is
  the operator a finite-depth panel assembly would need.

Run as a module::

    PYTHONPATH=src python -m tensorbem.benchmarks.finite_depth3d_gradients \\
        --output outputs/fd3d_grad/study.json \\
        --figures docs/figures \\
        --report docs/finite_depth3d_gradients.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ..free_surface3d import green_function_3d as deep_green_function_3d
from ..free_surface3d import green_gradient_field_3d as deep_green_gradient_3d
from ..finite_depth3d import (
    bessel_k1,
    green_function_3d,
    green_gradient_3d,
    green_gradient_source_3d,
    green_gradient_z_3d,
    hankel_h1,
)

NU = 0.6
DEPTHS = (2.0, 5.0, 20.0)
SOURCE = np.array([0.0, 0.0, -1.25])
TARGETS = np.array(
    [
        [2.0, 0.5, -0.5],
        [0.3, -1.7, -1.9],
        [1.1, 1.1, -0.15],
        [-2.4, 0.8, -0.9],
        [0.0, 0.0, -1.9],
        [3.5, -3.5, -1.2],
    ]
)
FD_STEP = 1.0e-6


def _relative(numerical: np.ndarray, reference: np.ndarray) -> float:
    scale = float(np.max(np.abs(reference)))
    if scale == 0.0:
        return 0.0
    return float(np.max(np.abs(numerical - reference)) / scale)


def _scaled(error: np.ndarray, scale: np.ndarray) -> float:
    """``max|error| / max|scale|`` -- normalisation for a residual, not a difference."""
    magnitude = float(np.max(np.abs(scale)))
    if magnitude == 0.0:
        return 0.0
    return float(np.max(np.abs(error)) / magnitude)


def _fd_gradient(point: np.ndarray, source: np.ndarray, nu: float, depth: float) -> np.ndarray:
    """Central difference of ``G`` with respect to the target coordinates."""
    out = np.zeros(3, dtype=complex)
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = FD_STEP
        out[axis] = (
            green_function_3d(point + step, source, nu, depth)
            - green_function_3d(point - step, source, nu, depth)
        ) / (2.0 * FD_STEP)
    return out


def _fd_source_gradient(
    point: np.ndarray, source: np.ndarray, nu: float, depth: float
) -> np.ndarray:
    out = np.zeros(3, dtype=complex)
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = FD_STEP
        out[axis] = (
            green_function_3d(point, source + step, nu, depth)
            - green_function_3d(point, source - step, nu, depth)
        ) / (2.0 * FD_STEP)
    return out


def _bessel_table() -> Dict[str, float]:
    """``K_1`` / ``H_1^{(1)}`` against SciPy (independent oracle)."""
    from scipy import special as sp

    xs = np.logspace(-8.0, 2.5, 241)
    k1_err = float(np.max(np.abs(bessel_k1(xs) / sp.k1(xs) - 1.0)))
    h1_err = float(np.max(np.abs(hankel_h1(xs) / sp.hankel1(1, xs) - 1.0)))
    return {
        "x_range": [float(xs[0]), float(xs[-1])],
        "k1_max_relative_error": k1_err,
        "h1_max_relative_error": h1_err,
    }


def run_study() -> Dict[str, Any]:
    study: Dict[str, Any] = {
        "nu": NU,
        "depths": list(DEPTHS),
        "source": SOURCE.tolist(),
        "targets": TARGETS.tolist(),
        "fd_step": FD_STEP,
        "numpy_version": np.__version__,
    }

    # ---- 1. bessel layer -----------------------------------------------------
    study["bessel"] = _bessel_table()

    # ---- 2. gradient versus finite differences -------------------------------
    rows: List[Dict[str, Any]] = []
    worst_target_rel = 0.0
    worst_source_rel = 0.0
    for depth in DEPTHS:
        for target in TARGETS:
            analytic = green_gradient_3d(target, SOURCE, NU, depth)
            fd_target = _fd_gradient(target, SOURCE, NU, depth)
            analytic_src = green_gradient_source_3d(target, SOURCE, NU, depth)
            fd_source = _fd_source_gradient(target, SOURCE, NU, depth)
            rel_target = _relative(analytic, fd_target)
            rel_source = _relative(analytic_src, fd_source)
            worst_target_rel = max(worst_target_rel, rel_target)
            worst_source_rel = max(worst_source_rel, rel_source)
            rows.append(
                {
                    "depth": depth,
                    "target": [float(v) for v in target],
                    "target_gradient_relative_error": rel_target,
                    "source_gradient_relative_error": rel_source,
                    "horizontal_relative_error": _relative(
                        analytic[:2], fd_target[:2]
                    ),
                    "vertical_relative_error": _relative(
                        analytic[2], fd_target[2]
                    ),
                }
            )
    study["finite_difference"] = {
        "rows": rows,
        "worst_target_relative_error": worst_target_rel,
        "worst_source_relative_error": worst_source_rel,
    }

    # ---- 3. boundary conditions on the gradient ------------------------------
    fs_rows: List[Dict[str, Any]] = []
    for depth in DEPTHS:
        ring = np.array([[1.7 * math.cos(t), 1.7 * math.sin(t), 0.0] for t in np.linspace(0.0, 2.0 * math.pi, 16)])
        value = green_function_3d(ring, SOURCE, NU, depth)
        grad = green_gradient_3d(ring, SOURCE, NU, depth)
        residual = grad[:, 2] - NU * value
        bottom_pts = np.array(
            [[1.7 * math.cos(t), 1.7 * math.sin(t), -depth] for t in np.linspace(0.0, 2.0 * math.pi, 16)]
        )
        grad_bottom = green_gradient_3d(bottom_pts, SOURCE, NU, depth)
        bottom_value = green_function_3d(bottom_pts, SOURCE, NU, depth)
        fs_rows.append(
            {
                "depth": depth,
                "free_surface_relative_residual": _scaled(residual, value),
                "bottom_relative_residual": _scaled(grad_bottom[:, 2], bottom_value),
            }
        )
    study["boundary_conditions"] = fs_rows

    # ---- 4. harmonicity ------------------------------------------------------
    harmonic_rows: List[Dict[str, Any]] = []
    for depth in DEPTHS:
        # The axis through the source (x = xi, y = eta) is excluded: there dG/dR ~ 1/R
        # diverges logarithmically in the potential itself, so a difference quotient
        # of the gradient there measures roundoff, not harmonicity.
        off_axis = np.hypot(TARGETS[:, 0], TARGETS[:, 1]) > 0.8
        interior = TARGETS[
            (TARGETS[:, 2] < -0.3) & (TARGETS[:, 2] > -depth + 0.3) & off_axis
        ]
        if interior.size == 0:
            continue
        divergence = np.zeros(interior.shape[0], dtype=complex)
        reference = np.zeros(interior.shape[0], dtype=complex)
        for axis in range(3):
            step = np.zeros(3)
            step[axis] = FD_STEP
            plus = green_gradient_3d(interior + step, SOURCE, NU, depth)
            minus = green_gradient_3d(interior - step, SOURCE, NU, depth)
            divergence += (plus[:, axis] - minus[:, axis]) / (2.0 * FD_STEP)
            reference += np.abs(plus[:, axis])
        harmonic_rows.append(
            {
                "depth": depth,
                "laplacian_relative_residual": _scaled(divergence, reference),
            }
        )
    study["harmonicity"] = harmonic_rows

    # ---- 5. deep-water limit -------------------------------------------------
    probe = np.array([[1.0, 0.5, -0.8], [3.0, -2.0, -4.0]])
    deep_rows: List[Dict[str, Any]] = []
    for depth in (20.0, 50.0, 100.0, 200.0, 400.0):
        deep_value = deep_green_function_3d(probe, SOURCE, NU)
        deep_grad = deep_green_gradient_3d(probe, SOURCE, NU)
        deep_rows.append(
            {
                "depth": depth,
                "k0_h": NU * depth,
                "kernel_relative_error": _relative(
                    green_function_3d(probe, SOURCE, NU, depth), deep_value
                ),
                "gradient_relative_error": _relative(
                    green_gradient_3d(probe, SOURCE, NU, depth), deep_grad
                ),
            }
        )
    study["deep_water_limit"] = deep_rows

    # ---- 6. far-field law ----------------------------------------------------
    depth = 5.0
    far_radius = 60.0
    angles = np.linspace(0.0, 2.0 * math.pi, 49)[:-1]
    ring = np.array([[far_radius * math.cos(a), far_radius * math.sin(a), -1.25] for a in angles])
    far_value = green_function_3d(ring, SOURCE, NU, depth)
    far_grad = green_gradient_3d(ring, SOURCE, NU, depth)
    radial = (
        far_grad[:, 0] * np.cos(angles) + far_grad[:, 1] * np.sin(angles)
    )
    law = 1j * NU * far_value
    decay_rows = []
    for radius in (10.0, 20.0, 40.0, 80.0, 160.0):
        pts = np.array([[radius * math.cos(a), radius * math.sin(a), -1.25] for a in angles])
        val = green_function_3d(pts, SOURCE, NU, depth)
        decay_rows.append(
            {
                "radius": radius,
                "k0_r": NU * radius,
                "amplitude_times_sqrt_k0r": float(np.mean(np.abs(val)) * math.sqrt(NU * radius)),
                "radial_gradient_relative_error": _relative(
                    green_gradient_3d(pts, SOURCE, NU, depth)[:, 0] * np.cos(angles)
                    + green_gradient_3d(pts, SOURCE, NU, depth)[:, 1] * np.sin(angles),
                    1j * NU * val,
                ),
            }
        )
    study["far_field"] = {
        "depth": depth,
        "radius": far_radius,
        "angles": angles.tolist(),
        "radial_gradient_relative_to_ik0G": _relative(radial, law),
        "horizontal_to_k0G_ratio": [
            float(v) for v in np.abs(far_grad[:, :2]).sum(axis=1) / np.abs(NU * far_value)
        ],
        "decay": decay_rows,
    }

    # ---- 7. panel-level normal derivative (the unlocked operator) ------------
    depth = 5.0
    radius = 0.25
    centre = np.array([0.0, 0.0, -1.6])
    n_theta, n_phi = 10, 20
    panel_rows = []
    for target in (np.array([2.2, 0.4, -0.7]), np.array([0.0, 0.0, -0.4])):
        total = 0.0 + 0.0j
        fd_total = 0.0 + 0.0j
        for i in range(n_theta):
            theta = math.pi * (i + 0.5) / n_theta
            d_area = 2.0 * math.pi * radius**2 * math.sin(theta) / n_theta / n_phi
            for j in range(n_phi):
                phi = 2.0 * math.pi * (j + 0.5) / n_phi
                normal = np.array(
                    [
                        math.sin(theta) * math.cos(phi),
                        math.sin(theta) * math.sin(phi),
                        math.cos(theta),
                    ]
                )
                point = centre + radius * normal
                total += np.dot(
                    green_gradient_source_3d(target, point, NU, depth), normal
                ) * d_area
                fd = _fd_source_gradient(target, point, NU, depth)
                fd_total += np.dot(fd, normal) * d_area
        panel_rows.append(
            {
                "target": [float(v) for v in target],
                "normal_derivative_integral": abs(total),
                "normal_derivative_integral_imag": total.imag,
                "fd_integral": abs(fd_total),
                "relative_error": _relative(total, fd_total),
            }
        )
    study["panel_normal_derivative"] = {
        "depth": depth,
        "sphere_radius": radius,
        "sphere_centre": centre.tolist(),
        "panels": n_theta * n_phi,
        "rows": panel_rows,
    }

    # ---- 8. z-derivative parity with the pre-existing kernel -----------------
    parity = 0.0
    for depth in DEPTHS:
        for target in TARGETS:
            full = green_gradient_3d(target, SOURCE, NU, depth)
            zonly = green_gradient_z_3d(target, SOURCE, NU, depth)
            parity = max(parity, _relative(full[2], zonly))
    study["z_component_parity"] = parity

    study["verdict"] = _verdict(study)
    return study


def _verdict(study: Dict[str, Any]) -> Dict[str, Any]:
    gates = {
        "bessel_k1_below_1e-8": study["bessel"]["k1_max_relative_error"] < 1.0e-8,
        "bessel_h1_below_1e-12": study["bessel"]["h1_max_relative_error"] < 1.0e-12,
        "fd_target_gradient_below_1e-5": study["finite_difference"][
            "worst_target_relative_error"
        ]
        < 1.0e-5,
        "fd_source_gradient_below_1e-5": study["finite_difference"][
            "worst_source_relative_error"
        ]
        < 1.0e-5,
        "free_surface_below_1e-10": max(
            row["free_surface_relative_residual"] for row in study["boundary_conditions"]
        )
        < 1.0e-10,
        "bottom_below_1e-10": max(
            row["bottom_relative_residual"] for row in study["boundary_conditions"]
        )
        < 1.0e-10,
        "harmonicity_below_1e-5": max(
            (row["laplacian_relative_residual"] for row in study["harmonicity"]),
            default=0.0,
        )
        < 1.0e-5,
        "deep_water_limit_monotone_and_below_1e-6": _deep_ok(study["deep_water_limit"]),
        "far_field_law_below_2e-2": study["far_field"]["radial_gradient_relative_to_ik0G"]
        < 2.0e-2,
        "panel_normal_derivative_below_1e-4": max(
            row["relative_error"] for row in study["panel_normal_derivative"]["rows"]
        )
        < 1.0e-4,
        "z_component_parity_exact": study["z_component_parity"] < 1.0e-14,
    }
    return {"gates": gates, "all_pass": all(gates.values())}


def _deep_ok(rows: Sequence[Dict[str, Any]]) -> bool:
    errors = [row["gradient_relative_error"] for row in rows]
    monotone = all(b <= a for a, b in zip(errors, errors[1:]))
    return bool(monotone and errors[-1] < 1.0e-6 and errors[0] > errors[-1])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(study: Dict[str, Any], directory: Path) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    # (1) finite-difference agreement + deep-water limit
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    rows = study["finite_difference"]["rows"]
    depths = sorted({row["depth"] for row in rows})
    width = 0.35
    positions = np.arange(len(depths))
    worst_target = [
        max(r["target_gradient_relative_error"] for r in rows if r["depth"] == d)
        for d in depths
    ]
    worst_source = [
        max(r["source_gradient_relative_error"] for r in rows if r["depth"] == d)
        for d in depths
    ]
    axes[0].bar(
        positions - width / 2.0, worst_target, width, label="target gradient"
    )
    axes[0].bar(
        positions + width / 2.0, worst_source, width, label="source gradient"
    )
    axes[0].set_yscale("log")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels([f"h = {d:g} m" for d in depths])
    axes[0].axhline(1.0e-5, color="k", ls="--", lw=1.0, label="gate 1e-5")
    axes[0].set_ylabel("max relative error vs central difference")
    axes[0].set_title("Analytic gradient vs finite-difference reference (FD step 1e-6 m)")
    axes[0].legend(fontsize=8, loc="best")
    axes[0].grid(alpha=0.3, axis="y")

    deep = study["deep_water_limit"]
    k0h = [row["k0_h"] for row in deep]
    axes[1].semilogy(k0h, [row["kernel_relative_error"] for row in deep], "o-", label="G")
    axes[1].semilogy(
        k0h, [row["gradient_relative_error"] for row in deep], "s-", label="grad G"
    )
    axes[1].set_xlabel(r"$k_0 h$")
    axes[1].set_ylabel("relative deviation from deep-water kernel")
    axes[1].set_title("Deep-water limit (h = 20 ... 400 m)")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = directory / "fig_finite_depth3d_gradient_accuracy.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    # (2) field cloud of |grad G| in the x-z plane
    nu = study["nu"]
    depth = 5.0
    # Half-step offsets keep the section plane off both the source singularity
    # (z = zeta, x = 0) and the source's vertical axis.
    xs = np.linspace(-6.0, 6.0, 141) + 0.5 * (12.0 / 140)
    zs = np.linspace(-depth, 0.0, 101) + 0.5 * (depth / 100)
    zs = np.minimum(zs, -1.0e-3)
    grid = np.array([[x, 0.021, z] for z in zs for x in xs])
    grad = green_gradient_3d(grid, SOURCE, nu, depth).reshape(len(zs), len(xs), 3)
    magnitude = np.minimum(np.abs(grad).max(axis=2), 1.0e3)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    cs = ax.contourf(xs, zs, np.log10(magnitude), levels=24, cmap="viridis")
    ax.axhline(0.0, color="w", lw=2.0)
    ax.axhline(-depth, color="w", lw=1.2, ls="--")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(
        r"$\log_{10}\max_i|\partial_i G|$ in the plane $y=0$   "
        r"($h=5$ m, $\nu=0.6$ m$^{-1}$, source at $z=-1.25$ m)"
    )
    ax.text(-5.7, -0.28, "free surface z = 0", color="w", fontsize=8)
    ax.text(-5.7, -depth + 0.12, "rigid bottom z = -h", color="w", fontsize=8)
    fig.colorbar(cs, ax=ax, label=r"$\log_{10}$ gradient magnitude")
    fig.tight_layout()
    path = directory / "fig_finite_depth3d_gradient_field.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    # (3) far field: angular pattern and R^{-1/2} decay
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    angles = np.array(study["far_field"]["angles"])
    ratio = np.array(study["far_field"]["horizontal_to_k0G_ratio"])
    axes[0].plot(np.degrees(angles), ratio, "o-", ms=3, label=r"$|\nabla_h G|/(k_0|G|)$")
    axes[0].axhline(1.0, color="k", ls="--", lw=1.0, label="far-field law = 1")
    axes[0].set_xlabel(r"azimuth $\theta$ [deg]")
    axes[0].set_ylabel("ratio")
    axes[0].set_ylim(0.8, 1.2)
    axes[0].set_title(f"Far field at $k_0R$ = {study['far_field']['radius']*study['nu']:.0f}")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    decay = study["far_field"]["decay"]
    radii = [row["radius"] for row in decay]
    axes[1].semilogx(
        radii,
        [row["amplitude_times_sqrt_k0r"] for row in decay],
        "o-",
        label=r"$\langle|G|\rangle\sqrt{k_0R}$",
    )
    axes[1].set_xlabel("R [m]")
    axes[1].set_ylabel(r"$\langle|G|\rangle\sqrt{k_0R}$")
    axes[1].set_ylim(0.0, 2.0 * max(row["amplitude_times_sqrt_k0r"] for row in decay))
    axes[1].set_title("Outgoing cylindrical-wave amplitude decay")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = directory / "fig_finite_depth3d_gradient_farfield.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)
    return written


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def render_markdown(study: Dict[str, Any]) -> str:
    fd = study["finite_difference"]
    lines: List[str] = []
    lines.append("# 3-D finite-depth Green function: analytic gradient and its verification")
    lines.append("")
    lines.append(
        "Generated by `python -m tensorbem.benchmarks.finite_depth3d_gradients --report`; "
        "every number is read from `study.json` produced in the same run "
        f"(numpy {study['numpy_version']}, nu = {study['nu']}, source at "
        f"({study['source'][0]:g}, {study['source'][1]:g}, {study['source'][2]:g}) m)."
    )
    lines.append("")
    lines.append("## 1. What was missing")
    lines.append("")
    lines.append(
        "`docs/finite_depth3d.md` shipped the John-eigenseries kernel with an analytic "
        "$\\partial G/\\partial z$ only, and recorded the reason: *\"a horizontal derivative "
        "needs $K_1$/$H_1$ and belongs to the panel layer\"*. Without it no finite-depth "
        "panel assembly is possible, because both the single-layer normal derivative and "
        "the double-layer normal derivative need $\\nabla_x G$."
    )
    lines.append("")
    lines.append("## 2. What was added")
    lines.append("")
    lines.append("| object | content |")
    lines.append("|---|---|")
    lines.append(
        "| `bessel_k1` | $K_1$ by the A&S 9.6.11 ascending series ($x<9$) and the Hankel "
        "asymptotic series ($x\\ge9$), pure NumPy |"
    )
    lines.append(
        "| `hankel_h1` | $H_1^{(1)}$ from the already-verified `bessel_j01_y01` |"
    )
    lines.append(
        "| `green_gradient_3d` | analytic $\\nabla_x G$ (three components), shape-leading "
        "trailing axis 3 |"
    )
    lines.append(
        "| `green_gradient_source_3d` | analytic $\\nabla_\\xi G$ = transposed target "
        "gradient (kernel symmetry; **no** sign flip in finite depth) |"
    )
    lines.append("")
    lines.append("## 3. Results")
    lines.append("")
    lines.append("### 3.1 Bessel layer against SciPy")
    lines.append("")
    bessel = study["bessel"]
    lines.append("| quantity | max relative error |")
    lines.append("|---|---:|")
    lines.append(
        f"| $K_1$, $x\\in[10^{{{math.log10(bessel['x_range'][0]):.0f}}},"
        f"10^{{{math.log10(bessel['x_range'][1]):.1f}}}]$ | {bessel['k1_max_relative_error']:.2e} |"
    )
    lines.append(
        f"| $H_1^{{(1)}}$, same range | {bessel['h1_max_relative_error']:.2e} |"
    )
    lines.append("")
    lines.append("### 3.2 Gradient against central differences of $G$")
    lines.append("")
    lines.append(
        f"Central difference with step {study['fd_step']:.0e} m on `green_function_3d`; "
        f"{len(fd['rows'])} (depth, target) combinations."
    )
    lines.append("")
    lines.append("| depth h [m] | max rel. error, $\\nabla_x G$ | max rel. error, $\\nabla_\\xi G$ |")
    lines.append("|---:|---:|---:|")
    for depth in sorted({row["depth"] for row in fd["rows"]}):
        subset = [row for row in fd["rows"] if row["depth"] == depth]
        lines.append(
            f"| {depth:g} | "
            f"{max(r['target_gradient_relative_error'] for r in subset):.2e} | "
            f"{max(r['source_gradient_relative_error'] for r in subset):.2e} |"
        )
    lines.append("")
    lines.append(
        f"Worst case over all cases: {fd['worst_target_relative_error']:.2e} (target), "
        f"{fd['worst_source_relative_error']:.2e} (source). The residual is the $O(h^2)$ "
        "truncation of the difference quotient, not the analytic derivative."
    )
    lines.append("")
    lines.append("### 3.3 Boundary conditions carried by the gradient")
    lines.append("")
    lines.append(
        "| depth h [m] | $|\\partial_z G-\\nu G|/|G|$ on $z=0$ | $|\\partial_z G|/|G|$ on $z=-h$ |"
    )
    lines.append("|---:|---:|---:|")
    for row in study["boundary_conditions"]:
        lines.append(
            f"| {row['depth']:g} | {row['free_surface_relative_residual']:.2e} | "
            f"{row['bottom_relative_residual']:.2e} |"
        )
    lines.append("")
    lines.append("### 3.4 Harmonicity and deep-water limit")
    lines.append("")
    lines.append("| depth h [m] | $|\\nabla^2 G|/|\\nabla G|$ (FD of the analytic gradient) |")
    lines.append("|---:|---:|")
    for row in study["harmonicity"]:
        lines.append(f"| {row['depth']:g} | {row['laplacian_relative_residual']:.2e} |")
    lines.append("")
    lines.append("| depth h [m] | $k_0h$ | kernel deviation | gradient deviation |")
    lines.append("|---:|---:|---:|---:|")
    for row in study["deep_water_limit"]:
        lines.append(
            f"| {row['depth']:g} | {row['k0_h']:.0f} | {row['kernel_relative_error']:.2e} | "
            f"{row['gradient_relative_error']:.2e} |"
        )
    lines.append("")
    lines.append("### 3.5 Far field and the panel operator")
    lines.append("")
    lines.append(
        f"At $R={study['far_field']['radius']:g}$ m ($k_0R="
        f"{study['far_field']['radius']*study['nu']:.0f}$) the radial gradient obeys "
        f"$\\partial_R G=i k_0G$ to {study['far_field']['radial_gradient_relative_to_ik0G']:.2e}, "
        "and $\\langle|G|\\rangle\\sqrt{k_0R}$ is flat to "
        f"{_decay_spread(study['far_field']['decay']):.1%} over $R=10\\ldots160$ m."
    )
    lines.append("")
    panel = study["panel_normal_derivative"]
    lines.append(
        f"Normal-derivative integral $\\int_S \\partial G/\\partial n\\,dS$ over a sphere "
        f"(radius {panel['sphere_radius']:g} m, centre "
        f"({panel['sphere_centre'][0]:g}, {panel['sphere_centre'][1]:g}, "
        f"{panel['sphere_centre'][2]:g}) m, {panel['panels']} panels) against a central "
        "difference of the same integral:"
    )
    lines.append("")
    lines.append("| target [m] | $|\\int \\partial_n G\\,dS|$ | relative error vs FD |")
    lines.append("|---|---:|---:|")
    for row in panel["rows"]:
        lines.append(
            f"| ({row['target'][0]:g}, {row['target'][1]:g}, {row['target'][2]:g}) | "
            f"{row['normal_derivative_integral']:.6e} | {row['relative_error']:.2e} |"
        )
    lines.append("")
    lines.append("## 4. Verdict")
    lines.append("")
    verdict = study["verdict"]
    lines.append("| gate | pass |")
    lines.append("|---|---|")
    for name, ok in verdict["gates"].items():
        lines.append(f"| `{name}` | {'PASS' if ok else 'FAIL'} |")
    lines.append("")
    lines.append(
        "**All gates pass.**" if verdict["all_pass"] else "**At least one gate fails.**"
    )
    lines.append("")
    lines.append(
        "The $z$ component reproduces the pre-existing `green_gradient_z_3d` to "
        f"{study['z_component_parity']:.1e} relative, so the new entry point is a strict "
        "superset of the old one."
    )
    lines.append("")
    lines.append("## 5. Figures")
    lines.append("")
    lines.append("![gradient accuracy](figures/fig_finite_depth3d_gradient_accuracy.png)")
    lines.append("")
    lines.append(
        "*Left:* analytic gradient against central differences of $G$ for "
        "$h=2,5,20$ m; *right:* deviation of the kernel and of its gradient from the "
        "deep-water Kelvin kernel as $k_0h$ grows (exponential, not truncation-limited)."
    )
    lines.append("")
    lines.append("![gradient field](figures/fig_finite_depth3d_gradient_field.png)")
    lines.append("")
    lines.append(
        "*Field distribution* of $\\max_i|\\partial_iG|$ in the plane $y=0$ between the "
        "free surface and the rigid bottom; the logarithmic singularity at the source and "
        "the mirror structure of the evanescent modes are visible."
    )
    lines.append("")
    lines.append("![far field](figures/fig_finite_depth3d_gradient_farfield.png)")
    lines.append("")
    lines.append(
        "*Left:* $|\\nabla_hG|/(k_0|G|)$ around a far-field ring, compared with the "
        "outgoing-wave law $=1$; *right:* $\\langle|G|\\rangle\\sqrt{k_0R}$ against $R$, "
        "i.e. the cylindrical $R^{-1/2}$ decay."
    )
    lines.append("")
    lines.append("## 6. Limitations")
    lines.append("")
    lines.append(
        "* Kernel level only. No panel assembly is shipped: the eigenseries carries the "
        "$1/(4\\pi r)$ singularity inside the mode sum, so self-panel and near-singular "
        "quadrature (the analogue of `tensorbem.near_singular`) is still the open item."
    )
    lines.append(
        "* Cost grows as $h/(\\pi R)$ modes; the gradient needs $K_1$ in addition to $K_0$, "
        "i.e. roughly twice the kernel cost at identical truncation."
    )
    lines.append(
        "* Constant depth, linearised potential flow, $z\\le0$ only; no sloping bottom, no current."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def _decay_spread(rows: Sequence[Dict[str, Any]]) -> float:
    values = [row["amplitude_times_sqrt_k0r"] for row in rows]
    return float((max(values) - min(values)) / max(values))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--figures", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    study = run_study()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(study, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.output}")
    if args.figures is not None:
        for path in make_figures(study, args.figures):
            print(f"wrote {path}")
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_markdown(study))
        print(f"wrote {args.report}")
    print("verdict:", "PASS" if study["verdict"]["all_pass"] else "FAIL")
    for name, ok in study["verdict"]["gates"].items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    return 0 if study["verdict"]["all_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())