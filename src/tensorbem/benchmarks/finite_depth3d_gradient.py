"""Verification ledger for the 3-D finite-depth kernel *gradient* and its Bessel layer.

The 3-D finite-depth free-surface kernel (:mod:`tensorbem.finite_depth3d`) shipped
with ``G`` and ``dG/dz`` only; the README lists the finite-depth boundary-element
solver as the next increment, and a panel layer needs the full ``nabla G`` plus the
order-1 Bessel/Hankel functions ``K_1``/``H_1^{(1)}`` that enter the horizontal
derivative.  This module is the verification face of that increment:

1. **Bessel accuracy.**  ``K_1`` is new, and taking stock of ``K_0`` while adding
   it exposed a defect: the ascending series of ``K_0`` was used up to
   ``x = 9``, where ``1/K_0`` amplifies the cancellation between the
   ``-ln(x/2) I_0`` term and the series, leaving a **2.1e-8 relative cliff at
   x ~ 8.9** while the asymptotic series is not yet converged.  Both functions
   are now evaluated by the Laplace-integral quadrature on ``4 <= x < 16``.
   ``legacy`` rows below reproduce the old branches so the fix is measurable.
2. **Analytic-derivative consistency.**  ``nabla_x G`` is checked against
   central/Richardson differences of ``G`` for a ladder of steps, and
   ``nabla_x G(x, xi) == nabla_xi G(xi, x)`` (the kernel is symmetric) is checked
   as a closed identity.
3. **Physical closures.**  ``d/dx(dG/dz) = nu dG/dx`` on the free surface,
   ``d/dx(dG/dz) = 0`` on the bottom, and the horizontal gradient is curl free.
4. **Limits.**  The ``1/(4 pi r)`` singularity strength ``4 pi r^2 |nabla G| -> 1``
   and the deep-water collapse of the gradient against the independent Kelvin
   kernel of :mod:`tensorbem.free_surface3d`.

Every number is computed here; the document and the tests read this ledger.

Run::

    PYTHONPATH=src python3 -m tensorbem.benchmarks.finite_depth3d_gradient \\
        --output docs/m3673/outputs/finite_depth3d_gradient.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np

from tensorbem.finite_depth3d import (
    _K0_SERIES_LIMIT,
    _K_ASYMPTOTIC_LIMIT,
    _bessel_k0_k1_quad,
    _bessel_k0_large,
    _bessel_k0_small,
    _bessel_k1_large,
    _bessel_k1_small,
    bessel_k0,
    bessel_k1,
    finite_depth_wavenumber,
    green_function_3d,
    green_gradient_3d,
    green_gradient_source_3d,
)
from tensorbem.free_surface3d import (
    green_function_3d as deep_water_kernel,
    green_gradient_field_3d as deep_water_gradient,
)

NU = 0.6  # omega**2/g
DEPTH = 4.0
#: ``K_0``/``K_1`` references at 40 significant digits from ``mpmath.besselk``
#: (mpmath 1.3.0).  Chosen to straddle every branch seam of the evaluator
#: (``x = 4``, ``x = 16``) and to sample the cancellation gap densely.
K_REFERENCE: tuple[tuple[float, float, float], ...] = (
    (0.0001, 9.3262719134502756, 9999.9995086864037),
    (0.001, 7.0236888005623817, 999.99623815608561),
    (0.01, 4.7212447301610947, 99.973894118296244),
    (0.10000000000000001, 2.4270690247020164, 9.853844780870606),
    (0.5, 0.92441907122766587, 1.6564411200033009),
    (1, 0.42102443824070834, 0.60190723019723458),
    (2, 0.11389387274953344, 0.13986588181652243),
    (3, 0.034739504386279249, 0.040156431128194184),
    (3.9900000000000002, 0.01128522791654109, 0.012627149543333438),
    (4, 0.011159676085853025, 0.012483498887268431),
    (4.5, 0.0063998572432339747, 0.0070780949089680901),
    (5, 0.0036910983340425942, 0.0040446134454521646),
    (6, 0.0012439943280131232, 0.0013439197177355091),
    (7, 0.00042479574186923179, 0.00045418248688489695),
    (8, 0.00014647070522281539, 0.00015536921180500115),
    (8.9000000000000004, 5.6539599333854405e-05, 5.9635344082692677e-05),
    (10, 1.778006231616765e-05, 1.8648773453825585e-05),
    (11, 6.2430205476536768e-06, 6.5208606745808858e-06),
    (12, 2.2008253973114916e-06, 2.2907574647671878e-06),
    (13, 7.7845438614204961e-07, 8.0785884122023472e-07),
    (15.99, 3.53567012207157e-08, 3.6445990777642788e-08),
    (16, 3.4994116639364992e-08, 3.6071571175287797e-08),
    (20, 5.7412378153365248e-10, 5.8830579695570384e-10),
    (40, 8.3928611000995672e-19, 8.4971319548610387e-19),
    (100, 4.6566282291759019e-45, 4.6798537356369095e-45),
    (300, 3.7236948548891435e-132, 3.7298958583323724e-132),
)

#: Dense scan for the branch-error curves.  Restricted to ``1 <= x <= 60``:
#: there the Laplace-integral quadrature is the reference (``~5e-15``, derived
#: independently of both power series), and ``x <= 60`` keeps ``K_0(x)`` well
#: above the ``exp(-x)`` underflow floor.
_SCAN = np.concatenate(
    (
        np.logspace(0.0, math.log10(60.0), 240),
        np.linspace(3.5, 17.0, 91),
    )
)

TARGETS = np.array(
    [
        [0.7, 0.3, -0.4],
        [1.5, 1.0, -2.0],
        [2.0, -1.0, -3.0],
        [0.35, -0.55, -1.2],
        [1.1, 0.9, -3.5],
    ]
)
SOURCES = np.array([0.2, -0.1, -1.0])


# ---------------------------------------------------------------------------
# 1. Bessel layer
# ---------------------------------------------------------------------------
def _legacy_k0(x: np.ndarray) -> np.ndarray:
    """The shipped pre-fix ``K_0``: series below 9, asymptotic above."""
    out = np.empty_like(x)
    small = x < 9.0
    out[small] = _bessel_k0_small(x[small])
    out[~small] = _bessel_k0_large(x[~small])
    return out


def _legacy_k1(x: np.ndarray) -> np.ndarray:
    """The hypothetical pre-fix ``K_1`` (same 9.0 switch, no quadrature)."""
    out = np.empty_like(x)
    small = x < 9.0
    out[small] = _bessel_k1_small(x[small])
    out[~small] = _bessel_k1_large(x[~small])
    return out


def _max_relative_error(values: np.ndarray, reference: np.ndarray) -> float:
    return float(np.max(np.abs(values / reference - 1.0)))


def _scan_errors() -> dict[str, Any]:
    """Branch-error curves on ``_SCAN`` against the Laplace-integral quadrature."""
    x = np.sort(_SCAN)
    reference_k0, reference_k1 = _bessel_k0_k1_quad(x)
    current_k0 = bessel_k0(x)
    current_k1 = bessel_k1(x)
    legacy_k0 = _legacy_k0(x)
    legacy_k1 = _legacy_k1(x)
    rows = [
        {
            "x": float(xi),
            "k0_relative_error": float(abs(a / r - 1.0)),
            "k1_relative_error": float(abs(b / q - 1.0)),
            "k0_legacy_relative_error": float(abs(c / r - 1.0)),
            "k1_legacy_relative_error": float(abs(d / q - 1.0)),
        }
        for xi, a, b, c, d, r, q in zip(x, current_k0, current_k1, legacy_k0, legacy_k1,
                                        reference_k0, reference_k1)
    ]
    worst_k0 = max(rows, key=lambda row: row["k0_relative_error"])
    worst_legacy = max(rows, key=lambda row: row["k0_legacy_relative_error"])
    return {
        "points": int(x.size),
        "reference": "Laplace-integral Gauss-Legendre quadrature (independent of both series)",
        "current_k0_max_relative_error": worst_k0["k0_relative_error"],
        "current_k0_worst_x": worst_k0["x"],
        "current_k1_max_relative_error": max(row["k1_relative_error"] for row in rows),
        "legacy_k0_max_relative_error": worst_legacy["k0_legacy_relative_error"],
        "legacy_k0_worst_x": worst_legacy["x"],
        "legacy_k1_max_relative_error": max(row["k1_legacy_relative_error"] for row in rows),
        "rows": rows,
    }


def bessel_accuracy() -> dict[str, Any]:
    """Accuracy of the ``K_0``/``K_1`` evaluator, before and after the fix."""
    pinned_x = np.array([row[0] for row in K_REFERENCE])
    pinned_k0 = np.array([row[1] for row in K_REFERENCE])
    pinned_k1 = np.array([row[2] for row in K_REFERENCE])
    current_k0 = bessel_k0(pinned_x)
    current_k1 = bessel_k1(pinned_x)
    legacy_k0 = _legacy_k0(pinned_x)
    legacy_k1 = _legacy_k1(pinned_x)

    table = [
        {
            "x": float(x),
            "k0_reference": float(k0_ref),
            "k1_reference": float(k1_ref),
            "k0_measured": float(k0_now),
            "k1_measured": float(k1_now),
            "k0_relative_error": float(abs(k0_now / k0_ref - 1.0)),
            "k1_relative_error": float(abs(k1_now / k1_ref - 1.0)),
            "k0_legacy_relative_error": float(abs(k0_old / k0_ref - 1.0)),
            "k1_legacy_relative_error": float(abs(k1_old / k1_ref - 1.0)),
        }
        for x, k0_ref, k1_ref, k0_now, k1_now, k0_old, k1_old in zip(
            pinned_x, pinned_k0, pinned_k1, current_k0, current_k1, legacy_k0, legacy_k1
        )
    ]

    # Branch seams: the quadrature must agree with both neighbours it replaces.
    seam = np.array(
        [
            _K0_SERIES_LIMIT - 1e-9,
            _K0_SERIES_LIMIT + 1e-9,
            _K_ASYMPTOTIC_LIMIT - 1e-9,
            _K_ASYMPTOTIC_LIMIT + 1e-9,
        ]
    )
    seam_k0, seam_k1 = _bessel_k0_k1_quad(seam)
    seam_table = []
    for value, quad_k0, quad_k1 in zip(seam, seam_k0, seam_k1):
        point = np.array([value])
        seam_table.append(
            {
                "x": float(value),
                "k0_quadrature": float(quad_k0),
                "k0_series": float(_bessel_k0_small(point)[0]),
                "k0_asymptotic": float(_bessel_k0_large(point)[0]),
                "k1_quadrature": float(quad_k1),
                "k1_series": float(_bessel_k1_small(point)[0]),
                "k1_asymptotic": float(_bessel_k1_large(point)[0]),
            }
        )

    scan = _scan_errors()
    return {
        "pinned_nodes": len(K_REFERENCE),
        "k0_max_relative_error": _max_relative_error(current_k0, pinned_k0),
        "k1_max_relative_error": _max_relative_error(current_k1, pinned_k1),
        "k0_legacy_max_relative_error": _max_relative_error(legacy_k0, pinned_k0),
        "k1_legacy_max_relative_error": _max_relative_error(legacy_k1, pinned_k1),
        "k0_worst_pinned_x": float(
            pinned_x[int(np.argmax(np.abs(legacy_k0 / pinned_k0 - 1.0)))]
        ),
        "table": table,
        "seam_table": seam_table,
        "series_limit": _K0_SERIES_LIMIT,
        "asymptotic_limit": _K_ASYMPTOTIC_LIMIT,
        "scan": scan,
        "reference": "mpmath.besselk, 40 significant digits",
    }


# ---------------------------------------------------------------------------
# 2. Analytic derivatives vs finite differences
# ---------------------------------------------------------------------------
def _central_gradient(targets: np.ndarray, sources: np.ndarray, step: float) -> np.ndarray:
    gradient = np.zeros((targets.shape[0], 3), dtype=complex)
    for axis in range(3):
        plus = targets.copy()
        minus = targets.copy()
        plus[:, axis] += step
        minus[:, axis] -= step
        gradient[:, axis] = (
            green_function_3d(plus, sources, NU, DEPTH)
            - green_function_3d(minus, sources, NU, DEPTH)
        ) / (2.0 * step)
    return gradient


def finite_difference_consistency() -> dict[str, Any]:
    """``nabla G`` against central and fourth-order Richardson differences."""
    analytic = green_gradient_3d(TARGETS, SOURCES, NU, DEPTH)
    scale = np.max(np.abs(analytic))
    ladder = []
    for step in (1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6):
        coarse = _central_gradient(TARGETS, SOURCES, step)
        fine = _central_gradient(TARGETS, SOURCES, 0.5 * step)
        richardson = (4.0 * fine - coarse) / 3.0
        ladder.append(
            {
                "step": step,
                "central_max_relative_error": float(np.max(np.abs(coarse - analytic)) / scale),
                "richardson_max_relative_error": float(
                    np.max(np.abs(richardson - analytic)) / scale
                ),
            }
        )
    best = min(row["richardson_max_relative_error"] for row in ladder)
    per_component = {
        name: float(np.max(np.abs(analytic[:, i] - _central_gradient(TARGETS, SOURCES, 1e-5)[:, i]))
                    / np.max(np.abs(analytic[:, i])))
        for i, name in enumerate(("dx", "dy", "dz"))
    }
    return {
        "points": int(TARGETS.shape[0]),
        "ladder": ladder,
        "best_richardson_max_relative_error": best,
        "per_component_relative_error_at_1e-5": per_component,
    }


def swap_identity() -> dict[str, Any]:
    """``nabla_x G(x, xi) == nabla_xi G(xi, x)`` (kernel symmetry)."""
    field = green_gradient_3d(TARGETS, SOURCES, NU, DEPTH)
    source = green_gradient_source_3d(SOURCES, TARGETS, NU, DEPTH)
    return {
        "max_absolute_difference": float(np.max(np.abs(field - source))),
        "max_relative_difference": float(
            np.max(np.abs(field - source)) / np.max(np.abs(field))
        ),
    }


# ---------------------------------------------------------------------------
# 3. Physical closures of the gradient
# ---------------------------------------------------------------------------
def boundary_closures() -> dict[str, Any]:
    """Free-surface / bottom closures and the horizontal curl of the gradient."""
    step = 1.0e-5
    surface = np.array([[0.7, 0.3, 0.0], [1.5, 1.0, 0.0], [2.0, -1.0, 0.0]])
    bottom = np.array([[0.7, 0.3, -DEPTH], [1.5, 1.0, -DEPTH], [2.0, -1.0, -DEPTH]])

    def d_dx_of_dz(points: np.ndarray) -> np.ndarray:
        plus = points.copy()
        minus = points.copy()
        plus[:, 0] += step
        minus[:, 0] -= step
        from tensorbem.finite_depth3d import green_gradient_z_3d

        return (
            green_gradient_z_3d(plus, SOURCES, NU, DEPTH)
            - green_gradient_z_3d(minus, SOURCES, NU, DEPTH)
        ) / (2.0 * step)

    gx_surface = green_gradient_3d(surface, SOURCES, NU, DEPTH)[:, 0]
    free_surface_residual = float(
        np.max(np.abs(d_dx_of_dz(surface) - NU * gx_surface)) / np.max(np.abs(NU * gx_surface))
    )
    bottom_residual = float(
        np.max(np.abs(d_dx_of_dz(bottom))) / np.max(np.abs(NU * gx_surface))
    )

    def d_dy_gx(points: np.ndarray) -> np.ndarray:
        plus = points.copy()
        minus = points.copy()
        plus[:, 1] += step
        minus[:, 1] -= step
        return (
            green_gradient_3d(plus, SOURCES, NU, DEPTH)[:, 0]
            - green_gradient_3d(minus, SOURCES, NU, DEPTH)[:, 0]
        ) / (2.0 * step)

    def d_dx_gy(points: np.ndarray) -> np.ndarray:
        plus = points.copy()
        minus = points.copy()
        plus[:, 0] += step
        minus[:, 0] -= step
        return (
            green_gradient_3d(plus, SOURCES, NU, DEPTH)[:, 1]
            - green_gradient_3d(minus, SOURCES, NU, DEPTH)[:, 1]
        ) / (2.0 * step)

    curl = d_dy_gx(TARGETS) - d_dx_gy(TARGETS)
    return {
        "free_surface_residual": free_surface_residual,
        "bottom_residual": bottom_residual,
        "horizontal_curl_relative": float(np.max(np.abs(curl)) / np.max(np.abs(gx_surface))),
        "step": step,
    }


# ---------------------------------------------------------------------------
# 4. Limits
# ---------------------------------------------------------------------------
def singularity_strength() -> dict[str, Any]:
    """``4 pi r^2 |nabla G| -> 1`` and the direction ``-> -r_hat``."""
    rows = []
    for radius in (1.0e-1, 5.0e-2, 2.5e-2, 1.0e-2, 2.5e-3):
        target = np.array([[radius, 0.0, -1.0]])
        source = np.array([0.0, 0.0, -1.0 - 1.0e-9])
        gradient = green_gradient_3d(target, source, NU, DEPTH)[0]
        magnitude = float(abs(np.linalg.norm(gradient)))
        unit = gradient / np.linalg.norm(gradient)
        rows.append(
            {
                "radius": radius,
                "four_pi_r2_gradient": float(4.0 * math.pi * radius * radius * magnitude),
                "direction_x": float(unit[0].real),
                "direction_z_imag": float(unit[2].imag),
            }
        )
    return {
        "rows": rows,
        "finest_deviation": abs(rows[-1]["four_pi_r2_gradient"] - 1.0),
    }


def deep_water_gradient_limit() -> dict[str, Any]:
    """Gradient collapse onto the independent deep-water Kelvin kernel."""
    rows = []
    for depth in (4.0, 8.0, 16.0, 32.0, 64.0, 128.0):
        k0 = finite_depth_wavenumber(NU, depth)
        finite = green_gradient_3d(TARGETS, SOURCES, NU, depth)
        deep = deep_water_gradient(TARGETS, SOURCES, NU)
        finite_kernel = green_function_3d(TARGETS, SOURCES, NU, depth)
        deep_kernel = deep_water_kernel(TARGETS, SOURCES, NU)
        rows.append(
            {
                "depth": depth,
                "k0h": float(k0 * depth),
                "gradient_relative_deviation": float(
                    np.max(np.abs(finite - deep)) / np.max(np.abs(deep))
                ),
                "kernel_relative_deviation": float(
                    np.max(np.abs(finite_kernel - deep_kernel)) / np.max(np.abs(deep_kernel))
                ),
            }
        )
    return {"rows": rows, "finest_gradient_deviation": rows[-1]["gradient_relative_deviation"]}


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------
THRESHOLDS = {
    "k0_max_relative_error": 1e-12,
    "k1_max_relative_error": 1e-12,
    "swap_relative_difference": 1e-12,
    "best_richardson_max_relative_error": 1e-7,
    "free_surface_residual": 1e-6,
    "bottom_residual": 1e-6,
    "horizontal_curl_relative": 1e-6,
    "singularity_deviation": 1e-3,
    "deep_water_gradient_deviation": 1e-5,
}


def run() -> dict[str, Any]:
    bessel = bessel_accuracy()
    finite_difference = finite_difference_consistency()
    swap = swap_identity()
    closures = boundary_closures()
    singularity = singularity_strength()
    deep_water = deep_water_gradient_limit()

    measured = {
        "k0_max_relative_error": bessel["k0_max_relative_error"],
        "k1_max_relative_error": bessel["k1_max_relative_error"],
        "swap_relative_difference": swap["max_relative_difference"],
        "best_richardson_max_relative_error": finite_difference[
            "best_richardson_max_relative_error"
        ],
        "free_surface_residual": closures["free_surface_residual"],
        "bottom_residual": closures["bottom_residual"],
        "horizontal_curl_relative": closures["horizontal_curl_relative"],
        "singularity_deviation": singularity["finest_deviation"],
        "deep_water_gradient_deviation": deep_water["finest_gradient_deviation"],
    }
    checks = [
        {
            "name": name,
            "measured": float(value),
            "threshold": THRESHOLDS[name],
            "pass": bool(value <= THRESHOLDS[name]),
        }
        for name, value in measured.items()
    ]
    failures = [check["name"] for check in checks if not check["pass"]]

    return {
        "meta": {
            "module": "tensorbem.benchmarks.finite_depth3d_gradient",
            "nu": NU,
            "depth": DEPTH,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
        },
        "bessel": bessel,
        "finite_difference": finite_difference,
        "swap_identity": swap,
        "boundary_closures": closures,
        "singularity": singularity,
        "deep_water": deep_water,
        "checks": checks,
        "verdict": {"pass": not failures, "failures": failures},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="JSON ledger path")
    parser.add_argument("--quiet", action="store_true", help="do not echo the ledger")
    args = parser.parse_args(argv)

    ledger = run()
    payload = json.dumps(ledger, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    if not args.quiet:
        print(payload)
    return 0 if ledger["verdict"]["pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())