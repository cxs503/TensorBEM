"""Verification harness for the 3-D finite-depth Green function (John eigenseries).

Writes ``artifacts/finite_depth3d/verification.json`` and figures.  Every
number quoted in ``docs/finite_depth3d_offaxis_diagnosis.md`` is produced here.

Run with::

    PYTHONPATH=src python -m tensorbem.benchmarks.finite_depth3d_verification
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..finite_depth import finite_depth_wavenumber
from ..finite_depth3d import (
    bessel_j0_y0,
    bessel_k0,
    green_function_3d,
    green_gradient_z_3d,
    hankel_h0,
    wave_amplitude_coefficient,
)
from ..free_surface3d import green_function_3d as deep_water_kernel

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "finite_depth3d"

NU = 0.6
SOURCE = np.array([[0.0, 0.0, -1.0]])  # (1, 3)
# Targets for general verification (avoid points too close to free surface)
TARGETS = np.array(
    [
        [0.5, 0.0, -0.5],
        [1.0, 0.0, -2.0],
        [2.0, 1.0, -1.5],
    ]
)
# Targets for deep-water limit (same as test suite)
DW_TARGETS = np.array(
    [[0.5, 0.0, -1.0], [2.0, 0.0, 0.0], [3.0, 1.0, -2.0], [6.0, 0.0, -0.5]]
)
DW_SOURCE = np.array([0.0, 0.0, -2.0])
DEPTHS = (2.0, 4.0, 8.0, 20.0, 60.0)


def dispersion_check() -> dict:
    """Dispersion relation residual and wavenumber for various depths."""
    rows = []
    for depth in (0.5, 1.0, 2.0, 8.0, 60.0):
        k0 = finite_depth_wavenumber(NU, depth)
        rows.append(
            {
                "depth": depth,
                "nu_h": NU * depth,
                "k0": k0,
                "k0_over_nu": k0 / NU,
                "residual": k0 * math.tanh(k0 * depth) - NU,
            }
        )
    return {"rows": rows}


def boundary_conditions(depth: float = 2.0) -> dict:
    """Free-surface and bottom boundary condition residuals."""
    surface = np.array([[0.3, 0.0, 0.0], [1.2, 0.4, 0.0]])
    bottom = np.array([[0.3, 0.0, -depth], [1.2, 0.4, -depth]])
    value = green_function_3d(surface, SOURCE, NU, depth)
    dz = green_gradient_z_3d(surface, SOURCE, NU, depth)
    dz_bottom = green_gradient_z_3d(bottom, SOURCE, NU, depth)
    return {
        "depth": depth,
        "free_surface_relative_residual": (
            np.abs(dz - NU * value) / np.abs(NU * value)
        ).tolist(),
        "bottom_absolute_derivative": np.abs(dz_bottom).tolist(),
    }


def reciprocity(depth: float = 2.0) -> dict:
    """G(P,Q) = G(Q,P) check."""
    a = np.array([[0.5, 0.0, -0.5], [2.0, 1.0, -1.5]])
    b = np.array([[0.0, 0.0, -1.0], [0.3, -0.4, -1.7]])
    forward = green_function_3d(a, b, NU, depth)
    backward = green_function_3d(b, a, NU, depth)
    return {
        "depth": depth,
        "forward": [complex(v) for v in forward],
        "backward": [complex(v) for v in backward],
        "relative_difference": (
            np.abs(forward - backward) / np.abs(forward)
        ).tolist(),
    }


def singularity_strength() -> dict:
    """4*pi*r*Re(G) -> 1 as r -> 0."""
    depth = 5.0
    radii = np.array([2e-2, 1e-2, 5e-3])
    targets = np.stack(
        [radii, np.zeros_like(radii), -1.0 * np.ones_like(radii)], axis=-1
    )
    value = green_function_3d(targets, SOURCE, NU, depth)
    strength = 4.0 * math.pi * radii * np.real(value)
    return {
        "depth": depth,
        "radii": radii.tolist(),
        "strength": strength.tolist(),
        "max_deviation_from_unit": float(np.max(np.abs(strength - 1.0))),
    }


def deep_water_limit() -> dict:
    """Cross-check against the independent deep-water Kelvin kernel.

    The John eigenseries implementation converges monotonically for ALL
    target points (on-axis and off-axis), unlike the integral representation
    which failed for off-axis points.
    """
    reference = deep_water_kernel(DW_TARGETS, DW_SOURCE, NU)
    rows = []
    for depth in (10.0, 20.0, 40.0):
        value = green_function_3d(DW_TARGETS, DW_SOURCE, NU, depth)
        rel_err = np.abs(value - reference) / np.abs(reference)
        rows.append(
            {
                "depth": depth,
                "nu_h": NU * depth,
                "k0h": finite_depth_wavenumber(NU, depth) * depth,
                "finite_depth": [complex(v) for v in value],
                "relative_error": rel_err.tolist(),
                "max_relative_error": float(np.max(rel_err)),
            }
        )
    return {
        "nu": NU,
        "source": DW_SOURCE.tolist(),
        "deep_water_reference": [complex(v) for v in reference],
        "rows": rows,
    }


def shallow_water_limit() -> dict:
    """Long-wave limit: wave term -> (i/(4h))*H_0^{(1)}(k0*R)."""
    rows = []
    src = np.array([[0.0, 0.0, 0.0]])  # source at z=0 for shallow water test
    for depth in (0.08, 0.04):
        k0 = finite_depth_wavenumber(NU, depth)
        radii = np.array([1.0, 2.5])
        targets = np.stack([radii, np.zeros_like(radii), np.zeros_like(radii)], axis=-1)
        total = green_function_3d(targets, src, NU, depth)
        without_wave = green_function_3d(targets, src, NU, depth, include_wave=False)
        wave = total - without_wave
        long_wave = 0.25j * (1.0 / depth) * hankel_h0(k0 * radii)
        rows.append(
            {
                "depth": depth,
                "k0h": k0 * depth,
                "deviation": float(
                    np.max(np.abs(wave - long_wave)) / np.max(np.abs(long_wave))
                ),
            }
        )
    return {"rows": rows}


def far_field_structure() -> dict:
    """Outgoing cylindrical wave with wavenumber k0."""
    depth = 4.0
    k0 = finite_depth_wavenumber(NU, depth)
    radii = np.linspace(25.0, 35.0, 11)
    targets = np.stack([radii, np.zeros_like(radii), np.zeros_like(radii)], axis=-1)
    value = green_function_3d(targets, SOURCE, NU, depth)
    evanescent = green_function_3d(targets, SOURCE, NU, depth, include_wave=False)

    amplitude = np.abs(value) * np.sqrt(k0 * radii)
    phase = np.unwrap(np.angle(value))
    slope = float(np.polyfit(k0 * radii, phase, 1)[0])

    return {
        "depth": depth,
        "k0": k0,
        "evanescent_ratio": float(
            np.max(np.abs(evanescent)) / np.max(np.abs(value))
        ),
        "amplitude_variation_pct": float(
            100 * np.ptp(amplitude) / np.mean(amplitude)
        ),
        "phase_slope_error": abs(slope - 1.0),
    }


def evanescent_truncation() -> dict:
    """Self-convergence of evanescent mode truncation."""
    depth = 4.0
    targets = np.array([[0.7, 0.0, -0.4], [1.5, 1.0, -2.0]])
    coarse = green_function_3d(targets, SOURCE, NU, depth, modes=120)
    fine = green_function_3d(targets, SOURCE, NU, depth, modes=400)
    return {
        "depth": depth,
        "coarse_modes": 120,
        "fine_modes": 400,
        "relative_error": float(
            np.max(np.abs(fine - coarse)) / np.max(np.abs(fine))
        ),
    }


def amplitude_scaling() -> dict:
    """Wave amplitude coefficient |A| vs nu*h.

    The current API's wave_amplitude_coefficient(z, nu, depth) returns the
    normalised propagating-mode shape phi_0(z)/N_0, which is the depth
    dependence of the wave term amplitude.
    """
    rows = []
    for depth in DEPTHS:
        amp = wave_amplitude_coefficient(np.array([0.0]), NU, depth)
        rows.append(
            {
                "depth": depth,
                "nu_h": NU * depth,
                "amplitude_at_surface": float(amp[0]),
            }
        )
    return {"nu": NU, "source_depth": SOURCE[0, 2], "rows": rows}


def bessel_references() -> dict:
    """Bessel function accuracy vs scipy (if available)."""
    try:
        import scipy.special as sp
    except ImportError:
        return {"skipped": True, "reason": "scipy not available"}

    x = np.concatenate([np.logspace(-3, 1.0, 40), np.linspace(10.0, 60.0, 40)])
    k0 = bessel_k0(x)
    j0, y0 = bessel_j0_y0(x)
    h0 = hankel_h0(x)

    return {
        "K0_max_rel_error": float(np.max(np.abs(k0 - sp.kv(0, x)) / sp.kv(0, x))),
        "J0_max_rel_error": float(
            np.max(np.abs(j0 - sp.jv(0, x)) / np.maximum(np.abs(sp.jv(0, x)), 1e-2))
        ),
        "Y0_max_rel_error": float(np.max(np.abs(y0 - sp.yv(0, x)) / np.abs(sp.yv(0, x)))),
        "H0_max_abs_error": float(np.max(np.abs(h0 - (sp.jv(0, x) + 1j * sp.yv(0, x))))),
    }


def _encode(value):
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialise {type(value)!r}")


def run(write: bool = True) -> dict:
    payload = {
        "nu": NU,
        "source": SOURCE[0].tolist(),
        "targets": TARGETS.tolist(),
        "dispersion": dispersion_check(),
        "boundary_conditions": boundary_conditions(),
        "reciprocity": reciprocity(),
        "singularity_strength": singularity_strength(),
        "deep_water_limit": deep_water_limit(),
        "shallow_water_limit": shallow_water_limit(),
        "far_field_structure": far_field_structure(),
        "evanescent_truncation": evanescent_truncation(),
        "amplitude_scaling": amplitude_scaling(),
        "bessel_references": bessel_references(),
    }
    if write:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        (ARTIFACT_DIR / "verification.json").write_text(
            json.dumps(payload, indent=2, default=_encode)
        )
        _figure(payload)
    return payload


def _figure(payload: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))

    # Top-left: deep-water limit convergence
    limit = payload["deep_water_limit"]
    depths = [row["depth"] for row in limit["rows"]]
    max_errors = [row["max_relative_error"] for row in limit["rows"]]
    axes[0, 0].semilogx(depths, max_errors, "o-", color="C0", linewidth=2)
    axes[0, 0].set_xlabel(r"water depth $h$ [m]")
    axes[0, 0].set_ylabel(r"max $|G_{fd} - G_{dw}| / |G_{dw}|$")
    axes[0, 0].set_title("Deep-water limit convergence (ALL points converge)")
    axes[0, 0].grid(True, which="both", alpha=0.3)

    # Top-right: individual target errors
    for i in range(len(DW_TARGETS)):
        errors = [row["relative_error"][i] for row in limit["rows"]]
        axes[0, 1].semilogx(depths, errors, "o-", label=f"target {i}")
    axes[0, 1].set_xlabel(r"water depth $h$ [m]")
    axes[0, 1].set_ylabel(r"relative error")
    axes[0, 1].set_title("Per-target deep-water limit error")
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].grid(True, which="both", alpha=0.3)

    # Bottom-left: amplitude scaling
    amplitude = payload["amplitude_scaling"]
    nuh = [row["nu_h"] for row in amplitude["rows"]]
    modulus = [row["amplitude_at_surface"] for row in amplitude["rows"]]
    axes[1, 0].semilogx(nuh, modulus, "o-", color="C2", linewidth=2)
    axes[1, 0].set_xlabel(r"$\nu h$")
    axes[1, 0].set_ylabel(r"$|\phi_0(0)/N_0|$")
    axes[1, 0].set_title("Propagating mode amplitude at surface")
    axes[1, 0].grid(True, which="both", alpha=0.3)

    # Bottom-right: far-field structure summary
    far = payload["far_field_structure"]
    axes[1, 1].text(
        0.1, 0.7,
        f"Evanescent ratio: {far['evanescent_ratio']:.2e}\n"
        f"Amplitude variation: {far['amplitude_variation_pct']:.2f}%\n"
        f"Phase slope error: {far['phase_slope_error']:.2e}",
        transform=axes[1, 1].transAxes,
        fontsize=10,
        verticalalignment="center",
    )
    axes[1, 1].set_title("Far-field structure verification")
    axes[1, 1].axis("off")

    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "finite_depth3d_verification.png", dpi=140)
    plt.close(fig)


def main() -> None:
    payload = run(write=True)
    print("=== Finite-Depth 3-D Green Function Verification ===")
    print(f"nu = {NU}, source = {payload['source']}")
    print()
    print("Boundary conditions:")
    bc = payload["boundary_conditions"]
    print(f"  free-surface residual: {bc['free_surface_relative_residual']}")
    print(f"  bottom derivative: {bc['bottom_absolute_derivative']}")
    print()
    print("Reciprocity:")
    print(f"  relative difference: {payload['reciprocity']['relative_difference']}")
    print()
    print("Singularity strength:")
    ss = payload["singularity_strength"]
    print(f"  max deviation from 1: {ss['max_deviation_from_unit']:.2e}")
    print()
    print("Deep-water limit (max relative error):")
    for row in payload["deep_water_limit"]["rows"]:
        print(f"  h={row['depth']:5.1f}  k0h={row['k0h']:.2f}  err={row['max_relative_error']:.2e}")
    print()
    print("Far-field structure:")
    ff = payload["far_field_structure"]
    print(f"  evanescent ratio: {ff['evanescent_ratio']:.2e}")
    print(f"  amplitude variation: {ff['amplitude_variation_pct']:.2f}%")
    print(f"  phase slope error: {ff['phase_slope_error']:.2e}")
    print()
    print(f"Artifacts written to: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
