"""Verification harness for the finite-depth 3-D Green function (quadrature line).

Writes ``artifacts/finite_depth3d/verification.json`` and a figure.  Every
number quoted in ``docs/finite_depth3d_green_function.md`` is produced here.

Run with::

    PYTHONPATH=src python -m tensorbem.benchmarks.finite_depth3d_verification
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..finite_depth import finite_depth_wavenumber
from ..finite_depth3d_quadrature import (
    dispersion_derivative,
    green_function_finite_depth_3d,
    green_function_finite_depth_z_derivative_3d,
    wave_amplitude_coefficient,
)
from ..free_surface3d import green_function_3d

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "finite_depth3d"

NU = 0.6
SOURCE_DEPTH = -1.0
TARGETS = np.array(
    [
        [0.5, 0.0, -0.5],
        [1.0, 0.0, -2.0],
        [2.0, 1.0, -1.5],
        [0.0, 0.0, -0.2],
    ]
)
DEPTHS = (2.0, 4.0, 8.0, 20.0, 60.0)


def dispersion_check() -> dict:
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
                "slope": dispersion_derivative(NU, depth, k0),
            }
        )
    return {"rows": rows}


def boundary_conditions(depth: float = 2.0) -> dict:
    src = np.array([[0.0, 0.0, SOURCE_DEPTH]])
    surface = np.array([[0.3, 0.0, 0.0], [1.2, 0.4, 0.0]])
    bottom = np.array([[0.3, 0.0, -depth], [1.2, 0.4, -depth]])
    value = green_function_finite_depth_3d(surface, src, NU, depth)
    dz = green_function_finite_depth_z_derivative_3d(surface, src, NU, depth)
    dz_bottom = green_function_finite_depth_z_derivative_3d(bottom, src, NU, depth)
    return {
        "depth": depth,
        "free_surface_relative_residual": (
            np.abs(dz - NU * value) / np.abs(NU * value)
        ).tolist(),
        "bottom_absolute_derivative": np.abs(dz_bottom).tolist(),
    }


def grid_independence(depth: float = 8.0) -> dict:
    src = np.array([[0.0, 0.0, SOURCE_DEPTH]])
    rows = []
    reference = green_function_finite_depth_3d(TARGETS, src, NU, depth, nodes=16)
    for nodes in (8, 16, 24, 32):
        value = green_function_finite_depth_3d(TARGETS, src, NU, depth, nodes=nodes)
        rows.append(
            {
                "nodes": nodes,
                "real": value.real.tolist(),
                "imag": value.imag.tolist(),
                "max_abs_diff_vs_16": float(np.max(np.abs(value - reference))),
            }
        )
    return {"depth": depth, "rows": rows}


def reciprocity(depth: float = 2.0) -> dict:
    a = np.array([[0.5, 0.0, -0.5], [2.0, 1.0, -1.5]])
    b = np.array([[0.0, 0.0, SOURCE_DEPTH], [0.3, -0.4, -1.7]])
    forward = green_function_finite_depth_3d(a, b, NU, depth)
    backward = green_function_finite_depth_3d(b, a, NU, depth)
    return {
        "depth": depth,
        "forward": [complex(v) for v in forward],
        "backward": [complex(v) for v in backward],
        "relative_difference": (
            np.abs(forward - backward) / np.abs(forward)
        ).tolist(),
    }


def deep_water_limit() -> dict:
    """The cross-check that FAILS: reference is the verified deep-water kernel."""
    src = np.array([[0.0, 0.0, SOURCE_DEPTH]])
    reference = green_function_3d(TARGETS, src, NU)
    rows = []
    for depth in DEPTHS:
        value = green_function_finite_depth_3d(TARGETS, src, NU, depth)
        rows.append(
            {
                "depth": depth,
                "nu_h": NU * depth,
                "finite_depth": [complex(v) for v in value],
                "relative_error": (np.abs(value - reference) / np.abs(reference)).tolist(),
            }
        )
    return {
        "nu": NU,
        "source": SOURCE_DEPTH,
        "deep_water_reference": [complex(v) for v in reference],
        "rows": rows,
    }


def amplitude_scaling() -> dict:
    rows = []
    for depth in DEPTHS:
        rows.append(
            {
                "depth": depth,
                "nu_h": NU * depth,
                "amplitude_at_surface": complex(
                    wave_amplitude_coefficient(np.array([0.0]), SOURCE_DEPTH, NU, depth)[0]
                ),
            }
        )
    return {"nu": NU, "source_depth": SOURCE_DEPTH, "rows": rows}


def _encode(value):
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialise {type(value)!r}")


def run(write: bool = True) -> dict:
    payload = {
        "nu": NU,
        "source_depth": SOURCE_DEPTH,
        "targets": TARGETS.tolist(),
        "dispersion": dispersion_check(),
        "boundary_conditions": boundary_conditions(),
        "grid_independence": grid_independence(),
        "reciprocity": reciprocity(),
        "deep_water_limit": deep_water_limit(),
        "amplitude_scaling": amplitude_scaling(),
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

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    limit = payload["deep_water_limit"]
    depths = [row["depth"] for row in limit["rows"]]
    errors = np.array([row["relative_error"] for row in limit["rows"]])
    for index in range(errors.shape[1]):
        axes[0].semilogx(depths, errors[:, index], "o-", label=f"target {index}")
    axes[0].set_xlabel(r"water depth $h$ [m]")
    axes[0].set_ylabel(r"$|G_{finite} - G_{deep}| / |G_{deep}|$")
    axes[0].set_title(r"deep-water limit ($\nu h$ up to 36): FAILS")
    axes[0].set_ylim(0.0, 4.0)
    axes[0].legend(fontsize=7)
    axes[0].grid(True, which="both", alpha=0.3)

    amplitude = payload["amplitude_scaling"]
    nuh = [row["nu_h"] for row in amplitude["rows"]]
    modulus = [abs(row["amplitude_at_surface"]) for row in amplitude["rows"]]
    axes[1].semilogx(nuh, modulus, "o-")
    deep = 0.5 * NU * math.exp(NU * SOURCE_DEPTH)
    axes[1].axhline(deep, color="k", linestyle="--", label="deep-water limit")
    axes[1].set_xlabel(r"$\nu h$")
    axes[1].set_ylabel(r"$|A|$ at $z=0$")
    axes[1].set_title("radiated-wave amplitude coefficient")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "finite_depth3d_verification.png", dpi=140)
    plt.close(fig)


def main() -> None:
    payload = run(write=True)
    print("boundary conditions:",
          payload["boundary_conditions"]["free_surface_relative_residual"],
          payload["boundary_conditions"]["bottom_absolute_derivative"])
    print("grid independence:",
          [row["max_abs_diff_vs_16"] for row in payload["grid_independence"]["rows"]])
    print("reciprocity:", payload["reciprocity"]["relative_difference"])
    print("deep-water limit relative errors:")
    for row in payload["deep_water_limit"]["rows"]:
        print("   h =", row["depth"], row["relative_error"])


if __name__ == "__main__":
    main()