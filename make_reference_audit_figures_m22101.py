"""Figures for m22101: independent audit of the cylinder case's reference chain.

Produces two figures under ``artifacts/m22101/``:

* ``audit_convergence.png`` -- the Cauchy-quadrature derivative of the declared
  complex potential converges *spectrally* to the machine floor as the number of
  contour nodes grows; this is what licenses using it as an independent referee
  for the transcribed closed forms.  A referee that had its own error would
  prove nothing.
* ``audit_surface.png`` -- surface pressure coefficient of the rotating cylinder
  (Gamma = 3), transcribed closed form vs the numerically differentiated
  potential, with the residual on a twin axis.

Run: PYTHONPATH=src python3 make_reference_audit_figures_m22101.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorbem import lifting_cylinder  # noqa: E402
from tensorbem.benchmarks import case_cylinder_stream as case  # noqa: E402

OUT = Path("artifacts/m22101")
NODES = (8, 12, 16, 20, 24, 32, 48, 64, 128, 256)
RADIUS_FRACTION = 0.05


def continued_log(z: np.ndarray, radius: float) -> np.ndarray:
    return np.log(np.abs(z / radius)) + 1j * np.unwrap(np.angle(z / radius))


def free_potential(z: np.ndarray) -> np.ndarray:
    return case.SPEED * (z + case.RADIUS**2 / z)


def lifting_potential(z: np.ndarray, circulation: float) -> np.ndarray:
    return (
        case.SPEED * (z + case.RADIUS**2 / z)
        - 1j * circulation / (2.0 * math.pi) * continued_log(z, case.RADIUS)
    )


def cauchy_derivative(potential, z: np.ndarray, nodes: int) -> np.ndarray:
    angles = 2.0 * math.pi * np.arange(nodes) / nodes
    out = np.empty((len(z), 2))
    for index, point in enumerate(z):
        radius = RADIUS_FRACTION * abs(point)
        values = np.asarray(potential(point + radius * np.exp(1j * angles)), complex)
        derivative = np.sum(values * np.exp(-1j * angles)) / (nodes * radius)
        out[index] = (derivative.real, -derivative.imag)
    return out


def figure_convergence() -> Path:
    probes = case.probe_points()
    z = probes[:, 0] + 1j * probes[:, 1]
    stated = case.exact_velocity(probes)
    lifting = lifting_cylinder.exact_velocity(
        probes, case.RADIUS, case.SPEED, case.CIRCULATION
    )
    free_error = []
    lift_error = []
    for nodes in NODES:
        free_error.append(
            float(np.max(np.abs(cauchy_derivative(free_potential, z, nodes) - stated)))
        )
        lift_error.append(
            float(
                np.max(
                    np.abs(
                        cauchy_derivative(
                            lambda zz: lifting_potential(zz, case.CIRCULATION), z, nodes
                        )
                        - lifting
                    )
                )
            )
        )
    free_error = np.array(free_error)
    lift_error = np.array(lift_error)

    fig, axes = plt.subplots(1, 1, figsize=(6.4, 4.2), constrained_layout=True)
    axes.semilogy(NODES, free_error, "o-", ms=4.5, label="free stream $W=U(z+a^2/z)$")
    axes.semilogy(
        NODES,
        lift_error,
        "s--",
        ms=4.0,
        label=r"lifting $W-i\Gamma/2\pi\,\ln(z/a)$, $\Gamma=3$",
    )
    axes.axhline(2.2e-16, color="0.5", ls=":", lw=1.0, label="double-precision eps")
    axes.set_xscale("log")
    axes.set_xlabel("Cauchy contour nodes")
    axes.set_ylabel(r"$\max\,|\,v_{\rm quadrature}-v_{\rm closed\ form}\,| / U$")
    axes.set_title("independent referee: spectral convergence to the round-off floor")
    axes.grid(True, which="both", alpha=0.3)
    axes.legend(fontsize=8, loc="upper right")
    output = OUT / "audit_convergence.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def figure_surface() -> tuple[Path, dict]:
    samples = 512
    theta = 2.0 * math.pi * np.arange(samples) / samples
    z = case.RADIUS * np.exp(1j * theta)
    stated_speed = lifting_cylinder.exact_surface_speed(
        theta, case.RADIUS, case.SPEED, case.CIRCULATION
    )
    stated_cp = 1.0 - (stated_speed / case.SPEED) ** 2
    numeric_velocity = cauchy_derivative(
        lambda zz: lifting_potential(zz, case.CIRCULATION), z, 256
    )
    numeric_speed = np.linalg.norm(numeric_velocity, axis=1)
    numeric_cp = 1.0 - (numeric_speed / case.SPEED) ** 2
    residual = np.abs(numeric_cp - stated_cp)
    radial = numeric_velocity[:, 0] * np.cos(theta) + numeric_velocity[:, 1] * np.sin(theta)

    fig, axes = plt.subplots(1, 1, figsize=(7.0, 4.2), constrained_layout=True)
    axes.plot(np.degrees(theta), stated_cp, "-", lw=2.2, label="closed form $1-(v_\\theta/U)^2$")
    axes.plot(
        np.degrees(theta),
        numeric_cp,
        "o",
        ms=2.2,
        mfc="none",
        color="tab:red",
        label="Cauchy-quadrature derivative",
    )
    axes.set_xlabel(r"$\theta$ [deg]")
    axes.set_ylabel("$C_p$")
    axes.set_xlim(0.0, 360.0)
    twin = axes.twinx()
    twin.semilogy(np.degrees(theta), residual + 1e-18, "-", color="0.35", lw=1.0)
    twin.set_ylabel("$|C_p^{num}-C_p^{closed}|$")
    axes.set_title(r"rotating cylinder $\Gamma=3$: transcribed vs re-derived $C_p$")
    axes.legend(fontsize=8, loc="lower right")
    axes.grid(True, alpha=0.3)
    output = OUT / "audit_surface.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    metrics = {
        "cp_max_abs_residual": float(np.max(residual)),
        "speed_max_abs_residual_over_U": float(np.max(np.abs(numeric_speed - stated_speed))),
        "max_radial_velocity_over_U": float(np.max(np.abs(radial))),
        "cp_min_stated": float(np.min(stated_cp)),
        "cp_min_numeric": float(np.min(numeric_cp)),
        "stagnation_stated_deg": [
            math.degrees(a)
            for a in lifting_cylinder.stagnation_angles(
                case.RADIUS, case.SPEED, case.CIRCULATION
            )
        ],
    }
    return output, metrics


if __name__ == "__main__":
    import json

    first = figure_convergence()
    second, metrics = figure_surface()
    print(json.dumps({"figures": [str(first), str(second)], "metrics": metrics}, indent=2))