"""Programmatic audit metrics for m22101 (no hand-typed numbers anywhere).

Writes ``artifacts/m22101/audit_metrics.json`` with, for the declared complex
potentials of CASE-BEM-LAPLACE-CYLINDER:

* the Cauchy-quadrature derivative error vs the transcribed closed forms,
  for the free stream and for the lifting potential at every swept circulation;
* the same on the cylinder surface (speed, and the radial component that must
  vanish);
* the lift and drag implied by the pressure integral of the numerically
  differentiated velocity, against Joukowski's ``-rho U Gamma``;
* the bisection zeros of the numerical surface velocity, against the stated
  stagnation angles.

Run (light: <10 s, from the repo root): PYTHONPATH=src python3 audit_reference_m22101.py
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from tensorbem import lifting_cylinder
from tensorbem.benchmarks import case_cylinder_stream as case
from tests.test_case_cylinder_reference_independent import (
    _cauchy_velocity,
    _free_stream_potential,
    _lifting_potential,
    _surface_angles,
)

OUT = Path("artifacts/m22101/audit_metrics.json")


def _stagnation_from_bisection(circulation: float) -> list[float]:
    def tangential(angle: float) -> float:
        velocity = _cauchy_velocity(
            lambda zz: _lifting_potential(zz, case.RADIUS, case.SPEED, circulation),
            np.array([case.RADIUS * np.exp(1j * angle)]),
        )[0]
        return float(velocity @ np.array([-math.sin(angle), math.cos(angle)]))

    grid = np.linspace(0.0, 2.0 * math.pi, 721)
    values = [tangential(angle) for angle in grid]
    zeros: list[float] = []
    for index in range(len(grid) - 1):
        if values[index] * values[index + 1] > 0.0:
            continue
        low, high = grid[index], grid[index + 1]
        for _ in range(200):
            mid = 0.5 * (low + high)
            if tangential(low) * tangential(mid) <= 0.0:
                high = mid
            else:
                low = mid
            if high - low < 1e-13:
                break
        zeros.append(0.5 * (low + high))
    return [math.degrees(value) for value in zeros]


def main() -> None:
    started = time.time()
    probes = case.probe_points()
    z = probes[:, 0] + 1j * probes[:, 1]
    free_stated = case.exact_velocity(probes)
    free_numeric = _cauchy_velocity(
        lambda zz: _free_stream_potential(zz, case.RADIUS, case.SPEED), z
    )
    theta = _surface_angles(512)
    z_surface = case.RADIUS * np.exp(1j * theta)

    metrics: dict[str, object] = {
        "case_id": case.CASE_ID,
        "radius": case.RADIUS,
        "speed": case.SPEED,
        "probe_count": int(len(probes)),
        "free_stream": {
            "velocity_max_abs_error_over_U": float(
                np.max(np.abs(free_stated - free_numeric)) / case.SPEED
            ),
            "perturbation_potential_max_abs_error": float(
                np.max(
                    np.abs(
                        case.exact_perturbation_potential(probes)
                        - (
                            _free_stream_potential(z, case.RADIUS, case.SPEED).real
                            - case.SPEED * probes[:, 0]
                        )
                    )
                )
                / case.SPEED
            ),
            "cp_max_abs_residual": float(
                np.max(
                    np.abs(
                        case.exact_pressure_coefficient(probes)
                        - (1.0 - np.sum(free_numeric**2, axis=1) / case.SPEED**2)
                    )
                )
            ),
        },
        "lifting": [],
        "duration_of_audit_seconds": None,
    }

    for circulation in case.CIRCULATION_SWEEP:
        lifting_stated = lifting_cylinder.exact_velocity(
            probes, case.RADIUS, case.SPEED, circulation
        )
        lifting_numeric = _cauchy_velocity(
            lambda zz: _lifting_potential(zz, case.RADIUS, case.SPEED, circulation), z
        )
        surface_numeric = _cauchy_velocity(
            lambda zz: _lifting_potential(zz, case.RADIUS, case.SPEED, circulation),
            z_surface,
        )
        surface_speed_numeric = np.linalg.norm(surface_numeric, axis=1)
        surface_speed_stated = lifting_cylinder.exact_surface_speed(
            theta, case.RADIUS, case.SPEED, circulation
        )
        radial = surface_numeric[:, 0] * np.cos(theta) + surface_numeric[:, 1] * np.sin(
            theta
        )
        pressure = 0.5 * case.SPEED**2 - 0.5 * surface_speed_numeric**2
        weight = case.RADIUS * 2.0 * math.pi / len(theta)
        lift = float(-np.sum(pressure * np.sin(theta)) * weight)
        drag = float(-np.sum(pressure * np.cos(theta)) * weight)
        joukowski = lifting_cylinder.joukowski_lift(
            case.RADIUS, case.SPEED, circulation
        )
        stated_angles = sorted(
            math.degrees(angle)
            for angle in lifting_cylinder.stagnation_angles(
                case.RADIUS, case.SPEED, circulation
            )
        )
        found_angles = sorted(_stagnation_from_bisection(circulation))
        metrics["lifting"].append(
            {
                "circulation": circulation,
                "velocity_max_abs_error_over_U": float(
                    np.max(np.abs(lifting_stated - lifting_numeric)) / case.SPEED
                ),
                "surface_speed_max_abs_error_over_U": float(
                    np.max(np.abs(surface_speed_stated - surface_speed_numeric))
                    / case.SPEED
                ),
                "surface_radial_velocity_over_U": float(
                    np.max(np.abs(radial)) / case.SPEED
                ),
                "lift_from_pressure_integral": lift,
                "lift_joukowski": joukowski,
                "lift_relative_error": abs(lift - joukowski) / abs(joukowski),
                "drag_from_pressure_integral": drag,
                "drag_over_lift": abs(drag) / abs(joukowski),
                "stagnation_stated_deg": stated_angles,
                "stagnation_bisection_deg": found_angles,
                "stagnation_max_deviation_deg": max(
                    min(abs(((a - b + 180.0) % 360.0) - 180.0) for b in found_angles)
                    for a in stated_angles
                )
                if len(found_angles) == len(stated_angles) == 2
                else None,
            }
        )

    metrics["duration_of_audit_seconds"] = round(time.time() - started, 3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()