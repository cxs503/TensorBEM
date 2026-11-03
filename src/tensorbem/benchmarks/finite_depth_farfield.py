"""Far-field closure of the finite-depth radiation problem (energy identity).

Why this module exists
----------------------
:mod:`tensorbem.finite_depth_solver` produces two independent estimates of the
same damping coefficient:

* the **near field** ``B_near = omega*rho*Im(force)``, i.e. the work done by the
  pressure on the body, and
* the **far field** ``B_far = 2*rho*g*c_g*|eta_a|**2``, i.e. twice the mean
  energy flux carried by the two outgoing waves, with ``eta_a`` the complex
  elevation amplitude obtained by projecting the source density onto the
  propagating mode.

``docs/finite_depth_radiation.md`` reports ``B_far/B_near = 1.0106`` for every
water depth at 64 panels and flags that 1.06 % gap as unexplained.  The design
question is whether that gap is

1. a **normalisation defect** - the factor 2, the group velocity ``c_g``, the
   mode norm ``N_0`` or the kinematic factor ``i*omega/g`` being wrong, in which
   case the ratio would settle on a constant != 1 and no refinement would move
   it, or
2. **discretisation error** - the panel density being first-order accurate, in
   which case the *near* and *far* functionals each carry their own ``O(1/N)``
   error and the gap must shrink under refinement, and must do so at the
   observed first order.

The two explanations are decided by refinement, not by argument.  This module
refines the mesh, fits the observed order of the gap, cross-checks the far-field
coefficient against a *direct* evaluation of the radiating potential, and
quantifies the panel quadrature error so that it can be excluded as the cause.

Additional independent anchors
------------------------------
* **Deep-water collapse.** At ``h/a = 50`` (``k0*h = 30``) the finite-depth
  solution must reproduce the deep-water Ursell reference
  ``B = 2718.1540 kg/(m*s)``: the difference between the two kernels is
  ``O(exp(-2*k0*h)) = O(1e-26)``.  The *analytical* value is therefore a
  legitimate target for the finite-depth near field, and the panel error can be
  measured against it directly.  The panel result converges to it from above
  while the method of fundamental solutions (MFS, no panels) already sits at
  0.03 %, which says the residual is panel discretisation and nothing else.
* **Direct far field.** ``eta_a`` is obtained in the solver by projecting the
  density onto the propagating mode.  Here the same amplitude is recovered
  *directly* from the finite-depth Green function evaluated on the free surface
  far from the body: if the propagating mode, its norm, or the kinematic factor
  were misnormalised, the two amplitudes would differ by exactly that factor.

Fault injection
---------------
A closure test is only worth what it rejects.  :func:`run` therefore also
reports the ratio that the *wrong* coefficients would produce - dropping the
factor 2 (one outgoing direction instead of two) or replacing the group
velocity by the phase velocity - so that the tolerances in
``tests/test_finite_depth_farfield.py`` can be shown to sit between the correct
value and the nearest wrong one.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from tensorbem.benchmarks.finite_depth_radiation import submerged_circle_mesh
from tensorbem.finite_depth import (
    evanescent_wavenumbers,
    finite_depth_wavenumber,
    green_function,
    group_velocity,
    phase_velocity,
    propagating_mode_norm,
)
from tensorbem.finite_depth_solver import (
    _solve_sources,
    radiation_reference,
    solve_radiation,
    wave_amplitude,
)
from tensorbem.free_surface_solver import (
    far_field_damping as deep_far_field_damping,
    solve_radiation_indirect,
)

GRAVITY = 9.80665
DENSITY = 1025.0
RADIUS = 1.0
SUBMERGENCE = 2.0

#: Deep-water Ursell reference (same geometry, ``ka = 0.6``), used as the
#: ``k0*h -> inf`` target for the finite-depth near field.  The value is
#: reproduced from ``tensorbem.analytic`` in :func:`_ursell_reference` rather
#: than copied, and the constant below only pins the expected key.
def _ursell_reference(
    nu: float, radius: float, density: float, *, submergence: float = SUBMERGENCE,
    gravity: float = GRAVITY,
) -> dict:
    """Deep-water Ursell added mass / damping, recomputed (not copied)."""
    from tensorbem.analytic import ursell_cylinder_added_mass

    omega = math.sqrt(nu * gravity)
    added_mass, damping = ursell_cylinder_added_mass(
        radius, submergence, omega, density, gravity=gravity
    )
    return {"added_mass": float(added_mass), "damping": float(damping),
            "omega": omega}


def _panel_quadrature(mesh, densities: np.ndarray, nodes: int) -> np.ndarray:
    """Return ``(sigma, weight)`` sampled on ``nodes`` Gauss points per panel.

    The solver stores a piecewise-constant density (one value per panel) and the
    far-field projection multiplies it by the mode shape at the *panel centre*.
    Sampling the density at Gauss points instead lets the same integral be
    evaluated to machine accuracy, which separates the quadrature error from the
    discretisation error of ``sigma`` itself.
    """
    abscissa, weights = np.polynomial.legendre.leggauss(nodes)
    points = []
    density = []
    total_weight = []
    for start, end, length in zip(mesh.vertices, mesh.ends, mesh.lengths):
        midpoint = 0.5 * (start + end)
        half_vector = 0.5 * (end - start)
        points.append(midpoint[None, :] + abscissa[:, None] * half_vector[None, :])
        total_weight.append(0.5 * length * weights)
    return np.vstack(points), np.concatenate(total_weight)


def _amplitude_with_quadrature(
    mesh,
    sources: np.ndarray,
    nu: float,
    depth: float,
    *,
    nodes: int,
    densities: np.ndarray | None = None,
    gravity: float = GRAVITY,
) -> complex:
    """Elevation amplitude with the projection integral taken to high order."""
    k0 = finite_depth_wavenumber(nu, depth)
    norm = propagating_mode_norm(k0, depth)
    points, weights = _panel_quadrature(mesh, sources, nodes)
    if densities is None:
        densities = np.repeat(sources, nodes)
    mode = np.cosh(k0 * (points[:, 1] + depth))
    integrand = mode * np.exp(-1j * k0 * points[:, 0]) * densities * weights
    coefficient = -1j / (2.0 * k0 * norm) * np.sum(integrand)
    omega = math.sqrt(nu * gravity)
    return 1j * omega / gravity * math.cosh(k0 * depth) * coefficient


def _direct_far_field_amplitude(
    mesh,
    sources: np.ndarray,
    nu: float,
    depth: float,
    *,
    distances: Sequence[float],
    nodes: int = 8,
    gravity: float = GRAVITY,
) -> np.ndarray:
    """Recover ``eta_a`` from the Green function on the free surface far away.

    With ``eta(x) = eta_a * exp(i*k0*x)`` for ``x -> +inf``, dividing the
    potential by the carrier ``exp(i*k0*x)`` leaves the amplitude.  The
    evanescent modes decay as ``exp(-mu_1*x)``; at ``x = 30/k0`` and beyond their
    residue is below ``1e-5`` for the depths used here, so the estimate is
    independent of the distance - which is also checked.
    """
    points, weights = _panel_quadrature(mesh, sources, 8)
    densities = np.repeat(sources, 8)
    omega = math.sqrt(nu * gravity)
    k0 = finite_depth_wavenumber(nu, depth)
    amplitudes = []
    for distance in distances:
        kernel = green_function(
            np.array([distance]),
            np.array([0.0]),
            points[:, 0][None, :],
            points[:, 1][None, :],
            nu,
            depth,
        )[0]
        potential = complex(np.sum(kernel * densities * weights))
        amplitudes.append(1j * omega / gravity * potential * np.exp(-1j * k0 * distance))
    return np.array(amplitudes)


@dataclass(frozen=True)
class CaseResult:
    """One (kind, depth, panel count) row of the closure table."""

    kind: str
    depth: float | None
    panels: int
    near: float
    far: float
    mfs_near: float | None
    ratio: float
    amplitude_rel_dev: float
    quadrature_rel_dev: float
    coefficient_variants: dict

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "depth": self.depth,
            "panels": self.panels,
            "near_damping": self.near,
            "far_damping": self.far,
            "far_over_near": self.ratio,
            "mfs_near_damping": self.mfs_near,
            "direct_amplitude_rel_dev": self.amplitude_rel_dev,
            "midpoint_vs_gauss_amplitude_rel_dev": self.quadrature_rel_dev,
            "wrong_coefficient_ratios": self.coefficient_variants,
        }


def _coefficient_variants(
    mesh, sources, nu, depth, near: float, *, density: float, gravity: float
) -> dict:
    """Ratio the identity would show under each mis-normalisation."""
    k0 = finite_depth_wavenumber(nu, depth)
    amplitude = wave_amplitude(mesh, sources, nu, depth, gravity=gravity)
    c_g = group_velocity(nu, depth, gravity=gravity)
    c_p = phase_velocity(nu, depth, gravity=gravity)
    squared = float(abs(amplitude) ** 2)
    return {
        # correct: two outgoing directions, group velocity
        "correct": 2.0 * density * gravity * c_g * squared / near,
        # one direction only
        "half_factor": density * gravity * c_g * squared / near,
        # phase velocity instead of group velocity
        "phase_velocity": 2.0 * density * gravity * c_p * squared / near,
        # deep-water c_g = omega/(2*k0) used in finite depth
        "deep_water_cg": 2.0 * density * gravity * (0.5 * c_p) * squared / near,
    }


def _observed_order(ratios: Sequence[float], panels: Sequence[int]) -> float:
    """Order ``p`` from ``|ratio-1| ~ C*N**-p`` over the last two levels."""
    if len(ratios) < 2:
        return float("nan")
    first, second = abs(ratios[-2] - 1.0), abs(ratios[-1] - 1.0)
    if first <= 0.0 or second <= 0.0:
        return float("nan")
    return math.log(first / second) / math.log(panels[-1] / panels[-2])


def _extrapolated_ratio(ratios: Sequence[float], panels: Sequence[int], order: float) -> float:
    """Richardson limit of ``ratio`` from the last two levels."""
    if len(ratios) < 2 or not math.isfinite(order) or order <= 0.0:
        return float("nan")
    factor = (panels[-2] / panels[-1]) ** order
    return (ratios[-1] - ratios[-2] * factor) / (1.0 - factor)


def run(
    *,
    panels: Iterable[int] = (64, 128, 256),
    depths: Iterable[float] = (3.2, 5.0, 50.0),
    ka: float = 0.6,
    radius: float = RADIUS,
    submergence: float = SUBMERGENCE,
    density: float = DENSITY,
    include_deep_water: bool = True,
    include_mfs: bool = True,
    gravity: float = GRAVITY,
) -> dict:
    """Refine the mesh and report the near/far-field closure of every case."""
    nu = ka / radius
    panels = list(panels)
    depths = list(depths)
    rows: list[CaseResult] = []
    for panel_count in panels:
        mesh = submerged_circle_mesh(panel_count, radius, submergence)
        for depth in depths:
            result, far = solve_radiation(
                mesh, nu, depth, "heave", density=density, gravity=gravity
            )
            sources, _system, _data, _potential = _solve_sources(
                mesh, nu, depth, "heave", 8, None
            )
            amplitude = wave_amplitude(mesh, sources, nu, depth, gravity=gravity)
            refined = _amplitude_with_quadrature(
                mesh, sources, nu, depth, nodes=16, gravity=gravity
            )
            # The evanescent modes decay as exp(-mu_1*x); putting the probes
            # at 12 and 24 decay lengths makes their residue O(1e-5), so the
            # recovered amplitude is the propagating mode alone.
            decay_length = 1.0 / float(evanescent_wavenumbers(nu, depth, 1)[0])
            direct = _direct_far_field_amplitude(
                mesh,
                sources,
                nu,
                depth,
                distances=[12.0 * decay_length, 24.0 * decay_length],
                gravity=gravity,
            )
            mfs = (
                radiation_reference(mesh, nu, depth, "heave", density=density,
                                    gravity=gravity).damping[1]
                if include_mfs else None
            )
            near = float(result.damping[1])
            rows.append(
                CaseResult(
                    kind="finite_depth",
                    depth=depth,
                    panels=panel_count,
                    near=near,
                    far=float(far),
                    mfs_near=None if mfs is None else float(mfs),
                    ratio=float(far) / near,
                    amplitude_rel_dev=float(np.max(np.abs(direct / amplitude - 1.0))),
                    quadrature_rel_dev=float(abs(refined / amplitude - 1.0)),
                    coefficient_variants=_coefficient_variants(
                        mesh, sources, nu, depth, near,
                        density=density, gravity=gravity,
                    ),
                )
            )
        if include_deep_water:
            deep = solve_radiation_indirect(mesh, nu, "heave")
            deep_far = deep_far_field_damping(
                mesh, deep.potential, deep.normal_velocity, nu, density, gravity=gravity
            )
            deep_near = float(deep.damping[1])
            rows.append(
                CaseResult(
                    kind="deep_water",
                    depth=None,
                    panels=panel_count,
                    near=deep_near,
                    far=float(deep_far),
                    mfs_near=None,
                    ratio=float(deep_far) / deep_near,
                    amplitude_rel_dev=float("nan"),
                    quadrature_rel_dev=float("nan"),
                    coefficient_variants={},
                )
            )

    # ---- convergence summary per case -------------------------------------
    reference = _ursell_reference(nu, radius, density, submergence=submergence,
                                 gravity=gravity)
    summary = {}
    for kind in sorted({row.kind for row in rows}):
        for depth in sorted({row.depth for row in rows if row.kind == kind},
                            key=lambda value: (value is None, value)):
            family = [row for row in rows if row.kind == kind and row.depth == depth]
            ratios = [row.ratio for row in family]
            counts = [row.panels for row in family]
            if len(family) < 2:
                continue
            order = _observed_order(ratios, counts)
            key = f"{kind}" if depth is None else f"{kind}@h={depth:g}"
            summary[key] = {
                "panels": counts,
                "far_over_near": ratios,
                "observed_order": order,
                "extrapolated_ratio": _extrapolated_ratio(ratios, counts, order),
                "near_error_vs_ursell_deep_water": [
                    row.near / reference["damping"] - 1.0 for row in family
                ],
                "mfs_near_damping": [row.mfs_near for row in family],
                "direct_amplitude_rel_dev": [row.amplitude_rel_dev for row in family],
                "max_quadrature_rel_dev": max(row.quadrature_rel_dev for row in family),
                "wrong_coefficient_ratios": family[-1].coefficient_variants,
            }

    return {
        "benchmark": "finite_depth_farfield_closure",
        "problem": {
            "geometry": "submerged circular cylinder, 2-D, heave",
            "radius": radius,
            "submergence": submergence,
            "ka": ka,
            "nu": nu,
            "density": density,
            "gravity": gravity,
            "depths": depths,
            "panels": panels,
        },
        "deep_water_ursell_reference": reference,
        "cases": [row.as_dict() for row in rows],
        "summary": summary,
    }


def _plot(payload: dict, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    colors = {"finite_depth": "C0", "deep_water": "C3"}
    for key, entry in payload["summary"].items():
        kind = "deep_water" if key.startswith("deep_water") else "finite_depth"
        panels = np.array(entry["panels"], dtype=float)
        residual = np.abs(np.array(entry["far_over_near"]) - 1.0)
        label = "deep water" if kind == "deep_water" else f"finite depth {key.split('@')[1]}"
        left.loglog(panels, residual, "o-", color=colors[kind], label=label)
        if len(panels) >= 2:
            order = entry["observed_order"]
            guide = residual[-1] * (panels / panels[-1]) ** (-order)
            left.loglog(panels, guide, ":", color=colors[kind], lw=1.0)
    left.set_xlabel("panels N")
    left.set_ylabel(r"$|B_{\rm far}/B_{\rm near} - 1|$")
    left.set_title("Energy-identity closure vs mesh refinement")
    left.grid(True, which="both", alpha=0.3)
    left.legend(fontsize=8)

    reference = payload["deep_water_ursell_reference"]["damping"]
    for key, entry in payload["summary"].items():
        if not key.startswith("finite_depth@h=50"):
            continue
        panels = np.array(entry["panels"], dtype=float)
        errors = np.array(entry["near_error_vs_ursell_deep_water"])
        right.loglog(panels, errors, "o-", color="C0", label="panel near field")
        mfs = [value for value in entry["mfs_near_damping"] if value is not None]
        if mfs:
            right.loglog(panels[: len(mfs)], np.abs(np.array(mfs) / reference - 1.0),
                         "s--", color="C2", label="MFS near field (no panels)")
    right.set_xlabel("panels N")
    right.set_ylabel(r"$|B_{\rm near}/B_{\rm Ursell}-1|$")
    right.set_title(r"$h/a=50$: collapse onto the deep-water reference")
    right.grid(True, which="both", alpha=0.3)
    right.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--panels", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--depths", type=float, nargs="+", default=[3.2, 5.0, 50.0])
    parser.add_argument("--ka", type=float, default=0.6)
    parser.add_argument("--no-mfs", action="store_true")
    parser.add_argument("--out", type=str,
                        default="outputs/finite_depth_farfield/closure.json")
    parser.add_argument("--figure", type=str,
                        default="outputs/finite_depth_farfield/closure.png")
    arguments = parser.parse_args(argv)
    payload = run(
        panels=arguments.panels,
        depths=arguments.depths,
        ka=arguments.ka,
        include_mfs=not arguments.no_mfs,
    )
    for path in (arguments.out, arguments.figure):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
    with open(arguments.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    _plot(payload, arguments.figure)
    print(json.dumps(payload["summary"], indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())