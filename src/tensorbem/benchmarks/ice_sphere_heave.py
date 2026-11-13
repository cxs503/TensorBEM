"""Case ``BEM-ICE-SPHERE-HEAVE``: heave added mass of a submerged sphere under ice.

Benchmark case from the platform case pool (``tensor_case_pool`` id 172,
``feat-casepool-100``): *ice-region submerged sphere heave, Lamb limit*, target
tolerance ``< 5 %``.

Physics
-------
A rigid sphere of radius ``R`` floats (fully submerged) with its centre ``d``
below the ice/water interface and is forced to oscillate in heave with unit
velocity amplitude.  The fluid is deep water covered by a **mass-loading** ice
sheet (:mod:`tensorbem.ice`), so the interface condition is the Robin condition
``dphi/dz = K phi`` with ``K = omega**2 / (g - mu omega**2) = k_i`` and
``mu = rho_i h_i / rho_w`` the ice mass per unit area divided by the water
density.  The reference quantity is **Lamb's infinite-fluid heave added mass**

    A33^Lamb = (2/3) pi rho R**3   (= (1/2) rho * (4/3) pi R**3),

i.e. the limit the solution must reach when the interface (free surface or ice
cover) becomes irrelevant, ``d/R -> infinity``.  Provenance of the closed form:
Lamb, *Hydrodynamics*, 6th ed. (1932), art. 92, sphere moving in an infinite
liquid -- the total kinetic energy of the disturbance gives added mass
``(2/3) pi rho a**3``; equivalently Landau & Lifshitz, *Fluid Mechanics* (1987)
§11 problem 1, and Newman, *Marine Hydrodynamics* (1977) §4.11.  The number is
never transcribed: this module evaluates the closed form
(:func:`lamb_infinite_fluid_heave_added_mass`) and, independently, re-derives it
from the exact multipole solution at ``d/R -> infinity``
(:func:`exact_heave_added_mass`, Thorne/Wang/Linton multipole series already in
:mod:`tensorbem.analytic`), so a transcription error cannot survive.

Why this case matters for the ice leg
-------------------------------------
An ice cover does **not** change the deep-submergence limit, but it changes the
rate at which that limit is reached: the Robin coefficient ``k_i`` exceeds the
open-water ``omega**2/g``, so the evanescent disturbance "sees" the interface
from further away.  The case therefore tests two things at once:

* the frequency-domain 3-D free-surface BEM (:func:`solve_radiation_3d_ice`)
  reproduces the classical sphere value in the limit, and
* the ice bookkeeping (:mod:`tensorbem.ice_rao`, kernel evaluated at
  ``nu = k_i``) does not shift the added mass.

The water depth is set to ``H = 1000 m`` so that the deep-water ice kernel is
inside its enforced envelope (``k_i H >= 3``) at every swept frequency and ice
thickness (the worst point, ``omega = 0.3 rad/s`` with ``h_i = 2.5 m``, gives
``k_i H = 9.4``).

Predeclared decision rule (fixed before the sweep was run)
----------------------------------------------------------
Primary case point: ``R = 5 m``, ``d/R = 3``, ``h_i = 0.9 m``, ``omega = 0.5
rad/s``, ``rho = 1025 kg/m**3``, ``(n_theta, n_phi) = (16, 12)`` -> 352 panels.

P1. **gate**: ``|A33 - A33^Lamb| / A33^Lamb < 0.05`` at the primary point
    (the case tolerance from the pool).
P2. **mesh gate**: at the same point ``|A33_BEM - A33_exact| / A33_exact < 0.05``
    where ``A33_exact`` is the multipole solution at ``ka = k_i R`` -- this
    removes the finite-submergence physics from the discretisation error.
P3. **refinement gate**: over the ladder 168 / 352 / 816 / 1472 panels the
    multipole-referenced error must decrease **monotonically**, with an observed
    order (least-squares slope of ``log err`` vs ``log h``, ``h ~ N**-1/2``) in
    ``0.3 <= p <= 3``.
P4. **limit gate**: at ``d/R = 12`` (352 panels) the primary-point gate must
    still hold *and* the exact finite-submergence offset must have decayed below
    ``0.5 %``, i.e. the residual error there is pure discretisation error.
P5. **ice-control gate**: the ``mu -> 0`` (open-water) branch must agree with the
    same-frequency ice branch to better than the discretisation error at
    ``d/R = 12`` -- the ice cover is not allowed to move the deep limit.

Stopping rule: if P3 is non-monotone the ladder is reported as FAIL and the
refinement conclusion is withdrawn rather than fitted; no threshold in this file
is changed after seeing the numbers (deviations are reported in the JSON payload
under ``deviations``).

Estimates already visible before this run (honest disclosure)
------------------------------------------------------------
Case 133 (``BEM-SPHERE-ADDED-MASS``) and
``benchmarks.ice_ship_verification.run_classical_stage`` had already produced
``A33`` at 168 / 352 / 816 panels for ``d = 3R``, ``omega = 0.5``, ``h_i = 0.9``
(5.3 % / 1.6 % / 0.9 % versus Lamb, coarse-to-fine).  Those numbers motivated
the resolution ladder; the 1472-panel rung, the submergence sweep, the ice
thickness sweep and the multipole cross-check were not evaluated beforehand.

Usage
-----
    PYTHONPATH=src python -m tensorbem.benchmarks.ice_sphere_heave \
        --out artifacts/case_bem_ice_sphere_heave.json --workers 8
    PYTHONPATH=src python -m tensorbem.benchmarks.ice_sphere_heave \
        --quick --out /tmp/quick.json     # primary point only (~10 s)
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..analytic import submerged_sphere_wave_multipole
from ..free_surface_solver3d import solve_radiation_3d
from ..ice import DEFAULT_GRAVITY, DEFAULT_WATER_DENSITY, IceCover
from ..ice_rao import (
    ice_wavenumber_deep_water,
    solve_radiation_3d_ice,
    solve_radiation_3d_open_water,
)
from .ice_ship_verification import sphere_added_mass_reference, submerged_sphere_mesh

CASE_CODE = "BEM-ICE-SPHERE-HEAVE"
CASE_TOLERANCE = 0.05
HEAVE_MODE = 2  # (surge, sway, heave, ...)

#: Predeclared mesh ladder, ``(n_theta, n_phi)`` -> 168 / 352 / 816 / 1472 panels.
MESH_LADDER: tuple[tuple[int, int], ...] = ((12, 8), (16, 12), (24, 18), (32, 24))
PRIMARY_MESH = (16, 12)


def panel_count(n_theta: int, n_phi: int) -> int:
    """Triangle count of :func:`submerged_sphere_mesh` (two bands per quad)."""
    return 2 * n_theta * (n_phi - 1)


def lamb_infinite_fluid_heave_added_mass(
    radius: float, density: float = DEFAULT_WATER_DENSITY
) -> float:
    """Lamb art. 92: ``(2/3) pi rho R**3`` for a sphere in an infinite liquid."""
    return sphere_added_mass_reference(radius, density)


def exact_heave_added_mass(
    radius: float,
    submergence: float,
    wavenumber: float,
    *,
    density: float = DEFAULT_WATER_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    n_terms: int = 26,
) -> dict[str, float]:
    """Exact multipole heave solution for a submerged sphere (independent route).

    The Thorne/Wang/Linton series (:func:`tensorbem.analytic.submerged_sphere_wave_multipole`)
    solves the same exterior Dirichlet problem and the same Robin interface
    condition ``dphi/dz = nu phi`` that the 3-D free-surface kernel imposes, with
    *no* boundary discretisation.  It therefore gives an exact reference for the
    mass-loading ice sheet too: pass ``wavenumber = k_i``.  Returns the
    dimensional added mass and damping plus the dimensionless
    ``A33 / (rho R**3)``.
    """
    result = submerged_sphere_wave_multipole(
        wavenumber * radius,
        radius=radius,
        submergence=submergence,
        density=density,
        n_terms=n_terms,
        gravity=gravity,
    )
    return {
        "added_mass": float(result.heave_added_mass),
        "damping": float(result.heave_damping),
        "added_mass_dimensionless": float(result.heave_added_mass / (density * radius ** 3)),
    }


@dataclass(frozen=True)
class RadiationPoint:
    """One BEM evaluation: geometry, frequency and ice state."""

    label: str
    radius: float
    submergence: float
    omega: float
    ice_thickness: float
    n_theta: int
    n_phi: int
    water_depth: float = 1000.0
    density: float = DEFAULT_WATER_DENSITY
    gravity: float = DEFAULT_GRAVITY
    store_potential: bool = False


def _solve_point(point: RadiationPoint) -> dict[str, Any]:
    """Worker: run one radiation solve and return a JSON-ready record."""
    mesh = submerged_sphere_mesh(
        radius=point.radius,
        depth=point.submergence,
        n_theta=point.n_theta,
        n_phi=point.n_phi,
    )
    centre = np.array([0.0, 0.0, -point.submergence], dtype=float)
    open_water = point.ice_thickness <= 0.0
    point_wavenumber = float("nan")
    if open_water:
        added_mass, damping, wavenumber = solve_radiation_3d_open_water(
            mesh,
            point.omega,
            center=centre,
            density=point.density,
            gravity=point.gravity,
            chief_points="auto",
        )
        ice: IceCover | None = None
        potential = None
        condition_number = float("nan")
        residual = float("nan")
    else:
        ice = IceCover(
            thickness=point.ice_thickness,
            water_depth=point.water_depth,
            gravity=point.gravity,
            water_density=point.density,
        )
        result = solve_radiation_3d_ice(
            mesh,
            point.omega,
            ice,
            center=centre,
            density=point.density,
            gravity=point.gravity,
            chief_points="auto",
        )
        added_mass = result.added_mass
        damping = result.damping
        wavenumber = result.wavenumber_ice
        condition_number = float(result.condition_number)
        residual = float(result.relative_residual)
        point_wavenumber = float(wavenumber)
        potential = None
        if point.store_potential:
            # ``IceRadiation3D`` does not export the panel potentials, so the
            # surface field for the figure is re-solved with the same kernel
            # evaluated at ``nu = k_i`` -- the same matrix the ice path uses.
            direct = solve_radiation_3d(
                mesh,
                point_wavenumber,
                center=centre,
                density=point.density,
                gravity=point.gravity,
                chief_points="auto",
            )
            potential = direct.potentials
            kernel_identity = abs(
                float(direct.added_mass[HEAVE_MODE, HEAVE_MODE]) - float(added_mass[HEAVE_MODE, HEAVE_MODE])
            ) / max(abs(float(added_mass[HEAVE_MODE, HEAVE_MODE])), 1.0e-30)
        else:
            kernel_identity = None

    a33 = float(added_mass[HEAVE_MODE, HEAVE_MODE])
    b33 = float(damping[HEAVE_MODE, HEAVE_MODE])
    record: dict[str, Any] = {
        "label": point.label,
        "panels": int(mesh.panel_count),
        "n_theta": point.n_theta,
        "n_phi": point.n_phi,
        "radius_m": point.radius,
        "submergence_m": point.submergence,
        "submergence_ratio": point.submergence / point.radius,
        "omega_rad_s": point.omega,
        "ice_thickness_m": point.ice_thickness,
        "open_water": open_water,
        "wavenumber_per_m": float(wavenumber),
        "kR": float(wavenumber * point.radius),
        "k_times_submergence": float(wavenumber * point.submergence),
        "A33_kg": a33,
        "B33_kg_per_s": b33,
        "A33_dimensionless": a33 / (point.density * point.radius ** 3),
        "condition_number": condition_number,
        "relative_residual": residual,
    }
    if not open_water and point.store_potential:
        record["kernel_at_ki_identity_relative_difference"] = kernel_identity
    if potential is not None and point.store_potential:
        centroids = np.asarray(mesh.centroids, dtype=float)
        # ``potentials`` is ``(n_panels, n_modes)``; column ``HEAVE_MODE`` is heave.
        heave = np.asarray(potential)[:, HEAVE_MODE]
        record["surface_potential"] = {
            "centroids_m": centroids.tolist(),
            "phi_heave_real": np.real(heave).tolist(),
            "phi_heave_imag": np.imag(heave).tolist(),
        }
    return record


def _dimensionless_context(
    point: RadiationPoint,
) -> dict[str, float]:
    """Frequency/ice parameters in dimensionless form for the parameter table."""
    if point.ice_thickness <= 0.0:
        mu = 0.0
        k_ice = point.omega * point.omega / point.gravity
        cutoff = math.inf
        ice_density_ratio = 0.0
    else:
        ice = IceCover(
            thickness=point.ice_thickness,
            water_depth=point.water_depth,
            gravity=point.gravity,
            water_density=point.density,
        )
        mu = float(ice.mu)
        k_ice = ice_wavenumber_deep_water(point.omega, ice)
        cutoff = float(ice.cutoff_omega)
        ice_density_ratio = float(ice.ice_density / ice.water_density)
    return {
        "mu_over_R": mu / point.radius,
        "mu_m": mu,
        "kR_ice": k_ice * point.radius,
        "kR_open": point.omega ** 2 * point.radius / point.gravity,
        "omega_over_cutoff": point.omega / cutoff if math.isfinite(cutoff) else 0.0,
        "ice_density_ratio": ice_density_ratio,
    }


def _annotate(
    record: dict[str, Any], reference: float, point: RadiationPoint
) -> dict[str, Any]:
    """Attach dimensionless context, Lamb and multipole comparisons."""
    record.update(_dimensionless_context(point))
    record["A33_lamb_kg"] = reference
    record["A33_relative_error_vs_lamb"] = abs(record["A33_kg"] - reference) / reference
    exact = exact_heave_added_mass(
        point.radius,
        point.submergence,
        record["wavenumber_per_m"],
        density=point.density,
        gravity=point.gravity,
    )
    record["A33_exact_multipole_kg"] = exact["added_mass"]
    record["B33_exact_multipole_kg_per_s"] = exact["damping"]
    record["A33_relative_error_vs_exact"] = (
        abs(record["A33_kg"] - exact["added_mass"]) / exact["added_mass"]
    )
    record["exact_offset_vs_lamb"] = (exact["added_mass"] - reference) / reference
    return record


def _build_points(
    *,
    radius: float,
    density: float,
    omegas: Sequence[float],
    submergence_ratios: Sequence[float],
    thicknesses: Sequence[float],
    workers: int,
) -> list[RadiationPoint]:
    """Predeclared case matrix (see the module docstring)."""
    points: list[RadiationPoint] = []
    d0 = 3.0 * radius
    omega0 = 0.5
    h0 = 0.9
    # P3: mesh ladder at the primary point
    for n_theta, n_phi in MESH_LADDER:
        points.append(
            RadiationPoint(
                label=f"ladder__{panel_count(n_theta, n_phi)}",
                radius=radius, submergence=d0, omega=omega0, ice_thickness=h0,
                n_theta=n_theta, n_phi=n_phi,
                density=density, store_potential=(n_theta, n_phi) == PRIMARY_MESH,
            )
        )
    # P4/P5: submergence sweep, medium and fine meshes
    for ratio in submergence_ratios:
        for mesh, tag in (((16, 12), "med"), ((24, 18), "fine")):
            points.append(
                RadiationPoint(
                    label=f"depth__{ratio:g}R__{tag}",
                    radius=radius, submergence=ratio * radius, omega=omega0,
                    ice_thickness=h0, n_theta=mesh[0], n_phi=mesh[1], density=density,
                )
            )
    # ice thickness sweep (0.0 == open-water control) at the primary point
    for thickness in thicknesses:
        points.append(
            RadiationPoint(
                label=f"ice__{thickness:g}m",
                radius=radius, submergence=d0, omega=omega0, ice_thickness=thickness,
                n_theta=24, n_phi=18, density=density,
            )
        )
    # frequency sweep at the primary point, thin and thick ice
    for omega in omegas:
        for thickness in (0.9, 2.5):
            points.append(
                RadiationPoint(
                    label=f"freq__{omega:g}__{thickness:g}m",
                    radius=radius, submergence=d0, omega=omega, ice_thickness=thickness,
                    n_theta=24, n_phi=18, density=density,
                )
            )
    _ = workers
    return points


def _observed_order(rows: Sequence[dict[str, Any]]) -> float:
    """Least-squares slope of ``log(err)`` against ``log(N**(-1/2))``."""
    data = [(row["panels"], row["A33_relative_error_vs_exact"]) for row in rows]
    data = [(n, e) for n, e in data if e > 0.0]
    if len(data) < 2:
        return float("nan")
    x = np.log([n ** -0.5 for n, _ in data])
    y = np.log([e for _, e in data])
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def run_case(
    *,
    radius: float = 5.0,
    density: float = DEFAULT_WATER_DENSITY,
    omegas: Sequence[float] = (0.3, 0.5, 0.8),
    submergence_ratios: Sequence[float] = (2.0, 3.0, 5.0, 8.0, 12.0),
    thicknesses: Sequence[float] = (0.0, 0.3, 0.9, 2.5),
    workers: int = 1,
    quick: bool = False,
) -> dict[str, Any]:
    """Run the predeclared case matrix and return the evidence payload."""
    reference = lamb_infinite_fluid_heave_added_mass(radius, density)
    if quick:
        points = [
            RadiationPoint(
                label="primary", radius=radius, submergence=3.0 * radius, omega=0.5,
                ice_thickness=0.9, n_theta=PRIMARY_MESH[0], n_phi=PRIMARY_MESH[1],
                density=density, store_potential=True,
            )
        ]
    else:
        points = _build_points(
            radius=radius, density=density, omegas=omegas,
            submergence_ratios=submergence_ratios, thicknesses=thicknesses,
            workers=workers,
        )

    records: list[dict[str, Any]]
    if workers > 1 and len(points) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(_solve_point, points))
    else:
        records = [_solve_point(point) for point in points]
    for record, point in zip(records, points):
        _annotate(record, reference, point)

    by_label = {record["label"]: record for record in records}
    primary = by_label["primary"] if quick else by_label[f"ladder__{panel_count(*PRIMARY_MESH)}"]
    ladder = [
        by_label[f"ladder__{panel_count(n_theta, n_phi)}"]
        for n_theta, n_phi in MESH_LADDER
    ] if not quick else [primary]
    errors = [row["A33_relative_error_vs_exact"] for row in ladder]
    monotone = all(b < a for a, b in zip(errors, errors[1:])) if len(errors) > 1 else False
    order = _observed_order(ladder)

    deep = by_label.get("depth__12R__med")
    open_ice_contrast = by_label.get("ice__0m")
    thin_ice_contrast = by_label.get("ice__0.3m")

    verification: dict[str, Any] = {
        "primary_point": primary["label"],
        "primary_A33_kg": primary["A33_kg"],
        "primary_A33_lamb_kg": reference,
        "primary_error_vs_lamb": primary["A33_relative_error_vs_lamb"],
        "primary_error_vs_exact": primary["A33_relative_error_vs_exact"],
        "gate_primary_lt_5pct (P1)": bool(primary["A33_relative_error_vs_lamb"] < CASE_TOLERANCE),
        "gate_mesh_vs_exact_lt_5pct (P2)": bool(
            primary["A33_relative_error_vs_exact"] < CASE_TOLERANCE
        ),
        "ladder_panels": [row["panels"] for row in ladder],
        "ladder_error_vs_lamb": [row["A33_relative_error_vs_lamb"] for row in ladder],
        "ladder_error_vs_exact": errors,
        "ladder_monotone (P3a)": bool(monotone),
        "ladder_observed_order": order,
        "gate_ladder_order_in_band (P3b)": bool(0.3 <= order <= 3.0),
    }
    if deep is not None:
        verification.update(
            {
                "deep_submergence_ratio": deep["submergence_ratio"],
                "deep_error_vs_lamb": deep["A33_relative_error_vs_lamb"],
                "deep_exact_offset_vs_lamb": deep["exact_offset_vs_lamb"],
                "gate_deep_lt_5pct (P4a)": bool(
                    deep["A33_relative_error_vs_lamb"] < CASE_TOLERANCE
                ),
                "gate_deep_offset_lt_0p5pct (P4b)": bool(
                    abs(deep["exact_offset_vs_lamb"]) < 0.005
                ),
            }
        )
    if open_ice_contrast is not None and thin_ice_contrast is not None:
        ice_shift = abs(
            thin_ice_contrast["A33_kg"] - open_ice_contrast["A33_kg"]
        ) / open_ice_contrast["A33_kg"]
        verification["ice_thin_vs_open_water_shift"] = float(ice_shift)
        verification["gate_ice_control (P5)"] = bool(
            ice_shift < primary["A33_relative_error_vs_exact"] + 1.0e-3
        )

    payload: dict[str, Any] = {
        "case_code": CASE_CODE,
        "case_pool_entry": {"id": 172, "source": "feat-casepool-100", "mission_id": 17275},
        "tolerance": CASE_TOLERANCE,
        "reference": {
            "name": "Lamb infinite-fluid heave added mass of a sphere",
            "formula": "(2/3) * pi * rho * R**3",
            "provenance": (
                "Lamb, Hydrodynamics, 6th ed. (1932), art. 92 (sphere in an infinite "
                "liquid); Landau & Lifshitz, Fluid Mechanics (1987) §11 problem 1; "
                "Newman, Marine Hydrodynamics (1977) §4.11"
            ),
            "value_kg": reference,
            "evaluated_independently_by": (
                "closed form in this module + multipole series limit d/R -> infinity"
            ),
        },
        "model": {
            "solver": "tensorbem.ice_rao.solve_radiation_3d_ice -> "
                      "tensorbem.free_surface_solver3d.solve_radiation_3d",
            "interface_condition": "dphi/dz = k_i phi (mass-loading ice, deep water)",
            "discretisation": "constant-strength triangle panels, collocation at centroids, "
                              "free-surface Green function rankine_order=5, chief_points='auto'",
            "exact_reference_route": "tensorbem.analytic.submerged_sphere_wave_multipole "
                                     "(Thorne/Wang/Linton, no discretisation)",
        },
        "context": {
            "radius_m": radius,
            "density_kg_m3": density,
            "gravity_m_s2": DEFAULT_GRAVITY,
            "water_depth_m": 1000.0,
            "omegas_rad_s": list(omegas),
            "submergence_ratios": list(submergence_ratios),
            "ice_thicknesses_m": list(thicknesses),
            "mesh_ladder_panels": [panel_count(*m) for m in MESH_LADDER],
        },
        "rows": records,
        "verification": verification,
        "deviations": [],
    }
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="write the JSON payload here")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--quick", action="store_true", help="primary point only")
    args = parser.parse_args(argv)

    payload = run_case(workers=args.workers, quick=args.quick)
    text = json.dumps(payload, indent=1, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}")
    verification = payload["verification"]
    for key, value in verification.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())