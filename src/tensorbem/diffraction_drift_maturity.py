"""Frequency-domain diffraction/radiation and second-order drift: maturity surface.

This module is the *verification surface* (not a new solver) for the two halves of
the ``p41w3-n01`` brief that live in this repository:

1. **First order** -- wave exciting loads and heave/pitch RAO of the parabolic
   Wigley III hull, computed with the 3-D free-surface BEM of
   :mod:`tensorbem.free_surface_solver3d`, and checked by *three independent
   routes* for the exciting force:

   * near-field pressure integral of the total (incident + scattered) potential
     (:attr:`tensorbem.free_surface_solver3d.Diffraction3DResult.excitation_force`),
   * boundary-form Haskind integral over the radiation potentials,
   * far-field Kochin reciprocity ``F_j = rho g A K_j(beta + pi)``,

   together with the energy identity that ties the far-field Kochin function to
   the near-field radiation damping, the optical-theorem residual for the
   scattered field, and the low-frequency limits of the RAO.

2. **Second order** -- mean wave drift force on a fixed vertical circular
   cylinder: the exact MacCamy-Fuchs near-field/far-field pair of
   :mod:`tensorbem.cylinder_drift` (analytic reference, ``ka`` scan and the
   ``(ka)^3`` long-wave law) against the *same* quantity from a discretised BEM
   solution of the cylinder (:func:`cylinder_bem_case`), which also supplies the
   control-circle radius scan of :func:`tensorbem.drift_far_field3d.radius_scan`.

Design rules
------------
* Every number this module returns is computed, never typed in: the study driver
  (:mod:`tensorbem.benchmarks.bem_diffraction_drift_maturity`) only aggregates.
* A gate that cannot be evaluated is reported as ``evaluable = False`` with a
  reason; it is never replaced by an invented reference value.  In particular
  **no digitised towing-tank data set ships with this repository**, so the
  Wigley RAO comparison against experiment stays ``WITHHELD``.
* Mass and restoring matrices are assembled from closed-form hull integrals
  (:func:`wigley_hydrostatic_reference`), so the RAO check tests the BEM
  hydrodynamics and not a fitted mass model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from . import cylinder_drift as _cd
from .drift_far_field3d import (
    bem_kochin,
    maruo_far_field_drift,
    optical_theorem_residual,
    radius_scan,
)
from .free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    far_field_damping_matrix_3d,
    haskind_excitation_force_3d,
    solve_diffraction_3d,
    solve_radiation_3d,
)
from .geometry3d import WettedHullMesh, _oriented_triangle, _wall_strip
from .wigley_hydrostatics import wigley_hydrostatic_reference, wigley_wetted_hull_mesh

DEFAULT_DENSITY = 1025.0
DEFAULT_GRAVITY = 9.80665

#: Wigley III towing-tank geometry used by the repository's radiation campaign
#: (:mod:`tensorbem.benchmarks.cummins_wigley_hull`): L = 3 m, B = 0.3 m,
#: T = 0.1875 m, i.e. Cb = 4/9.
WIGLEY_L = 3.0
WIGLEY_B = 0.3
WIGLEY_T = 0.1875

#: Modes used by the 2-DOF heave/pitch motion solve (0=surge ... 5=yaw).
HEAVE, PITCH = 2, 4


# ---------------------------------------------------------------------------
# 1. Wigley III reference geometry, mass and restoring matrices
# ---------------------------------------------------------------------------
def wigley_reference(
    length: float = WIGLEY_L,
    beam: float = WIGLEY_B,
    draft: float = WIGLEY_T,
    *,
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
) -> dict[str, Any]:
    """Closed-form hydrostatics plus the 2x2 mass/restoring blocks of the hull.

    The Wigley hull ``y = (B/2)(1 - (2x/L)^2)(1 - (z/T)^2)`` has

    * ``nabla = 4LBT/9`` (so ``Cb = 4/9``), waterplane ``A_wp = 2LB/3``,
    * a longitudinal second moment of the displaced volume about the origin
      ``I_yy = rho T B L^3 / 45``, i.e. a uniform-density pitch radius of
      gyration ``k_yy = L/sqrt(20)``,

    so the pitch inertia about the centre of gravity ``z_G = -T/2`` is
    ``I55^G = I_yy - m z_G^2``.  Heave/pitch inertia coupling vanishes because
    ``x_G = y_G = 0``.

    **Frame convention.** The motion block returned here uses the *waterline
    origin* as the reference point, because that is the point about which
    :func:`tensorbem.free_surface_solver3d.solve_radiation_3d` and
    :func:`tensorbem.free_surface_solver3d.solve_diffraction_3d` express their
    added mass / damping / exciting moments, and the point about which
    :func:`tensorbem.wigley_hydrostatics.wigley_hydrostatic_reference` expresses
    ``C55`` (its ``z_B - z_G`` term already carries the CG height).  For a CG on
    the vertical through the origin the rigid-body block is therefore
    ``diag(m, I_yy)`` about the origin -- *not* the CG inertia
    ``I55^G = I_yy - m z_G^2``; the ``-m z_G^2`` term is exactly the parallel-axis
    transfer back to the origin.  Both radii of gyration are reported so that
    the choice is auditable.  Restoring terms are the analytic ``C33``, ``C55`` of
    :func:`tensorbem.wigley_hydrostatics.wigley_hydrostatic_reference`.
    """
    hydro = wigley_hydrostatic_reference(
        length, beam, draft, density=density, gravity=gravity
    )
    volume = float(hydro["volume"])
    mass = density * volume
    z_g = -0.5 * draft
    i_yy_origin = density * draft * beam * length**3 / 45.0
    i55_cg = i_yy_origin - mass * z_g**2
    mass_block = np.array([[mass, 0.0], [0.0, i_yy_origin]], dtype=float)
    # NOTE (naming trap, do not "simplify"): ``wigley_hydrostatic_reference``
    # stores ``C44 = rho g nabla GM_L`` and ``C55 = rho g nabla GM_T``, i.e. its
    # C44/C55 keys are swapped with respect to the roll/pitch convention used by
    # ``tensorbem.hydrostatic.compute_hydrostatic_restoring_matrix`` (there
    # ``C[3,3]`` roll carries GM_T and ``C[4,4]`` pitch carries GM_L -- pinned by
    # ``tests/test_wigley_hydrostatics.py``).  The physically correct pitch
    # restoring is therefore ``rho g nabla GM_L``, assembled here from ``GM_L``
    # instead of from the mislabelled key.
    c55 = float(density * gravity * volume * float(hydro["GM_L"]))
    restoring_block = np.array(
        [[float(hydro["C33"]), 0.0], [0.0, c55]], dtype=float
    )
    return {
        "length": float(length),
        "beam": float(beam),
        "draft": float(draft),
        "density": float(density),
        "gravity": float(gravity),
        "volume": volume,
        "Cb": volume / (length * beam * draft),
        "waterplane_area": float(hydro["waterplane_area"]),
        "mass": mass,
        "z_g": z_g,
        "pitch_inertia_about_origin": float(i_yy_origin),
        "pitch_inertia_about_cg": float(i55_cg),
        "pitch_radius_of_gyration": float(math.sqrt(i_yy_origin / mass)),
        "pitch_radius_of_gyration_cg": float(math.sqrt(i55_cg / mass)),
        "mass_block": mass_block,
        "restoring_block": restoring_block,
        "hydrostatic_reference": hydro,
        "modes": (HEAVE, PITCH),
    }


def geometry_gate(
    mesh: Any,
    reference: dict[str, Any],
    *,
    cb_tolerance: float = 2.0e-3,
    volume_tolerance: float = 2.0e-3,
) -> dict[str, Any]:
    """Precondition gate: panelised hull must reproduce ``Cb``, ``nabla``, ``A_wp``.

    A hull mesh that does not reproduce the block coefficient and the displaced
    volume cannot be used to judge hydrodynamics, so this gate runs first and is
    reported with the same weight as the physical gates.

    ``WettedHullMesh.volume`` is a midpoint (pyramid) quadrature of the wetted
    surface, so it under-reports the volume by ``O(h^2)``: 1.93 % on a 8x4 mesh
    and 0.122 % on the 32x16 mesh used by the study (ratio 15.8 ~ 16 = 4^2 for a
    factor-4 refinement, i.e. the deficit is pure panelisation).  The default
    tolerance is therefore 0.2 %, and the measured value is always reported.
    """
    volume = float(mesh.volume)
    waterplane = float(mesh.waterplane_area)
    ref_volume = float(reference["volume"])
    ref_wp = float(reference["waterplane_area"])
    cb_mesh = volume / (reference["length"] * reference["beam"] * reference["draft"])
    cb_ref = float(reference["Cb"])
    return {
        "panel_count": int(mesh.panel_count),
        "wetted_area": float(mesh.surface_area),
        "volume_mesh": volume,
        "volume_reference": ref_volume,
        "volume_rel_error": abs(volume - ref_volume) / ref_volume,
        "waterplane_mesh": waterplane,
        "waterplane_reference": ref_wp,
        "waterplane_rel_error": abs(waterplane - ref_wp) / ref_wp,
        "Cb_mesh": cb_mesh,
        "Cb_reference": cb_ref,
        "Cb_rel_error": abs(cb_mesh - cb_ref) / cb_ref,
        "passed": bool(
            abs(cb_mesh - cb_ref) / cb_ref <= cb_tolerance
            and abs(volume - ref_volume) / ref_volume <= volume_tolerance
        ),
    }


# ---------------------------------------------------------------------------
# 2. 2-DOF heave/pitch RAO from BEM coefficients
# ---------------------------------------------------------------------------
def rao_2dof(
    omega: float,
    added_mass: Any,
    damping: Any,
    mass_block: Any,
    restoring_block: Any,
    excitation: Any,
) -> dict[str, Any]:
    """Solve ``[-omega^2 (M + A) + i omega B + C] xi = F`` for heave and pitch.

    Parameters are the 2x2 ``(heave, pitch)`` blocks in the order used by
    :data:`HEAVE`, :data:`PITCH`.  ``excitation`` is the complex 2-vector
    ``(F_3, F_5)``.  Returns the complex displacements, the RAO amplitudes
    ``|xi_3|/A`` and ``|xi_5|/(k A)`` (``A`` is the unit incident amplitude and
    ``k`` is recovered from ``omega`` through the deep-water dispersion
    relation), and the linear-system residual as a fraction of ``|F|``.
    """
    om = float(omega)
    if not math.isfinite(om) or om <= 0.0:
        raise ValueError("omega must be positive and finite")
    a = np.asarray(added_mass, dtype=float)
    b = np.asarray(damping, dtype=float)
    m = np.asarray(mass_block, dtype=float)
    c = np.asarray(restoring_block, dtype=float)
    f = np.asarray(excitation, dtype=np.complex128)
    for name, mat in (("added_mass", a), ("damping", b), ("mass_block", m),
                      ("restoring_block", c)):
        if mat.shape != (2, 2):
            raise ValueError(f"{name} must be a 2x2 matrix")
    if f.shape != (2,):
        raise ValueError("excitation must be a complex 2-vector")
    dynamic = -om * om * (m + a) + 1.0j * om * b + c
    try:
        xi = np.linalg.solve(dynamic, f)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive
        raise ValueError("heave/pitch system is singular") from exc
    residual = float(np.linalg.norm(dynamic @ xi - f) / max(np.linalg.norm(f), 1.0e-300))
    wavenumber = om * om / DEFAULT_GRAVITY
    return {
        "omega": om,
        "wavenumber": wavenumber,
        "displacement": xi,
        "heave_rao": float(abs(xi[0])),
        "pitch_rao_per_ka": float(abs(xi[1]) / wavenumber),
        "pitch_rao_per_amplitude": float(abs(xi[1])),
        "heave_phase_deg": float(math.degrees(np.angle(xi[0]))),
        "pitch_phase_deg": float(math.degrees(np.angle(xi[1]))),
        "system_residual": residual,
        "dynamic_matrix": dynamic,
    }


# ---------------------------------------------------------------------------
# 3. One Wigley head-seas frequency: three-route exciting force + RAO
# ---------------------------------------------------------------------------
def wigley_head_seas_case(
    nu: float,
    reference: dict[str, Any],
    mesh: Any,
    *,
    amplitude: float = 1.0,
    n_azimuth: int = 120,
    operators: Any = None,
    density: float | None = None,
    gravity: float | None = None,
) -> dict[str, Any]:
    """One frequency of the Wigley head-seas verification surface.

    ``nu`` is the incident wavenumber ``k`` (rad/m).  The function returns

    * the three routes for ``F_3`` and ``F_5`` and their pairwise relative
      differences,
    * the near-field and far-field (Kochin) radiation damping for the two modes,
      i.e. the energy-conservation check of the Kochin function,
    * the optical-theorem residual of the scattered field,
    * the heave/pitch RAO of the 2-DOF motion solve.
    """
    k = float(nu)
    if not math.isfinite(k) or k <= 0.0:
        raise ValueError("nu must be positive and finite")
    rho = float(reference["density"] if density is None else density)
    g = float(reference["gravity"] if gravity is None else gravity)
    om = math.sqrt(g * k)
    if operators is None:
        operators = assemble_free_surface_operators_3d(mesh, k)

    diffraction = solve_diffraction_3d(
        mesh, k, amplitude=amplitude, density=rho, gravity=g, operators=operators
    )
    radiation = solve_radiation_3d(
        mesh, k, modes=(HEAVE, PITCH), density=rho, gravity=g, operators=operators
    )
    haskind = haskind_excitation_force_3d(
        mesh, radiation, amplitude=amplitude, density=rho, gravity=g
    )
    near = np.asarray(diffraction.excitation_force, dtype=np.complex128)[[HEAVE, PITCH]]
    boundary = np.asarray(haskind["boundary_haskind"], dtype=np.complex128)
    kochin_force = np.asarray(haskind["kochin_haskind"], dtype=np.complex128)

    def _rel(x: complex, y: complex) -> float:
        scale = max(abs(x), abs(y), 1.0e-300)
        return float(abs(x - y) / scale)

    routes = {
        # stored as [re, im] pairs so that the study JSON is plain JSON
        "near_field": [[v.real, v.imag] for v in near],
        "boundary_haskind": [[v.real, v.imag] for v in boundary],
        "kochin_haskind": [[v.real, v.imag] for v in kochin_force],
        "rel_near_vs_boundary": [_rel(near[i], boundary[i]) for i in range(2)],
        "rel_near_vs_kochin": [_rel(near[i], kochin_force[i]) for i in range(2)],
        "rel_boundary_vs_kochin": [_rel(boundary[i], kochin_force[i]) for i in range(2)],
    }

    azimuth = np.linspace(0.0, 2.0 * math.pi, int(n_azimuth), endpoint=False)
    kochin = bem_kochin(mesh, diffraction, n_azimuth=int(n_azimuth))
    b_far = far_field_damping_matrix_3d(mesh, radiation, density=rho, gravity=g,
                                        n_azimuth=int(n_azimuth))
    b_near = np.asarray(radiation.damping, dtype=float)
    energy = {
        "damping_near": b_near.tolist(),
        "damping_far": np.asarray(b_far, dtype=float).tolist(),
        "damping_rel_diff": (
            float(np.abs(np.asarray(b_far) - b_near).max()
                  / max(np.abs(b_near).max(), 1.0e-300))
        ),
        "kochin_azimuth_rel_change": float(kochin.azimuth_rel_change),
        "optical_theorem": optical_theorem_residual(
            kochin.kochin, kochin.azimuth, k, radius=0.5 * float(reference["beam"])
        ),
    }

    rao = rao_2dof(
        om,
        np.asarray(radiation.added_mass, dtype=float),
        b_near,
        reference["mass_block"],
        reference["restoring_block"],
        near,
    )
    rao.pop("dynamic_matrix", None)

    return {
        "wavenumber": k,
        "ka_beam": k * float(reference["beam"]) / 2.0,
        "kL": k * float(reference["length"]),
        "omega": om,
        "wavelength_over_length": 2.0 * math.pi / (k * float(reference["length"])),
        "panel_count": int(mesh.panel_count),
        "excitation_routes": routes,
        "added_mass": np.asarray(radiation.added_mass, dtype=float).tolist(),
        "damping": b_near.tolist(),
        "energy": energy,
        "rao": rao,
        "solver_relative_residual": float(
            max(diffraction.relative_residual, radiation.relative_residual)
        ),
    }


# ---------------------------------------------------------------------------
# 4. Cylinder: BEM drift vs the exact MacCamy-Fuchs reference
# ---------------------------------------------------------------------------
def vertical_cylinder_hull_mesh(
    radius: float,
    draft: float,
    *,
    n_theta: int = 48,
    n_levels: int = 7,
    grading: float = 2.5,
) -> "WettedHullMesh":
    """Closed wall-sided circular cylinder in ``z <= 0`` with graded rings.

    The rings sit at ``z_j = -draft * (j / n_levels) ** grading``, i.e. they are
    clustered towards the free surface, which is where the deep-water potential
    ``exp(k z)`` and the free-surface term of the Green function live.  Bottom
    rings only have to make the body watertight once
    ``exp(-2 k * draft)`` is negligible (see :func:`cylinder_bem_case`).

    Triangle orientation is delegated to the private helpers of
    :mod:`tensorbem.geometry3d` so that the outward-normal convention and the
    waterline-loop detection stay single-sourced with the other hull builders.
    """
    if not (radius > 0.0) or not (draft > 0.0):
        raise ValueError("radius and draft must be positive")
    n_t = int(n_theta)
    n_l = int(n_levels)
    if n_t < 8 or n_l < 1:
        raise ValueError("n_theta must be >= 8 and n_levels >= 1")
    angles = (2.0 * math.pi / n_t) * np.arange(n_t)
    coords: list[tuple[float, float, float]] = []
    rings: list[np.ndarray] = []
    for j in range(n_l + 1):
        z = -float(draft) * (j / n_l) ** float(grading)
        loop = []
        for ang in angles:
            loop.append(len(coords))
            coords.append((radius * math.cos(ang), radius * math.sin(ang), z))
        rings.append(np.asarray(loop, dtype=np.int64))
    faces: list[tuple[int, int, int]] = []
    for j in range(n_l):
        _wall_strip(rings[j + 1], rings[j], coords, faces)
    down = np.array([0.0, 0.0, -1.0])
    centre = len(coords)
    coords.append((0.0, 0.0, -float(draft)))
    bottom = rings[-1]
    for k in range(n_t):
        k2 = (k + 1) % n_t
        p0 = np.array(coords[bottom[k]], dtype=float)
        p1 = np.array(coords[bottom[k2]], dtype=float)
        p2 = np.array(coords[centre], dtype=float)
        order = _oriented_triangle(p0, p1, p2, down)
        if order is None:  # pragma: no cover - defensive
            raise ValueError("degenerate bottom triangle")
        tri = (int(bottom[k]), int(bottom[k2]), centre)
        faces.append(tuple(tri[order[i]] for i in range(3)))
    return WettedHullMesh(np.asarray(coords, dtype=float), np.asarray(faces, dtype=np.int64))


def cylinder_bem_case(
    ka: float,
    *,
    radius: float = 1.0,
    panels: int = 64,
    depth_factor: float = 16.0,
    n_theta: int = 48,
    n_levels: int = 7,
    grading: float = 2.5,
    n_azimuth: int = 96,
    density: float = DEFAULT_DENSITY,
    gravity: float = DEFAULT_GRAVITY,
    radii: Sequence[float] = (5.0, 7.5, 10.0, 15.0, 20.0),
) -> dict[str, Any]:
    """Discretised BEM solution of a deep, bottom-mounted circular cylinder.

    The cylinder is modelled as a wall-sided body of draft ``depth_factor * radius``
    so that the deep-water Green function sees the bottom through
    ``exp(-2 k H) <= exp(-2 * depth_factor * ka)`` -- at the default
    ``depth_factor = 16`` and ``ka = 1`` that is ``1e-14``, i.e. the finite draft
    is numerically invisible.  ``panels`` is the number of azimuthal panels
    (``n_theta``); the vertical grading and ring count are exposed separately
    because the free-surface region is what sets the accuracy.  Returns the Maruo far-field drift force from the
    BEM Kochin function, its control-circle radius scan, the optical-theorem
    residual, and the comparison against the exact MacCamy-Fuchs pair.
    """
    k = float(ka) / float(radius)
    mesh = vertical_cylinder_hull_mesh(
        radius, depth_factor * radius, n_theta=int(panels),
        n_levels=int(n_levels), grading=grading,
    )
    diffraction = solve_diffraction_3d(
        mesh, k, density=density, gravity=gravity, amplitude=1.0
    )
    kochin = bem_kochin(mesh, diffraction, n_azimuth=int(n_azimuth))
    far = maruo_far_field_drift(
        kochin.kochin, kochin.azimuth, k, density=density, gravity=gravity
    )
    scan = radius_scan(
        kochin.kochin, kochin.azimuth, k, radii=tuple(radii),
        density=density, gravity=gravity,
    )
    exact_near = _cd.near_field_cylinder_drift(
        ka, radius=radius, density=density, gravity=gravity
    )
    exact_far = _cd.far_field_cylinder_drift(
        ka, radius=radius, density=density, gravity=gravity
    )
    bem_force = float(far["force_x"])
    exact_force = float(exact_far["mean_drift_force_x"])
    return {
        "ka": float(ka),
        "radius": float(radius),
        "draft": depth_factor * radius,
        "panel_count": int(mesh.panel_count),
        "wavenumber": k,
        "bem_drift_force": bem_force,
        "exact_far_field_force": exact_force,
        "exact_near_field_force": float(exact_near["mean_drift_force_x"]),
        "bem_vs_exact_rel_error": abs(bem_force - exact_force) / abs(exact_force),
        "exact_near_vs_far_rel_error": (
            abs(exact_near["mean_drift_force_x"] - exact_force) / abs(exact_force)
        ),
        "radius_scan": scan,
        "kochin_azimuth_rel_change": float(kochin.azimuth_rel_change),
        "optical_theorem": optical_theorem_residual(
            kochin.kochin, kochin.azimuth, k, radius=radius
        ),
        "solver_relative_residual": float(diffraction.relative_residual),
    }


def cylinder_mesh_convergence(
    ka: float,
    *,
    panel_counts: Sequence[int] = (32, 64, 96),
    radius: float = 1.0,
    depth_factor: float = 16.0,
    n_theta: int = 48,
    n_levels: int = 7,
    grading: float = 2.5,
    n_azimuth: int = 96,
) -> dict[str, Any]:
    """Panel convergence of the BEM drift force against the exact value."""
    rows = []
    for n in panel_counts:
        case = cylinder_bem_case(
            ka, radius=radius, panels=int(n), depth_factor=depth_factor,
            n_levels=n_levels, grading=grading, n_azimuth=n_azimuth,
        )
        rows.append(
            {
                "panels": int(case["panel_count"]),
                "force": case["bem_drift_force"],
                "rel_error": case["bem_vs_exact_rel_error"],
            }
        )
    errors = [row["rel_error"] for row in rows]
    return {
        "ka": float(ka),
        "rows": rows,
        "error_ratio_last_two": (
            errors[-2] / errors[-1] if errors[-1] > 0.0 else float("inf")
        ),
        "converged": bool(errors[-1] <= errors[0]),
    }


# ---------------------------------------------------------------------------
# 5. Analytic cylinder gates (exact reference) used by the driver
# ---------------------------------------------------------------------------
def cylinder_energy_balance(
    ka: float = 1.0,
    *,
    probes: Sequence[float] = (5.0, 10.0, 20.0),
    radius: float = 1.0,
) -> dict[str, Any]:
    """Kochin energy-balance residual of the exact field at several probe radii.

    ``cylinder_drift.energy_balance`` evaluates the *un-approximated* flux of the
    exact MacCamy-Fuchs field on a probe circle (no stationary-phase step), so
    this is the sharpest available statement of brief item (c): a fixed rigid body
    does no work, hence ``flux_scattered + flux_interference = 0`` at **every**
    radius.  The spread of the residual over the probe radii is the
    radius-independence measure that the finite-radius *drift* route lacks.
    """
    rows = []
    for probe in probes:
        try:
            res = _cd.energy_balance(float(ka), radius=radius, radius_probe=float(probe))
        except ValueError as exc:  # k*R > 35: series accuracy limit, report it
            rows.append({"probe": float(probe), "evaluable": False, "reason": str(exc)})
            continue
        rows.append(
            {
                "probe": float(probe),
                "kR": float(res["probe_radius"]) * float(ka) / float(radius),
                "residual": float(res["residual"]),
                "pair_residual": float(res["pair_residual"]),
                "evaluable": True,
            }
        )
    values = [r["residual"] for r in rows if r["evaluable"]]
    return {
        "ka": float(ka),
        "rows": rows,
        "max_residual": max(values) if values else None,
        "residual_spread": (max(values) - min(values)) if values else None,
    }


def cylinder_asymptotic_gate(
    ka_scan: Sequence[float] = tuple(np.linspace(0.2, 2.0, 8)),
    ka_law: Sequence[float] = (0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3),
) -> dict[str, Any]:
    """``(ka)^3`` law, near/far agreement and long-wave coefficient of the exact pair."""
    rows = []
    for ka in ka_scan:
        near = _cd.near_field_cylinder_drift(float(ka))
        far = _cd.far_field_cylinder_drift(float(ka))
        rows.append(
            {
                "ka": float(ka),
                "near_field_force": float(near["mean_drift_force_x"]),
                "far_field_force": float(far["mean_drift_force_x"]),
                "rel_diff": abs(
                    near["mean_drift_force_x"] - far["mean_drift_force_x"]
                ) / abs(far["mean_drift_force_x"]),
            }
        )
    x = np.array([float(v) for v in ka_law], dtype=float)
    forces = np.array(
        [_cd.far_field_cylinder_drift(float(v))["mean_drift_force_x"] for v in ka_law]
    )
    slope, intercept = np.polyfit(np.log(x), np.log(forces), 1)
    closed_form = _cd.long_wave_drift_coefficient()
    normalised = forces / (_cd.DEFAULT_DENSITY * _cd.DEFAULT_GRAVITY * x**3)
    coefficient_error = float(abs(normalised[0] - closed_form) / closed_form)
    return {
        "scan_rows": rows,
        "max_near_vs_far_rel_diff": float(max(r["rel_diff"] for r in rows)),
        "law_slope": float(slope),
        "law_intercept": float(intercept),
        "closed_form_coefficient": float(closed_form),
        "coefficient_error_at_ka_min": coefficient_error,
        "law_rows": [
            {"ka": float(xi), "force": float(fi), "normalised": float(ni)}
            for xi, fi, ni in zip(x, forces, normalised)
        ],
    }


def wigley_mesh(n_sections: int = 40, n_levels: int = 12, **kwargs: Any) -> Any:
    """Convenience wrapper: Wigley III wetted hull mesh at the default geometry."""
    return wigley_wetted_hull_mesh(
        float(kwargs.get("length", WIGLEY_L)),
        float(kwargs.get("beam", WIGLEY_B)),
        float(kwargs.get("draft", WIGLEY_T)),
        n_sections,
        n_levels,
    )


@dataclass(frozen=True)
class StudyVerdict:
    """A named machine-checkable gate."""

    gate: str
    title: str
    threshold: str
    value: float | None
    passed: bool
    evaluable: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "title": self.title,
            "threshold": self.threshold,
            "value": self.value,
            "passed": bool(self.passed),
            "evaluable": bool(self.evaluable),
        }


def gate(gate_id: str, title: str, threshold: str, value: float | None,
         ok: bool | None) -> StudyVerdict:
    """Build a verdict; ``ok=None`` means *not evaluable* (never a silent pass)."""
    if ok is None:
        return StudyVerdict(gate_id, title, threshold, value, False, False)
    return StudyVerdict(gate_id, title, threshold, value, bool(ok))


__all__ = [
    "DEFAULT_DENSITY",
    "DEFAULT_GRAVITY",
    "HEAVE",
    "PITCH",
    "StudyVerdict",
    "WIGLEY_B",
    "WIGLEY_L",
    "WIGLEY_T",
    "cylinder_asymptotic_gate",
    "cylinder_bem_case",
    "cylinder_energy_balance",
    "cylinder_mesh_convergence",
    "gate",
    "geometry_gate",
    "rao_2dof",
    "wigley_head_seas_case",
    "wigley_mesh",
    "vertical_cylinder_hull_mesh",
    "wigley_reference",
]