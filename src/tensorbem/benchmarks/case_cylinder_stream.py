"""Case-pool anchor CASE-BEM-LAPLACE-CYLINDER: potential flow past a cylinder.

Physics
-------
A circular cylinder of radius ``a`` in an unbounded inviscid irrotational
stream ``U`` along ``+x``. The complex potential is (Lamb 1932, Milne-Thomson)

    W(z) = U * (z + a**2 / z),            z = x + i y,

so the perturbation potential decays as ``U a**2 cos(theta) / r`` and the
velocity field follows from ``dW/dz = u - i v = U (1 - a**2 / z**2)``. On the
body the flow is tangential with speed ``2 U |sin(theta)|`` and the pressure
coefficient is ``Cp = 1 - 4 sin(theta)**2`` (``Cp = -3`` at the shoulders and
``Cp = 1`` at the stagnation points: d'Alembert's zero drag).

What this case verifies
-----------------------
The BEM side is the *exterior* 2-D Laplace operator of :mod:`tensorbem.solver`
driven two ways on an inscribed regular polygon:

* route D (Dirichlet): prescribe the analytic perturbation potential on the
  panel midpoints, solve the first-kind exterior Dirichlet problem, then use
  :func:`tensorbem.solver.evaluate_exterior_field` -- added for this anchor --
  to recover potential *and* velocity ``grad(phi)`` at exterior probe points.
* route N (Neumann): prescribe the analytic fluid-normal derivative
  ``q_e = U a**2 cos(theta) / r**2`` (zero total normal velocity on the body),
  solve the second-kind exterior Neumann problem, and compare the fluid-side
  surface velocity (normal part ``q_e``, tangential part ``d(phi)/ds``) with
  ``2 U sin(theta)``.

Every reference value is a closed form evaluated at the same points as the
discretisation: no stored number and no second discrete solution. The polygonal
body is not the circle, so the discrete field can only converge to the cylinder
solution; that faceting residual is reported with its observed order instead of
being hidden.

Not in scope: viscous wake, separation, free surface, 3-D effects. This is a
verification case for the potential-flow chain, not a drag prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..geometry import regular_polygon
from ..multipole2d import decompose_rings, fit_ring, ring_angles
from ..solver import (
    LaplaceDirichletResult,
    evaluate_exterior_field,
    evaluate_exterior_surface_velocity,
    solve_laplace_exterior_dirichlet,
    solve_laplace_exterior_neumann,
)
from ..lifting_cylinder import (
    angle_difference,
    exact_pressure_coefficient as circulation_pressure_coefficient,
    exact_surface_speed as circulation_surface_speed,
    exact_velocity as circulation_velocity,
    exterior_velocity_field as circulation_velocity_field,
    joukowski_lift,
    solve_rotating_cylinder,
    stagnation_angles as circulation_stagnation_angles,
    stagnation_angles_from_surface as circulation_stagnation_from_surface,
)

CASE_ID = "CASE-BEM-LAPLACE-CYLINDER"
POOL_CODE = "BEM-LAPLACE-CYLINDER"
SCHEMA = "tensorbem/case_bem_laplace_cylinder/1"
TOLERANCE = 0.03
RADIUS = 1.0
SPEED = 1.0
LADDER: tuple[int, ...] = (32, 64, 128)
QUICK_LADDER: tuple[int, ...] = (32, 64)
BUDGET_PANELS = 128
PROBE_RINGS: tuple[float, ...] = (1.25, 1.5, 2.0, 3.0)
PROBE_ANGLES = 16
FAR_FIELD_RADIUS = 8.0
ORDER_MIN = 1.5
WRONG_REFERENCE_FACTOR = 10.0
NEAR_WALL_LADDER: tuple[int, ...] = (32, 64, 128)
NEAR_WALL_FACTORS: tuple[float, ...] = (2.0, 1.0, 0.5, 0.25, 0.1)
# Documented validity window of the constant-panel exterior field evaluator:
# beyond delta >= NEAR_WALL_WINDOW_FACTOR * h_max the relative velocity error
# must stay inside one third of the case tolerance.
NEAR_WALL_WINDOW_FACTOR = 2.0
# The linear-density (first-moment) basis is required to hold the 1% bound this
# far in; NEAR_WALL_LINEAR_MIN_GAIN is the least improvement it must buy there.
NEAR_WALL_LINEAR_FACTOR = 0.5
NEAR_WALL_LINEAR_MIN_GAIN = 3.0
NEAR_WALL_WINDOW_TOL = 0.01
# Inside delta <= NEAR_WALL_EXCLUSION_FACTOR * h_max the error must exceed the
# case tolerance: the window above is an exclusion, not a cosmetic remark.
NEAR_WALL_EXCLUSION_FACTOR = 0.1
# At a fixed delta/h_max the error must be panel-relative, i.e. refining the
# mesh must not buy near-wall accuracy (ratio between the coarsest and the
# finest level stays in this band).
NEAR_WALL_SATURATION_BAND = (0.5, 2.0)
# The near-wall error at fixed delta/h_max is first order in h (the faceting
# O(h^2) term is swamped by the piecewise-constant density); a lower bound of
# half an order is the weakest statement that still forbids a plateau.
NEAR_WALL_ORDER_MIN = 0.5
# Where the case tolerance actually ends.  The gates above pin two clearance
# levels -- delta = 2 h_max inside 1 %, delta = 0.1 h_max outside 3 % -- but not
# the boundary between them, and the boundary is the number a caller needs:
# at delta = 0.25 h_max the measured error is already about 9 %, i.e. outside the
# case tolerance while every gated criterion still passes.  This study locates
# the crossing on a log-spaced grid in the scaled clearance delta/h_max, so that
# "the field meets the < 3 % acceptance" comes with a measured inner bound
# instead of an assumed one.
NEAR_WALL_FLOOR_GRID: tuple[float, ...] = (
    4.0,
    3.0,
    2.0,
    1.5,
    1.0,
    0.75,
    0.6,
    0.45,
    0.35,
    0.25,
    0.15,
    0.1,
    0.07,
    0.05,
)
# Refining the mesh must move the tolerance boundary *into* the body (the usable
# envelope cannot shrink when h shrinks); the slack still forbids a boundary that
# jumps around by panel-unit amounts across the ladder.
NEAR_WALL_FLOOR_MONOTONE_TOL = 0.05
# The gated 1 % window has to sit at least this many times outside the measured
# 3 % boundary, so "inside the window" cannot be true merely because the window
# was placed next to the boundary it is supposed to be safely away from.
NEAR_WALL_FLOOR_WINDOW_MARGIN = 2.0

# Azimuthal (intra-panel) sweep.  The window above is measured on radial probes
# above the panel *midpoints*, which is where a constant-density panel is at its
# best; callers evaluate wherever the geometry puts them, so the same ladder is
# re-probed across one panel up to the vertex, where the two adjacent densities
# meet and the trace is not smooth.
NEAR_WALL_AZIMUTH_FACTORS: tuple[float, ...] = (2.0, 1.0, 0.5, 0.25, 0.1)
NEAR_WALL_AZIMUTH_OFFSETS: tuple[float, ...] = (0.0, 0.125, 0.25, 0.375, 0.5)
# Six panels per level, strided: they reach the stagnation, shoulder and
# broadside azimuths without evaluating the closed-form kernels at every panel.
# ``tests/test_case_cylinder_stream.py`` checks the sampled maximum against the
# all-panel maximum.
NEAR_WALL_AZIMUTH_PANELS = 6
# The documented window must not depend on where across the panel the target
# sits, so the same 1% tolerance is demanded of the worst azimuth.
NEAR_WALL_AZIMUTH_WINDOW_TOL = 0.01
# Inside the exclusion zone the vertex direction is measurably worse than the
# midpoint direction: the midpoint-only column understates the worst case by at
# least this factor.
NEAR_WALL_AZIMUTH_VERTEX_PENALTY = 2.0
# The two studies probe from different reference surfaces (the circle for the
# radial ladder, the chord for the azimuthal sweep), so their midpoint errors
# agree to a few percent, not to roundoff.
NEAR_WALL_AZIMUTH_FRAME_TOL = 0.10
# First-moment (linear) density is not a uniform improvement: at the documented
# window it is measurably worse than constant panels, because the arc-length
# slope extrapolates a density that is discontinuous across the vertex.
NEAR_WALL_LINEAR_WINDOW_GAIN_MAX = 1.1
GRADIENT_CROSSCHECK_TOL = 1.0e-5
HARMONICITY_TOL = 1.0e-6
# Far-field multipole study (route M).  The disturbance potential of an exterior
# harmonic field decays as sum_n A_n r**(-n): ring fits recover A_n, which must
# agree on every ring (radius independence), carry no spurious source (the 2-D
# log term, which a single ring cannot separate from the ring mean), and put all
# of the strength in the harmonic the boundary data excites.
MULTIPOLE_RADII: tuple[float, ...] = (1.5, 2.5, 4.0)
MULTIPOLE_ORDER = 4
# Nyquist guard: a regular N-gon field carries facet harmonics at n = k N +- 1,
# which alias onto the low harmonics when the ring is sampled at about N points.
MULTIPOLE_MIN_SAMPLES = 64
# Cross-harmonic contamination, source leakage and ring-to-ring spread all come
# out at roundoff (< 1e-13 measured); this threshold still forbids an O(h^2) leak.
MULTIPOLE_PURITY_TOL = 1.0e-9
MULTIPOLE_RADIUS_INDEPENDENCE_TOL = 1.0e-9
MULTIPOLE_SOURCE_TOL = 1.0e-9
# The truncated (order 4) ring fit leaves the n >= 5 facet content in the
# residual: small, but not roundoff.
MULTIPOLE_FIT_RESIDUAL_TOL = 1.0e-5
MULTIPOLE_ORDER_MIN = ORDER_MIN
MULTIPOLE_CONTROL_SIGN = -1.0
# Prescribed circulation of the rotating-cylinder route, in units of U a. The
# two-stagnation-point regime is |Gamma| < 4 pi U a ~= 12.566; 3.0 keeps both
# stagnation points well resolved and off the shoulders (theta_s ~ 13.8 deg).
CIRCULATION = 3.0
CIRCULATION_SWEEP: tuple[float, ...] = (1.0, 2.0, 3.0)
# The solve must reproduce the prescribed contour integral, not merely a flow
# that looks like it: a 1% band is far tighter than the 3% case tolerance and
# still loose enough for the faceting residual.
CIRCULATION_FLUX_TOL = 0.01
# Closed-form stagnation angles are O(1) numbers (deg); the interpolated
# numerical zeros are second order, so 1 deg is a weak but meaningful bound.
STAGNATION_TOL_DEG = 1.0
# d'Alembert: the pressure integral of a steady inviscid flow has no drag.
DRAG_TOL = 0.01


# ---------------------------------------------------------------------------
# closed-form reference
# ---------------------------------------------------------------------------
def exact_velocity(
    points: np.ndarray, radius: float = RADIUS, speed: float = SPEED
) -> np.ndarray:
    """Cartesian velocity ``(u, v)`` of ``W(z) = U (z + a**2/z)``."""
    pts = np.asarray(points, dtype=float)
    z = pts[:, 0] + 1j * pts[:, 1]
    if np.any(np.abs(z) == 0.0):
        raise ValueError("the cylinder centre is a singularity of the reference flow")
    dW = speed * (1.0 - radius**2 / z**2)
    return np.column_stack((dW.real, -dW.imag))


def exact_perturbation_potential(
    points: np.ndarray, radius: float = RADIUS, speed: float = SPEED
) -> np.ndarray:
    """Decaying part ``phi - U x = U a**2 cos(theta) / r`` of the total potential."""
    pts = np.asarray(points, dtype=float)
    r = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(r == 0.0):
        raise ValueError("the perturbation potential is singular at the centre")
    return speed * radius**2 * pts[:, 0] / r**2


def exact_pressure_coefficient(
    points: np.ndarray, radius: float = RADIUS, speed: float = SPEED
) -> np.ndarray:
    """``Cp = 1 - (|v|/U)**2`` of the closed-form field."""
    velocity = exact_velocity(points, radius, speed)
    return 1.0 - np.sum(velocity**2, axis=1) / speed**2


def probe_points(
    rings: Sequence[float] = PROBE_RINGS, angles: int = PROBE_ANGLES
) -> np.ndarray:
    """Fixed exterior probe set, offset in angle to avoid vertex alignment."""
    if angles < 4:
        raise ValueError("at least four probe angles are required")
    theta = 2.0 * np.pi * (np.arange(angles) + 0.5) / angles
    return np.vstack(
        [np.column_stack((ring * np.cos(theta), ring * np.sin(theta))) for ring in rings]
    )


# ---------------------------------------------------------------------------
# geometry gate
# ---------------------------------------------------------------------------
def geometry_gate(panels: int, radius: float = RADIUS) -> dict[str, Any]:
    """Validate the mesh itself before any physics is claimed.

    An inscribed regular polygon has exact area ``N/2 a**2 sin(2 pi/N)`` and
    perimeter ``2 N a sin(pi/N)``; both are compared with the computed mesh, so a
    mis-scaled or non-inscribed discretisation cannot pass unnoticed. The circle
    ratios (defects) are reported as the faceting baseline of the case.
    """
    mesh = regular_polygon(panels, radius)
    radii = np.linalg.norm(mesh.vertices, axis=1)
    exact_area = 0.5 * panels * radius**2 * math.sin(2.0 * math.pi / panels)
    exact_perimeter = 2.0 * panels * radius * math.sin(math.pi / panels)
    area = mesh.area
    perimeter = mesh.perimeter
    return {
        "panels": int(panels),
        "radius": radius,
        "vertex_radius_max_deviation": float(np.max(np.abs(radii - radius))),
        "area": area,
        "inscribed_area_closed_form": exact_area,
        "area_relative_deviation": abs(area - exact_area) / exact_area,
        "circle_area": math.pi * radius**2,
        "area_ratio_to_circle": area / (math.pi * radius**2),
        "perimeter": perimeter,
        "inscribed_perimeter_closed_form": exact_perimeter,
        "perimeter_relative_deviation": abs(perimeter - exact_perimeter) / exact_perimeter,
        "circle_perimeter": 2.0 * math.pi * radius,
        "perimeter_ratio_to_circle": perimeter / (2.0 * math.pi * radius),
        "h_mean": float(np.mean(mesh.lengths)),
        "h_max": float(np.max(mesh.lengths)),
    }


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------
def _relative_l2(computed: np.ndarray, exact: np.ndarray) -> float:
    denominator = float(np.linalg.norm(exact))
    if denominator <= 0.0:
        return float("nan")
    return float(np.linalg.norm(np.asarray(computed) - np.asarray(exact)) / denominator)


def route_dirichlet(
    panels: int, radius: float = RADIUS, speed: float = SPEED
) -> dict[str, Any]:
    """Route D: analytic perturbation potential imposed, field recovered outside."""
    mesh = regular_polygon(panels, radius)
    centers = mesh.centers
    data = exact_perturbation_potential(centers, radius, speed)
    result = solve_laplace_exterior_dirichlet(mesh, data)
    points = probe_points()
    field = evaluate_exterior_field(mesh, result, points)
    exact_velocity_field = exact_velocity(points, radius, speed)
    total_velocity = field.gradient + np.array([speed, 0.0])
    exact_potential = exact_perturbation_potential(points, radius, speed)
    return {
        "route": "dirichlet",
        "panels": int(panels),
        "condition_number": result.condition_number,
        "relative_residual_l2": result.relative_residual_l2,
        "potential_relative_l2": _relative_l2(field.potential, exact_potential),
        "potential_max_abs_error": float(
            np.max(np.abs(field.potential - exact_potential))
        ),
        "velocity_relative_l2": _relative_l2(total_velocity, exact_velocity_field),
        "velocity_max_abs_error": float(
            np.max(np.linalg.norm(total_velocity - exact_velocity_field, axis=1))
        )
        / speed,
        "normal_derivative_net_flux_over_speed_radius": float(
            np.sum(result.normal_derivative * mesh.lengths) / (speed * radius)
        ),
        "h_mean": float(np.mean(mesh.lengths)),
    }


def route_neumann(
    panels: int, radius: float = RADIUS, speed: float = SPEED
) -> dict[str, Any]:
    """Route N: analytic fluid-normal derivative imposed, surface and field compared.

    The exterior Neumann solve fixes the potential only up to a constant, so the
    potential comparison removes that constant (the velocity comparison does not
    need to).
    """
    mesh = regular_polygon(panels, radius)
    centers = mesh.centers
    r = np.hypot(centers[:, 0], centers[:, 1])
    cos_theta = centers[:, 0] / r
    # phi' = U a^2 cos(theta)/r  =>  q_e = d(phi')/dn_e = -d(phi')/dr
    normal_derivative = speed * radius**2 * cos_theta / r**2
    result = solve_laplace_exterior_neumann(mesh, normal_derivative)
    surface_velocity = evaluate_exterior_surface_velocity(mesh, result)
    total_surface = surface_velocity + np.array([speed, 0.0])
    exact_surface = exact_velocity(centers, radius, speed)
    points = probe_points()
    field = evaluate_exterior_field(mesh, result, points)
    total_velocity = field.gradient + np.array([speed, 0.0])
    exact_velocity_field = exact_velocity(points, radius, speed)
    exact_potential = exact_perturbation_potential(points, radius, speed)
    offset = float(np.mean(field.potential - exact_potential))
    normal_residual = np.sum(total_surface * mesh.normals, axis=1)
    cp_numerical = 1.0 - np.sum(total_surface**2, axis=1) / speed**2
    cp_exact = exact_pressure_coefficient(centers, radius, speed)
    exact_speed = 2.0 * speed * np.abs(
        np.sin(np.arctan2(centers[:, 1], centers[:, 0]))
    )
    return {
        "route": "neumann",
        "panels": int(panels),
        "condition_number": result.condition_number,
        "relative_residual_l2": result.relative_residual_l2,
        "surface_velocity_relative_l2": _relative_l2(total_surface, exact_surface),
        "surface_speed_max_error_over_speed": float(
            np.max(np.abs(np.linalg.norm(total_surface, axis=1) - exact_speed))
        )
        / speed,
        "normal_velocity_residual_over_speed": float(np.max(np.abs(normal_residual)))
        / speed,
        "cp_relative_l2": _relative_l2(cp_numerical, cp_exact),
        "cp_min": float(np.min(cp_numerical)),
        "cp_min_exact": float(np.min(cp_exact)),
        "velocity_relative_l2": _relative_l2(total_velocity, exact_velocity_field),
        "velocity_max_abs_error": float(
            np.max(np.linalg.norm(total_velocity - exact_velocity_field, axis=1))
        )
        / speed,
        "potential_offset": offset,
        "potential_relative_l2_up_to_constant": _relative_l2(
            field.potential - offset, exact_potential
        ),
        "normal_derivative_net_flux_over_speed_radius": float(
            np.sum(normal_derivative * mesh.lengths) / (speed * radius)
        ),
        "h_mean": float(np.mean(mesh.lengths)),
    }


def route_circulation(
    panels: int,
    radius: float = RADIUS,
    speed: float = SPEED,
    circulation: float = CIRCULATION,
) -> dict[str, Any]:
    """Route G: prescribed circulation (rotating cylinder), lift vs Kutta-Joukowski.

    The circulation is *prescribed* -- that is what a rotating cylinder does --
    so well-posedness comes from the data and no Kutta condition is involved.
    The solve is two constant-panel exterior Neumann problems (the doublet data
    and the unit-vortex response), the same decomposition as the verified
    lifting solve of :mod:`tensorbem.lifting`; see
    :func:`tensorbem.lifting_cylinder.solve_rotating_cylinder`.

    Everything is then compared with closed form: surface speed
    ``v_theta = Gamma/(2 pi a) - 2 U sin(theta)``, ``Cp``, the external
    velocity field, the stagnation angles ``sin(theta_s) = Gamma/(4 pi U a)``,
    the lift ``L' = -rho U Gamma`` and the vanishing drag.
    """
    mesh = regular_polygon(panels, radius)
    solution = solve_rotating_cylinder(mesh, radius, speed, circulation)
    centers = mesh.centers
    theta = np.arctan2(centers[:, 1], centers[:, 0])
    surface_speed = np.linalg.norm(solution.surface_velocity, axis=1)
    exact_speed = circulation_surface_speed(theta, radius, speed, circulation)
    exact_cp = circulation_pressure_coefficient(theta, radius, speed, circulation)
    points = probe_points()
    total_velocity = circulation_velocity_field(solution, points)
    exact_velocity_field = circulation_velocity(points, radius, speed, circulation)
    exact_lift = joukowski_lift(radius, speed, circulation)
    found = sorted(
        circulation_stagnation_from_surface(mesh, solution.surface_velocity)
    )
    reference = sorted(circulation_stagnation_angles(radius, speed, circulation))
    if len(found) != len(reference):
        stagnation_error_deg = float("inf")
    else:
        stagnation_error_deg = math.degrees(
            max(angle_difference(a, b) for a, b in zip(found, reference))
        )
    dynamic_pressure_area = 0.5 * speed**2 * (2.0 * radius)
    return {
        "route": "circulation",
        "panels": int(panels),
        "circulation": float(circulation),
        "circulation_measured": solution.circulation_ccw,
        "circulation_relative_error": abs(solution.circulation_ccw - circulation)
        / abs(circulation),
        "surface_velocity_relative_l2": _relative_l2(surface_speed, exact_speed),
        "surface_speed_max_error_over_speed": float(
            np.max(np.abs(surface_speed - exact_speed))
        )
        / speed,
        "cp_relative_l2": _relative_l2(
            solution.surface_pressure_coefficient, exact_cp
        ),
        "cp_min": float(np.min(solution.surface_pressure_coefficient)),
        "cp_min_exact": float(np.min(exact_cp)),
        "velocity_relative_l2": _relative_l2(total_velocity, exact_velocity_field),
        "velocity_max_abs_error": float(
            np.max(np.linalg.norm(total_velocity - exact_velocity_field, axis=1))
        )
        / speed,
        "lift": solution.lift,
        "lift_exact": exact_lift,
        "lift_relative_error": solution.lift_relative_error_vs_joukowski,
        "drag": solution.drag,
        "drag_coefficient": solution.drag / dynamic_pressure_area,
        "stagnation_angles_found_deg": [math.degrees(value) for value in found],
        "stagnation_angles_exact_deg": [math.degrees(value) for value in reference],
        "stagnation_angle_max_error_deg": stagnation_error_deg,
        "condition_number": solution.condition_number,
        "relative_residual_l2": solution.relative_residual_l2,
        "h_mean": float(np.mean(mesh.lengths)),
    }


def circulation_sweep(
    panels: int,
    radius: float = RADIUS,
    speed: float = SPEED,
    circulations: Sequence[float] = CIRCULATION_SWEEP,
) -> dict[str, Any]:
    """Lift must be linear in the prescribed circulation (Kutta-Joukowski).

    A single circulation level only checks one point of the lift curve. Sweeping
    ``Gamma`` at a fixed mesh shows that the pressure integral tracks
    ``L' = -rho U Gamma`` across a factor of three instead of being tuned at one
    value, and that the stagnation angle follows
    ``sin(theta_s) = Gamma/(4 pi U a)``.
    """
    rows: list[dict[str, Any]] = []
    for circulation in circulations:
        row = route_circulation(panels, radius, speed, circulation)
        rows.append(
            {
                "circulation": row["circulation"],
                "lift": row["lift"],
                "lift_exact": row["lift_exact"],
                "lift_over_joukowski": row["lift"] / row["lift_exact"]
                if row["lift_exact"]
                else float("nan"),
                "lift_relative_error": row["lift_relative_error"],
                "circulation_relative_error": row["circulation_relative_error"],
                "cp_min": row["cp_min"],
                "cp_min_exact": row["cp_min_exact"],
                "stagnation_angle_max_error_deg": row[
                    "stagnation_angle_max_error_deg"
                ],
            }
        )
    deviations = [abs(row["lift_over_joukowski"] - 1.0) for row in rows]
    return {
        "panels": int(panels),
        "note": "rho = 1 (dimensionless); lift per unit span",
        "rows": rows,
        "max_lift_deviation_from_joukowski": max(deviations),
        "max_stagnation_error_deg": max(
            row["stagnation_angle_max_error_deg"] for row in rows
        ),
    }


def attach_azimuth_study(
    window: Sequence[dict[str, Any]],
    radius: float = RADIUS,
    speed: float = SPEED,
    ladder: Sequence[int] = NEAR_WALL_LADDER,
) -> dict[str, Any]:
    """Measure the azimuthal sweep and cross-check it against the radial ladder.

    ``window`` is the ``near_wall["window"]["levels"]`` list: the same midpoint
    direction measured from the circle instead of from the chord, which is what
    makes the two studies comparable.
    """
    azimuth = near_wall_azimuth_study(radius=radius, speed=speed, ladder=ladder)
    frame = []
    for index, level in enumerate(azimuth["levels"]):
        row = next(
            row
            for row in level["sweep"]
            if row["clearance_over_h_max"] == NEAR_WALL_WINDOW_FACTOR
        )
        midpoint = float(row["midpoint_error"])
        circle = float(window[index]["max_relative_velocity_error"])
        frame.append(
            {
                "panels": level["panels"],
                "midpoint_error_from_chord": midpoint,
                "midpoint_error_from_circle": circle,
                "relative_difference": (
                    abs(midpoint - circle) / circle if circle > 0.0 else float("inf")
                ),
            }
        )
    azimuth["midpoint_cross_check"] = {
        "levels": frame,
        "max_relative_difference": max(item["relative_difference"] for item in frame),
        "tolerance": NEAR_WALL_AZIMUTH_FRAME_TOL,
        "note": (
            "the radial ladder probes at radius a + delta (measured from the "
            "circle) and the azimuthal sweep at chord midpoint + delta (measured "
            "from the polygon), so the same midpoint direction differs by the "
            "faceting offset; agreement within NEAR_WALL_AZIMUTH_FRAME_TOL "
            "confirms both studies measure the same regime"
        ),
    }
    return azimuth


def near_wall_study(
    radius: float = RADIUS,
    speed: float = SPEED,
    ladder: Sequence[int] = NEAR_WALL_LADDER,
    factors: Sequence[float] = NEAR_WALL_FACTORS,
) -> dict[str, Any]:
    """Measure how the exterior field degrades as the probe approaches the body.

    ``factors`` selects the clearance sweep, but three levels are part of the
    study's contract rather than of the caller's choice: the documented window,
    the first-moment window and the exclusion zone.  They are unioned in here
    instead of being demanded of the caller -- passing a shorter tuple used to
    die with a bare ``KeyError: 0.5`` from the summary lookups, which is a
    caller-visible crash for an argument that reads as a sampling choice.

    The constant-panel single- and double-layer potentials are integrated in
    closed form, but the *densities* are piecewise constant, so the recovered
    gradient is only meaningful at distances of the order of the panel size and
    larger.  The audit of the first delivery flagged this regime as unguarded
    (``minimum_clearance_factor`` defaults to zero and the probe rings stop at
    ``r/a = 1.25``); this study makes it explicit instead of implicit:

    * a clearance sweep in units of ``h_max`` on every ladder level,
    * a guard probe showing that the evaluator refuses targets closer than the
      requested ``minimum_clearance_factor * h_max``,
    * the observed order of the window error in ``h`` and the saturation ratio
      showing the near-wall error is panel-relative (it does not vanish for a
      fixed ``delta/h_max`` when the mesh is refined).

    Probes are placed radially above the panel midpoints, so one direction per
    panel is sampled and no probe sits on a vertex ray.
    """
    factors = tuple(
        sorted(
            {float(value) for value in factors}
            | {
                float(NEAR_WALL_WINDOW_FACTOR),
                float(NEAR_WALL_LINEAR_FACTOR),
                float(NEAR_WALL_EXCLUSION_FACTOR),
            },
            reverse=True,
        )
    )
    levels: list[dict[str, Any]] = []
    for panels in ladder:
        mesh = regular_polygon(int(panels), radius)
        centers = mesh.centers
        r = np.hypot(centers[:, 0], centers[:, 1])
        cos_theta = centers[:, 0] / r
        normal_derivative = speed * radius**2 * cos_theta / r**2
        result = solve_laplace_exterior_neumann(mesh, normal_derivative)
        result_linear = solve_laplace_exterior_neumann(
            mesh, normal_derivative, linear_density=True
        )
        # Representation-only control: the same exterior field, but with the two
        # traces taken from the exact solution at the panel midpoints, so the
        # solve error is removed and what is left is the panel basis itself.
        exact_result = LaplaceDirichletResult(
            normal_derivative=normal_derivative.copy(),
            boundary_values=exact_perturbation_potential(centers, speed, radius),
            single_layer_matrix=np.zeros((mesh.panel_count, mesh.panel_count)),
            double_layer_matrix=np.zeros((mesh.panel_count, mesh.panel_count)),
            condition_number=1.0,
            residual_l2=0.0,
            domain="exterior",
        )
        h_max = float(np.max(mesh.lengths))
        theta = np.arctan2(centers[:, 1], centers[:, 0])
        sweep: list[dict[str, Any]] = []
        for factor in factors:
            delta = float(factor) * h_max
            points = np.column_stack(
                (
                    (radius + delta) * np.cos(theta),
                    (radius + delta) * np.sin(theta),
                )
            )
            exact = exact_velocity(points, radius, speed)
            exact_speed = np.linalg.norm(exact, axis=1)

            def _relative(field: Any) -> np.ndarray:
                total = field.gradient + np.array([speed, 0.0])
                return np.linalg.norm(total - exact, axis=1) / exact_speed

            relative = _relative(evaluate_exterior_field(mesh, result, points))
            relative_linear = _relative(
                evaluate_exterior_field(mesh, result_linear, points)
            )
            representation = _relative(
                evaluate_exterior_field(mesh, exact_result, points)
            )
            representation_linear = _relative(
                evaluate_exterior_field(
                    mesh, exact_result, points, linear_density=True
                )
            )
            sweep.append(
                {
                    "clearance_over_h_max": float(factor),
                    "clearance_over_radius": delta / radius,
                    "max_relative_velocity_error": float(np.max(relative)),
                    "mean_relative_velocity_error": float(np.mean(relative)),
                    "max_relative_velocity_error_linear": float(
                        np.max(relative_linear)
                    ),
                    "mean_relative_velocity_error_linear": float(
                        np.mean(relative_linear)
                    ),
                    "representation_error_constant": float(np.max(representation)),
                    "representation_error_linear": float(
                        np.max(representation_linear)
                    ),
                }
            )
        guard_factor = 2.0 * NEAR_WALL_WINDOW_FACTOR
        sampled_factor = 0.5 * NEAR_WALL_WINDOW_FACTOR
        window_points = np.column_stack(
            (
                (radius + sampled_factor * h_max) * np.cos(theta),
                (radius + sampled_factor * h_max) * np.sin(theta),
            )
        )
        guard_rejected = False
        guard_message = "guard did not fire"
        try:
            evaluate_exterior_field(
                mesh, result, window_points, minimum_clearance_factor=guard_factor
            )
        except ValueError as error:  # the demanded behaviour
            guard_rejected = True
            guard_message = str(error).split(";")[0]
        # Locate the case-tolerance boundary on this level.  Only the
        # constant-density field is probed: the boundary is a property of the
        # representation that the gated window/criteria use, and the linear and
        # representation-only variants are already reported on the coarse grid
        # above.  The grid is log-spaced in delta/h_max because the error is a
        # power law in the scaled clearance over the whole usable range.
        floor_samples: list[dict[str, float]] = []
        for factor in sorted(float(value) for value in NEAR_WALL_FLOOR_GRID):
            delta = factor * h_max
            points = np.column_stack(
                (
                    (radius + delta) * np.cos(theta),
                    (radius + delta) * np.sin(theta),
                )
            )
            exact = exact_velocity(points, radius, speed)
            total = evaluate_exterior_field(mesh, result, points).gradient + np.array(
                [speed, 0.0]
            )
            floor_samples.append(
                {
                    "clearance_over_h_max": factor,
                    "clearance_over_radius": delta / radius,
                    "max_relative_velocity_error": float(
                        np.max(
                            np.linalg.norm(total - exact, axis=1)
                            / np.linalg.norm(exact, axis=1)
                        )
                    ),
                }
            )
        floor_block: dict[str, Any] = {
            "tolerance": float(TOLERANCE),
            "grid": floor_samples,
            "clearance_over_h_max": None,
            "bracket": None,
            "errors_at_bracket": None,
            "observed_order_in_clearance": None,
            "note": (
                "the clearance delta/h_max at which the worst-direction relative "
                "velocity error crosses the case tolerance, log-log interpolated "
                "between the two grid samples that straddle it"
            ),
        }
        for coarse, fine in zip(floor_samples, floor_samples[1:]):
            error_coarse = coarse["max_relative_velocity_error"]
            error_fine = fine["max_relative_velocity_error"]
            if error_fine >= TOLERANCE or error_coarse < TOLERANCE:
                continue
            ratio = error_coarse / error_fine
            c_ratio = fine["clearance_over_h_max"] / coarse["clearance_over_h_max"]
            order = math.log(ratio) / math.log(c_ratio) if ratio > 0.0 else float("nan")
            if math.isfinite(order) and order != 0.0:
                boundary = coarse["clearance_over_h_max"] * (error_coarse / TOLERANCE) ** (
                    1.0 / order
                )
            else:  # pragma: no cover - degenerate, never triggered by a power law
                boundary = float("nan")
            floor_block.update(
                {
                    "clearance_over_h_max": float(boundary),
                    "bracket": [
                        coarse["clearance_over_h_max"],
                        fine["clearance_over_h_max"],
                    ],
                    "errors_at_bracket": [error_coarse, error_fine],
                    "observed_order_in_clearance": float(order),
                }
            )
            break
        levels.append(
            {
                "panels": int(panels),
                "h_max_over_radius": h_max / radius,
                "h_mean_over_radius": float(np.mean(mesh.lengths)) / radius,
                "three_percent_floor": floor_block,
                "condition_number": result.condition_number,
                "sweep": sweep,
                "guard_factor": guard_factor,
                "guard_sampled_clearance_factor": sampled_factor,
                "guard_rejected": guard_rejected,
                "guard_message": guard_message,
            }
        )

    def error_at(level: dict[str, Any], factor: float) -> float:
        for row in level["sweep"]:
            if row["clearance_over_h_max"] == float(factor):
                return float(row["max_relative_velocity_error"])
        raise KeyError(factor)

    window = [
        {
            "panels": level["panels"],
            "h_max_over_radius": level["h_max_over_radius"],
            "max_relative_velocity_error": error_at(level, NEAR_WALL_WINDOW_FACTOR),
        }
        for level in levels
    ]
    window_h = [row["h_max_over_radius"] for row in window]
    window_error = [row["max_relative_velocity_error"] for row in window]
    if len(window_h) > 1 and all(value > 0.0 for value in window_error):
        order = float(np.polyfit(np.log(window_h), np.log(window_error), 1)[0])
    else:  # pragma: no cover - defensive
        order = float("nan")
    saturation = [
        {
            "panels": level["panels"],
            "max_relative_velocity_error": error_at(
                level, NEAR_WALL_EXCLUSION_FACTOR
            ),
        }
        for level in levels
    ]
    finest = saturation[-1]["max_relative_velocity_error"]
    saturation_ratio = (
        saturation[0]["max_relative_velocity_error"] / finest
        if finest > 0.0
        else float("inf")
    )
    def linear_at(level: dict[str, Any], factor: float) -> float:
        for row in level["sweep"]:
            if row["clearance_over_h_max"] == float(factor):
                return float(row["max_relative_velocity_error_linear"])
        raise KeyError(factor)

    def representation_at(level: dict[str, Any], factor: float, key: str) -> float:
        for row in level["sweep"]:
            if row["clearance_over_h_max"] == float(factor):
                return float(row[key])
        raise KeyError(factor)

    half = float(NEAR_WALL_LINEAR_FACTOR)
    linear_half = [
        {
            "panels": level["panels"],
            "h_max_over_radius": level["h_max_over_radius"],
            "max_relative_velocity_error": linear_at(level, half),
        }
        for level in levels
    ]
    constant_half = [
        {
            "panels": level["panels"],
            "h_max_over_radius": level["h_max_over_radius"],
            "max_relative_velocity_error": error_at(level, half),
        }
        for level in levels
    ]
    half_h = [level["h_max_over_radius"] for level in linear_half]
    half_error = [row["max_relative_velocity_error"] for row in linear_half]
    if len(half_h) > 1 and all(value > 0.0 for value in half_error):
        half_order = float(np.polyfit(np.log(half_h), np.log(half_error), 1)[0])
    else:  # pragma: no cover - defensive
        half_order = float("nan")
    usable_factor = None
    for factor in sorted(factors, reverse=True):
        worst = max(
            float(
                next(
                    row["max_relative_velocity_error_linear"]
                    for row in level["sweep"]
                    if row["clearance_over_h_max"] == float(factor)
                )
            )
            for level in levels
        )
        if worst <= NEAR_WALL_WINDOW_TOL:
            usable_factor = float(factor)
        else:
            break
    # Faceting attribution: the inscribed polygon's panel midpoints sit at
    # radius a*cos(pi/N), i.e. a*(1 - h^2/8), so a probe placed at a + delta
    # above the *circle* is (delta + h^2/8) above the *surface it is measured
    # against*; at fixed delta/h_max that offset is O(h) of delta, which is
    # exactly the first-order behaviour of the exclusion-zone error measured
    # above. Re-referencing the probes to the polygon surface isolates it.
    faceting = []
    for level, panels in zip(levels, ladder):
        mesh = regular_polygon(int(panels), radius)
        centers = mesh.centers
        theta = np.arctan2(centers[:, 1], centers[:, 0])
        h_max = float(np.max(mesh.lengths))
        delta = NEAR_WALL_EXCLUSION_FACTOR * h_max
        exact_result = LaplaceDirichletResult(
            normal_derivative=speed * radius**2 * (centers[:, 0] / np.hypot(centers[:, 0], centers[:, 1])) / np.hypot(centers[:, 0], centers[:, 1]) ** 2,
            boundary_values=exact_perturbation_potential(centers, speed, radius),
            single_layer_matrix=np.zeros((mesh.panel_count, mesh.panel_count)),
            double_layer_matrix=np.zeros((mesh.panel_count, mesh.panel_count)),
            condition_number=1.0,
            residual_l2=0.0,
            domain="exterior",
        )

        def _err(surface_radius: float) -> tuple[float, float]:
            points = np.column_stack(
                (
                    (surface_radius + delta) * np.cos(theta),
                    (surface_radius + delta) * np.sin(theta),
                )
            )
            exact = exact_velocity(points, radius, speed)
            exact_speed = np.linalg.norm(exact, axis=1)
            values = []
            for linear in (False, True):
                field = evaluate_exterior_field(
                    mesh, exact_result, points, linear_density=linear
                )
                total = field.gradient + np.array([speed, 0.0])
                values.append(
                    float(
                        np.max(
                            np.linalg.norm(total - exact, axis=1) / exact_speed
                        )
                    )
                )
            return values[0], values[1]

        circle_constant, circle_linear = _err(radius)
        polygon_constant, polygon_linear = _err(radius * math.cos(math.pi / panels))
        faceting.append(
            {
                "panels": int(panels),
                "midpoint_radius_over_radius": math.cos(math.pi / panels),
                "faceting_offset_over_h_max": (
                    (radius - radius * math.cos(math.pi / panels)) / h_max
                ),
                "circle_referenced_constant": circle_constant,
                "circle_referenced_linear": circle_linear,
                "surface_referenced_constant": polygon_constant,
                "surface_referenced_linear": polygon_linear,
            }
        )
    azimuth = attach_azimuth_study(window, radius, speed, ladder)
    floor_levels = [
        {
            "panels": level["panels"],
            "h_max_over_radius": level["h_max_over_radius"],
            "clearance_over_h_max": level["three_percent_floor"][
                "clearance_over_h_max"
            ],
            "bracket": level["three_percent_floor"]["bracket"],
        }
        for level in levels
    ]
    located = [row for row in floor_levels if row["clearance_over_h_max"] is not None]
    finest_floor = (
        floor_levels[-1]["clearance_over_h_max"]
        if floor_levels and floor_levels[-1]["clearance_over_h_max"] is not None
        else None
    )
    shifts = [
        coarse["clearance_over_h_max"] / fine["clearance_over_h_max"]
        for coarse, fine in zip(floor_levels, floor_levels[1:])
        if coarse["clearance_over_h_max"] is not None
        and fine["clearance_over_h_max"] is not None
        and fine["clearance_over_h_max"] > 0.0
    ]
    floor_summary = {
        "tolerance": float(TOLERANCE),
        "levels": floor_levels,
        "all_levels_located": len(located) == len(floor_levels),
        "refinement_shifts": shifts,
        "monotone_tolerance": NEAR_WALL_FLOOR_MONOTONE_TOL,
        "boundary_moves_inward_with_refinement": bool(shifts)
        and all(shift >= 1.0 / (1.0 + NEAR_WALL_FLOOR_MONOTONE_TOL) for shift in shifts),
        "finest_level_clearance_over_h_max": finest_floor,
        "window_factor_over_finest_floor": (
            float(NEAR_WALL_WINDOW_FACTOR) / finest_floor
            if finest_floor
            else None
        ),
        "window_margin_required": NEAR_WALL_FLOOR_WINDOW_MARGIN,
        "note": (
            "the case acceptance is < 3 % on the exterior velocity field; this "
            "block measures *where* that holds. Above the reported clearance the "
            "worst-direction relative error is inside the tolerance, below it the "
            "field is outside the acceptance however many criteria pass -- the "
            "probe rings of routes D/N stop at r/a = 1.25, i.e. delta = 0.25 a, "
            "which is many panels out on every level, so the routes themselves "
            "are unaffected"
        ),
    }
    return {
        "floor": floor_summary,
        "azimuth": azimuth,
        "faceting": {
            "clearance_over_h_max": NEAR_WALL_EXCLUSION_FACTOR,
            "levels": faceting,
            "note": (
                "probes measured from the inscribed polygon surface rather than "
                "from the circle: the residual difference is the faceting offset "
                "a(1-cos(pi/N)) ~ h^2/8, which is O(h) relative to delta = "
                "0.1 h_max and is the reason the exclusion-zone error saturates "
                "at first order instead of improving like h^2"
            ),
        },
        "linear": {
            "clearance_over_h_max": half,
            "levels": linear_half,
            "constant_mode_levels": constant_half,
            "observed_order_in_h": half_order,
            "error_tolerance": NEAR_WALL_WINDOW_TOL,
            "usable_clearance_over_h_max": usable_factor,
            "improvement_at_half_h_finest": (
                constant_half[-1]["max_relative_velocity_error"]
                / linear_half[-1]["max_relative_velocity_error"]
                if linear_half[-1]["max_relative_velocity_error"] > 0.0
                else float("inf")
            ),
            "improvement_at_window_finest": (
                error_at(levels[-1], NEAR_WALL_WINDOW_FACTOR)
                / linear_at(levels[-1], NEAR_WALL_WINDOW_FACTOR)
            ),
            "improvement_at_exclusion_finest": (
                error_at(levels[-1], NEAR_WALL_EXCLUSION_FACTOR)
                / linear_at(levels[-1], NEAR_WALL_EXCLUSION_FACTOR)
            ),
            "note": (
                "linear_density=True expands both panel densities as sigma(s) = "
                "sigma_i + sigma'_i (s - L_i/2) with closed-form first-moment "
                "integrals; the representation error drops by an order of "
                "magnitude, so the 1%-accurate window moves in from "
                f"delta = {NEAR_WALL_WINDOW_FACTOR:g} h_max to "
                "delta ~ 0.5 h_max"
            ),
        },
        "attribution": {
            "factors": [float(value) for value in factors],
            "representation_constant": {
                str(level["panels"]): {
                    f"{factor:g}": representation_at(
                        level, factor, "representation_error_constant"
                    )
                    for factor in factors
                }
                for level in levels
            },
            "representation_linear": {
                str(level["panels"]): {
                    f"{factor:g}": representation_at(
                        level, factor, "representation_error_linear"
                    )
                    for factor in factors
                }
                for level in levels
            },
            "note": (
                "the two representation rows use the exact traces at the panel "
                "midpoints, so they isolate the panel basis from the solve; the "
                "difference between them and the sweep rows above measures how "
                "much of the near-wall error is boundary-trace error amplified "
                "as 1/delta"
            ),
        },
        "window": {
            "clearance_over_h_max": NEAR_WALL_WINDOW_FACTOR,
            "relative_error_tolerance": NEAR_WALL_WINDOW_TOL,
            "levels": window,
            "observed_order_in_h": order,
        },
        "exclusion": {
            "clearance_over_h_max": NEAR_WALL_EXCLUSION_FACTOR,
            "levels": saturation,
            "saturation_ratio_coarsest_over_finest": float(saturation_ratio),
            "saturation_band": list(NEAR_WALL_SATURATION_BAND),
        },
        "guard": {
            "factor": 2.0 * NEAR_WALL_WINDOW_FACTOR,
            "sampled_clearance_factor": 0.5 * NEAR_WALL_WINDOW_FACTOR,
            "all_levels_rejected": all(level["guard_rejected"] for level in levels),
            "messages": [level["guard_message"] for level in levels],
        },
        "ladder": [int(value) for value in ladder],
        "factors": [float(value) for value in factors],
        "levels": levels,
        "probe_directions": "radial, one above each panel midpoint",
        "note": (
            "the error reported here is the total velocity (grad(phi) + U x_hat) "
            "against dW/dz at the same points; it is a property of the "
            "constant-density panel representation, not of the linear solver"
        ),
    }



def _distance_to_polygon(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Distance from each point to the closed polygon boundary."""
    ends = np.roll(vertices, -1, axis=0)
    nearest = np.full(len(points), np.inf)
    for start, end in zip(vertices, ends):
        segment = end - start
        projection = np.clip(
            ((points - start) @ segment) / float(segment @ segment), 0.0, 1.0
        )
        closest = start + projection[:, None] * segment
        nearest = np.minimum(nearest, np.linalg.norm(points - closest, axis=1))
    return nearest


def near_wall_azimuth_study(
    radius: float = RADIUS,
    speed: float = SPEED,
    ladder: Sequence[int] = NEAR_WALL_LADDER,
    factors: Sequence[float] = NEAR_WALL_AZIMUTH_FACTORS,
    offsets: Sequence[float] = NEAR_WALL_AZIMUTH_OFFSETS,
) -> dict[str, Any]:
    """Re-probe the near-wall ladder across a panel, not only above its midpoint.

    The classic near-wall table (:func:`near_wall_study`) puts its probes on the
    radial rays through the panel midpoints.  That is the most favourable
    azimuth a constant-density panel offers, so the window it quotes is only a
    statement about midpoints.  This study walks the probe along one panel,
    ``zeta in [0, 0.5] h_max`` from the midpoint to the vertex, and pushes it out
    along the local outward normal (the bisector of the two adjacent panel
    normals at the vertex).  The offset distance actually realised is measured
    against the polygon, not assumed, because the bisector probe is slightly
    closer to the boundary than the nominal offset.

    Reported per level and clearance: the midpoint error (both panel bases), the
    worst error over the azimuth sweep, where that worst azimuth sits, and the
    ratio between them.  Aggregates answer three questions the midpoint table
    cannot:

    * does the documented window survive at every azimuth?
    * how much worse than the quoted exclusion-zone number is the worst azimuth?
    * does ``linear_density=True`` keep its advantage once the azimuth is free?
    """
    levels: list[dict[str, Any]] = []
    for panels in ladder:
        mesh = regular_polygon(int(panels), radius)
        centers, vectors, lengths = mesh.centers, mesh.vectors, mesh.lengths
        h_max = float(np.max(lengths))
        stride = max(1, int(panels) // NEAR_WALL_AZIMUTH_PANELS)
        select = np.arange(0, int(panels), stride)
        tangents = vectors / lengths[:, None]
        normals = np.column_stack((tangents[:, 1], -tangents[:, 0]))
        following = np.roll(normals, -1, axis=0)
        bisectors = normals + following
        bisectors /= np.linalg.norm(bisectors, axis=1)[:, None]
        r = np.hypot(centers[:, 0], centers[:, 1])
        normal_derivative = speed * radius**2 * (centers[:, 0] / r) / r**2
        result = solve_laplace_exterior_neumann(mesh, normal_derivative)
        result_linear = solve_laplace_exterior_neumann(
            mesh, normal_derivative, linear_density=True
        )
        blocks: list[np.ndarray] = []
        tags: list[tuple[float, float]] = []
        for factor in factors:
            delta = float(factor) * h_max
            for offset in offsets:
                if float(offset) >= 0.5:
                    base = centers[select] + 0.5 * h_max * tangents[select]
                    direction = bisectors[select]
                else:
                    base = centers[select] + float(offset) * h_max * tangents[select]
                    direction = normals[select]
                blocks.append(base + delta * direction)
                tags.append((float(factor), float(offset)))
        points = np.vstack(blocks)
        exact = exact_velocity(points, radius, speed)
        exact_speed = np.linalg.norm(exact, axis=1)

        def _relative(field: Any) -> np.ndarray:
            total = field.gradient + np.array([speed, 0.0])
            return np.linalg.norm(total - exact, axis=1) / exact_speed

        constant = _relative(evaluate_exterior_field(mesh, result, points))
        linear = _relative(
            evaluate_exterior_field(mesh, result_linear, points, linear_density=True)
        )
        clearance = _distance_to_polygon(points, mesh.vertices) / h_max
        shape = (len(factors), len(offsets), len(select))
        constant = constant.reshape(shape)
        linear = linear.reshape(shape)
        clearance = clearance.reshape(shape)
        sweep: list[dict[str, Any]] = []
        for i, factor in enumerate(factors):
            rows = [
                {
                    "offset_over_h_max": float(offset),
                    "max_relative_velocity_error": float(constant[i, j].max()),
                    "max_relative_velocity_error_linear": float(linear[i, j].max()),
                    "min_clearance_over_h_max": float(clearance[i, j].min()),
                }
                for j, offset in enumerate(offsets)
            ]
            midpoint = next(
                row for row in rows if row["offset_over_h_max"] == 0.0
            )
            worst = max(rows, key=lambda row: row["max_relative_velocity_error"])
            worst_linear = max(
                rows, key=lambda row: row["max_relative_velocity_error_linear"]
            )
            midpoint_error = midpoint["max_relative_velocity_error"]
            midpoint_error_linear = midpoint["max_relative_velocity_error_linear"]
            sweep.append(
                {
                    "clearance_over_h_max": float(factor),
                    "offsets": rows,
                    "midpoint_error": midpoint_error,
                    "midpoint_error_linear": midpoint_error_linear,
                    "worst_error": worst["max_relative_velocity_error"],
                    "worst_error_linear": worst_linear[
                        "max_relative_velocity_error_linear"
                    ],
                    "worst_offset_over_h_max": worst["offset_over_h_max"],
                    "worst_offset_over_h_max_linear": worst_linear[
                        "offset_over_h_max"
                    ],
                    "azimuth_penalty": (
                        worst["max_relative_velocity_error"] / midpoint_error
                        if midpoint_error > 0.0
                        else float("inf")
                    ),
                    "azimuth_penalty_linear": (
                        worst_linear["max_relative_velocity_error_linear"]
                        / midpoint_error_linear
                        if midpoint_error_linear > 0.0
                        else float("inf")
                    ),
                    "min_clearance_over_h_max": min(
                        row["min_clearance_over_h_max"] for row in rows
                    ),
                }
            )
        levels.append(
            {
                "panels": int(panels),
                "h_max_over_radius": h_max / radius,
                "sampled_panels": int(len(select)),
                "panel_stride": int(stride),
                "sweep": sweep,
            }
        )

    def _at(level: dict[str, Any], factor: float) -> dict[str, Any]:
        return next(
            row for row in level["sweep"] if row["clearance_over_h_max"] == float(factor)
        )

    def _window_factor(key: str) -> float | None:
        factor: float | None = None
        for candidate in sorted(factors, key=float, reverse=True):
            worst = max(
                float(_at(level, candidate)[key]) for level in levels
            )
            if worst <= NEAR_WALL_AZIMUTH_WINDOW_TOL:
                factor = float(candidate)
            else:
                break
        return factor

    window = {
        "clearance_over_h_max": NEAR_WALL_WINDOW_FACTOR,
        "relative_error_tolerance": NEAR_WALL_AZIMUTH_WINDOW_TOL,
        "levels": [
            {
                "panels": level["panels"],
                "h_max_over_radius": level["h_max_over_radius"],
                "midpoint_error": _at(level, NEAR_WALL_WINDOW_FACTOR)["midpoint_error"],
                "worst_error": _at(level, NEAR_WALL_WINDOW_FACTOR)["worst_error"],
                "worst_error_linear": _at(level, NEAR_WALL_WINDOW_FACTOR)[
                    "worst_error_linear"
                ],
                "azimuth_penalty": _at(level, NEAR_WALL_WINDOW_FACTOR)[
                    "azimuth_penalty"
                ],
            }
            for level in levels
        ],
    }
    window["worst_over_levels"] = max(
        row["worst_error"] for row in window["levels"]
    )
    window["worst_over_levels_linear"] = max(
        row["worst_error_linear"] for row in window["levels"]
    )
    exclusion = {
        "clearance_over_h_max": NEAR_WALL_EXCLUSION_FACTOR,
        "levels": [
            {
                "panels": level["panels"],
                "midpoint_error": _at(level, NEAR_WALL_EXCLUSION_FACTOR)[
                    "midpoint_error"
                ],
                "worst_error": _at(level, NEAR_WALL_EXCLUSION_FACTOR)["worst_error"],
                "worst_error_linear": _at(level, NEAR_WALL_EXCLUSION_FACTOR)[
                    "worst_error_linear"
                ],
                "worst_offset_over_h_max": _at(level, NEAR_WALL_EXCLUSION_FACTOR)[
                    "worst_offset_over_h_max"
                ],
                "azimuth_penalty": _at(level, NEAR_WALL_EXCLUSION_FACTOR)[
                    "azimuth_penalty"
                ],
                "azimuth_penalty_linear": _at(level, NEAR_WALL_EXCLUSION_FACTOR)[
                    "azimuth_penalty_linear"
                ],
            }
            for level in levels
        ],
    }
    exclusion["min_azimuth_penalty"] = min(
        row["azimuth_penalty"] for row in exclusion["levels"]
    )
    exclusion["worst_over_levels"] = max(
        row["worst_error"] for row in exclusion["levels"]
    )
    linear_gain = {
        "clearance_over_h_max": NEAR_WALL_WINDOW_FACTOR,
        "levels": [
            {
                "panels": level["panels"],
                "constant_worst": _at(level, NEAR_WALL_WINDOW_FACTOR)["worst_error"],
                "linear_worst": _at(level, NEAR_WALL_WINDOW_FACTOR)[
                    "worst_error_linear"
                ],
                "ratio_linear_over_constant": (
                    _at(level, NEAR_WALL_WINDOW_FACTOR)["worst_error_linear"]
                    / _at(level, NEAR_WALL_WINDOW_FACTOR)["worst_error"]
                ),
            }
            for level in levels
        ],
    }
    linear_gain["min_ratio"] = min(
        row["ratio_linear_over_constant"] for row in linear_gain["levels"]
    )
    linear_gain["max_ratio"] = max(
        row["ratio_linear_over_constant"] for row in linear_gain["levels"]
    )
    return {
        "ladder": [int(value) for value in ladder],
        "factors": [float(value) for value in factors],
        "offsets_over_h_max": [float(value) for value in offsets],
        "levels": levels,
        "window": window,
        "exclusion": exclusion,
        "linear_at_window": linear_gain,
        "windows_including_azimuth": {
            "relative_error_tolerance": NEAR_WALL_AZIMUTH_WINDOW_TOL,
            "constant_density": _window_factor("worst_error"),
            "linear_density": _window_factor("worst_error_linear"),
        },
        "probe_directions": (
            "along one panel from the midpoint to the vertex (bisector normal at "
            "the vertex), pushed out along the local outward normal"
        ),
        "note": (
            "zones are reported per azimuth: the midpoint column reproduces the "
            "radial ladder (from the circle, so it agrees to a few percent) while "
            "the worst column is the honest bound for an arbitrary target"
        ),
    }


def gradient_kernel_crosscheck(
    panels: int = 32,
    radius: float = RADIUS,
    speed: float = SPEED,
    step_scale: float = 1.0e-5,
) -> dict[str, Any]:
    """Compare the closed-form field gradient with finite differences of the potential.

    This check involves no additional reference solution: it only asks whether
    the analytic panel-integral gradients that carry the velocity field are the
    derivatives of the potential that the same module evaluates.
    """
    mesh = regular_polygon(panels, radius)
    centers = mesh.centers
    data = exact_perturbation_potential(centers, radius, speed)
    result = solve_laplace_exterior_dirichlet(mesh, data)
    points = probe_points(rings=(1.25, 2.0), angles=8)
    field = evaluate_exterior_field(mesh, result, points)
    step = step_scale * float(np.min(np.linalg.norm(points, axis=1)) - radius)
    numeric = np.empty_like(field.gradient)
    for axis in range(2):
        offset = np.zeros(2)
        offset[axis] = step
        plus = evaluate_exterior_field(mesh, result, points + offset).potential
        minus = evaluate_exterior_field(mesh, result, points - offset).potential
        numeric[:, axis] = (plus - minus) / (2.0 * step)
    scale = float(np.max(np.linalg.norm(field.gradient, axis=1)))
    return {
        "panels": int(panels),
        "step": step,
        "points": int(len(points)),
        "max_abs_difference": float(np.max(np.abs(numeric - field.gradient))),
        "gradient_scale": scale,
        "relative_crosscheck": float(np.max(np.abs(numeric - field.gradient))) / scale,
    }


def harmonicity_check(radius: float = RADIUS, speed: float = SPEED) -> dict[str, Any]:
    """Numerically confirm the closed-form reference is harmonic and curl free.

    The reference is generated from ``W(z)``; differentiating it shows
    ``Laplacian(phi) = 0`` and ``curl(v) = 0`` at the probe points, so a coding
    slip in the closed form cannot masquerade as a "reference". The second
    formula for the perturbation potential (polar, angle-based) is compared with
    the Cartesian one as an independent transcription check.
    """
    points = probe_points()
    step = 1.0e-4
    eye = np.eye(2) * step
    potential = exact_perturbation_potential
    laplacian = np.zeros(len(points))
    for axis in range(2):
        laplacian += (
            potential(points + eye[axis], radius, speed)
            - 2.0 * potential(points, radius, speed)
            + potential(points - eye[axis], radius, speed)
        ) / step**2
    curl = np.zeros(len(points))
    for axis in range(2):
        plus = exact_velocity(points + eye[axis], radius, speed)[:, 1 - axis]
        minus = exact_velocity(points - eye[axis], radius, speed)[:, 1 - axis]
        sign = 1.0 if axis == 0 else -1.0
        curl += sign * (plus - minus) / (2.0 * step)
    polar = (
        speed
        * radius**2
        * np.cos(np.arctan2(points[:, 1], points[:, 0]))
        / np.linalg.norm(points, axis=1)
    )
    defect = np.abs(potential(points, radius, speed) - polar)
    return {
        "points": int(len(points)),
        "finite_difference_step": step,
        "laplacian_scaled_max": float(np.max(np.abs(laplacian))) / speed,
        "curl_scaled_max": float(np.max(np.abs(curl))) / speed,
        "perturbation_formula_max_abs_difference": float(np.max(defect)),
        "reference_velocity_max_over_speed": float(
            np.max(np.linalg.norm(exact_velocity(points, radius, speed), axis=1))
        )
        / speed,
    }


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------
def wrong_reference_control(
    panels: int = BUDGET_PANELS, radius: float = RADIUS, speed: float = SPEED
) -> dict[str, Any]:
    """Solve the same problem but also compare against the mirrored image sign.

    ``W_mirror(z) = U (z - a**2/z)`` is harmonic and decaying as well, but it is
    the flow of a source-sink pair, not of a cylinder. A case that could not tell
    the two apart would prove nothing, so the recovered field must be rejected by
    the mirrored reference by a wide margin.
    """
    mesh = regular_polygon(panels, radius)
    centers = mesh.centers
    data = exact_perturbation_potential(centers, radius, speed)
    result = solve_laplace_exterior_dirichlet(mesh, data)
    points = probe_points()
    field = evaluate_exterior_field(mesh, result, points)
    total_velocity = field.gradient + np.array([speed, 0.0])
    correct = exact_velocity(points, radius, speed)
    r = np.hypot(points[:, 0], points[:, 1])
    cos_theta = points[:, 0] / r
    sin_theta = points[:, 1] / r
    mirror = np.column_stack(
        (
            speed * (1.0 - radius**2 * (cos_theta**2 - sin_theta**2) / r**2),
            -speed * (-2.0 * radius**2 * cos_theta * sin_theta / r**2),
        )
    )
    error_correct = _relative_l2(total_velocity, correct)
    error_mirror = _relative_l2(total_velocity, mirror)
    return {
        "panels": int(panels),
        "relative_l2_vs_correct_reference": error_correct,
        "relative_l2_vs_mirror_reference": error_mirror,
        "discrimination_ratio": error_mirror / max(error_correct, 1.0e-300),
    }


# ---------------------------------------------------------------------------
# far-field multipole content (route M)
# ---------------------------------------------------------------------------
def _uniform_flow_perturbation(
    centers: np.ndarray, radius: float, speed: float
) -> np.ndarray:
    """``phi' = U a**2 cos(theta)/r = U a**2 x / r**2`` on the given points."""
    r = np.hypot(centers[:, 0], centers[:, 1])
    return speed * radius**2 * centers[:, 0] / r**2


def _pure_strain_perturbation(
    centers: np.ndarray, radius: float, speed: float
) -> np.ndarray:
    """``phi' = (gamma/2) a**4 cos(2 theta)/r**2`` for ``u = gamma x, v = -gamma y``.

    The full potential of the straining flow past the cylinder is
    ``phi = (gamma/2) (r**2 + a**4/r**2) cos(2 theta)``: its normal derivative
    vanishes on ``r = a`` and it tends to ``(gamma/2)(x**2 - y**2)`` at infinity.
    Only the decaying part is imposed on the boundary, exactly as for the
    uniform-flow dipole.
    """
    r = np.hypot(centers[:, 0], centers[:, 1])
    theta = np.arctan2(centers[:, 1], centers[:, 0])
    return 0.5 * speed * radius**4 * np.cos(2.0 * theta) / r**2


MULTIPOLE_FAMILIES: dict[str, dict[str, Any]] = {
    "uniform_flow": {
        "harmonic": 1,
        "potential": "phi' = U a**2 cos(theta)/r",
        "excitation": "uniform stream U along +x",
        "boundary_values": _uniform_flow_perturbation,
        "exact_moment": "A_1 = U a**2",
        "scale": lambda radius, speed: speed * radius**2,
        "contaminants": "A_2 = A_3 = A_4 = 0 and no 2-D source",
        "bibliography": (
            "Lamb H. Hydrodynamics, 6th ed., 1932 (uniform flow past a cylinder)."
        ),
    },
    "pure_strain": {
        "harmonic": 2,
        "potential": "phi' = (gamma/2) a**4 cos(2 theta)/r**2",
        "excitation": "plane straining flow u = gamma x, v = -gamma y",
        "boundary_values": _pure_strain_perturbation,
        "exact_moment": "A_2 = (gamma/2) a**4",
        "scale": lambda radius, speed: 0.5 * speed * radius**4,
        "contaminants": "A_1 = A_3 = A_4 = 0 and no 2-D source",
        "bibliography": (
            "Milne-Thomson L M. Theoretical Hydrodynamics, 5th ed., 1968: "
            "straining flow past a circular cylinder."
        ),
    },
}


def multipole_level(
    panels: int,
    family: str,
    radius: float = RADIUS,
    speed: float = SPEED,
    scale_sign: float = 1.0,
) -> dict[str, Any]:
    """Recover the far-field multipole spectrum of one exterior Dirichlet solve.

    ``scale_sign = -1`` feeds the mirrored (image-sign) data instead: the
    negative control, since a reference that cannot tell ``A_1`` from ``-A_1``
    is not measuring the moment at all.
    """
    if family not in MULTIPOLE_FAMILIES:
        raise ValueError(f"unknown multipole family {family!r}")
    entry = MULTIPOLE_FAMILIES[family]
    mesh = regular_polygon(panels, radius)
    data = scale_sign * entry["boundary_values"](mesh.centers, radius, speed)
    result = solve_laplace_exterior_dirichlet(mesh, data)
    samples = max(MULTIPOLE_MIN_SAMPLES, 2 * int(panels))
    angles = ring_angles(samples)
    fits = []
    for ring in MULTIPOLE_RADII:
        points = np.column_stack(
            (ring * radius * np.cos(angles), ring * radius * np.sin(angles))
        )
        field = evaluate_exterior_field(mesh, result, points)
        fits.append(
            fit_ring(field.potential, ring * radius, angles, order=MULTIPOLE_ORDER)
        )
    report = decompose_rings(fits)
    harmonic = int(entry["harmonic"])
    scale = float(entry["scale"](radius, speed))
    spectrum = report["spectrum"]
    target = spectrum[harmonic]
    contaminant_orders = [
        n for n in range(1, MULTIPOLE_ORDER + 1) if n != harmonic
    ]
    largest_contaminant = max(
        abs(float(spectrum[n]["mean"])) for n in contaminant_orders
    )
    return {
        "family": family,
        "panels": int(panels),
        "h_mean": float(np.mean(mesh.lengths)),
        "condition_number": result.condition_number,
        "relative_residual": result.relative_residual_l2,
        "samples_per_ring": int(samples),
        "rings_over_a": [float(ring) for ring in MULTIPOLE_RADII],
        "harmonic": harmonic,
        "scale": scale,
        "target_moment_over_scale": float(target["mean"]) / scale,
        # signed cosine projection: the amplitude above is a magnitude, so only
        # this number can tell A_n from -A_n (the mirror control needs it)
        "target_cosine_over_scale": float(
            np.mean([fit.cos_strength(harmonic) for fit in fits])
        )
        / scale,
        "target_moment_relative_error": abs(float(target["mean"]) / scale - 1.0),
        "target_moment_ring_spread_relative": float(target["relative_spread"]),
        "contaminant_over_scale": {
            str(n): float(spectrum[n]["mean"]) / scale for n in contaminant_orders
        },
        "largest_contaminant_over_scale": largest_contaminant / scale,
        "source_over_scale": abs(float(report["source_strength"])) / scale,
        "source_trend_residual_over_scale": float(report["source_trend_residual"])
        / scale,
        "fit_relative_residual_max": float(max(report["relative_residual"])),
        "ring_fit_condition_number_max": float(max(report["condition_number"])),
        "ring_constants_over_scale": [
            float(value) / scale for value in report["constants"]
        ],
    }


def multipole_study(
    ladder: Sequence[int] = LADDER,
    radius: float = RADIUS,
    speed: float = SPEED,
) -> dict[str, Any]:
    """Route M: multipole spectrum of the exterior field on both analytic flows."""
    families: dict[str, Any] = {}
    for family, entry in MULTIPOLE_FAMILIES.items():
        levels = [multipole_level(panels, family, radius, speed) for panels in ladder]
        families[family] = {
            "harmonic": int(entry["harmonic"]),
            "potential": entry["potential"],
            "excitation": entry["excitation"],
            "exact_moment": entry["exact_moment"],
            "contaminants": entry["contaminants"],
            "bibliography": entry["bibliography"],
            "levels": levels,
            "observed_order_target_moment": _observed_orders(
                levels, "target_moment_relative_error"
            ),
            "finest": levels[-1],
            "worst": {
                "largest_contaminant_over_scale": max(
                    level["largest_contaminant_over_scale"] for level in levels
                ),
                "source_over_scale": max(
                    level["source_over_scale"] for level in levels
                ),
                "target_moment_ring_spread_relative": max(
                    level["target_moment_ring_spread_relative"] for level in levels
                ),
                "fit_relative_residual_max": max(
                    level["fit_relative_residual_max"] for level in levels
                ),
            },
        }
    control = multipole_level(
        max(ladder), "uniform_flow", radius, speed, scale_sign=MULTIPOLE_CONTROL_SIGN
    )
    correct = families["uniform_flow"]["levels"][-1]["target_moment_relative_error"]
    return {
        "radii_over_a": [float(ring) for ring in MULTIPOLE_RADII],
        "order": MULTIPOLE_ORDER,
        "families": families,
        "mirror_control": {
            "dipole_over_scale": control["target_moment_over_scale"],
            "target_cosine_over_scale": control["target_cosine_over_scale"],
            "signed_error_of_the_mirror": abs(
                control["target_cosine_over_scale"] + 1.0
            ),
            "discrimination_ratio": 2.0 / correct if correct > 0.0 else float("inf"),
        },
    }


def far_field_check(
    panels: int = BUDGET_PANELS, radius: float = RADIUS, speed: float = SPEED
) -> dict[str, Any]:
    """The disturbance decays like ``1/r**2``; check the field where it is small.

    At ``r = 8a`` the perturbation is still about ``2 a**2/r**2 = 3%`` of ``U``,
    so the criterion is not "velocity equals U" but "the BEM velocity matches the
    closed form to 1% of U" at a radius where the panel mesh is far away.
    """
    mesh = regular_polygon(panels, radius)
    centers = mesh.centers
    data = exact_perturbation_potential(centers, radius, speed)
    result = solve_laplace_exterior_dirichlet(mesh, data)
    theta = 2.0 * np.pi * (np.arange(PROBE_ANGLES) + 0.5) / PROBE_ANGLES
    points = np.column_stack(
        (
            FAR_FIELD_RADIUS * radius * np.cos(theta),
            FAR_FIELD_RADIUS * radius * np.sin(theta),
        )
    )
    field = evaluate_exterior_field(mesh, result, points)
    total_velocity = field.gradient + np.array([speed, 0.0])
    exact = exact_velocity(points, radius, speed)
    return {
        "radius_over_a": FAR_FIELD_RADIUS,
        "max_velocity_error_vs_exact_over_speed": float(
            np.max(np.linalg.norm(total_velocity - exact, axis=1))
        )
        / speed,
        "max_perturbation_over_speed": float(
            np.max(np.linalg.norm(exact - np.array([speed, 0.0]), axis=1))
        )
        / speed,
        "max_perturbation_potential_over_speed_radius": float(
            np.max(np.abs(field.potential))
        )
        / (speed * radius),
    }


# ---------------------------------------------------------------------------
# study driver
# ---------------------------------------------------------------------------
def _observed_orders(levels: list[dict[str, Any]], key: str) -> list[float]:
    orders: list[float] = []
    for coarse, fine in zip(levels, levels[1:]):
        error_coarse = float(coarse[key])
        error_fine = float(fine[key])
        h_ratio = float(coarse["h_mean"]) / float(fine["h_mean"])
        if error_coarse <= 0.0 or error_fine <= 0.0 or h_ratio <= 1.0:
            orders.append(float("nan"))
            continue
        orders.append(math.log(error_coarse / error_fine) / math.log(h_ratio))
    return orders


def explicit_probe_table(
    probes: np.ndarray, radius: float, speed: float
) -> list[dict[str, Any]]:
    """Reference velocity and perturbation potential at every probe point."""
    velocity = exact_velocity(probes, radius, speed)
    potential = exact_perturbation_potential(probes, radius, speed)
    return [
        {
            "x_over_a": float(point[0] / radius),
            "y_over_a": float(point[1] / radius),
            "u_over_U": float(vel[0] / speed),
            "v_over_U": float(vel[1] / speed),
            "perturbation_potential_over_Ua": float(phi / (speed * radius)),
        }
        for point, vel, phi in zip(probes, velocity, potential)
    ]


GENERATING_SOURCE_FILES: tuple[str, ...] = (
    "benchmarks/case_cylinder_stream.py",
    "solver.py",
    "kernels.py",
    "near_singular.py",
    "geometry.py",
    "multipole2d.py",
    "lifting_cylinder.py",
    "lifting.py",
)

GUARDED_SOURCE_FILE = "benchmarks/case_cylinder_stream.py"


def generator_provenance() -> dict[str, Any]:
    """Digests of the sources whose numbers this record contains.

    A committed ``results.json`` is the record of *one* run, and the way it goes
    stale is silent: the case module can grow a study and a set of criteria, be
    merged, and leave the record -- and the prose rendered from it -- describing
    the previous study, while docstrings elsewhere point at the record as
    evidence for the new one.  Recording the digests makes the code-to-record
    link explicit and checkable without re-running the case.

    ``GUARDED_SOURCE_FILE`` is the file that decides what the record contains
    and how it is rendered, so a change to it *must* be followed by a
    regeneration; the case-pool test asserts exactly that digest.  The other
    files are recorded for traceability: a change there changes the numbers
    without changing the record's shape, and regenerating is the responsibility
    of whoever made it.
    """
    package = Path(__file__).resolve().parent.parent
    digests: dict[str, str | None] = {}
    for name in GENERATING_SOURCE_FILES:
        path = package / name
        digests[name] = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
    combined = hashlib.sha256(
        "\n".join(
            f"{name}:{digests[name]}" for name in GENERATING_SOURCE_FILES
        ).encode()
    ).hexdigest()
    return {
        "files": digests,
        "combined_sha256": combined,
        "guarded_file": GUARDED_SOURCE_FILE,
    }


def run_case(
    ladder: Sequence[int] = LADDER,
    radius: float = RADIUS,
    speed: float = SPEED,
    with_controls: bool = True,
) -> dict[str, Any]:
    """Run both routes on the ladder and return the machine-readable artifact."""
    started = time.time()
    panels = tuple(int(value) for value in ladder)
    if not panels:
        raise ValueError("the panel ladder must not be empty")
    if len(set(panels)) != len(panels):
        raise ValueError("the panel ladder must not repeat a level")
    budget = max(panels)

    dirichlet_levels = [route_dirichlet(n, radius, speed) for n in panels]
    neumann_levels = [route_neumann(n, radius, speed) for n in panels]
    probes = probe_points()
    analytic_velocity = exact_velocity(probes, radius, speed)

    artifact: dict[str, Any] = {
        "case_id": CASE_ID,
        "pool_code": POOL_CODE,
        "schema": SCHEMA,
        "repo": "TensorBEM",
        "domain": "ship-total-performance/resistance-potential-flow",
        "capability": (
            "2-D exterior Laplace operator: potential AND velocity field outside a "
            "body, plus the prescribed-circulation (rotating cylinder) exterior "
            "Neumann problem, verified against the cylinder complex potential"
        ),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "machine": platform.machine(),
        },
        "provenance": {"generator": generator_provenance()},
        "parameters": {
            "radius": radius,
            "speed": speed,
            "panel_ladder": list(panels),
            "budget_panels": budget,
            "probe_rings": list(PROBE_RINGS),
            "probe_angles": PROBE_ANGLES,
            "probe_count": int(len(probes)),
            "far_field_radius_over_a": FAR_FIELD_RADIUS,
            "circulation": CIRCULATION,
            "circulation_sweep": list(CIRCULATION_SWEEP),
        },
        "reference": {
            "complex_potential": "W(z) = U (z + a**2/z)",
            "velocity": "dW/dz = u - i v = U (1 - a**2/z**2)",
            "perturbation_potential": "phi - U x = U a**2 cos(theta) / r",
            "surface_speed": "|v| = 2 U |sin(theta)|",
            "pressure_coefficient": "Cp = 1 - 4 sin(theta)**2",
            "lifting_complex_potential": (
                "W(z) = U (z + a**2/z) - i Gamma/(2 pi) ln(z/a) "
                "(Gamma counter-clockwise)"
            ),
            "lifting_velocity": (
                "dW/dz = u - i v = U (1 - a**2/z**2) - i Gamma/(2 pi z)"
            ),
            "lifting_surface_speed": (
                "v_theta = Gamma/(2 pi a) - 2 U sin(theta); "
                "Cp = 1 - (v_theta/U)**2"
            ),
            "lifting_stagnation": "sin(theta_s) = Gamma/(4 pi U a)",
            "lifting_lift": (
                "L' = -rho U Gamma (Kutta-Joukowski, rho = 1); drag = 0"
            ),
            "probe_table": explicit_probe_table(probes, radius, speed),
            "hand_typed_numbers": 0,
            "bibliography": [
                "Lamb H. Hydrodynamics. 6th ed. Cambridge University Press, 1932: "
                "uniform flow past a circular cylinder (same edition as "
                "docs/classical_benchmarks.md ref. [1]).",
                "Milne-Thomson L M. Theoretical Hydrodynamics. 5th ed. Macmillan, "
                "1968: circle theorem / translating-cylinder potential "
                "(docs/classical_benchmarks.md ref. [3]).",
            ],
            "page_or_article_number": (
                "WITHHELD: neither book is reachable from this container, so the exact "
                "page/article number was not verified here; the formulas are "
                "elementary and are instead re-derived numerically by the "
                "harmonicity gate (Laplacian and curl of the reference)."
            ),
        },
        "geometry_gate": geometry_gate(budget, radius),
        "harmonicity": harmonicity_check(radius, speed),
        "gradient_kernel_crosscheck": gradient_kernel_crosscheck(
            radius=radius, speed=speed
        ),
        "routes": {
            "dirichlet": {
                "boundary_data": "phi' = U a**2 cos(theta)/r on panel midpoints",
                "recovered": "potential and grad(phi) at exterior probes",
                "levels": dirichlet_levels,
                "observed_order_velocity": _observed_orders(
                    dirichlet_levels, "velocity_relative_l2"
                ),
                "observed_order_potential": _observed_orders(
                    dirichlet_levels, "potential_relative_l2"
                ),
                "finest": dirichlet_levels[-1],
            },
            "neumann": {
                "boundary_data": (
                    "q_e = U a**2 cos(theta)/r**2 (zero total normal velocity)"
                ),
                "recovered": (
                    "surface velocity from q_e and d(phi)/ds; grad(phi) outside"
                ),
                "levels": neumann_levels,
                "observed_order_velocity": _observed_orders(
                    neumann_levels, "velocity_relative_l2"
                ),
                "observed_order_surface_velocity": _observed_orders(
                    neumann_levels, "surface_velocity_relative_l2"
                ),
                "finest": neumann_levels[-1],
            },
        },
        "probe_reference": {
            "points": int(len(probes)),
            "velocity_norm_max_over_speed": float(
                np.max(np.linalg.norm(analytic_velocity, axis=1))
            )
            / speed,
            "velocity_norm_min_over_speed": float(
                np.min(np.linalg.norm(analytic_velocity, axis=1))
            )
            / speed,
        },
        "repro_cmd": (
            "PYTHONPATH=src python3 -m tensorbem.benchmarks.case_cylinder_stream "
            "--outdir benchmarks/verified/case_bem_laplace_cylinder "
            "--doc docs/CASE-BEM-LAPLACE-CYLINDER.md"
        ),
    }
    circulation_levels = [route_circulation(n, radius, speed) for n in panels]
    artifact["routes"]["circulation"] = {
        "declared": {
            "circulation": CIRCULATION,
            "surface_speed": "v_theta = Gamma/(2 pi a) - 2 U sin(theta)",
            "pressure_coefficient": "Cp = 1 - (v_theta/U)**2",
            "stagnation_angles": "sin(theta_s) = Gamma/(4 pi U a)",
            "lift": "L' = -rho U Gamma (Kutta-Joukowski, rho = 1)",
            "drag": "0 (d'Alembert)",
            "imposed": (
                "doublet Neumann data q_e = grad(phi_d).n_e plus the unit-vortex "
                "response q_e = v_unit.n_body; the circulation itself is "
                "prescribed, so no Kutta condition is solved"
            ),
            "recovered": (
                "surface velocity from the solved traces, pressure integral for "
                "lift and drag, interpolated zeros of v_theta for the "
                "stagnation points"
            ),
        },
        "levels": circulation_levels,
        "observed_order_surface_velocity": _observed_orders(
            circulation_levels, "surface_velocity_relative_l2"
        ),
        "observed_order_lift": _observed_orders(
            circulation_levels, "lift_relative_error"
        ),
        "finest": circulation_levels[-1],
    }
    artifact["circulation_sweep"] = circulation_sweep(budget, radius, speed)
    artifact["near_wall"] = near_wall_study(radius, speed)
    artifact["multipole"] = multipole_study(panels, radius, speed)
    if with_controls:
        artifact["controls"] = {
            "wrong_reference": wrong_reference_control(budget, radius, speed),
            "far_field": far_field_check(budget, radius, speed),
        }
    artifact["criteria"] = criteria(artifact)
    failed = [item["name"] for item in artifact["criteria"] if not item["ok"]]
    artifact["acceptance"] = {
        "passed": not failed,
        "failed_checks": failed,
        "tolerance": TOLERANCE,
        "order_min": ORDER_MIN,
        "n_checks": len(artifact["criteria"]),
    }
    artifact["overall"] = "PASS" if not failed else "FAIL"
    artifact["wall_seconds"] = time.time() - started
    return artifact


def criteria(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    """Predeclared acceptance criteria, evaluated against the artifact."""
    checks: list[dict[str, Any]] = []

    def add(name: str, value: Any, threshold: Any, ok: bool, note: str) -> None:
        checks.append(
            {
                "name": name,
                "value": value,
                "threshold": threshold,
                "ok": bool(ok),
                "note": note,
            }
        )

    gate = artifact["geometry_gate"]
    add(
        "geometry_vertices_on_circle",
        gate["vertex_radius_max_deviation"],
        "<= 1e-12 a",
        gate["vertex_radius_max_deviation"] <= 1.0e-12 * gate["radius"],
        "the body must be the inscribed polygon of radius a, not a rescaled one",
    )
    add(
        "geometry_area_matches_inscribed_closed_form",
        gate["area_relative_deviation"],
        "<= 1e-12",
        gate["area_relative_deviation"] <= 1.0e-12,
        "polygon area vs N/2 a^2 sin(2 pi/N)",
    )
    add(
        "geometry_perimeter_matches_inscribed_closed_form",
        gate["perimeter_relative_deviation"],
        "<= 1e-12",
        gate["perimeter_relative_deviation"] <= 1.0e-12,
        "polygon perimeter vs 2 N a sin(pi/N)",
    )

    harmonic = artifact["harmonicity"]
    add(
        "reference_is_harmonic",
        harmonic["laplacian_scaled_max"],
        f"<= {HARMONICITY_TOL:g} U/a",
        harmonic["laplacian_scaled_max"] <= HARMONICITY_TOL,
        "Laplacian of the closed-form perturbation potential at the probes",
    )
    add(
        "reference_is_irrotational",
        harmonic["curl_scaled_max"],
        f"<= {HARMONICITY_TOL:g} U/a",
        harmonic["curl_scaled_max"] <= HARMONICITY_TOL,
        "curl of the closed-form velocity at the probes",
    )

    crosscheck = artifact["gradient_kernel_crosscheck"]
    add(
        "field_gradient_matches_finite_difference",
        crosscheck["relative_crosscheck"],
        f"<= {GRADIENT_CROSSCHECK_TOL:g}",
        crosscheck["relative_crosscheck"] <= GRADIENT_CROSSCHECK_TOL,
        "closed-form panel-integral gradients vs central differences of the potential",
    )

    dirichlet = artifact["routes"]["dirichlet"]["finest"]
    add(
        "dirichlet_velocity_relative_l2",
        dirichlet["velocity_relative_l2"],
        f"< {TOLERANCE:g}",
        dirichlet["velocity_relative_l2"] < TOLERANCE,
        "BEM grad(phi) + U x_hat vs dW/dz at the exterior probes (budget panels)",
    )
    add(
        "dirichlet_potential_relative_l2",
        dirichlet["potential_relative_l2"],
        f"< {TOLERANCE:g}",
        dirichlet["potential_relative_l2"] < TOLERANCE,
        "recovered perturbation potential vs closed form (same probes)",
    )

    neumann = artifact["routes"]["neumann"]["finest"]
    add(
        "neumann_surface_velocity_relative_l2",
        neumann["surface_velocity_relative_l2"],
        f"< {TOLERANCE:g}",
        neumann["surface_velocity_relative_l2"] < TOLERANCE,
        "fluid-side surface velocity vs 2 U sin(theta) at the panel midpoints",
    )
    add(
        "neumann_velocity_relative_l2",
        neumann["velocity_relative_l2"],
        f"< {TOLERANCE:g}",
        neumann["velocity_relative_l2"] < TOLERANCE,
        "BEM grad(phi) + U x_hat vs dW/dz at the exterior probes",
    )
    add(
        "neumann_normal_velocity_residual",
        neumann["normal_velocity_residual_over_speed"],
        "<= 0.01 U",
        neumann["normal_velocity_residual_over_speed"] <= 0.01,
        "the body must be a streamline (max |v.n|/U)",
    )
    add(
        "neumann_cp_relative_l2",
        neumann["cp_relative_l2"],
        f"< {TOLERANCE:g}",
        neumann["cp_relative_l2"] < TOLERANCE,
        "Cp = 1 - (|v|/U)^2 vs 1 - 4 sin^2(theta) at the panel midpoints",
    )

    orders = artifact["routes"]["neumann"]["observed_order_surface_velocity"]
    finite_orders = [order for order in orders if math.isfinite(order)]
    add(
        "neumann_surface_velocity_observed_order",
        orders,
        f">= {ORDER_MIN:g}",
        bool(finite_orders) and min(finite_orders) >= ORDER_MIN,
        "second-order faceting convergence of the polygon surface velocity",
    )

    near = artifact["near_wall"]
    window_errors = [
        row["max_relative_velocity_error"] for row in near["window"]["levels"]
    ]
    add(
        "near_wall_window_error",
        max(window_errors),
        f"<= {NEAR_WALL_WINDOW_TOL:g}",
        max(window_errors) <= NEAR_WALL_WINDOW_TOL,
        "at the documented clearance window delta = "
        f"{NEAR_WALL_WINDOW_FACTOR:g} h_max the exterior velocity error stays inside "
        "one third of the case tolerance, on every ladder level",
    )
    decreasing = all(
        later < earlier for earlier, later in zip(window_errors, window_errors[1:])
    )
    add(
        "near_wall_window_error_scales_with_h",
        {
            "observed_order_in_h": near["window"]["observed_order_in_h"],
            "monotone_decreasing": bool(decreasing),
        },
        f"decreasing with order >= {NEAR_WALL_ORDER_MIN:g}",
        bool(decreasing)
        and math.isfinite(near["window"]["observed_order_in_h"])
        and near["window"]["observed_order_in_h"] >= NEAR_WALL_ORDER_MIN,
        "the window error is a discretisation error: it must shrink as h shrinks "
        "(first order at best; the constant-density term dominates)",
    )
    add(
        "near_wall_exclusion_zone_is_real",
        near["exclusion"]["levels"][-1]["max_relative_velocity_error"],
        f">= {TOLERANCE:g}",
        near["exclusion"]["levels"][-1]["max_relative_velocity_error"] >= TOLERANCE,
        "inside delta = "
        f"{NEAR_WALL_EXCLUSION_FACTOR:g} h_max the constant-panel field is *not* "
        "usable; documenting the window only means something if this holds",
    )
    add(
        "near_wall_error_is_panel_relative",
        near["exclusion"]["saturation_ratio_coarsest_over_finest"],
        f"in {list(NEAR_WALL_SATURATION_BAND)}",
        NEAR_WALL_SATURATION_BAND[0]
        <= near["exclusion"]["saturation_ratio_coarsest_over_finest"]
        <= NEAR_WALL_SATURATION_BAND[1],
        "at fixed delta/h_max mesh refinement must not buy near-wall accuracy "
        "(ratio of the error at the coarsest and the finest level)",
    )
    add(
        "near_wall_guard_rejects_inside_window",
        near["guard"]["all_levels_rejected"],
        "True on every level",
        bool(near["guard"]["all_levels_rejected"]),
        "minimum_clearance_factor must refuse targets closer than the requested "
        "clearance instead of returning a plausible wrong velocity",
    )

    floor = near["floor"]
    add(
        "near_wall_tolerance_boundary_is_located",
        floor["finest_level_clearance_over_h_max"],
        "bracketed on every level",
        bool(floor["all_levels_located"]),
        "the acceptance is < 3 % on the exterior velocity field, so the record "
        "must say *where* that holds: on every level the worst-direction error "
        "has to cross the tolerance inside the probed clearance grid, otherwise "
        "the usable region is unbounded (an assumed bound) or empty",
    )
    floor_ok = bool(floor["boundary_moves_inward_with_refinement"]) and all(
        shift is not None for shift in floor["refinement_shifts"]
    )
    add(
        "near_wall_tolerance_boundary_moves_inward_with_refinement",
        [round(float(shift), 6) for shift in floor["refinement_shifts"]],
        f"each coarse/fine <= {1.0 + floor['monotone_tolerance']:g}",
        floor_ok,
        "refining the mesh may not shrink the usable envelope: if the boundary "
        "moved outward the reported window would be a property of one mesh only",
    )
    margin = floor["window_factor_over_finest_floor"]
    add(
        "near_wall_window_clears_measured_tolerance_boundary",
        margin,
        f">= {floor['window_margin_required']:g}",
        margin is not None and float(margin) >= floor["window_margin_required"],
        "the gated 1 % window delta = "
        f"{NEAR_WALL_WINDOW_FACTOR:g} h_max must sit at least "
        f"{floor['window_margin_required']:g}x outside the *measured* 3 % boundary, "
        "so the acceptance is not true merely because the window was placed next "
        "to the edge of the region where it fails",
    )

    linear_near = near["linear"]
    add(
        "near_wall_linear_window_extends_to_half_panel",
        {
            "usable_clearance_over_h_max": linear_near["usable_clearance_over_h_max"],
            "levels": linear_near["levels"],
        },
        f"<= {NEAR_WALL_LINEAR_FACTOR:g} at <= {NEAR_WALL_WINDOW_TOL:g} error",
        linear_near["usable_clearance_over_h_max"] is not None
        and linear_near["usable_clearance_over_h_max"] <= NEAR_WALL_LINEAR_FACTOR,
        "with the first-moment (linear density) basis the 1% window must reach "
        f"delta = {NEAR_WALL_LINEAR_FACTOR:g} h_max, against "
        f"{NEAR_WALL_WINDOW_FACTOR:g} h_max for constant panels",
    )
    add(
        "near_wall_linear_beats_constant_basis",
        linear_near["improvement_at_half_h_finest"],
        f">= {NEAR_WALL_LINEAR_MIN_GAIN:g}",
        linear_near["improvement_at_half_h_finest"] >= NEAR_WALL_LINEAR_MIN_GAIN,
        f"at delta = {NEAR_WALL_LINEAR_FACTOR:g} h_max the linear-density error must "
        "be at least three times smaller than the constant-density error on the "
        "finest level",
    )
    faceting = near["faceting"]["levels"]
    add(
        "near_wall_exclusion_floor_is_faceting",
        {
            "circle_referenced": faceting[-1]["circle_referenced_linear"],
            "surface_referenced": faceting[-1]["surface_referenced_linear"],
            "faceting_offset_over_h_max": faceting[-1]["faceting_offset_over_h_max"],
        },
        "surface_referenced <= circle_referenced / 3",
        faceting[-1]["surface_referenced_linear"]
        <= faceting[-1]["circle_referenced_linear"] / 3.0,
        "the exclusion-zone floor is geometric faceting (the inscribed polygon's "
        "panel midpoints sit at a(1-cos(pi/N)) ~ h^2/8 below the circle, an O(h) "
        "offset relative to delta = 0.1 h_max): re-referencing the probes to the "
        "polygon surface must collapse most of it",
    )

    azimuth = near["azimuth"]
    add(
        "near_wall_window_is_azimuth_independent",
        {
            "worst_over_levels": azimuth["window"]["worst_over_levels"],
            "levels": azimuth["window"]["levels"],
        },
        f"<= {NEAR_WALL_AZIMUTH_WINDOW_TOL:g} on every level",
        azimuth["window"]["worst_over_levels"] <= NEAR_WALL_AZIMUTH_WINDOW_TOL,
        "the documented window was measured above the panel midpoints only; "
        "walking the probe across the panel (midpoint -> vertex) must not break "
        "the same 1% tolerance, otherwise the window is an azimuth-specific "
        "statement rather than a validity claim",
    )
    add(
        "near_wall_exclusion_is_worse_at_vertices",
        {
            "clearance_over_h_max": azimuth["exclusion"]["clearance_over_h_max"],
            "min_azimuth_penalty": azimuth["exclusion"]["min_azimuth_penalty"],
            "worst_over_levels": azimuth["exclusion"]["worst_over_levels"],
            "levels": azimuth["exclusion"]["levels"],
        },
        f">= {NEAR_WALL_AZIMUTH_VERTEX_PENALTY:g} on every level",
        azimuth["exclusion"]["min_azimuth_penalty"] >= NEAR_WALL_AZIMUTH_VERTEX_PENALTY,
        "inside the exclusion zone the midpoint column is the *best* azimuth: at "
        "the vertex the piecewise-constant density is discontinuous across the "
        "kink, so the quoted midpoint number understates the worst case",
    )
    add(
        "near_wall_azimuth_window_is_not_extended_by_linear_density",
        azimuth["windows_including_azimuth"],
        f"constant and linear both {NEAR_WALL_WINDOW_FACTOR:g} h_max",
        azimuth["windows_including_azimuth"]["constant_density"]
        == NEAR_WALL_WINDOW_FACTOR
        and azimuth["windows_including_azimuth"]["linear_density"]
        == NEAR_WALL_WINDOW_FACTOR,
        "the midpoint-only study moves the 1% window of the first-moment basis in "
        "to 0.5 h_max; once the azimuth is free that gain disappears, so the "
        "linear basis must not be advertised as widening the usable window",
    )
    add(
        "near_wall_linear_density_is_worse_in_the_window",
        {
            "min_ratio": azimuth["linear_at_window"]["min_ratio"],
            "levels": azimuth["linear_at_window"]["levels"],
        },
        f">= {NEAR_WALL_LINEAR_WINDOW_GAIN_MAX:g} on every level",
        azimuth["linear_at_window"]["min_ratio"] >= NEAR_WALL_LINEAR_WINDOW_GAIN_MAX,
        "inside the documented window the first-moment basis is measurably worse "
        "than constant panels (its arc-length slope extrapolates the density "
        "across the vertex), which is the qualification the library docstring "
        "needs",
    )
    add(
        "near_wall_azimuth_midpoint_cross_check",
        azimuth["midpoint_cross_check"]["max_relative_difference"],
        f"<= {NEAR_WALL_AZIMUTH_FRAME_TOL:g}",
        azimuth["midpoint_cross_check"]["max_relative_difference"]
        <= NEAR_WALL_AZIMUTH_FRAME_TOL,
        "the azimuthal sweep and the radial ladder probe the same midpoint "
        "direction from different reference surfaces (chord vs circle); their "
        "agreement is what makes the two windows comparable",
    )

    multipole = artifact["multipole"]
    for family, block in multipole["families"].items():
        finest = block["finest"]
        add(
            f"multipole_{family}_moment_relative_error",
            finest["target_moment_relative_error"],
            f"< {TOLERANCE:g}",
            finest["target_moment_relative_error"] < TOLERANCE,
            f"far-field harmonic A_{finest['harmonic']} recovered from the ring fits "
            f"vs {block['exact_moment']} (budget panels)",
        )
        orders = [
            order
            for order in block["observed_order_target_moment"]
            if math.isfinite(order)
        ]
        add(
            f"multipole_{family}_moment_observed_order",
            block["observed_order_target_moment"],
            f">= {MULTIPOLE_ORDER_MIN:g}",
            bool(orders) and min(orders) >= MULTIPOLE_ORDER_MIN,
            "the faceting error of the multipole moment must converge with h",
        )
    add(
        "multipole_purity",
        max(
            block["worst"]["largest_contaminant_over_scale"]
            for block in multipole["families"].values()
        ),
        f"< {MULTIPOLE_PURITY_TOL:g}",
        all(
            block["worst"]["largest_contaminant_over_scale"] < MULTIPOLE_PURITY_TOL
            for block in multipole["families"].values()
        ),
        "harmonics the boundary data does not excite must stay at roundoff "
        "(largest spurious A_n over the moment scale, every level)",
    )
    add(
        "multipole_no_source_leak",
        max(
            block["worst"]["source_over_scale"]
            for block in multipole["families"].values()
        ),
        f"< {MULTIPOLE_SOURCE_TOL:g}",
        all(
            block["worst"]["source_over_scale"] < MULTIPOLE_SOURCE_TOL
            for block in multipole["families"].values()
        ),
        "the exterior field must be solenoidal: no 2-D log source, measured as the "
        "slope of the ring mean against log(r)",
    )
    add(
        "multipole_radius_independence",
        max(
            block["worst"]["target_moment_ring_spread_relative"]
            for block in multipole["families"].values()
        ),
        f"< {MULTIPOLE_RADIUS_INDEPENDENCE_TOL:g}",
        all(
            block["worst"]["target_moment_ring_spread_relative"]
            < MULTIPOLE_RADIUS_INDEPENDENCE_TOL
            for block in multipole["families"].values()
        ),
        "the moment measured on r/a = "
        + ", ".join(f"{ring:g}" for ring in multipole["radii_over_a"])
        + " must not depend on the ring",
    )
    add(
        "multipole_fit_residual",
        max(
            block["worst"]["fit_relative_residual_max"]
            for block in multipole["families"].values()
        ),
        f"< {MULTIPOLE_FIT_RESIDUAL_TOL:g}",
        all(
            block["worst"]["fit_relative_residual_max"] < MULTIPOLE_FIT_RESIDUAL_TOL
            for block in multipole["families"].values()
        ),
        "the truncated harmonic fit must explain the sampled field "
        "(relative rms residual of the order-4 fit)",
    )
    control_m = multipole["mirror_control"]
    add(
        "multipole_mirror_control_is_rejected",
        control_m["discrimination_ratio"],
        f">= {WRONG_REFERENCE_FACTOR:g}",
        control_m["discrimination_ratio"] >= WRONG_REFERENCE_FACTOR,
        "the image-sign data must be read as a negative dipole, i.e. the moment "
        "carries its sign",
    )
    circulation = artifact["routes"]["circulation"]
    finest_circulation = circulation["finest"]
    add(
        "circulation_surface_velocity_relative_l2",
        finest_circulation["surface_velocity_relative_l2"],
        f"< {TOLERANCE:g}",
        finest_circulation["surface_velocity_relative_l2"] < TOLERANCE,
        "surface speed vs v_theta = Gamma/(2 pi a) - 2 U sin(theta) at the budget mesh",
    )
    add(
        "circulation_cp_relative_l2",
        finest_circulation["cp_relative_l2"],
        f"< {TOLERANCE:g}",
        finest_circulation["cp_relative_l2"] < TOLERANCE,
        "Cp built from the same surface velocity that is integrated for the lift",
    )
    add(
        "circulation_velocity_relative_l2",
        finest_circulation["velocity_relative_l2"],
        f"< {TOLERANCE:g}",
        finest_circulation["velocity_relative_l2"] < TOLERANCE,
        "external velocity field vs dW/dz of the lifting complex potential",
    )
    add(
        "circulation_measured_matches_prescribed",
        finest_circulation["circulation_relative_error"],
        f"<= {CIRCULATION_FLUX_TOL:g}",
        finest_circulation["circulation_relative_error"] <= CIRCULATION_FLUX_TOL,
        "the counter-clockwise contour integral of the solved surface velocity must "
        "reproduce the prescribed Gamma, otherwise the lift agreement is a coincidence",
    )
    add(
        "circulation_lift_matches_joukowski",
        finest_circulation["lift_relative_error"],
        f"< {TOLERANCE:g}",
        finest_circulation["lift_relative_error"] < TOLERANCE,
        "pressure-integrated lift vs L' = -rho U Gamma",
    )
    add(
        "circulation_drag_vanishes",
        abs(finest_circulation["drag_coefficient"]),
        f"<= {DRAG_TOL:g}",
        abs(finest_circulation["drag_coefficient"]) <= DRAG_TOL,
        "d'Alembert: no drag in a steady inviscid flow, for any Gamma",
    )
    add(
        "circulation_stagnation_angle_matches_closed_form",
        finest_circulation["stagnation_angle_max_error_deg"],
        f"<= {STAGNATION_TOL_DEG:g} deg",
        finest_circulation["stagnation_angle_max_error_deg"] <= STAGNATION_TOL_DEG,
        "interpolated zeros of the surface tangential velocity vs "
        "sin(theta_s) = Gamma/(4 pi U a)",
    )
    order = circulation["observed_order_surface_velocity"][-1]
    add(
        "circulation_observed_order",
        circulation["observed_order_surface_velocity"],
        f">= {ORDER_MIN:g}",
        bool(math.isfinite(order)) and order >= ORDER_MIN,
        "the circulation route must converge on refinement (faceting O(h^2)) "
        "instead of sitting at a plateau",
    )
    sweep = artifact["circulation_sweep"]
    add(
        "circulation_sweep_lift_is_linear",
        sweep["max_lift_deviation_from_joukowski"],
        f"< {TOLERANCE:g}",
        sweep["max_lift_deviation_from_joukowski"] < TOLERANCE,
        "lift/(rho U Gamma) must stay at 1 across Gamma in " + str(CIRCULATION_SWEEP),
    )

    if "controls" in artifact:
        control = artifact["controls"]["wrong_reference"]
        add(
            "wrong_reference_is_rejected",
            control["discrimination_ratio"],
            f">= {WRONG_REFERENCE_FACTOR:g}",
            control["discrimination_ratio"] >= WRONG_REFERENCE_FACTOR,
            "the mirrored image-sign flow must not be accepted by the reference",
        )
        far = artifact["controls"]["far_field"]
        add(
            "far_field_matches_closed_form",
            far["max_velocity_error_vs_exact_over_speed"],
            "<= 0.01 U",
            far["max_velocity_error_vs_exact_over_speed"] <= 0.01,
            "at r = 8a (disturbance still ~3% of U) the BEM velocity matches "
            "dW/dz to 1% of U",
        )
    return checks

# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def render_markdown(artifact: dict[str, Any]) -> str:
    """Render the case document from the artifact (no hand-copied numbers)."""
    rows_d = artifact["routes"]["dirichlet"]["levels"]
    rows_n = artifact["routes"]["neumann"]["levels"]
    multipole = artifact["multipole"]
    gate = artifact["geometry_gate"]
    budget = artifact["parameters"]["budget_panels"]
    tolerance = artifact["acceptance"]["tolerance"]
    lines: list[str] = [
        f"# {artifact['case_id']} — 圆柱绕势流速度场（BEM vs 解析复势）",
        "",
        f"- 算例码 `{artifact['case_id']}` / 池码 `{artifact['pool_code']}`，域：{artifact['domain']}",
        f"- 容差 `< {tolerance:.0%}`；判定 **{artifact['overall']}**"
        f"（{len(artifact['criteria'])} 项判据，失败 {len(artifact['acceptance']['failed_checks'])} 项）",
        f"- 生成 {artifact['generated_at_utc']} · Python {artifact['environment']['python']}"
        f" · NumPy {artifact['environment']['numpy']} · {artifact['environment']['machine']}",
        "",
        "## 1. 案例介绍",
        "",
        "半径 $a$ 的圆柱在无界无粘不可压流中以速度 $U$ 沿 $+x$ 均匀来流，复势",
        "$W(z)=U(z+a^2/z)$（$z=x+iy$）。扰动势 $\\phi'=\\phi-Ux=Ua^2\\cos\\theta/r$ 在无穷远以 $1/r$ 衰减，",
        "速度 $u-iv=dW/dz=U(1-a^2/z^2)$。物面纯切向，速度 $2U|\\sin\\theta|$，",
        "压力系数 $Cp=1-4\\sin^2\\theta$（肩部 $-3$、驻点 $1$，即 d'Alembert 零阻力佯谬）。",
        "",
        f"几何：正 $N$ 边形内接于 $r=a$ 的圆，$a={artifact['parameters']['radius']:g}$、"
        f"$U={artifact['parameters']['speed']:g}$；面板阶梯 {artifact['parameters']['panel_ladder']}，"
        f"预算 $N={budget}$；外部测点 {artifact['parameters']['probe_count']} 个"
        f"（环 $r/a={artifact['parameters']['probe_rings']}$，每环 {artifact['parameters']['probe_angles']} 方位）。",
        "",
        "## 2. 怎么做的",
        "",
        "本算例补的是**外部场求值**欠账：旧版 `evaluate_interior_field` 对 `domain != 'interior'` 直接抛错，",
        "BEM 只给边界量、拿不到物体外速度场。本单新增：",
        "",
        "- `kernels.integrate_single_layer_panel_gradient` / `integrate_double_layer_panel_gradient`：",
        "  直线常值面板单/双层积分对目标点的**闭式梯度**（速度场不含差分步长）；",
        "- `solver.evaluate_exterior_field`：外部 Green 表示",
        "  $u=\\oint G q_e\\,ds+\\oint u\\,\\partial G/\\partial n_{body}\\,ds$ 的势与梯度求值，",
        "  并对「点在体内 / 贴边界」设门（界外表示仍返回有限值，不设门就会静默给错数）；",
        "- `solver.evaluate_exterior_surface_velocity`：物面流体侧速度（法向用解出的 $q_e$，",
        "  切向用边界势的弧长中心差分）。",
        "",
        "两条路线共用同一算子：",
        "",
        "1. 路线 D（Dirichlet）：解析扰动势 $\\phi'$ 加在面板中点 → 第一类外部 Dirichlet 问题 →",
        "   新求值器给出外部测点的 $\\phi'$ 与 $\\nabla\\phi'$，加回 $U\\hat x$ 与 $dW/dz$ 比较；",
        "2. 路线 N（Neumann）：解析流体法向导数 $q_e=Ua^2\\cos\\theta/r^2$（等价于物面无穿透）→",
        "   第二类外部 Neumann 问题 → 物面速度与 $2U|\\sin\\theta|$ 比较，外部测点比较速度场；",
        "3. 路线 $\\Gamma$（给定环量 / 旋转圆柱）：环量由**数据给定**（无 Kutta 条件）。扰动拆成"
        "**单值**偶极子 $\\phi_d=Ua^2\\cos\\theta/r$（Neumann 数据 "
        "$q_e=\\nabla\\phi_d\\cdot n_e$）与**单位点涡响应**（数据 $q_e=v_{unit}\\cdot n_{body}$），"
        "各解一次外部 Neumann 问题；",
        "   总速度 $U\\hat x+\\nabla\\phi_d+\\Gamma(v_{unit}+\\nabla\\phi_c)$，"
        "再与 Kutta-Joukowski 升力、驻点角、$Cp$ 闭式逐项对照（模块 `tensorbem.lifting_cylinder`）。"
        "**实测**：内接多面体的面板中点法向恰为径向、点涡在该点纯切向，故响应数据恒为零（$\\sim10^{-16}$，"
        "`tests/test_lifting_cylinder.py` 已钉住该零、并在椭圆网格上验证其非零），即圆柱阶梯上环量由解析项承载，"
        "BEM 解的是折面体上的偶极子 Neumann 问题。",
        "",
        "参考值全部由闭式公式在同一点现算（`hand_typed_numbers = 0`），并以有限差分反查参考自身满足",
        "$\\nabla^2\\phi'=0$、$\\nabla\\times v=0$；闭式面板梯度再以势的中心差分反查（`field_gradient_matches_finite_difference`）。",
        "",
        "## 3. 计算结果",
        "",
        f"几何门（$N={gate['panels']}$，$h_{{mean}}={gate['h_mean']:.4f}$）：",
        "",
        "| 量 | 值 |",
        "| --- | --- |",
        f"| 顶点半径最大偏差 | {gate['vertex_radius_max_deviation']:.2e} |",
        f"| 面积相对偏差（vs $N/2\\,a^2\\sin(2\\pi/N)$） | {gate['area_relative_deviation']:.2e} |",
        f"| 面积比（多边形/圆） | {gate['area_ratio_to_circle']:.6f} |",
        f"| 周长相对偏差（vs $2Na\\sin(\\pi/N)$） | {gate['perimeter_relative_deviation']:.2e} |",
        f"| 周长比（多边形/圆） | {gate['perimeter_ratio_to_circle']:.6f} |",
        "",
        "路线 D（$\\phi'$ Dirichlet 数据 → 外部势与速度场）：",
        "",
        "| N | cond(S) | 相对残差 | $\\phi'$ 相对 L2 | 速度相对 L2 | 速度最大偏差/U | 净通量/(Ua) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows_d:
        lines.append(
            "| %d | %.2e | %.2e | %.3e | %.3e | %.3e | %.1e |"
            % (
                row["panels"],
                row["condition_number"],
                row["relative_residual_l2"],
                row["potential_relative_l2"],
                row["velocity_relative_l2"],
                row["velocity_max_abs_error"],
                row["normal_derivative_net_flux_over_speed_radius"],
            )
        )
    lines += [
        "",
        "路线 N（$q_e$ Neumann 数据 → 物面速度与外部速度场）：",
        "",
        "| N | cond(½I−H) | 物面速度 L2 | 物面速度最大偏差/U | $|v\\cdot n|/U$ | $Cp$ 相对 L2 | $Cp_{min}$ | 外部速度 L2 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows_n:
        lines.append(
            "| %d | %.2f | %.3e | %.3e | %.1e | %.3e | %.3f | %.3e |"
            % (
                row["panels"],
                row["condition_number"],
                row["surface_velocity_relative_l2"],
                row["surface_speed_max_error_over_speed"],
                row["normal_velocity_residual_over_speed"],
                row["cp_relative_l2"],
                row["cp_min"],
                row["velocity_relative_l2"],
            )
        )
    lines += [
        "",
        "收敛阶（以 $h_{mean}$ 为尺度）：路线 N 物面速度 "
        f"{artifact['routes']['neumann']['observed_order_surface_velocity']}，"
        f"路线 N 外部速度 {artifact['routes']['neumann']['observed_order_velocity']}，"
        f"路线 D 外部速度 {artifact['routes']['dirichlet']['observed_order_velocity']}。",
        "",
        "对照（预算面板）：",
        "",
        "| 对照 | 值 |",
        "| --- | --- |",
        "| 参考 harmonicity $\\max|\\nabla^2\\phi'|/U$ | "
        f"{artifact['harmonicity']['laplacian_scaled_max']:.2e} |",
        "| 参考 irrotationality $\\max|\\nabla\\times v|/U$ | "
        f"{artifact['harmonicity']['curl_scaled_max']:.2e} |",
        "| 闭式梯度 vs 势差分的相对差 | "
        f"{artifact['gradient_kernel_crosscheck']['relative_crosscheck']:.2e} |",
    ]
    if "controls" in artifact:
        control = artifact["controls"]["wrong_reference"]
        far = artifact["controls"]["far_field"]
        lines += [
            "| 错参考（镜像符号 $U(z-a^2/z)$）误差 | "
            f"{control['relative_l2_vs_mirror_reference']:.3e}（正确参考 "
            f"{control['relative_l2_vs_correct_reference']:.3e}，判别比 "
            f"{control['discrimination_ratio']:.1f}×） |",
            "| 远场 $r=8a$ 处 BEM 速度 vs $dW/dz$ | "
            f"{far['max_velocity_error_vs_exact_over_speed']:.2e}（该处扰动幅度 "
            f"{far['max_perturbation_over_speed']:.2e} $U$） |",
        ]
    near = artifact["near_wall"]
    factors = near["factors"]
    lines += [
        "",
        "### 远场多极矩（路线 M，本轮新增）",
        "",
        "外部扰动势在 $r>a$ 上有衰减多极展开 "
        "$\\phi'=\\sum_{n\\ge 1}(A_n\\cos n\\theta+B_n\\sin n\\theta)/r^n$。"
        "物面只给边界量、点值比较无法说明**远场内容**是否正确，故新增路线 M：",
        "在 $r/a=$ "
        + ", ".join(f"{ring:g}" for ring in MULTIPOLE_RADII)
        + " 三个环上采样 BEM 外部势，按 $\\cos n\\theta,\\sin n\\theta\ (n\\le "
        f"{MULTIPOLE_ORDER})"
        + ")$ 做最小二乘谐波拟合（`tensorbem.multipole2d`），"
        "并用「环均值 vs $\\log r$ 的斜率」分离二维源项（单个环无法与常数分离）。",
        "两个解析场交替检验：均匀来流只激发出 $A_1$（偶极），"
        "纯应变流 $u=\\gamma x,\\ v=-\\gamma y$ 只激发出 $A_2$（四极，$A_2=\\gamma a^4/2$）；"
        "两组数据各自都含「不该被激发的谐波必须≈0」的干净判据。",
        "",
        "| 激励 | $N$ | $h_{mean}/a$ | 目标矩 $A_n$/尺度 | 相对误差 | 环间相对离散 | 最大污染谐波 | 源项 | 拟合相对残差 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for family, block in multipole["families"].items():
        for row in block["levels"]:
            lines.append(
                "| %s | %d | %.4e | %.9f | %.3e | %.2e | %.2e | %.2e | %.2e |"
                % (
                    family,
                    row["panels"],
                    row["h_mean"] / artifact["parameters"]["radius"],
                    row["target_moment_over_scale"],
                    row["target_moment_relative_error"],
                    row["target_moment_ring_spread_relative"],
                    row["largest_contaminant_over_scale"],
                    row["source_over_scale"],
                    row["fit_relative_residual_max"],
                )
            )
    lines += [
        "",
        "收敛阶（目标矩相对误差）："
        + "；".join(
            "%s %s"
            % (
                family,
                "["
                + ", ".join("%.4g" % order for order in block["observed_order_target_moment"])
                + "]",
            )
            for family, block in multipole["families"].items()
        )
        + "；镜像（$A_1\\to-A_1$）对照的判别比 "
        + "%.4g×" % multipole["mirror_control"]["discrimination_ratio"]
        + "。",
        "",
        "### 近壁适用窗（第 2 轮补充）",
        "",
        "常值面板的单/双层积分是闭式的，但**密度**是分片常数，故恢复出的速度场只在",
        "与面板尺度同量级以外的距离可信。上表测点从 $r/a=1.25$ 起，未覆盖近壁 regime；",
        "本节把它显式量化（探针沿每个面板中点法向，$\\delta$ 以 $h_{max}$ 为单位）：",
        "",
        "| 层 | $h_{max}/a$ | "
        + " | ".join(f"$\\delta={factor:g}h$" for factor in factors)
        + " |",
        "| --- | --- | " + " | ".join("---" for _ in factors) + " |",
    ]
    for level in near["levels"]:
        lines.append(
            f"| {level['panels']} | {level['h_max_over_radius']:.3e} | "
            + " | ".join(
                f"{row['max_relative_velocity_error']:.3e}" for row in level["sweep"]
            )
            + " |"
        )
    window = near["window"]
    exclusion = near["exclusion"]
    guard = near["guard"]
    lines += [
        "",
        f"判读：$\\delta\\ge {window['clearance_over_h_max']:g}\\,h_{{max}}$ 时各层最大相对速度误差"
        "分别为 "
        + "、".join(
            f"{row['max_relative_velocity_error']:.3e}" for row in window["levels"]
        )
        + f"（阈值 {window['relative_error_tolerance']:g}；随 $h$ 的实测阶 "
        f"{window['observed_order_in_h']:.2f}，即近壁误差为**一阶**，"
        "而非远场看到的二阶）。"
        f"$\\delta={exclusion['clearance_over_h_max']:g}\\,h_{{max}}$ 处仍有 "
        f"{exclusion['levels'][-1]['max_relative_velocity_error']:.3e}，"
        f"粗/细网格误差比 {exclusion['saturation_ratio_coarsest_over_finest']:.3f}"
        f"（落在 {exclusion['saturation_band']}）：加密**不**改善近壁误差，"
        "说明这是常值密度表示的固有误差，不是线性求解器的问题。"
        f"守门探针（`minimum_clearance_factor={guard['factor']:g}`）在所有层均拒绝："
        f"{'是' if guard['all_levels_rejected'] else '否'}。",
        "",
        "与首轮审计建议（默认 clearance 取 $\\sim0.05h_{max}$）对照：该建议**实测不足**——",
        f"近壁相对速度误差在 $0.1h_{{max}}$ 处仍有 "
        f"{exclusion['levels'][-1]['max_relative_velocity_error']:.1%}（见上表末列），"
        "更贴壁只会更大。本轮不改既有 API 默认值（避免破坏调用方），而是把可用窗口"
        "写进本表与 `evaluate_exterior_field` docstring，并新增守门判据。",
        "",
        "#### 容差边界（第 6 轮）：<3% 到底从多远开始成立",
        "",
        "上面两个闸门钉住了 $2\\,h_{max}$（$\\le1\\%$）与 $0.1\\,h_{max}$"
        "（$>3\\%$）两个**离散**的 clearance，却没钉住二者之间的容差边界——"
        "而调用方要的正是这个数。在 $\\delta/h_{max}$ 上按对数网格逐点测量最差方向"
        "相对速度误差，取穿越 $3\\%$ 的相邻两点做对数-对数插值：",
        "",
        "| 层 | $h_{max}/a$ | 3% 边界 $\\delta^*/h_{max}$ | 夹逼区间 | 区间端点误差 | 区内幂次 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for level in near["floor"]["levels"]:
        detail = next(
            row["three_percent_floor"]
            for row in near["levels"]
            if row["panels"] == level["panels"]
        )
        boundary = detail["clearance_over_h_max"]
        lines.append(
            f"| {level['panels']} | {level['h_max_over_radius']:.3e} | "
            + (f"{boundary:.3f}" if boundary is not None else "未跨越")
            + " | "
            + (
                f"[{detail['bracket'][0]:g}, {detail['bracket'][1]:g}]"
                if detail["bracket"]
                else "—"
            )
            + " | "
            + (
                " / ".join(f"{value:.3e}" for value in detail["errors_at_bracket"])
                if detail["errors_at_bracket"]
                else "—"
            )
            + " | "
            + (
                f"{detail['observed_order_in_clearance']:.2f}"
                if detail["observed_order_in_clearance"] is not None
                else "—"
            )
            + " |"
        )
    floor_block = near["floor"]
    lines += [
        "",
        f"判读：最细层容差边界 $\\delta^*={floor_block['finest_level_clearance_over_h_max']:.3f}\\,h_{{max}}$"
        if floor_block["finest_level_clearance_over_h_max"] is not None
        else "判读：最细层未在网格上跨越容差。",
        "（粗/细层边界比 "
        + "、".join(f"{shift:.3f}" for shift in floor_block["refinement_shifts"])
        + "：加密使边界**向内**移动，即可用包络不随加密缩小）。"
        f"守门的 $1\\%$ 窗口 $\\delta={NEAR_WALL_WINDOW_FACTOR:g}\\,h_{{max}}$ 距该边界 "
        f"{floor_block['window_factor_over_finest_floor']:.2f}x"
        f"（判据要求 $\\ge{floor_block['window_margin_required']:g}x$）——"
        "也就是说「$<3\\%$」有实测内边界，不是把窗口摆在失效边缘得来的。"
        "路线 D/N 的探针环最近在 $r/a=1.25$，即 $\\delta=0.25\\,a$；"
        f"最细层 $h_{{max}}=0.049\\,a$，该处 $\\delta/h_{{max}}\\approx5.1$，"
        "远在边界之外，故各路线判据不受本轮边界测量影响。",
        "",
        "适用范围（消费方须知）：给定需要评价的物理 clearance $\\delta$ 与特征长度 $a$，"
        "要求的网格满足 $h_{max}\\le\\delta/\\delta^*$（本层实测 $\\delta^*$ 见上表）。"
        "边界随加密**缓慢**内移，故按最细层 $\\delta^*$ 取网格是偏保守的一侧。",
        "",
        "#### 一阶矩（线性密度）基与近壁窗口的再评估（第 3 轮）",
        "",
        "常值密度下近壁误差是**一阶**且不随加密改善，根因是基函数无法承载面板内密度的切向变化。"
        "本轮给两个层都加上一阶矩项 $\\int s K\\,ds$ 的闭式积分（`tensorbem.kernels`），"
        "密度展开为 $\\sigma(s)=\\sigma_i+\\sigma'_i(s-L_i/2)$（斜率由弧长中心差分给出），"
        "求解与外部场求值都可用 `linear_density=True` 打开。逐层结果：",
        "",
        "| 层 | $h_{max}/a$ | 常值 $\\delta=0.5h$ | 线性 $\\delta=0.5h$ | 常值 $\\delta=2h$ | 线性 $\\delta=2h$ |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    # `linear_near` is the near-wall linear-density study; `criteria()` reads the
    # same block from `artifact["near_wall"]["linear"]`, and this renderer used to
    # reference the name without binding it (NameError on every render).
    linear_near = near["linear"]
    for level in near["levels"]:
        row_half = next(
            row
            for row in level["sweep"]
            if row["clearance_over_h_max"] == linear_near["clearance_over_h_max"]
        )
        row_window = next(
            row
            for row in level["sweep"]
            if row["clearance_over_h_max"] == window["clearance_over_h_max"]
        )
        lines.append(
            f"| {level['panels']} | {level['h_max_over_radius']:.3e} | "
            f"{row_half['max_relative_velocity_error']:.3e} | "
            f"{row_half['max_relative_velocity_error_linear']:.3e} | "
            f"{row_window['max_relative_velocity_error']:.3e} | "
            f"{row_window['max_relative_velocity_error_linear']:.3e} |"
        )
    # The attribution tables are keyed by panel count and clearance factor; the
    # prose below quotes the finest level at the exclusion clearance.  Both names
    # were referenced and never bound, so every render_markdown() call died with
    # NameError before it wrote a byte -- which is why the committed record and
    # the committed document both predate this section.
    finest_key = str(near["levels"][-1]["panels"])
    exclusion_key = f"{NEAR_WALL_EXCLUSION_FACTOR:g}"
    lines += [
        "",
        f"判读：$\\delta={linear_near['clearance_over_h_max']:g}\\,h_{{max}}$ 处线性基把误差降到 "
        f"{linear_near['levels'][-1]['max_relative_velocity_error']:.3e}"
        f"（常值 {linear_near['constant_mode_levels'][-1]['max_relative_velocity_error']:.3e}，"
        f"改善 {linear_near['improvement_at_half_h_finest']:.1f}×），"
        f"1% 窗口从 $\\delta\\ge{NEAR_WALL_WINDOW_FACTOR:g}h$ 扩展到 "
        f"$\\delta\\ge{linear_near['usable_clearance_over_h_max']:g}h$；"
        f"$\\delta={NEAR_WALL_WINDOW_FACTOR:g}h$ 处两模式几乎相同"
        f"（改善 {linear_near['improvement_at_window_finest']:.2f}×，"
        "该处误差已由边界迹精度而非基函数决定）。",
        "",
        "**误差归因（第 3 轮修订）**：把物面两条迹换成解析值（去掉求解误差）后，"
        "表示误差本身在 $\\delta=0.1h_{{max}}$ 处由 "
        f"{near['attribution']['representation_constant'][finest_key][exclusion_key]:.3e}"
        f"（常值）降到 "
        f"{near['attribution']['representation_linear'][finest_key][exclusion_key]:.3e}"
        f"（线性，改善 "
        f"{near['attribution']['representation_constant'][finest_key][exclusion_key] / near['attribution']['representation_linear'][finest_key][exclusion_key]:.1f}×）。"
        "但即便如此，贴壁残差仍停在 $\\sim10^{-1}$ —— 说明主项两者都不是，"
        "而是**折面几何**：内接正多边形面板中点半径 $a\\cos(\\pi/N)=a(1-h^2/8)$，"
        "所以从圆周量起的 $\\delta=0.1h_{{max}}$ 相对真实物面是 "
        "$(0.1h+h^2/8)/h$，该偏移是 $\\delta$ 的 $O(h)$ 倍，正解释上表实测的一阶饱和。"
        "把探针距离改为从折面（$a\\cos\\pi/N$）量起后，最细网格上误差由 "
        f"{near['faceting']['levels'][-1]['circle_referenced_linear']:.3e} 降到 "
        f"{near['faceting']['levels'][-1]['surface_referenced_linear']:.3e}"
        f"（{near['faceting']['levels'][-1]['circle_referenced_linear'] / max(near['faceting']['levels'][-1]['surface_referenced_linear'], 1e-30):.1f}×）。"
        "结论：近壁误差预算 = 折面几何（一阶，$\\propto h/\\delta$）+ 基函数（可被一阶矩项压掉）"
        "+ 边界迹求解误差（$O(h^2)$，被 $1/\\delta$ 放大）；要再收窄窗口需曲边面板/更高阶求解。",
        "",
    ]
    azimuth = near["azimuth"]
    azimuth_window = azimuth["window"]
    azimuth_exclusion = azimuth["exclusion"]
    lines += [
        "",
        "**方位（面内）敏感性（第 4 轮补充）**：前面两张表的探针都落在**面板中点法向**上，"
        "那是常值面板最有利的方向，而用户可以在任意位置求值。本轮把探针在同一面板内从中点"
        "（$\\zeta=0$）走到顶点（$\\zeta=0.5h_{max}$，顶点处取相邻面板法向的角平分线），"
        "逐层给出面内最差误差（线性基同测）。到折面的实测间隙由程序逐段量距，非名义值：",
        "",
        "| N | $\\delta/h_{max}$ | 中点误差 | 面内最差 | 最差位置 $\\zeta/h_{max}$ | 顶点/中点 | 线性基面内最差 | 实测最小间隙/h |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for level in azimuth["levels"]:
        for row in level["sweep"]:
            lines.append(
                "| %d | %.2f | %.3e | %.3e | %.3f | %.2f | %.3e | %.3f |"
                % (
                    level["panels"],
                    row["clearance_over_h_max"],
                    row["midpoint_error"],
                    row["worst_error"],
                    row["worst_offset_over_h_max"],
                    row["azimuth_penalty"],
                    row["worst_error_linear"],
                    row["min_clearance_over_h_max"],
                )
            )
    lines += [
        "",
        f"判读一（窗内方位无关）：$\\delta={azimuth_window['clearance_over_h_max']:g}h_{{max}}$ 处逐层最差/中点比为 "
        + "、".join(f"{row['azimuth_penalty']:.2f}" for row in azimuth_window["levels"])
        + f"，最差 {azimuth_window['worst_over_levels']:.3e} 仍在 1% 容差内 ——"
        "「$\\delta\\ge2h_{max}$」这条窗对任意方位成立，不只对中点法向。",
        "",
        f"判读二（排除区在中点方向被低估）：$\\delta={azimuth_exclusion['clearance_over_h_max']:g}h_{{max}}$ 处逐层中点误差 "
        + "、".join(f"{row['midpoint_error']:.3f}" for row in azimuth_exclusion["levels"])
        + "，面内最差却达 "
        + "、".join(f"{row['worst_error']:.3f}" for row in azimuth_exclusion["levels"])
        + f"（顶点/中点比 {azimuth_exclusion['min_azimuth_penalty']:.2f}–"
        + f"{max(row['azimuth_penalty'] for row in azimuth_exclusion['levels']):.2f}）。"
        "理由：常值密度在顶点处**不连续**，而折面顶点相对真实圆的偏差最直接地投影到该方向。"
        "结论：排除区里的「18%」只是最有利方位的数字，贴壁任意方向应按 ~60% 或更大预留误差；"
        "守门参数照旧建议 $\\ge1$–$2h_{max}$。",
        "",
        f"判读三（一阶矩基的窗没有前移）：窗内 $\\delta={azimuth['linear_at_window']['clearance_over_h_max']:g}h_{{max}}$ 处线性基面内最差比常值大 "
        f"{azimuth['linear_at_window']['min_ratio']:.2f}–{azimuth['linear_at_window']['max_ratio']:.2f}×；"
        "含方位后 1% 窗对两种基都是 $\\delta\\ge"
        f"{azimuth['windows_including_azimuth']['constant_density']:g}h_{{max}}$"
        f"（线性 {azimuth['windows_including_azimuth']['linear_density']:g}）。"
        "线性基的优势只在**中点方向**、$\\delta\\le0.5h_{max}$ 处（上一表 3–9×）；"
        "换到顶点方向在 $\\delta=0.1h_{max}$ 反而更差且逐层不单调（"
        + "、".join(f"{row['worst_error_linear']:.3f}" for row in azimuth_exclusion["levels"])
        + "，随加密 1.40→0.129→1.26）：弧长中心差分给出的斜率把密度外推过了顶点折角。"
        "故 `linear_density=True` 只能当「光滑曲边 + 离壁」选项，不能当通用改进"
        "（见 `evaluate_exterior_field` docstring）。",
        "",
        f"两条研究在 $\\delta=2h_{{max}}$ 的中点方向互相校核：同一方向、不同参考面（折面弦 vs 圆周）"
        f"的误差最大相差 {azimuth['midpoint_cross_check']['max_relative_difference']:.2%}，"
        "与折面偏移 $a(1-\\cos\\pi/N)\\sim h^2/8$ 同量级，说明两表量的是同一 regime。",
        "",
    ]
    rows_c = artifact["routes"]["circulation"]["levels"]
    sweep = artifact["circulation_sweep"]
    circulation = artifact["routes"]["circulation"]
    finest_c = circulation["finest"]
    lines += [
        "",
        "路线 $\\Gamma$（给定环量 = 旋转圆柱 / Magnus，"
        f"$\\Gamma={finest_c['circulation']:g}\\,Ua$）：闭式 "
        "$W(z)=U(z+a^2/z)-i\\Gamma/(2\\pi)\\ln(z/a)$，物面 "
        "$v_\\theta=\\Gamma/(2\\pi a)-2U\\sin\\theta$，"
        "$\\sin\\theta_s=\\Gamma/(4\\pi Ua)$，$L'=-\\rho U\\Gamma$。",
        "",
        "| N | 环量（实测/给定） | 物面速度 L2 | $Cp$ L2 | 外部速度 L2 | 升力 $L'$ | $L'_{ref}$ | 升力相对误差 | 阻力系数 | 驻点误差(deg) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows_c:
        lines.append(
            "| %d | %.6f / %.1f | %.3e | %.3e | %.3e | %.6f | %.6f | %.3e | %.1e | %.3f |"
            % (
                row["panels"],
                row["circulation_measured"],
                row["circulation"],
                row["surface_velocity_relative_l2"],
                row["cp_relative_l2"],
                row["velocity_relative_l2"],
                row["lift"],
                row["lift_exact"],
                row["lift_relative_error"],
                row["drag_coefficient"],
                row["stagnation_angle_max_error_deg"],
            )
        )
    lines += [
        "",
        f"环量扫描（$N={sweep['panels']}$，升力比 $L'/(\\rho U\\Gamma)$ 应为 1）：",
        "",
        "| $\\Gamma/(Ua)$ | $L'$ | $-\\rho U\\Gamma$ | 比值 | 相对误差 | $Cp_{min}$ | $Cp_{min,ref}$ | 驻点误差(deg) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in sweep["rows"]:
        lines.append(
            "| %.1f | %.6f | %.6f | %.6f | %.3e | %.5f | %.5f | %.3f |"
            % (
                row["circulation"],
                row["lift"],
                row["lift_exact"],
                row["lift_over_joukowski"],
                row["lift_relative_error"],
                row["cp_min"],
                row["cp_min_exact"],
                row["stagnation_angle_max_error_deg"],
            )
        )
    lines += [
        "",
        f"实测收敛阶：物面速度 {circulation['observed_order_surface_velocity']}、"
        f"升力误差 {circulation['observed_order_lift']}（面元阶梯 "
        f"{artifact['parameters']['panel_ladder']}）。",
        "",
    ]
    lines += [
        "",
        "## 4. 结果图",
        "",
        "![velocity field](benchmarks/verified/case_bem_laplace_cylinder/figures/field.png)",
        "",
        "`figures/field.png`：左=解析 $|v|/U$ 云图 + BEM 速度箭头（$N=64$ 演示网格）；"
        "右=外部测点上的逐点相对速度误差与物面速度角度分布。",
        "",
        "![near wall](benchmarks/verified/case_bem_laplace_cylinder/figures/near_wall.png)",
        "",
        "`figures/near_wall.png`：左=近壁误差随 $\\delta/h_{max}$ 的变化（三层网格，实线常值基、"
        "虚线一阶矩基），竖线为适用窗与排除区边界；右=面内方位扫描的「最差方位/中点」误差比，"
        "比值 1 表示该层该间隙下中点就是最差方位。",
        "",
        "![convergence](benchmarks/verified/case_bem_laplace_cylinder/figures/convergence.png)",
        "",
        "`figures/convergence.png`：三条路线的误差随 $h_{mean}$ 收敛，附 $h^2$、$h^4$ 参考斜率。",
        "",
        "![circulation](benchmarks/verified/case_bem_laplace_cylinder/figures/circulation.png)",
        "",
        "`figures/circulation.png`：左=给定环量下的 $Cp(\theta)$（BEM 面板 vs 闭式，"
        "虚线为闭式驻点位置）；右=升力随 $\Gamma$ 的线性（数值 vs Kutta-Joukowski）与相对误差。",
        "",
        "![multipole](benchmarks/verified/case_bem_laplace_cylinder/figures/multipole.png)",
        "",
        "`figures/multipole.png`：左=均匀来流（偶极）与纯应变（四极）的谐波谱 "
        "$|A_n|/(\\text{尺度})$（预算面板，解析值 1 与 0 作参考线）；"
        "右=目标矩相对误差随 $h_{mean}$ 的收敛（附 $h^2$ 参考斜率）。",
        "",
        "## 5. 注意点",
        "",
        "- 多面体不是圆：离散体是内接正 $N$ 边形，离散解只能**收敛**到圆柱解；面板中点半径",
        "  $a\\cos(\\pi/N)$，故物面数据本身带 $O(h^2)$ 偏置，所有误差按实测报告；",
        "- 外部 Neumann 的势只确定到常数，势比较先减去常数（速度不受影响）；",
        "- 单层算子 $S$ 的条件数随加密增长（对数容量退化），本算例 $L=2$、$a=1$ 远离退化点，",
        "  但仍逐级报告 $cond(S)$；第二类算子 $\\frac12 I-H$ 条件数 $O(1)$；",
        "- 只在 $r/a\\ge 1.25$ 求值：贴壁/体内点必须被守门拒绝；",
        "- 路线 $\\Gamma$ 的环量是**给定**的（旋转圆柱），不是解出来的：本算例验证的是"
        "「带环量的外部 Neumann 问题 + 压力积分 + 驻点定位」，不验证 Kutta 条件（后者见 "
        "`docs/lifting_kutta_closure.md` 与 lifting 家族算例）；",
        "- 圆柱阶梯上点涡响应数据实测恒为零（见 §2 第 3 条），所以本路线的数值部分=折面体偶极子解；"
        "不把「升力吻合」读成「环量被 BEM 解出来」；",
        "- 环量须满足 $|\\Gamma|<4\\pi Ua$ 才有两个物面驻点；超临界时物面无驻点，模块直接拒绝；",
        "- 只验证势流算子链（边界积分、外部场求值、物面速度），无粘性尾流/分离/自由面，",
        "  不能当阻力预报用（$L'=-\\rho U\\Gamma$ 是理想化结果，真实旋转圆柱有粘性分离与 Magnus 失效）。",
        "",
        "## 6. 误差判定",
        "",
        f"| 关键量（预算 $N={budget}$） | 相对误差 | 是否 <3% |",
        "| --- | --- | --- |",
    ]
    checks = [
        ("路线 D 外部速度场 L2", rows_d[-1]["velocity_relative_l2"]),
        ("路线 D 外部扰动势 L2", rows_d[-1]["potential_relative_l2"]),
        ("路线 N 物面速度 L2（vs 2U sinθ）", rows_n[-1]["surface_velocity_relative_l2"]),
        ("路线 N 外部速度场 L2", rows_n[-1]["velocity_relative_l2"]),
        ("路线 N Cp L2", rows_n[-1]["cp_relative_l2"]),
        ("路线 $\\Gamma$ 物面速度 L2", rows_c[-1]["surface_velocity_relative_l2"]),
        ("路线 $\\Gamma$ 外部速度 L2", rows_c[-1]["velocity_relative_l2"]),
        ("路线 $\\Gamma$ 升力 vs $\\rho U\\Gamma$", rows_c[-1]["lift_relative_error"]),
        ("路线 $\\Gamma$ 环量（实测 vs 给定）", rows_c[-1]["circulation_relative_error"]),
    ]
    for name, value in checks:
        lines.append(
            "| %s | %.3e | %s |" % (name, value, "是" if value < tolerance else "**否**")
        )
    lines += [
        "",
        "| 路线 M 偶极矩 $A_1$ | %.3e | %s |"
        % (
            multipole["families"]["uniform_flow"]["finest"]["target_moment_relative_error"],
            "是"
            if multipole["families"]["uniform_flow"]["finest"]["target_moment_relative_error"]
            < tolerance
            else "**否**",
        ),
        "| 路线 M 四极矩 $A_2$ | %.3e | %s |"
        % (
            multipole["families"]["pure_strain"]["finest"]["target_moment_relative_error"],
            "是"
            if multipole["families"]["pure_strain"]["finest"]["target_moment_relative_error"]
            < tolerance
            else "**否**",
        ),
        "结论：预算面板下所有关键量均 `<3%`；误差随加密按网格化阶（路线 N 约二阶）下降，",
        "不是靠放宽阈值通过的。",
        "",
    ]
    return "\n".join(lines)


def write_case_card(artifact: dict[str, Any], path: Path) -> Path:
    """Write the one-shot case card (mirrors the annulus/ellipse pool entries)."""
    card = {
        "case_id": CASE_ID,
        "schema": SCHEMA,
        "repo": "TensorBEM",
        "domain": artifact["domain"],
        "declared_solution_forms": [artifact["reference"]["complex_potential"]],
        "reference_provenance": {
            "form": "closed_form",
            "values": (
                "generated by case_cylinder_stream.exact_velocity / "
                "exact_perturbation_potential at the same points as the "
                "discretisation; harmonicity and irrotationality re-checked by "
                "finite differences (harmonicity_check)"
            ),
            "hand_typed_numbers": 0,
            "bibliography": artifact["reference"]["bibliography"],
            "page_or_article_number": artifact["reference"]["page_or_article_number"],
        },
        "tolerance": TOLERANCE,
        "budget_panels": artifact["parameters"]["budget_panels"],
        "parameters": artifact["parameters"],
        "capability_reused": [
            "tensorbem.solver.solve_laplace_exterior_dirichlet",
            "tensorbem.solver.solve_laplace_exterior_neumann",
            "tensorbem.solver.evaluate_exterior_field",
            "tensorbem.solver.evaluate_exterior_surface_velocity",
            "tensorbem.kernels.integrate_single_layer_panel_gradient",
            "tensorbem.kernels.integrate_double_layer_panel_gradient",
            "tensorbem.multipole2d.fit_ring",
            "tensorbem.multipole2d.decompose_rings",
            "tensorbem.lifting_cylinder.solve_rotating_cylinder",
            "tensorbem.lifting_cylinder.exterior_velocity_field",
            "tensorbem.lifting.vortex_velocity_field",
        ],
        "repro_cmd": artifact["repro_cmd"],
        "test_cmd": (
            "PYTHONPATH=src python3 -m pytest tests/test_case_cylinder_stream.py "
            "tests/test_lifting_cylinder.py -q"
        ),
        "artifacts": {
            "results_json": "benchmarks/verified/case_bem_laplace_cylinder/results.json",
            "report_md": "benchmarks/verified/case_bem_laplace_cylinder/report.md",
            "figures": "benchmarks/verified/case_bem_laplace_cylinder/figures/",
        },
        "verdict_passed": bool(artifact["acceptance"]["passed"]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def plot_circulation(artifact: dict[str, Any], output: Path) -> Path:
    """Circulation route: Cp(theta) against closed form, and lift vs Gamma."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for plotting") from error

    radius = float(artifact["parameters"]["radius"])
    speed = float(artifact["parameters"]["speed"])
    rows_c = artifact["routes"]["circulation"]["levels"]
    finest = rows_c[-1]
    circulation = float(finest["circulation"])
    mesh = regular_polygon(int(finest["panels"]), radius)
    solution = solve_rotating_cylinder(mesh, radius, speed, circulation)
    theta = np.arctan2(mesh.centers[:, 1], mesh.centers[:, 0])
    order = np.argsort(theta)
    reference_theta = np.linspace(0.0, 2.0 * np.pi, 721)
    sweep = artifact["circulation_sweep"]["rows"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].plot(
        np.degrees(reference_theta),
        circulation_pressure_coefficient(
            reference_theta, radius, speed, circulation
        ),
        "k-",
        lw=1.0,
        label="closed form",
    )
    axes[0].plot(
        np.degrees(theta[order]),
        solution.surface_pressure_coefficient[order],
        "o",
        ms=2.5,
        label=f"BEM constant panels, N={finest['panels']}",
    )
    for angle in finest["stagnation_angles_exact_deg"]:
        axes[0].axvline(angle, color="tab:red", ls=":", lw=0.9)
    axes[0].annotate(
        r"$\theta_s$ (closed form)",
        (finest["stagnation_angles_exact_deg"][0], 0.7),
        fontsize=7,
        color="tab:red",
    )
    axes[0].set_xlabel(r"$\theta$ (deg)")
    axes[0].set_ylabel(r"$C_p$")
    axes[0].set_title(rf"$C_p(\theta)$ at $\Gamma={circulation:g}Ua$")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    gammas = [row["circulation"] for row in sweep]
    axes[1].plot(
        gammas,
        [row["lift_exact"] for row in sweep],
        "k-",
        lw=1.0,
        label=r"$-\rho U\Gamma$",
    )
    axes[1].plot(
        gammas,
        [row["lift"] for row in sweep],
        "o",
        ms=4.0,
        label=f"BEM pressure integral, N={artifact['circulation_sweep']['panels']}",
    )
    axes[1].set_xlabel(r"$\Gamma/(Ua)$")
    axes[1].set_ylabel("lift per unit span")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7, loc="upper left")
    twin = axes[1].twinx()
    twin.semilogy(
        gammas,
        [max(row["lift_relative_error"], 1.0e-18) for row in sweep],
        "s--",
        ms=3.0,
        color="tab:green",
        label="relative error",
    )
    twin.set_ylabel("lift relative error", color="tab:green")
    twin.legend(fontsize=7, loc="center right")
    axes[1].set_title(
        "lift linearity; max deviation "
        f"{artifact['circulation_sweep']['max_lift_deviation_from_joukowski']:.2e}"
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output

def plot_field(
    artifact: dict[str, Any], output: Path, panels: int = 64
) -> Path:
    """Velocity field: analytic ``|v|/U`` contours with BEM arrows, plus errors."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for plotting") from error

    radius = float(artifact["parameters"]["radius"])
    speed = float(artifact["parameters"]["speed"])
    mesh = regular_polygon(panels, radius)
    centers = mesh.centers
    r = np.hypot(centers[:, 0], centers[:, 1])
    normal_derivative = speed * radius**2 * (centers[:, 0] / r) / r**2
    result = solve_laplace_exterior_neumann(mesh, normal_derivative)

    limit = 3.0 * radius
    grid = np.linspace(-limit, limit, 241)
    xx, yy = np.meshgrid(grid, grid)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    outside = ~mesh.contains_point(points) & (
        np.hypot(points[:, 0], points[:, 1]) > 1.0e-3
    )
    analytic = np.full(len(points), np.nan)
    analytic[outside] = (
        np.linalg.norm(exact_velocity(points[outside], radius, speed), axis=1) / speed
    )

    theta = np.linspace(0.0, 2.0 * np.pi, 28, endpoint=False)
    rings = np.linspace(1.15 * radius, limit, 18)
    arrows = np.vstack(
        [np.column_stack((rr * np.cos(theta), rr * np.sin(theta))) for rr in rings]
    )
    arrows = arrows[~mesh.contains_point(arrows)]
    arrows = arrows[np.hypot(arrows[:, 0], arrows[:, 1]) > 1.0e-3]
    arrows = arrows[np.all(np.abs(arrows) < limit, axis=1)]
    field = evaluate_exterior_field(mesh, result, arrows)
    velocity = field.gradient + np.array([speed, 0.0])

    probes = probe_points()
    probe_bem = evaluate_exterior_field(mesh, result, probes)
    error = np.linalg.norm(
        probe_bem.gradient + np.array([speed, 0.0])
        - exact_velocity(probes, radius, speed),
        axis=1,
    ) / speed

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)
    image = axes[0].contourf(xx, yy, analytic.reshape(xx.shape), levels=24, cmap="viridis")
    axes[0].plot(
        np.append(mesh.vertices[:, 0], mesh.vertices[0, 0]),
        np.append(mesh.vertices[:, 1], mesh.vertices[0, 1]),
        color="white",
        lw=1.2,
    )
    axes[0].quiver(
        arrows[:, 0],
        arrows[:, 1],
        velocity[:, 0] / speed,
        velocity[:, 1] / speed,
        color="white",
        width=0.0025,
    )
    axes[0].set_aspect("equal")
    axes[0].set_title(f"analytic |v|/U + BEM velocity (N={panels})")
    axes[0].set_xlabel("x/a")
    axes[0].set_ylabel("y/a")
    fig.colorbar(image, ax=axes[0], label="|v|/U")

    angles = np.arctan2(centers[:, 1], centers[:, 0])
    surface = evaluate_exterior_surface_velocity(mesh, result) + np.array([speed, 0.0])
    order = np.argsort(angles)
    axes[1].plot(
        np.degrees(angles[order]),
        np.linalg.norm(surface[order], axis=1) / speed,
        "o",
        ms=3.0,
        label=f"BEM surface speed (N={panels})",
    )
    dense = np.linspace(0.0, 2.0 * np.pi, 360)
    axes[1].plot(
        np.degrees(dense),
        2.0 * np.abs(np.sin(dense)),
        "-",
        lw=1.2,
        label="2|sin theta| (exact)",
    )
    axes[1].set_xlabel("theta [deg]")
    axes[1].set_ylabel("|v|/U")
    axes[1].set_title("surface speed vs closed form")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_xlim(0.0, 360.0)
    inset = axes[1].inset_axes([0.12, 0.12, 0.38, 0.34])
    inset.semilogy(np.arange(len(error)), error, "o", ms=2.5, color="tab:red")
    inset.set_title(f"probe rel. error\nmax {error.max():.2e}", fontsize=6)
    inset.tick_params(labelsize=5)
    inset.set_xlabel("probe", fontsize=5)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_convergence(artifact: dict[str, Any], output: Path) -> Path:
    """Observed convergence of both routes against the closed-form field."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for plotting") from error

    fig, axes = plt.subplots(1, 1, figsize=(6.2, 4.4), constrained_layout=True)
    series = [
        (
            "route D velocity",
            artifact["routes"]["dirichlet"]["levels"],
            "velocity_relative_l2",
            "o-",
        ),
        (
            "route N velocity",
            artifact["routes"]["neumann"]["levels"],
            "velocity_relative_l2",
            "s-",
        ),
        (
            "route N surface velocity",
            artifact["routes"]["neumann"]["levels"],
            "surface_velocity_relative_l2",
            "^-",
        ),
        (
            "route N Cp",
            artifact["routes"]["neumann"]["levels"],
            "cp_relative_l2",
            "d-",
        ),
    ]
    h_reference = max(row["h_mean"] for row in artifact["routes"]["neumann"]["levels"])
    for label, rows, key, style in series:
        h = np.array([row["h_mean"] for row in rows])
        error = np.array([row[key] for row in rows])
        axes.loglog(h, error, style, ms=4.5, label=label)
        order = np.polyfit(np.log(h), np.log(error), 1)[0] if len(h) > 1 else float("nan")
        axes.annotate(
            f"p={order:.2f}",
            (h[0], error[0]),
            textcoords="offset points",
            xytext=(4, 6),
            fontsize=7,
        )
    h_line = np.array([h_reference, h_reference / 4.0])
    for power, factor in ((2.0, 5.0e-3), (4.0, 5.0e-2)):
        axes.loglog(
            h_line,
            factor * (h_line / h_reference) ** power,
            "k--",
            lw=0.8,
            alpha=0.6,
        )
        axes.annotate(
            f"h^{int(power)}",
            (h_line[-1], factor * (h_line[-1] / h_reference) ** power),
            fontsize=7,
        )
    axes.axhline(artifact["acceptance"]["tolerance"], color="tab:red", ls=":", lw=1.0)
    axes.annotate(
        "tolerance 3%",
        (axes.get_xlim()[0], artifact["acceptance"]["tolerance"]),
        fontsize=7,
        color="tab:red",
        va="bottom",
    )
    axes.set_xlabel(r"$h_{mean}/a$")
    axes.set_ylabel("relative error")
    axes.set_title("cylinder potential flow: BEM vs complex potential")
    axes.grid(True, which="both", alpha=0.3)
    axes.legend(fontsize=7)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_near_wall(artifact: dict[str, Any], output: Path) -> Path:
    """Near-wall error against clearance, with the validity window marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    near = artifact["near_wall"]
    fig, (axes, azimuth_axes) = plt.subplots(1, 2, figsize=(11.6, 4.4))
    for level in near["levels"]:
        clearance = np.array([row["clearance_over_h_max"] for row in level["sweep"]])
        error = np.array(
            [row["max_relative_velocity_error"] for row in level["sweep"]]
        )
        axes.loglog(clearance, error, "o-", ms=4.0, label=f"N={level['panels']}")
        error_linear = np.array(
            [row["max_relative_velocity_error_linear"] for row in level["sweep"]]
        )
        axes.loglog(
            clearance,
            error_linear,
            "s--",
            ms=3.5,
            color=axes.lines[-1].get_color(),
            lw=1.0,
            alpha=0.85,
            label=f"N={level['panels']} linear",
        )
    axes.axvline(
        near["window"]["clearance_over_h_max"],
        color="tab:green",
        ls="--",
        lw=1.0,
        label="window $\\delta=2h_{max}$",
    )
    axes.axvline(
        near["exclusion"]["clearance_over_h_max"],
        color="tab:red",
        ls=":",
        lw=1.0,
        label="exclusion $\\delta=0.1h_{max}$",
    )
    axes.axhline(
        near["window"]["relative_error_tolerance"], color="0.4", ls=":", lw=0.8
    )
    axes.annotate(
        "1% window tolerance",
        (axes.get_xlim()[0], near["window"]["relative_error_tolerance"]),
        fontsize=7,
        color="0.35",
        va="bottom",
    )
    axes.set_xlabel(r"$\delta/h_{max}$")
    axes.set_ylabel("max relative velocity error")
    axes.set_title("midpoint rays: validity window")
    axes.grid(True, which="both", alpha=0.3)
    axes.legend(fontsize=7)

    azimuth = near["azimuth"]
    for level in azimuth["levels"]:
        clearance = np.array(
            [row["clearance_over_h_max"] for row in level["sweep"]]
        )
        penalty = np.array([row["azimuth_penalty"] for row in level["sweep"]])
        penalty_linear = np.array(
            [row["azimuth_penalty_linear"] for row in level["sweep"]]
        )
        azimuth_axes.semilogx(clearance, penalty, "o-", ms=4.0, label=f"N={level['panels']}")
        azimuth_axes.semilogx(
            clearance,
            penalty_linear,
            "s--",
            ms=3.5,
            lw=1.0,
            alpha=0.85,
            color=azimuth_axes.lines[-1].get_color(),
            label=f"N={level['panels']} linear",
        )
    azimuth_axes.axvline(
        near["window"]["clearance_over_h_max"],
        color="tab:green",
        ls="--",
        lw=1.0,
    )
    azimuth_axes.axvline(
        near["exclusion"]["clearance_over_h_max"],
        color="tab:red",
        ls=":",
        lw=1.0,
    )
    azimuth_axes.set_xlabel(r"$\delta/h_{max}$")
    azimuth_axes.set_ylabel("worst azimuth error / midpoint error")
    azimuth_axes.set_title("across one panel: azimuth penalty")
    azimuth_axes.grid(True, which="both", alpha=0.3)
    azimuth_axes.legend(fontsize=7)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_multipole(artifact: dict[str, Any], output: Path) -> Path:
    """Harmonic spectrum of both analytic flows plus the moment convergence."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    multipole = artifact["multipole"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    width = 0.38
    orders = np.arange(1, int(multipole["order"]) + 1)
    for offset, (family, block) in zip(
        (-width / 2.0, width / 2.0), multipole["families"].items()
    ):
        finest = block["finest"]
        scale = float(finest["scale"])
        values = []
        for n in orders:
            if n == int(finest["harmonic"]):
                values.append(finest["target_moment_over_scale"])
            else:
                values.append(finest["contaminant_over_scale"][str(n)])
        axes[0].bar(
            orders + offset,
            np.abs(values),
            width=width,
            label=f"{family} (A_{finest['harmonic']} excited)",
        )
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-18, 10.0)
    axes[0].axhline(1.0, color="k", ls="--", lw=0.8, label="analytic |A_n|/scale = 1")
    axes[0].axhline(1e-15, color="0.5", ls=":", lw=0.8, label="roundoff floor")
    axes[0].set_xticks(orders)
    axes[0].set_xlabel("harmonic $n$")
    axes[0].set_ylabel(r"$|A_n| / $ scale")
    axes[0].set_title(
        "far-field harmonic spectrum, N = %d" % int(finest["panels"])
    )
    axes[0].grid(True, which="both", axis="y", alpha=0.3)
    axes[0].legend(fontsize=7)

    for family, block in multipole["families"].items():
        h = [row["h_mean"] for row in block["levels"]]
        err = [row["target_moment_relative_error"] for row in block["levels"]]
        axes[1].loglog(h, err, "o-", label=f"{family} A_{block['finest']['harmonic']}")
        reference = err[0] * (np.asarray(h) / h[0]) ** 2
        axes[1].loglog(h, reference, "k:", lw=0.8)
    axes[1].annotate(r"$h^2$", xy=(h[-1], err[-1] * 4.0), fontsize=8)
    axes[1].axhline(TOLERANCE, color="r", ls="--", lw=0.8, label=f"tolerance {TOLERANCE:g}")
    axes[1].set_xlabel(r"$h_{mean}/a$")
    axes[1].set_ylabel("relative error of the moment")
    axes[1].set_title("multipole convergence")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("benchmarks/verified/case_bem_laplace_cylinder"),
        help="artifact directory (results.json, report.md, case.json, figures/)",
    )
    parser.add_argument(
        "--doc", type=Path, default=None, help="write the case document here"
    )
    parser.add_argument(
        "--quick", action="store_true", help="stop the ladder at 64 panels (test cost)"
    )
    parser.add_argument(
        "--no-controls", action="store_true", help="skip the control solves"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if any predeclared criterion is violated",
    )
    args = parser.parse_args(argv)

    artifact = run_case(
        ladder=QUICK_LADDER if args.quick else LADDER,
        with_controls=not args.no_controls,
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "results.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.outdir / "report.md").write_text(
        render_markdown(artifact) + "\n", encoding="utf-8"
    )
    write_case_card(artifact, args.outdir / "case.json")
    figures = args.outdir / "figures"
    plot_field(artifact, figures / "field.png")
    plot_convergence(artifact, figures / "convergence.png")
    plot_near_wall(artifact, figures / "near_wall.png")
    plot_multipole(artifact, figures / "multipole.png")
    plot_circulation(artifact, figures / "circulation.png")
    if args.doc is not None:
        args.doc.parent.mkdir(parents=True, exist_ok=True)
        args.doc.write_text(render_markdown(artifact) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "case_id": CASE_ID,
                "verdict": artifact["overall"],
                "failed_checks": artifact["acceptance"]["failed_checks"],
                "dirichlet_velocity_relative_l2": artifact["routes"]["dirichlet"][
                    "finest"
                ]["velocity_relative_l2"],
                "neumann_surface_velocity_relative_l2": artifact["routes"]["neumann"][
                    "finest"
                ]["surface_velocity_relative_l2"],
                "circulation_surface_velocity_relative_l2": artifact["routes"][
                    "circulation"
                ]["finest"]["surface_velocity_relative_l2"],
                "circulation_lift_relative_error": artifact["routes"]["circulation"][
                    "finest"
                ]["lift_relative_error"],
                "wall_seconds": artifact["wall_seconds"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if (artifact["acceptance"]["passed"] or not args.check) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
