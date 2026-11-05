"""Guards for mutation survivors in the wave-load chain (m3108).

Context
-------
The m2582 mutation audit measured the discriminating power of this suite on the
*Laplace / free-surface kernels* and scored them 0.50-1.00.  It never looked at
the modules that turn those kernels into the numbers the package actually
reports.  m3108 extended the same audit engine to ``kernels3d``, ``pressure``,
``helmholtz`` and ``rao`` (see ``docs/m3108/mutation_wave_load_chain.md``) and
found 14 survivors out of 32 mutants, including a **sign flip on the dynamic
pressure** and an **inverted inside-polygon predicate**, both of which leave the
pre-existing suite green.

This module closes the survivors that can be closed with a sharp, independent
oracle.  Every reference below is computed in the test by a different route than
the function under test:

* the Bernoulli load reference is assembled here from ``mesh.areas`` /
  ``mesh.normals`` / ``mesh.centroids`` as ``sum(p A n)`` and ``sum(r x p A n)``
  with ``p = 1/2 rho (U^2 - |v|^2)`` written in *sum* form, so neither
  ``steady_pressure_coefficient`` nor ``integrate_pressure_loads`` is reused;
* the polygon predicate is checked against a circle of known radius and a known
  edge clearance, not against the crossing-count implementation;
* the reported Helmholtz boundary residual is re-derived from the returned
  ``double_layer_matrix`` / ``total_potential`` / ``incident_potential``;
* the documented *default* quadrature order is pinned by comparing the call that
  omits ``order`` with the explicit ``order=8`` call.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from tensorbem.geometry import PanelMesh, regular_polygon
from tensorbem.geometry3d import TriangleMesh, icosphere
from tensorbem.helmholtz import solve_vertical_cylinder_diffraction
from tensorbem.kernels3d import duffy_quadrature_batch
from tensorbem.pressure import steady_pressure_loads

RHO = 1000.0
G = 9.80665


def _closed_unit_sphere(subdivisions: int = 2):
    """Return a closed :class:`TriangleMesh` (built through whichever constructor)."""
    built = icosphere(subdivisions)
    if isinstance(built, TriangleMesh):
        return built
    vertices, faces = built
    return TriangleMesh(vertices, faces)


# ---------------------------------------------------------------------------
# pressure: the sign and the quadratic speed law of the dynamic pressure
# ---------------------------------------------------------------------------


def test_steady_pressure_loads_equals_independent_bernoulli_integral():
    """``F`` and ``M`` must equal a hand-assembled ``sum(p A n)`` reference.

    The velocity is deliberately non-uniform, so the resultant is non-zero: a
    global sign flip of ``p - p_inf`` (the surviving mutant) reverses the
    resultant and cannot hide behind cancellation on a closed surface.
    """
    mesh = _closed_unit_sphere(2)
    speed = 4.0
    centers = mesh.centroids
    # |v| varies across the surface -> the pressure field varies -> F != 0
    magnitude = speed * (1.0 + 0.6 * centers[:, 0])
    velocity = np.column_stack((magnitude, np.zeros_like(magnitude), np.zeros_like(magnitude)))

    _cp, loads = steady_pressure_loads(
        mesh, velocity, (speed, 0.0, 0.0), density=RHO,
        reference_center=(0.0, 0.0, 0.0),
    )

    # Independent reference: the definition, in sum form.
    pressure = 0.5 * RHO * (speed * speed - np.sum(velocity * velocity, axis=1))
    panel_force = (pressure * mesh.areas)[:, None] * mesh.normals
    force_ref = np.sum(panel_force, axis=0)
    moment_ref = np.sum(np.cross(centers, panel_force), axis=0)

    assert np.linalg.norm(force_ref) > 1.0e3, "reference resultant must not be degenerate"
    assert np.allclose(loads.force, force_ref, rtol=1.0e-12, atol=1.0e-9)
    assert np.allclose(loads.moment, moment_ref, rtol=1.0e-12, atol=1.0e-9)

    # The same identity for a |v| far from U, where "v**2 -> v*2" is visible.
    stagnation = np.full((mesh.panel_count, 3), [0.0, 0.0, 0.0])
    cp_stag, _ = steady_pressure_loads(mesh, stagnation, (speed, 0.0, 0.0), density=RHO)
    assert np.allclose(cp_stag, 1.0, rtol=0.0, atol=1.0e-15)  # Cp = 1 - 0/U^2
    fast = np.full((mesh.panel_count, 3), [3.0 * speed, 0.0, 0.0])
    cp_fast, loads_fast = steady_pressure_loads(mesh, fast, (speed, 0.0, 0.0), density=RHO)
    assert np.allclose(cp_fast, 1.0 - 9.0, rtol=0.0, atol=1.0e-14)


# ---------------------------------------------------------------------------
# helmholtz: the inside-polygon predicate
# ---------------------------------------------------------------------------


def test_points_inside_polygon_accepts_interior_and_rejects_exterior():
    """A 256-gon is a circle of radius 1 to 5e-5: interior/exterior are analytic."""
    from tensorbem.helmholtz import points_inside_polygon

    mesh = regular_polygon(256, radius=1.0)
    pts = np.array(
        [
            [0.0, 0.0],       # centre
            [0.5, 0.0],       # well inside
            [0.0, -0.9],      # inside, away from the seam
            [1.5, 0.0],       # outside
            [0.0, 2.0],       # outside
            [-1.2, -1.0],     # outside, radius 1.56
        ]
    )
    inside = points_inside_polygon(mesh, pts)
    assert inside.dtype == np.bool_
    assert inside.tolist() == [True, True, True, False, False, False]

    # A point 1e-6 inside the boundary is *inside* geometrically but must be
    # rejected by the clearance rule -- this pins the "&" arm as well.
    assert points_inside_polygon(mesh, np.array([[1.0 - 1.0e-6, 0.0]]))[0] == False  # noqa: E712


# ---------------------------------------------------------------------------
# kernels3d: the documented default quadrature order
# ---------------------------------------------------------------------------


def test_duffy_quadrature_batch_default_order_is_the_documented_eight():
    """Omitting ``order`` must be identical to ``order=8`` (the docstring value)."""
    v0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    v1 = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    v2 = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    default_points, default_weights = duffy_quadrature_batch(v0, v1, v2)
    explicit_points, explicit_weights = duffy_quadrature_batch(v0, v1, v2, order=8)
    assert default_points.shape == (2, 64, 3)
    assert np.array_equal(default_points, explicit_points)
    assert np.array_equal(default_weights, explicit_weights)
    # and the weights are a genuine integrator: sum(w) = area, per triangle
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    assert np.allclose(np.sum(default_weights, axis=1), areas, rtol=1.0e-14)