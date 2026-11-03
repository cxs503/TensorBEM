"""End-to-end validation of the finite-depth radiation path.

The tests here are the acceptance criteria for driving the boundary-element
solver with :mod:`tensorbem.finite_depth` instead of the deep-water kernel, and
for the finite-depth far-field energy identity. They are written so that each
one fails for a different reason:

* the far-field normalisation is checked with no BEM solve in it at all, so a
  wrong ``c_g`` or mode norm cannot hide behind discretisation error;
* the ``depth -> inf`` limit is checked against Ursell's closed-form deep-water
  values through the method of fundamental solutions, which shares no panels
  with the panel method;
* the near-field/far-field identity is checked *relative to the deep-water
  residual on the same mesh*, because that residual is a property of the
  piecewise-constant representation, not of the depth;
* the panel solve itself is checked against that same independent MFS
  reference, and against its own refinement.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem import finite_depth
from tensorbem import free_surface_solver as solver
from tensorbem.benchmarks.finite_depth_radiation import (
    NU,
    REFERENCE_ADDED_MASS,
    REFERENCE_DAMPING,
)
from tensorbem.benchmarks.ursell_cylinder import submerged_circle_mesh

FINITE_DEPTH = 16.0


@pytest.fixture(scope="module")
def mesh():
    return submerged_circle_mesh(64)


@pytest.fixture(scope="module")
def deep_solution(mesh):
    return solver.solve_radiation_indirect(mesh, NU, "heave")


@pytest.fixture(scope="module")
def finite_solution(mesh):
    return solver.solve_radiation_indirect(mesh, NU, "heave", depth=FINITE_DEPTH)


def test_kernel_rankine_split_is_exact():
    """``include_rankine=False`` must differ from the full kernel by log(r)/2pi."""
    points = np.array([0.3, 0.9, 2.0])
    sources = np.array([0.35, 0.8, 2.6])
    full = finite_depth.green_function(points, -1.0, sources, -2.0, 0.6, 5.0)
    smooth = finite_depth.green_function(
        points, -1.0, sources, -2.0, 0.6, 5.0, include_rankine=False
    )
    radius = np.hypot(points - sources, -1.0 + 2.0)
    expected = np.log(radius) / (2.0 * math.pi)
    assert np.allclose(full - smooth, expected, rtol=0.0, atol=1e-14)


def test_amplitude_coefficient_is_overflow_free():
    """The deep-water limit of the mode shape is exp(nu*z), without overflow."""
    z = np.array([-0.5, -2.0, -10.0])
    shape = finite_depth.wave_amplitude_coefficient(z, NU, 800.0)
    assert np.all(np.isfinite(shape))
    assert np.allclose(shape, np.exp(NU * z), rtol=1e-11)
    slope = finite_depth.wave_amplitude_slope(z, NU, 800.0)
    assert np.allclose(slope, NU * np.exp(NU * z), rtol=1e-11)


def test_far_field_constant_is_depth_independent(mesh, deep_solution):
    """Same field, four depths: the far-field damping must not move.

    This pins the ``c_g``/mode-normalisation constant of
    :func:`tensorbem.free_surface_solver.far_field_damping` on its own: the
    potential, the normal velocity and the mesh are identical in every term,
    so the deep-water value must be reproduced as ``h -> inf``.
    """
    deep_value = solver.far_field_damping(
        mesh, deep_solution.potential, deep_solution.normal_velocity, NU
    )
    for depth in (50.0, 200.0, 800.0):
        finite_value = solver.far_field_damping(
            mesh, deep_solution.potential, deep_solution.normal_velocity, NU,
            depth=depth,
        )
        assert abs(finite_value / deep_value - 1.0) < 1e-12


def test_deep_water_limit_recovers_ursell_reference(mesh):
    """The MFS coefficients converge onto Ursell's deep-water values in ``h``.

    At ``nu*h = 9.6`` the water is still shallow enough to matter (the residual
    is a physical finite-depth effect, not an error), so the bound there is the
    loose one; at ``nu*h = 19.2`` the coefficients must be within 0.5 % of
    Ursell's closed-form values, which is what makes the ``h -> inf`` limit a
    statement about the solver rather than about the sea bed.
    """
    errors = {}
    for depth in (16.0, 32.0):
        reference = solver.radiation_reference(mesh, NU, "heave", depth=depth)
        errors[depth] = (
            abs(reference.added_mass[1] / REFERENCE_ADDED_MASS - 1.0),
            abs(reference.damping[1] / REFERENCE_DAMPING - 1.0),
        )
    assert errors[16.0][0] < 2e-2 and errors[16.0][1] < 2e-2
    assert errors[32.0][0] < 5e-3 and errors[32.0][1] < 5e-3
    assert errors[32.0][0] < errors[16.0][0]
    assert errors[32.0][1] < errors[16.0][1]


def test_deep_water_limit_is_approached_monotonically(mesh, deep_solution):
    """Deeper water must approach the deep-water panel solution."""
    deep_added_mass = float(deep_solution.added_mass[1])
    errors = []
    for depth in (16.0, 32.0):
        solution = solver.solve_radiation_indirect(
            mesh, NU, "heave", depth=depth
        )
        errors.append(abs(float(solution.added_mass[1]) / deep_added_mass - 1.0))
    assert errors[1] < errors[0]
    assert errors[1] < 5e-3


def test_energy_identity_is_depth_independent(mesh, deep_solution, finite_solution):
    """Near-field and far-field damping agree as well in depth as in deep water."""
    deep_far = solver.far_field_damping(
        mesh, deep_solution.potential, deep_solution.normal_velocity, NU
    )
    deep_residual = abs(deep_far / float(deep_solution.damping[1]) - 1.0)

    finite_far = solver.far_field_damping(
        mesh, finite_solution.potential, finite_solution.normal_velocity, NU,
        depth=FINITE_DEPTH,
    )
    finite_residual = abs(finite_far / float(finite_solution.damping[1]) - 1.0)

    assert deep_residual < 2e-2
    assert finite_residual < 2e-2
    # The residual is the representation error of piecewise constant sources,
    # so it may not grow measurably just because the kernel has a sea bed.
    assert finite_residual < 1.15 * deep_residual + 1e-4


def test_mfs_satisfies_the_energy_identity_tighter_than_panels(
    mesh, finite_solution
):
    """The identity is exact for a smooth field: the MFS residual is smaller."""
    reference = solver.radiation_reference(mesh, NU, "heave", depth=FINITE_DEPTH)
    reference_far = solver.far_field_damping(
        mesh, reference.potential, reference.normal_velocity, NU, depth=FINITE_DEPTH
    )
    reference_residual = abs(reference_far / float(reference.damping[1]) - 1.0)
    panel_far = solver.far_field_damping(
        mesh, finite_solution.potential, finite_solution.normal_velocity, NU,
        depth=FINITE_DEPTH,
    )
    panel_residual = abs(panel_far / float(finite_solution.damping[1]) - 1.0)
    assert reference_residual < panel_residual
    assert reference_residual < 2e-3


def test_panel_solution_matches_independent_mfs(mesh, finite_solution):
    """The panel solve agrees with the panel-free representation in depth."""
    reference = solver.radiation_reference(mesh, NU, "heave", depth=FINITE_DEPTH)
    assert abs(
        float(finite_solution.added_mass[1]) / float(reference.added_mass[1]) - 1.0
    ) < 4e-2
    assert abs(
        float(finite_solution.damping[1]) / float(reference.damping[1]) - 1.0
    ) < 4e-2
    assert finite_solution.relative_residual < 1e-10


def test_body_must_clear_the_sea_bed(mesh):
    """A body below the sea bed is a user error, not a silent wrong answer."""
    with pytest.raises(ValueError, match="above the sea bed"):
        solver.solve_radiation_indirect(mesh, NU, "heave", depth=2.5)
    with pytest.raises(ValueError, match="above the sea bed"):
        solver.assemble_indirect_matrix(mesh, NU, depth=2.5)


@pytest.mark.parametrize("depth", [0.0, -3.0, float("inf"), float("nan")])
def test_invalid_depth_is_rejected(mesh, depth):
    with pytest.raises(ValueError):
        solver.solve_radiation_indirect(mesh, NU, "heave", depth=depth)


def test_finite_depth_conditioning_is_bounded(finite_solution):
    """The depth-averaged kernel must not wreck the assembled system."""
    assert np.isfinite(finite_solution.condition_number)
    assert finite_solution.condition_number < 50.0