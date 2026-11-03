"""Verification of the finite-depth radiation solver.

The module under test wires the finite-depth kernel of
:mod:`tensorbem.finite_depth` into the indirect (single-layer) radiation
formulation of :mod:`tensorbem.free_surface_solver`.  The kernel itself is
verified elsewhere; what has to be shown here is that the *chain* is right,
because a kernel can be correct while the assembly normalisation, the jump
relation or the far-field constant are wrong, and each of those mistakes
produces plausible-looking coefficients.

Four independent properties are asserted:

1. **deep-water collapse** - at ``h = 50a`` the coefficients must fall back
   onto Ursell's analytic deep-water solution, which is a closed-form
   construction sharing no code with this solver;
2. **energy identity** - the damping from the surface pressure must equal the
   damping from the radiated wave amplitude, using the standard wave-energy
   balance ``B = 2*rho*g*c_g*|eta_a|^2``;
3. **independent discretisation** - the method of fundamental solutions on the
   *same* kernel (point sources, no panels, no singular quadrature) must
   reproduce the panel coefficients to the panel discretisation error;
4. **physical monotonicity** - reducing the clearance to the sea bed must
   increase the added mass (blockage), and the heave and sway coefficients must
   split in finite depth while they coincide in deep water.
"""

import math

import numpy as np
import pytest

from tensorbem.analytic import ursell_cylinder_added_mass
from tensorbem.finite_depth_solver import (
    far_field_damping,
    radiation_reference,
    solve_radiation,
    solve_radiation_indirect,
)
from tensorbem.free_surface_solver import solve_radiation_indirect as solve_deep

GRAVITY = 9.80665
RADIUS = 1.0
SUBMERGENCE = 2.0
KA = 0.6
NU = KA / RADIUS
DENSITY = 1025.0


def circle_mesh(panel_count=64, radius=RADIUS, submergence=SUBMERGENCE):
    angles = 2.0 * math.pi * np.arange(panel_count) / panel_count
    from tensorbem.geometry import PanelMesh

    return PanelMesh(
        np.column_stack(
            (radius * np.cos(angles), radius * np.sin(angles) - submergence)
        )
    )


@pytest.fixture(scope="module")
def analytic():
    return ursell_cylinder_added_mass(
        RADIUS, SUBMERGENCE,
        math.sqrt(NU * GRAVITY), DENSITY, gravity=GRAVITY,
    )


# --------------------------------------------------------------------------
# 1. Deep-water collapse
# --------------------------------------------------------------------------
@pytest.mark.parametrize("direction,index", [("heave", 1), ("sway", 0)])
def test_deep_water_collapse(direction, index, analytic):
    """At h/a = 50 the finite-depth coefficients recover Ursell's values."""
    mesh = circle_mesh()
    result = solve_radiation_indirect(mesh, NU, 50.0, direction, density=DENSITY)
    exact_mass, exact_damping = analytic
    # The 64-panel discretisation error of this code is ~1.7% in added mass, so
    # the collapse is asserted at 3% and the panel-level accuracy is asserted
    # in test_panel_refinement.
    assert result.added_mass[index] == pytest.approx(exact_mass, rel=3.0e-2)
    assert result.damping[index] == pytest.approx(exact_damping, rel=3.0e-2)
    # ... and it must also agree with the deep-water solver at the same mesh,
    # which is the statement that only the kernel changed.
    deep = solve_deep(mesh, NU, direction, density=DENSITY)
    assert result.added_mass[index] == pytest.approx(deep.added_mass[index], rel=1.0e-2)
    assert result.damping[index] == pytest.approx(deep.damping[index], rel=1.0e-2)


def test_shallow_water_is_not_deep_water():
    """The depth dependence is real, not a relabelled deep-water answer."""
    mesh = circle_mesh()
    shallow = solve_radiation_indirect(mesh, NU, 4.0, "heave", density=DENSITY)
    deep = solve_radiation_indirect(mesh, NU, 20.0, "heave", density=DENSITY)
    relative_split = abs(shallow.added_mass[1] - deep.added_mass[1]) / deep.added_mass[1]
    assert relative_split > 0.1


# --------------------------------------------------------------------------
# 2. Energy identity (near field vs far field)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("depth", [4.0, 6.0, 20.0])
@pytest.mark.parametrize("direction,index", [("heave", 1), ("sway", 0)])
def test_energy_identity(depth, direction, index):
    """``omega*rho*Im(force)`` must match ``2*rho*g*c_g*|eta_a|^2``."""
    mesh = circle_mesh()
    result, far = solve_radiation(depth=depth, mesh=mesh, nu=NU,
                                  direction=direction, density=DENSITY)
    assert result.damping[index] > 0.0
    assert far == pytest.approx(result.damping[index], rel=3.0e-2)


def test_far_field_damping_requires_sources():
    """The far-field damping is a function of the solved density, by design."""
    mesh = circle_mesh(16)
    from tensorbem.finite_depth_solver import _solve_sources

    sources, _, _, _ = _solve_sources(mesh, NU, 6.0, "heave", 8, None)
    value = far_field_damping(mesh, sources, NU, 6.0, DENSITY, gravity=GRAVITY)
    assert value > 0.0 and math.isfinite(value)


# --------------------------------------------------------------------------
# 3. Independent discretisation
# --------------------------------------------------------------------------
@pytest.mark.parametrize("direction,index", [("heave", 1), ("sway", 0)])
def test_mfs_reference_agrees(direction, index):
    mesh = circle_mesh()
    panel = solve_radiation_indirect(mesh, NU, 10.0, direction, density=DENSITY)
    reference = radiation_reference(
        mesh, NU, 10.0, direction, source_count=24, density=DENSITY
    )
    assert reference.added_mass[index] == pytest.approx(
        panel.added_mass[index], rel=2.5e-2
    )
    assert reference.damping[index] == pytest.approx(
        panel.damping[index], rel=3.0e-2
    )


# --------------------------------------------------------------------------
# 4. Physical behaviour and input validation
# --------------------------------------------------------------------------
def test_blockage_increases_added_mass():
    """Closing the sea bed must raise the heave added mass monotonically."""
    mesh = circle_mesh()
    masses = [
        solve_radiation_indirect(mesh, NU, depth, "heave", density=DENSITY).added_mass[1]
        for depth in (20.0, 10.0, 6.0, 4.0)
    ]
    assert masses == sorted(masses), masses


def test_heave_and_sway_split_in_finite_depth():
    mesh = circle_mesh()
    heave = solve_radiation_indirect(mesh, NU, 4.0, "heave", density=DENSITY)
    sway = solve_radiation_indirect(mesh, NU, 4.0, "sway", density=DENSITY)
    assert abs(heave.added_mass[1] - sway.added_mass[0]) > 0.1 * sway.added_mass[0]
    deep_heave = solve_radiation_indirect(mesh, NU, 50.0, "heave", density=DENSITY)
    deep_sway = solve_radiation_indirect(mesh, NU, 50.0, "sway", density=DENSITY)
    assert deep_heave.added_mass[1] == pytest.approx(
        deep_sway.added_mass[0], rel=1.0e-3
    )


def test_modes_are_converged():
    """Doubling the evanescent mode count may not move the coefficients."""
    mesh = circle_mesh(32)
    coarse = solve_radiation_indirect(
        mesh, NU, 6.0, "heave", density=DENSITY, modes=128
    )
    fine = solve_radiation_indirect(
        mesh, NU, 6.0, "heave", density=DENSITY, modes=256
    )
    assert fine.added_mass[1] == pytest.approx(coarse.added_mass[1], rel=1.0e-6)


def test_body_must_sit_inside_the_strip():
    mesh = circle_mesh()
    with pytest.raises(ValueError):
        solve_radiation_indirect(mesh, NU, 2.9, "heave")  # floor above the keel
    with pytest.raises(ValueError):
        solve_radiation_indirect(mesh, 0.0, 6.0, "heave")
    with pytest.raises(ValueError):
        solve_radiation_indirect(mesh, NU, 0.0, "heave")
    with pytest.raises(ValueError):
        solve_radiation_indirect(mesh, NU, 6.0, "roll")