"""Back-filled guards for the mutation-audit blind spots in ``tensorbem.analytic``.

``tensorbem.analytic`` is the oracle the rest of the suite is judged against: a
wrong factor here does not fail one benchmark, it moves *every* reference at
once.  The 2026-11-03 audit left it unaudited (its unit-test file was measured
at >100 s); the 2026-11-04 sweep re-measured it at ~30 s and seeded 24 defects.
Three of them survived the whole suite.

Each test below is written against an *independent* criterion -- a documented
physical identity, a documented input contract, or a default-argument
round-trip -- not by copying the implementation formula back into the test, so
it keeps its discriminating power if the implementation is later rewritten.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.analytic import (
    ellipsoid_added_mass,
    ellipsoid_shape_factors,
    submerged_sphere_wave_multipole,
    ursell_cylinder_added_mass,
)
from tensorbem.benchmarks.mutation_audit import _family_names, sibling_tests


def test_submerged_sphere_far_field_damping_is_consistent_with_haskind() -> None:
    """The nondimensional far-field column must equal the radiation column.

    ``submerged_sphere_wave_multipole`` returns two routes to the same damping:
    the radiation solve (``*_damping_dimless``) and the Kochin far-field integral
    of the excitation force (``*_far_field_damping_dimless``).  The Haskind
    identity equates them exactly for an exact multipole solution, and the module
    itself reports ``haskind_error_*`` on the *physical* pair only.  The
    dimensionless columns -- the ones the published tables quote -- were never
    compared, so a wrong normalisation (divide vs multiply by rho*omega*V) went
    unnoticed.  Comparing the two published columns to 1e-8 is a physical
    identity, not a copied formula.
    """
    result = submerged_sphere_wave_multipole(0.8)

    pairs = (
        (result.heave_damping_dimless, result.heave_far_field_damping_dimless, "heave"),
        (result.surge_damping_dimless, result.surge_far_field_damping_dimless, "surge"),
    )
    for radiation, far_field, name in pairs:
        assert radiation > 0.0, f"{name} damping must be positive"
        assert far_field > 0.0, f"{name} far-field damping must be positive"
        relative = abs(far_field - radiation) / abs(radiation)
        assert relative < 1.0e-8, (
            f"{name} far-field damping {far_field} disagrees with the radiation "
            f"value {radiation} by {relative:.3e}; the two columns are the same "
            "quantity by Haskind's identity"
        )


def test_submerged_sphere_physical_damping_follows_the_dimensionless_column() -> None:
    """``*_damping`` (N s/m) = rho*omega*V times ``*_damping_dimless``.

    ``displaced_volume`` and ``omega`` are reported by the result itself, so the
    normalisation can be checked without importing the constant density: the
    ratio must be the same positive number for heave and surge.  This pins the
    dimensionalisation direction (multiply, not divide) that the swap mutant in
    ``submerged_sphere_wave_multipole`` flipped.
    """
    result = submerged_sphere_wave_multipole(0.8)
    volume = result.displaced_volume
    omega = result.omega
    assert volume > 0.0 and omega > 0.0

    scale_heave = result.heave_damping / (omega * volume * result.heave_damping_dimless)
    scale_surge = result.surge_damping / (omega * volume * result.surge_damping_dimless)
    assert scale_heave > 0.0
    assert scale_heave == pytest.approx(scale_surge, rel=1.0e-12)
    # Sea-water density: 1025 kg/m^3 for the ship-hydrodynamics convention.
    assert scale_heave == pytest.approx(1025.0, rel=1.0e-12)
    assert result.heave_added_mass == pytest.approx(
        1025.0 * volume * result.heave_added_mass_dimless, rel=1.0e-12
    )


def test_ursell_cylinder_default_density_is_seawater() -> None:
    """The documented default ``density=1025.0`` must be the value actually used.

    ``ursell_cylinder_added_mass`` is called without ``density`` throughout the
    benchmark scripts, so a sign flip in the default is invisible to every test
    that passes the density explicitly.  The default result must be positive and
    equal (to machine precision) the result with ``density=1025.0`` spelled out.
    """
    default_mass, default_damping = ursell_cylinder_added_mass(1.0, 1.5, 2.0)
    explicit_mass, explicit_damping = ursell_cylinder_added_mass(
        1.0, 1.5, 2.0, density=1025.0
    )
    assert default_mass > 0.0
    assert default_damping > 0.0
    assert default_mass == pytest.approx(explicit_mass, rel=1.0e-14)
    assert default_damping == pytest.approx(explicit_damping, rel=1.0e-14)
    # The default must also be linear in the density: doubling it doubles both
    # outputs.  A negative default breaks positivity, a wrong default breaks the
    # equality above, and a non-linear normalisation breaks this ratio.
    double_mass, double_damping = ursell_cylinder_added_mass(
        1.0, 1.5, 2.0, density=2050.0
    )
    assert double_mass == pytest.approx(2.0 * default_mass, rel=1.0e-12)
    assert double_damping == pytest.approx(2.0 * default_damping, rel=1.0e-12)


def test_ellipsoid_shape_factors_accept_the_documented_minimum_node_count() -> None:
    """``nodes=16`` is the documented lower limit, so it must not be rejected as input.

    The docstring and the error message both say "nodes must be an integer >=
    16", and the source multiplies the node count by a safety factor to raise the
    truncation.  Incrementing the threshold (16 -> 17) makes the documented
    minimum fail *input validation* instead of running, which the suite only
    noticed by timing out.  The criterion below is the contract, not the value:
    ``nodes=16`` may still hit the quadrature's own "did not converge" error for
    a hard aspect ratio, but it must never be rejected as too small, while
    ``nodes=15`` must be.
    """
    semi_axes = np.array([1.5, 1.0, 0.75])

    with pytest.raises(ValueError) as too_small:
        ellipsoid_shape_factors(semi_axes, nodes=15)
    assert "nodes must be an integer >= 16" in str(too_small.value)

    try:
        ellipsoid_shape_factors(semi_axes, nodes=16)
    except ValueError as error:
        # A convergence complaint is legitimate at the minimum truncation; an
        # input-validation complaint is not.
        assert "nodes must be an integer >= 16" not in str(error)

    # The default call must still reproduce the contraction identity.
    default_factors = ellipsoid_shape_factors(semi_axes)
    assert len(default_factors) == 3
    assert all(0.0 < value < 2.0 for value in default_factors)
    assert math.fsum(default_factors) == pytest.approx(2.0, abs=1.0e-9)

    added = ellipsoid_added_mass(semi_axes)
    assert len(added) == 3
    assert all(value > 0.0 for value in added)


def test_private_helper_family_includes_the_public_parent() -> None:
    """A defect seeded in a private helper must be re-checked against its caller's tests.

    Tests name public entry points, never ``_helpers``.  The name resolution is
    asserted directly (rather than through ``sibling_tests``, which globs this
    worktree's ``tests/`` directory) so the guard also holds inside the audit's
    isolated shadow tree.
    """
    assert _family_names("submerged_sphere_wave_multipole._solve_azimuthal") == (
        "submerged_sphere_wave_multipole",
    )
    assert _family_names("ellipsoid_shape_factors") == ("ellipsoid_shape_factors",)
    assert _family_names("SomeClass._helper") == ("SomeClass",)
    # A module-level private helper with no public parent resolves to nothing,
    # which makes stage 2 a no-op; registered as a known limitation.
    assert _family_names("_ursell_multipole_system") == ()
    assert _family_names("<module>") == ()
    # The directory walk still finds the public entry point's tests.
    assert "tests/test_published_reference_columns.py" in sibling_tests(
        "ellipsoid_shape_factors"
    )
