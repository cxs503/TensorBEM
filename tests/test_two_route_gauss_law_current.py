"""Regression tests for the two-route (surface vs far-field Gauss law) current
cross-validation, XVAL-BEM-TWO-ROUTE-GAUSS-LAW-CURRENT.

What is pinned here:

1. the closed-form spheroid spreading current used as the *independent* anchor
   (sphere limit, thin-rod limit, thin-disc limit ``8 b``, axis-permutation
   invariance, guard against non-spheroid inputs).  The disc limit is the
   discriminating one: with the arctan argument written as ``e`` instead of
   ``e*b/a`` the formula is still exact for a sphere but returns ``16 b`` for a
   disc, i.e. a factor 2 -- exactly the defect this study found and fixed;
2. the two routes agree on a sphere and on a spheroid at a coarse panel level
   (the ledger acceptance is 15 %, the internal budget here is tighter);
3. the far-field route is genuinely a Gauss-law/monopole extraction: the *raw*
   per-radius currents are radius independent, and an off-centre sampling
   sphere is detected (mutant control, so the comparison is not vacuous);
4. disconnected conductors: merging two spheres keeps the mesh valid and the
   two-route identity holds for the pair, with the pair current below the
   isolated-double bound.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.analytic import (
    spheroid_geometric_capacitance,
    spheroid_spreading_current,
)
from tensorbem.benchmarks.two_route_gauss_law_current import (
    CRITERIA,
    case_current,
    merge_meshes,
    mixed_bc_spheroid,
    offcentre_mutant,
    solve_uniform_potential,
)
from tensorbem.cathodic_protection import (
    SEAWATER_CONDUCTIVITY,
    far_field_current,
)
from tensorbem.geometry3d import ellipsoid_mesh, icosphere


# ---------------------------------------------------------------------------
# 1. closed-form anchor
# ---------------------------------------------------------------------------
def test_sphere_limit_of_both_branches():
    radius = 2.0
    exact = 4.0 * math.pi * radius
    assert spheroid_geometric_capacitance(np.array([radius, radius, radius])) == pytest.approx(exact, rel=1e-15)
    # near-sphere prolate and oblate approach the same value from opposite sides
    prolate = spheroid_geometric_capacitance(np.array([radius * 1.001, radius, radius]))
    oblate = spheroid_geometric_capacitance(np.array([radius, radius * 1.001, radius * 1.001]))
    assert prolate == pytest.approx(exact, rel=1e-2)
    assert oblate == pytest.approx(exact, rel=1e-2)


def test_thin_disc_limit_is_eight_radius():
    """Oblate spheroid, polar axis -> 0: kappa -> 8 b (disc capacitance 8 eps0 b)."""
    for equatorial in (1.0, 2.5):
        kappa = spheroid_geometric_capacitance(np.array([1e-9, equatorial, equatorial]))
        assert kappa == pytest.approx(8.0 * equatorial, rel=1e-6)


def test_thin_rod_limit_vanishes_and_is_monotone():
    values = [
        spheroid_geometric_capacitance(np.array([1.0, minor, minor]))
        for minor in (1.0, 0.5, 0.1, 0.01, 1e-4)
    ]
    assert values[0] == pytest.approx(4.0 * math.pi, rel=1e-15)
    assert all(later < earlier for earlier, later in zip(values, values[1:]))
    # Logarithmic (not algebraic) decay: kappa -> 8 pi a / ln(4 a^2 / b^2)
    for minor, kappa in zip((0.01, 1e-4), values[-2:]):
        asymptotic = 8.0 * math.pi / math.log(4.0 / minor**2)
        assert kappa == pytest.approx(asymptotic, rel=0.05)


def test_oblate_branch_sits_between_disc_and_enclosing_sphere():
    """8 b (thin disc) < kappa < 4 pi b (sphere through the equator).

    The spheroid contains the equatorial disc as the polar axis shrinks and is
    contained in the equatorial sphere, so its capacitance is bracketed; both
    bounds are exact and independent of the formula under test.
    """
    polar, equatorial = 0.5, 1.0
    kappa = spheroid_geometric_capacitance(np.array([polar, equatorial, equatorial]))
    assert 8.0 * equatorial < kappa < 4.0 * math.pi * equatorial
    # and it must also exceed the insphere value 4 pi * polar
    assert kappa > 4.0 * math.pi * polar


def test_axis_permutation_invariance_and_guards():
    reference = spheroid_geometric_capacitance(np.array([1.0, 2.0, 2.0]))
    for axes in ((2.0, 1.0, 2.0), (2.0, 2.0, 1.0)):
        assert spheroid_geometric_capacitance(np.array(axes)) == pytest.approx(reference, rel=1e-15)
    with pytest.raises(ValueError):
        spheroid_geometric_capacitance(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        spheroid_geometric_capacitance(np.array([1.0, -2.0, -2.0]))
    with pytest.raises(ValueError):
        spheroid_spreading_current(np.array([1.0, 1.0, 1.0]), potential=float("nan"))


def test_spreading_current_scales_linearly_with_conductivity_and_potential():
    axes = np.array([2.0, 1.0, 1.0])
    base = spheroid_spreading_current(axes, potential=-1.0, conductivity=SEAWATER_CONDUCTIVITY)
    assert base < 0.0
    doubled = spheroid_spreading_current(axes, potential=-2.0, conductivity=SEAWATER_CONDUCTIVITY)
    assert doubled == pytest.approx(2.0 * base, rel=1e-15)
    halved_sigma = spheroid_spreading_current(axes, potential=-1.0, conductivity=0.5 * SEAWATER_CONDUCTIVITY)
    assert halved_sigma == pytest.approx(0.5 * base, rel=1e-15)


# ---------------------------------------------------------------------------
# 2. two-route agreement on coarse meshes
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def sphere_case():
    return case_current(
        "sphere_r1_coarse",
        icosphere(1, radius=1.0),
        semi_axes=(1.0, 1.0, 1.0),
        radii=(3.0, 5.0, 8.0),
        n_directions=24,
    )


@pytest.fixture(scope="module")
def spheroid_case():
    return case_current(
        "prolate_2to1_coarse",
        ellipsoid_mesh(1, np.array([2.0, 1.0, 1.0])),
        semi_axes=(2.0, 1.0, 1.0),
        radii=(6.0, 10.0, 16.0),
        n_directions=24,
    )


def test_sphere_two_routes_agree_and_match_analytic(sphere_case):
    assert sphere_case["surface_rel_error"] < 0.10          # 80 panels
    assert sphere_case["two_route_rel"] < CRITERIA["two_route_rel"] * 5.0
    assert sphere_case["far_field_rel_error"] < 0.10


def test_spheroid_two_routes_agree_and_match_analytic(spheroid_case):
    """A non-spherical body: the far-field route must still isolate the monopole."""
    assert spheroid_case["surface_rel_error"] < 0.15        # ledger acceptance
    assert spheroid_case["two_route_rel"] < CRITERIA["two_route_rel"] * 5.0
    assert spheroid_case["far_field_rel_error"] < 0.15


def test_raw_gauss_law_current_is_radius_independent(spheroid_case):
    """The angular mean is the monopole: I_R must not depend on R."""
    spread = spheroid_case["radius_independence"]["spread_rel"]
    assert spread < 0.05, spread
    # ... and every raw radius current matches the surface route on its own
    for rel in spheroid_case["radius_independence"]["rel_by_radius"]:
        assert abs(rel) < 0.05


# ---------------------------------------------------------------------------
# 3. mutant control: the comparison has teeth
# ---------------------------------------------------------------------------
def test_offcentre_sampling_sphere_is_detected(spheroid_case):
    mesh = ellipsoid_mesh(1, np.array([2.0, 1.0, 1.0]))
    result = solve_uniform_potential(mesh)
    surface = float(result.anode_current_total)
    extent = float(np.max(np.linalg.norm(mesh.vertices - mesh.vertices.mean(0), axis=1)))
    radii = tuple(scale * extent for scale in (1.5, 2.0, 3.0, 4.0, 6.0))
    base = far_field_current(
        mesh, result, radii=radii, n_directions=24
    )["far_field_vs_surface_rel_error"]
    mutant = offcentre_mutant(
        mesh, result, radii=radii, surface_current=surface, n_directions=24
    )
    assert mutant["offset_fraction"] == 1.0
    assert mutant["intercept_rel_error"] > 1.0e-2, mutant
    assert mutant["intercept_rel_error"] > 100.0 * base


# ---------------------------------------------------------------------------
# 4. mixed BC and disconnected conductors
# ---------------------------------------------------------------------------
def test_mixed_bc_two_routes_agree():
    record = mixed_bc_spheroid(
        (2.0, 1.0, 1.0),
        subdivisions=1,
        radii=(6.0, 10.0, 16.0),
        n_directions=24,
    )
    assert 0 < record["n_anode_panels"] < record["n_panels"]
    assert record["two_route_rel"] < 0.05, record["two_route_rel"]
    assert record["surface_current_A"] < 0.0


def test_merged_mesh_is_valid_and_pair_current_is_below_isolated_double():
    radius = 1.0
    single = icosphere(1, radius=radius)
    isolated = abs(float(solve_uniform_potential(single).anode_current_total))
    separation = 6.0
    merged = merge_meshes(
        (icosphere(1, radius=radius), icosphere(1, radius=radius)),
        ((-0.5 * separation, 0.0, 0.0), (0.5 * separation, 0.0, 0.0)),
    )
    assert merged.panel_count == 2 * single.panel_count
    result = solve_uniform_potential(merged)
    pair = abs(float(result.anode_current_total))
    ratio = pair / (2.0 * isolated)
    assert math.log(2.0) < ratio < 1.0, ratio
    far = far_field_current(merged, result, radii=(6.0, 9.0, 14.0), n_directions=24)
    assert far["far_field_vs_surface_rel_error"] < 0.05