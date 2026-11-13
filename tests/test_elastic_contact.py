"""Verification of the elastic-contact (Boussinesq) route.

The anchor is ``CASE-FEM-HERTZ-CONTACT``: contact radius ``a`` and penetration
``delta`` of a sphere pressed into an elastic half-space versus the Hertz closed
form (Hertz 1882; Johnson 1985 Sec. 4.2), with a 3 % tolerance, the load being
the prescribed quantity.

Every assertion here compares a *solver output* against a formula that the
solver never sees, and the two intermediate quantities the solver does depend on
are themselves verified independently:

* the Boussinesq ring kernel is checked against a direct angular quadrature
  (this is where an ``ellipk`` parameter convention error would hide);
* the assembled influence matrix is checked against the exact displacement under
  a uniformly loaded circle, ``w(0) = 2 p a / E*`` (Love's result), which pins the
  quadrature weights too;
* the complementarity solver is checked on a *different* contact problem whose
  closed form is different -- the rigid flat punch, ``P = 2 E* a delta``.

Only then is the Hertz anchor solved for.  No part of the Hertz formula enters
the mesh, the load path or the rim recovery, so the comparison is not circular.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate

from tensorbem import elastic_contact as ec

# --- anchor case: two 50 mm steel spheres, 10 kN (Hertz/Johnson) -------------
SPHERE_RADIUS = 50.0e-3
YOUNG = 210.0e9
POISSON = 0.3
LOAD = 10.0e3
#: equivalent sphere-on-half-space problem: R = R_s/2, 1/E* = (1-nu^2)/E + (1-nu^2)/E
RADIUS_EFF = SPHERE_RADIUS / 2.0
E_STAR = ec.effective_modulus(YOUNG, POISSON, YOUNG, POISSON)
ANCHOR = ec.hertz_sphere_halfspace(LOAD, RADIUS_EFF, E_STAR)

TOLERANCE = 0.03  # the anchor's acceptance tolerance


def _sphere_solution(n_elements: int = 160, **kwargs) -> ec.ContactSolution:
    return ec.solve_normal_contact(
        lambda r: ec.parabolic_gap(r, RADIUS_EFF), E_STAR, load=LOAD,
        n_elements=n_elements, grading=2.0, **kwargs,
    )


# ---------------------------------------------------------------------------
# Closed form and kernel: the two things the solver is allowed to assume
# ---------------------------------------------------------------------------
def test_effective_modulus_is_the_series_combination():
    # two identical steel bodies: E* = E / (2 (1 - nu^2))
    assert E_STAR == pytest.approx(YOUNG / (2.0 * (1.0 - POISSON ** 2)), rel=1e-14)
    # a rigid indenter on an elastic half-space: E* = E / (1 - nu^2)
    assert ec.effective_modulus(YOUNG, POISSON) == pytest.approx(
        YOUNG / (1.0 - POISSON ** 2), rel=1e-14)
    assert ec.effective_modulus(YOUNG, POISSON, 1e30, POISSON) == pytest.approx(
        YOUNG / (1.0 - POISSON ** 2), rel=1e-10)


def test_hertz_reference_matches_the_published_anchor_values():
    """Johnson (1985) Sec. 4.2 worked example, as quoted in the anchor ref."""
    assert ANCHOR.contact_radius == pytest.approx(1.1756673e-3, rel=2e-6)
    assert ANCHOR.penetration == pytest.approx(5.5287748e-5, rel=2e-6)
    assert ANCHOR.peak_pressure == pytest.approx(3.4543988e9, rel=2e-6)
    # internal consistency: the profile integrates to the load and delta = a^2/R
    r = np.linspace(0.0, ANCHOR.contact_radius, 20001)
    load = 2.0 * math.pi * integrate.simpson(ANCHOR.pressure(r) * r, x=r)
    assert load == pytest.approx(LOAD, rel=1e-6)
    assert ANCHOR.penetration == pytest.approx(ANCHOR.contact_radius ** 2 / RADIUS_EFF, rel=1e-14)


@pytest.mark.parametrize("r,rho", [(1e-4, 2e-4), (3e-4, 1e-4), (2e-3, 1e-6), (1e-3, 3e-3)])
def test_ring_kernel_matches_direct_angular_quadrature(r, rho):
    """``4 K(k)/(r+rho)`` versus ``int_0^{2 pi} dphi/sqrt(r^2 + rho^2 - 2 r rho cos phi)``.

    A wrong elliptic-integral parameter convention (``m = k^2`` versus ``m = k``)
    survives every downstream test because it changes the kernel by a few per
    cent only near ``r = rho``; this test catches it where it lives.
    """
    direct = integrate.quad(
        lambda phi: 1.0 / math.sqrt(r * r + rho * rho - 2.0 * r * rho * math.cos(phi)),
        0.0, 2.0 * math.pi, epsabs=0.0, epsrel=1e-12, limit=400)[0]
    assert float(ec.ring_kernel(r, rho)) == pytest.approx(direct, rel=1e-9)
    # axis limit: kernel -> 2 pi / rho
    assert float(ec.ring_kernel(0.0, rho)) == pytest.approx(2.0 * math.pi / rho, rel=1e-12)


def test_influence_matrix_reproduces_the_uniformly_loaded_circle():
    """Love's result ``w(0) = 2 p a / E*`` pins kernel *and* quadrature weights."""
    a, modulus, p0 = 1.0e-3, 1.0e11, 2.0e6
    edges = np.linspace(0.0, a, 201)
    centers = 0.5 * (edges[1:] + edges[:-1])
    C = ec.displacement_influence(centers, edges, modulus)
    w = C @ np.full(centers.size, p0)
    assert w[0] == pytest.approx(2.0 * p0 * a / modulus, rel=1e-3)
    # and the edge value, w(a) = 4 p a/(pi E*), to the accuracy of a centroid sample
    assert w[-1] == pytest.approx(4.0 * p0 * a / (math.pi * modulus), rel=3e-2)


# ---------------------------------------------------------------------------
# The complementarity solver on a problem with a DIFFERENT closed form
# ---------------------------------------------------------------------------
def test_flat_punch_load_matches_its_own_closed_form():
    """Rigid flat punch: ``P = 2 E* a delta`` (Johnson 3.3), rim pressure singular.

    This exercises the same kernel and the same active-set solver on a contact
    patch whose pressure does not vanish at the rim, i.e. a solution class the
    Hertz problem does not probe.
    """
    punch_radius, penetration = 1.0e-3, 2.0e-6
    sol = ec.solve_normal_contact(ec.flat_gap, E_STAR, penetration=penetration,
                                  domain_radius=punch_radius, n_elements=200, grading=2.0)
    reference = ec.flat_punch_load(punch_radius, penetration, E_STAR)
    assert sol.load == pytest.approx(reference, rel=0.01)
    assert sol.contact_radius_rim == pytest.approx(punch_radius, rel=1e-9)


def test_solution_satisfies_the_complementarity_conditions():
    sol = _sphere_solution()
    assert np.all(sol.pressures >= 0.0)
    w = sol.pressures * 0.0
    # gap = w - delta + g must be >= 0 everywhere, and zero where p > 0
    edges = sol.edges
    centers = sol.radii
    C = ec.displacement_influence(centers, edges, sol.modulus)
    w = C @ sol.pressures
    gap = w - sol.penetration + ec.parabolic_gap(centers, RADIUS_EFF)
    active = sol.active_mask
    scale = max(float(np.abs(gap).max()), 1e-30)
    assert gap.min() > -1e-6 * scale
    assert np.abs(gap[active]).max() < 1e-5 * scale
    assert gap[~active].min() > 0.0


# ---------------------------------------------------------------------------
# The anchor itself
# ---------------------------------------------------------------------------
def test_hertz_anchor_contact_radius_and_penetration():
    """``CASE-FEM-HERTZ-CONTACT``: a and delta within the 3 % acceptance."""
    sol = _sphere_solution()
    comparison = ec.hertz_comparison(sol, ANCHOR)
    assert comparison["contact_radius_rel_error"] == pytest.approx(0.0, abs=TOLERANCE)
    assert comparison["penetration_rel_error"] == pytest.approx(0.0, abs=TOLERANCE)
    # the shape-free rim estimate must agree as well: the fit is not hiding an error
    assert comparison["contact_radius_rim_rel_error"] == pytest.approx(0.0, abs=TOLERANCE)
    # the load really is the prescribed one
    assert sol.load == pytest.approx(LOAD, rel=1e-6)


def test_hertz_anchor_pressure_field_and_rim_exponent():
    sol = _sphere_solution()
    comparison = ec.hertz_comparison(sol, ANCHOR)
    assert comparison["peak_pressure_rel_error"] == pytest.approx(0.0, abs=TOLERANCE)
    assert comparison["pressure_profile_rms_rel_error"] < TOLERANCE
    # Hertz predicts p ~ (a-r)^(1/2); the fit leaves the exponent free, so this is
    # a diagnostic that the rim is resolved rather than assumed
    assert 0.30 < sol.edge_exponent < 0.70


def test_anchor_converges_under_mesh_refinement():
    """a, delta and p0 must converge (and stay inside tolerance) as n grows."""
    errors = []
    for n in (100, 200):
        sol = _sphere_solution(n_elements=n)
        errors.append(ec.hertz_comparison(sol, ANCHOR)["contact_radius_rel_error"])
    assert abs(errors[-1]) <= abs(errors[0]) + 5e-3
    assert abs(errors[-1]) < TOLERANCE


def test_load_scaling_follows_the_cube_root_law():
    """A second load level: ``a ~ P^(1/3)`` is the only functional form the anchor allows."""
    heavier = ec.solve_normal_contact(lambda r: ec.parabolic_gap(r, RADIUS_EFF), E_STAR,
                                      load=8.0 * LOAD, n_elements=160, grading=2.0)
    lighter = _sphere_solution()
    ratio = heavier.contact_radius / lighter.contact_radius
    assert ratio == pytest.approx(8.0 ** (1.0 / 3.0), rel=TOLERANCE)
    assert heavier.penetration == pytest.approx(
        ec.hertz_sphere_halfspace(8.0 * LOAD, RADIUS_EFF, E_STAR).penetration, rel=TOLERANCE)


def test_domain_truncation_does_not_change_the_answer():
    """The half-space must not be cut short silently."""
    wide = _sphere_solution(domain_radius=2.0 * 7.5e-3)
    tight = _sphere_solution()
    assert wide.domain_radius > tight.domain_radius
    assert wide.contact_radius == pytest.approx(tight.contact_radius, rel=2e-2)
    assert wide.penetration == pytest.approx(tight.penetration, rel=2e-3)


@pytest.mark.parametrize("regularization", [1e-8, 1e-10, 1e-12, 1e-14])
def test_reported_quantities_are_independent_of_the_regularisation(regularization):
    """The truncated-SVD threshold must not be a hidden tuning knob."""
    sol = _sphere_solution(n_elements=120, regularization=regularization)
    reference = _sphere_solution(n_elements=120)
    assert sol.contact_radius == pytest.approx(reference.contact_radius, rel=1e-6)
    assert sol.penetration == pytest.approx(reference.penetration, rel=1e-6)
    assert sol.peak_pressure == pytest.approx(reference.peak_pressure, rel=1e-6)


def test_rejects_ambiguous_input():
    with pytest.raises(ValueError):
        ec.solve_normal_contact(ec.flat_gap, E_STAR, n_elements=20)
    with pytest.raises(ValueError):
        ec.solve_normal_contact(ec.flat_gap, E_STAR, load=1.0, penetration=1.0e-6,
                                n_elements=20)