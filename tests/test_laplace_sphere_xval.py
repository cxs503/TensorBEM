"""Cross-validation (XVAL) of the exterior-Laplace BIE kernel against OSS.

Case: sphere of radius ``a`` in an unbounded electrolyte, Dirichlet data
``phi(a, mu) = phi_a (1 + mu)``.  The ``l = 1`` Legendre mode is what makes
these tests discriminating: it is the first mode that a wrong quadrature, a
wrong double-layer sign or a missing ``sigma`` can corrupt, while the uniform
(``l = 0``) conducting-sphere anchor used elsewhere in the suite cannot see any
of them.

References: analytic Legendre series (truth), Capytaine influence matrices in
the Rankine limit (OSS BEM, element-wise assembly comparison), scikit-fem
axisymmetric Galerkin (OSS FEM, independent discretisation family).
"""

from __future__ import annotations

import math
import types

import numpy as np
import pytest

from tensorbem.benchmarks import laplace_sphere_xval as xval
from tensorbem.cathodic_protection import evaluate_exterior_potential
from tensorbem.geometry3d import icosphere
from tensorbem.solver3d import solve_laplace_surface

HAVE_CAPYTAINE = xval._capytaine_module() is not None
HAVE_SKFEM = xval._skfem_modules() is not None


@pytest.fixture(scope="module")
def bem_levels():
    return xval.route_tensorbem(subdivisions=(1, 2, 3), quadrature_order=8)


def test_exact_series_reproduces_boundary_data():
    mesh = icosphere(2, 1.0)
    mu = xval.panel_mu(mesh)
    pts = mesh.centroids
    phi = xval.exact_potential(pts)
    assert np.allclose(phi, -1.0 * (1.0 + mu), rtol=0.0, atol=1e-12)
    # the exact density is the one the BEM solves for: q = -dphi/dn_body
    radial = -(-1.0) / 1.0 - 2.0 * (-1.0) * mu / 1.0
    assert np.allclose(xval.exact_density(mu), radial, rtol=0.0, atol=1e-12)


def test_exterior_potential_dipole_sign():
    """The double-layer term of the representation formula must be ADDED.

    Pure ``l = 1`` mode: ``phi = mu`` on the sphere, exact exterior solution
    ``(a/r)^2 mu``, exact density ``q = 2 mu / a``.  Substituting the exact data
    into the representation formula isolates the quadrature + sign convention:
    adding the double layer reproduces the dipole to the geometry error of the
    faceted sphere, subtracting it is off by ~2/3.  This is the regression for
    the sign defect found by the round-2 XVAL (uniform-potential cases are
    blind to it because the double layer of a constant vanishes outside).
    """
    mesh = icosphere(3, 1.0)
    mu = xval.panel_mu(mesh)
    values = mu.copy()
    density = 2.0 * mu
    probes = xval.probe_points()
    phi = xval.representation_potential(mesh, density, values, probes)
    ref = xval.exact_potential(probes, phi_a=1.0) - 1.0  # strip the monopole term
    err = np.max(np.abs(phi - ref)) / np.max(np.abs(ref))
    assert err < 0.05, f"dipole representation error {err:.3%}"

    # the production post-processor must agree with the module-level formula
    fake = types.SimpleNamespace(potential=values, single_layer_density=density)
    assert np.allclose(
        evaluate_exterior_potential(mesh, fake, probes), phi, rtol=1e-10, atol=1e-12
    )


def test_current_anchor_within_ledger_band(bem_levels):
    finest = bem_levels["levels"][-1]
    assert finest["current_rel_error"] < 0.005
    assert finest["dimensionless_current"] == pytest.approx(1.0, abs=0.005)


def test_density_and_potential_converge(bem_levels):
    levels = bem_levels["levels"]
    for key in ("density_linf_rel_error", "probe_potential_l2_rel_error"):
        errors = [lvl[key] for lvl in levels]
        assert errors[-1] < errors[0]
        assert errors[-1] < 0.01
    orders = xval._observed_order(levels, "probe_potential_l2_rel_error")
    assert orders[-1] > 1.2  # second-order collocation on plane triangles


def test_probe_potential_matches_exact(bem_levels):
    finest = bem_levels["levels"][-1]
    assert finest["probe_potential_linf_rel_error"] < 0.01


@pytest.mark.skipif(not HAVE_CAPYTAINE, reason="capytaine not installed")
def test_capytaine_assembly_agreement():
    """Element-wise: TensorBEM assembly == capytaine Rankine influence matrices."""
    mesh = icosphere(2, 1.0)
    single, double, info = xval.capytaine_matrices(mesh)
    a_ref, d_ref = xval.solver3d.assemble_surface_matrices(mesh)
    assert single.shape == a_ref.shape
    assert np.linalg.norm(-single - a_ref) / np.linalg.norm(a_ref) < 1e-2
    assert np.linalg.norm(-double - d_ref) / np.linalg.norm(d_ref) < 1e-2
    assert info["mesh_panels"] == mesh.panel_count


@pytest.mark.skipif(not HAVE_CAPYTAINE, reason="capytaine not installed")
def test_capytaine_route_end_to_end(bem_levels):
    capy = xval.route_capytaine(subdivisions=(2,))
    level = capy["levels"][0]
    assert level["current_rel_error"] < 0.02
    assert level["probe_potential_l2_rel_error"] < 0.05
    bem = bem_levels["levels"][1]  # subdivisions = 2
    assert abs(level["current_A"] - bem["current_A"]) / abs(bem["current_A"]) < 0.005


@pytest.mark.skipif(not HAVE_SKFEM, reason="scikit-fem not installed")
def test_skfem_axisymmetric_reference():
    fem = xval.route_skfem(outer_ratios=(4.0,), resolutions=((24, 16), (48, 32)), order=2)
    fine = fem["levels"][-1]
    assert fine["current_rel_error"] < 5e-3
    assert fine["probe_potential_linf_rel_error"] < 1e-5
    assert fine["current_conservation_rel"] < 5e-3   # inner vs outer sphere flux
    extrapolated = fem["richardson_current"][0]
    assert extrapolated["richardson_rel_error"] < 5e-6


def test_sigma_scaling_is_exact():
    """The dimensionless current must be invariant under (sigma, a, phi_a)."""
    ratios = []
    for sigma, radius, phi_a in ((3.5, 1.0, -1.0), (4.0, 2.0, -0.9), (1.0, 0.5, -1.2)):
        mesh = icosphere(2, radius)
        mu = xval.panel_mu(mesh, radius)
        values = phi_a * (1.0 + mu)
        res = solve_laplace_surface(mesh, values, domain="exterior")
        current = sigma * float(np.sum(res.normal_derivative * mesh.areas))
        exact = xval.exact_current(radius, phi_a, sigma)
        ratios.append(current / exact)
    assert max(ratios) - min(ratios) < 1e-12
    assert math.isclose(ratios[0], 0.987, rel_tol=0.02)