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
from tensorbem.solver3d import assemble_surface_matrices, solve_laplace_surface

HAVE_CAPYTAINE = xval._capytaine_module() is not None
HAVE_SKFEM = xval._skfem_modules() is not None


@pytest.fixture(scope="module")
def bem_levels():
    return xval.route_tensorbem(subdivisions=(1, 2, 3), quadrature_order=8)


def test_exact_series_reproduces_boundary_data():
    mesh = icosphere(2, 1.0)
    mu = xval.panel_mu(mesh)
    pts = mesh.centroids / np.linalg.norm(mesh.centroids, axis=1)[:, None]
    phi = xval.exact_potential(pts)
    assert np.allclose(phi, -1.0 * (1.0 + mu), rtol=0.0, atol=1e-12)
    # the exact density is the one the BEM solves for: q = -dphi/dn_body
    assert np.allclose(xval.exact_density(mu), -(1 + 2 * pts[:, 2]), rtol=0.0, atol=1e-12)
    delta = 1e-5
    radial = -(xval.exact_potential((1 + delta) * pts)
               - xval.exact_potential((1 - delta) * pts)) / (2 * delta)
    assert np.allclose(xval.exact_density(mu), radial, rtol=0.0, atol=1e-9)


def test_exterior_potential_dipole_sign(record_property):
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
    ref = probes[:, 2] / np.linalg.norm(probes, axis=1)**3
    err = np.max(np.abs(phi - ref)) / np.max(np.abs(ref))
    record_property("dipole_relative_error", float(err))
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


def test_density_and_potential_converge(bem_levels, record_property):
    levels = bem_levels["levels"]
    for key in ("density_linf_rel_error", "probe_potential_l2_rel_error"):
        errors = [lvl[key] for lvl in levels]
        for level, error in zip(levels, errors):
            record_property(f"{key}_{level['panels']}_panels", error)
        assert errors[-1] < errors[0]
        assert errors[-1] < 0.01
    orders = xval._observed_order(levels, "probe_potential_l2_rel_error")
    record_property("final_probe_order_in_h", orders[-1])
    assert orders[-1] > 1.2  # second-order collocation on plane triangles


def test_probe_potential_matches_exact(bem_levels):
    finest = bem_levels["levels"][-1]
    assert finest["probe_potential_linf_rel_error"] < 0.01


@pytest.mark.skipif(not HAVE_CAPYTAINE, reason="capytaine not installed")
def test_capytaine_assembly_agreement():
    """Element-wise: TensorBEM assembly == capytaine Rankine influence matrices."""
    mesh = icosphere(2, 1.0)
    single, double, info = xval.capytaine_matrices(mesh)
    a_ref, d_ref = assemble_surface_matrices(mesh)
    # Capytaine includes the boundary jump in the double-layer diagonal.
    assert np.allclose(np.diag(double), 0.5, rtol=0.0, atol=1e-12)
    principal_value = -double.copy()
    np.fill_diagonal(principal_value, 0.0)
    assert single.shape == a_ref.shape
    assert np.linalg.norm(-single - a_ref) / np.linalg.norm(a_ref) < 1e-2
    assert np.linalg.norm(principal_value - d_ref) / np.linalg.norm(d_ref) < 1e-2
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

@pytest.mark.parametrize("domain", ["interior", "exterior"])
def test_nonaxisymmetric_sphere_quadrupole_field_and_energy(domain, record_property):
    from tensorbem.solver3d import evaluate_surface_field, evaluate_surface_velocity

    radius = 1.4
    eigenvalue = (2 if domain == "interior" else 3) / radius
    # u(a)=n_x*n_y and integral over the unit sphere of n_x²*n_y² is 4π/15.
    exact_energy = eigenvalue * radius**2 * 4*np.pi/15
    directions = np.array([[1., 2., 1.], [-2., 3., 1.], [1., -1., 2.]])
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    radii = np.array([.35, .55, .75] if domain == "interior" else [1.35, 1.7, 2.2])
    probes = radii[:, None] * radius * directions
    exact = directions[:, 0]*directions[:, 1]*radii**(2 if domain == "interior" else -3)
    # Surface gradient of n_x*n_y is its Cartesian gradient projected
    # onto the tangent plane. The radial derivative carries the signed degree.
    degree = 2 if domain == "interior" else -3
    angular_gradient = np.column_stack((directions[:, 1], directions[:, 0], np.zeros(len(probes))))
    angular_gradient -= 2*(directions[:, 0]*directions[:, 1])[:, None]*directions
    exact_radial = degree*exact/(radius*radii)
    exact_tangent = (radii**(degree-1)/radius)[:, None]*angular_gradient
    errors = []
    for level in (1, 2, 3):
        mesh = icosphere(level, radius)
        normal = mesh.centroids / np.linalg.norm(mesh.centroids, axis=1)[:, None]
        trace = normal[:, 0]*normal[:, 1]
        result = solve_laplace_surface(mesh, trace, domain=domain, quadrature_order=8)
        sampled = evaluate_surface_field(mesh, result, probes)
        velocity = evaluate_surface_velocity(mesh, result, probes)
        radial = np.sum(velocity*directions, axis=1)
        tangent = velocity-radial[:, None]*directions
        radial_error = np.linalg.norm(radial-exact_radial)/np.linalg.norm(exact_radial)
        tangent_error = np.linalg.norm(tangent-exact_tangent)/np.linalg.norm(exact_tangent)
        energy = np.dot(mesh.areas*trace, result.normal_derivative)
        field_error = np.linalg.norm(sampled-exact)/np.linalg.norm(exact)
        energy_error = abs(energy/exact_energy-1)
        net_flux = abs(mesh.areas @ result.normal_derivative)
        for name, value in (("relative_field_error", field_error), ("relative_energy_error", energy_error),
                            ("relative_radial_gradient_error", radial_error),
                            ("relative_tangential_gradient_error", tangent_error),
                            ("net_flux_m", net_flux), ("condition_number", result.condition_number)):
            record_property(f"{name}_{mesh.panel_count}", float(value))
        assert net_flux < 1e-11
        errors.append((field_error, energy_error, radial_error, tangent_error))
    for coarse, fine in zip(errors, errors[1:]):
        assert all(a/b > 2.5 for a, b in zip(coarse, fine))


@pytest.mark.parametrize("domain", ["interior", "exterior"])
def test_near_surface_quadrupole_at_fixed_and_panel_scaled_clearance(domain, record_property):
    from tensorbem.solver3d import evaluate_surface_field_near_surface, evaluate_surface_velocity_near_surface

    from tensorbem import recover_surface_velocity_at_offsets

    radius = 1.4
    sign = -1 if domain == "interior" else 1
    directions = np.array([[1., 2., 1.], [-2., 3., 1.], [1., -1., 2.]])
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    fixed_errors = []
    panel_field_errors = []
    recovered_errors = []

    def exact_field(points):
        x, y, _ = points.T
        exact = x*y/radius**2
        gradient = np.column_stack((y, x, np.zeros_like(x)))/radius**2
        if domain == "exterior":
            r = np.linalg.norm(points, axis=1)
            factor = (radius/r)**5
            gradient = factor[:, None]*(gradient-5*exact[:, None]*points/r[:, None]**2)
            exact *= factor
        return exact, gradient

    for level in (1, 2, 3):
        mesh = icosphere(level, radius)
        normal = mesh.centroids / np.linalg.norm(mesh.centroids, axis=1)[:, None]
        field = solve_laplace_surface(mesh, normal[:, 0]*normal[:, 1], domain=domain)
        panels = np.argmax(directions @ normal.T, axis=1)
        for label, probes in (
            ("fixed", (1+sign*.10)*radius*directions),
            ("panel", mesh.centroids[panels]+sign*.01*mesh.h_max*mesh.normals[panels]),
        ):
            # The sphere harmonics extend smoothly to points just off the
            # faceted surface, including where it lies inside the exact sphere.
            exact, gradient = exact_field(probes)
            sampled = evaluate_surface_field_near_surface(mesh, field, probes)
            velocity = evaluate_surface_velocity_near_surface(mesh, field, probes)
            errors = [np.linalg.norm(sampled-exact)/np.linalg.norm(exact),
                      np.linalg.norm(velocity-gradient)/np.linalg.norm(gradient)]
            for name, error in zip(("field", "gradient"), errors):
                record_property(f"{label}_{name}_relative_error_{mesh.panel_count}", float(error))
            if label == "fixed":
                fixed_errors.append(errors)
            else:
                panel_field_errors.append(errors[0])
                recovered = recover_surface_velocity_at_offsets(mesh, field, .01*mesh.h_max)
                recovered_error = np.linalg.norm(recovered[panels]-gradient)/np.linalg.norm(gradient)
                recovered_errors.append(recovered_error)
                record_property(f"panel_recovered_gradient_relative_error_{mesh.panel_count}",
                                float(recovered_error))
                # Isolate the normal Taylor correction from the baseline trace error.
                wall = recover_surface_velocity_at_offsets(mesh, field, 0.)[panels]
                _, exact_wall = exact_field(mesh.centroids[panels])
                change = gradient-exact_wall
                correction_error = np.linalg.norm(recovered[panels]-wall-change)/np.linalg.norm(change)
                record_property(f"normal_increment_relative_error_{mesh.panel_count}", float(correction_error))
                if level == 3:
                    assert correction_error < 0.5
                # At shrinking clearance the continuum gradient includes panel
                # discretization error. Check differentiation of the *same*
                # discrete potential separately using centered differences.
                delta = mesh.h_max*1e-4
                offsets = delta*np.eye(3)
                plus = (probes[:, None, :]+offsets).reshape(-1, 3)
                minus = (probes[:, None, :]-offsets).reshape(-1, 3)
                values = evaluate_surface_field_near_surface(mesh, field, np.vstack((plus, minus)))
                hi, lo = np.split(values, 2)
                derivative = ((hi-lo)/(2*delta)).reshape(-1, 3)
                error = np.linalg.norm(velocity-derivative)/np.linalg.norm(derivative)
                record_property(f"panel_gradient_fd_relative_error_{mesh.panel_count}", float(error))
                assert error < 1e-5
    for coarse, fine in zip(fixed_errors, fixed_errors[1:]):
        assert all(a > b for a, b in zip(coarse, fine))
    assert all(a/b > 3 for a, b in zip(panel_field_errors, panel_field_errors[1:]))
    assert max(fixed_errors[-1]) < 0.015
    assert recovered_errors[-1] < 0.05
    assert all(a > b for a, b in zip(recovered_errors, recovered_errors[1:]))
