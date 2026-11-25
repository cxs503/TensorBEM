import numpy as np
import pytest

from tensorbem import regular_polygon, solve_laplace_dirichlet
from tensorbem.solver import assemble_matrices, evaluate_interior_field


def test_solver_returns_finite_solution_and_small_algebraic_residual():
    mesh = regular_polygon(24)
    values = mesh.centers[:, 0]
    result = solve_laplace_dirichlet(mesh, values)
    assert result.normal_derivative.shape == (24,)
    assert np.isfinite(result.normal_derivative).all()
    assert result.residual_l2 < 1.0e-10
    assert result.condition_number > 0.0


def test_solver_rejects_wrong_boundary_value_shape():
    mesh = regular_polygon(12)
    with pytest.raises(ValueError, match="shape"):
        solve_laplace_dirichlet(mesh, np.zeros(11))


def test_constant_boundary_value_has_near_zero_flux():
    mesh = regular_polygon(48)
    result = solve_laplace_dirichlet(mesh, np.ones(48))
    assert np.sqrt(np.mean(result.normal_derivative**2)) < 3.0e-2


def test_manufactured_linear_harmonic_solution_improves_with_refinement():
    errors = []
    for n in (24, 48):
        mesh = regular_polygon(n)
        result = solve_laplace_dirichlet(mesh, mesh.centers[:, 0])
        theta = np.arctan2(mesh.centers[:, 1], mesh.centers[:, 0])
        errors.append(np.sqrt(np.mean((result.normal_derivative - np.cos(theta)) ** 2)))
    assert errors[1] < errors[0]


def test_interior_field_matches_linear_harmonic_solution():
    mesh = regular_polygon(64)
    result = solve_laplace_dirichlet(mesh, mesh.centers[:, 0])
    points = np.array([[0.1, 0.2], [-0.3, 0.1], [0.6, -0.2]])
    values = evaluate_interior_field(mesh, result, points)
    assert values.shape == (len(points),)
    assert np.sqrt(np.mean((values - points[:, 0]) ** 2)) < 2.0e-2


def test_interior_field_rejects_invalid_evaluation_points():
    mesh = regular_polygon(16)
    result = solve_laplace_dirichlet(mesh, mesh.centers[:, 0])
    with pytest.raises(ValueError, match=r"shape \(M, 2\)"):
        evaluate_interior_field(mesh, result, np.array([0.0, 0.0]))
    with pytest.raises(ValueError, match="finite"):
        evaluate_interior_field(mesh, result, np.array([[np.nan, 0.0]]))


def test_assembled_double_layer_matrix_has_zero_self_terms_and_finite_entries():
    single, double = assemble_matrices(regular_polygon(32))
    assert np.isfinite(single).all()
    assert np.isfinite(double).all()
    assert np.allclose(np.diag(double), 0.0)
    assert np.all(np.diag(single) > 0.0)


def test_exact_panel_integration_preserves_refinement_trend():
    errors = []
    for n in (16, 32, 64):
        mesh = regular_polygon(n)
        result = solve_laplace_dirichlet(mesh, mesh.centers[:, 0])
        theta = np.arctan2(mesh.normals[:, 1], mesh.normals[:, 0])
        errors.append(np.sqrt(np.mean((result.normal_derivative - np.cos(theta)) ** 2)))
    assert all(b < a for a, b in zip(errors, errors[1:]))


def test_dirichlet_solve_survives_an_infinite_condition_estimate(monkeypatch):
    """The condition-number gate must not reject a solvable system.

    The gate exists to reject the logarithmic-capacity degeneracy, not to
    depend on the LAPACK SVD succeeding: on the loongarch64 nodes
    ``numpy.linalg.cond`` returned ``inf`` for the 128-panel single-layer
    matrix whose true condition number is ~76, which made every case that
    touches this solve unrunnable there.  With the SVD unavailable the solve
    must still succeed, keep its algebraic residual, and record that the
    condition number came from the LU-based fallback.
    """
    mesh = regular_polygon(32)
    values = mesh.centers[:, 0]
    reference = solve_laplace_dirichlet(mesh, values)
    assert reference.condition_number_source == "svd"

    monkeypatch.setattr(np.linalg, "cond", lambda *a, **k: float("inf"))
    result = solve_laplace_dirichlet(mesh, values)
    assert result.condition_number_source == "power_iteration"
    assert np.isfinite(result.condition_number)
    assert result.relative_residual_l2 < 1.0e-10
    assert np.allclose(result.normal_derivative, reference.normal_derivative, rtol=0.0, atol=1.0e-12)


def test_dirichlet_gate_still_rejects_a_singular_system(monkeypatch):
    """If the fallback also fails to bound the condition number, raise."""
    import tensorbem.solver as solver_module

    mesh = regular_polygon(16)
    monkeypatch.setattr(np.linalg, "cond", lambda *a, **k: float("inf"))
    monkeypatch.setattr(
        solver_module,
        "condition_number_with_fallback",
        lambda matrix: (float("inf"), "power_iteration"),
    )
    with pytest.raises(ValueError, match="numerically singular"):
        solve_laplace_dirichlet(mesh, mesh.centers[:, 0])


@pytest.mark.parametrize(("size", "origin"), [(1.0, 0.0), (1.0, 2.0**45), (1e-60, 0.0), (1e60, 0.0)])
def test_rectangle_harmonic_field_in_scaled_coordinates(size, origin, record_property):
    from tensorbem.geometry import PanelMesh

    corners = np.array([[0, 0], [4, 0], [4, 2], [0, 2]])
    edges = np.roll(corners, -1, axis=0) - corners
    fractions = np.arange(16) / 16
    vertices = corners[:, None, :] + fractions[None, :, None] * edges[:, None, :]
    mesh = PanelMesh(origin + size * vertices.reshape(-1, 2))
    values = (mesh.centers[:, 0] - origin) / size
    solution = solve_laplace_dirichlet(mesh, values, reference_length=8 * size)
    probes = np.array([[1.0, 0.75], [2.5, 1.25]])
    computed = evaluate_interior_field(mesh, solution, origin + size * probes)
    error = float(np.max(abs(computed - probes[:, 0])))
    record_property("max_absolute_potential_error", error)
    assert error < 6e-6


@pytest.mark.parametrize("linear", [False, True])
def test_interior_field_gradient_cubic_harmonic_on_ellipse(linear, record_property):
    from tensorbem import evaluate_interior_field_and_gradient
    from tensorbem.geometry import ellipse_mesh

    points = np.array([[.1,.12], [-.42,.13], [.34,-.22], [-.18,-.28]])
    z = points[:,0] + 1j*points[:,1]
    exact = (z**3).real
    gradient = np.column_stack(((3*z*z).real, -(3*z*z).imag))
    errors = []
    for count in (24,48,96):
        mesh = ellipse_mesh(count)
        centers = mesh.centers[:,0] + 1j*mesh.centers[:,1]
        result = solve_laplace_dirichlet(mesh, (centers**3).real, linear_density=linear)
        values, computed = evaluate_interior_field_and_gradient(mesh, result, points)
        row = (np.max(np.abs(values-exact)), np.max(np.linalg.norm(computed-gradient, axis=1)))
        errors.append(row)
        record_property(f"max_potential_error_{count}", row[0])
        record_property(f"max_gradient_error_{count}", row[1])
        if not linear:
            np.testing.assert_allclose(values, evaluate_interior_field(mesh,result,points), atol=2e-15)
    for coarse, fine in zip(errors,errors[1:]):
        assert all(a/b > 3 for a,b in zip(coarse,fine))
    with pytest.raises(ValueError, match="outside"):
        evaluate_interior_field_and_gradient(mesh, result, np.array([[5.,5.]]))


@pytest.mark.parametrize("linear", [False, True])
def test_interior_gradient_at_fixed_and_panel_scaled_clearance(linear, record_property):
    from tensorbem import evaluate_interior_field_and_gradient

    # u=x²-y². Sample eight azimuths, including oblique panels where the
    # tangential derivative is large, and two locations along each panel.
    previous = {}
    for count in (24, 48, 96):
        mesh = regular_polygon(count, 1.)
        ids = np.arange(8)*count//8
        x, y = mesh.centers.T
        result = solve_laplace_dirichlet(mesh, x*x-y*y, reference_length=2., linear_density=linear)
        for shift in (0., .4):
            feet = mesh.centers[ids] + shift*mesh.vectors[ids]
            for label, depth in (("fixed", .15), ("scaled", .2*mesh.lengths[0])):
                points = feet-depth*mesh.normals[ids]
                _, grad = evaluate_interior_field_and_gradient(mesh, result, points)
                exact = 2*points*np.array([1., -1.])
                error = float(np.max(np.linalg.norm(grad-exact, axis=1)))
                record_property(f"gradient_error_{label}_shift_{shift}_{count}", error)
                key = (label, shift)
                if count > 24 and label == "fixed":
                    assert error < previous[key]/3.
                if count > 24 and label == "scaled" and linear:
                    assert error < previous[key]/1.45
                previous[key] = error
    assert max(previous["scaled", s] for s in (0., .4)) < (.008 if linear else .75)


@pytest.mark.parametrize("linear", [False, True])
def test_interior_surface_gradient_tracks_harmonic_quadratic(linear, record_property):
    from dataclasses import replace
    from tensorbem import evaluate_interior_surface_gradient
    from tensorbem.geometry import ellipse_mesh

    errors = []
    for n in (24, 48, 96):
        mesh = ellipse_mesh(n)
        x, y = mesh.centers.T
        # Re((1-i/2) z²) gives both normal and tangential variation.
        result = solve_laplace_dirichlet(mesh, x*x-y*y+x*y, linear_density=linear)
        recovered = evaluate_interior_surface_gradient(mesh, result)
        exact = np.column_stack((2*x+y, x-2*y))
        normal = np.sum(recovered*mesh.normals, axis=1)
        np.testing.assert_allclose(normal, result.normal_derivative, atol=1e-14, rtol=0.)
        difference = recovered-exact
        tangents = mesh.vectors/mesh.lengths[:,None]
        normal_error = np.max(np.abs(np.sum(difference*mesh.normals, axis=1)))
        tangent_error = np.max(np.abs(np.sum(difference*tangents, axis=1)))
        record_property(f"max_normal_error_{n}", normal_error)
        record_property(f"max_tangential_error_{n}", tangent_error)
        errors.append(float(np.max(np.linalg.norm(difference,axis=1))))
        record_property(f"max_gradient_error_{n}", errors[-1])
    assert all(a/b > 3.0 for a,b in zip(errors,errors[1:]))
    shifted = replace(result, boundary_values=result.boundary_values+8.)
    np.testing.assert_allclose(evaluate_interior_surface_gradient(mesh,shifted), recovered,
                               atol=1e-13, rtol=0.)
    with pytest.raises(ValueError, match="interior solve"):
        evaluate_interior_surface_gradient(mesh, replace(result, domain="exterior"))


def test_interior_gradient_at_rectangle_corners(record_property):
    from tensorbem import evaluate_interior_surface_gradient
    from tensorbem.geometry import PanelMesh

    corners = np.array([[0.,0.], [2.,0.], [2.,1.], [0.,1.]])
    edges = np.roll(corners,-1,axis=0)-corners
    for n in (4, 8, 16):
        fraction = (np.arange(n)/n)**1.25
        vertices = corners[:,None,:] + fraction[None,:,None]*edges[:,None,:]
        mesh = PanelMesh(np.roll(vertices.reshape(-1,2),1,axis=0))
        indices = 1 + n*np.arange(4)
        x,y = mesh.centers.T
        result = solve_laplace_dirichlet(mesh, x*x-y*y+x*y, reference_length=4.)
        exact = np.column_stack((2*x+y, x-2*y))
        tangent = mesh.vectors / mesh.lengths[:,None]
        recovered = evaluate_interior_surface_gradient(mesh,result,corner_vertices=indices[::-1])
        difference = recovered-exact
        adjacent = np.r_[indices, indices-1]
        tangent_error = float(np.max(np.abs(np.sum(difference*tangent,axis=1)[adjacent])))
        normal_error = float(np.max(np.abs(np.sum(difference*mesh.normals,axis=1)[adjacent])))
        default = evaluate_interior_surface_gradient(mesh,result)
        centered_error = float(np.max(np.abs(np.sum((default-exact)*tangent,axis=1)[adjacent])))
        record_property(f"one_sided_tangent_error_{n}",tangent_error)
        record_property(f"periodic_tangent_error_{n}",centered_error)
        record_property(f"normal_error_{n}",normal_error)
        assert tangent_error < 2e-13
        np.testing.assert_allclose(np.sum(recovered*mesh.normals,axis=1),result.normal_derivative,
                                   atol=1e-14,rtol=0.)
    for invalid in ([0, 1], [0, 0], [-1], [1.5], [mesh.panel_count]):
        with pytest.raises(ValueError):
            evaluate_interior_surface_gradient(mesh,result,corner_vertices=invalid)


def test_interior_corner_segments_with_two_panels_use_secants():
    from tensorbem import evaluate_interior_surface_gradient
    from tensorbem.geometry import PanelMesh

    corners = np.array([[0.,0.], [2.,0.], [0.,1.]])
    vertices = np.stack((corners,.5*(corners+np.roll(corners,-1,axis=0))),axis=1)
    mesh = PanelMesh(vertices.reshape(-1,2))
    x,y = mesh.centers.T
    result = solve_laplace_dirichlet(mesh, 2*x-3*y, reference_length=4.)
    grad = evaluate_interior_surface_gradient(mesh,result,corner_vertices=[0,2,4])
    tangent = mesh.vectors/mesh.lengths[:,None]
    np.testing.assert_allclose(np.sum(grad*tangent,axis=1), tangent @ [2.,-3.], atol=1e-14)


@pytest.mark.parametrize("quadratic", [False, True])
def test_corner_delimited_density_recovers_rectangle_normal_flux(quadratic, record_property):
    from tensorbem import evaluate_interior_field_and_gradient, evaluate_interior_surface_gradient
    from tensorbem.geometry import PanelMesh

    corners = np.array([[0.,0.], [2.,0.], [2.,1.], [0.,1.]])
    edges = np.roll(corners,-1,axis=0)-corners
    errors = []
    for n in (4,8,16):
        fraction = (np.arange(n)/n)**1.25
        vertices = corners[:,None,:] + fraction[None,:,None]*edges[:,None,:]
        mesh = PanelMesh(np.roll(vertices.reshape(-1,2),1,axis=0))
        ids = 1+n*np.arange(4)
        x,y = mesh.centers.T
        values = x*x-y*y+x*y if quadratic else 2*x-3*y
        exact = np.column_stack((2*x+y,x-2*y)) if quadratic else np.array([2.,-3.])
        result = solve_laplace_dirichlet(mesh,values,reference_length=4.,linear_density=True,
                                         corner_vertices=ids[::-1])
        normal_error = np.abs(result.normal_derivative - np.sum(exact*mesh.normals,axis=1))
        error = float(np.max(normal_error[np.r_[ids,ids-1]]))
        errors.append(error)
        record_property(f"corner_adjacent_normal_error_{n}",error)
        probes = np.array([[.4,.25],[1.2,.6]])
        xp,yp = probes.T
        potential, gradient = evaluate_interior_field_and_gradient(mesh,result,probes)
        np.testing.assert_allclose(evaluate_interior_field(mesh,result,probes),potential)
        if not quadratic:
            np.testing.assert_allclose(potential,2*xp-3*yp,atol=2e-12,rtol=0.)
            np.testing.assert_allclose(gradient, np.tile(exact,(2,1)),atol=2e-12,rtol=0.)
            surface = evaluate_interior_surface_gradient(mesh,result)
            np.testing.assert_allclose(surface,np.tile(exact,(4*n,1)),atol=2e-12,rtol=0.)
    if quadratic:
        assert errors[0] > errors[1] > errors[2]
    else:
        assert max(errors) < 2e-12


@pytest.mark.parametrize("grading", [1, 2])
def test_reentrant_corner_flux_against_singular_wedge_mode(grading, record_property):
    from tensorbem.geometry import PanelMesh

    # The removed quadrant is x>0, y<0. The two rays from the origin are
    # Dirichlet walls of the 3π/2 wedge. The first harmonic mode has finite
    # flux integral -R^(2/3) on either ray despite its singular derivative.
    corners = np.array([[0.,0.], [1.,0.], [1.,1.], [-1.,1.], [-1.,-1.], [0.,-1.]])
    edges = np.roll(corners, -1, axis=0) - corners
    errors = []
    for n in (4, 8, 16, 32):
        fractions = np.tile(np.arange(n)/n, (6,1))
        fractions[0] **= grading
        fractions[-1] = 1-(1-fractions[-1])**grading
        vertices = corners[:,None,:] + fractions[:,:,None]*edges[:,None,:]
        mesh = PanelMesh(vertices.reshape(-1,2))
        x, y = mesh.centers.T
        radius = np.hypot(x,y)
        theta = np.mod(np.arctan2(y,x), 2*np.pi)
        values = radius**(2/3)*np.sin(2*theta/3)
        result = solve_laplace_dirichlet(mesh, values, reference_length=4.,
                                         linear_density=True, corner_vertices=n*np.arange(6))
        q = result.normal_derivative.reshape(6,n)
        lengths = mesh.lengths.reshape(6,n)
        cumulative_errors, tip_errors = [], []
        for side, step in ((0,1),(-1,-1)):
            widths = lengths[side,::step]
            densities = q[side,::step]
            # The midpoint density integrates exactly for a linear panel.
            integrated = np.cumsum(widths*densities)
            cumulative_errors.append(np.max(np.abs(integrated+np.cumsum(widths)**(2/3))))
            tip_exact = -(2/3)*(widths[0]/2)**(-1/3)
            tip_errors.append(abs(densities[0]/tip_exact-1))
        errors.append(float(max(cumulative_errors)))
        balance = abs(np.sum(q*lengths))
        record_property(f"max_prefix_flux_error_{n}", errors[-1])
        record_property(f"max_tip_midpoint_relative_error_{n}", max(tip_errors))
        record_property(f"net_flux_residual_{n}", balance)
        record_property(f"condition_number_{n}", result.condition_number)
        assert balance < .025/n
    assert all(a/b > (1.5 if grading==1 else 2.4) for a,b in zip(errors, errors[1:]))
    assert errors[-1] < (.002 if grading==1 else .0005)
