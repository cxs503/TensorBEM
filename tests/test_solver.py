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
