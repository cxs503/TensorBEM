"""Guards for the 3-D BEM solve-stage health audit.

Background (dev report m20910): on this ecosystem's loongarch64 cluster nodes
(numpy 2.0.1) ``np.linalg.lstsq`` returns a *full-rank, perfectly conditioned,
wrong* solution of the CHIEF-augmented hemisphere system -- relative residual
4.9e-1 against 1.8e-4 on the control plane, with a bit-identical system matrix.
The solver had no way to notice, so it published a 51 %-wrong damping
coefficient without a single diagnostic firing.

These tests pin the two invariants that make the failure loud (or repaired):

1. the stationarity measure ``||A^H r|| / (||A|| ||r||)`` is O(eps) at the
   least-squares minimum *even when the system is genuinely inconsistent*, so
   the guard cannot be satisfied by simply returning any low-residual vector;
2. a driver that returns a wrong vector is detected and replaced by the
   Householder QR path, with a warning, and the recovered vector is the true
   least-squares minimiser.

The negative controls matter as much as the positive ones: a guard that fires
on the healthy path would either break every board case or be disabled.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tensorbem.free_surface_solver3d import (
    FreeSurface3DOperators,
    _solve_least_squares,
    _solve_square,
    _solve_system,
    least_squares_stationarity,
    solve_radiation_3d,
)


def _inconsistent_complex_system(m: int = 40, n: int = 30, seed: int = 7):
    """Over-determined complex system with no exact solution."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(m, n)) + 1.0j * rng.normal(size=(m, n))
    rhs = rng.normal(size=(m, 1)) + 1.0j * rng.normal(size=(m, 1))
    return matrix, rhs


def test_stationarity_is_zero_at_the_minimum_of_an_inconsistent_system():
    """The CHIEF system is inconsistent by construction; stationarity is not fooled."""
    matrix, rhs = _inconsistent_complex_system()
    sol, method = _solve_least_squares(matrix, rhs)
    assert method == "lstsq"  # healthy driver: no fallback, no warning
    resid = matrix @ sol - rhs
    rel_res = float(np.linalg.norm(resid) / np.linalg.norm(rhs))
    # The residual is large on purpose (1e-2 scale) ...
    assert rel_res > 1e-6
    # ... yet the point is stationary, i.e. the guard tolerates inconsistency.
    assert least_squares_stationarity(matrix, sol, rhs) < 1e-10


def test_stationarity_flags_a_perturbed_solution():
    """Negative control: a 1 % perturbation of a correct solve is not stationary."""
    matrix, rhs = _inconsistent_complex_system()
    sol, _ = _solve_least_squares(matrix, rhs)
    healthy = least_squares_stationarity(matrix, sol, rhs)
    perturbed = sol * (1.0 + 1.0e-2)
    flagged = least_squares_stationarity(matrix, perturbed, rhs)
    assert flagged > 1e-5
    assert flagged / max(healthy, 1e-300) > 1e6


def test_fallback_recovers_the_minimiser_when_the_driver_lies(monkeypatch):
    """A driver returning zeros is caught and the QR path returns the real answer."""
    matrix, rhs = _inconsistent_complex_system()
    reference, _ = _solve_least_squares(matrix, rhs)

    def broken_lstsq(a, b, rcond=None):  # noqa: ARG001 - signature compatibility
        garbage = np.zeros((a.shape[1], b.shape[1]), dtype=np.complex128)
        return garbage, np.zeros(1), int(a.shape[1]), np.zeros(1)

    monkeypatch.setattr(np.linalg, "lstsq", broken_lstsq)
    with pytest.warns(RuntimeWarning, match="stationarity audit"):
        sol, method = _solve_least_squares(matrix, rhs)
    assert method == "qr"
    np.testing.assert_allclose(sol, reference, rtol=0, atol=1e-12)
    assert least_squares_stationarity(matrix, sol, rhs) < 1e-10


def test_both_paths_failing_is_loud_not_silent(monkeypatch):
    """If neither driver works we raise; returning a wrong vector is not an option."""
    matrix, rhs = _inconsistent_complex_system()

    def broken_lstsq(a, b, rcond=None):  # noqa: ARG001
        garbage = np.zeros((a.shape[1], b.shape[1]), dtype=np.complex128)
        return garbage, np.zeros(1), int(a.shape[1]), np.zeros(1)

    def broken_solve(a, b):
        return np.zeros_like(b)

    monkeypatch.setattr(np.linalg, "lstsq", broken_lstsq)
    monkeypatch.setattr(np.linalg, "solve", broken_solve)
    with pytest.raises(RuntimeError, match="cannot be trusted"):
        _solve_least_squares(matrix, rhs)


def test_square_path_guard_repairs_and_passes_the_healthy_route():
    rng = np.random.default_rng(11)
    matrix = rng.normal(size=(20, 20)) + 1.0j * rng.normal(size=(20, 20))
    matrix += 20.0 * np.eye(20)  # well conditioned on purpose
    sol_ref = rng.normal(size=(20, 1)) + 0.0j
    rhs = matrix @ sol_ref
    sol, method = _solve_square(matrix, rhs)
    assert method == "solve"
    np.testing.assert_allclose(sol, sol_ref, rtol=0, atol=1e-10)


def _small_ops(chief: bool) -> FreeSurface3DOperators:
    """The 64-panel hemisphere cell, radiation operators only."""
    from tensorbem.geometry3d import hemisphere_hull_mesh
    from tensorbem.free_surface_solver3d import assemble_free_surface_operators_3d

    mesh = hemisphere_hull_mesh(2, radius=1.0)
    return assemble_free_surface_operators_3d(
        mesh, 2.0, chief_points="auto" if chief else None
    )


@pytest.mark.parametrize("chief", [True, False])
def test_real_hemisphere_system_is_stationary_where_the_residual_is_not(chief):
    """On the production system the guard must be quiet; the residual is not tiny."""
    ops = _small_ops(chief)
    n = ops.system_matrix.shape[1]
    rng = np.random.default_rng(3)
    q_ext = (rng.normal(size=(n, 1)) + 0.0j).astype(np.complex128)
    rhs_body = ops.single_layer @ q_ext
    if chief:
        rhs = np.vstack((rhs_body, ops.chief_single_layer @ q_ext))
        sol, method = _solve_least_squares(ops.system_matrix, rhs)
        assert method == "lstsq"
        assert least_squares_stationarity(ops.system_matrix, sol, rhs) < 1e-8
        # the CHIEF residual is finite and *larger* than the stationarity measure
        rel_res = float(
            np.linalg.norm(ops.system_matrix @ sol - rhs) / np.linalg.norm(rhs)
        )
        assert rel_res > 1e-8
    else:
        rhs = rhs_body
        sol, method = _solve_square(ops.system_matrix, rhs)
        assert method == "solve"


def test_solve_system_still_matches_published_cell_values():
    """Healthy path is bit-for-bit unchanged: the graded cell is not disturbed."""
    from tensorbem.benchmarks.hulme_hemisphere_case import run_cell

    cell = run_cell(2.0, 2)
    assert cell["A11_dimless"] == pytest.approx(0.255802, abs=5e-7)
    assert cell["A33_dimless"] == pytest.approx(0.358460, abs=5e-7)
    assert cell["B11_dimless"] == pytest.approx(0.313958, abs=5e-7)
    assert cell["B33_dimless"] == pytest.approx(0.123144, abs=5e-7)


def test_no_spurious_warnings_on_a_normal_radiation_solve():
    """The guard must never fire on the healthy path (it would be turned off)."""
    from tensorbem.geometry3d import hemisphere_hull_mesh

    ops = _small_ops(True)
    mesh = hemisphere_hull_mesh(2, radius=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rad = solve_radiation_3d(mesh, 2.0, modes=(0, 2), operators=ops)
    assert rad.relative_residual < 1e-2