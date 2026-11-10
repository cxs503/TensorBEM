"""Tests for :mod:`tensorbem.nk_panel3d` (mission m10003).

The tests are grouped by the property they defend:

* kernel algebra      -- Rankine image kernel, exponential-integral representation
* kernel physics      -- free-surface condition, harmonicity, radiation direction,
                         Kelvin wedge angle, theta-quadrature convergence
* panel geometry      -- Wigley half-body panels, normals, areas
* operator assembly   -- single-layer matrices, symmetry, near-panel quadrature
* solve / outputs     -- boundary-condition residual, elevation field, far field
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem import nk_panel3d as nk

L, B, T = 10.0, 1.0, 0.625
G = 9.80665
RHO = 1025.0


# ---------------------------------------------------------------------------
# theta quadrature
# ---------------------------------------------------------------------------


def test_theta_nodes_cover_interval():
    th, w = nk.theta_nodes(32)
    assert th.shape == w.shape == (32,)
    assert np.all(th > -0.5 * math.pi) and np.all(th < 0.5 * math.pi)
    assert abs(float(np.sum(w)) - math.pi) < 1e-12
    assert np.allclose(th, -th[::-1], atol=1e-12)


def test_theta_nodes_rejects_small_order():
    with pytest.raises(ValueError):
        nk.theta_nodes(4)


# ---------------------------------------------------------------------------
# kernel algebra
# ---------------------------------------------------------------------------


def test_rankine_kernel_matches_closed_form():
    p = np.array([[0.3, 0.2, -0.4]])
    q = np.array([[0.0, 0.0, -0.5]])
    Gk = nk.rankine_kernel(p, q)[0][0, 0]
    r = math.dist(p[0], q[0])
    r1 = math.hypot(0.3, 0.2, -0.4 - 0.5)
    assert Gk == pytest.approx(1.0 / (4 * math.pi * r) + 1.0 / (4 * math.pi * r1), rel=1e-12)


def test_rankine_kernel_mirror_free_surface_condition():
    # the image term makes dG_R/dz vanish on z = 0
    src = np.array([[0.0, 0.0, -0.5]])
    h = 1e-6
    Gp = nk.rankine_kernel(np.array([[0.7, 0.1, h]]), src)[0][0, 0]
    Gm = nk.rankine_kernel(np.array([[0.7, 0.1, -h]]), src)[0][0, 0]
    dGdz = (Gp - Gm) / (2 * h)
    scale = 1.0 / (4 * math.pi * 0.7**2)
    assert abs(dGdz) / scale < 1e-4


def test_rankine_kernel_gradient_matches_finite_difference():
    p = np.array([[0.4, -0.3, -0.7]])
    q = np.array([[0.1, 0.2, -0.4]])
    _, Gx, Gy, Gz = nk.rankine_kernel(p, q)
    h = 1e-6
    for axis, ana in enumerate((Gx[0, 0], Gy[0, 0], Gz[0, 0])):
        ep = p.copy()
        em = p.copy()
        ep[0, axis] += h
        em[0, axis] -= h
        fd = (
            nk.rankine_kernel(ep, q)[0][0, 0] - nk.rankine_kernel(em, q)[0][0, 0]
        ) / (2 * h)
        assert fd == pytest.approx(ana, rel=1e-5, abs=1e-9)


def test_exp_ei_matches_direct_product_in_safe_range():
    beta = np.array([[0.7 - 0.4j, 1.5 + 0.9j, 0.2 - 2.0j]])
    from scipy.special import expi

    direct = np.exp(-beta) * expi(beta)
    assert np.allclose(nk._exp_ei(beta), direct, rtol=1e-12)


def test_exp_ei_asymptotic_branch_matches_direct_product():
    """The |beta| > 60 asymptotic branch must agree with ``exp(-beta) Ei(beta)``."""
    from scipy.special import expi

    for mag in (59.0, 60.5, 120.0):
        beta = np.array([[mag - 4.0j]])
        direct = np.exp(-beta) * expi(beta)
        assert abs(complex(nk._exp_ei(beta)[0, 0] - direct[0, 0])) < 1e-9


def test_wave_kernel_gradient_x_matches_finite_difference():
    kappa = 1.0
    src = np.array([[0.0, 0.0, -0.4]])
    p = np.array([[0.8, 0.3, -0.2]])
    h = 1e-5
    _, Gx, _, _ = nk.nk_wave_kernel(p, src, kappa, 64, "halfplane", -1.0)
    ep = p.copy()
    em = p.copy()
    ep[0, 0] += h
    em[0, 0] -= h
    fd = (
        float(nk.nk_wave_kernel(ep, src, kappa, 64, "halfplane", -1.0)[0].ravel()[0])
        - float(nk.nk_wave_kernel(em, src, kappa, 64, "halfplane", -1.0)[0].ravel()[0])
    ) / (2 * h)
    assert fd == pytest.approx(float(Gx[0, 0]), rel=2e-4, abs=1e-8)


def test_wave_kernel_gradient_z_matches_finite_difference():
    kappa = 1.0
    src = np.array([[0.0, 0.0, -0.4]])
    p = np.array([[0.8, 0.3, -0.2]])
    h = 1e-5
    _, _, _, Gz = nk.nk_wave_kernel(p, src, kappa, 64, "halfplane", -1.0)
    ep = p.copy()
    em = p.copy()
    ep[0, 2] += h
    em[0, 2] -= h
    fd = (
        float(nk.nk_wave_kernel(ep, src, kappa, 64, "halfplane", -1.0)[0].ravel()[0])
        - float(nk.nk_wave_kernel(em, src, kappa, 64, "halfplane", -1.0)[0].ravel()[0])
    ) / (2 * h)
    assert fd == pytest.approx(float(Gz[0, 0]), rel=2e-4, abs=1e-8)


def test_wave_kernel_rejects_bad_kappa():
    with pytest.raises(ValueError):
        nk.nk_wave_kernel(np.zeros((1, 3)), np.zeros((1, 3)) + 1.0, 0.0)


def test_wave_kernel_rejects_bad_shape():
    with pytest.raises(ValueError):
        nk.nk_wave_kernel(np.zeros((1, 2)), np.zeros((1, 3)) + 1.0, 1.0)


# ---------------------------------------------------------------------------
# kernel physics
# ---------------------------------------------------------------------------


def test_free_surface_condition_is_satisfied():
    for mode in ("none", "halfplane"):
        res = nk.verify_free_surface_condition(1.0, 64, mode, -1.0, h=2e-3)
        assert res["max_relative_residual"] < 1e-3


def test_wave_kernel_is_harmonic():
    res = nk.verify_laplace(1.0, 64, "halfplane", -1.0)
    assert abs(res["laplacian"]) < 1e-3 * max(abs(res["value"]), 1.0) + 1e-6


def test_radiation_waves_behind_source():
    res = nk.verify_radiation_direction(1.0, 64, "halfplane", -1.0)
    assert res["ratio_down_over_up"] > 1.5


def test_radiation_mirror_choice_is_unphysical():
    res = nk.verify_radiation_direction(1.0, 64, "halfplane", +1.0)
    assert res["ratio_down_over_up"] < 0.7


def test_radiation_rejects_unknown_mode():
    with pytest.raises(ValueError):
        nk.nk_wave_kernel(
            np.array([[1.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, -0.2]]),
            1.0,
            32,
            "sideways",
            -1.0,
        )


def test_kelvin_wake_extent_is_finite():
    res = nk.kelvin_wake_extent(1.0, 64, "halfplane", -1.0, span=12.0, n=81)
    assert math.isfinite(res["mean_wedge_extent_deg"])
    assert res["theoretical_deg"] == pytest.approx(math.degrees(math.asin(1 / 3)), rel=1e-9)


def test_kelvin_wedge_gap_is_an_open_limitation():
    """Regression guard on a *known* limitation (mission m10003).

    The far field of the present kernel does not collapse onto the classical
    Kelvin caustic: the measured wake extent (``~40-75 deg``) is far wider than
    ``19.47 deg``.  This test pins the gap so that a future fix cannot pass
    unnoticed -- when the wedge is reproduced the assertion must be inverted.
    """
    res = nk.kelvin_wake_extent(1.0, 64, "halfplane", -1.0, span=12.0, n=81,
                                threshold=0.1)
    assert res["mean_wedge_extent_deg"] > 1.5 * res["theoretical_deg"]


def test_theta_quadrature_converges():
    src = np.array([[0.0, 0.0, -0.3]])
    p = np.array([[1.1, 0.4, -0.2]])
    vals = [
        float(nk.nk_wave_kernel(p, src, 1.0, nt, "halfplane", -1.0)[0].ravel()[0])
        for nt in (32, 64, 128)
    ]
    assert abs(vals[1] - vals[2]) / abs(vals[2]) < 1e-5
    assert abs(vals[0] - vals[2]) / abs(vals[2]) < 5e-3


# ---------------------------------------------------------------------------
# panel geometry
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mesh():
    return nk.wigley_half_panels(L, B, T, 16, 8)


def test_wigley_panel_count(mesh):
    assert mesh.n_panels == 16 * 8
    assert mesh.centroids.shape == (mesh.n_panels, 3)


def test_wigley_panel_normals_are_unit(mesh):
    assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0)


def test_wigley_panel_normals_point_outwards(mesh):
    assert np.all(mesh.normals[:, 1] > 0.0)


def test_wigley_panel_centroids_lie_on_surface(mesh):
    x = mesh.centroids[:, 0]
    y = mesh.centroids[:, 1]
    z = mesh.centroids[:, 2]
    # panel centroid of a bilinear quad on a curved surface sits slightly inside
    y_surf = 0.5 * B * (1 - (2 * x / L) ** 2) * (1 - (z / T) ** 2)
    # flat facets on a curved surface: the bilinear centroid stays within a
    # fraction of the local half-breadth
    assert np.all(np.abs(y - y_surf) < 0.06 * y_surf + 1e-3 * B)


def test_wigley_panel_area_converges(mesh):
    def integrand(x, z):
        fx = -(4.0 * B / L**2) * x * (1.0 - (z / T) ** 2)
        fz = -(B / T**2) * (1.0 - (2.0 * x / L) ** 2) * z
        return np.sqrt(1.0 + fx * fx + fz * fz)

    n = 400
    xs = (np.arange(n) + 0.5) / n * L - 0.5 * L
    zs = (np.arange(n) + 0.5) / n * T - T
    XX, ZZ = np.meshgrid(xs, zs, indexing="ij")
    area = float(np.sum(integrand(XX, ZZ)) * (L / n) * (T / n))
    # flat facet approximation of a curved surface: < 1 % on a 16x8 half body
    assert abs(mesh.total_area - area) / area < 0.01


def test_wigley_panels_reject_bad_input():
    with pytest.raises(ValueError):
        nk.wigley_half_panels(-1.0, B, T, 8, 4)
    with pytest.raises(ValueError):
        nk.wigley_half_panels(L, B, T, 2, 4)


def test_centreplane_panel_area():
    m = nk.centreplane_panels(L, B, T, 10, 5)
    assert m.total_area == pytest.approx(L * T, rel=1e-12)


# ---------------------------------------------------------------------------
# operator assembly
# ---------------------------------------------------------------------------


def test_rankine_influence_potential_is_symmetric(mesh):
    S, _ = nk.rankine_influences(mesh, mirror=True, near_factor=1.5)
    cen = mesh.centroids
    dist = np.linalg.norm(cen[:, None, :] - cen[None, :, :], axis=2)
    far = dist > 3.0 * np.sqrt(mesh.areas).max()
    # S[i, j] = G_ij * area_j, so the *kernel* S_ij / area_j is the symmetric one
    A = mesh.areas
    asym = np.abs(S / A[None, :] - (S / A[None, :]).T)[far] / np.maximum(
        np.abs(S / A[None, :])[far], 1e-30
    )
    assert np.all(asym < 0.02)


def test_rankine_influence_diagonal_positive(mesh):
    S, _ = nk.rankine_influences(mesh, mirror=True, near_factor=1.5)
    diag = np.diag(S)
    assert np.all(diag > 0.0)
    assert np.all(np.isfinite(S))


def test_rankine_influence_normal_derivative_finite(mesh):
    _, K = nk.rankine_influences(mesh, mirror=True, near_factor=1.5, nsub=2)
    assert np.all(np.isfinite(K))
    assert np.allclose(np.diag(K), 0.0)


# ---------------------------------------------------------------------------
# solve and output quantities
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_solution():
    m = nk.wigley_half_panels(L, B, T, 8, 4)
    U = 0.3 * math.sqrt(G * L)
    return m, nk.solve_nk_panel(
        m,
        U,
        rho=RHO,
        g=G,
        n_theta=32,
        radiation="halfplane",
        sign=-1.0,
        s_ref=L * T,
        L_ref=L,
    )


def test_solver_boundary_condition_residual(small_solution):
    _, res = small_solution
    assert res.bc_residual < 1e-8


def test_solver_returns_finite_fields(small_solution):
    _, res = small_solution
    assert np.all(np.isfinite(res.sigma))
    assert np.all(np.isfinite(res.phi_hull))
    assert np.all(np.isfinite(res.dphidx_hull))
    assert math.isfinite(res.extra["condition_number"])


def test_solver_rejects_nonpositive_speed(mesh):
    with pytest.raises(ValueError):
        nk.solve_nk_panel(mesh, 0.0)


def test_solver_froude_number(small_solution):
    _, res = small_solution
    assert res.froude == pytest.approx(0.3, rel=1e-12)


def test_free_surface_elevation_shape(small_solution):
    m, res = small_solution
    U = res.speed
    pts = np.column_stack([np.linspace(-L, 0.5 * L, 7), np.zeros(7)])
    zeta = nk.free_surface_elevation(pts, m, res.sigma, U, g=G, n_theta=32,
                                     radiation="halfplane", sign=-1.0)
    assert zeta.shape == (7,)
    assert np.all(np.isfinite(zeta))


def test_free_surface_elevation_rejects_bad_shape(small_solution):
    m, res = small_solution
    with pytest.raises(ValueError):
        nk.free_surface_elevation(np.zeros((3, 3)), m, res.sigma, res.speed)


def test_kochin_amplitude_matches_direct_sum():
    kappa = 1.0
    mesh = nk.centreplane_panels(L, B, T, 4, 2)
    sigma = np.linspace(1.0, 2.0, mesh.n_panels)
    th = np.array([0.0, 0.3])
    H = nk.kochin_amplitude(sigma, mesh, kappa, th, mirror=False)
    manual = []
    for t in th:
        c, s = math.cos(t), math.sin(t)
        sec2 = 1.0 / c**2
        acc = 0.0
        for j in range(mesh.n_panels):
            x, y, z = mesh.centroids[j]
            acc += sigma[j] * mesh.areas[j] * np.exp(
                kappa * sec2 * (z + 1j * (x * c + y * s))
            )
        manual.append(acc)
    assert np.allclose(H, np.array(manual))


def test_far_field_scales_with_prefactor():
    mesh = nk.centreplane_panels(L, B, T, 6, 3)
    sigma = np.linspace(-1.0, 1.0, mesh.n_panels)
    a = nk.wave_resistance_far(sigma, mesh, 1.0, n_theta=181, mirror=False, prefactor=1.0)[0]
    b = nk.wave_resistance_far(sigma, mesh, 1.0, n_theta=181, mirror=False, prefactor=2.0)[0]
    assert b == pytest.approx(2.0 * a, rel=1e-12)


def test_thin_ship_reference_is_finite():
    U = 0.3 * math.sqrt(G * L)
    ts = nk.thin_ship_reference(L, B, T, U, nx=20, nz=10)
    assert math.isfinite(ts["rw_far"])
    assert np.all(np.isfinite(ts["sigma"]))
    assert ts["mesh"].total_area == pytest.approx(L * T, rel=1e-12)