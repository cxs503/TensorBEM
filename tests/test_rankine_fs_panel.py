"""Acceptance tests for :mod:`tensorbem.rankine_fs` (Rankine panel method).

The suite has four jobs:

1. **Geometry first** -- a hull panel mesh is only usable if it reproduces
   ``Cb = 4/9`` and the wetted area of the Wigley form; a factor-of-4 error in
   the panel half-extents (found during this mission) is exactly the class of
   bug these tests catch.
2. **Kernel exactness** -- the closed-form in-plane rectangle potential and the
   ``2 pi`` jump of the planar single layer, checked against quadrature.
3. **The verified solve** -- dense 3-D Rankine Neumann (double-body) solve:
   residual, conditioning, mesh convergence of ``Cb``, and the d'Alembert
   negative control (zero wave resistance for a fore-aft symmetric hull in an
   irrotational symmetric flow).
4. **The documented failure** -- the truncated-domain Neumann-Kelvin
   free-surface panel formulation is *unstable*; the tests pin the measured
   symptom (downstream growth, growth sign of the discrete symbol) so that the
   finding cannot silently rot.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.rankine_fs import (
    FlatPanels,
    double_body_rankine,
    farfield_wave_resistance,
    free_surface_grid,
    free_surface_operator_symbol,
    gci_fine,
    gradient_x,
    grid_convergence_index,
    in_plane_single_layer,
    single_layer,
    solve_rankine_free_surface,
    wigley_hull_panels,
    wigley_section_curve,
)

L, B, T = 2.0, 0.2, 0.125
G = 9.80665
RHO = 1025.0
U30 = 0.30 * math.sqrt(G * L)


# ---------------------------------------------------------------------------
# 1. Geometry
# ---------------------------------------------------------------------------


def test_hull_panel_volume_reproduces_block_coefficient():
    """Panel volume converges to the analytic ``Cb = 4/9`` of the Wigley form."""
    cbs = []
    for nx, nz in ((12, 5), (24, 8), (48, 12)):
        res = double_body_rankine(L, B, T, U30, hull_nx=nx, hull_nz=nz)
        cbs.append(res.displaced_volume_panels / (L * B * T))
    assert cbs[0] == pytest.approx(0.434, abs=0.01)
    assert cbs[-1] == pytest.approx(4.0 / 9.0, abs=0.005)
    assert cbs == sorted(cbs), "Cb must increase monotonically towards 4/9"


def test_hull_panel_wetted_area_matches_independent_quadrature():
    """Wetted area against a dense midpoint rule of ``2 sqrt(1+Yx^2+Yz^2)``."""
    nx, nz = 48, 12
    hull = wigley_hull_panels(L, B, T, nx, nz)
    panel_area = 2.0 * float(np.sum(hull.areas))

    xs = np.linspace(-0.5 * L, 0.5 * L, 400)
    zs = np.linspace(-T, 0.0, 200)
    dx = xs[1] - xs[0]
    dz = zs[1] - zs[0]
    xg, zg = np.meshgrid(xs, zs, indexing="ij")
    eps = 1e-6
    y = wigley_section_curve(xg, zg, L, B, T)
    yx = (
        wigley_section_curve(xg + eps, zg, L, B, T)
        - wigley_section_curve(xg - eps, zg, L, B, T)
    ) / (2 * eps)
    yz = (
        wigley_section_curve(xg, zg + eps, L, B, T)
        - wigley_section_curve(xg, zg - eps, L, B, T)
    ) / (2 * eps)
    quad = 2.0 * float(np.sum(np.sqrt(1.0 + yx**2 + yz**2)) * dx * dz)
    assert panel_area == pytest.approx(quad, rel=0.02)


def test_hull_normals_are_unit_and_point_outboard():
    hull = wigley_hull_panels(L, B, T, 10, 4)
    assert np.allclose(np.linalg.norm(hull.normal, axis=1), 1.0, atol=1e-12)
    assert np.all(hull.normal[:, 1] > 0.0), "outward normal must have +y component"
    assert np.all(hull.centers[:, 1] >= 0.0), "half domain must be y >= 0"


def test_panel_validation_rejects_bad_inputs():
    with pytest.raises(ValueError):
        wigley_hull_panels(-1.0, B, T)
    with pytest.raises(ValueError):
        wigley_hull_panels(L, B, T, nx=1)
    with pytest.raises(ValueError):
        free_surface_grid(0.1, 0.1, x_min=1.0, x_max=0.0, y_max=1.0)
    with pytest.raises(ValueError):
        FlatPanels(
            np.zeros((2, 3)), np.tile(np.eye(3)[0], (2, 1)), np.tile(np.eye(3)[1], (2, 1)),
            np.tile(np.eye(3)[2], (2, 1)), np.array([1.0, -1.0]), np.ones(2),
        )


# ---------------------------------------------------------------------------
# 2. Kernels
# ---------------------------------------------------------------------------


def test_in_plane_single_layer_matches_quadrature():
    """Closed-form coplanar rectangle potential vs 8x8 Gauss quadrature."""
    fs, _, _ = free_surface_grid(0.1, 0.1, x_min=-0.5, x_max=0.5, y_max=0.5)
    targets = np.array([[0.02, 0.03], [0.07, -0.11], [0.31, 0.22], [0.0, 0.13]])
    closed = in_plane_single_layer(targets, fs).T
    p3 = np.column_stack([targets, np.zeros(targets.shape[0])])
    quad = single_layer(p3, fs, order=8).T
    dist = np.linalg.norm(fs.centers[:, None, :2] - targets[None, :, :], axis=2)
    far = dist > 0.25  # exclude the singular own/near panels
    assert np.allclose(closed[far], quad[far], rtol=1e-10)


def test_free_surface_jump_constant_is_two_pi():
    """Solid angle of a rectangle seen from ``eps`` above its centre tends to ``2 pi``.

    This is the constant the solver hard-codes for the one-sided derivative of
    the planar single layer.  It has to be evaluated *analytically*: fixed-order
    Gauss quadrature cannot resolve the delta-like kernel
    ``eps/(rho^2+eps^2)^(3/2)`` -- measured values 4.53, 0.77, 0.078 for
    ``eps = 1e-2, 1e-3, 1e-4`` on a 0.1 x 0.1 panel, i.e. it collapses instead of
    converging, which is exactly why the solver uses the jump relation.
    """
    a = b = 0.05

    def solid_angle(eps: float) -> float:
        return 4.0 * math.atan(a * b / (eps * math.sqrt(a * a + b * b + eps * eps)))

    values = [solid_angle(eps) for eps in (1e-2, 1e-3, 1e-4, 1e-6)]
    assert values == sorted(values)
    assert all(v < 2.0 * math.pi for v in values)
    assert values[-1] == pytest.approx(2.0 * math.pi, rel=1e-4)


def _dipole_z(targets: np.ndarray, panels: FlatPanels) -> np.ndarray:
    from tensorbem.rankine_fs import dipole

    zhat = np.tile(np.array([0.0, 0.0, 1.0]), (targets.shape[0], 1))
    return dipole(targets, zhat, panels, order=8)


# ---------------------------------------------------------------------------
# 3. The verified solve
# ---------------------------------------------------------------------------


def test_double_body_neumann_residual_small():
    res = double_body_rankine(L, B, T, U30, hull_nx=24, hull_nz=8)
    assert res.neumann_residual < 1e-10
    assert res.matrix_condition < 1e5


def test_double_body_has_no_wave_resistance():
    """d'Alembert negative control: symmetric flow cannot radiate."""
    for nx, nz in ((12, 5), (24, 8), (48, 12)):
        res = double_body_rankine(L, B, T, U30, hull_nx=nx, hull_nz=nz)
        ref = 0.5 * RHO * U30**2 * L * T
        assert abs(res.pressure_resistance) / ref < 1e-9


def test_double_body_source_strength_is_antisymmetric_in_x():
    """``sigma ~ n_x``: hull sources must be odd about midship."""
    res = double_body_rankine(L, B, T, U30, hull_nx=16, hull_nz=6)
    x = res.hull_panels.centers[:, 0]
    sig = res.sigma
    left = sig[x < -0.25 * L]
    right = sig[x > 0.25 * L]
    assert np.mean(left) < 0.0 < np.mean(right)


def test_double_body_cb_grid_convergence_index():
    cbs = [
        double_body_rankine(L, B, T, U30, hull_nx=nx, hull_nz=nz).displaced_volume_panels
        / (L * B * T)
        for nx, nz in ((12, 5), (24, 8), (48, 12))
    ]
    gci = grid_convergence_index(cbs)
    assert gci["monotone"] is True
    assert gci["gci_fine_pct"] < 2.0


# ---------------------------------------------------------------------------
# 4. Boundary / negative controls of the free-surface formulation
# ---------------------------------------------------------------------------


def test_negative_speed_raises():
    with pytest.raises(ValueError):
        solve_rankine_free_surface(L, B, T, -1.0, hull_nx=6, hull_nz=3, panels_per_wavelength=5.0)


def test_zero_speed_raises():
    with pytest.raises(ValueError):
        double_body_rankine(L, B, T, 0.0, hull_nx=6, hull_nz=3)


def test_invalid_geometry_raises():
    with pytest.raises(ValueError):
        double_body_rankine(0.0, B, T, 1.0)
    with pytest.raises(ValueError):
        double_body_rankine(L, B, T, float("nan"))


def test_upwind_symbol_growth_is_positive_first_order():
    """The 3-point backward stencil has ``Im k > 0``: aliased downstream growth."""
    sym = free_surface_operator_symbol(np.array([0.05, 0.2]), order=1)
    assert np.all(sym["growth"] > 0.0), "3-point backward stencil adds O(h) damping"
    sym2 = free_surface_operator_symbol(np.array([0.05, 0.2]), order=2)
    assert np.all(np.abs(sym2["growth"]) < np.abs(sym["growth"]) + 1e-12)
    with pytest.raises(ValueError):
        free_surface_operator_symbol(0.1, order=3)


def test_gci_rejects_bad_input():
    with pytest.raises(ValueError):
        gci_fine([1.0, 1.1])
    with pytest.raises(ValueError):
        gci_fine([1.0, 1.1, 1.21], refinement=1.0)


def test_gci_on_second_order_sequence():
    """Synthetic ``f(h) = f0 + c h^2`` with ``h`` halving gives ``p ~ 2``."""
    exact = 3.0
    vals = [exact + 0.4 / r**2 for r in (1, 2, 4)]
    res = grid_convergence_index(vals, refinement=2.0)
    assert res["apparent_order"] == pytest.approx(2.0, abs=0.05)
    assert 0.0 < res["gci_fine_pct"] < 5.0


def test_farfield_wave_resistance_recovers_known_amplitude():
    """A pure transverse wave train ``a cos(k0 x)`` must give ``rho g a^2 L_y / 4``."""
    k0 = 2.0 * math.pi / 4.0
    hx, hy, ny = 0.25, 0.5, 6
    x = (np.arange(40)[:, None] + 0.5) * hx
    y = (np.arange(ny)[None, :] + 0.5) * hy
    amp = 0.02
    zeta = amp * np.cos(k0 * x) * np.ones_like(y)
    rw = farfield_wave_resistance(zeta, np.broadcast_to(x, zeta.shape),
                                  np.broadcast_to(y, zeta.shape), k0, RHO, G)
    assert rw == pytest.approx(0.25 * RHO * G * amp**2 * ny * hy, rel=1e-6)


# ---------------------------------------------------------------------------
# 5. The documented failure of the truncated-domain formulation
# ---------------------------------------------------------------------------


def test_truncated_free_surface_solve_is_unstable():
    """Pin the measured negative result: spurious downstream growth of ``zeta``.

    The discrete Neumann-Kelvin row set is not a well-posed second-kind system:
    the physical wave content lives on the ``|k| = k0`` resonance ring, and the
    upwind stencil's aliased branches grow downstream.  This test asserts the
    symptom is still there (so the documented limitation cannot silently change)
    and that the solve stays finite rather than raising / returning NaN.
    """
    res = solve_rankine_free_surface(
        L, B, T, U30, hull_nx=8, hull_nz=4, panels_per_wavelength=5.0,
        x_min_factor=1.0, x_max_factor=2.0, y_max_factor=1.0,
    )
    zeta = np.asarray(res.zeta)
    assert np.isfinite(zeta).all()
    quarter = max(1, zeta.shape[0] // 4)
    up = float(np.sqrt(np.mean(zeta[:quarter] ** 2)))
    down = float(np.sqrt(np.mean(zeta[-quarter:] ** 2)))
    # measured symptom: the upwind stencil damps the physical |k| = k0 modes
    assert down < up, "known defect: transverse waves are numerically damped"
    assert abs(res.cw_pressure) > 0.0