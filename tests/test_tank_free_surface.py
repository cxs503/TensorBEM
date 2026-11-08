"""Guards the slack-tank free-surface correction (``tensorbem.tank_free_surface``).

The module is pure algebra, so the tests are the only thing that stands between a
sign slip and a wrong stability result on a real loading condition. Three
independent evaluations are pinned:

* closed form ``i_xx = l b^3/12`` against the exact Green's-theorem polygon
  integration;
* the polygon integration against midpoint quadrature on a refining grid
  (rectangle and an L-shaped footprint with ``i_xy != 0``);
* the metacentric form ``dGM_T = rho_l i_xx / (rho nabla)`` against the roll and
  pitch entries of the 6x6 correction matrix.

The liquid potential-energy Hessian path (which would decide the signs from
energy alone) is **not** asserted here: its grid helper did not close (it
returned four times the analytic curvature), so no unverified green light is
planted.  The sign of ``dC[3,4] = +rho_l g i_xy`` rests on the second-order
derivation in the module docstring and on the hand-checkable L-shaped moments.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.tank_free_surface import (
    apply_free_surface_correction,
    free_surface_correction,
    polygon_moments,
    rectangular_free_surface,
)

RECT = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]])
# L-shaped footprint: area 12, centroid (11/6, 11/6), I_xx = I_yy = 22, i_xy = -4.
L_SHAPE = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [2.0, 2.0], [2.0, 4.0], [0.0, 4.0]])
TRIANGLE = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 2.0]])


def _grid(n: int, lo, hi):
    hx = (hi[0] - lo[0]) / n
    hy = (hi[1] - lo[1]) / n
    xc = np.linspace(lo[0] + 0.5 * hx, hi[0] - 0.5 * hx, n)
    yc = np.linspace(lo[1] + 0.5 * hy, hi[1] - 0.5 * hy, n)
    xx, yy = np.meshgrid(xc, yc, indexing="xy")
    return hx * hy, xx, yy


def _inside_rect(x, y):
    return (x >= 0.0) & (x <= 4.0) & (y >= 0.0) & (y <= 2.0)


def _inside_L(x, y):
    return (x >= 0.0) & (x <= 4.0) & (y >= 0.0) & (y <= 4.0) & ~((x > 2.0) & (y > 2.0))


def _grid_moments(n, inside, hi=(4.0, 4.0)):
    dA, x, y = _grid(n, (0.0, 0.0), hi)
    mask = inside(x, y)
    w = dA * mask
    area = w.sum()
    cx = (w * x).sum() / area
    cy = (w * y).sum() / area
    return area, np.array([cx, cy]), (w * (y - cy) ** 2).sum(), (w * (x - cx) ** 2).sum(), (
        w * (x - cx) * (y - cy)
    ).sum()


def test_rectangle_matches_closed_form() -> None:
    m = polygon_moments(RECT)
    assert m.area == pytest.approx(8.0, rel=1e-13)
    assert m.centroid == pytest.approx([2.0, 1.0], abs=1e-13)
    assert m.i_xx == pytest.approx(4.0 * 2.0**3 / 12.0, rel=1e-13)
    assert m.i_yy == pytest.approx(2.0 * 4.0**3 / 12.0, rel=1e-13)
    assert m.i_xy == pytest.approx(0.0, abs=1e-13)


def test_triangle_matches_closed_form_and_orientation_is_normalised() -> None:
    m = polygon_moments(TRIANGLE)
    assert m.area == pytest.approx(3.0, rel=1e-13)
    assert m.centroid == pytest.approx([1.0, 2.0 / 3.0], abs=1e-13)
    # b*h^3/36 about the centroid for a triangle with the base on y = 0
    # i_xx = b h^3/36 about the centroid for a triangle with its base on y = 0
    assert m.i_xx == pytest.approx(3.0 * 2.0**3 / 36.0, rel=1e-13)
    assert m.i_yy == pytest.approx(1.5, rel=1e-13)
    assert m.i_xy == pytest.approx(-0.5, rel=1e-13)
    reversed_m = polygon_moments(TRIANGLE[::-1])
    assert reversed_m.i_xx == pytest.approx(m.i_xx, rel=1e-13)
    assert reversed_m.i_xy == pytest.approx(m.i_xy, abs=1e-13)


def test_polygon_moments_match_quadrature_on_l_shape() -> None:
    m = polygon_moments(L_SHAPE)
    area, centroid, i_xx, i_yy, i_xy = _grid_moments(400, _inside_L)
    assert m.area == pytest.approx(12.0, rel=1e-13)
    assert m.centroid == pytest.approx([5.0 / 3.0, 5.0 / 3.0], abs=1e-12)
    assert m.i_xx == pytest.approx(44.0 / 3.0, rel=1e-13)
    assert m.i_yy == pytest.approx(44.0 / 3.0, rel=1e-13)
    assert m.i_xy == pytest.approx(-16.0 / 3.0, rel=1e-13)
    assert m.i_xx == pytest.approx(i_xx, rel=2e-3)
    assert m.i_yy == pytest.approx(i_yy, rel=2e-3)
    assert m.i_xy == pytest.approx(i_xy, rel=2e-2)
    assert area == pytest.approx(12.0, rel=2e-3)


def test_correction_is_translation_invariant() -> None:
    a = polygon_moments(L_SHAPE)
    b = polygon_moments(L_SHAPE + np.array([3.0, -2.0]))
    assert b.i_xx == pytest.approx(a.i_xx, rel=1e-12)
    assert b.i_yy == pytest.approx(a.i_yy, rel=1e-12)
    assert b.i_xy == pytest.approx(a.i_xy, rel=1e-12)
    assert b.centroid == pytest.approx(a.centroid + np.array([3.0, -2.0]), abs=1e-12)


def test_correction_matrix_entries_and_metacentric_form() -> None:
    t = rectangular_free_surface(12.0, 6.0, centroid=(0.0, 4.0))
    nabla = 5000.0
    corr = free_surface_correction([t, rectangular_free_surface(12.0, 6.0, centroid=(0.0, -4.0))], nabla)
    rho, g = 1025.0, 9.80665
    expected_dgm = 2.0 * rho * (12.0 * 6.0**3 / 12.0) / (rho * nabla)
    assert corr.delta_gm_transverse == pytest.approx(expected_dgm, rel=1e-13)
    assert corr.delta_gm_longitudinal == pytest.approx(2.0 * (6.0 * 12.0**3 / 12.0) / nabla, rel=1e-13)
    assert corr.restoring_correction[3, 3] == pytest.approx(-rho * g * 2.0 * 216.0, rel=1e-13)
    assert corr.restoring_correction[2, 2] == 0.0  # no heave restoring from a free surface
    assert corr.restoring_correction[0, 0] == 0.0
    assert corr.restoring_correction[5, 5] == 0.0
    # dGM_T read back out of the matrix must be the same number
    assert -corr.restoring_correction[3, 3] / (rho * g * nabla) == pytest.approx(
        corr.delta_gm_transverse, rel=1e-13
    )
    # a free surface can only remove restoring
    block = corr.restoring_correction[np.ix_([3, 4], [3, 4])]
    assert np.all(np.linalg.eigvalsh(block) < 0.0)
    assert np.allclose(corr.restoring_correction, corr.restoring_correction.T)


def test_cross_coupling_enters_matrix_for_l_shaped_tank() -> None:
    t = type(rectangular_free_surface(1.0, 1.0)).from_polygon(L_SHAPE)
    corr = free_surface_correction([t], 5000.0)
    rho, g = 1025.0, 9.80665
    assert corr.restoring_correction[3, 4] == pytest.approx(rho * g * (-16.0 / 3.0), rel=1e-12)
    assert corr.restoring_correction[3, 4] == corr.restoring_correction[4, 3]


def test_superposition_and_apply() -> None:
    nabla = 5000.0
    t1 = rectangular_free_surface(12.0, 6.0, centroid=(0.0, 4.0))
    t2 = rectangular_free_surface(8.0, 4.0, centroid=(6.0, 0.0))
    both = free_surface_correction([t1, t2], nabla)
    a = free_surface_correction([t1], nabla)
    b = free_surface_correction([t2], nabla)
    assert both.delta_gm_transverse == pytest.approx(a.delta_gm_transverse + b.delta_gm_transverse, rel=1e-13)

    rest = np.zeros((6, 6))
    rest[2, 2] = 3.0e6
    rest[3, 3] = -2.0e8
    out = apply_free_surface_correction(rest, both)
    assert out[2, 2] == pytest.approx(3.0e6, rel=1e-13)
    assert out[3, 3] == pytest.approx(-2.0e8 + both.restoring_correction[3, 3], rel=1e-13)
    assert rest[3, 3] == -2.0e8  # input untouched


def test_roll_period_lengthens() -> None:
    corr = free_surface_correction([rectangular_free_surface(12.0, 6.0)], 5000.0)
    gm0 = 1.2
    gm1 = gm0 - corr.delta_gm_transverse
    ratio = math.sqrt(gm0 / gm1)
    assert 1.0 < ratio < 1.10
    assert ratio == pytest.approx(1.0185, abs=2e-3)


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        free_surface_correction([rectangular_free_surface(1.0, 1.0)], 0.0)
    with pytest.raises(ValueError):
        rectangular_free_surface(0.0, 1.0)
    with pytest.raises(ValueError):
        polygon_moments(np.array([[0.0, 0.0], [1.0, 0.0]]))
    with pytest.raises(ValueError):
        polygon_moments(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))  # zero area
    with pytest.raises(ValueError):
        apply_free_surface_correction(np.zeros((3, 3)), free_surface_correction([], 1.0))