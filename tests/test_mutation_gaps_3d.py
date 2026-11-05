"""Back-filled gates for the mutation blind spots of the 3-D kernels.

A mutation sweep of the three 3-D modules (``kernels3d``, ``free_surface3d``,
``finite_depth3d``; summary in ``docs/m3328/mutation_summary.md``, raw dumps
archived out of git under ``artifacts/m3328/``
produced survivors -- defects that every mapped test let through.  This file
turns the *observable* survivors into gates.  Each test names the mutant it is
supposed to kill, so the file doubles as documentation of what was blind.

Three distinct failure modes were found, all of the same shape: **the test
geometry was too symmetric to see the defect**.

1. ``kernels3d:44`` -- ``double_layer_kernel_3d`` divided by ``r2/sqrt(r2)``
   instead of ``r2*sqrt(r2)``, i.e. ``1/r`` in place of ``1/r**3``.  Every
   existing assertion probed ``r = 1``, where the two agree exactly.
2. ``kernels3d:180`` -- a sign flip in the barycentric numerator
   ``d1*c22 - d2*c12``.  Every existing panel was axis-aligned, so ``c12 = 0``
   and the flipped term never entered the result.
3. ``finite_depth3d:334`` -- ``y_target - y_source`` perturbed to
   ``y_target + y_source``.  Every existing case put the source at ``y = 0``,
   where the perturbation is a no-op.

Two API-contract gaps are covered as well: ``duffy_quadrature*``'s default
order was never exercised (only explicit ``order=``), and ``depth = 0`` was
never probed so ``depth <= 0`` could be relaxed to ``depth < 0`` unnoticed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.finite_depth3d import depth_modes, green_function_3d as finite_depth_green_3d
from tensorbem.free_surface3d import (
    green_function_3d as deep_green_3d,
    verify_free_surface_bc_3d,
    verify_laplace_3d,
)
from tensorbem.kernels3d import (
    double_layer_kernel_3d,
    duffy_quadrature,
    duffy_quadrature_batch,
    integrate_single_layer_triangle,
    triangle_area,
)

FOUR_PI = 4.0 * math.pi

#: A deliberately skewed triangle in ``z = 0``: ``c12 = e1 . e2 = 1.31 != 0``.
SKEW_TRIANGLE = np.array([[0.0, 0.0, 0.0], [2.0, 0.3, 0.0], [0.4, 1.7, 0.0]])


# ---------------------------------------------------------------------------
# kernels3d:44 -- the double-layer point kernel must decay like 1/r**3
# ---------------------------------------------------------------------------
def test_double_layer_point_kernel_scales_like_inverse_cube():
    """Kills ``swap_binop in double_layer_kernel_3d: r2*sqrt(r2) -> r2/sqrt(r2)``.

    ``dG/dn_y = (x - y).n / (4*pi*|x - y|**3)``.  The pre-existing assertions
    used ``|x - y| = 1``, the unique distance at which ``1/r`` and ``1/r**3``
    coincide; two off-unit distances fix the power.
    """
    source = np.array([0.1, 0.2, 0.3])
    normal = np.array([0.0, 0.0, 1.0])
    for distance in (2.0, 0.5, 1.0):
        target = source + np.array([0.0, 0.0, distance])
        delta = target - source
        expected = float(np.dot(delta, normal)) / (FOUR_PI * distance**3)
        got = double_layer_kernel_3d(target, source, normal)
        assert got == pytest.approx(expected, rel=1.0e-14), f"r = {distance}"

    # On the normal axis delta.n = r, so the kernel is 1/(4*pi*r**2): halving
    # the separation multiplies it by four.  A 1/r kernel (the mutant) would
    # give two -- and so would a 1/r**3 kernel probed at a fixed r, which is
    # exactly why the r = 1 assertions above could not see the defect.
    near = double_layer_kernel_3d(source + np.array([0.0, 0.0, 0.5]), source, normal)
    far = double_layer_kernel_3d(source + np.array([0.0, 0.0, 1.0]), source, normal)
    assert near / far == pytest.approx(4.0, rel=1.0e-12)


# ---------------------------------------------------------------------------
# kernels3d:107 -- the documented default Gauss order is part of the contract
# ---------------------------------------------------------------------------
def test_duffy_quadrature_defaults_and_exactness_on_skewed_panel():
    """Kills ``negate_constant in duffy_quadrature_batch: order = 8 -> -8``.

    Callers (:mod:`tensorbem.solver3d`, :mod:`tensorbem.near_singular`) rely on
    the default order; every pre-existing test passed ``order=`` explicitly.
    The skewed panel also rejects any weight/Gauss-node bookkeeping error, since
    an axis-aligned panel cannot tell ``c12 = 0`` from a sign error.
    """
    area = triangle_area(SKEW_TRIANGLE)
    for points, weights in (
        duffy_quadrature(SKEW_TRIANGLE),
        (
            duffy_quadrature_batch(
                SKEW_TRIANGLE[None, 0], SKEW_TRIANGLE[None, 1], SKEW_TRIANGLE[None, 2]
            )[0][0],
            duffy_quadrature_batch(
                SKEW_TRIANGLE[None, 0], SKEW_TRIANGLE[None, 1], SKEW_TRIANGLE[None, 2]
            )[1][0],
        ),
    ):
        assert points.shape == (64, 3)  # default order 8 -> 8*8 nodes

        # exactness on a skewed panel: moments of the reference triangle
        cx = (SKEW_TRIANGLE[0, 0] + SKEW_TRIANGLE[1, 0] + SKEW_TRIANGLE[2, 0]) / 3.0
        cy = (SKEW_TRIANGLE[0, 1] + SKEW_TRIANGLE[1, 1] + SKEW_TRIANGLE[2, 1]) / 3.0
        assert np.sum(weights) == pytest.approx(area, rel=1.0e-13)
        assert np.sum(weights * points[:, 0]) == pytest.approx(area * cx, rel=1.0e-13)
        assert np.sum(weights * points[:, 1]) == pytest.approx(area * cy, rel=1.0e-13)

        # every node must lie inside the panel
        bary = np.linalg.solve(
            np.array(
                [
                    [SKEW_TRIANGLE[1, 0] - SKEW_TRIANGLE[0, 0], SKEW_TRIANGLE[2, 0] - SKEW_TRIANGLE[0, 0]],
                    [SKEW_TRIANGLE[1, 1] - SKEW_TRIANGLE[0, 1], SKEW_TRIANGLE[2, 1] - SKEW_TRIANGLE[0, 1]],
                ]
            ),
            (points[:, :2] - SKEW_TRIANGLE[0, :2]).T,
        ).T
        assert np.all(bary >= -1.0e-14)
        assert np.all(bary.sum(axis=1) <= 1.0 + 1.0e-14)


# ---------------------------------------------------------------------------
# kernels3d:180 -- in-plane containment on a skewed panel
# ---------------------------------------------------------------------------
def test_in_plane_singular_detection_on_skewed_panel():
    """Kills ``swap_binop in _barycentric_in_plane: d1*c22 - d2*c12 -> +``.

    The singular-panel guard asks whether an in-plane target falls inside the
    panel.  With ``c12 = 0`` (axis-aligned panel) the sign of that term is
    unobservable, so the guard was only ever exercised where it cannot fail.
    """
    inside = SKEW_TRIANGLE.mean(axis=0)  # centroid: strictly interior
    with pytest.raises(ValueError, match="singular"):
        integrate_single_layer_triangle(inside, SKEW_TRIANGLE)

    # a point in the same plane but outside the panel must stay legal
    outside = np.array([-0.6, 0.9, 0.0])
    value = integrate_single_layer_triangle(outside, SKEW_TRIANGLE, order=12)
    coarse = integrate_single_layer_triangle(outside, SKEW_TRIANGLE, order=20)
    assert value == pytest.approx(coarse, rel=1.0e-8)


# ---------------------------------------------------------------------------
# finite_depth3d:240 -- h = 0 is not a fluid domain
# ---------------------------------------------------------------------------
def test_finite_depth_rejects_zero_depth():
    """Kills ``flip_compare in depth_modes: depth <= 0 -> depth < 0``."""
    with pytest.raises(ValueError, match="depth"):
        depth_modes(0.9, 0.0)
    with pytest.raises(ValueError, match="depth"):
        depth_modes(0.9, -1.0)
    with pytest.raises(ValueError, match="depth"):
        finite_depth_green_3d(np.array([1.0, 0.0, -0.5]), np.array([0.0, 0.0, -1.0]), 0.9, 0.0)


# ---------------------------------------------------------------------------
# finite_depth3d:334 -- the kernel depends on (y_target - y_source)
# ---------------------------------------------------------------------------
def test_finite_depth_kernel_with_off_axis_source_recovers_deep_water():
    """Kills ``swap_binop in green_function_3d: dy = tgt - src -> tgt + src``.

    Perturbing the ``y`` difference alone is invisible whenever the source sits
    at ``y = 0``, which is true of every pre-existing case.  An off-axis source
    makes the mutant a *reflected* kernel, which the deep-water cross-check
    (independent Kelvin kernel) rejects at O(1).
    """
    nu = 0.6
    sources = np.array([0.3, 0.7, -2.0])
    targets = np.array([[0.5, -0.4, -1.0], [2.0, 1.1, -0.5], [3.0, -1.6, -2.0]])

    errors = []
    for depth in (10.0, 20.0, 40.0):
        finite = finite_depth_green_3d(targets, sources, nu, depth)
        deep = deep_green_3d(targets, sources, nu)
        errors.append(np.max(np.abs(finite - deep)) / np.max(np.abs(deep)))

    assert errors[-1] < 2.0e-3
    assert errors[-1] < errors[1] < errors[0]


def test_finite_depth_kernel_is_symmetric_in_horizontal_separation():
    """The kernel may depend on ``|x - y|`` only: mirror both points in ``y``.

    A guarding identity rather than a reproduction: it is cheap and it pins the
    geometry convention the previous test's mutant violates.
    """
    nu = 0.8
    depth = 6.0
    target = np.array([1.2, 0.9, -1.4])
    source = np.array([-0.4, 0.35, -2.6])
    base = finite_depth_green_3d(target, source, nu, depth)
    mirrored = finite_depth_green_3d(
        target * np.array([1.0, -1.0, 1.0]), source * np.array([1.0, -1.0, 1.0]), nu, depth
    )
    assert complex(mirrored) == pytest.approx(complex(base), rel=1.0e-13, abs=1.0e-15)


# ---------------------------------------------------------------------------
# free_surface3d -- branch selection of the Kelvin wave block
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "xy",
    [
        (1.5, 2.0),  # reg_a: y <= 4.5, x > 0.35y  (series-in-cone branch excluded)
        (3.0, 4.0),  # reg_a, near the y = 4.5 seam
        (0.05, 1.0),  # cone branch: x <= 0.35y, near-vertical field point
        (1.2, 0.4),  # cone branch, off-axis
        (4.0, 5.0),  # reg_b: y > 4.5
        (30.0, 40.0),  # reg_c: y > 4.5, d >= 26 (multipole branch)
    ],
)
def test_wave_block_branches_satisfy_the_free_surface_condition(xy):
    """Physics gate over every branch of ``_eval_wave_block``.

    ``_eval_wave_block`` selects one of four representations by region; the
    pre-existing gates only sampled points in reg_b/reg_c and the cone, so a
    mask perturbation (``0.35*y`` -> ``0.35/y``, ``y > 4.5`` -> ``y > -4.5``)
    could route a point to a branch that is not valid there without any test
    noticing.  The free-surface condition ``dG/dz - nu*G = 0`` is independent of
    the chosen representation, so it catches a mis-routed point.
    """
    nu = 1.1
    source = np.array([0.2, -0.3, -0.8])
    residual = verify_free_surface_bc_3d(np.array([xy]), source, nu=nu)
    scale = abs(complex(deep_green_3d(np.array([xy[0], xy[1], 0.0]), source, nu)))
    assert abs(complex(residual[0])) < 1.0e-9 * max(scale, 1.0)


@pytest.mark.parametrize("point", [(1.5, 2.0, -1.3), (3.0, 4.0, -0.7), (0.05, 1.0, -1.5)])
def test_wave_block_branches_are_harmonic(point):
    """``grad^2 G = 0`` off the source, in the branches the mask can mis-route."""
    nu = 1.1
    source = np.array([0.2, -0.3, -0.8])
    target = np.array(point, dtype=float)
    residual = verify_laplace_3d(target, source, nu=nu)
    scale = abs(complex(deep_green_3d(target, source, nu))) / 1.0e-8
    assert abs(residual) < 1.0e-2 * scale


def test_wave_value_is_continuous_across_the_cone_boundary():
    """The cone/reg_a split is a representation change, not a physical one."""
    nu = 1.1
    x = 0.35 * 1.5  # exactly on the cone mask boundary for y = 1.5
    targets = np.array([[x - 1.0e-6, 1.5, -0.9], [x + 1.0e-6, 1.5, -0.9]])
    sources = np.array([0.0, 0.0, -1.4])
    values = deep_green_3d(targets, sources, nu)
    assert abs(complex(values[1] - values[0])) < 1.0e-6 * abs(complex(values[0]))