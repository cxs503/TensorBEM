"""Gates for the source gradient of the quadrature finite-depth 3-D kernel.

The m13811 fix made ``green_function_finite_depth_3d`` agree with the John
eigenseries kernel off axis (``docs/finite_depth3d_quadrature_offaxis_fix.md``).
That kernel was still unusable inside the BEM solver, which needs
``nabla_xi G`` for the double-layer operator; m14568 adds
``green_gradient_source_finite_depth_3d`` (horizontal entries from ``dG/dR`` via
``J_1``, vertical entry from the transposed ``d/dz`` call).  These tests pin it
against two independent references -- the eigenseries gradient and central
differences of the quadrature kernel itself -- and re-pin the value component so
the shared-component refactor cannot silently move ``G``.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorbem.finite_depth3d import (
    green_function_3d as eigenseries_value,
    green_gradient_source_3d as eigenseries_gradient,
)
from tensorbem.finite_depth3d_quadrature import (
    green_function_finite_depth_3d as quadrature_value,
    green_function_finite_depth_z_derivative_3d as quadrature_dz,
    green_gradient_source_finite_depth_3d as quadrature_gradient,
)

NU = 0.6
#: Off-axis targets/sources from the m13811 diagnosis (R = 0.5 .. 5.8).
TARGETS = np.array(
    [[0.5, 0.0, -0.5], [2.0, 1.0, -1.5], [1.5, -0.7, -0.2], [5.0, 3.0, -0.9]]
)
SOURCES = np.array([[0.0, 0.0, -1.0], [1.0, -0.5, -1.5]])
DEPTHS = (4.0, 20.0)


def _rel(got: np.ndarray, ref: np.ndarray) -> float:
    return float(np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1.0e-12)))


def test_value_component_is_unchanged_by_the_component_refactor():
    """``G`` itself must still match the eigenseries after adding ``J_1``."""
    tgt = TARGETS[:, None, :]
    src = SOURCES[None, :, :]
    for depth in DEPTHS:
        got = quadrature_value(tgt, src, NU, depth)
        ref = eigenseries_value(tgt, src, NU, depth)
        assert _rel(got, ref) < 1.0e-8


def test_dz_component_still_matches_finite_differences():
    """The ``order=1`` path shares ``_q_values``; keep it pinned to ``dG/dz``.

    The perturbation is on the *field* point: in finite depth the vertical mode
    shapes depend on ``z`` and ``zeta`` separately, so ``dG/dz`` is not the
    source-depth derivative (that one is obtained by the transposed call, which
    :func:`test_source_gradient_matches_central_differences` pins).
    """
    tgt = TARGETS[:, None, :]
    src = SOURCES[None, :, :]
    step = np.zeros(3)
    step[2] = 1.0e-6
    for depth in DEPTHS:
        analytic = quadrature_dz(tgt, src, NU, depth)
        finite = (
            quadrature_value(tgt + step, src, NU, depth)
            - quadrature_value(tgt - step, src, NU, depth)
        ) / (2.0 * step[2])
        assert _rel(analytic, finite) < 1.0e-4


def test_source_gradient_matches_the_eigenseries_kernel():
    """Two independent representations must agree to the quadrature tolerance."""
    tgt = TARGETS[:, None, :]
    src = SOURCES[None, :, :]
    for depth in DEPTHS:
        got = quadrature_gradient(tgt, src, NU, depth)
        ref = eigenseries_gradient(tgt, src, NU, depth)
        assert got.shape == ref.shape
        assert _rel(got, ref) < 1.0e-8


def test_source_gradient_matches_central_differences():
    """Central differences of the same kernel: independent of the eigenseries."""
    tgt = TARGETS[:, None, :]
    src = SOURCES[None, :, :]
    for depth in DEPTHS:
        got = quadrature_gradient(tgt, src, NU, depth)
        for axis, tol in ((0, 1.0e-6), (1, 1.0e-6), (2, 1.0e-3)):
            step = np.zeros(3)
            step[axis] = 1.0e-6
            finite = (
                quadrature_value(tgt, src + step, NU, depth)
                - quadrature_value(tgt, src - step, NU, depth)
            ) / (2.0 * step[axis])
            # Truncation of the difference formula grows fastest for d/dz at
            # small R, hence the looser vertical tolerance.
            assert _rel(got[..., axis], finite) < tol


def test_horizontal_gradient_vanishes_on_the_source_axis():
    """At ``R = 0`` the horizontal entries are zero by symmetry, not NaN."""
    tgt = np.array([[0.0, 0.0, -0.7]])
    src = np.array([[0.0, 0.0, -1.3]])
    grad = quadrature_gradient(tgt, src, NU, 4.0)
    assert np.all(np.isfinite(grad))
    assert abs(grad[0, 0]) < 1.0e-12
    assert abs(grad[0, 1]) < 1.0e-12
    assert abs(grad[0, 2]) > 1.0e-3


def test_gradient_rejects_points_outside_the_fluid_layer():
    with pytest.raises(ValueError):
        quadrature_gradient(np.array([[0.0, 0.0, 0.5]]), np.array([[0.0, 0.0, -1.0]]), NU, 4.0)
    with pytest.raises(ValueError):
        quadrature_gradient(np.array([[0.0, 0.0, -5.0]]), np.array([[0.0, 0.0, -1.0]]), NU, 4.0)


def test_gradient_is_singular_at_the_source():
    with pytest.raises(ValueError):
        quadrature_gradient(np.array([[0.0, 0.0, -1.0]]), np.array([[0.0, 0.0, -1.0]]), NU, 4.0)