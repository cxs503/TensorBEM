"""Solver-level gate for the two finite-depth 3-D free-surface kernels.

``tensorbem.finite_depth3d_quadrature`` used to be a verification-only module:
correct after m13811, but unreachable from any solver, so a regression there
could not move a single engineering number.  m14568 adds the source gradient and
the ``kernel=`` switch, and this test holds the two ends together: the *main*
radiation solver is assembled on the eigenseries kernel and on the independent
quadrature kernel, and the 3-DOF added mass and damping matrices must agree.

The quadrature kernel costs one wavenumber integral per (target, source-depth)
pair, so the gate runs on an 8-panel octahedron with ``kernel_nodes=8``; it is
the same case as ``scripts/m14568_quadrature_kernel_solver.py``, which reports
the finer geodesic-sphere rows.

Integration point under test: ``src/tensorbem/finite_depth_solver3d.py``
(``kernel_functions`` -> ``_smooth_remainder``), reached from
``solve_radiation_finite_depth_3d(..., kernel="quadrature")``; the kernel itself
is ``src/tensorbem/finite_depth3d_quadrature.py:
green_gradient_source_finite_depth_3d``.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorbem.finite_depth_solver3d import (
    KERNELS,
    assemble_finite_depth_operators_3d,
    solve_radiation_finite_depth_3d,
)
from tensorbem.geometry3d import TriangleMesh

NU = 0.6
DEPTH = 4.0
MODES = (0, 2, 4)


def _octahedron(half_size: float = 1.0, draught: float = 2.0) -> TriangleMesh:
    a = float(half_size)
    vertices = np.array(
        [
            [a, 0.0, 0.0],
            [-a, 0.0, 0.0],
            [0.0, a, 0.0],
            [0.0, -a, 0.0],
            [0.0, 0.0, a],
            [0.0, 0.0, -a],
        ]
    )
    vertices[:, 2] -= float(draught)
    faces = np.array(
        [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ]
    )
    return TriangleMesh(vertices, faces)


def _frob(got: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


@pytest.fixture(scope="module")
def octahedron():
    return _octahedron()


def test_kernel_registry_names_the_two_kernels():
    assert KERNELS == ("eigenseries", "quadrature")


def test_solver_on_the_quadrature_kernel_reproduces_the_eigenseries_solver(octahedron):
    """A33/B33 from both kernels must agree; a kernel defect moves them by tens of percent."""
    ref = solve_radiation_finite_depth_3d(octahedron, NU, DEPTH, modes=MODES)
    got = solve_radiation_finite_depth_3d(
        octahedron, NU, DEPTH, modes=MODES, kernel="quadrature", kernel_nodes=8
    )
    assert ref.added_mass.shape == got.added_mass.shape == (3, 3)
    # The two kernels are different representations of the same problem, so the
    # residual difference is kernel quadrature error only (measured 6e-5 at
    # nodes=16, 1e-4 at nodes=8 on this body).
    assert _frob(got.added_mass, ref.added_mass) < 1.0e-3
    assert _frob(got.damping, ref.damping) < 1.0e-3
    # The eigenseries defect that would matter (the m13811 34%/320% off-axis
    # error) moves the diagonal by O(0.1-1), so a loose-looking 1e-3 is still
    # ~500x tighter than the failure mode it guards.
    for k in range(3):
        assert abs(got.added_mass[k, k] - ref.added_mass[k, k]) / abs(ref.added_mass[k, k]) < 1.0e-3
        assert abs(got.damping[k, k] - ref.damping[k, k]) / abs(ref.damping[k, k]) < 1.0e-3


def test_operators_carry_the_kernel_label(octahedron):
    # The quadrature label is exercised by the solve above; here only the cheap
    # eigenseries assembly is repeated so the gate does not pay for it twice.
    ops = assemble_finite_depth_operators_3d(octahedron, NU, DEPTH)
    assert ops.kernel == "eigenseries"
    assert np.isfinite(ops.system_matrix).all()


def test_quadrature_kernel_needs_strictly_submerged_panels():
    """A waterline-touching body is rejected with an actionable message."""
    waterline = _octahedron(draught=1.0)  # top vertex sits exactly at z = 0
    with pytest.raises(ValueError, match="strictly submerged"):
        assemble_finite_depth_operators_3d(waterline, NU, DEPTH, kernel="quadrature")
    # The eigenseries kernel handles the same body.
    ops = assemble_finite_depth_operators_3d(waterline, NU, DEPTH)
    assert np.isfinite(ops.system_matrix).all()


@pytest.mark.parametrize("kwargs", [{"kernel": "kelvin"}, {"kernel_nodes": 2}])
def test_kernel_options_are_validated(octahedron, kwargs):
    with pytest.raises(ValueError):
        assemble_finite_depth_operators_3d(octahedron, NU, DEPTH, **kwargs)