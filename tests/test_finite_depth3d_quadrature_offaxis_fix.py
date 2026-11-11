"""Gate for the m13811 finite-depth 3-D off-axis fix.

The kernel ran its wavenumber integral with ``np.minimum(tgt, 0.0)`` applied to
the *whole* point array, which sets ``x = y = 0``; every pair was therefore
evaluated on the source axis and the recorded "off-axis quadrature defect"
(34 % / 320 % vs deep water, flat in depth) was an input-projection artefact,
not a quadrature error.  This gate pins the three properties that make that
statement checkable:

1. off-axis output differs from the on-axis output (the bug collapsed them);
2. the quadrature agrees with the independent John eigenseries off-axis;
3. it converges to the deep-water Kelvin kernel as ``h -> inf``.

Numbers are measured in the test, and the same computation is published as
``outputs/m13811/fix_verification.json`` by
``scripts/m13811_finite_depth3d_quadrature_offaxis_fix.py``.
"""
import numpy as np
import pytest

from tensorbem.finite_depth3d import green_function_3d as eigenseries
from tensorbem.finite_depth3d_quadrature import (
    green_function_finite_depth_3d as quadrature,
)
from tensorbem.free_surface3d import green_function_3d as deep_water

NU = 0.6
SOURCE = np.array([0.0, 0.0, -1.0])
OFFAXIS = np.array([[0.5, 0.0, -0.5], [2.0, 1.0, -1.5], [1.5, -0.7, -0.2], [5.0, 3.0, -0.9]])


def _rel(value, ref):
    return np.abs(value - ref) / np.abs(ref)


@pytest.mark.parametrize("depth", [4.0, 8.0, 20.0, 60.0, 128.0])
def test_quadrature_matches_eigenseries_offaxis(depth):
    """Two independent representations of the same kernel: roundoff agreement."""
    value = quadrature(OFFAXIS, SOURCE, NU, depth)
    reference = eigenseries(OFFAXIS, SOURCE, NU, depth)
    assert np.all(_rel(value, reference) < 1e-8), _rel(value, reference)


def test_deep_water_limit_offaxis_is_first_order_in_1_over_h():
    """Measured 2.5e-2 -> 1.8e-7 over h = 4 -> 128 (monotone, no plateau)."""
    reference = deep_water(OFFAXIS, SOURCE, NU)
    errors = [_rel(quadrature(OFFAXIS, SOURCE, NU, h), reference).max()
              for h in (4.0, 8.0, 20.0, 60.0, 128.0)]
    assert errors[-1] < 1e-6, errors
    assert all(b <= a for a, b in zip(errors, errors[1:])), errors


def test_horizontal_coordinates_are_not_clamped():
    """Regression gate: the pre-fix bug projected targets onto x = y = 0."""
    offaxis = np.array([[0.75, -0.25, -0.6]])
    on_axis = np.array([[0.0, 0.0, -0.6]])
    value_off = np.ravel(quadrature(offaxis, SOURCE, NU, 8.0))[0]
    value_axis = np.ravel(quadrature(on_axis, SOURCE, NU, 8.0))[0]
    assert abs(value_off - value_axis) / abs(value_axis) > 1e-2
    # and the mirror symmetry of the horizontal plane still holds
    mirrored = np.array([[-0.75, 0.25, -0.6]])
    assert np.isclose(np.ravel(quadrature(mirrored, SOURCE, NU, 8.0))[0], value_off,
                      rtol=0, atol=1e-12)
