"""Off-axis deep-water limit: eigenseries vs quadrature split (m6304).

The 3-D finite-depth kernel ships two independent implementations:

* ``tensorbem.finite_depth3d.green_function_3d`` -- John (1950) eigenseries.
* ``tensorbem.finite_depth3d_quadrature.green_function_finite_depth_3d`` --
  wavenumber-integral (kernel-subtraction) form.

``docs/finite_depth3d_offaxis_diagnosis.md`` (m2930) recorded that the
*quadrature* form fails the off-axis deep-water cross-check (34 % / 320 %)
and that the residual is systematic, not a resolution artefact.  This module
adds the missing half of the picture: the *eigenseries* converges to the
deep-water Kelvin kernel at every off-axis point as ``h -> inf`` (the
eigenseries is singular only on the source axis ``R = 0``, where
``K_0(mu_n R)`` diverges -- a known representation limit, not a defect).

The gate below pins the positive result (eigenseries off-axis convergence)
so the two implementations cannot silently swap roles, and records the
quadrature off-axis defect as a documented, measured open item.
"""
import numpy as np
import pytest

from tensorbem.finite_depth3d import green_function_3d as eigenseries
from tensorbem.finite_depth3d_quadrature import (
    green_function_finite_depth_3d as quadrature,
)
from tensorbem.free_surface3d import green_function_3d as deep_water

SOURCE = np.array([[0.0, 0.0, -1.0]])
NU = 0.6
# Off-axis targets only: the eigenseries carries K_0(mu_n R) and is singular
# on the source axis R = 0, so on-axis points are out of scope for this gate.
OFFAXIS = np.array([[0.5, 0.0, -0.5], [2.0, 1.0, -1.5]])


def _rel_err(value, ref):
    return np.abs(value - ref) / np.abs(ref)


def test_eigenseries_offaxis_converges_to_deep_water():
    ref = deep_water(OFFAXIS, SOURCE, NU)
    prev = None
    for depth in (8.0, 32.0, 128.0):
        err = _rel_err(eigenseries(OFFAXIS, SOURCE, NU, depth), ref)
        assert np.all(err < 1e-2), f"h={depth}: {err}"
        if prev is not None:
            assert np.all(err <= prev + 1e-12), f"non-monotone at h={depth}"
        prev = err
    assert np.all(prev < 1e-3), f"final off-axis error {prev} not < 1e-3"


def test_quadrature_offaxis_defect_is_closed():
    """The quadrature form now matches both references off-axis (m13811).

    History: this test used to pin the *presence* of a 34 % / 320 % off-axis
    error in the quadrature form.  Root cause was a whole-array coordinate
    clamp in ``_green`` (``np.minimum(tgt, 0.0)`` put every target on
    ``x = y = 0``, i.e. on the source axis), not the wavenumber integral.
    ``docs/finite_depth3d_quadrature_offaxis_fix.md``.
    """
    ref_eig = eigenseries(OFFAXIS, SOURCE, NU, 64.0)
    ref_dw = deep_water(OFFAXIS, SOURCE, NU)
    quad = quadrature(OFFAXIS, SOURCE, NU, 64.0)
    # Independent implementation agreement is roundoff-level.
    assert np.all(_rel_err(quad, ref_eig) < 1e-8), _rel_err(quad, ref_eig)
    # Deep-water limit is reached at h = 64.
    assert np.all(_rel_err(quad, ref_dw) < 1e-4), _rel_err(quad, ref_dw)


def test_offaxis_targets_are_not_projected_onto_the_source_axis():
    """Regression gate for the m13811 clamp bug itself.

    The pre-fix kernel silently evaluated every pair at ``R = 0``, so the
    off-axis value collapsed onto the on-axis value.  That is a cheap,
    implementation-independent signature: assert the two differ, and differ by
    more than the (small) roundoff-level agreement above.
    """
    offaxis = np.array([[0.5, 0.0, -0.5]])
    on_axis = np.array([[0.0, 0.0, -0.5]])
    value_off = np.ravel(quadrature(offaxis, SOURCE, NU, 20.0))[0]
    value_axis = np.ravel(quadrature(on_axis, SOURCE, NU, 20.0))[0]
    assert abs(value_off - value_axis) > 1e-3 * abs(value_axis)
