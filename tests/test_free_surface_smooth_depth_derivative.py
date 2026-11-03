"""Finite-difference guard for the *depth* derivative of the deep-water kernel.

``docs/mutation_audit_tensorbem.md`` §5 lists one surviving mutant as
"unasserted path (todo)":

    ``free_surface.py:243`` — ``df_d_depth``: ``2.0 * (elementary_cos +
    frequency * cos_integral)`` mutated to ``2.0 / (...)`` and the whole suite
    stayed green, because that quantity only fed ``dg_d_depth`` and nothing
    compared a derivative *with respect to depth* against a finite difference.

That was a real defect: ``dg_d_depth`` was computed and thrown away, and
``dg_d_source_z`` was written out by hand from the same terms.  The kernel now
routes both through one helper (``tensorbem.free_surface._depth_derivative``),
so the depth-derivative path is part of the returned gradient and is guarded
here directly.

Reference used (no recorded numbers, no second implementation):

    ``A = -(z + source_z)``  ⇒  ``d/dsource_z = -d/dA``

so ``-smooth_depth_derivative`` must equal the central difference of
``green_function(..., include_rankine=False)`` in ``source_z``.  Removing the
Rankine term is what makes the identity exact: ``r = |P - Q|`` depends on
``source_z`` outside ``A``, so only the smooth part is a pure function of
``(X, A)``.
"""

import numpy as np
import pytest

from tensorbem.free_surface import (
    green_function,
    green_gradient,
    smooth_depth_derivative,
)

NU = 1.0
STEP = 1.0e-6
# Field/source pairs strictly below the free surface, with unequal depths so
# that neither dA nor the (z - source_z) term can vanish identically.
PAIRS = [
    ((0.7, -1.3), (0.0, -2.1)),
    ((-1.9, -0.6), (0.4, -3.4)),
    ((2.6, -2.2), (-0.8, -0.9)),
]


@pytest.mark.parametrize("field, source", PAIRS)
def test_smooth_depth_derivative_matches_a_source_depth_finite_difference(
    field, source
):
    x, z = field
    sx, sz = source

    analytic = smooth_depth_derivative(x, z, sx, sz, NU)

    plus = green_function(x, z, sx, sz + STEP, NU, include_rankine=False)
    minus = green_function(x, z, sx, sz - STEP, NU, include_rankine=False)
    finite_difference = (plus - minus) / (2.0 * STEP)

    # d/dsource_z = -d/dA
    assert analytic == pytest.approx(-finite_difference, rel=1.0e-8, abs=1.0e-10)


@pytest.mark.parametrize("field, source", PAIRS)
def test_depth_derivative_is_not_the_trivial_zero_or_real_only(field, source):
    """Guard against a silently dropped term (a zero or real-only result)."""
    x, z = field
    sx, sz = source
    analytic = smooth_depth_derivative(x, z, sx, sz, NU)
    assert abs(analytic) > 1.0e-3
    assert abs(analytic.imag) > 1.0e-6


@pytest.mark.parametrize("field, source", PAIRS)
def test_source_gradient_vertical_component_is_consistent_with_the_smooth_part(
    field, source
):
    """``dg_d_source_z`` must still be the full derivative of ``G``.

    The exposed depth derivative covers the smooth part only; the assembled
    kernel gradient adds the Rankine term.  This pins the assembly, i.e. the
    ``include_rankine=False`` smooth part plus the explicit singular term.
    """
    x, z = field
    sx, sz = source

    full_x, full_z = green_gradient(x, z, sx, sz, NU)
    plus = green_function(x, z, sx, sz + STEP, NU)
    minus = green_function(x, z, sx, sz - STEP, NU)
    assert full_z == pytest.approx(
        (plus - minus) / (2.0 * STEP), rel=1.0e-7, abs=1.0e-9
    )
    # d/dsource_x = -d/dx
    plus_x = green_function(x + STEP, z, sx, sz, NU)
    minus_x = green_function(x - STEP, z, sx, sz, NU)
    assert full_x == pytest.approx(
        -(plus_x - minus_x) / (2.0 * STEP), rel=1.0e-7, abs=1.0e-9
    )


def test_outgoing_false_conjugates_the_depth_derivative():
    x, z, sx, sz = 0.7, -1.3, 0.0, -2.1
    outgoing = smooth_depth_derivative(x, z, sx, sz, NU, outgoing=True)
    incoming = smooth_depth_derivative(x, z, sx, sz, NU, outgoing=False)
    assert incoming == pytest.approx(np.conj(outgoing))