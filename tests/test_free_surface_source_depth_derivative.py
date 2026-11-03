"""``green_gradient`` vertical source derivative, checked by finite differences.

The 2-D free-surface kernel sees the source depth in two ways: through the
elementary depth sum ``A = -(z + z0)`` and through the Rankine term's difference
``z - z0``.  The analytic vertical source derivative is therefore

    dG/dz0 = -dG/dA - (1/2pi) (z - z0)/r^2

(in the module's convention ``d/dz0 = -d/dA`` because ``A`` decreases as the
source goes deeper, and the Rankine contribution picks up
``d/dz0 log r = -(z - z0)/r^2``).

Until 2026-11-03 the ``dG/dA`` piece was *restated inline* inside the source
derivative, and the local that held it (``df_d_depth`` -> ``dg_d_depth``) was
therefore never read.  The mutation audit recorded the consequence: flipping the
``2.0 *`` factor of ``df_d_depth`` to ``2.0 /`` in
``src/tensorbem/free_surface.py`` survived the entire test suite, because the
mutated value was discarded.  This file guards the derivative itself (so the
mutant now dies on a physics assertion), and the module now composes the source
derivative from ``dg_d_depth`` instead of restating it.

The finite-difference reference is a central difference of
:func:`tensorbem.free_surface.green_function` with step ``1e-6``; it shares no
algebra with :func:`tensorbem.free_surface.green_gradient`.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorbem.free_surface import green_function, green_gradient

NU = 0.6
POINTS = [
    (1.0, -1.0, 0.0, -1.2),  # the point pinned by the mutation audit
    (-2.3, -0.8, 0.4, -2.5),  # off-origin, deeper source
    (0.0, -0.5, -1.0, -0.6),  # shallow source: the depth sum dominates
]


@pytest.mark.parametrize("x, z, source_x, source_z", POINTS)
def test_vertical_source_derivative_matches_a_finite_difference(
    x: float, z: float, source_x: float, source_z: float
) -> None:
    """``dg/ds_z`` equals a central difference of ``G`` at every probe point."""
    step = 1.0e-6
    _, dg_d_source_z = green_gradient(x, z, source_x, source_z, NU)
    plus = green_function(x, z, source_x, source_z + step, NU)
    minus = green_function(x, z, source_x, source_z - step, NU)
    finite_difference = (plus - minus) / (2.0 * step)
    assert complex(dg_d_source_z) == pytest.approx(finite_difference, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("x, z, source_x, source_z", POINTS)
def test_horizontal_source_derivative_matches_a_finite_difference(
    x: float, z: float, source_x: float, source_z: float
) -> None:
    """``dg/ds_x`` equals a central difference of ``G`` (sign included).

    Included because the vertical derivative is now built from the *depth*
    derivative; a sign slip in that composition would be invisible if only the
    vertical component were checked against a differently-derived expression.
    """
    step = 1.0e-6
    dg_d_source_x, _ = green_gradient(x, z, source_x, source_z, NU)
    plus = green_function(x, z, source_x + step, source_z, NU)
    minus = green_function(x, z, source_x - step, source_z, NU)
    finite_difference = (plus - minus) / (2.0 * step)
    assert complex(dg_d_source_x) == pytest.approx(finite_difference, rel=1e-6, abs=1e-9)


def test_depth_derivative_is_not_discarded() -> None:
    """The reported source derivative must react to the depth-integral terms.

    ``green_function(include_rankine=False)`` drops the singular Rankine piece, so
    a central difference of it isolates

        (1/2pi) d/dz0 [ -log r1 - 2 PV ]

whose depth content is exactly ``dg_d_depth``.  Comparing the two paths pins the
composition ``dG/dz0 = -dG/dA + Rankine`` end to end: if the ``A``-derivative
were dropped (or its factor mutated) the two sides disagree by O(1), not by
rounding.
    """
    x, z, source_x, source_z = -1.4, -0.9, 0.3, -1.7
    step = 1.0e-6
    smooth_plus = green_function(x, z, source_x, source_z + step, NU, include_rankine=False)
    smooth_minus = green_function(x, z, source_x, source_z - step, NU, include_rankine=False)
    smooth_derivative = (smooth_plus - smooth_minus) / (2.0 * step)

    full_plus = green_function(x, z, source_x, source_z + step, NU)
    full_minus = green_function(x, z, source_x, source_z - step, NU)
    rankine_derivative = (full_plus - full_minus) / (2.0 * step) - smooth_derivative

    _, dg_d_source_z = green_gradient(x, z, source_x, source_z, NU)
    total = complex(dg_d_source_z)
    assert total == pytest.approx(smooth_derivative + rankine_derivative, rel=1e-6, abs=1e-9)
    assert abs(smooth_derivative) > 0.1 * abs(total)  # the depth path is not negligible here