"""Attribution of the finite-depth far/near damping offset (verification debt).

``docs/finite_depth_radiation.md`` §5 (limitation 1) left the ~1 % gap between
the far-field damping ``B_far = 2*rho*g*c_g*|eta_a|^2`` and the near-field
surface-pressure damping ``B_near = omega*rho*Im(force)`` unattributed.  The
benchmark :mod:`tensorbem.benchmarks.finite_depth_energy_identity` shows the
offset is a first-order panel discretisation error that tends to zero, so the
far-field constant is correct.  These tests pin that attribution down so a
future change that introduces a *systematic* factor (a non-zero limiting
offset) is caught.

The four asserted signatures are:

1. the offset ``B_far/B_near - 1`` halves (first order) as the panel count
   doubles;
2. the offset is bounded (``< 2 %`` at 64 panels, ``< 1 %`` at 128);
3. the finite-depth and deep-water solvers bracket the truth from opposite
   sides (opposite signs) - a hidden constant in either formula would break
   this;
4. the offset is depth-independent to within the discretisation noise.
"""

import math

import pytest

from tensorbem.benchmarks.finite_depth_energy_identity import (
    deep_water_ratio,
    finite_depth_ratio,
)

RADIUS = 1.0
SUBMERGENCE = 2.0
KA = 0.6
DENSITY = 1025.0
REF_DEPTH = 5.0


@pytest.fixture(scope="module")
def ladder():
    """Finite-depth refinement ladder at the reference depth (heave)."""
    return {
        n: finite_depth_ratio(REF_DEPTH, n, "heave")
        for n in (32, 64, 128)
    }


@pytest.fixture(scope="module")
def deep():
    """Deep-water ratio at the same meshes (heave)."""
    return {n: deep_water_ratio(n, "heave") for n in (64, 128)}


def test_offset_halves_with_refinement(ladder):
    """First order in the panel size: the offset halves for each doubling."""
    off = {n: abs(c["offset"]) for n, c in ladder.items()}
    assert off[64] < 0.6 * off[32], (
        f"offset did not halve 32->64: {off[32]:.5f} -> {off[64]:.5f}"
    )
    assert off[128] < 0.6 * off[64], (
        f"offset did not halve 64->128: {off[64]:.5f} -> {off[128]:.5f}"
    )


def test_offset_is_bounded(ladder):
    """The residual is a discretisation error, not a systematic factor."""
    assert abs(ladder[64]["offset"]) < 2.0e-2
    assert abs(ladder[128]["offset"]) < 1.0e-2


def test_bracketing_from_opposite_sides(ladder, deep):
    """Finite depth and deep water approach 1 from opposite sides.

    The two solvers use different (equivalent) far-field formulae; a correct
    pair brackets the truth.  A hidden constant in either formula would make
    the two offsets share a sign or fail to converge together.
    """
    fd = ladder[64]["offset"]
    dw = deep[64]["offset"]
    assert fd * dw < 0.0, (
        f"expected opposite signs, got finite-depth {fd:+.5f}, "
        f"deep-water {dw:+.5f}"
    )
    # and the magnitudes are of the same order (both O(h) discretisation)
    assert 0.3 < abs(fd) / abs(dw) < 3.0


def test_depth_independence(ladder):
    """The offset is a property of the discretisation, not of the depth."""
    ref = ladder[64]["offset"]
    for depth in (3.2, 50.0):
        off = finite_depth_ratio(depth, 64, "heave")["offset"]
        assert abs(off - ref) < 2.0e-3, (
            f"offset at h/a={depth} ({off:+.5f}) differs from h/a={REF_DEPTH} "
            f"({ref:+.5f}) by more than the discretisation noise"
        )


def test_far_field_is_positive_and_finite(ladder):
    for case in ladder.values():
        assert case["B_far"] > 0.0 and math.isfinite(case["B_far"])
        assert case["residual"] < 1.0e-10
