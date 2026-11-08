"""Patch the free-surface tests: pin the energy-Hessian path (C8).

Mission m6502.
"""
from pathlib import Path

p = Path("/root/tp-agent/work/m6502/tests/test_tank_free_surface.py")
s = p.read_text()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new)


rep(
    '''* the metacentric form ``dGM_T = rho_l i_xx / (rho nabla)`` against the roll and
  pitch entries of the 6x6 correction matrix.

The liquid potential-energy Hessian path (which would decide the signs from
energy alone) is **not** asserted here: its grid helper did not close (it
returned four times the analytic curvature), so no unverified green light is
planted.  The sign of ``dC[3,4] = +rho_l g i_xy`` rests on the second-order
derivation in the module docstring and on the hand-checkable L-shaped moments.
''',
    '''* the metacentric form ``dGM_T = rho_l i_xx / (rho nabla)`` against the roll and
  pitch entries of the 6x6 correction matrix;
* the **liquid potential-energy Hessian** (mission m6502): the energy of the free
  surface minus that of the liquid frozen to the hull, differenced twice in
  ``(phi, theta)``.  This decides the signs *and* magnitudes of ``dC[3,3]``,
  ``dC[4,4]`` and ``dC[3,4]`` from energy, independently of the derivation in
  the module docstring.  Two traps are reproduced here as regressions, because
  they were the reason the path was abandoned in m6452: writing the
  free-surface plane about the footprint origin instead of its centroid (the
  volume then stops being conserved and the roll curvature comes out exactly
  4x), and forgetting to subtract the frozen baseline (14.5x, the rigid
  weight-at-height energy).
''',
)

s += '''

# --------------------------------------------------------------------------
# Energy-Hessian cross-check (checks.C8 of the verification module).
# --------------------------------------------------------------------------
from tensorbem.benchmarks import tank_free_surface_verification as tfsv  # noqa: E402

RHO = tfsv.DEFAULT_DENSITY
G = tfsv.DEFAULT_GRAVITY
DEPTH = 3.0


@pytest.mark.parametrize("eps", [1.0e-2, 1.0e-3])
def test_energy_hessian_diagonal_matches_rho_g_moments(eps: float) -> None:
    """Second derivative of (free - frozen) energy == -rho g i_xx, -rho g i_yy."""
    for verts in (tfsv.RECT, tfsv.L_SHAPE):
        moments = tfsv.polygon_moments(verts)
        h = tfsv._energy_hessian(verts, 200, DEPTH, eps, RHO, G)
        assert h["roll"] == pytest.approx(-RHO * G * moments.i_xx, rel=1.0e-4)
        assert h["pitch"] == pytest.approx(-RHO * G * moments.i_yy, rel=1.0e-4)
        # a slanted footprint can only remove restoring: both diagonal entries < 0
        assert h["roll"] < 0.0 and h["pitch"] < 0.0
        # volume conservation A*H under the centroidal free-surface plane
        assert h["volume_relerr"] < 1.0e-12


def test_energy_hessian_cross_term_sign_and_magnitude() -> None:
    """d2 U / dphi dtheta == +rho_l g i_xy, sign included (L-shape has i_xy < 0)."""
    moments = tfsv.polygon_moments(tfsv.L_SHAPE)
    assert moments.i_xy < 0.0
    h = tfsv._energy_hessian(tfsv.L_SHAPE, 200, DEPTH, 1.0e-3, RHO, G)
    target = RHO * G * moments.i_xy
    assert h["roll_pitch"] == pytest.approx(target, rel=1.0e-4)
    assert h["roll_pitch"] < 0.0
    # a symmetric footprint has no roll-pitch coupling: the FD value is rounding noise
    rect = tfsv._energy_hessian(tfsv.RECT, 200, DEPTH, 1.0e-3, RHO, G)
    assert abs(rect["roll_pitch"]) < 1.0e-6 * abs(rect["roll"])


def test_energy_hessian_rest_state_is_zero_and_quadratic() -> None:
    """delta U(0, 0) == 0 and delta U(eps, 0) == (1/2) dC[3,3] eps^2 to 1e-6."""
    eps = 1.0e-3
    h = tfsv._energy_hessian(tfsv.RECT, 200, DEPTH, eps, RHO, G)
    assert abs(h["rest"]) == 0.0
    moments = tfsv.polygon_moments(tfsv.RECT)
    u = tfsv._liquid_energies(tfsv.RECT, 200, DEPTH, eps, 0.0, RHO, G)
    expected = -0.5 * RHO * G * moments.i_xx * eps ** 2
    assert u["delta"] == pytest.approx(expected, rel=1.0e-4)


def test_origin_based_plane_reproduces_the_historical_factor_four() -> None:
    """Regression pin on the m6452 trap: plane about the origin -> exactly 4x.

    The two defects that made the energy path look broken are kept executable so
    nobody re-derives them: an origin-based free-surface plane (volume not
    conserved once the tank tilts) gives 4.0000x the analytic roll curvature, and
    dropping the frozen baseline gives 14.5x.
    """
    moments = tfsv.polygon_moments(tfsv.RECT)
    target = -RHO * G * moments.i_xx
    buggy = tfsv._energy_hessian(tfsv.RECT, 200, DEPTH, 1.0e-3, RHO, G, centroidal=False)
    assert buggy["roll"] / target == pytest.approx(4.0, rel=1.0e-3)
    no_ref = tfsv._energy_hessian(tfsv.RECT, 200, DEPTH, 1.0e-3, RHO, G, reference=False)
    assert no_ref["roll"] / target == pytest.approx(14.5, rel=1.0e-3)
    good = tfsv._energy_hessian(tfsv.RECT, 200, DEPTH, 1.0e-3, RHO, G)
    assert good["roll"] / target == pytest.approx(1.0, rel=1.0e-4)


def test_energy_hessian_grid_independence() -> None:
    """Quadrature error only: 100 vs 400 cells agree to the midpoint O(h^2) floor."""
    fine = tfsv._energy_hessian(tfsv.RECT, 400, DEPTH, 1.0e-3, RHO, G)
    coarse = tfsv._energy_hessian(tfsv.RECT, 100, DEPTH, 1.0e-3, RHO, G)
    assert abs(fine["roll"] - coarse["roll"]) / abs(fine["roll"]) < 1.0e-4
'''

p.write_text(s)
print("patched tests")