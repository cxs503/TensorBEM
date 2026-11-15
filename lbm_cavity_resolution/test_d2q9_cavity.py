"""Fast regression tests for the independence/correctness of the cavity solver."""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d2q9_cavity import (  # noqa: E402
    GHIA100_U, GHIA100_V, GHIA100_X, GHIA100_Y, CavityConfig, CavitySolver,
    error_at_ghia_points, errors_vs_ghia, observed_order,
)


def test_ghia_tables_sorted_and_consistent():
    for a, b in ((GHIA100_Y, GHIA100_U), (GHIA100_X, GHIA100_V)):
        assert a.shape == b.shape and len(a) == 17
        assert np.all(np.diff(a) > 0), "station abscissae must be increasing"
        assert a[0] == 0.0 and a[-1] == 1.0
    # u = 1 at the lid, v = 0 at both side walls
    assert GHIA100_U[0] == 0.0 and GHIA100_U[-1] == 1.0
    assert GHIA100_V[0] == 0.0 and GHIA100_V[-1] == 0.0
    # Table I and Table II use *different* station lists (a defect in the
    # TensorLBM transcription collapses them onto Table I's stations)
    assert not np.allclose(GHIA100_Y, GHIA100_X)
    # sign structure of Ghia Re=100 Table II: v>0 on the left half, v<0 on right
    assert GHIA100_V[np.argmin(abs(GHIA100_X - 0.2266))] > 0
    assert GHIA100_V[np.argmin(abs(GHIA100_X - 0.8594))] < 0


def test_ladd_momentum_injected_in_lid_direction():
    """One step from rest: the first fluid row must gain +x momentum."""
    for col in ("bgk", "trt", "rlbm"):
        s = CavitySolver(CavityConfig(n=17, re=100.0, u_lid=0.1, collision=col,
                                      max_steps=1))
        s.step()
        _, ux, _ = s.macroscopic()
        row = ux[-2, 1:-1]
        assert np.allclose(row, 0.1 * 6 * (1 / 36) * 2 / 2, atol=1e-12) or \
            np.all(row > 0), f"{col}: lid does not drag fluid in +x: {row}"
        assert np.all(row > 0)


def test_equilibrium_is_fixed_point():
    s = CavitySolver(CavityConfig(n=17, re=100.0, u_lid=1e-12, collision="rlbm"))
    f0 = s.f.copy()
    s.step()
    assert np.abs(s.f - f0).max() < 1e-12


def test_errors_zero_on_reference_profile():
    y = np.linspace(0, 1, 129)
    x = np.linspace(0, 1, 129)
    u = np.interp(y, GHIA100_Y, GHIA100_U)
    v = np.interp(x, GHIA100_X, GHIA100_V)
    e = errors_vs_ghia(y, u, x, v)
    assert e["max_uv"] < 1e-12
    # The station-list metric cannot be exact: the 129-point grid does not
    # align with the Ghia stations, so a piecewise-linear kink is cut once.
    p = error_at_ghia_points(y, u, x, v)
    assert p["max_uv_pts"] < 1e-3
    # a wrong (legacy) Table II must be detected as *not* matching
    e_leg = errors_vs_ghia(y, u, x, v, legacy_v=True)
    assert e_leg["max_v"] > 0.05


def test_observed_order_recovers_synthetic_slope():
    h = [1 / 64, 1 / 96, 1 / 128]
    e = [0.05 * (hi / h[0]) ** 2.0 for hi in h]
    fit = observed_order(h, e)
    assert abs(fit["slope"] - 2.0) < 1e-9
    assert all(abs(p["p"] - 2.0) < 1e-9 for p in fit["pairs"])


def test_short_run_is_physical():
    """n=33, a few thousand steps: velocity inside the box, u_lid scale."""
    s = CavitySolver(CavityConfig(n=33, re=100.0, u_lid=0.1, collision="rlbm",
                                  max_steps=1500, check_every=500))
    s.run()
    _, ux, uy = s.macroscopic()
    assert np.abs(ux).max() <= 0.11 and np.abs(uy).max() <= 0.11
    assert np.abs(ux).max() > 1e-3   # the flow did start
