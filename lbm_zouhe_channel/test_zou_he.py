"""Unit + physical tests for the Zou-He inlet/outlet reconstruction.

The unit tests assert the *invariants* the reconstruction is supposed to
satisfy (imposed velocity/density reproduced exactly, known populations left
untouched, conservation consistent), not merely that the code runs.  The
physical tests assert the error against the analytic plane-Poiseuille solution
is inside the ledger acceptance band (< 15 %) and that the anchor's one-lattice
inlet-parabola offset is visible as an error increase.

Run with:  python3 -m pytest test_zou_he.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

import channel
from zou_he import (
    OPP,
    W,
    apply_inlet_velocity,
    apply_outlet_pressure,
    equilibrium,
    macroscopic,
    poiseuille_profile,
)


def _random_state(ny=8, seed=0, u=0.05):
    rng = np.random.default_rng(seed)
    f = equilibrium(np.ones((ny, 3)), np.full((ny, 3), u), np.zeros((ny, 3)))
    # add a non-equilibrium part so the reconstruction is not tested on a
    # trivially-consistent state
    f = f + rng.normal(scale=1e-3, size=f.shape)
    return f


def test_inlet_imposes_velocity_exactly():
    ny = 8
    f = _random_state(ny)
    rows = slice(1, ny - 1)
    u_in = poiseuille_profile(ny, 0.08)[:]
    rho = apply_inlet_velocity(f, u_in, rows=rows)
    rho_full, ux, uy = macroscopic(f)
    assert np.allclose(ux[rows, 0], u_in[rows], atol=1e-14)
    # the returned and the recovered density must agree
    assert np.allclose(rho[rows], rho_full[rows, 0], atol=1e-14)


def test_inlet_leaves_known_populations_untouched():
    ny = 8
    f0 = _random_state(ny, seed=1)
    f = f0.copy()
    apply_inlet_velocity(f, poiseuille_profile(ny, 0.08), rows=slice(0, ny))
    for i in (0, 2, 3, 4, 6, 7):
        assert np.allclose(f[:, 0, i], f0[:, 0, i]), f"known population f{i} modified"


def test_inlet_recovers_equilibrium_density():
    """For an exact equilibrium state with u_y=0 the Zou-He density formula is
    analytically exact: rho = (1-u)/(1-u) = 1 for any u."""
    ny, u = 12, 0.12
    f = equilibrium(np.ones((ny, 2)), np.full((ny, 2), u), np.zeros((ny, 2)))
    rho = apply_inlet_velocity(f, np.full(ny, u))
    assert np.allclose(rho, 1.0, atol=1e-13)
    _, ux, _ = macroscopic(f)
    assert np.allclose(ux[:, 0], u, atol=1e-14)


def test_outlet_imposes_density_and_conserves_moments():
    ny = 8
    f = _random_state(ny, seed=2)
    ux_bc = apply_outlet_pressure(f, rho_out=1.0)
    rho, ux, uy = macroscopic(f)
    assert np.allclose(rho[:, -1], 1.0, atol=1e-14)
    assert np.allclose(ux[:, -1], ux_bc, atol=1e-14)
    for i in (0, 1, 2, 4, 5, 8):
        assert np.allclose(f[:, -1, i], _random_state(ny, seed=2)[:, -1, i]), "known pop modified"


def test_outlet_general_density():
    """The reconstruction must be exact for any prescribed rho_out, not just 1."""
    ny = 8
    f = _random_state(ny, seed=3)
    for rho_out in (0.98, 1.0, 1.02):
        g = f.copy()
        apply_outlet_pressure(g, rho_out=rho_out)
        r, _, _ = macroscopic(g)
        assert np.allclose(r[:, -1], rho_out, atol=1e-14)


def test_reconstruction_is_non_negative_in_operating_range():
    ny = 24
    for u in (-0.02, 0.0, 0.05, 0.12):
        f = _random_state(ny, seed=4, u=u)
        apply_inlet_velocity(f, np.full(ny, u))
        apply_outlet_pressure(f, 1.0)
        assert np.all(f > -1e-12), f"negative population at u={u}"


def test_uniform_flow_is_a_fixed_point_of_the_reconstruction():
    """A uniform state at rho=1 is reproduced exactly, so the BC pair does not
    perturb a state that is already the steady solution of the channel."""
    ny, u = 16, 0.05
    f = equilibrium(np.ones((ny, 4)), np.full((ny, 4), u), np.zeros((ny, 4)))
    f_ref = f.copy()
    apply_inlet_velocity(f, np.full(ny, u))
    apply_outlet_pressure(f, 1.0)
    # column 0 (inlet) and column -1 (outlet) unchanged; interior untouched
    assert np.allclose(f[:, 0, :], f_ref[:, 0, :], atol=1e-13)
    assert np.allclose(f[:, -1, :], f_ref[:, -1, :], atol=1e-13)
    assert np.allclose(f[:, 1:-1, :], f_ref[:, 1:-1, :])


def test_profile_geometry():
    ny = 32
    u = poiseuille_profile(ny, 0.08, offset="corrected")
    # the discrete sample points straddle the centreline, so the sampled peak is
    # a hair below the continuous peak (0.07991 for ny=32)
    assert u.max() == pytest.approx(0.08, rel=2e-3)
    assert u.max() < 0.08
    # vanishes halfway into each wall (walls at rows 0 and ny-1)
    assert u[0] == pytest.approx(-0.00542, abs=1e-5)
    assert u[-1] == pytest.approx(u[0], rel=1e-12)
    # symmetric about the channel centre
    assert np.allclose(u[: ny // 2], u[::-1][: ny // 2], atol=1e-12)

    ref = poiseuille_profile(ny, 0.08, offset="reference")
    # the anchor offset makes the profile asymmetric and negative at the
    # bottom wall row: u(0)/U = -21 %
    assert ref[0] < 0 and abs(ref[0]) / 0.08 > 0.15
    assert ref[-1] > 0
    assert not np.allclose(ref[: ny // 2], ref[::-1][: ny // 2], atol=1e-6)


def test_streaming_is_conservative_without_boundaries():
    ny, nx = 6, 8
    f = np.zeros((ny, nx, 9))
    # populations with c_y = 0 never cross the (non-periodic) y walls, so the
    # periodic-in-x streaming must conserve them exactly
    f[:, :, [0, 1, 3]] = np.random.default_rng(0).random((ny, nx, 3))
    out = channel.stream_periodic_x(f)
    assert out.sum() == pytest.approx(f.sum(), rel=1e-14)
    # single-direction shift
    g = np.zeros((1, 4, 9))
    g[0, 1, 1] = 1.0  # population moving in +x
    h = channel.stream(g)
    assert h[0, 2, 1] == 1.0 and h.sum() == 1.0


@pytest.mark.parametrize("variant", ["reference", "corrected"])
def test_channel_reaches_analytic_poiseuille(variant):
    """Physical gate: the resolved channel profile must be within the ledger's
    15 % acceptance band of the analytic parabola (it is in fact far inside)."""
    rows = "all" if variant == "reference" else "interior"
    m = channel.run_zouhe_channel(ny=16, nx=32, u_peak=0.08, nu=0.05,
                                  profile=variant, rows=rows,
                                  max_steps=12000, tol=1e-8, check_every=500)
    assert m["converged"]
    assert m["err_l2_profile"] < 0.15
    assert m["err_shape"] < 0.15
    assert m["Q_rel_drift"] < 1e-6


def test_anchor_offset_is_a_detectable_defect():
    """The anchor's one-lattice inlet offset must produce a strictly larger
    profile error than the corrected parabola at the same resolution."""
    kw = dict(ny=16, nx=32, u_peak=0.08, nu=0.05, max_steps=12000, tol=1e-8,
              check_every=500)
    ref = channel.run_zouhe_channel(profile="reference", rows="all", **kw)
    cor = channel.run_zouhe_channel(profile="corrected", rows="interior", **kw)
    assert ref["err_l2_profile"] > cor["err_l2_profile"]
    assert ref["err_shape"] > cor["err_shape"]


def test_periodic_forced_channel_matches_analytic():
    """The reference-free lattice baseline must reproduce Poiseuille as well."""
    m = channel.run_forced_channel(ny=16, nx=32, u_peak=0.08, nu=0.05,
                                   max_steps=20000, tol=1e-8, check_every=500)
    assert m["converged"]
    assert m["err_l2_profile"] < 0.05
    assert m["Q_rel_drift"] < 1e-6