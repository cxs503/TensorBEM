"""Verification of the periodic-strip BEM mass-loading dispersion capability.

Every assertion compares the new BEM route with a route that does not share its
code path:

* the Weitz-Keller closed form ``omega^2 = g k tanh(kH)/(1 + mu k tanh(kH))``
  (already pinned independently in ``tensorbem.ice``);
* the two limits of that closed form (deep water, long wave);
* the interior solution of the driven mode, ``cos(kx) cosh(k(z+H))/cosh(kH)``,
  which the BEM never sees as data on the bottom or on the periodic walls;
* panel refinement, where the error must fall towards the closed form.

The negative cases pin input validation and the tie bookkeeping.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem import ice as ice_module
from tensorbem import ice_mass_loading_dispersion as disp

MU = disp.mass_loading_mu(1.0)          # rho_i h_i / rho_w for a 1 m sheet
G = disp.DEFAULT_GRAVITY


# ---------------------------------------------------------------------------
# reference side: the closed form and its limits
# ---------------------------------------------------------------------------
def test_mass_loading_parameter_matches_ice_cover():
    cover = ice_module.IceCover(thickness=1.0)
    assert disp.mass_loading_mu(1.0, ice_density=cover.ice_density,
                                water_density=cover.water_density) == pytest.approx(
        cover.mu, rel=1e-15
    )


@pytest.mark.parametrize("kH", [20.0, 50.0, 100.0])
def test_deep_water_limit_of_the_closed_form(kH: float):
    k = 0.01
    depth = kH / k
    finite = disp.weitz_keller_omega2(k, depth=depth, mu=MU)
    deep = disp.weitz_keller_deep_omega2(k, mu=MU)
    assert finite == pytest.approx(deep, rel=1e-6)


@pytest.mark.parametrize("kH", [1e-3, 1e-2])
def test_long_wave_limit_of_the_closed_form(kH: float):
    k = 0.001
    depth = kH / k
    finite = disp.weitz_keller_omega(k, depth=depth, mu=MU)
    long_wave = disp.long_wave_omega(k, depth=depth)
    assert finite == pytest.approx(long_wave, rel=2e-3)


def test_closed_form_agrees_with_the_ice_module():
    """The new closed form must reproduce ``tensorbem.ice`` exactly."""
    cover = ice_module.IceCover(thickness=0.9, water_depth=120.0)
    for k in (0.02, 0.1, 0.3):
        assert disp.weitz_keller_omega(k, depth=cover.water_depth, mu=cover.mu,
                                       gravity=cover.gravity) == pytest.approx(
            ice_module.mass_loading_omega(k, cover), rel=1e-14
        )


def test_robin_inversion_round_trips():
    k, depth = 0.12, 30.0
    lam = disp.weitz_keller_depth_factor(k, depth)
    omega2 = disp.mass_loading_omega2_from_depth_factor(lam, mu=MU)
    # invert by hand: Lam = omega^2 / (g - mu omega^2)
    assert omega2 / (G - MU * omega2) == pytest.approx(lam, rel=1e-12)


def test_cutoff_is_the_short_wave_limit():
    cutoff = disp.mass_loading_cutoff_omega(MU)
    assert cutoff == pytest.approx(math.sqrt(G / MU), rel=1e-15)
    # omega grows monotonically towards the cutoff with k, never past it
    omegas = [disp.weitz_keller_omega(k, depth=1e6, mu=MU) for k in (0.1, 1.0, 10.0)]
    assert all(b > a for a, b in zip(omegas[:-1], omegas[1:]))
    assert all(w < cutoff for w in omegas)


# ---------------------------------------------------------------------------
# mesh / tie bookkeeping
# ---------------------------------------------------------------------------
def test_mesh_is_one_period_with_outward_normals():
    k, depth = 0.5, 10.0
    mesh = disp.build_periodic_strip(k, depth, panels_per_wavelength=12)
    assert mesh.period == pytest.approx(2.0 * math.pi / k, rel=1e-14)
    fs = mesh.index["free_surface"]
    bot = mesh.index["bottom"]
    left = mesh.index["left"]
    right = mesh.index["right"]
    assert np.allclose(mesh.normals[fs], [0.0, 1.0])
    assert np.allclose(mesh.normals[bot], [0.0, -1.0])
    assert np.allclose(mesh.normals[left], [-1.0, 0.0])
    assert np.allclose(mesh.normals[right], [1.0, 0.0])
    # every panel lies on the boundary of the cell
    centres = mesh.centres
    assert np.all(np.isclose(centres[fs, 1], 0.0))
    assert np.all(np.isclose(centres[bot, 1], -depth))
    # the free surface spans exactly one period
    assert float(np.sum(mesh.lengths[fs])) == pytest.approx(mesh.period, rel=1e-12)


def test_tie_operators_cover_every_panel_once():
    mesh = disp.build_periodic_strip(0.5, 10.0, panels_per_wavelength=8)
    p, q = disp.tie_operators(mesh)
    assert p.shape == q.shape
    assert p.shape[0] == mesh.n_panels
    assert p.shape[1] == mesh.n_panels
    # the free surface is Dirichlet data: no unknown potential there
    assert np.all(p[mesh.index["free_surface"], :] == 0.0)
    # the bottom is rigid: no unknown flux there
    assert np.all(q[mesh.index["bottom"], :] == 0.0)
    # periodic image: same potential, opposite flux
    left, right = mesh.index["left"], mesh.index["right"]
    assert np.allclose(p[left, :], p[right, :])
    assert np.allclose(q[left, :], -q[right, :])


def test_tie_operators_reject_mismatched_walls():
    mesh = disp.build_periodic_strip(0.5, 10.0, panels_per_wavelength=8)
    broken = disp.StripMesh(
        period=mesh.period, depth=mesh.depth, starts=mesh.starts, ends=mesh.ends,
        normals=mesh.normals, lengths=mesh.lengths,
        index={
            "free_surface": mesh.index["free_surface"],
            "bottom": mesh.index["bottom"],
            "left": mesh.index["left"],
            "right": mesh.index["right"][:-1],
        },
    )
    with pytest.raises(ValueError):
        disp.tie_operators(broken)


# ---------------------------------------------------------------------------
# BEM side: depth factor and frequency
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k,depth,tol", [(0.1, 200.0, 0.02), (0.4, 200.0, 0.02),
                                         (0.1, 2.0, 0.03), (0.2, 2.0, 0.03)])
def test_depth_factor_reproduces_k_tanh(k: float, depth: float, tol: float):
    solve = disp.strip_bem_depth_factor(k, depth, panels_per_wavelength=32)
    assert solve.depth_factor_rq / solve.exact_depth_factor == pytest.approx(
        1.0, abs=tol
    )
    assert solve.residual < 1e-10


def test_dtn_route_agrees_with_the_driven_route():
    for k, depth in ((0.1, 200.0), (0.1, 2.0)):
        driven = disp.strip_bem_depth_factor(k, depth, panels_per_wavelength=16)
        dtn = disp.dtn_depth_factor(k, depth, panels_per_wavelength=16)
        assert dtn == pytest.approx(driven.depth_factor_rq, rel=1e-10)


def test_boundary_values_off_the_free_surface_are_recovered():
    """The bottom and wall potentials were not prescribed: recovering the exact
    mode there is a genuine check of the interior solution."""
    solve = disp.strip_bem_depth_factor(0.1, 200.0, panels_per_wavelength=32)
    assert solve.bv_error < 0.02


def test_interior_field_matches_the_exact_mode():
    field = disp.strip_bem_field(0.1, 200.0, panels_per_wavelength=32, nx=21, nz=11)
    assert field.relative_error < 0.03
    assert field.points.shape[0] == 20 * 11


def test_bem_frequency_matches_the_closed_form():
    for k, depth in ((0.05, 200.0), (0.2, 2.0)):
        row = disp.dispersion_row(k, depth=depth, ice_thickness=1.0,
                                 panels_per_wavelength=32)
        assert abs(row["omega_rel_error"]) < 0.05
        assert row["omega_got"] == pytest.approx(row["omega_ref"], rel=0.05)


def test_refinement_reduces_the_depth_factor_error():
    rows = disp.convergence_study(0.1, depth=200.0, levels=(8, 16, 32))
    errors = [abs(r["rel_error"]) for r in rows]
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 0.01


def test_higher_modes_of_the_same_cell():
    sweep = disp.strip_bem_mode_sweep(0.1, 200.0, n_modes=3, panels_per_wavelength=32)
    assert [int(r["mode"]) for r in sweep] == [1, 2, 3]
    for row in sweep:
        assert row["wavenumber"] == pytest.approx(int(row["mode"]) * 0.1)
        assert abs(row["rel_error"]) < 0.03


def test_ice_module_callers_use_the_bem():
    """Integration: ``tensorbem.ice`` exposes the BEM frequency."""
    cover = ice_module.IceCover(thickness=1.0, water_depth=200.0)
    k = 0.1
    from_ice = ice_module.mass_loading_omega_bem(k, cover, panels_per_wavelength=16)
    from_module = disp.strip_bem_mass_loading_omega(
        k, depth=cover.water_depth, mu=cover.mu, gravity=cover.gravity,
        panels_per_wavelength=16,
    )
    assert from_ice == pytest.approx(from_module, rel=1e-15)
    closed_form = ice_module.mass_loading_omega(k, cover)
    assert from_ice == pytest.approx(closed_form, rel=0.05)
    # the DtN route through the same caller
    dtn = ice_module.mass_loading_omega_bem(
        k, cover, panels_per_wavelength=16, route="dtn"
    )
    assert dtn == pytest.approx(from_ice, rel=1e-10)


def test_dispersion_row_schema():
    row = disp.dispersion_row(0.1, depth=20.0, ice_thickness=1.0,
                             panels_per_wavelength=12)
    for key in ("k", "kH", "mu", "omega2_ref", "omega2_got", "omega_ref",
                "omega_got", "omega_rel_error", "depth_factor_bem",
                "depth_factor_ref", "cutoff_omega"):
        assert key in row
    assert row["omega_got"] < row["cutoff_omega"]


# ---------------------------------------------------------------------------
# negative cases
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kwargs", [
    {"k": 0.0, "depth": 10.0},
    {"k": -0.1, "depth": 10.0},
    {"k": 0.1, "depth": 0.0},
    {"k": 0.1, "depth": -5.0},
])
def test_invalid_wave_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        disp.strip_bem_depth_factor(kwargs["k"], kwargs["depth"])


def test_invalid_mass_loading_parameter_raises():
    with pytest.raises(ValueError):
        disp.mass_loading_omega2_from_depth_factor(-0.1, mu=MU)
    with pytest.raises(ValueError):
        disp.mass_loading_mu(0.0)
    with pytest.raises(ValueError):
        disp.mass_loading_cutoff_omega(0.0)


def test_invalid_mesh_and_mode_inputs_raise():
    with pytest.raises(ValueError):
        disp.build_periodic_strip(0.1, 10.0, panels_per_wavelength=3)
    with pytest.raises(ValueError):
        disp.strip_bem_depth_factor(0.1, 10.0, mode=0)
    with pytest.raises(ValueError):
        disp.strip_bem_mass_loading_omega2(
            0.1, depth=10.0, mu=MU, route="not-a-route"
        )


# ---------------------------------------------------------------------------
# the benchmark driver
# ---------------------------------------------------------------------------
def test_benchmark_driver_runs_and_stays_within_tolerance():
    from tensorbem.benchmarks import ice_mass_loading_dispersion as benchmark

    report = benchmark.run(
        panels_per_wavelength=32,
        convergence_levels=(8, 16),
        n_modes=2,
        field_grid=(11, 7),
    )
    assert report["case"] == "CASE-BEM-ICE-DISPERSION"
    rows = report["series"]["deep"]["rows"] + report["series"]["shallow"]["rows"]
    assert len(rows) == 9
    assert max(abs(r["omega_rel_error"]) for r in rows) < benchmark.TOLERANCE
    # the two numerical routes are the same operator, so they must coincide
    assert report["criteria"]["C2_routes_agree"]["passed"]
    assert report["criteria"]["C1_omega_error_below_tolerance"]["passed"]
    assert set(report["criteria"]) == {
        "C1_omega_error_below_tolerance", "C2_routes_agree",
        "C3_refinement_reduces_error", "C4_interior_field_error",
        "C5_deep_water_anchor", "C6_long_wave_anchor",
    }
    document = benchmark.render_document(report)
    assert "CASE-BEM-ICE-DISPERSION" in document
    assert "Weitz & Keller (1950)" in document