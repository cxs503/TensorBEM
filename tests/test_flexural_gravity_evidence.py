"""Evidence tests for the flexural-gravity verification face (mission m14560).

These tests pin the machine-checkable claims of
``docs/flexural_gravity_verification/evidence.json``:

* the coupled plate/fluid dispersion (eq. 3) is numerically identical to the
  independent implementation in :mod:`tensorbem.hydroelastic`, which now
  delegates to it (integration caller);
* the short-wave limit reproduces the in-vacuo Euler-Bernoulli plate wave;
* the ice-edge matching conserves energy;
* the inertia-dominated long waves travel slower under the ice.
"""
from __future__ import annotations

import math

import pytest

from tensorbem import flexural_gravity as fg
from tensorbem import ice as ice_module
from tensorbem import hydroelastic as hyd

G = fg.DEFAULT_GRAVITY
RHO_W = fg.DEFAULT_WATER_DENSITY
RHO_I = fg.DEFAULT_ICE_DENSITY
E_ICE = fg.DEFAULT_YOUNG_MODULUS
NU = fg.DEFAULT_POISSON_RATIO
DEPTH = 200.0
THICKNESSES = (0.5, 1.0, 2.0)
K_GRID = (1e-2, 5e-2, 0.2, 1.0, 5.0, 20.0)


def ice_plate(h: float, *, E: float = E_ICE) -> fg.IcePlate:
    return fg.IcePlate(thickness=h, young_modulus=E, poisson_ratio=NU,
                       water_density=RHO_W, ice_density=RHO_I,
                       water_depth=DEPTH, gravity=G)


def elastic_panel(h: float) -> hyd.ElasticPanel:
    return hyd.ElasticPanel(
        length=1.0, breadth=1.0, thickness=h, young_modulus=E_ICE,
        poisson_ratio=NU, plate_density=RHO_I, fluid_density=RHO_W,
        gravity=G, water_depth=DEPTH,
    )


def test_hydroelastic_delegates_dispersion_to_flexural_gravity() -> None:
    """The main plate/fluid solver must call the shared implementation."""
    assert hyd.plate_fluid_omega is fg.plate_fluid_omega


@pytest.mark.parametrize("h", THICKNESSES)
def test_eq3_matches_independent_hydroelastic_implementation(h: float) -> None:
    plate = ice_plate(h)
    panel = elastic_panel(h)
    for k in K_GRID:
        got = fg.flexural_gravity_omega_mass_loading(k, plate)
        ref = hyd.flexural_gravity_omega(k, panel)
        assert got == pytest.approx(ref, rel=1e-12), (h, k, got, ref)


@pytest.mark.parametrize("h", THICKNESSES)
def test_plate_fluid_omega_matches_eq3_ice_plate(h: float) -> None:
    plate = ice_plate(h)
    for k in K_GRID:
        got = fg.plate_fluid_omega(
            k, rigidity=plate.flexural_rigidity,
            mass_per_area=RHO_I * h, fluid_density=RHO_W,
            water_depth=DEPTH, gravity=G,
        )
        assert got == pytest.approx(
            fg.flexural_gravity_omega_mass_loading(k, plate), rel=1e-12)


def test_plate_fluid_omega_deep_water_limit_matches_finite_depth() -> None:
    for k in (0.5, 1.0, 5.0):
        finite = fg.plate_fluid_omega(
            k, rigidity=5.0e7, mass_per_area=400.0, fluid_density=RHO_W,
            water_depth=DEPTH, gravity=G)
        deep = fg.plate_fluid_omega(
            k, rigidity=5.0e7, mass_per_area=400.0, fluid_density=RHO_W,
            water_depth=None, gravity=G)
        assert finite == pytest.approx(deep, rel=1e-4)


@pytest.mark.parametrize("h", THICKNESSES)
def test_eq3_short_wave_limit_is_in_vacuo_plate_wave(h: float) -> None:
    """k -> inf: omega^2 -> D k^4 / (rho_i h) (Timoshenko plate theory)."""
    plate = ice_plate(h)
    k = 1000.0
    got = fg.flexural_gravity_omega2_mass_loading(k, plate)
    ref = plate.flexural_rigidity * k ** 4 / (RHO_I * h)
    assert got == pytest.approx(ref, rel=2.0e-2)


@pytest.mark.parametrize("h", THICKNESSES)
def test_zero_rigidity_is_mass_loading(h: float) -> None:
    """D = 0 in the coupled dispersion reproduces tensorbem.ice mass loading."""
    cover = ice_module.IceCover(thickness=h, ice_density=RHO_I,
                               water_density=RHO_W, water_depth=DEPTH, gravity=G)
    plate = ice_plate(h)
    for k in K_GRID:
        dep = fg.plate_fluid_omega(
            k, rigidity=0.0, mass_per_area=RHO_I * h, fluid_density=RHO_W,
            water_depth=DEPTH, gravity=G)
        assert dep == pytest.approx(ice_module.mass_loading_omega(k, cover), rel=1e-12)
        # ... and the algebraic identity with the IcePlate eq.(3) helper
        assert dep == pytest.approx(
            math.sqrt(fg.open_water_omega2(k, DEPTH, G)
                      / (1.0 + fg.mass_loading_parameter(plate) * k
                         * math.tanh(k * DEPTH))), rel=1e-12)


@pytest.mark.parametrize("h", THICKNESSES)
@pytest.mark.parametrize("omega", [0.2, 0.3, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("inertia", [False, True])
def test_ice_edge_energy_is_conserved(h: float, omega: float, inertia: bool) -> None:
    res = fg.ice_edge_coefficients(omega, ice_plate(h), include_plate_inertia=inertia)
    assert abs(res.energy_residual) <= 1e-4
    assert abs(res.R_energy ** 2 + res.T_energy ** 2 - 1.0) <= 1e-4


@pytest.mark.parametrize("h", THICKNESSES)
def test_long_inertia_dominated_waves_are_shorter_under_ice(h: float) -> None:
    plate = ice_plate(h)
    for omega in (0.2, 0.3):
        k0 = fg.open_water_wavenumber(omega, DEPTH, G)
        k1 = fg.flexural_gravity_wavenumber_mass_loading(omega, plate)
        assert k1 > k0                      # lambda_ice = 2 pi / k1 < lambda_open
        assert 2.0 * math.pi / k1 < 2.0 * math.pi / k0


@pytest.mark.parametrize("h", THICKNESSES)
def test_dispersion_root_round_trip(h: float) -> None:
    plate = ice_plate(h)
    for omega in (0.2, 0.5, 1.0, 2.0, 3.0):
        k = fg.flexural_gravity_wavenumber_mass_loading(omega, plate)
        assert fg.flexural_gravity_omega_mass_loading(k, plate) == pytest.approx(
            omega, rel=1e-10)

# ---------------------------------------------------------------------------
# independent finite-difference depth solve
# ---------------------------------------------------------------------------
from tensorbem import flexural_gravity_numeric as fgn  # noqa: E402


@pytest.mark.parametrize("k", [1e-2, 5e-2, 0.2, 1.0, 5.0, 20.0])
def test_fd_depth_factor_matches_k_tanh_kh(k: float) -> None:
    g0, dz, nodes = fgn.fd_depth_gradient_ratio(k, DEPTH)
    exact = k * math.tanh(k * DEPTH)
    assert g0 == pytest.approx(exact, rel=2e-3), (k, dz, nodes)


def test_fd_depth_factor_is_second_order() -> None:
    k = 1.0
    exact = k * math.tanh(k * DEPTH)
    errors = []
    for dz in (0.05, 0.025, 0.0125):
        g0, _, _ = fgn.fd_depth_gradient_ratio(k, DEPTH, dz=dz)
        errors.append(abs(g0 - exact) / exact)
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert all(r > 3.5 for r in ratios), (errors, ratios)


@pytest.mark.parametrize("h", THICKNESSES)
@pytest.mark.parametrize("k", [1e-2, 0.2, 1.0, 5.0, 20.0])
def test_fd_dispersion_matches_closed_form(h: float, k: float) -> None:
    plate = ice_plate(h)
    got = fgn.fd_omega2(
        k, depth=DEPTH, rigidity=plate.flexural_rigidity,
        mass_per_area=RHO_I * h, fluid_density=RHO_W, gravity=G)
    ref = fg.flexural_gravity_omega2_mass_loading(k, plate)
    assert got == pytest.approx(ref, rel=1e-3)


def test_fd_open_water_limit_matches_gravity_wave() -> None:
    for k in (0.01, 0.1, 1.0, 10.0):
        got = fgn.fd_omega(k, depth=DEPTH, rigidity=0.0, mass_per_area=0.0)
        ref = math.sqrt(fg.open_water_omega2(k, DEPTH, G))
        assert got == pytest.approx(ref, rel=2e-3)


def test_omega2_from_depth_factor_rearranges_the_free_surface_condition() -> None:
    k, g0 = 2.0, 1.9
    rho, g, D, m_pp = RHO_W, G, 5.0e7, 400.0
    omega2 = fgn.omega2_from_depth_factor(
        g0, k=k, rigidity=D, mass_per_area=m_pp,
        fluid_density=rho, gravity=g)
    residual = (rho * g + D * k ** 4 - m_pp * omega2) * g0 - rho * omega2
    assert abs(residual) <= 1e-9 * max(1.0, rho * omega2)
