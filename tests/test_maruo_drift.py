"""Verification suite for the Maruo far-field mean drift force (m6271).

The module under test returns a *vector* force from a sampled Kochin table, so
the only honest way to check it is against closed-form azimuthal integrals and
against frame symmetries.  All reference values below are analytic, not
recorded program output.

Closed form used throughout.  With ``|K(theta)|^2 = C^2 (1 + a cos(theta-gamma))``

    S  = int |K|^2 dtheta            = 2 pi C^2
    Ci = int |K|^2 cos(theta) dtheta = a pi C^2 cos(gamma)
    Di = int |K|^2 sin(theta) dtheta = a pi C^2 sin(gamma)

and the covariant Maruo weight ``e_beta - e_theta`` gives

    F(beta) = (rho omega^2 / 4k) [ S e_beta - (Ci, Di) ]
            = (rho g / 4) [ 2 pi C^2 e_beta - a pi C^2 e_gamma ]

using ``omega^2 = g k``, i.e. the prefactor is ``rho g / 4`` and does *not*
depend on the wave number.

Reference: Maruo (1957, 1960); Newman (1977) Marine Hydrodynamics sec. 7.3.
The identical weight is implemented, and separately verified, by the
control-surface derivation in ``tensorbem.drift.far_field_drift_3d``.
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pytest

from tensorbem.maruo_drift import (
    maruo_mean_drift_force,
    maruo_mean_drift_force_general,
)

RHO = 1025.0
G = 9.81
K_WAVE = 0.5
# rho*omega^2/(4k) with omega^2 = g k
COEFF = RHO * G / 4.0

# 3600 azimuth samples of the Kochin table give a linear-interpolation floor of
# ~5e-7 relative on the closed forms below (measured, see
# docs/maruo_drift.md table 3); tolerances are set above that floor.
N_TABLE = 3600
RTOL = 1e-4
# forces are O(1e4) N here; the interpolation floor is ~5e-3 N
ATOL = 1.0


def _table(n=N_TABLE, gamma=0.0, c=1.0, a=0.5):
    """Sampled Kochin table with |K|^2 = c^2 (1 + a cos(theta - gamma))."""
    theta = np.linspace(0.0, 2.0 * math.pi, n + 1)[:-1]
    pattern = np.maximum(1.0 + a * np.cos(theta - gamma), 0.0)
    return c * np.sqrt(pattern), theta


def _closed_form(beta, gamma=0.0, c=1.0, a=0.5):
    """(Fx, Fy) from the covariant Maruo integral, analytic in theta."""
    s = 2.0 * math.pi * c * c
    moment = a * math.pi * c * c * np.array([math.cos(gamma), math.sin(gamma)])
    e_beta = np.array([math.cos(beta), math.sin(beta)])
    return COEFF * (s * e_beta - moment)


def _force(result):
    return np.array([result["mean_drift_force_x"], result["mean_drift_force_y"]])


# ---------------------------------------------------------------------------
# closed-form azimuthal integrals (prefactor, quadrature, weight)
# ---------------------------------------------------------------------------
def test_isotropic_kochin_pushes_along_propagation():
    """|K|^2 = const: S = 2 pi C^2 and both moments vanish -> F along e_beta."""
    c = 0.7
    koch = np.full(N_TABLE, c, dtype=np.complex128) * np.exp(0.3j)  # phase is irrelevant
    _, theta = _table()
    for beta in (0.0, 0.9, math.pi / 2, 2.7):
        force = _force(maruo_mean_drift_force_general(koch, theta, K_WAVE, wave_angle=beta))
        expected = _closed_form(beta, c=c, a=0.0)
        np.testing.assert_allclose(force, expected, rtol=RTOL, atol=ATOL)
        assert np.linalg.norm(force) == pytest.approx(COEFF * 2.0 * math.pi * c * c, rel=RTOL)


def test_dipole_scatterer_force_is_along_propagation():
    """|K|^2 = C^2 cos^2(theta): S = pi C^2, moments vanish for every beta.

    A fore-aft symmetric dipole (the long-wave vertical cylinder) therefore
    feels a drift force of magnitude ``(rho g/4) pi C^2`` pointing exactly in
    the direction of wave propagation, at *any* heading.
    """
    theta = np.linspace(0.0, 2.0 * math.pi, N_TABLE, endpoint=False)
    koch = np.cos(theta)  # |K|^2 = cos^2(theta): S = pi, both moments vanish
    magnitude = COEFF * math.pi
    for beta in (0.0, 0.7, math.pi / 2, 2.6):
        force = _force(maruo_mean_drift_force_general(koch, theta, K_WAVE, wave_angle=beta))
        assert np.linalg.norm(force) == pytest.approx(magnitude, rel=RTOL)
        assert math.atan2(force[1], force[0]) == pytest.approx(beta, abs=1e-6)


def test_beam_and_following_seas_for_asymmetric_scatterer():
    """Regression pin for the pre-m6271 weight (cos beta - cos(theta-beta)).

    With ``|K|^2 = C^2 (1 + 0.5 cos theta)`` (fore-aft asymmetric scattering
    pattern) the old weight returned F_x = 0 in beam seas and 1.18e4 N of
    drag in following seas; the covariant weight gives -3.95e3 N and
    -1.97e4 N.  Errors were 100 % / 25 % and 40 %.
    """
    koch, theta = _table(gamma=0.0, a=0.5)

    beam = _force(maruo_mean_drift_force_general(koch, theta, K_WAVE, wave_angle=math.pi / 2))
    expected_beam = _closed_form(math.pi / 2, a=0.5)
    np.testing.assert_allclose(beam, expected_beam, rtol=RTOL, atol=ATOL)
    # lateral component: the incident-flux term minus the radiated moment
    assert beam[0] == pytest.approx(-COEFF * 0.5 * math.pi, rel=RTOL)
    assert beam[1] == pytest.approx(COEFF * 2.0 * math.pi, rel=RTOL)

    following = _force(
        maruo_mean_drift_force_general(koch, theta, K_WAVE, wave_angle=math.pi)
    )
    np.testing.assert_allclose(
        following, _closed_form(math.pi, a=0.5), rtol=RTOL, atol=ATOL
    )
    # the radiated moment adds to the isotropic term in following seas
    assert abs(following[0]) == pytest.approx(COEFF * 2.5 * math.pi, rel=RTOL)


@pytest.mark.parametrize(
    ("gamma", "beta"),
    [(0.0, math.pi / 2), (0.0, math.pi), (0.0, math.pi / 4), (0.7, 1.1), (2.3, -0.4)],
)
def test_closed_form_for_rotated_scattering_pattern(gamma, beta):
    koch, theta = _table(gamma=gamma, a=0.5)
    force = _force(maruo_mean_drift_force_general(koch, theta, K_WAVE, wave_angle=beta))
    np.testing.assert_allclose(
        force, _closed_form(beta, gamma=gamma, a=0.5), rtol=RTOL, atol=ATOL
    )


# ---------------------------------------------------------------------------
# frame symmetries
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("alpha", "beta"), [(0.7, math.pi / 3), (1.3, 0.2), (2.5, -0.9)])
def test_rotational_covariance(alpha, beta):
    """Rotating body and incident waves together by alpha rotates the force."""
    gamma = 0.3
    koch0, theta0 = _table(gamma=gamma)
    koch1, theta1 = _table(gamma=gamma + alpha)
    f0 = _force(maruo_mean_drift_force_general(koch0, theta0, K_WAVE, wave_angle=beta))
    f1 = _force(maruo_mean_drift_force_general(koch1, theta1, K_WAVE, wave_angle=beta + alpha))
    rot = np.array([[math.cos(alpha), -math.sin(alpha)], [math.sin(alpha), math.cos(alpha)]])
    np.testing.assert_allclose(f1, rot @ f0, rtol=1e-6, atol=1e-9)


def test_drift_force_has_non_negative_projection_on_propagation():
    """F . e_beta = (rho g/4)(S - int|K|^2 cos(theta-beta)) >= 0 since cos <= 1.

    Energy argument: a fixed body can only extract momentum from the wave
    train, so the force never points against the direction of propagation.
    """
    rng = np.random.default_rng(20261106)
    theta = np.linspace(0.0, 2.0 * math.pi, 720, endpoint=False)
    for _ in range(5):
        raw = rng.normal(size=theta.size) + 1j * rng.normal(size=theta.size)
        koch = np.convolve(raw, np.ones(9) / 9.0, mode="same")  # smooth table
        for beta in (0.0, 0.6, math.pi / 2, 2.9):
            force = _force(maruo_mean_drift_force_general(koch, theta, K_WAVE, wave_angle=beta))
            projection = force @ np.array([math.cos(beta), math.sin(beta)])
            assert projection >= 0.0


def test_head_sea_entry_matches_general_entry_at_zero_angle():
    koch, theta = _table(gamma=0.0, a=0.5)
    head = maruo_mean_drift_force(koch, theta, K_WAVE)
    general = maruo_mean_drift_force_general(koch, theta, K_WAVE, wave_angle=0.0)
    assert head["mean_drift_force_x"] == pytest.approx(general["mean_drift_force_x"], rel=1e-12)
    assert general["mean_drift_force_y"] == pytest.approx(0.0, abs=1e-9)
    assert head["flux_integral"] == pytest.approx(general["flux_integral_x"], rel=1e-12)


def test_prefactor_is_wavenumber_independent():
    """rho omega^2/(4k) = rho g/4 in deep water: F must not depend on k."""
    koch, theta = _table()
    values = [
        maruo_mean_drift_force(koch, theta, k)["mean_drift_force_x"]
        for k in (0.05, 0.5, 2.0, 8.0)
    ]
    np.testing.assert_allclose(values, values[0], rtol=1e-12)
    assert values[0] == pytest.approx(COEFF * math.pi * (2.0 - 0.5), rel=RTOL)


def test_zero_kochin_gives_zero_force():
    _, theta = _table()
    zeros = np.zeros(theta.size, dtype=np.complex128)
    assert maruo_mean_drift_force(zeros, theta, K_WAVE)["mean_drift_force_x"] == 0.0
    general = maruo_mean_drift_force_general(zeros, theta, K_WAVE, wave_angle=1.1)
    assert general["mean_drift_force_x"] == 0.0
    assert general["mean_drift_force_y"] == 0.0


def test_scaling_in_density_and_gravity():
    koch, theta = _table()
    base = maruo_mean_drift_force(koch, theta, K_WAVE)["mean_drift_force_x"]
    doubled = maruo_mean_drift_force(koch, theta, K_WAVE, density=2.0 * RHO)["mean_drift_force_x"]
    assert doubled == pytest.approx(2.0 * base, rel=1e-12)


# ---------------------------------------------------------------------------
# documented accuracy limits
# ---------------------------------------------------------------------------
def test_azimuth_table_not_quadrature_limits_accuracy():
    """The linear interpolation of the sampled table, not n_theta, sets the floor.

    A dipole table of 24 samples is ~1.1e-2 off; refining n_theta from 128 to
    16384 changes nothing, because the integrand is already the interpolant.
    Callers must supply a table with spacing well below the smallest
    azimuthal scale of K.
    """
    theta = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
    coarse = np.cos(theta)
    magnitude = COEFF * math.pi
    errors = [
        abs(maruo_mean_drift_force(coarse, theta, K_WAVE, n_theta=n)["mean_drift_force_x"] - magnitude)
        / magnitude
        for n in (128, 16384)
    ]
    assert 5e-3 < errors[0] < 5e-2
    assert errors[1] == pytest.approx(errors[0], rel=2e-2)

    fine_theta = np.linspace(0.0, 2.0 * math.pi, N_TABLE, endpoint=False)
    fine = abs(
        maruo_mean_drift_force(np.cos(fine_theta), fine_theta, K_WAVE)["mean_drift_force_x"]
        - magnitude
    ) / magnitude
    assert fine < 1e-6


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"wavenumber": 0.0}, "wavenumber"),
        ({"wavenumber": -1.0}, "wavenumber"),
        ({"wavenumber": 0.5, "gravity": 0.0}, "gravity"),
        ({"wavenumber": 0.5, "density": -1.0}, "density"),
        ({"wavenumber": 0.5, "n_theta": 127}, "n_theta"),
    ],
)
def test_invalid_arguments_raise(kwargs, match):
    koch, theta = _table(n=64)
    with pytest.raises(ValueError, match=match):
        maruo_mean_drift_force(koch, theta, **kwargs)
    with pytest.raises(ValueError, match=match):
        maruo_mean_drift_force_general(koch, theta, wave_angle=0.4, **kwargs)


def test_mismatched_table_lengths_raise():
    koch, theta = _table(n=64)
    with pytest.raises(ValueError, match="same length"):
        maruo_mean_drift_force(koch[:-1], theta, K_WAVE)
    with pytest.raises(ValueError, match="same length"):
        maruo_mean_drift_force_general(koch, theta[:-2], K_WAVE, wave_angle=0.2)


def test_non_one_dimensional_table_raises():
    theta = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
    with pytest.raises(ValueError, match="one-dimensional"):
        maruo_mean_drift_force(np.cos(theta).reshape(8, 8), theta, K_WAVE)

# ---------------------------------------------------------------------------
# verification driver quoted by docs/maruo_drift.md
# ---------------------------------------------------------------------------
def test_verification_driver_reproduces_the_documented_verdicts():
    from tensorbem.benchmarks import maruo_drift_verification as driver

    report = driver.study()
    summary = report["summary"]
    assert summary["rel_err_covariant_max"] < 1e-5
    assert summary["closed_form_within_3pct_covariant"] is True
    assert summary["rel_err_legacy_max"] > 0.3
    assert summary["closed_form_within_3pct_legacy"] is False
    assert summary["covariance_residual_covariant_max"] < 1e-9
    assert summary["covariance_residual_legacy_max"] > 1e-2

    beam = [r for r in report["headings"] if abs(r["beta_deg"] - 90.0) < 1e-9][0]
    # the pre-m6271 weight annihilated the beam-sea lateral force entirely
    assert abs(beam["fx_legacy"]) < 1e-6
    assert beam["fx_covariant"] < -1e3
    # table resolution, not quadrature, sets the floor
    table = [r for r in report["resolution"]["table"] if r["n_table"] == 24][0]
    quad = [r for r in report["resolution"]["quadrature"] if r["n_theta"] == 16384][0]
    assert table["rel_err"] > 5e-3 > quad["rel_err"]


def test_report_figures_render(tmp_path):
    pytest.importorskip("matplotlib")
    from tensorbem.benchmarks import maruo_drift_verification as driver

    report = driver.study()
    driver.figure_polar(tmp_path / "polar.png", report)
    driver.figure_resolution(tmp_path / "resolution.png", report)
    assert (tmp_path / "polar.png").stat().st_size > 10_000
    assert (tmp_path / "resolution.png").stat().st_size > 10_000


# ---------------------------------------------------------------------------
# the published m6287 evidence bundle must reproduce a fresh build (m6287)
# ---------------------------------------------------------------------------
def _repo_root():
    return pathlib.Path(__file__).resolve().parents[1]


def _build_report_module():
    path = _repo_root() / "docs" / "m6287" / "build_report.py"
    if not path.is_file():
        pytest.skip("docs/m6287 evidence bundle is absent")
    import importlib.util

    spec = importlib.util.spec_from_file_location("m6287_build_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_docs_m6287_artifact_reproduces_against_the_module(tmp_path):
    """The JSON and Markdown in docs/m6287 are build output, not prose.

    A fresh build must reproduce both byte for byte (so a hand-edited number or
    a changed module fails here), and the verdicts it publishes must be the
    measured ones: the shipped covariant weight inside 3 %, the pre-m6271
    weight outside the 15 % acceptance limit.
    """
    bundle = _repo_root() / "docs" / "m6287"
    builder = _build_report_module()
    json_path = bundle / "maruo_general_heading_fix.json"
    md_path = bundle / "maruo_general_heading_fix.md"
    recorded = json.loads(json_path.read_text(encoding="utf-8"))

    fresh = builder.main(path=tmp_path / "fresh.json", markdown=tmp_path / "fresh.md")
    assert fresh["summary"] == recorded["summary"]
    assert json.loads((tmp_path / "fresh.json").read_text(encoding="utf-8")) == recorded
    assert (tmp_path / "fresh.md").read_text(encoding="utf-8") == md_path.read_text(encoding="utf-8")

    summary = recorded["summary"]
    assert summary["rel_err_covariant_max"] < 0.03
    assert summary["closed_form_within_3pct_covariant"] is True
    assert summary["rel_err_legacy_max"] > 0.15
    assert summary["closed_form_within_15pct_legacy"] is False
    assert summary["worst_legacy_heading_deg"] == pytest.approx(180.0)

    beam = recorded["key_cases"]["beam_seas"]
    # the pre-m6271 weight annihilated the beam-sea side force; the shipped one does not
    assert abs(beam["fx_legacy"]) < 1e-6
    assert beam["fx_covariant"] < -1e3
    assert beam["rel_err_legacy"] > 0.3


def test_docs_m6287_figure_renders(tmp_path):
    pytest.importorskip("matplotlib")
    import importlib.util

    path = _repo_root() / "docs" / "m6287" / "render_report.py"
    if not path.is_file():
        pytest.skip("docs/m6287 evidence bundle is absent")
    spec = importlib.util.spec_from_file_location("m6287_render_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    report = json.loads(
        (_repo_root() / "docs" / "m6287" / "maruo_general_heading_fix.json").read_text(encoding="utf-8")
    )
    out = module.render(report, tmp_path / "fig1.png")
    assert out.stat().st_size > 20_000
    published = _repo_root() / "docs" / "m6287" / "figures" / "fig1_heading_error.png"
    assert published.is_file() and published.stat().st_size > 20_000
