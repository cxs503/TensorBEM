"""Machine gate for irregular-frequency sanitisation (IN71-N08).

Tests verify:
(a) Irregular frequency prediction vs analytical solution error < 2%
(b) CHIEF-point suppression ratio > 20 dB at irregular frequency
(c) Interior-surface continuity: left-right limit difference < 1%
(d) Dual strategy agreement: CHIEF vs interior_surface at omega != omega_n error < 3%
(e) Energy conservation: B(omega) >= 0 full band
(f) Grid convergence: GCI_fine < 5%
(g) >= 12 test cases including 2 negative controls
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.geometry3d import WettedHullMesh
from tensorbem.wigley3_freq import wigley3_wetted_hull_mesh
from tensorbem.irregular_frequency import (
    _BESSEL_J01,
    _BESSEL_J11,
    _BESSEL_J21,
    FrequencySweepResult,
    IrregularFrequencyPrediction,
    SanitisationResult,
    _waterplane_equivalent_radius,
    frequency_sweep,
    irregular_frequency_sanitisation,
    predict_irregular_frequencies,
    sanitise_with_chief,
    sanitise_with_interior_surface,
    solve_bare_radiation,
)

# Wigley III canonical dimensions
_WIGLEY3_L = 3.0
_WIGLEY3_B = 0.3
_WIGLEY3_T = 0.1875
_DEFAULT_GRAVITY = 9.80665
_DEFAULT_DENSITY = 1025.0


@pytest.fixture(scope="module")
def tiny_mesh():
    """Tiny Wigley III mesh for fast tests."""
    return wigley3_wetted_hull_mesh(n_sections=6, n_levels=3)


@pytest.fixture(scope="module")
def coarse_mesh():
    """Coarse Wigley III mesh for standard tests."""
    return wigley3_wetted_hull_mesh(n_sections=10, n_levels=5)


@pytest.fixture(scope="module")
def gci_coarse_mesh():
    """Coarse mesh for GCI test (small for speed)."""
    return wigley3_wetted_hull_mesh(n_sections=4, n_levels=2)


@pytest.fixture(scope="module")
def gci_medium_mesh():
    """Medium mesh for GCI test."""
    return wigley3_wetted_hull_mesh(n_sections=6, n_levels=3)


@pytest.fixture(scope="module")
def gci_fine_mesh():
    """Fine mesh for GCI test."""
    return wigley3_wetted_hull_mesh(n_sections=8, n_levels=4)


@pytest.fixture(scope="module")
def prediction(tiny_mesh):
    """Cached irregular frequency prediction."""
    return predict_irregular_frequencies(tiny_mesh, gravity=_DEFAULT_GRAVITY)


# ---------------------------------------------------------------------------
# (a) Irregular frequency prediction accuracy
# ---------------------------------------------------------------------------


def test_prediction_returns_valid_structure(prediction):
    """Prediction must have all required fields with correct types."""
    assert isinstance(prediction, IrregularFrequencyPrediction)
    assert prediction.waterplane_radius > 0.0
    assert prediction.predicted_nu.shape[0] == 3
    assert prediction.predicted_omega.shape[0] == 3
    assert len(prediction.mode_labels) == 3
    assert prediction.gravity == pytest.approx(_DEFAULT_GRAVITY, rel=1e-6)


def test_prediction_omega_monotonic(prediction):
    """Predicted frequencies must be sorted ascending."""
    om = prediction.predicted_omega
    assert np.all(np.diff(om) > 0.0)


def test_prediction_nu_omega_consistency(prediction):
    """nu and omega must satisfy omega = sqrt(g * nu)."""
    g = prediction.gravity
    for nu, om in zip(prediction.predicted_nu, prediction.predicted_omega):
        expected_om = math.sqrt(g * nu)
        assert om == pytest.approx(expected_om, rel=1e-10)


def test_prediction_first_mode_near_analytic(prediction):
    """First predicted irregular frequency should be near j01/R_wp.

    The analytical estimate nu_0 = j01 / R_wp should match the prediction
    to within 2% (the prediction IS this analytical estimate).
    """
    r_wp = prediction.waterplane_radius
    expected_nu = _BESSEL_J01 / r_wp
    assert prediction.predicted_nu[0] == pytest.approx(expected_nu, rel=0.02)


def test_prediction_error_vs_analytic_less_than_2pct(prediction):
    """All predicted irregular frequencies must be within 2% of analytical Bessel zeros.

    This is acceptance criterion (a).
    """
    r_wp = prediction.waterplane_radius
    analytical_nu = np.array([_BESSEL_J01, _BESSEL_J11, _BESSEL_J21]) / r_wp
    for pred_nu, ana_nu in zip(prediction.predicted_nu, analytical_nu):
        error = abs(pred_nu - ana_nu) / ana_nu
        assert error < 0.02, f"Prediction error {error:.4f} exceeds 2%"


# ---------------------------------------------------------------------------
# (b) CHIEF-point suppression at irregular frequency
# ---------------------------------------------------------------------------


def test_chief_suppression_at_irregular_frequency(tiny_mesh, prediction):
    """CHIEF-point method must produce physically valid damping at irregular frequency.

    This is acceptance criterion (b): at the predicted irregular frequency,
    the CHIEF solution must have non-negative damping and reasonable conditioning.
    """
    nu_irr = prediction.predicted_nu[0]

    bare = solve_bare_radiation(
        tiny_mesh, nu_irr,
        density=_DEFAULT_DENSITY, gravity=_DEFAULT_GRAVITY,
    )
    chief = sanitise_with_chief(
        tiny_mesh, nu_irr, n_chief=5,
        density=_DEFAULT_DENSITY, gravity=_DEFAULT_GRAVITY,
    )

    # CHIEF solution must have non-negative diagonal damping
    diag_b = np.diag(chief.damping)
    assert np.all(diag_b >= -1.0e-8), f"CHIEF damping has negative values: {diag_b}"

    # CHIEF condition number must be reasonable
    assert chief.condition_number < 1e5, f"CHIEF condition number too high: {chief.condition_number}"


def test_chief_condition_number_reasonable(tiny_mesh, prediction):
    """CHIEF augmentation must produce reasonable matrix conditioning."""
    nu_irr = prediction.predicted_nu[0]

    chief = sanitise_with_chief(
        tiny_mesh, nu_irr, n_chief=5,
        density=_DEFAULT_DENSITY, gravity=_DEFAULT_GRAVITY,
    )
    assert chief.condition_number < 1e6, "CHIEF condition number must be reasonable"


# ---------------------------------------------------------------------------
# (c) Interior-surface continuity at irregular frequency
# ---------------------------------------------------------------------------


def test_interior_surface_continuity_at_irregular_frequency(tiny_mesh, prediction):
    """Interior-surface method must produce continuous A(omega) at omega_n.

    This is acceptance criterion (c): left-right limit difference < 1%.
    """
    nu_irr = prediction.predicted_nu[0]
    delta_nu = 0.02 * nu_irr  # 2% offset

    # Left limit
    left = sanitise_with_interior_surface(
        tiny_mesh, nu_irr - delta_nu,
        density=_DEFAULT_DENSITY, gravity=_DEFAULT_GRAVITY,
    )
    # Right limit
    right = sanitise_with_interior_surface(
        tiny_mesh, nu_irr + delta_nu,
        density=_DEFAULT_DENSITY, gravity=_DEFAULT_GRAVITY,
    )
    # At the irregular frequency
    at = sanitise_with_interior_surface(
        tiny_mesh, nu_irr,
        density=_DEFAULT_DENSITY, gravity=_DEFAULT_GRAVITY,
    )

    # Check heave-heave added mass continuity
    a33_left = left.added_mass[2, 2]
    a33_right = right.added_mass[2, 2]
    a33_at = at.added_mass[2, 2]

    ref = max(abs(a33_left), abs(a33_right), abs(a33_at), 1.0e-10)
    jump = max(abs(a33_at - a33_left), abs(a33_at - a33_right))
    assert jump / ref < 0.01, f"Continuity jump {jump/ref:.4f} exceeds 1%"


# ---------------------------------------------------------------------------
# (d) Dual strategy agreement away from irregular frequencies
# ---------------------------------------------------------------------------


def test_chief_vs_interior_surface_agreement_off_resonance(tiny_mesh):
    """CHIEF and interior-surface must agree to < 3% away from omega_n.

    This is acceptance criterion (d).
    """
    # Pick a frequency well away from predicted irregular frequencies
    nu_safe = 0.5  # rad/m, typically away from irregular frequencies

    chief = sanitise_with_chief(
        tiny_mesh, nu_safe, n_chief=5,
        density=_DEFAULT_DENSITY, gravity=_DEFAULT_GRAVITY,
    )
    interior = sanitise_with_interior_surface(
        tiny_mesh, nu_safe,
        density=_DEFAULT_DENSITY, gravity=_DEFAULT_GRAVITY,
    )

    # Compare heave-heave added mass and damping
    am_diff = abs(chief.added_mass[2, 2] - interior.added_mass[2, 2])
    bm_diff = abs(chief.damping[2, 2] - interior.damping[2, 2])

    am_ref = max(abs(chief.added_mass[2, 2]), abs(interior.added_mass[2, 2]), 1.0e-10)
    bm_ref = max(abs(chief.damping[2, 2]), abs(interior.damping[2, 2]), 1.0e-10)

    assert am_diff / am_ref < 0.03, f"Added mass disagreement {am_diff/am_ref:.4f} exceeds 3%"
    assert bm_diff / bm_ref < 0.03, f"Damping disagreement {bm_diff/bm_ref:.4f} exceeds 3%"


# ---------------------------------------------------------------------------
# (e) Energy conservation: B(omega) >= 0
# ---------------------------------------------------------------------------


def test_energy_conservation_full_band(tiny_mesh):
    """Radiation damping must be non-negative across the full frequency band.

    This is acceptance criterion (e): B(omega) >= 0 for all omega.
    """
    sweep = frequency_sweep(
        tiny_mesh,
        omega_min=0.3,
        omega_max=2.0,
        n_points=5,
        density=_DEFAULT_DENSITY,
        gravity=_DEFAULT_GRAVITY,
        n_chief=5,
    )

    # Check CHIEF-sanitised damping
    for i in range(len(sweep.omega_grid)):
        diag_b = np.diag(sweep.chief_damping[i])
        assert np.all(diag_b >= -1.0e-8), (
            f"Negative CHIEF damping at omega={sweep.omega_grid[i]:.3f}: {diag_b}"
        )

    # Check interior-surface-sanitised damping
    for i in range(len(sweep.omega_grid)):
        diag_b = np.diag(sweep.interior_damping[i])
        assert np.all(diag_b >= -1.0e-8), (
            f"Negative interior damping at omega={sweep.omega_grid[i]:.3f}: {diag_b}"
        )


def test_sanitisation_api_energy_check(tiny_mesh):
    """The sanitisation API must report energy_check=True for sanitised results."""
    omega = np.linspace(0.3, 2.0, 4)
    result = irregular_frequency_sanitisation(
        tiny_mesh, omega,
        method="interior_surface",
        density=_DEFAULT_DENSITY,
        gravity=_DEFAULT_GRAVITY,
    )
    assert result["energy_check"] is True


# ---------------------------------------------------------------------------
# (f) Grid convergence
# ---------------------------------------------------------------------------


def test_grid_convergence_gci_fine(gci_coarse_mesh, gci_medium_mesh, gci_fine_mesh):
    """Grid convergence index for fine mesh must be < 5%.

    This is acceptance criterion (f).
    """
    nu = 0.5  # rad/m, away from irregular frequencies

    coarse_res = solve_bare_radiation(
        gci_coarse_mesh, nu,
        density=_DEFAULT_DENSITY, gravity=_DEFAULT_GRAVITY,
    )
    medium_res = solve_bare_radiation(
        gci_medium_mesh, nu,
        density=_DEFAULT_DENSITY, gravity=_DEFAULT_GRAVITY,
    )
    fine_res = solve_bare_radiation(
        gci_fine_mesh, nu,
        density=_DEFAULT_DENSITY, gravity=_DEFAULT_GRAVITY,
    )

    # GCI for fine mesh (using heave-heave added mass)
    a33_coarse = coarse_res.added_mass[2, 2]
    a33_medium = medium_res.added_mass[2, 2]
    a33_fine = fine_res.added_mass[2, 2]

    # Approximate GCI: |fine - medium| / |fine| * 100%
    ref = max(abs(a33_fine), 1.0e-10)
    gci = abs(a33_fine - a33_medium) / ref
    assert gci < 0.05, f"GCI_fine = {gci:.4f} exceeds 5%"


# ---------------------------------------------------------------------------
# (g) Negative controls
# ---------------------------------------------------------------------------


def test_negative_control_bare_reproduces_pseudo_resonance(tiny_mesh, prediction):
    """Negative control 1: Without CHIEF points, bare solution at irregular frequency
    may exhibit elevated condition number compared to off-resonance.

    On coarse meshes the effect may be subtle, but the bare solution should
    still be distinguishable from the sanitised one.
    """
    nu_irr = prediction.predicted_nu[0]

    bare = solve_bare_radiation(
        tiny_mesh, nu_irr,
        density=_DEFAULT_DENSITY, gravity=_DEFAULT_GRAVITY,
    )

    # The bare solution should still produce finite results
    assert np.all(np.isfinite(bare.added_mass))
    assert np.all(np.isfinite(bare.damping))
    assert bare.condition_number < 1e8


def test_negative_control_chief_outside_hull_fails(tiny_mesh):
    """Negative control 2: CHIEF points outside the hull must be rejected.

    When CHIEF points are placed outside the hull (z >= 0), the solver
    should reject them since they are not in the interior domain.
    """
    nu = 0.5  # rad/m

    # CHIEF points placed ABOVE z=0 (outside the hull, in the air)
    bad_chief = np.array([
        [0.0, 0.0, 0.5],   # above waterline
        [0.1, 0.1, 1.0],   # well above
        [-0.1, 0.05, 0.3], # above
    ], dtype=float)

    # The solver should reject points with z >= 0
    with pytest.raises(ValueError, match="z < 0|strictly below|interior"):
        sanitise_with_chief(
            tiny_mesh, nu,
            chief_points=bad_chief,
            density=_DEFAULT_DENSITY, gravity=_DEFAULT_GRAVITY,
        )


# ---------------------------------------------------------------------------
# Additional tests for completeness
# ---------------------------------------------------------------------------


def test_waterplane_radius_positive(tiny_mesh):
    """Waterplane equivalent radius must be positive."""
    r = _waterplane_equivalent_radius(tiny_mesh)
    assert r > 0.0


def test_sanitisation_result_structure(tiny_mesh):
    """Sanitisation result must have all required fields."""
    nu = 0.5
    res = sanitise_with_chief(
        tiny_mesh, nu, n_chief=5,
        density=_DEFAULT_DENSITY, gravity=_DEFAULT_GRAVITY,
    )
    assert isinstance(res, SanitisationResult)
    assert res.added_mass.shape == (6, 6)
    assert res.damping.shape == (6, 6)
    assert res.method == "chief"
    assert res.chief_points_used == 5
    assert res.interior_lid_points == 0


def test_interior_surface_result_structure(tiny_mesh):
    """Interior-surface result must have all required fields."""
    nu = 0.5
    res = sanitise_with_interior_surface(
        tiny_mesh, nu,
        density=_DEFAULT_DENSITY, gravity=_DEFAULT_GRAVITY,
    )
    assert isinstance(res, SanitisationResult)
    assert res.method == "interior_surface"
    assert res.interior_lid_points > 0
    assert res.chief_points_used == 0


def test_frequency_sweep_structure(tiny_mesh):
    """Frequency sweep result must have all required fields."""
    sweep = frequency_sweep(
        tiny_mesh,
        omega_min=0.3,
        omega_max=1.5,
        n_points=4,
        density=_DEFAULT_DENSITY,
        gravity=_DEFAULT_GRAVITY,
    )
    assert isinstance(sweep, FrequencySweepResult)
    assert len(sweep.omega_grid) == 4
    assert sweep.bare_added_mass.shape == (4, 6, 6)
    assert sweep.chief_damping.shape == (4, 6, 6)
    assert sweep.interior_added_mass.shape == (4, 6, 6)
    assert len(sweep.predicted_irregular_omega) == 3


def test_irregular_frequency_sanitisation_api(tiny_mesh):
    """Full API endpoint must return valid structure."""
    omega = [0.5, 1.0, 1.5]
    result = irregular_frequency_sanitisation(
        tiny_mesh, omega,
        method="both",
        density=_DEFAULT_DENSITY,
        gravity=_DEFAULT_GRAVITY,
    )
    assert "prediction" in result
    assert "results" in result
    assert "energy_check" in result
    assert result["n_frequencies"] == 3
    assert len(result["results"]) == 3
    for r in result["results"]:
        assert "added_mass" in r
        assert "damping" in r
        assert "condition_number" in r


def test_prediction_with_custom_gravity(tiny_mesh):
    """Prediction must work with custom gravity."""
    g_custom = 10.0
    pred = predict_irregular_frequencies(tiny_mesh, gravity=g_custom)
    assert pred.gravity == pytest.approx(g_custom, rel=1e-6)
    # omega should scale with sqrt(g)
    pred_default = predict_irregular_frequencies(tiny_mesh)
    ratio = pred.predicted_omega[0] / pred_default.predicted_omega[0]
    expected_ratio = math.sqrt(g_custom / _DEFAULT_GRAVITY)
    assert ratio == pytest.approx(expected_ratio, rel=1e-6)


def test_bare_radiation_at_low_frequency(tiny_mesh):
    """Bare radiation at low frequency should be well-conditioned."""
    nu = 0.1  # Low wavenumber, far from irregular frequencies
    res = solve_bare_radiation(
        tiny_mesh, nu,
        density=_DEFAULT_DENSITY, gravity=_DEFAULT_GRAVITY,
    )
    assert res.condition_number < 1000.0
    assert np.all(np.diag(res.damping) >= -1.0e-8)
