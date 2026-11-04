"""Machine gate for the irregular-frequency band study.

These tests re-derive every claim of
:mod:`tensorbem.benchmarks.irregular_frequency_study` from the solver itself, so
neither the documentation nor the committed artifact can drift away from the
code.  Panels are limited to 64 to keep the module under ~30 s.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.analytic import BESSEL_J0_FIRST_ZERO, BESSEL_J1_FIRST_ZERO
from tensorbem.benchmarks.irregular_frequency_study import (
    BESSEL_J2_FIRST_ZERO,
    RESONANCES,
    amplification_band,
    bessel_j,
    dirichlet_zeros,
    locate_resonance,
    nodal_chief_points,
    probe,
    run,
)
from tensorbem.geometry import regular_polygon

RADIUS = 1.0
PANELS = 64
DEPTH = 3.0


@pytest.fixture(scope="module")
def mesh():
    return regular_polygon(PANELS, radius=RADIUS)


@pytest.fixture(scope="module")
def dipole_resonance(mesh):
    """Discrete dipole resonance of the 64-panel mesh (cached for the module)."""
    _, analytic = RESONANCES["j11_dipole"]
    return locate_resonance(mesh, analytic - 0.15, analytic + 0.15, radius=RADIUS)


def test_dirichlet_zeros_match_published_tables():
    """The Bessel zeros are the reference values the study keys on."""
    assert dirichlet_zeros(0, 1)[0] == pytest.approx(BESSEL_J0_FIRST_ZERO, rel=1e-12)
    assert dirichlet_zeros(1, 1)[0] == pytest.approx(BESSEL_J1_FIRST_ZERO, rel=1e-12)
    assert dirichlet_zeros(2, 1)[0] == pytest.approx(BESSEL_J2_FIRST_ZERO, rel=1e-12)
    # second roots, Abramowitz & Stegun table 9.5
    assert dirichlet_zeros(0, 2)[1] == pytest.approx(5.520078110286310, rel=1e-11)
    assert dirichlet_zeros(1, 2)[1] == pytest.approx(7.015586669815618, rel=1e-11)


def test_recurrence_bessel_matches_zero_locations():
    """J_2 built by recurrence must vanish on the J_2 roots (guards the recurrence)."""
    for root in dirichlet_zeros(2, 2):
        assert abs(float(bessel_j(2, root))) < 1e-10
    with pytest.raises(ValueError):
        bessel_j(-1, 2.0)


def test_nodal_chief_points_lie_on_resonant_nodal_rays():
    """Nodal control points must sit on theta = (2p+1) pi / (2m) rays, inside the body."""
    points = nodal_chief_points("j11_dipole", radius=RADIUS)
    assert points.shape == (8, 2)
    # m = 1 -> nodal rays are theta = +-pi/2, i.e. the y axis: x == 0
    assert np.allclose(points[:, 0], 0.0, atol=1e-15)
    assert np.all(np.hypot(points[:, 0], points[:, 1]) < RADIUS)

    quad = nodal_chief_points("j21_quadrupole", radius=RADIUS)
    angles = np.arctan2(quad[:, 1], quad[:, 0])
    # m = 2 -> nodal rays at pi/4 and 3pi/4 (mod pi)
    expected = np.array([np.pi / 4.0, 3.0 * np.pi / 4.0])
    folded = np.mod(np.abs(angles), np.pi)
    delta = np.min(np.abs(folded[:, None] - expected[None, :]), axis=1)
    assert np.all(delta < 1e-12)

    with pytest.raises(ValueError):
        nodal_chief_points("j01_axisymmetric")


def test_discrete_resonance_is_shifted_upwards_and_peaks_harder(mesh, dipole_resonance):
    """The mesh resonance sits ~h^2 above the analytic zero and is much sharper."""
    analytic = RESONANCES["j11_dipole"][1]
    detuning = (dipole_resonance["ka"] - analytic) / analytic
    # inscribed 64-gon: sqrt(area/pi)/R = 0.99840 -> +0.12% in ka, allow 0.05-0.3%
    assert 5.0e-4 < detuning < 3.0e-3
    at_zero = probe(mesh, analytic, "unaugmented", radius=RADIUS)
    assert dipole_resonance["condition_number"] > 10.0 * at_zero.condition_number


def test_unaugmented_solver_is_wrong_at_the_resonance_and_chief_fixes_it(
    mesh, dipole_resonance
):
    """Headline claim: >3% error at resonance, CHIEF restores sub-percent accuracy."""
    ka = dipole_resonance["ka"]
    bare = probe(mesh, ka, "unaugmented", radius=RADIUS)
    chief = probe(mesh, ka, "auto", radius=RADIUS)

    assert bare.potential_relative_error > 0.10  # measured ~0.42
    assert chief.potential_relative_error < 5.0e-3  # measured ~4e-3
    assert bare.force_relative_error > 0.03  # measured ~1.0 (100%)
    assert chief.force_relative_error < 0.01  # measured ~3e-3
    assert chief.condition_number < 50.0
    assert bare.condition_number > 20.0 * chief.condition_number


def test_nodal_chief_placement_leaves_the_cure_inert(mesh, dipole_resonance):
    """Negative control for the solver docstring's nodal-curve caveat."""
    ka = dipole_resonance["ka"]
    bare = probe(mesh, ka, "unaugmented", radius=RADIUS)
    nodal = probe(mesh, ka, "nodal", resonance="j11_dipole", radius=RADIUS)
    chief = probe(mesh, ka, "auto", radius=RADIUS)

    assert nodal.chief_point_count == 8
    # inert: the condition number stays within a factor ~1.5 of the bare system
    assert nodal.condition_number > 0.5 * bare.condition_number
    assert nodal.condition_number > 20.0 * chief.condition_number
    # inert: the error is unchanged to within 1%
    assert nodal.potential_relative_error == pytest.approx(
        bare.potential_relative_error, rel=1e-2
    )
    assert nodal.force_relative_error == pytest.approx(
        bare.force_relative_error, rel=1e-2
    )


def test_off_resonance_solver_is_accurate_and_well_conditioned(mesh):
    """Sanity anchor: away from any resonance the unaugmented BIE is fine."""
    bare = probe(mesh, 1.0, "unaugmented")
    assert bare.potential_relative_error < 1.0e-2
    assert bare.force_relative_error < 1.0e-3
    assert bare.condition_number < 20.0


def test_amplification_band_reports_a_bounded_spike(mesh):
    """The scan must return a bounded, finite window estimate (sampling-limited)."""
    half_width_used = 0.03
    rows = amplification_band(
        mesh,
        RESONANCES["j11_dipole"][1],
        half_width=half_width_used,
        samples=13,
        thresholds=(0.03,),
        radius=RADIUS,
    )
    row = rows[0]
    assert math.isfinite(row["peak_potential_error"]) and row["peak_potential_error"] > 0.0
    assert 0.0 <= row["full_width"] <= 2.0 * half_width_used
    assert 0.0 <= row["relative_half_width"] <= 1.0


def test_study_report_is_self_consistent():
    """The JSON-able report must carry every measured quantity the docs quote."""
    study = run(panel_counts=(64,), half_width=0.15, band_samples=9)
    assert study["bessel_zeros"]["j11"] == pytest.approx(BESSEL_J1_FIRST_ZERO, rel=1e-12)
    for name, entry in study["resonances"].items():
        row = entry["panel_counts"]["64"]
        assert set(row) >= {
            "discrete_resonance",
            "detuning_relative",
            "peak_to_zero_kappa_ratio",
            "at_analytic_zero",
            "at_discrete_resonance",
            "band",
            "amplification",
        }
        variants = {p["variant"] for p in row["at_discrete_resonance"]}
        assert variants == {"unaugmented", "auto"} | (
            {"nodal"} if entry["order"] >= 1 else set()
        )
        for probe_row in row["at_discrete_resonance"]:
            assert math.isfinite(probe_row["potential_relative_error"])
            assert probe_row["potential_relative_error"] >= 0.0