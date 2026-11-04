"""Guards the Kramers-Kronig causality audit of the Ursell cylinder.

``tensorbem.benchmarks.kkr_causality`` sweeps a frequency band, pushes the
``(A, B)`` pairs of several solutions through the Kramers-Kronig transform of
:mod:`tensorbem.causality`, and reports a *causality residual* next to the
ordinary error against Ursell's analytical solution.  Before this module the
benchmark had no test at all: its three published claims -- the transform's
floor, the high-frequency added-mass limit that the transform requires, and the
ordering of the panel residuals against that floor -- were prose.

The claims pinned here, all from the analytical side of the audit so they stay
cheap:

* the ``1/(k a)`` extrapolation of the infinite-frequency added mass agrees with
  an independent wide-tail fit (``k a`` up to 120) to better than 0.5 %;
* the residual of the *exactly causal* multipole pair is dominated by band
  truncation: it falls by an order of magnitude from ``k a_max = 1`` to
  ``k a_max = 30``, and refining the samples on a fixed narrow band does not
  reproduce that fall;
* a deliberately non-causal gain on the damping is still detected, above the
  floor, at the widest band;
* 40 multipole terms reproduce the 80-term values over the whole audited band,
  which is what makes the cheap reference sweep legitimate.

One slower test runs the benchmark itself and pins the statement the audit
actually makes about the panels: at the default band the panel residual sits
*below* the analytical floor, i.e. it is not resolvable, so the audit may not
claim the panels are causal.

Run: ``pytest tests/test_kkr_causality.py -q`` (the panel test is ``slow``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorbem.analytic import ursell_cylinder_added_mass
from tensorbem.benchmarks import kkr_causality
from tensorbem.causality import kkr_check

DENSITY = kkr_causality._DENSITY
GRAVITY = kkr_causality._GRAVITY
RADIUS = 1.0
SUBMERGENCE = 2.0


def _pair(ka_max: float, points: int, ka_min: float = 0.05):
    from tensorbem.benchmarks.kkr_causality import _analytic_pair, _sweep

    ka, omega = _sweep(ka_min, ka_max, points, RADIUS, GRAVITY)
    mass, damping = _analytic_pair(omega, RADIUS, SUBMERGENCE, DENSITY, GRAVITY)
    return ka, omega, mass, damping


@pytest.fixture(scope="module")
def floor_ladder():
    """Truncation ladder of the exactly causal reference pair."""
    return kkr_causality.truncation_floor(
        RADIUS,
        SUBMERGENCE,
        DENSITY,
        GRAVITY,
        points=24,
        bands=kkr_causality.TRUNCATION_BANDS,
        band_of_interest=3.0,
    )


def test_series_terms_are_sufficient_over_the_audited_band():
    """40 terms must reproduce 80 terms, else the cheap sweep is not valid."""
    worst = 0.0
    for ka in np.geomspace(0.05, 30.0, 12):
        omega = math.sqrt(float(ka) * GRAVITY / RADIUS)
        coarse = ursell_cylinder_added_mass(
            RADIUS, SUBMERGENCE, omega, DENSITY, gravity=GRAVITY, terms=40
        )
        exact = ursell_cylinder_added_mass(
            RADIUS, SUBMERGENCE, omega, DENSITY, gravity=GRAVITY, terms=80
        )
        worst = max(
            worst,
            abs(coarse[0] - exact[0]) / abs(exact[0]),
            abs(coarse[1] - exact[1]) / max(abs(exact[1]), 1e-12),
        )
    assert worst < 1.0e-10, f"40-term series is not converged at ka<=30: {worst:.3e}"


def test_high_frequency_limit_agrees_with_independent_wide_tail():
    """The A_inf that the transform needs must not be an extrapolation artefact."""
    limit, residual = kkr_causality.ursell_high_frequency_limit(
        RADIUS, SUBMERGENCE, DENSITY, GRAVITY
    )
    wide_ka = np.array([30.0, 60.0, 120.0])
    wide = np.array(
        [
            ursell_cylinder_added_mass(
                RADIUS,
                SUBMERGENCE,
                math.sqrt(ka * GRAVITY / RADIUS),
                DENSITY,
                gravity=GRAVITY,
            )[0]
            for ka in wide_ka
        ]
    )
    design = np.column_stack((np.ones_like(wide_ka), 1.0 / wide_ka))
    coefficients, *_ = np.linalg.lstsq(design, wide, rcond=None)
    wide_limit = float(coefficients[0])
    assert residual < 1.0, f"1/(ka) model does not fit the tail: residual {residual:.3f}"
    assert abs(limit - wide_limit) / wide_limit < 5.0e-3
    # Rigid-wall sanity: the free surface stiffens with frequency, so the limit
    # exceeds the value at the widest sampled ka but stays below the isolated
    # circle's infinite-fluid added mass rho*pi*a^2.
    isolated = DENSITY * math.pi * RADIUS**2
    assert wide[-1] < limit < isolated


def test_truncation_floor_falls_with_band_width(floor_ladder):
    """The floor is truncation driven: widening the band removes it.

    It is *not* a sample-count effect: at a fixed narrow band the residual
    wanders between 0.19 and 0.31 for 16-64 samples.  The two regimes meet at
    ``k a_max ~ 10``; beyond that the interpolation error over the span is
    larger than the tail still left out, so the ladder saturates at a few
    percent instead of going to zero.
    """
    bands = floor_ladder["bands"]
    assert [row["ka_max"] for row in bands] == [1.0, 3.0, 10.0, 30.0]
    damping = [row["max_damping_relative_error"] for row in bands]
    assert damping[0] > 0.5
    assert damping[1] < 0.5 * damping[0]
    assert damping[2] < 0.5 * damping[1]
    assert damping[3] < 1.5 * damping[2]
    assert damping[3] < 0.08
    mass = [row["max_added_mass_relative_error"] for row in bands]
    assert mass[2] < mass[0]
    assert mass[3] < 0.02
    # ... and the row reported for the audited band is that band's row.
    assert floor_ladder["floor_at_band"]["ka_max"] == 3.0
    assert floor_ladder["floor_at_band"]["max_damping_relative_error"] == pytest.approx(
        damping[1]
    )


def test_refinement_does_not_reproduce_the_truncation_floor():
    """Fixed-band refinement leaves the residual where truncation put it."""
    narrow, wide = [], []
    for points in (16, 48):
        _, omega, mass, damping = _pair(3.0, points)
        narrow.append(kkr_check(omega, mass, damping).max_damping_relative_error)
        _, omega, mass, damping = _pair(30.0, points)
        wide.append(kkr_check(omega, mass, damping).max_damping_relative_error)
    # On the narrow band the three-fold sample increase is not a three-fold
    # error reduction; the wide band converges much closer.
    assert min(narrow) / max(narrow) > 0.5
    assert wide[-1] < 0.5 * narrow[-1]


def test_non_causal_gain_is_detected_above_the_floor():
    """The audit must see a 50 % damping gain at the widest band."""
    limit, _ = kkr_causality.ursell_high_frequency_limit(
        RADIUS, SUBMERGENCE, DENSITY, GRAVITY
    )
    floor = kkr_causality.truncation_floor(
        RADIUS,
        SUBMERGENCE,
        DENSITY,
        GRAVITY,
        points=24,
        bands=(30.0,),
        band_of_interest=30.0,
    )["floor_at_band"]["max_damping_relative_error"]
    _, omega, mass, damping = _pair(30.0, 24)
    clean = kkr_check(omega, mass, damping, added_mass_inf=limit)
    corrupted = kkr_check(omega, mass, damping * 1.5, added_mass_inf=limit)
    assert clean.max_damping_relative_error == pytest.approx(floor, rel=1e-9)
    assert corrupted.max_damping_relative_error > 3.0 * floor


@pytest.mark.slow
def test_benchmark_reports_the_floor_and_the_panel_residual_is_not_resolvable():
    """The audit's own output must carry the floor it is read against."""
    record = kkr_causality.run(
        panel_counts=(32,),
        points=16,
        include_3d=False,
        floor_points=12,
    )
    floor = record["analytic_truncation_floor"]
    assert floor["floor_at_band"]["ka_max"] == pytest.approx(record["ka_max"])
    panel = record["cases"]["panels_32"]
    panel_damping = panel["kkr_with_limit_added_mass"]["max_damping_relative_error"]
    reference = record["cases"]["analytic_multipole"]
    reference_damping = reference["kkr_with_limit_added_mass"]["max_damping_relative_error"]
    # The reference residual is the floor, and at the default band it is large.
    assert reference_damping >= 0.1
    assert panel_damping < reference_damping
    # The panel solution is still accurate against Ursell itself, which is the
    # claim the audit *can* make at this band.
    assert panel["added_mass_relative_error_max"] < 0.1
    assert panel["damping_relative_error_max"] < 0.2
    assert record["added_mass_limit_over_displaced_mass"] < 1.0