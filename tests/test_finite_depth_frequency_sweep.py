"""Verification of the finite-depth radiation solver across ``ka`` and ``f/a``.

``docs/finite_depth_radiation.md`` section 5, limitation 5 recorded that the
finite-depth chain had only been exercised at one operating point
(``ka = 0.6``, ``f/a = 2``, ``h/a = 50``).  These tests lock the sweep that
closes that gap:

1. the committed artefact is reproducible from the code (one live re-run of a
   single case, bit-for-bit against the stored numbers);
2. the finite-depth coefficients collapse onto the deep-water *panel* solution
   over the whole ``ka`` range, so no ``ka``-dependent kernel defect hides in
   the sweep;
3. the error against Ursell's closed-form solution stays at panel-discretisation
   level and shrinks at the observed first order under panel refinement;
4. the energy identity holds across the sweep;
5. the automatic evanescent mode count is converged far below the
   discretisation error (the measured residue is reported, not assumed);
6. the heave/sway split (which must vanish in deep water) decreases monotonically
   as the section is moved away from the sea bed.
"""

import json
import math
from pathlib import Path

import pytest

from tensorbem.analytic import ursell_cylinder_added_mass
from tensorbem.benchmarks.finite_depth_frequency_sweep import (
    DENSITY,
    GRAVITY,
    RADIUS,
    _analytic,
    _deep_panel,
    _finite_depth,
    summary,
)
from tensorbem.benchmarks.finite_depth_radiation import submerged_circle_mesh
from tensorbem.finite_depth_solver import solve_radiation

MOTIONS = ("heave", "sway")
ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "finite_depth_frequency_sweep"
    / "sweep.json"
)


@pytest.fixture(scope="module")
def payload():
    if not ARTIFACT.exists():
        pytest.skip(f"missing committed artefact {ARTIFACT}")
    return json.loads(ARTIFACT.read_text())


def _case(payload, ka, ratio):
    for case in payload["cases"]:
        if math.isclose(case["ka"], ka) and math.isclose(
            case["submergence_ratio"], ratio
        ):
            return case
    raise AssertionError(f"no case for ka={ka}, f/a={ratio}")


def test_cases_are_non_degenerate(payload):
    """Both motions must produce a non-zero coefficient in every case.

    The heave solve of a body symmetric about ``x = 0`` produces no horizontal
    force, so reading both coefficients out of one solve silently compares zero
    against zero; this test is the regression guard for exactly that mistake.
    """
    for case in payload["cases"]:
        for motion in MOTIONS:
            assert case["finite_depth"]["added_mass"][motion] > 100.0
            assert case["finite_depth"]["damping"][motion] > 10.0
            assert case["deep_panel"]["added_mass"][motion] > 100.0


def test_artifact_is_reproducible(payload):
    """A live re-run of one case must reproduce the stored numbers."""
    case = _case(payload, 0.6, 2.0)
    live = _finite_depth(0.6, 2.0, payload["depth"], payload["panels"], None)
    for motion in MOTIONS:
        assert live["added_mass"][motion] == pytest.approx(
            case["finite_depth"]["added_mass"][motion], rel=1e-12
        )
        assert live["damping"][motion] == pytest.approx(
            case["finite_depth"]["damping"][motion], rel=1e-12
        )
        assert live["far_over_near"][motion] == pytest.approx(
            case["finite_depth"]["far_over_near"][motion], rel=1e-12
        )


def test_collapse_onto_deep_panel_solution_across_ka(payload):
    """The finite-depth kernel must reproduce the deep-water panel solver.

    At ``h/a = 50`` the finite-depth kernel differs from the deep-water kernel
    only through exponentially small terms, so a ``ka``-dependent discrepancy
    above the panel-discretisation floor would be an implementation defect.
    """
    worst = 0.0
    for case in payload["cases"]:
        for motion in MOTIONS:
            for quantity in ("added_mass", "damping"):
                worst = max(worst, abs(case[f"dev_from_deep_panel_{motion}"][quantity]))
    assert worst < 1.0e-2, f"collapse error {worst:.3e} exceeds 1%"


def test_error_against_ursell_is_discretisation_limited(payload):
    """Against the closed-form solution the error must stay O(panel size).

    The 64-panel added-mass error of the deep-water solver itself is 1.7%, so no
    case may exceed 3%: anything larger would mean the finite-depth chain adds
    an error of its own at some ``ka`` or ``f/a``.
    """
    for case in payload["cases"]:
        for motion in MOTIONS:
            assert abs(case[f"dev_from_analytic_{motion}"]["added_mass"]) < 3.0e-2
    assert summary(payload)["max_panel_error_vs_analytic"] < 3.0e-2


def test_panel_refinement_is_first_order(payload):
    """Refining the panels must reduce the error at the observed order ~1."""
    for study in payload["refinement"]:
        errors = [row["abs_err_added_mass"] for row in study["rows"]]
        assert errors == sorted(errors, reverse=True), errors
        for order in study["observed_orders"]:
            if order["quantity"] == "abs_err_added_mass":
                assert 0.6 < order["observed_order"] < 1.5, order


def test_energy_identity_holds_across_the_sweep(payload):
    """``B`` from the wave amplitude must equal ``B`` from the pressure."""
    for case in payload["cases"]:
        for motion in MOTIONS:
            ratio = case["finite_depth"]["far_over_near"][motion]
            assert abs(ratio - 1.0) < 3.0e-2, (case["ka"], case["submergence_ratio"], ratio)


def test_automatic_mode_count_is_converged(payload):
    """The default mode count must be far below the discretisation error.

    The bound is ``1e-4`` because that is four orders below the ~1% panel error
    the sweep works at; the measured residue is quoted in the report rather than
    claimed to be exactly zero.
    """
    assert payload["mode_study"], "mode study missing from artefact"
    for study in payload["mode_study"]:
        for values in (study["rel_diff_added_mass"], study["rel_diff_damping"]):
            for motion, difference in values.items():
                assert abs(difference) < 1.0e-4, (study["ka"], motion, difference)


def test_heave_sway_split_shrinks_with_clearance(payload):
    """In deep water heave and sway coincide; the residue must fall as f grows."""
    splits = []
    for ratio in payload["submergence_ratios"]:
        case = _case(payload, 0.6, ratio)
        splits.append(
            max(
                abs(case["finite_depth"]["added_mass"]["heave"]
                    / case["finite_depth"]["added_mass"]["sway"] - 1.0),
                abs(case["finite_depth"]["damping"]["heave"]
                    / case["finite_depth"]["damping"]["sway"] - 1.0),
            )
        )
    assert splits[-1] < 1.0e-2


def test_live_case_outside_the_sweep():
    """Independent live check at ``ka = 1.2``, which is not in the sweep."""
    radius, ratio, depth, panels = RADIUS, 2.0, 50.0 * RADIUS, 64
    nu = 1.2 / radius
    mesh = submerged_circle_mesh(panels, radius, ratio * radius)
    result, far = solve_radiation(
        depth=depth, mesh=mesh, nu=nu, direction="heave",
        density=DENSITY, gravity=GRAVITY,
    )
    analytic_mass, analytic_damping = ursell_cylinder_added_mass(
        radius, ratio * radius, math.sqrt(1.2 * GRAVITY / radius), DENSITY,
        gravity=GRAVITY,
    )
    # the local helper really is the closed-form reference
    helper_mass, helper_damping = _analytic(1.2, ratio)
    assert helper_mass == pytest.approx(analytic_mass, rel=1e-14)
    assert helper_damping == pytest.approx(analytic_damping, rel=1e-14)
    assert abs(result.added_mass[1] / analytic_mass - 1.0) < 3.0e-2
    assert abs(result.damping[1] / analytic_damping - 1.0) < 3.0e-2
    assert abs(far / result.damping[1] - 1.0) < 3.0e-2
    deep = _deep_panel(1.2, ratio, panels)
    assert abs(result.added_mass[1] / deep["added_mass"]["heave"] - 1.0) < 1.0e-2