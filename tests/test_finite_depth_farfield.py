"""Regression tests for the finite-depth far-field closure.

The gap these tests guard is the one left open by
``docs/finite_depth_radiation.md``: at 64 panels the far-field damping exceeds
the near-field damping by 1.06 %, at every water depth, and the document could
not say whether that is a normalisation defect or panel discretisation.  The
tests below pin the answer:

* a *constant* mis-normalisation (factor 2, group velocity, mode norm,
  ``i*omega/g``) would leave a residual that does **not** shrink under
  refinement, so the closure test refines the mesh and demands the residual
  behave like ``O(1/N)``;
* the far-field projection is cross-checked against a direct evaluation of the
  radiating potential, so a wrong mode shape or norm shows up as a discrepancy
  between two independently computed amplitudes;
* the panel quadrature error of that projection is measured, so that the 1 %
  gap cannot be blamed on the quadrature;
* wrong coefficients are injected to show that the ±2 % tolerance really does
  exclude them.

The heavy, publication-quality run is
``python -m tensorbem.benchmarks.finite_depth_farfield`` (writes
``outputs/finite_depth_farfield/closure.json``); here the panel counts are
small on purpose so the suite stays fast.
"""

import math

import pytest

from tensorbem.benchmarks.finite_depth_farfield import _ursell_reference, run

NU = 0.6          # ka with radius a = 1
SHALLOW = 3.2     # depth ratio h/a, clearance h - f - a = 0.2
FAST_PANELS = (32, 64)


@pytest.fixture(scope="module")
def short_run():
    """One refined pair of meshes at a shallow depth, MFS and deep water off."""
    return run(
        panels=FAST_PANELS,
        depths=(SHALLOW,),
        include_deep_water=False,
        include_mfs=False,
    )


def test_ursell_reference_is_recomputed_not_copied():
    """The deep-water target comes from the analytic module, not a literal."""
    reference = _ursell_reference(NU, 1.0, 1025.0)
    assert reference["added_mass"] == pytest.approx(2545.1672, rel=1.0e-6)
    assert reference["damping"] == pytest.approx(2718.1540, rel=1.0e-6)
    assert reference["omega"] == pytest.approx(math.sqrt(NU * 9.80665), rel=1.0e-12)


def test_closure_residual_is_first_order_and_extrapolates_to_one(short_run):
    """Refinement must shrink |B_far/B_near - 1| like 1/N and extrapolate to 0.

    A constant mis-normalisation of the far field would keep the residual
    constant; the observed order and the Richardson limit are the two numbers
    that distinguish the two explanations.
    """
    entry = short_run["summary"][f"finite_depth@h={SHALLOW:g}"]
    ratios = entry["far_over_near"]
    residuals = [abs(value - 1.0) for value in ratios]
    assert residuals[1] < residuals[0], "refinement did not reduce the closure gap"
    assert 1.5 < residuals[0] / residuals[1] < 2.6, (
        f"closure residual is not first order: ratio of residuals "
        f"{residuals[0] / residuals[1]:.3f}"
    )
    assert 0.7 < entry["observed_order"] < 1.3
    assert entry["extrapolated_ratio"] == pytest.approx(1.0, abs=5.0e-3)


def test_direct_far_field_matches_the_mode_projection(short_run):
    """The projected amplitude equals the one recovered from the Green function.

    The probe sits 12 and 24 evanescent decay lengths downrange, so only the
    propagating mode survives.  Agreement to 1e-3 means the propagating mode
    shape, its norm and the kinematic factor are consistent with the kernel.
    """
    entry = short_run["summary"][f"finite_depth@h={SHALLOW:g}"]
    assert max(entry["direct_amplitude_rel_dev"]) < 1.0e-3


def test_panel_quadrature_is_far_too_small_to_explain_the_gap(short_run):
    """Midpoint vs 16-point Gauss on the projection integral: sub-1e-3."""
    entry = short_run["summary"][f"finite_depth@h={SHALLOW:g}"]
    assert entry["max_quadrature_rel_dev"] < 1.0e-3
    # And the observed gap is an order of magnitude larger, so the quadrature
    # cannot be its origin.
    residuals = [abs(value - 1.0) for value in entry["far_over_near"]]
    assert min(residuals) > 20.0 * entry["max_quadrature_rel_dev"]


def test_wrong_coefficients_are_rejected_by_the_tolerance(short_run):
    """Fault injection: every mis-normalisation leaves the ±2 % band."""
    entry = short_run["summary"][f"finite_depth@h={SHALLOW:g}"]
    variants = entry["wrong_coefficient_ratios"]
    assert variants["correct"] == pytest.approx(entry["far_over_near"][-1], rel=1.0e-12)
    assert abs(variants["half_factor"] - 1.0) > 0.4      # one direction instead of two
    assert abs(variants["phase_velocity"] - 1.0) > 0.2   # c_p instead of c_g
    assert abs(variants["deep_water_cg"] - 1.0) > 0.1    # c_g = omega/(2 k0) in shallow water
    # The correct value is inside the band the identity test uses.
    assert abs(entry["far_over_near"][-1] - 1.0) < 2.0e-2


def test_near_field_energy_balance_is_the_only_reference_needed(short_run):
    """Sanity: the near-field damping is positive and of the right magnitude."""
    cases = [case for case in short_run["cases"] if case["kind"] == "finite_depth"]
    assert len(cases) == len(FAST_PANELS)
    for case in cases:
        assert case["near_damping"] > 0.0
        assert 2000.0 < case["near_damping"] < 3500.0
        assert case["far_damping"] > 0.0