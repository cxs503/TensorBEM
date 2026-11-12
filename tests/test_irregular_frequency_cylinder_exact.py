"""Three-route verification of the irregular-frequency predictor (m16777).

The pre-existing gate on ``predict_irregular_frequencies`` compared the prediction
with the right-hand side of its own formula ("the prediction IS this analytical
estimate").  These tests replace that with three routes that are not the formula:

* **Route A** -- the *rigid-lid* interior eigenproblem, which the formula solves
  exactly.  Checked by substitution (PDE / wall / lid residuals) and by binding
  the library predictor to it on a real cylinder mesh.
* **Route B** -- the *free-surface-consistent* interior eigenproblem
  (``phi_z = nu phi`` on the interior waterplane), solved exactly for the one
  separable geometry.  Checked against an independent numerical integration of
  the vertical ODE, and against its own deep-draft limit (Route A).
* **Route C** -- the repository's own bare BEM, whose irregular band is located
  by the condition-number peak.  This is the only route that can say whether the
  predictor is *useful*; it is asserted within a loose band and costs a handful of
  small radiation solves.

Route B is expected to disagree with Route A (that is the finding).  The tests
therefore pin the *gap* rather than asserting agreement, so the numbers in
``docs/irregular_frequency_cylinder_exact.md`` cannot drift silently.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorbem.benchmarks.irregular_frequency_cylinder_exact import (
    J01,
    exact_cylinder_modes,
    locate_peak,
    mode_residuals,
    observed_spurious_band,
    residuals_rigid_lid,
    rigid_lid_wavenumbers,
    run_study,
)

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "docs" / "m16777" / "irregular_frequency_cylinder_exact.json"
DOC = REPO / "docs" / "irregular_frequency_cylinder_exact.md"


# ---------------------------------------------------------------------------
# Route A: the formula is the rigid-lid eigenvalue, proved by substitution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", [0, 1, 2])
def test_rigid_lid_mode_satisfies_the_interior_problem(order: int) -> None:
    residuals = residuals_rigid_lid(1.0, order=order)
    assert residuals["pde_residual"] < 1.0e-8
    assert residuals["wall_residual"] < 1.0e-12
    assert residuals["lid_residual"] == 0.0


def test_rigid_lid_wavenumbers_are_the_bessel_zeros_over_radius() -> None:
    wavenumbers = rigid_lid_wavenumbers(2.0)
    assert wavenumbers[0] == pytest.approx(J01 / 2.0, rel=1.0e-15)
    assert np.all(np.diff(wavenumbers) > 0.0)


def test_library_predictor_is_the_rigid_lid_eigenvalue_on_a_cylinder_mesh() -> None:
    """The predictor's number, on a body whose waterplane really is a circle.

    This is the identity claim that makes Route A the right description of the
    library formula; the mesh polygon under-estimates the waterplane area by
    O((2*pi/n)^2), hence the 2 % band.
    """

    from tensorbem.diffraction_drift_maturity import vertical_cylinder_hull_mesh
    from tensorbem.irregular_frequency import predict_irregular_frequencies

    radius = 1.0
    mesh = vertical_cylinder_hull_mesh(radius, 0.5 * radius, n_theta=32, n_levels=3)
    prediction = predict_irregular_frequencies(mesh)
    expected = rigid_lid_wavenumbers(prediction.waterplane_radius)
    for got, want in zip(prediction.predicted_nu, expected):
        assert got == pytest.approx(want, rel=1.0e-12)
    # The polygon waterplane radius is within 2 % of the true radius.
    assert prediction.waterplane_radius == pytest.approx(radius, rel=0.02)


# ---------------------------------------------------------------------------
# Route B: the free-surface-consistent reference, exact for the cylinder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", [0.25, 0.5, 1.0, 2.0, 4.0])
def test_route_b_residuals_vanish(ratio: float) -> None:
    mode = exact_cylinder_modes(1.0, ratio, angular_orders=(0,))[0]
    residuals = mode_residuals(mode, ratio)
    assert residuals["dispersion_residual"] < 1.0e-14
    assert residuals["robin_residual"] < 1.0e-10  # ODE integrated, not tan()
    assert residuals["pde_residual"] < 1.0e-10


def test_route_b_collapses_onto_route_a_as_the_draft_grows() -> None:
    mode = exact_cylinder_modes(1.0, 64.0, angular_orders=(0,))[0]
    assert abs(mode.radial_wavenumber - mode.nu) / mode.nu < 5.0e-3
    assert mode.vertical_wavenumber < 0.05


@pytest.mark.parametrize("ratio", [2.0, 4.0, 8.0, 16.0])
def test_route_b_is_above_route_a_at_deep_drafts(ratio: float) -> None:
    """Any finite draft has ``k_z > 0``, so Route B must exceed Route A."""

    mode = exact_cylinder_modes(1.0, ratio, angular_orders=(0,))[0]
    assert mode.nu > mode.radial_wavenumber
    assert mode.vertical_wavenumber > 0.0


def test_route_b_gap_is_not_monotone_and_that_is_documented() -> None:
    """The reference has a branch structure: the smallest root jumps at ``j T = 1``.

    Honest pinning of the shape of the reference, so nobody can later present the
    gap as a monotone "draft correction" (the falsified hypothesis of m16777).
    """

    gaps = {}
    for ratio in (0.25, 0.5, 1.0, 2.0):
        mode = exact_cylinder_modes(1.0, ratio, angular_orders=(0,))[0]
        gaps[ratio] = (mode.nu - mode.radial_wavenumber) / mode.radial_wavenumber
    assert gaps[0.25] < gaps[0.5]  # non-monotone in T/R
    assert gaps[0.5] > gaps[1.0] > gaps[2.0]  # decreasing again above the switch
    # Branch switch happens where j_{0,1} * T/R crosses 1.
    shallow = exact_cylinder_modes(1.0, 0.25, angular_orders=(0,))[0]
    assert shallow.vertical_wavenumber * 0.25 < 0.75 * math.pi  # first branch


def test_route_b_shallow_limit_is_pi_over_four() -> None:
    """For ``j T << 1`` the first-branch root tends to ``nu T = pi/4``.

    ``tan(k_z T) = k_z/nu`` with ``nu T = sqrt((jT)^2 + (k_z T)^2) -> k_z T`` gives
    ``tan(x) = 1``, i.e. ``x = nu T -> pi/4``.
    """

    mode = exact_cylinder_modes(1.0, 0.05, angular_orders=(0,))[0]
    assert mode.nu * 0.05 == pytest.approx(0.25 * math.pi, rel=0.02)


def test_route_b_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        exact_cylinder_modes(0.0, 1.0)
    with pytest.raises(ValueError):
        exact_cylinder_modes(1.0, 0.0)


# ---------------------------------------------------------------------------
# Route C: the repository's own BEM
# ---------------------------------------------------------------------------


def test_locate_peak_finds_a_synthetic_peak() -> None:
    """Negative control: the locator must not just return the largest sample."""

    grid = np.linspace(2.0, 3.0, 21)
    values = -(grid - 2.633) ** 2 + 5.0
    _, position = locate_peak(grid, values)
    assert position == pytest.approx(2.633, abs=0.01)
    _, flat = locate_peak(grid, np.ones_like(grid))
    assert flat == pytest.approx(grid[0])


def test_empirical_band_tracks_route_a_within_ten_percent() -> None:
    """The bare solve loses conditioning near the rigid-lid eigenvalue, not Route B."""

    band = observed_spurious_band(
        1.0, 1.0, mode=2, nu_min=2.15, nu_max=2.75, n_points=5, n_theta=16, n_levels=3
    )
    route_a = rigid_lid_wavenumbers(1.0)[0]
    route_b = exact_cylinder_modes(1.0, 1.0, angular_orders=(0,))[0].nu
    assert abs(band["nu_observed"] - route_a) / route_a < 0.10
    assert abs(band["nu_observed"] - route_b) / route_b > 0.20
    assert band["max_condition_number"] > 10.0
    assert band["mesh"]["panels"] > 100


# ---------------------------------------------------------------------------
# The published numbers may not drift
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def artifact() -> dict:
    assert ARTIFACT.exists(), f"missing artifact {ARTIFACT}"
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_artifact_is_reproducible_without_the_empirical_route(artifact: dict) -> None:
    recomputed = run_study(empirical_ratios=())
    assert recomputed["draft_ratio_sweep"] == artifact["draft_ratio_sweep"]
    assert (
        recomputed["angular_mode_table_at_T_over_R_0p5"]
        == artifact["angular_mode_table_at_T_over_R_0p5"]
    )


def test_artifact_empirical_rows_are_the_ones_quoted(artifact: dict) -> None:
    empirical = artifact["empirical"]
    assert len(empirical) == 2
    labels = {row["label"] for row in empirical}
    assert labels == {"coarse", "refined"}
    for row in empirical:
        # every empirical row is within 5 % of Route A and > 40 % away from Route B
        assert abs(row["error_vs_route_A"]) < 0.05
        assert abs(row["error_vs_route_B"]) > 0.40
        assert row["passivity_violation"] is True


def test_document_publishes_the_artifact_numbers(artifact: dict) -> None:
    """Every number in the document's headline table comes from the artifact."""

    markdown = DOC.read_text(encoding="utf-8")
    for row in artifact["draft_ratio_sweep"]:
        cell = f"{row['nu_route_B_free_surface']:.4f}"
        assert cell in markdown, cell
    for row in artifact["empirical"]:
        assert f"{row['nu_observed']:.4f}" in markdown, row
    assert "exact by construction" in markdown  # the retired claim is quoted