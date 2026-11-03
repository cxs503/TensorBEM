"""Guards for the 3-D per-quantity accuracy envelope.

The envelope is only useful if the claims it makes are executable:

* the damping converges with a much larger error constant than the added mass,
  so a mesh validated on added mass is *not* validated for damping;
* the near-field and far-field damping estimators agree at fixed mesh, i.e. the
  residual error belongs to the discrete potentials and not to either estimator;
* the published envelope JSON still matches the solver.

Runtime is kept at a few seconds: the full sweep lives in
``artifacts/accuracy_envelope/envelope.json``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorbem.benchmarks.accuracy_envelope import (
    _panels_per_wavelength,
    run_sphere_envelope,
    summarise,
)

JSON_PATH = Path(__file__).resolve().parents[1] / "artifacts/accuracy_envelope/envelope.json"


@pytest.fixture(scope="module")
def coarse_sphere() -> list[dict]:
    return run_sphere_envelope(subdivisions=(0, 1), ka_values=(0.8,))


def test_panels_per_wavelength_definition() -> None:
    # radius 1, ka = 1 -> k = 1, lambda = 2 pi; h = 0.2 -> lambda/h = 10 pi
    assert _panels_per_wavelength(1.0, 1.0, 0.2) == pytest.approx(10.0 * math.pi)
    assert _panels_per_wavelength(1.0, 1.0, 0.0) == float("inf")


def test_added_mass_converges_second_order(coarse_sphere: list[dict]) -> None:
    ladder = sorted(coarse_sphere, key=lambda r: r["panels"])
    assert [r["panels"] for r in ladder] == [20, 80]
    for quantity in ("a33_err", "a11_err"):
        coarse, fine = ladder[0][quantity], ladder[1][quantity]
        assert fine < coarse, f"{quantity} did not decrease under refinement"
        order = math.log(coarse / fine) / math.log(ladder[0]["h_max"] / ladder[1]["h_max"])
        assert 1.7 <= order <= 2.4, f"{quantity} observed order {order:.3f} outside [1.7, 2.4]"


def test_damping_is_the_slow_quantity(coarse_sphere: list[dict]) -> None:
    """A mesh that is adequate for added mass is not adequate for damping."""
    row = max(coarse_sphere, key=lambda r: r["panels"])
    assert row["a33_err"] < 0.10
    assert row["b33_near_err"] > 2.0 * row["a33_err"], (
        "heave damping should carry the larger error constant at fixed mesh; "
        f"measured A33 {row['a33_err']:.3e} vs B33 {row['b33_near_err']:.3e}"
    )


def test_near_and_far_field_damping_agree_at_fixed_mesh(coarse_sphere: list[dict]) -> None:
    """Two different estimators of the same continuum quantity.

    Their agreement means the residual damping error is discretisation of the
    potentials, not a defect of either estimation formula.
    """
    row = max(coarse_sphere, key=lambda r: r["panels"])
    reference = row["b33_ref"]
    gap = abs(row["b33_near"] - row["b33_far"]) / reference
    assert gap < 0.02, f"near/far damping disagreement {gap:.3e} of the reference"


def test_summary_extremes_are_derived_from_rows(coarse_sphere: list[dict]) -> None:
    summary = summarise(coarse_sphere)["submerged_sphere"]
    assert summary["finest_panels"] == 80
    worst = summary["worst_error_at_finest_mesh"]
    assert worst["a33_err"] == pytest.approx(
        max(r["a33_err"] for r in coarse_sphere if r["panels"] == 80)
    )
    # 80 panels is not within 3% anywhere on the grid, so no admissible band
    assert summary["largest_ka_with_error_le_3pct_at_finest_mesh"]["b33_near_err"] is None


@pytest.mark.skipif(not JSON_PATH.exists(), reason="envelope JSON not committed")
def test_published_envelope_numbers_are_reproducible() -> None:
    """Pin the published envelope (artifacts/accuracy_envelope/envelope.json)."""
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    rows = data["rows"]
    sphere = {
        (r["panels"], round(r["ka"], 3)): r for r in rows if r["body"] == "submerged_sphere"
    }
    hemi = {
        (r["panels"], round(r["ka"], 3)): r
        for r in rows if r["body"] == "floating_hemisphere"
    }
    assert len(rows) == 58
    # submerged sphere, 320 panels: added mass inside the 3% standard everywhere,
    # damping outside it in the long-wave band.
    assert sphere[(320, 0.8)]["a33_err"] == pytest.approx(1.85e-2, rel=0.05)
    assert sphere[(320, 0.8)]["b33_near_err"] == pytest.approx(5.74e-2, rel=0.05)
    assert sphere[(320, 4.0)]["b33_near_err"] < 0.03
    assert sphere[(320, 0.4)]["b33_near_err"] > 0.03
    # the sphere never exceeds the platform's 15% acceptance line at 320 panels
    for (panels, _ka), row in sphere.items():
        if panels == 320:
            assert row["b33_near_err"] < 0.15, row
            assert row["a33_err"] < 0.15, row
    # floating hemisphere: 64 panels is nowhere near converged, 256 panels is
    assert hemi[(64, 2.0)]["b33_near_err"] > 0.10
    assert hemi[(256, 0.8)]["b33_near_err"] < 0.01
    # near the first interior Dirichlet resonance (ka ~ 2.58) the far-field
    # damping degrades faster than the near-field one
    assert hemi[(256, 2.5)]["b33_far_err"] > 3.0 * hemi[(256, 2.5)]["b33_near_err"]