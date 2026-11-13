"""m17243 — CASE-BEM-RIGID-LID-HEMISPHERE: rigid-lid hemisphere added mass (ka->0).

Pins the benchmark-pool case: the 3-D radiation solver at ka=1e-3 must land
within 5 % of the even-parity rigid-lid endpoint (0.830954148) on the 1024-panel
mesh, and the committed payload must be self-consistent (criteria re-derived
from the raw rows, reference re-derived from the even-mode series).
"""

from __future__ import annotations

import json
import math
import pathlib

import pytest

from tensorbem.benchmarks.rigid_lid_hemisphere_attribution import (
    even_mode_added_mass_dimless,
)

ARTIFACTS = (
    pathlib.Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "verified"
    / "bem_rigid_lid_hemisphere_ka0"
)

TOLERANCE = 0.05


@pytest.fixture(scope="module")
def payload() -> dict:
    path = ARTIFACTS / "results.json"
    assert path.exists(), f"missing artifact: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_reference_rederived_from_series(payload: dict) -> None:
    """The committed reference must equal the even-mode series value."""
    excel = even_mode_added_mass_dimless(degree_max=60, quad_order=400)
    committed = payload["reference"]["rigid_lid_added_mass_dimless"]
    assert abs(committed - excel["rigid_lid_added_mass_dimless"]) < 1e-9
    # the refuted anchor is 1.0; the correct endpoint is well below it
    assert committed < 0.9
    # series is converged
    tail = abs(
        excel["partial_sums_even"][-1] - excel["partial_sums_even"][-2]
    )
    assert tail < 1e-6


def test_headline_within_5pct(payload: dict) -> None:
    """Headline (1024 panels, ka=1e-3) within the task tolerance."""
    head = payload["headline"]
    assert head["panels"] == 1024
    assert head["ka"] == 1e-3
    assert abs(head["rel_error_vs_ref"]) < TOLERANCE
    assert head["within_5pct"] is True


def test_rows_consistent_with_reference(payload: dict) -> None:
    """Every row's rel_error_vs_ref must match its dimless value and the ref."""
    ref = payload["reference"]["rigid_lid_added_mass_dimless"]
    for row in payload["mesh_ladder"] + payload["ka_ladder"]:
        expected = row["added_mass_dimless"] / ref - 1.0
        assert abs(row["rel_error_vs_ref"] - expected) < 1e-12


def test_mesh_ladder_brackets_reference(payload: dict) -> None:
    """256 and 1024 panel values bracket the reference (convergence evidence)."""
    rows = {r["panels"]: r["added_mass_dimless"] for r in payload["mesh_ladder"]}
    ref = payload["reference"]["rigid_lid_added_mass_dimless"]
    assert rows[256] < ref < rows[1024]
    # absolute error decreases monotonically down the ladder
    errs = [abs(r["rel_error_vs_ref"]) for r in payload["mesh_ladder"]]
    assert errs[0] > errs[1] > errs[2]


def test_all_criteria_pass(payload: dict) -> None:
    assert payload["verdict"] == "ALL PASS"
    for key, crit in payload["criteria"].items():
        assert crit["pass"], f"criterion {key} failed"


def test_hulme_crosscheck_bounded(payload: dict) -> None:
    """Independent Hulme (1982) cross-check stays under 1 % over the band."""
    worst = max(
        abs(r["rel_error_vs_hulme"]) for r in payload["ka_ladder"]
    )
    assert worst < 0.01
