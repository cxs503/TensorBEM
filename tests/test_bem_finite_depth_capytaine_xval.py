"""Regression lock for the finite-depth TensorBEM/capytaine cross-validation.

The artifact ``benchmarks/verified/bem_finite_depth_capytaine_xval/xval_results.json``
is assembled by ``assemble_xval.py`` from the per-case cache written by
``run_cases.py``.  capytaine is **not** a test dependency: the numbers are frozen
into the JSON.

What these tests assert
-----------------------
* the *measured* state of the pre-registered gates -- G3 (capytaine
  self-convergence) passes, G1/G2/G5 fail by the recorded margins.  The failing
  gates are pinned, not papered over: the numbers below are the evidence for the
  open defect tracked in the report.
* the study status is ``partial`` and the blocking defects are written down, so a
  later round cannot silently claim a complete cross-validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT_FILE = (
    REPO_ROOT / "benchmarks" / "verified" / "bem_finite_depth_capytaine_xval"
    / "xval_results.json"
)


@pytest.fixture(scope="module")
def xval_data():
    if not RESULT_FILE.exists():
        pytest.skip(f"{RESULT_FILE} not found; run run_cases.py + assemble_xval.py")
    return json.loads(RESULT_FILE.read_text())


class TestGates:
    """Measured state of the pre-registered gates (2026-11 round)."""

    def test_g3_capytaine_self_convergence_passes(self, xval_data):
        gate = xval_data["comparison"]["gates"]["G3_cpt_self_conv_am"]
        assert gate["n"] > 0
        assert gate["passed"], gate
        assert gate["max_error"] < 0.05

    @pytest.mark.parametrize("name,lo,hi,n", [
        ("G1_tbem_cpt_am_heave", 0.10, 0.13, 16),
        ("G2_tbem_cpt_b_heave", 0.30, 0.36, 16),
        ("G5_dw_limit_am", 0.08, 0.09, 5),
    ])
    def test_known_failing_gates_are_pinned(self, xval_data, name, lo, hi, n):
        """The gate fails, and it fails by exactly the recorded margin."""
        gate = xval_data["comparison"]["gates"][name]
        assert gate["n"] == n
        assert not gate["passed"]
        assert lo <= gate["max_error"] <= hi, gate

    def test_g4_not_evaluated(self, xval_data):
        gate = xval_data["comparison"]["gates"]["G4_tbem_self_conv_am"]
        assert gate["n"] == 0
        assert not gate["passed"]
        assert "not evaluated" in gate["note"]


class TestAddMassBias:
    """TensorBEM A33 is low by a nearly frequency-independent margin."""

    def test_bias_is_frequency_independent_at_h12(self, xval_data):
        rows = [r for r in xval_data["comparison"]["comparison"]
                if r["depth_ratio"] == 12.0 and "am_rel_error" in r]
        assert len(rows) == 5
        errs = [r["am_rel_error"] for r in rows]
        assert all(0.08 < e < 0.13 for e in errs), errs
        # spread across the ka grid is small -> a geometry/discretisation bias,
        # not a kernel or phase error
        assert max(errs) - min(errs) < 0.04

    def test_bias_sign_flips_with_depth(self, xval_data):
        """The error is *not* a one-sided area deficit.

        At h/R = 3, 5, 20 TensorBEM is low by 5-8 %; at h/R = 12 it is high by
        9-12 %.  A pure flat-panel area deficit cannot change sign, so the
        depth-dependent part is an evanescent-mode truncation error -- which is
        why G1 is pinned as a failure rather than explained away.
        """
        rows = [r for r in xval_data["comparison"]["comparison"] if "am_rel_error" in r]
        assert len(rows) == 16
        high = [r for r in rows if r["tbem_am_heave"] > r["cpt_am_heave"]]
        assert {r["depth_ratio"] for r in high} == {12.0}
        assert len(high) == 5


class TestAnalyticAnchor:
    """The infinite-depth multipole reference is evaluated, not transcribed."""

    def test_analytic_deep_water_added_mass(self, xval_data):
        ana = xval_data["analytic_infinite_depth"]
        assert "error" not in ana, ana.get("error")
        # A33' -> 0.5 (Lamb 1932 art. 92) as ka -> 0 for a deeply submerged sphere
        a03 = ana["0.3"]["added_mass_heave"]
        displaced = ana["0.3"]["displaced_volume"] * 1025.0
        assert 0.5 < a03 / displaced < 0.60, a03 / displaced

    def test_capytaine_deep_water_vs_analytic(self, xval_data):
        """Only ka = 0.3 has a deep-water capytaine station in the cache."""
        ana = xval_data["analytic_infinite_depth"]
        err = ana["0.3"]["am_rel_error_dimful_vs_cpt_inf"]
        assert err is not None and err < 0.10, err

    def test_tbem_is_low_against_the_analytic_anchor(self, xval_data):
        """At the deepest station TensorBEM sits below the closed-form value.

        ka = 1.0: analytic A33 = 1923.8 kg, TensorBEM (h/R = 20) = 1809.5 kg
        (-5.9 %), capytaine (h/R = 20) = 1973.5 kg (+2.6 %).  Two numerical
        codes bracket the analytic value, so neither discretisation is
        "obviously right"; TensorBEM is the one that is low.
        """
        ana = xval_data["analytic_infinite_depth"]["1.0"]["added_mass_heave"]
        rows = [r for r in xval_data["comparison"]["comparison"]
                if r["depth_ratio"] == 20.0 and r["ka"] == 1.0]
        assert len(rows) == 1
        assert rows[0]["tbem_am_heave"] < ana < rows[0]["cpt_am_heave"]


class TestStudyStatus:
    """A partial study must say so."""

    def test_status_partial(self, xval_data):
        assert xval_data["study_status"]["state"] == "partial"

    def test_blocking_defects_recorded(self, xval_data):
        defects = xval_data["study_status"]["blocking_defects"]
        assert len(defects) >= 2
        assert any("mode" in d or "eigenseries" in d for d in defects)

    def test_degenerate_station_is_flagged(self, xval_data):
        """h/R = 3 puts the sphere exactly on the sea bed: zero clearance."""
        assert xval_data["geometry"]["seabed_clearance_m"]["3.0"] == 0.0

    def test_panel_counts_are_the_real_ones(self, xval_data):
        assert xval_data["parameters"]["tbem_panels"] == {"2": 320, "3": 1280}
        assert xval_data["parameters"]["cpt_faces"]["24x48"] == 1152
