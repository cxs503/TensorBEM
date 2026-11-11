"""The m13787 report cannot drift away from the code (m15140 round 2).

``dev_report_m13787_ellipsoid_kochin_verification.md`` was merged as PR#622, sat
frozen through the m15106 accuracy envelope and the m14562 benchmark driver, and
came back as a closed PR#658.  Its §4 tables quote a ``< 10 %`` green tick for
the ``panels(source)`` route that the tree does not meet, and several literals
are mislabelled rows of another study.

These tests close that hole mechanically:

* the committed artifact ``docs/m15140/m13787_revalidation.json`` must equal a
  **fresh** :func:`tensorbem.benchmarks.m13787_revalidation.build`
  (bit-for-bit: the module is deterministic), so no number can go stale;
* the auto-generated §9 block of the report must equal
  :func:`render_markdown` of that artifact, so the prose cannot diverge from the
  judged numbers;
* the judgements themselves are pinned: a row that misses the criterion *the
  report itself quoted* must be ``withdrawn`` (not quietly re-thresholded), and
  the mislabelled literals must keep pointing at the ``(route, k, n)`` that
  actually produces them;
* the geometry gate (``Cb`` vs ``pi/6``) is checked before any hydrodynamics,
  per the platform rule.

Nothing here is a hand-transcribed number: every assertion reads the artifact,
which is itself rebuilt from the solver.
"""

from __future__ import annotations

import importlib
import json
import math
import pathlib
import re
import sys

import pytest

from tensorbem.benchmarks import m13787_revalidation as rev

ROOT = pathlib.Path(__file__).resolve().parents[1]
SIBLING_TEST = ROOT / "tests" / "test_ellipsoid_kochin_three_route.py"

ARTIFACT = rev.ARTIFACT
REPORT = rev.REPORT


@pytest.fixture(scope="module")
def art() -> dict:
    assert ARTIFACT.exists(), f"missing committed artifact {ARTIFACT}"
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def report_text() -> str:
    assert REPORT.exists(), f"missing report {REPORT}"
    return REPORT.read_text(encoding="utf-8")


def _row(art: dict, key: str) -> dict:
    matches = [row for row in art["rows"] if row["key"] == key]
    assert len(matches) == 1, f"expected exactly one row {key!r}, got {len(matches)}"
    return matches[0]


# ---------------------------------------------------------------------------
# Geometry gate: verify Cb / L / B before trusting any hydrodynamic number
# ---------------------------------------------------------------------------


class TestGeometryGate:
    def test_principal_dimensions(self, art):
        gate = art["geometry_gate"]
        assert gate["L"] == pytest.approx(10.0)  # 2a
        assert gate["B"] == pytest.approx(2.0)  # 2b
        assert gate["T"] == pytest.approx(0.5)  # c

    def test_block_coefficient_matches_the_half_ellipsoid(self, art):
        gate = art["geometry_gate"]
        assert gate["Cb"] == pytest.approx(math.pi / 6.0, rel=1.0e-12)
        assert gate["Cb_rel_err"] < 1.0e-14
        assert gate["pass"] is True

    def test_thin_ship_parameter_sets_the_modelling_error(self, art):
        gate = art["geometry_gate"]
        assert gate["thinness_b_over_a"] == pytest.approx(0.2, rel=1.0e-12)
        # O((b/a)^2) = 4 % is the floor of the panels-vs-exact disagreement
        assert gate["thinness_b_over_a"] ** 2 == pytest.approx(0.04, rel=1.0e-12)


# ---------------------------------------------------------------------------
# The artifact and the report §9 must both be reproducible from the code
# ---------------------------------------------------------------------------


class TestArtifactIsCurrent:
    def test_artifact_equals_a_fresh_recomputation(self, art):
        """Deterministic module -> the committed JSON must reproduce exactly."""
        fresh = rev.build()
        assert json.loads(json.dumps(fresh)) == art

    def test_check_entry_point_reports_green(self):
        ok, message = rev.check()
        assert ok, message

    def test_report_section_9_equals_the_rendered_block(self, art, report_text):
        block = rev.render_markdown(art).rstrip("\n")
        assert block in report_text, "§9 drifted: rerun --write"
        assert report_text.count(rev.BEGIN) == 1
        assert report_text.count(rev.END) == 1

    def test_report_carries_the_round_2_banner(self, report_text):
        assert "Round 2 说明（m15140）" in report_text
        assert "1 项 withdrawn" in report_text


# ---------------------------------------------------------------------------
# The judgements: what the old report claimed vs what the code does
# ---------------------------------------------------------------------------


class TestJudgements:
    def test_every_declared_row_has_a_verdict(self, art):
        assert len(art["rows"]) == len(rev.DECLARED)
        for row in art["rows"]:
            assert row["verdict"] in {"reproduced", "stale", "withdrawn"}
            assert math.isfinite(row["recomputed"])
            assert row["recomputed"] >= 0.0

    def test_withdrawn_rows_really_miss_their_own_criterion(self, art):
        """A withdrawn ✅ must be provably false, not just relabelled."""
        withdrawn = [row for row in art["rows"] if row["verdict"] == "withdrawn"]
        assert withdrawn, "the 10 % panels(source) claim should be withdrawn"
        for row in withdrawn:
            assert row["criterion"] is not None
            assert row["recomputed"] > row["criterion"], row["key"]

    def test_panels_source_claim_is_the_withdrawn_one(self, art):
        row = _row(art, "panels_source_in_envelope_max")
        assert row["verdict"] == "withdrawn"
        assert row["criterion"] == pytest.approx(rev.REPORTED_PANEL_BUDGET)
        # 16.1 % in-envelope, i.e. the report's "< 10 %" tick was wrong
        assert row["recomputed"] == pytest.approx(0.1614, abs=5.0e-3)

    def test_current_budget_is_met_inside_the_envelope(self, art):
        metrics = art["metrics"]
        assert metrics["panels_source_in_envelope_max"] <= rev.CURRENT_PANEL_BUDGET
        assert metrics["panels_integrated_in_envelope_max"] <= rev.CURRENT_PANEL_BUDGET
        assert art["summary"]["in_envelope_panels_meet_current_budget"] is True

    def test_ordinate_error_is_an_out_of_envelope_limit(self, art):
        """The 30.2 % ordinate figure is a k a = 20 discretisation limit."""
        metrics = art["metrics"]
        assert metrics["panels_ordinate_all_max"] == pytest.approx(0.302, abs=5.0e-3)
        assert metrics["panels_ordinate_all_argmax_k"] * art["geometry_gate"]["a"] > (
            rev.K_A_ENVELOPE
        )
        assert metrics["panels_ordinate_in_envelope_max"] < 0.05

    def test_exact_routes_agree_at_roundoff(self, art):
        """analytic vs quadrature is the load-bearing cross-validation claim."""
        metrics = art["metrics"]
        assert metrics["quadrature_magnitude_max"] < 1.0e-10
        assert metrics["quadrature_phase_max_deg"] < 1.0e-10

    def test_resistance_quadrature_is_the_same_integral(self, art):
        """R_q and R_a share the Kelvin integral, so they agree to round-off."""
        assert art["metrics"]["resistance_quadrature_max"] < 1.0e-10

    def test_resistance_panels_is_within_the_modelling_error(self, art):
        value = art["metrics"]["resistance_panels_max"]
        assert value < rev.CURRENT_PANEL_BUDGET
        assert value < 0.02  # actually ~1 %: much better than the quoted 8.5 %


class TestProvenanceOfStaleLiterals:
    """Where the mislabelled literals actually come from."""

    def test_source_max_literal_is_the_k03_row(self, art):
        row = _row(art, "panels_source_in_envelope_max")
        src = row["declared_traced_to"]
        assert src is not None, "0.073 should be traceable"
        assert src["route"] == "panels_source"
        assert src["k"] == pytest.approx(0.3)
        assert src["n_r"] == 32

    def test_source_integrated_literal_is_another_route_study_row(self, art):
        row = _row(art, "panels_integrated_in_envelope_max")
        src = row["declared_traced_to"]
        assert src is not None, "0.081 should be traceable"
        assert src["route"] == "panels_source"  # not source_integrated
        assert src["n_r"] == 64
        assert src["k"] == pytest.approx(1.0)

    def test_resistance_literals_are_not_traced_to_the_kochin_sweep(self, art):
        """Honest 'not traced': these literals cannot be reproduced at all."""
        for key in ("resistance_quadrature_max", "resistance_panels_max"):
            assert _row(art, key)["declared_traced_to"] is None


# ---------------------------------------------------------------------------
# Convergence: what the refinement studies actually show
# ---------------------------------------------------------------------------


class TestConvergence:
    def test_panels_study_is_monotone(self, art):
        curves = art["convergence"]["panels"]
        assert [row["n_r"] for row in curves] == list(rev.CONVERGENCE_PANELS)
        errors = [row["rel_err"] for row in curves]
        assert all(later < earlier for earlier, later in zip(errors, errors[1:]))
        assert errors[0] == pytest.approx(0.632, abs=5.0e-3)  # not the quoted 15 %
        assert errors[-1] == pytest.approx(0.081, abs=5.0e-3)  # ~ the quoted 7 %

    def test_quadrature_study_is_at_the_roundoff_floor(self, art):
        """Both endpoints are ~1e-15, so '0.5 % -> 0.01 %' is not reproducible."""
        curves = art["convergence"]["quadrature"]
        assert [row["n_psi"] for row in curves] == list(rev.CONVERGENCE_NPSI)
        assert art["summary"]["quadrature_at_roundoff"] is True
        assert all(row["rel_err"] < 1.0e-12 for row in curves)

    def test_sibling_test_no_longer_asserts_noise_monotonicity(self):
        """The m15140 defect fix: the flaky round-off monotonicity is gone."""
        source = SIBLING_TEST.read_text(encoding="utf-8")
        block = source.split("def test_quadrature_converges_to_analytic", 1)[1]
        block = block.split("def ", 1)[0]
        assert "errors[-1] < errors[0]" not in block
        assert "all(err < 1.0e-12 for err in errors)" in block


# ---------------------------------------------------------------------------
# The gates in the two files must not drift apart
# ---------------------------------------------------------------------------


def test_envelope_and_budget_match_the_sibling_test():
    """Read the sibling test's constants; a silent re-threshold must fail here."""
    source = SIBLING_TEST.read_text(encoding="utf-8")
    envelope = re.search(r"^K_A_ENVELOPE = ([\d.]+)$", source, re.MULTILINE)
    assert envelope is not None, "sibling test lost its K_A_ENVELOPE constant"
    assert float(envelope.group(1)) == rev.K_A_ENVELOPE

    budgets = {float(value) for value in re.findall(r"rel_err < ([\d.]+)", source)}
    assert rev.CURRENT_PANEL_BUDGET in budgets, (
        f"current panel budget {rev.CURRENT_PANEL_BUDGET} is not the one the "
        f"sibling test asserts ({sorted(budgets)})"
    )


def test_report_has_no_runnable_scratch_path(report_text):
    """The pre-round-2 §7 ran pytest from a worktree that no longer exists.

    The path may still be *named* once, as provenance for its own removal; it
    must not appear in a command a reader would follow.
    """
    assert "cd /root/tp-agent" not in report_text
    for match in re.finditer(r"/root/tp-agent/\S*", report_text):
        tail = report_text[match.end() : match.end() + 160]
        assert "失效" in tail, f"dead scratch path outside the removal note: {match.group()}"


def test_report_test_count_matches_the_sibling_file(report_text):
    """§2 quotes a test count; keep it equal to the number of test functions."""
    source = SIBLING_TEST.read_text(encoding="utf-8")
    count = len(re.findall(r"^    def test_", source, re.MULTILINE))
    assert count == 17, f"sibling test file grew/shrank to {count} tests"
    assert f"{count} 个 pytest" in report_text


def test_module_cli_help_exits_zero():
    """Same shallow contract as tests/test_benchmark_cli_contract.py."""
    import subprocess

    env = {"PYTHONPATH": str(ROOT / "src")}
    completed = subprocess.run(
        [sys.executable, "-m", "tensorbem.benchmarks.m13787_revalidation", "--help"],
        capture_output=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr.decode()
    assert b"--write" in completed.stdout


def test_figure_is_committed():
    assert rev.FIGURE.exists()
    assert rev.FIGURE.stat().st_size > 10_000  # a real 150 dpi plot, not a blank canvas


def test_module_imports_cleanly():
    module = importlib.import_module("tensorbem.benchmarks.m13787_revalidation")
    assert module is rev