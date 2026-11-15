"""Durability gate for the independent audit of ``CASE-BEM-RIGID-BODY-MASS``.

Decision this file supports
---------------------------
The anchor's independent review is produced by a *script* that rewrites the
committed artefacts::

    PYTHONPATH=src python3 scripts/audit_m19935_rigid_body_mass.py

and that command is quoted in the case-pool registry next to the note that
``docs/case_bem_rigid_body_mass/adversarial_audit.json`` is a reviewed artefact.
Until this gate existed, re-running it was **destructive**: the generator
rendered sections 1-5 only, while the committed document had grown a section 6
(the RAO-caller replay addendum of a later change set), so the documented
reproduction command silently deleted 34 lines of committed review text and
reported nothing.  Two later change sets also refactored
``src/tensorbem/rigid_body_mass.py`` with no test that the four probes of the
audit still reproduce.

Two competing hypotheses were on the table:

* H1 "the audit is durable": regenerating is a no-op on a clean checkout and
  the four probes still reproduce the committed numbers after the refactors.
* H2 "the audit has drifted": some probe no longer reproduces, so the published
  ``<3%`` disposition is stale and must be re-derived.

Predeclared thresholds (chosen before running anything here): the review must
match **character for character**, the numbers must agree to ``1e-9`` relative
(they are deterministic on one machine; the tolerance only absorbs summation
order), and the drift check must *fire* on a single mutated number -- a gate
that cannot fail is not evidence.

Outcome: H1 holds, and the generator was repaired so that it stays true.  The
tests below pin (a) byte-identical regeneration, (b) reproductibility of the
four probes, and (c) the discriminating power of the check.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "audit_m19935_rigid_body_mass.py"
OUTDIR = REPO / "docs" / "case_bem_rigid_body_mass"


def _load_audit_module():
    """Import the audit script by path (``scripts/`` is not a package)."""
    spec = importlib.util.spec_from_file_location("audit_m19935_rigid_body_mass", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit():
    return _load_audit_module()


@pytest.fixture(scope="module")
def committed_payload():
    return json.loads((OUTDIR / "adversarial_audit.json").read_text())


@pytest.fixture(scope="module")
def committed_review():
    return (OUTDIR / "ADVERSARIAL_REVIEW.md").read_text()


@pytest.fixture(scope="module")
def live_payload(audit):
    """Recompute the four probes exactly as the documented command does."""
    return audit.main(write=False)


class TestRegenerationIsNonDestructive:
    def test_render_reproduces_the_committed_review(self, audit, committed_payload, committed_review):
        rendered = audit.render_review(committed_payload)
        assert rendered == committed_review, (
            "the documented reproduction command no longer reproduces "
            "docs/case_bem_rigid_body_mass/ADVERSARIAL_REVIEW.md"
        )

    def test_render_ends_with_a_newline(self, audit, committed_payload):
        assert audit.render_review(committed_payload).endswith("\n")

    def test_addendum_survives_regeneration(self, audit, committed_payload):
        """Section 6 was authored in a later change set; the generator must keep it."""
        rendered = audit.render_review(committed_payload)
        assert "## 6. Addendum — the end-to-end claim was not in the verdict (fixed)" in rendered
        assert "rao_caller" in rendered

    def test_legacy_generator_would_have_truncated_the_review(
        self, audit, committed_payload, committed_review
    ):
        """Control: the pre-fix generator (no addendum) does *not* reproduce the file.

        Without this control the equality above could be satisfied by a
        committed document that simply had nothing extra in it.
        """
        addendum_free = "\n".join(
            line for line in audit.render_review(committed_payload).split("\n")
            if not any(line == entry for entry in audit.ADDENDUM_SECTION)
        )
        assert addendum_free != committed_review
        removed = len(committed_review.split("\n")) - len(addendum_free.split("\n"))
        assert removed >= 30, f"expected the addendum to be a substantial section, removed {removed} lines"


class TestProbesStillReproduce:
    def test_payload_matches_committed_numbers(self, audit, live_payload, committed_payload):
        live = {k: v for k, v in live_payload.items() if k != "environment"}
        reference = {k: v for k, v in committed_payload.items() if k != "environment"}
        diffs = audit._walk_diff(reference, live)
        assert diffs == [], "audit probes drifted from the committed artefacts: " + "; ".join(diffs[:5])

    def test_committed_artefacts_pass_the_check_mode(self, audit, live_payload, committed_review):
        committed_payload = json.loads((OUTDIR / "adversarial_audit.json").read_text())
        status = audit.compare_committed(live_payload)
        assert status["review_matches"] is True
        assert status["payload_diff_count"] == 0
        assert status["figure_present"] is True
        assert status["ok"] is True
        assert committed_review == audit.render_review(committed_payload)

    def test_symbolic_hessian_still_closes(self, live_payload):
        """Probe A: Hessian(T) of the kinetic energy equals the block form exactly."""
        sym = live_payload["symbolic_energy_hessian"]
        assert sym["symbolic_difference_is_zero"] is True
        assert sym["symbolic_nonzero_entries_of_difference"] == 0
        scale = sym["hessian_vs_shipped_scale"]
        assert scale > 0.0
        assert sym["hessian_vs_shipped_max_abs"] <= 1e-12 * scale

    def test_quadrature_converges_at_second_order(self, live_payload):
        """Probe B: the 1.4 % anchor figure is panelisation error, not a bias."""
        conv = live_payload["quadrature_convergence"]
        assert conv["log_log_slope"] < -2.5, conv["log_log_slope"]
        accepted = [r for r in conv["levels"] if r["subdivisions"] >= 3]
        assert accepted, conv["levels"]
        assert max(r["worst_entry_rel_error"] for r in accepted) < 0.03

    def test_guard_rejects_every_corrupted_surface(self, live_payload):
        """Probe C: the finding of the review is still enforced by the shipped guard."""
        cases = live_payload["topology_controls"]["cases"]
        assert cases["closed"]["guard_outcome"] == "accepted"
        for corrupted in ("open_patch", "flipped_triangle", "duplicated_triangle"):
            assert cases[corrupted]["guard_outcome"] == "rejected", corrupted
            assert cases[corrupted]["legacy_volume_rel_error"] < 0.03, corrupted
        controls = live_payload["topology_controls"]
        assert controls["post_fix_all_rejected"] is True
        assert controls["max_prefix_silent_rel_error"] < 0.03
        assert controls["prefix_but_inside_tolerance"], controls

    def test_metric_is_not_diluted_by_the_matrix_scale(self, live_payload):
        """Probe D: the reported residual equals the injected entry-relative error."""
        met = live_payload["metric_sensitivity"]
        assert met["reported_matches_definition_residual"] < 1e-9
        coupling = met["coupling_injection"][0]
        assert coupling["reported_max_rel"] > 10.0 * coupling["injected_rel"]


class TestCheckModeHasDiscriminatingPower:
    def test_small_numeric_drift_is_detected(self, audit, live_payload):
        """A 1e-6 relative drift is below the review's 2-decimal text, not below the gate."""
        mutated = json.loads(json.dumps(live_payload))
        mutated["quadrature_convergence"]["log_log_slope"] *= 1.0 + 1e-6
        status = audit.compare_committed(mutated)
        assert status["payload_diff_count"] >= 1
        assert any("log_log_slope" in d for d in status["payload_diffs"])
        assert status["ok"] is False
        # the rendered review formats the slope to two decimals, so only the
        # numeric comparison can see this; that is why both layers are checked.
        assert status["review_matches"] is True

    def test_material_numeric_drift_reaches_the_review_text(self, audit, live_payload):
        mutated = json.loads(json.dumps(live_payload))
        mutated["quadrature_convergence"]["log_log_slope"] *= 1.10
        status = audit.compare_committed(mutated)
        assert status["ok"] is False
        assert status["review_matches"] is False

    def test_changed_review_text_is_detected(self, audit, live_payload, tmp_path, committed_review):
        outdir = tmp_path / "case_bem_rigid_body_mass"
        outdir.mkdir()
        (outdir / "adversarial_audit.json").write_text(
            (OUTDIR / "adversarial_audit.json").read_text())
        (outdir / "fig_adversarial_audit.png").write_bytes(
            (OUTDIR / "fig_adversarial_audit.png").read_bytes())
        (outdir / "ADVERSARIAL_REVIEW.md").write_text(committed_review.replace("<3 %", "<1 %"))
        status = audit.compare_committed(live_payload, outdir=outdir)
        assert status["review_matches"] is False
        assert status["ok"] is False

    def test_missing_artefact_is_detected(self, audit, live_payload, tmp_path, committed_review):
        outdir = tmp_path / "case_bem_rigid_body_mass"
        outdir.mkdir()
        (outdir / "adversarial_audit.json").write_text(
            (OUTDIR / "adversarial_audit.json").read_text())
        (outdir / "ADVERSARIAL_REVIEW.md").write_text(committed_review)
        status = audit.compare_committed(live_payload, outdir=outdir)
        assert status["figure_present"] is False
        assert status["ok"] is False