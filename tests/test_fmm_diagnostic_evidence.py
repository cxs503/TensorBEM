"""Far-fraction diagnostic evidence for :mod:`tensorbem.fmm` (m18196).

The diagnostic capability has two halves, both pinned here:

* ``FMMOperator.pair_partition`` splits every (target, source-leaf) pair into a
  **disjoint** far / near-geometric / near-box bucket, and the far bucket is
  exactly what ``matvec`` hands to the multipole path.  The round-trip test
  instruments ``matvec`` and compares.
* ``docs/fmm_far_fraction_diagnostic/{evidence.json,tables.md,fig_*.png}`` are
  *generated* by ``scripts/fmm_far_fraction_diagnostic.py``.  The last test
  re-runs the generator's row builder on the committed geometry and requires the
  committed numbers to reproduce, so a hand-edited evidence file fails CI.

Every test in this module fails on the pre-m18196 code (``pair_partition`` and
the bucket attribution do not exist there).
"""
from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import sys

import numpy as np
import pytest

from tensorbem import fmm as fmm_mod
from tensorbem.fmm import FMMOperator, PairPartition

ROOT = pathlib.Path(__file__).resolve().parents[1]
DIAG_PATH = ROOT / "scripts" / "fmm_far_fraction_diagnostic.py"
DOCS = ROOT / "docs" / "fmm_far_fraction_diagnostic"


def _load_diagnostic():
    spec = importlib.util.spec_from_file_location("fmm_far_fraction_diagnostic", DIAG_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("fmm_far_fraction_diagnostic", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def diag():
    return _load_diagnostic()


@pytest.fixture(scope="module")
def cases(diag):
    return diag._cases()


@pytest.fixture(scope="module")
def evidence():
    return json.loads((DOCS / "evidence.json").read_text())


# ---------------------------------------------------------------------------
# bucket accounting
# ---------------------------------------------------------------------------

class TestPairPartitionAccounting:
    def test_buckets_are_disjoint_and_complete(self, cases):
        for name in ("surface", "shell", "separated"):
            case = cases[name]
            op = FMMOperator(p=4, max_leaf=case["max_leaf"])
            op.setup(case["sources"])
            part = op.pair_partition(case["targets"])
            assert part.n_targets == len(case["targets"])
            assert part.total_pairs == part.n_targets * part.n_leaves
            assert (
                part.far_pairs + part.near_geometric_pairs + part.near_box_pairs
                == part.total_pairs
            ), name
            assert part.near_pairs == part.near_geometric_pairs + part.near_box_pairs
            # every pair covers exactly one source of the cloud
            assert part.total_sources == part.n_targets * len(case["sources"])

    def test_far_fraction_is_the_pair_fraction(self, cases):
        for name in ("surface", "separated"):
            case = cases[name]
            op = FMMOperator(p=4, max_leaf=case["max_leaf"])
            op.setup(case["sources"])
            part = op.pair_partition(case["targets"])
            assert op.far_fraction(case["targets"]) == pytest.approx(
                part.far_pair_fraction, abs=0.0
            )
            assert 0.0 <= part.far_pair_fraction <= 1.0
            assert 0.0 <= part.far_source_fraction <= 1.0

    def test_empty_targets_partition_is_all_zero(self, cases):
        case = cases["separated"]
        op = FMMOperator(p=4, max_leaf=case["max_leaf"])
        op.setup(case["sources"])
        part = op.pair_partition(np.zeros((0, 3)))
        assert isinstance(part, PairPartition)
        assert part.total_pairs == 0 and part.far_pairs == 0
        assert part.far_pair_fraction == 0.0 and part.far_source_fraction == 0.0

    def test_partition_before_setup_fails_closed(self, cases):
        case = cases["separated"]
        op = FMMOperator(p=4, max_leaf=case["max_leaf"])
        with pytest.raises(RuntimeError, match="pair_partition"):
            op.pair_partition(case["targets"])


# ---------------------------------------------------------------------------
# round-trip against the matvec
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_matvec_far_pairs_equal_partition_far_pairs(self, diag, cases):
        for name in ("surface", "separated"):
            case = cases[name]
            op = diag._CountingOperator(p=4, max_leaf=case["max_leaf"])
            op.setup(case["sources"])
            op.matvec(case["targets"], case["sources"], case["charges"])
            part = op.pair_partition(case["targets"])
            assert op.far_pairs_used == part.far_pairs, name
            # the instrumented hook is hit once per non-empty target block
            assert sum(op.far_pairs_per_block) == op.far_pairs_used

    def test_committed_evidence_roundtrip_rows_pass(self, evidence):
        assert evidence["roundtrip_ok"] is True
        assert len(evidence["roundtrip"]) >= 6
        for row in evidence["roundtrip"]:
            assert row["matvec_far_pairs"] == row["partition_far_pairs"], row
            assert row["match"] is True, row


# ---------------------------------------------------------------------------
# what the far band is made of
# ---------------------------------------------------------------------------

class TestFarBandComposition:
    def test_targets_outside_the_cloud_have_no_geometric_near_pairs(self, cases):
        """All pairs are admissible for a target far outside the source ball, so
        every directly-summed pair is the conservative box margin, not geometry."""
        for name in ("shell", "separated"):
            case = cases[name]
            op = FMMOperator(p=4, max_leaf=case["max_leaf"])
            op.setup(case["sources"])
            part = op.pair_partition(case["targets"])
            assert part.near_geometric_pairs == 0, name
            assert part.near_box_pairs > 0, name

    def test_box_margin_is_the_only_thing_keeping_pairs_direct(self, cases):
        """Dropping it makes every shell/separated pair multipole-evaluated."""
        for name in ("shell", "separated"):
            case = cases[name]
            op = _load_diagnostic()._AdmissibilityOnlyOperator(p=4, max_leaf=case["max_leaf"])
            op.setup(case["sources"])
            part = op.pair_partition(case["targets"])
            assert part.far_pairs == part.total_pairs, name
            assert part.far_pair_fraction == 1.0

    def test_pair_fraction_is_not_the_work_fraction(self, cases):
        """Uneven leaf populations: the source-weighted fraction is lower."""
        case = cases["separated"]
        op = FMMOperator(p=4, max_leaf=case["max_leaf"])
        op.setup(case["sources"])
        part = op.pair_partition(case["targets"])
        assert part.far_source_fraction < 0.75 * part.far_pair_fraction

    def test_single_leaf_tree_is_a_pure_direct_sum(self, diag, cases):
        """Degenerate coarse tree: far_fraction == 0 and the answer is exact."""
        case = cases["surface"]
        op = FMMOperator(p=4, max_leaf=10_000)
        op.setup(case["sources"])
        part = op.pair_partition(case["targets"])
        assert part.n_leaves == 1 and part.far_pairs == 0
        ref = diag.direct_laplace_matvec(
            case["targets"], case["sources"], case["charges"]
        )
        got = op.matvec(case["targets"], case["sources"], case["charges"])
        assert np.max(np.abs(got - ref)) / np.max(np.abs(ref)) < 1e-12


# ---------------------------------------------------------------------------
# committed evidence is generated, not hand-copied
# ---------------------------------------------------------------------------

class TestCommittedEvidence:
    def test_evidence_reproduces(self, diag, cases, evidence):
        assert evidence["mission"] == "m18196"
        for row in evidence["rows"]:
            if row["p"] != 4:
                continue
            name = row["geometry"]
            case = cases[name]
            cls = (
                diag.FMMOperator
                if row["policy"] == "current"
                else diag._AdmissibilityOnlyOperator
            )
            fresh = diag._row(name, case, 4, cls)
            for key in (
                "far_pairs",
                "near_geometric_pairs",
                "near_box_pairs",
                "far_sources",
                "far_pair_fraction",
                "far_source_fraction",
            ):
                assert fresh[key] == row[key], (name, row["policy"], key)
            assert fresh["rel_error_vs_dense"] == pytest.approx(
                row["rel_error_vs_dense"], rel=1e-12, abs=1e-18
            ), (name, row["policy"])

    def test_tables_are_generated_from_the_evidence(self, evidence):
        tables = (DOCS / "tables.md").read_text()
        assert tables.startswith("<!-- generated by scripts/fmm_far_fraction_diagnostic.py")
        for heading in ("T1", "T2", "T3", "T4"):
            assert heading in tables
        # one formatted row per evidence row, no hand edits possible
        for row in evidence["rows"]:
            cell = "| {ff:.4f} | {fsf:.4f} |".format(
                ff=row["far_pair_fraction"], fsf=row["far_source_fraction"]
            )
            assert cell in tables, row["geometry"]

    def test_readme_numbers_are_injected_from_the_evidence(self, evidence):
        readme = (DOCS / "README.md").read_text()
        assert "<!--" not in readme  # narrative file, not a generated stub
        for row in evidence["rows"]:
            if row["p"] != 4 or row["policy"] != "current":
                continue
            cell = f"| {row['far_pair_fraction']:.4f} | {row['far_source_fraction']:.4f} |"
            assert cell in readme, row["geometry"]

    def test_figure_exists_and_is_not_a_placeholder(self):
        fig = DOCS / "fig_far_fraction.png"
        assert fig.exists()
        assert fig.stat().st_size > 10_000
        assert fig.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    def test_evidence_records_the_admissibility_factor(self, evidence):
        assert evidence["admissibility_factor"] == pytest.approx(
            float(fmm_mod._ADMISSIBILITY)
        )
        assert math.isfinite(evidence["admissibility_factor"])

class TestLinearityEvidenceConsistency:
    """m16000 documented ``far_fraction: 0.0`` as a self-contradiction; its cause
    was the pre-m16109 fail-open call.  The committed value is now the number the
    current code reproduces on that bench geometry (m18196)."""

    def test_linearity_evidence_far_fraction_is_current(self):
        bench = ROOT / "scripts" / "fmm_linearity_bench.py"
        spec = importlib.util.spec_from_file_location("fmm_linearity_bench", bench)
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("fmm_linearity_bench", module)
        spec.loader.exec_module(module)
        fresh = module.error_versus_order()["far_fraction"]
        committed = json.loads(
            (ROOT / "docs" / "fmm_operator_linearity" / "evidence.json").read_text()
        )["error_versus_order"]["far_fraction"]
        assert committed > 0.0, "the m16000 '0.0 on a set-up operator' value is back"
        assert committed == pytest.approx(fresh, rel=0.0, abs=1e-12)
