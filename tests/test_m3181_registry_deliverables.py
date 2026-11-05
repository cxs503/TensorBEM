"""Regression gate for the m3181 3-D free-surface registry deliverables.

``docs/m3181/`` holds a recomputation ledger, three figures and a generated block
inside ``dev_report_m3181_three_d_free_surface_registry.md``. Those artifacts are
committed, so they can rot in three independent ways, and each gets a test here:

* the registry gains/loses/renames a cell but the ledger is not re-run
  (:func:`test_ledger_matches_the_current_registry`);
* a solver change moves a measured value but the ledger still claims PASS
  (:func:`test_every_registered_cell_still_passes_the_ledger`, plus the heavy
  re-run behind ``TENSORBEM_HEAVY_TESTS``);
* the report or the figures drift away from the ledger
  (:func:`test_report_block_matches_the_ledger`,
  :func:`test_figures_exist_and_are_png`).

The default tests never solve a panel system; the end-to-end re-run costs about
two minutes and is therefore opt-in.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tensorbem.benchmarks.declared_numbers import (
    DECLARED_NUMBERS,
    _chief_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs" / "m3181"
LEDGER = DOCS / "outputs" / "three_d_free_surface_registry.json"
REPORT = REPO_ROOT / "dev_report_m3181_three_d_free_surface_registry.md"
FIGURES = DOCS / "figures"

GROUPS = ("hulme_hemisphere", "hulme_chief", "submerged_sphere")

#: The four cells that were quoted in tables 11-12 but unregistered before m3181.
GAP_CELLS = (
    "hulme_chief_2_58_exact_a33_dimless",
    "hulme_chief_2_58_exact_b33_dimless",
    "hulme_chief_2_58_chief_point_count",
    "submerged_sphere_20_rotational_coupling",
)

HEAVY = pytest.mark.skipif(
    not os.environ.get("TENSORBEM_HEAVY_TESTS"),
    reason="set TENSORBEM_HEAVY_TESTS=1 to re-run the 3-D free-surface benchmarks",
)


@pytest.fixture(scope="module")
def ledger() -> dict:
    assert LEDGER.exists(), f"missing recomputation ledger: {LEDGER}"
    return json.loads(LEDGER.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ledger_rows(ledger: dict) -> dict[str, dict]:
    return {row["key"]: row for row in ledger["checks"]}


# --------------------------------------------------------------------------
# The ledger against the registry
# --------------------------------------------------------------------------


def test_ledger_matches_the_current_registry(ledger, ledger_rows):
    """Every registered 3-D cell is in the ledger, with the identical declared value.

    This is the drift gate that does not need a solve: renaming a key, retyping a
    documented digit or adding a cell without re-running the benchmarks all fail
    here rather than silently leaving a stale evidence file in the tree.
    """
    registered = {entry.key: entry for entry in DECLARED_NUMBERS if entry.group in GROUPS}
    assert set(ledger_rows) == set(registered), (
        f"missing from the ledger: {sorted(set(registered) - set(ledger_rows))}; "
        f"stale in the ledger: {sorted(set(ledger_rows) - set(registered))}"
    )
    for key, entry in registered.items():
        row = ledger_rows[key]
        assert row["declared"] == entry.declared, key
        assert row["group"] == entry.group, key
        assert row["tier"] == entry.tier, key
        assert row["comparison"] == entry.kind, key
        assert row["source"] == entry.source, key
        assert row["parameters"] == entry.parameters, key
    assert tuple(ledger["groups"]) == GROUPS
    assert set(ledger["tiers"]) == {"fast", "standard", "heavy"}


def test_every_registered_cell_still_passes_the_ledger(ledger):
    summary = ledger["summary"]
    assert summary["checked"] == len(ledger["checks"]) == 66
    assert summary["passed"] == 66
    assert summary["drifted"] == 0
    assert summary["drifted_keys"] == []
    assert summary["verdict"] == "PASS"
    drifted = [row["key"] for row in ledger["checks"] if row["verdict"] != "PASS"]
    assert drifted == []


def test_value_cells_stay_inside_their_tolerance(ledger_rows):
    for row in ledger_rows.values():
        if row["comparison"] != "value":
            continue
        assert abs(row["relative_deviation"]) <= row["tolerance"], row["key"]


# --------------------------------------------------------------------------
# The four cells this mission added
# --------------------------------------------------------------------------


def test_table_11_and_12_gap_cells_are_registered(ledger_rows):
    for key in GAP_CELLS:
        assert key in ledger_rows, key
    # the Hulme reference solution reproduces the two printed reference digits
    assert abs(ledger_rows["hulme_chief_2_58_exact_a33_dimless"]["relative_deviation"]) < 1e-3
    assert abs(ledger_rows["hulme_chief_2_58_exact_b33_dimless"]["relative_deviation"]) < 1e-3
    # "5 点三维内部 CHIEF": the automatic selection still picks five points
    assert ledger_rows["hulme_chief_2_58_chief_point_count"]["measured"] == 5.0
    # the quoted 2.5e-32 is a machine-zero claim, so it is judged as an upper bound
    bound = ledger_rows["submerged_sphere_20_rotational_coupling"]
    assert bound["comparison"] == "upper"
    assert abs(bound["measured"]) <= abs(bound["declared"])


def test_chief_regularisation_story_holds_in_the_ledger(ledger_rows):
    """Sign flip and conditioning drop, read off the committed ledger."""
    plain_b33 = ledger_rows["hulme_chief_2_58_unregularized_b33"]["measured"]
    chief_b33 = ledger_rows["hulme_chief_2_58_chief_b33"]["measured"]
    exact_b33 = ledger_rows["hulme_chief_2_58_exact_b33_dimless"]["measured"]
    assert plain_b33 < 0.0 < chief_b33
    assert abs(chief_b33 - exact_b33) < abs(plain_b33 - exact_b33)
    plain_cond = ledger_rows["hulme_chief_2_58_unregularized_condition_number"]["measured"]
    chief_cond = ledger_rows["hulme_chief_2_58_chief_condition_number"]["measured"]
    assert chief_cond < 0.05 * plain_cond


def test_chief_rows_flatten_the_reference_solution():
    """``_chief_rows`` must not drop the row-level ``exact_*`` scalars again."""
    row = {
        "ka": 2.58,
        "exact_a33": 0.4,
        "exact_b33": 0.05,
        "unregularized": {"condition_number": 145.0, "a33_dimless": 0.35, "passivity_violated": True},
        "chief_regularized": {"chief_point_count": 5, "a33_dimless": 0.39},
    }
    values = _chief_rows([row])
    assert values["ka2.58_exact_a33_dimless"] == pytest.approx(0.4)
    assert values["ka2.58_exact_b33_dimless"] == pytest.approx(0.05)
    assert values["ka2.58_unregularized_condition_number"] == pytest.approx(145.0)
    assert values["ka2.58_chief_regularized_chief_point_count"] == pytest.approx(5.0)
    # booleans are flags, not numbers to register
    assert not any("passivity_violated" in key for key in values)


# --------------------------------------------------------------------------
# The artifacts built from the ledger
# --------------------------------------------------------------------------


def test_report_block_matches_the_ledger():
    """The report's tables are generated; ``--check`` proves they are current."""
    completed = subprocess.run(
        [sys.executable, str(DOCS / "build_report.py"), "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    text = REPORT.read_text(encoding="utf-8")
    assert "<!-- BEGIN REGISTRY TABLES" in text and "<!-- END REGISTRY TABLES -->" in text
    assert text.count("<!-- BEGIN REGISTRY TABLES") == 1


def test_figures_exist_and_are_png():
    expected = {
        "fig_registry_3d_free_surface.png",
        "fig_registry_3d_free_surface_small.png",
        "fig_three_d_convergence_registry.png",
        "fig_three_d_convergence_registry_small.png",
        "fig_chief_regularization.png",
        "fig_chief_regularization_small.png",
    }
    present = {path.name for path in FIGURES.glob("*.png")}
    assert expected <= present, sorted(expected - present)
    for name in sorted(expected):
        data = (FIGURES / name).read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n"), name
        assert len(data) > 5_000, name


def test_small_figures_stay_report_sized():
    """The report API rejects oversized payloads; keep the embedded copies small."""
    total = sum((FIGURES / f"{stem}_small.png").stat().st_size for stem in (
        "fig_registry_3d_free_surface",
        "fig_three_d_convergence_registry",
        "fig_chief_regularization",
    ))
    assert total < 200_000, f"embedded figures total {total} bytes"


# --------------------------------------------------------------------------
# End-to-end, opt-in
# --------------------------------------------------------------------------


@HEAVY
def test_end_to_end_recomputation_reproduces_the_committed_ledger(tmp_path):
    """Re-run the three benchmarks and compare against the committed ledger."""
    output = tmp_path / "ledger.json"
    completed = subprocess.run(
        [sys.executable, str(DOCS / "run_registry_check.py"), "--strict", "--output", str(output)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    fresh = json.loads(output.read_text(encoding="utf-8"))
    committed = json.loads(LEDGER.read_text(encoding="utf-8"))
    fresh_rows = {row["key"]: row for row in fresh["checks"]}
    committed_rows = {row["key"]: row for row in committed["checks"]}
    assert set(fresh_rows) == set(committed_rows)
    for key, row in fresh_rows.items():
        assert row["declared"] == committed_rows[key]["declared"], key
        # deterministic benchmarks: the two runs agree to floating-point noise
        assert row["measured"] == pytest.approx(committed_rows[key]["measured"], rel=1e-9), key