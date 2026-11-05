"""The numbers in ``docs/three_dimensional_bem.md`` become a red build when they rot.

The document is the package's 3-D Laplace BEM paper: four refinement tables, a
diagnostics list and a surface-velocity study.  Before ``m3085`` nothing in the
test suite referred to it, so any kernel, mesh or solver change was free to
invalidate it silently -- the failure mode that already produced three doc-drift
incidents in this repository.

What these tests pin, in order of decreasing cheapness:

* the markdown tables still parse, with the row/column layout the registry
  assumes, so a restyled table is a test failure rather than a silent skip;
* every registry cell of every table reproduces from the committed artifacts
  inside the tolerance implied by the digits the document prints;
* the two ``<=`` round-off claims of section 6.5 hold with the documented margin,
  and the original (platform-specific) numbers stay in the registry as evidence;
* the ``p_surf`` ambiguity is real and is documented: the printed value follows
  the ``N^{-1/2}`` convention while the ``p`` column of the same table follows
  ``h_max``, and the document now says so;
* a *live* two-level run of the study reproduces the committed artifact, so the
  artifacts are not a stale transcript that the document was fitted to.

The heavy 1280-panel levels are **not** recomputed in CI: they are pinned against
the committed artifacts (``docs/m3085/outputs/``), which
``--recompute`` regenerates in about three CPU-minutes.
"""

from __future__ import annotations

import json
import math

import pytest

from tensorbem.benchmarks import three_dimensional_doc_pins as pins


@pytest.fixture(scope="module")
def report():
    return pins.check()


def test_tables_parse_with_the_registry_layout():
    tables = pins.parse_tables(pins.DOC.read_text())
    assert set(tables) == {"interior", "added_mass", "flux", "velocity"}
    assert set(tables["interior"].rows) == {20, 80, 320, 1280}
    assert set(tables["added_mass"].rows) == {20, 80, 320, 1280}
    assert set(tables["flux"].rows) == {20, 80, 320, 1280}
    # The velocity table only starts at 80 panels: the 20-panel point is absent
    # because the representation path is already unusable there.
    assert set(tables["velocity"].rows) == {80, 320, 1280}

    # Every registered column must exist in the parsed table.
    for spec in pins.REGISTRY:
        width = max(len(row) for row in tables[spec.table].rows.values())
        assert spec.column < width, f"{spec.table} column {spec.column} out of range"


def test_every_declared_number_reproduces(report):
    assert report["verdict"] == "PASS"
    assert report["cells_drifted"] == 0
    graded = report["cells_graded"]
    assert graded == report["cells_passed"]
    # Four tables, 25 registry entries: 86 graded cells, five of them "no value".
    assert graded >= 80
    drifted = [c for c in report["checks"] if c["status"] == "DRIFT"]
    assert drifted == []


def test_value_cells_agree_within_printed_precision(report):
    """The tolerance must come from the printed digits, so check the worst case."""
    values = [c for c in report["checks"] if c["kind"] == "value" and c["status"] == "PASS"]
    assert values
    for cell in values:
        assert cell["tolerance"] is not None
        deviation = abs(cell["measured"] - cell["declared"]) / abs(cell["declared"])
        assert deviation <= cell["tolerance"], (cell["quantity"], cell["panels"], deviation)


def test_roundoff_bounds_hold_with_the_documented_margin(report):
    bounds = {b["quantity"]: b for b in report["bounds"]}
    assert set(bounds) == set(pins.DOCUMENTED_BOUNDS)
    for quantity, entry in bounds.items():
        assert entry["status"] == "PASS"
        assert entry["measured_worst"] <= pins.ROUNDOFF_CEILING
        assert entry["measured_worst"] <= entry["documented_bound"]
    # The audit trail must keep the original, platform-specific numbers.
    assert bounds["double_layer_row_sum_error"]["original_documented_bound"] == 1.6e-15
    assert bounds["relative_residual_l2"]["original_documented_bound"] == 6.6e-16
    # And at least one of them really is exceeded on this platform: if that ever
    # stops being true the correction note in the document is stale and should be
    # revisited rather than left as decoration.
    assert any(entry["exceeds_original_bound"] for entry in bounds.values())


def test_roundoff_columns_are_not_used_as_accuracy_claims(report):
    """Row sums and residuals must not be graded like a physics error."""
    for cell in report["checks"]:
        if cell["kind"] == "roundoff":
            assert cell["status"] in {"PASS", "DRIFT"}
            if cell["status"] == "PASS":
                assert cell["measured"] <= pins.ROUNDOFF_CEILING


def test_velocity_order_convention_ambiguity_is_real_and_documented():
    """The printed p_surf follows N^{-1/2}; the p column of the same table uses h_max."""
    artifacts = pins.load_artifacts()
    measured = pins.measure(artifacts)
    declared = pins.cell_to_float(pins.parse_tables(pins.DOC.read_text())["velocity"].cell(320, 6))
    assert declared is not None
    by_panels = measured[("velocity", 320, 6, "panels")]
    by_h_max = measured[("velocity", 320, 6, "h_max")]
    assert abs(by_panels - declared) / declared <= 2.0e-2
    assert abs(by_h_max - declared) / declared > 2.0e-2
    # And the document must say so, otherwise a reader cannot compare the columns.
    text = pins.DOC.read_text()
    assert "N^{-1/2}" in text or "N^-1/2" in text or "\\sqrt{N}" in text


def test_result_tables_are_fully_protected(report):
    """Coverage: every printed number of the four tables is registry-backed."""
    coverage = pins.coverage()
    assert coverage["protected_in_tables"] >= 60
    assert coverage["protected_fraction"] > 0.2
    # The prose numbers are grouped by section so a later shift can see what is
    # still unguarded instead of assuming full coverage.
    assert isinstance(coverage["unguarded_by_section"], dict)
    assert coverage["numeric_tokens_total"] == coverage["protected_in_tables"] + coverage["unguarded"]


def test_pins_file_is_consistent_with_the_document():
    if not pins.PINS_FILE.is_file():
        pytest.skip("pin file not generated yet")
    payload = json.loads(pins.PINS_FILE.read_text())
    assert payload["verdict"] == "PASS"
    assert payload["document"] == "docs/three_dimensional_bem.md"
    assert len(payload["cells"]) >= 80
    assert all(cell["status"] in {"PASS", "N/A"} for cell in payload["cells"])


def test_live_two_level_run_reproduces_the_committed_artifact(tmp_path):
    """Provenance: the artifacts are what this code computes, not a transcript.

    Only subdivisions 0 and 1 (20 and 80 panels) are recomputed -- the 1280-panel
    levels are pinned against the committed artifacts to keep CI cheap.
    """
    written = pins.recompute(tmp_path, levels=(0, 1))
    live = pins.load_artifacts(tmp_path)
    committed = pins.load_artifacts()

    checked = 0
    artifacts_live = {"study": live["study"], "surface_study": live["surface_study"],
                      "sphere_velocity": committed["sphere_velocity"]}
    measured_live = pins.measure(artifacts_live)
    measured_committed = pins.measure(committed)
    tables = pins.parse_tables(pins.DOC.read_text())
    for spec in pins.REGISTRY:
        if spec.table == "velocity":
            continue
        for panels in (20, 80):
            key = (spec.table, panels, spec.column)
            if key not in measured_live or key not in measured_committed:
                continue
            declared_cell = tables[spec.table].cell(panels, spec.column)
            declared = pins.cell_to_float(declared_cell)
            if declared is None:
                continue
            tolerance = spec.tolerances(declared_cell)[0]
            live_value = measured_live[key]
            committed_value = measured_committed[key]
            assert pins._agrees(declared, live_value, tolerance), (
                spec.quantity, panels, declared, live_value
            )
            if spec.kind == "roundoff":
                assert live_value <= pins.ROUNDOFF_CEILING
            else:
                assert abs(live_value - committed_value) <= tolerance * max(abs(committed_value), 1e-300), (
                    spec.quantity, panels, live_value, committed_value
                )
            checked += 1
    assert checked >= 20, checked
    assert written  # the CLI wrote every artifact it claims to