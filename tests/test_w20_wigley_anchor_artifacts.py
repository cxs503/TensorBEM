"""W20 item 4 / mission 9647: the Wigley anchor artifacts must stay self-consistent.

These tests do not run any solver.  They check the two claims the report makes
about its own artifacts:

1. the field-level transcription really covers **every** scalar leaf of both
   repository anchors, and every pointer round-trips to the same value/type
   (criterion 1 is 100 %, not "most fields");
2. every verdict declared in the truncation comparison is reproducible from the
   numbers stored next to it, and no row is labelled ``experiment`` although the
   repository holds no model-test data (criterion 2 discipline).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TRANS = REPO / "papers/data/w20-anchors/wigley_repo_benchmark_transcription.json"
COMP = REPO / "papers/data/w20-anchors/wigley_truncation_comparison.json"
SOURCES = {
    "wave_resistance": "benchmarks/verified/bem_wigley_wave_resistance/results.json",
    "hydrostatics": "benchmarks/verified/bem_wigley_hydrostatics/results.json",
}


def resolve(doc, pointer: str):
    cur = doc
    for tok in pointer.split("/")[1:]:
        tok = tok.replace("~1", "/").replace("~0", "~")
        cur = cur[int(tok)] if isinstance(cur, list) else cur[tok]
    return cur


def walk(node, prefix=""):
    if isinstance(node, dict):
        for key, val in node.items():
            tok = str(key).replace("~", "~0").replace("/", "~1")
            yield from walk(val, f"{prefix}/{tok}")
    elif isinstance(node, list):
        for idx, val in enumerate(node):
            yield from walk(val, f"{prefix}/{idx}")
    else:
        yield prefix, node


@pytest.fixture(scope="module")
def trans():
    return json.loads(TRANS.read_text())


@pytest.fixture(scope="module")
def comp():
    return json.loads(COMP.read_text())


def test_transcription_is_one_hundred_percent(trans):
    total = 0
    for study, rel in SOURCES.items():
        src = json.loads((REPO / rel).read_text())
        leaves = dict(walk(src))
        stored = {leaf["pointer"]: leaf["value"]
                  for leaf in trans["field_inventory"][study]["leaves"]}
        assert set(stored) == set(leaves), f"{study}: pointer set differs from source"
        for pointer, value in leaves.items():
            back = resolve(src, pointer)
            assert back == value
            assert stored[pointer] == value, f"{study}{pointer} value drift"
        total += len(leaves)
    assert trans["completeness"]["leaves_total"] == total
    assert trans["completeness"]["leaves_transcribed"] == total
    assert trans["completeness"]["rate"] == 1.0
    assert trans["completeness"]["roundtrip_failures"] == []
    assert trans["completeness"]["required_classes_all_present"] is True


def test_source_hashes_match_the_files_on_disk(trans):
    for entry in trans["sources"]:
        digest = hashlib.sha256((REPO / entry["path"]).read_bytes()).hexdigest()
        assert entry["sha256"] == digest, entry["path"]


def test_no_row_is_labelled_experiment(comp, trans):
    rows = (comp["comparison_table"]["wave_resistance"]
            + comp["comparison_table"]["hydrostatics"])
    assert rows
    for row in rows:
        assert row["reference_type"] in ("analytic", "numerical")
    assert comp["criterion_verdicts"]["criterion_2_comparison_table"]["experiment_rows"] == 0
    ledger = trans["reference_type_ledger"]
    assert ledger["experiment_anchor"]["status"] == "WITHHELD"
    for entry in ledger["entries"]:
        assert entry["is_experiment"] is False


def test_truncation_verdicts_follow_from_the_stored_numbers(comp):
    gates = comp["predeclared_gates"]
    for dom in comp["domains_by_froude"].values():
        base = dom["lateral"][-1]["rw_pressure_N"]
        for row in dom["lateral"]:
            expected = 100.0 * abs(row["rw_pressure_N"] - base) / abs(base)
            assert row["rel_change_vs_reference_domain_pct"] == pytest.approx(
                expected, rel=1e-12)
        for row in dom["downstream"]:
            base_x = dom["downstream"][-1]["rw_pressure_N"]
            expected = 100.0 * abs(row["rw_pressure_N"] - base_x) / abs(base_x)
            assert row["rel_change_vs_reference_domain_pct"] == pytest.approx(
                expected, rel=1e-12)

    c3 = comp["criterion_verdicts"]["criterion_3_truncation_conclusion"]
    assert c3["G1_verdict"] == ("PASS" if c3["G1_lateral_max_pct"]
                                <= gates["G1_lateral_truncation_pct"] else "FAIL")
    assert c3["G2_verdict"] == ("PASS" if c3["G2_downstream_max_pct"]
                                <= gates["G2_downstream_truncation_pct"] else "FAIL")
    assert c3["G3_verdict"] == ("PASS" if c3["G3_radiation_energy_gap_pct"]
                                <= gates["G3_radiation_energy_gap_pct"] else "FAIL")
    # the honest FAIL must survive: no threshold was relaxed after the fact
    assert gates["G3_radiation_energy_gap_pct"] == 5.0
    assert c3["G3_verdict"] == "FAIL"
    assert comp["criterion_verdicts"]["criterion_3_truncation_conclusion"]["physics_verdict"] == "FAIL"


def test_damping_zone_is_declared_not_supported(comp):
    dz = comp["damping_zone"]
    assert dz["status"] == "NOT_SUPPORTED"
    # the negative evidence must still be true in the source tree
    text = (REPO / dz["evidence"]["path"]).read_text().lower()
    assert "no sponge layer" in text


FIGURE_NAMES = (
    "w20_wigley_truncation_domain.png",
    "w20_wigley_energy_and_accuracy.png",
    "w20_wigley_gates_and_mesh.png",
)


def _figure_builder():
    """Load ``papers/data/w20-anchors/build_wigley_truncation.py`` by path.

    The directory name is not importable (hyphen), and the module only defines
    functions at import time, so a ``spec_from_file_location`` load is safe.
    """
    import importlib.util

    path = REPO / "papers/data/w20-anchors/build_wigley_truncation.py"
    spec = importlib.util.spec_from_file_location("w20_build_wigley_truncation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_artifact_and_figures_are_reproducible(tmp_path):
    """The report's three figures must be re-renderable from the committed record.

    A committed ``*.png`` under ``papers/`` is not a durable invariant: the
    platform push gate drops ``*.png`` outside ``docs/`` (see
    ``docs/dev_report_m13806_push_gate_figure_integrity.md``), so the images the
    report links never reach ``main``.  The invariant kept here is the
    reproducible one -- the figures are rendered from
    ``papers/data/w20-anchors/wigley_truncation_comparison.json`` into a
    temporary directory, must be non-trivial, and the report must still link
    exactly those names.
    """
    report = REPO / "papers/reports/w20_seakeeping_truncation4.md"
    assert report.exists() and report.stat().st_size > 5000

    doc = json.loads(COMP.read_text())
    figs = _figure_builder().make_figures(doc, out_dir=tmp_path)
    assert [f["name"] for f in figs] == list(FIGURE_NAMES)
    assert all(f["caption"].strip() for f in figs)

    from _figure_integrity import assert_reproducible_figures

    assert_reproducible_figures(
        [tmp_path / name for name in FIGURE_NAMES],
        minimum=3,
        minimum_bytes=10_000,
        embedded_in=report.read_text(encoding="utf-8"),
    )