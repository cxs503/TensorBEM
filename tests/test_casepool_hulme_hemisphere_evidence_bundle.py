"""Contract tests for the CASE-BEM-HULME-HEMISPHERE evidence bundle.

The bundle (``benchmarks/verified/casepool_bem_hulme_hemisphere/evidence/`` plus
the case ``README.md``) exists so that a reader -- or a reviewer of a pull
request -- can check the case's claims without re-running any BEM solve, and so
that a hand edit cannot pass unnoticed.  These tests pin the four properties the
bundle is supposed to have:

1. **Reproducible** -- ``evidence/build_evidence.py`` regenerates all six files
   byte for byte from the committed run artifacts, so no number in the bundle can
   have been typed in by hand.
2. **Self-consistent** -- every per-cell relative error stored in
   ``evidence/benchmark.json`` is re-derived here from the pair
   ``(computed, reference)`` in that same file, and the reference series matches
   ``results.json``/``reference.json``; the harness does not just trust the
   builder's own re-derivation.
3. **Integrity-checked** -- every sha256 in ``evidence/artifacts_sha256.json``
   matches the file on disk, so a silent edit of an artifact or of a bundle file
   breaks the test.
4. **Honest about provenance** -- the anchor registry must *not* claim a
   literature table it does not have, must declare itself ineligible for the
   external-anchor ledger, and the third-party (capytaine) axis must be present
   with its own verdict, including the post-hoc observation that is deliberately
   excluded from the gates.

No solver run is required: the tests read committed artifacts only.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

CASEPOOL_DIR = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "verified"
    / "casepool_bem_hulme_hemisphere"
)
EVIDENCE_DIR = CASEPOOL_DIR / "evidence"
BUILDER = EVIDENCE_DIR / "build_evidence.py"
README = CASEPOOL_DIR / "README.md"
TOLERANCE = 0.05
GENERATED = (
    "evidence/reference.json",
    "evidence/benchmark.json",
    "evidence/independent.json",
    "evidence/anchor_registry.json",
    "evidence/artifacts_sha256.json",
    "README.md",
)
QUANTITIES = ("A11", "B11", "A33", "B33", "Fx", "Fz")
FIELD = {tag: f"{tag}_dimless" for tag in QUANTITIES}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def builder():
    assert BUILDER.is_file(), f"missing generator: {BUILDER}"
    spec = importlib.util.spec_from_file_location("hulme_evidence_builder", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bundle() -> dict[str, dict]:
    out = {}
    for rel in GENERATED:
        path = CASEPOOL_DIR / rel
        assert path.is_file(), f"missing bundle file: {rel}"
        if rel.endswith(".json"):
            out[rel] = _load_json(path)
    return out


@pytest.fixture(scope="module")
def benchmark(bundle) -> dict:
    return bundle["evidence/benchmark.json"]


@pytest.fixture(scope="module")
def registry(bundle) -> dict:
    return bundle["evidence/anchor_registry.json"]


@pytest.fixture(scope="module")
def independent(bundle) -> dict:
    return bundle["evidence/independent.json"]


@pytest.fixture(scope="module")
def reference_doc(bundle) -> dict:
    return bundle["evidence/reference.json"]


# --- 1. reproducibility ---------------------------------------------------

def test_bundle_regenerates_byte_identical(builder, tmp_path):
    """The generator must reproduce every committed bundle file exactly."""
    builder.build(CASEPOOL_DIR, tmp_path)
    for rel in GENERATED:
        committed = CASEPOOL_DIR / rel
        rebuilt = tmp_path / rel
        assert rebuilt.is_file(), f"generator did not write {rel}"
        assert rebuilt.read_bytes() == committed.read_bytes(), (
            f"{rel} is stale: committed bytes differ from a fresh generation "
            f"(run evidence/build_evidence.py after any change to the artifacts)")


def test_bundle_has_no_nan_or_inf(bundle):
    def walk(node, where):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{where}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{where}[{index}]")
        elif isinstance(node, float):
            assert math.isfinite(node), f"non-finite number at {where}"

    for rel, payload in bundle.items():
        walk(payload, rel)


# --- 2. self-consistency of the stored numbers ----------------------------

def test_case_mesh_errors_rederive_from_values_and_reference(benchmark):
    """Independent re-derivation: the test, not the builder, does the division."""
    rows = [row for row in benchmark["rows"]
            if row["kind"] == "band" and row["subdivisions"] == benchmark["meshes"]["case"]["subdivisions"]]
    assert len(rows) == 6, "expected six band frequencies on the case mesh"
    for row in rows:
        for tag in QUANTITIES:
            computed = row["values"][tag]
            reference = row["reference"][tag]
            stored = row["rel_error"][tag]
            assert stored == pytest.approx((computed - reference) / reference, abs=1e-12), (
                f"stored rel_error of {tag} at ka={row['ka']} does not follow from "
                f"the values in the same file")


def test_reference_series_matches_run_artifact(reference_doc):
    results = _load_json(CASEPOOL_DIR / "results.json")
    for ka, series in reference_doc["series"].items():
        stored = results["reference"][ka]
        for tag in QUANTITIES:
            assert series[FIELD[tag]] == pytest.approx(stored[FIELD[tag]], rel=0, abs=1e-15)
    band = [float(ka) for ka in reference_doc["series"]]
    assert band == sorted(band), "reference series must be stored in increasing ka"
    assert [float(ka) for ka in results["reference"]] == pytest.approx(band)


def test_reference_truncation_drift_matches_ladder(reference_doc):
    """The two independent stores of the truncation drift must agree."""
    results = _load_json(CASEPOOL_DIR / "results.json")
    for ka, drift in reference_doc["truncation_drift"].items():
        ladder = results["ref_ladder"][ka]["max_rel_delta"]
        for step, value in drift.items():
            assert value == pytest.approx(ladder[step], abs=1e-18)


def test_verdict_and_gates_are_pass_and_inside_the_band(benchmark):
    assert benchmark["verdict"]["verdict"] == "PASS"
    assert benchmark["verdict"]["all_pass"] is True
    criteria = benchmark["verdict"]["criteria"]
    assert set(criteria) == {
        "G1_case_mesh_band_error_below_5pct",
        "G2_geometry_precondition",
        "G3_refinement_does_not_worsen",
        "G4_reference_self_converged",
    }
    for name, crit in criteria.items():
        assert crit["pass"] is True, f"{name} did not pass: {crit}"

    worst = benchmark["summary"]["case_mesh_worst"]
    assert abs(worst["rel_error"]) < TOLERANCE
    assert benchmark["summary"]["cells_inside_band"] == "36/36"
    # the headline number in the summary must be the true maximum of the rows
    rows = [row for row in benchmark["rows"]
            if row["kind"] == "band" and row["subdivisions"] == 4]
    peak = max(abs(row["rel_error"][tag]) for row in rows for tag in QUANTITIES)
    assert abs(worst["rel_error"]) == pytest.approx(peak, abs=1e-15)


def test_tolerance_is_the_predeclared_one_and_not_loosened(benchmark, independent):
    assert benchmark["tolerance_relative"] == TOLERANCE
    assert independent["tolerance_relative"] == TOLERANCE
    # an inscribed constant-panel mesh under-estimates the displaced volume; the
    # gate is on the absolute value and is tightened, never loosened
    assert 0 < abs(benchmark["geometry"]["volume_rel_error"]) < 0.02
    assert 0 < abs(benchmark["geometry"]["waterplane_rel_error"]) < 0.02


# --- 3. integrity of the sha256 manifest ----------------------------------

def test_manifest_hashes_match_the_files_on_disk(bundle):
    manifest = bundle["evidence/artifacts_sha256.json"]
    assert manifest["algorithm"] == "sha256"
    listed = {item["path"]: item for item in manifest["files"]}
    assert "evidence/artifacts_sha256.json" not in listed, (
        "a manifest cannot list itself")
    for rel, item in listed.items():
        path = CASEPOOL_DIR / rel
        assert path.is_file(), f"manifest lists a missing file: {rel}"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == item["sha256"], f"sha256 mismatch for {rel}"
        assert item["bytes"] == path.stat().st_size
    # every other bundle file must be covered
    for rel in GENERATED:
        if rel != "evidence/artifacts_sha256.json":
            assert rel in listed, f"{rel} is not integrity-checked by the manifest"


def test_manifest_aggregate_is_reproducible(bundle):
    manifest = bundle["evidence/artifacts_sha256.json"]
    canonical = "\n".join(
        f"{item['path']}:{item['sha256']}" for item in manifest["files"])
    assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == manifest["aggregate_sha256"]


# --- 4. provenance honesty ------------------------------------------------

def test_anchor_registry_does_not_claim_a_literature_table(registry, reference_doc):
    assert reference_doc["source_class"] == "LITERATURE_CLOSED_FORM"
    assert registry["anchors"], "the registry must carry at least one anchor"
    for anchor in registry["anchors"]:
        assert anchor["source_class"] == "LITERATURE_CLOSED_FORM"
        assert anchor["transcribed_digits"] is None, (
            "no digit of Hulme's printed table is available in this repository; "
            "claiming transcribed digits would be a provenance defect")
        # the registry must name the source class it is *not*
        assert "NOT a LITERATURE_TABLE anchor" in reference_doc["source_class_note"]


def test_anchor_registry_declares_ledger_ineligibility_with_a_reason(registry, reference_doc):
    """The ineligibility claim must follow from the repository's own gate rule."""
    assert registry["external_ledger_eligible"] is False
    verdict = registry["external_ledger_verdict"]
    assert verdict["result"] == "FAIL"
    assert verdict["independent_of_repo_package"] is False
    assert "src/" in verdict["why"], "the reason must name the in-package reference"
    assert registry["external_ledger"].endswith("benchmarks/anchors/external/index.json")
    assert registry["external_ledger"] == "benchmarks/anchors/external/index.json"

    # recompute the marker hit from the gate's own constants, not from the bundle
    gate_path = CASEPOOL_DIR.parents[1] / "anchors" / "external_anchor_gate.py"
    spec = importlib.util.spec_from_file_location("tensorbem_external_anchor_gate_test",
                                                  gate_path)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    expected = [marker for marker in gate.REPO_MARKERS
                if marker in reference_doc["implementation"]]
    assert expected, ("the gate no longer flags the reference implementation as in-repo; "
                      "the registry record would be stale")
    assert verdict["repo_markers_hit"] == expected
    assert "reference_method.independent_of_repo_package must be true" in gate_path.read_text()


def test_independent_axis_covers_all_six_quantities(independent):
    table = independent["study_table"]
    covered = {(row["quantity"], row["ka"]) for row in table}
    band = {0.2, 0.5, 0.8, 1.0, 1.5, 2.0}
    assert {quantity for quantity, _ in covered} == set(QUANTITIES)
    assert {ka for _, ka in covered} == band
    assert len(table) == 36
    assert independent["verdict"] == "PASS"
    for name, crit in independent["gates"].items():
        assert crit["pass"] is True, f"{name} did not pass: {crit}"
    # the strictest reading is reported but must not be dressed up as a gate
    observation = independent["post_hoc_observation"]
    assert "NOT a pre-registered gate" in observation["kind"]
    assert abs(observation["measured"]) < TOLERANCE, (
        "if the post-hoc observation left the 5 % band the case would have to be "
        "re-opened, not silently re-gated")
    assert set(independent["gates"]).isdisjoint({"solver_vs_capytaine"})


def test_limit_context_is_declared_not_independent(independent, reference_doc):
    """The ka -> 0 limit is context: it comes from the same package, so it is labelled."""
    context = independent["limit_context"]
    assert context["independent"] is False
    assert "not a second code" in context["why_not_independent"]
    source = _load_json(
        CASEPOOL_DIR.parents[0] / "bem_rigid_lid_hemisphere" / "results.json")
    even = source["criteria"]["R1_even_mode_reference_refutes_rho_nabla"]["measured"]
    assert context["ka_to_zero_even_mode_A33"] == pytest.approx(even, abs=0.0, rel=0.0)
    band = [float(ka) for ka in reference_doc["series"]]
    lowest = context["lowest_band_point"]
    assert lowest["ka"] == min(band)
    # the reported gap must follow from the two numbers in the bundle
    assert lowest["gap_to_limit"] == pytest.approx(
        lowest["A33_dimless"] / even - 1.0, abs=1e-15)


# --- 5. README is a rendering of the bundle, not a second copy ------------

def test_readme_reports_the_bundle_numbers(bundle):
    text = README.read_text(encoding="utf-8")
    benchmark = bundle["evidence/benchmark.json"]
    worst = benchmark["summary"]["case_mesh_worst"]
    independent = bundle["evidence/independent.json"]

    assert benchmark["verdict"]["verdict"] in text
    assert f"{100.0 * worst['rel_error']:+.2f} %" in text
    assert f"ka = {worst['ka']:g}" in text
    assert f"`{worst['quantity']}`" in text
    assert f"{100.0 * independent['worst_overall_vs_reference']['measured']:+.2f} %" in text
    for rel in GENERATED:
        if rel == "README.md":
            continue
        assert rel in text, f"README does not point at {rel}"
    assert "evidence/build_evidence.py" in text
    assert "not** in" in text  # the ledger note