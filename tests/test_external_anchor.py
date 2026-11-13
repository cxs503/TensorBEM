"""External-anchor pipeline for TensorBEM, with a provenance gate (M-R16-03).

What this module locks down
---------------------------
Report 16348 graded a floating-hemisphere case against values produced by the
repository's **own** Hulme multipole routine while the report labelled them
"Hulme (1982) original table".  Every numeric gate passed; the provenance was
fabricated.  The pipeline under test makes that impossible to hide:

* ``benchmarks/anchors/external/index.json`` -- machine-readable registry of
  external anchors (source class, DOI/ISBN/URL, locator, conditions, units).
* ``benchmarks/anchors/external/maccamy_fuchs_cylinder.json`` -- the anchor used
  here: the MacCamy & Fuchs (1954) closed-form diffraction force on a vertical
  circular cylinder, re-derived with ``scipy.special`` only (never through
  ``tensorbem``).
* ``benchmarks/anchors/external_anchor_gate.py`` -- three independent checks:
  provenance schema, reference reproducibility from the cited equation (1e-6),
  and the 5 % ledger tolerance of the solver against the anchor.
* ``benchmarks/anchors/external/evidence/maccamy_fuchs_error_table.json`` (and the
  ``artifacts/external_anchors/`` copy) -- the committed error table, one row per quantity with
  ``{quantity, computed, reference, error_pct, pass_5pct}``.

Five red-team fixtures under ``benchmarks/anchors/external/redteam/`` each break
exactly one rule; the Hulme one additionally proves the point of the design: its
numbers *do* reproduce, so only the provenance rule can reject it.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANCHOR_DIR = REPO_ROOT / "benchmarks" / "anchors" / "external"
GATE_PATH = REPO_ROOT / "benchmarks" / "anchors" / "external_anchor_gate.py"
#: Canonical (git-tracked) error table.  ``artifacts/`` is git-ignored in this
#: repository, so the durable copy lives next to the anchor record and the
#: ``artifacts/`` copy is a local convenience for downstream consumers.
EVIDENCE = ANCHOR_DIR / "evidence" / "maccamy_fuchs_error_table.json"
ARTIFACT = REPO_ROOT / "artifacts" / "external_anchors" / "maccamy_fuchs_error_table.json"
FIGURE = REPO_ROOT / "docs" / "external_anchors_bem_error.png"

_spec = importlib.util.spec_from_file_location("external_anchor_gate", GATE_PATH)
gate_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate_mod)

REQUIRED_ROW_KEYS = ("quantity", "computed", "reference", "error_pct", "pass_5pct")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def registry() -> dict:
    return gate_mod.load_registry(ANCHOR_DIR)


@pytest.fixture(scope="module")
def anchor(registry: dict) -> dict:
    """The (single) registered external anchor record."""
    return registry["records"][0]


@pytest.fixture(scope="module")
def computed_32(anchor: dict) -> dict:
    """Cheap solver run: 32 panels (~0.5 s) -- sufficient to exercise the gate."""
    return gate_mod.compute_quantities(anchor, panels=32)


@pytest.fixture(scope="module")
def computed_64(anchor: dict) -> dict:
    """Finer run used for the tightened 1 % claim (~2 s)."""
    return gate_mod.compute_quantities(anchor, panels=64)


@pytest.fixture(scope="module")
def table() -> dict:
    assert EVIDENCE.exists(), f"missing committed error table {EVIDENCE}"
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def _redteam(name: str) -> dict:
    return json.loads((ANCHOR_DIR / "redteam" / name).read_text(encoding="utf-8"))


def _codes(verdict: dict) -> set[str]:
    """All failure tags of a gate verdict, in a comparable form."""
    codes = set(verdict["failures"])
    for finding in verdict["checks"]["provenance"]["findings"]:
        codes.add(finding["code"])
    return codes


# ---------------------------------------------------------------------------
# 1. The registry is machine-readable and every entry is an external source
# ---------------------------------------------------------------------------


def test_registry_schema_and_files(registry: dict) -> None:
    assert registry["schema"] == gate_mod.REGISTRY_SCHEMA
    assert registry["anchors"], "registry must list at least one anchor"
    for entry in registry["anchors"]:
        assert (ANCHOR_DIR / entry["file"]).is_file()
        assert entry["source_class"] in gate_mod.EXTERNAL_SOURCE_CLASSES
        assert entry["status"] == "verified"


def test_registered_anchor_has_no_schema_findings(anchor: dict) -> None:
    findings = gate_mod.validate_record(anchor)
    assert findings == [], f"anchor fails the provenance schema: {findings}"


def test_anchor_carries_citation_identifier_and_locator(anchor: dict) -> None:
    source = anchor["source"]
    assert source["class"] == "LITERATURE_CLOSED_FORM"
    identifiers = source["identifiers"]
    assert any(gate_mod.identifier_ok(item) for item in identifiers), identifiers
    # the cited works must be bibliographically well formed, not free text
    assert any(item["type"] == "isbn" and gate_mod.isbn_checksum_ok(item["value"]) for item in identifiers)
    assert source["primary"]["locator"].strip()
    assert source["primary"]["year"] <= 2000  # a classical reference, not a self-citation


def test_reference_method_is_independent_of_the_package(anchor: dict) -> None:
    method = anchor["reference_method"]
    assert method["independent_of_repo_package"] is True
    lowered = method["implementation"].lower()
    for marker in gate_mod.REPO_MARKERS:
        assert marker.lower() not in lowered, (marker, method["implementation"])
    # the uncertainty of the reference must be stated
    assert method["uncertainty"]["relative"] <= 1e-6


# ---------------------------------------------------------------------------
# 2. Identifier validation is real validation (not a truthiness check)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["0-262-14026-8", "981-02-0421-3", "978-0-262-14026-3"])
def test_valid_isbns_pass_checksum(value: str) -> None:
    assert gate_mod.isbn_checksum_ok(value)


@pytest.mark.parametrize("value", ["0-262-14026-7", "981-02-0421-4", "978-0-262-14026-4", "1234"])
def test_invalid_isbns_fail_checksum(value: str) -> None:
    assert not gate_mod.isbn_checksum_ok(value)


def test_identifier_forms() -> None:
    assert gate_mod.doi_form_ok("10.1017/S0022112082001967")
    assert not gate_mod.doi_form_ok("S0022112082001967")
    assert gate_mod.url_form_ok("https://example.org/report.pdf")
    assert not gate_mod.url_form_ok("src/tensorbem/analytic.py")
    assert not gate_mod.identifier_ok({"type": "isbn", "value": "n/a"})
    assert not gate_mod.identifier_ok({"type": "handwave", "value": "trust me"})


# ---------------------------------------------------------------------------
# 3. The stored reference values follow from the cited equation
# ---------------------------------------------------------------------------


def test_stored_references_reproduce_from_the_closed_form(anchor: dict) -> None:
    check = gate_mod.check_reference_reproducible(anchor)
    assert check["passed"], check
    assert check["max_relative_deviation"] <= 1e-9
    assert len(check["rows"]) == len(anchor["quantities"])


def test_reference_is_not_the_package_analytic_module(anchor: dict) -> None:
    """The gate's reference must differ from ``tensorbem.analytic`` only by
    round-off -- two independent implementations of the same published formula.
    If they disagreed, the anchor would be grading the wrong equation."""
    import sys

    src = REPO_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from tensorbem.analytic import maccamy_fuchs_diffraction

    conditions = anchor["conditions"]
    for quantity in anchor["quantities"]:
        if quantity["kind"] != "force_amplitude":
            continue
        ka = quantity["case"]["ka"]
        package = complex(
            maccamy_fuchs_diffraction(
                ka,
                radius=conditions["radius_m"],
                depth=math.inf,
                amplitude=conditions["wave_amplitude_m"],
                density=conditions["rho_kg_m3"],
                gravity=conditions["g_m_s2"],
            )["force_x"]
        )
        independent = gate_mod.recompute_reference(anchor, quantity)
        assert abs(abs(package) - independent) / independent < 1e-6, quantity["id"]


# ---------------------------------------------------------------------------
# 4. Solver vs anchor: the 5 % verdict (and the tighter 1 % claim)
# ---------------------------------------------------------------------------


def test_all_quantities_pass_the_5pct_ledger_tolerance(anchor: dict, computed_32: dict) -> None:
    verdict = gate_mod.gate(anchor, computed_32)
    assert verdict["verdict"] == "PASS", verdict["failures"]
    for row in verdict["checks"]["tolerance"]["rows"]:
        assert row["pass_5pct"], row
        assert set(REQUIRED_ROW_KEYS) <= set(row)


def test_coarse_run_is_also_within_1pct(anchor: dict, computed_64: dict) -> None:
    """64 panels already beats 1 %: the anchor is discriminating, not loose."""
    numeric = gate_mod.check_tolerance(anchor, computed_64)
    assert numeric["passed"]
    assert max(row["error_pct"] for row in numeric["rows"]) < 1.0, numeric["rows"]


def test_error_trend_in_ka_is_u_shaped_not_monotone(computed_64: dict) -> None:
    """Measured trend, recorded so nobody re-invents a monotone story.

    At 64 panels the worst relative error sits at the *lowest* ka (0.05) because
    the force amplitude is largest there, dips by a factor ~2 around ka = 0.5 and
    rises again towards ka = 3 (finer wavelength per panel).  Both ends are far
    inside 1 %; the shape only matters for reading convergence studies.
    """
    errors = {
        qid: abs(computed_64[qid] - value) / value
        for qid, value in _force_references().items()
    }
    assert max(errors, key=errors.get) == "force_amplitude_ka_0.05", errors
    assert errors["force_amplitude_ka_3.00"] > 1.5 * errors["force_amplitude_ka_0.50"], errors


def _force_references() -> dict[str, float]:
    anchor = json.loads((ANCHOR_DIR / "maccamy_fuchs_cylinder.json").read_text(encoding="utf-8"))
    return {
        item["id"]: float(item["reference"])
        for item in anchor["quantities"]
        if item["kind"] == "force_amplitude"
    }


# ---------------------------------------------------------------------------
# 5. The committed error table is complete, self-consistent and matches the anchor
# ---------------------------------------------------------------------------


def test_error_table_rows_have_the_required_fields(table: dict) -> None:
    assert table["schema"] == "tensorbem.external_anchor_error_table/v1"
    assert table["anchor_id"] == "XANCHOR-BEM-MACCAMY-FUCHS-1954"
    assert table["quantities"], "error table must have rows"
    for row in table["quantities"]:
        assert set(REQUIRED_ROW_KEYS) <= set(row), row
        assert isinstance(row["pass_5pct"], bool)


def test_error_table_answers_the_5pct_question_explicitly(table: dict, anchor: dict) -> None:
    assert table["verdict"] == "PASS"
    assert table["tolerance_relative"] == pytest.approx(0.05)
    assert table["all_pass_5pct"] is True
    assert table["all_pass_1pct"] is True
    # explicit answers, not implied by the absence of failures
    for row in table["quantities"]:
        assert row["error_pct"] is not None
        assert row["pass_5pct"] == (row["error_pct"] <= 5.0)


def test_error_table_references_are_the_anchor_references(table: dict, anchor: dict) -> None:
    expected = {item["id"]: float(item["reference"]) for item in anchor["quantities"]}
    expected.update({item["id"]: float(item["reference"]) for item in anchor["limit_checks"]})
    seen = {}
    for row in table["quantities"]:
        seen[row["quantity"]] = row["reference"]
        assert row["reference"] == pytest.approx(expected[row["quantity"]], rel=1e-12)
    assert set(seen) == set(expected), "every anchor quantity must appear in the table"


def test_error_table_errors_recompute_from_its_own_numbers(table: dict) -> None:
    """No hand-edited ``error_pct``: each row must follow from computed/reference."""
    for row in table["quantities"]:
        expected = 100.0 * abs(row["computed"] - row["reference"]) / abs(row["reference"])
        assert row["error_pct"] == pytest.approx(expected, rel=1e-9, abs=1e-12)


def test_error_table_numbers_survive_a_fresh_run(table: dict, computed_64: dict) -> None:
    """Anti-drift: the committed 128-panel numbers must agree with a fresh run."""
    by_id = {row["quantity"]: row["computed"] for row in table["quantities"]}
    for qid, fresh in computed_64.items():
        assert qid in by_id, qid
        assert abs(fresh - by_id[qid]) / abs(by_id[qid]) < 5e-3, (qid, fresh, by_id[qid])


def test_error_table_records_solver_provenance(table: dict) -> None:
    solver = table["solver"]
    assert solver["route"] == "tensorbem.helmholtz.solve_vertical_cylinder_diffraction"
    assert solver["panels"] == 128
    assert solver["water_depth_over_radius"] == "inf"
    assert table["reference"]["source_class"] == "LITERATURE_CLOSED_FORM"
    assert table["reference"]["identifiers"], "the table must carry the citation"


def test_convergence_against_the_external_anchor_is_monotone(table: dict) -> None:
    rows = table["convergence"]
    assert [row["panels"] for row in rows] == [32, 64, 128]
    errors = [row["error_pct"] for row in rows]
    assert errors == sorted(errors, reverse=True), errors  # refinement reduces the error
    assert errors[-1] < 0.1


def test_artifacts_copy_matches_the_tracked_evidence(table: dict) -> None:
    """The two paths must not drift: same bytes, or the artifacts copy is absent."""
    if not ARTIFACT.exists():
        pytest.skip("artifacts/ copy not generated in this worktree")
    assert json.loads(ARTIFACT.read_text(encoding="utf-8")) == table


def test_figure_exists_and_is_not_empty() -> None:
    assert FIGURE.is_file(), f"missing figure {FIGURE}"
    assert FIGURE.stat().st_size > 10_000


# ---------------------------------------------------------------------------
# 6. Red team: each fixture must FAIL, with the expected rule named
# ---------------------------------------------------------------------------


def test_redteam_provenance_swap_fails_and_numbers_alone_would_not(anchor: dict, computed_32: dict) -> None:
    """Report 16348 reproduced: the reference is the repo's own series.

    The numbers still reproduce from the in-repo-derived reference (proving a
    pure tolerance gate is blind to it); only the provenance rule rejects it.
    """
    record = _redteam("provenance_swap_hulme_in_repo.json")
    verdict = gate_mod.gate(record, computed_32)
    assert verdict["verdict"] == "FAIL"
    codes = _codes(verdict)
    assert "provenance_in_repo_derived" in codes, codes
    assert "provenance_points_into_repo" in codes, codes
    # the trap: the numeric side is clean
    assert verdict["checks"]["reference_reproducible"]["passed"] is True
    assert _redteam_expected("provenance_swap_hulme_in_repo.json") == "FAIL"


def test_redteam_value_tamper_fails_reproducibility_and_tolerance(anchor: dict, computed_32: dict) -> None:
    record = _redteam("value_tamper_10pct.json")
    verdict = gate_mod.gate(record, computed_32)
    assert verdict["verdict"] == "FAIL"
    codes = _codes(verdict)
    assert "reference:not_reproducible_from_cited_equation" in codes, codes
    assert "tolerance:exceeded" in codes, codes
    tampered = next(r for r in verdict["checks"]["tolerance"]["rows"] if r["quantity"] == "force_amplitude_ka_1.00")
    assert tampered["error_pct"] > 9.0
    assert tampered["pass_5pct"] is False


def test_redteam_missing_identifier_fails_provenance(anchor: dict, computed_32: dict) -> None:
    verdict = gate_mod.gate(_redteam("missing_identifier.json"), computed_32)
    assert verdict["verdict"] == "FAIL"
    assert "missing_external_identifier" in _codes(verdict)


def test_redteam_reference_sharing_the_solver_fails_provenance(anchor: dict, computed_32: dict) -> None:
    verdict = gate_mod.gate(_redteam("reference_shares_solver.json"), computed_32)
    assert verdict["verdict"] == "FAIL"
    codes = _codes(verdict)
    assert "reference_method_not_independent" in codes, codes
    assert "provenance_points_into_repo" in codes, codes


def test_redteam_loosened_tolerance_fails_provenance(anchor: dict, computed_32: dict) -> None:
    verdict = gate_mod.gate(_redteam("loose_tolerance.json"), computed_32)
    assert verdict["verdict"] == "FAIL"
    assert "tolerance_invalid" in _codes(verdict)


def test_redteam_registry_lists_every_fixture_with_its_expectation(registry: dict) -> None:
    listed = {entry["file"] for entry in registry["redteam"]}
    on_disk = {f"redteam/{path.name}" for path in (ANCHOR_DIR / "redteam").glob("*.json")}
    assert listed == on_disk
    for entry in registry["redteam"]:
        record = _redteam(Path(entry["file"]).name)
        assert record["_redteam"]["expect"] == entry["expect"] == "FAIL"


def _redteam_expected(name: str) -> str:
    return _redteam(name)["_redteam"]["expect"]


# ---------------------------------------------------------------------------
# 7. Positive control: a perturbed *solver* value fails tolerance only
# ---------------------------------------------------------------------------


def test_perturbed_solver_value_fails_tolerance_only(anchor: dict, computed_32: dict) -> None:
    shaken = dict(computed_32)
    shaken["force_amplitude_ka_1.00"] = shaken["force_amplitude_ka_1.00"] * 1.06
    verdict = gate_mod.gate(anchor, shaken)
    assert verdict["verdict"] == "FAIL"
    assert verdict["failures"] == ["tolerance:exceeded"], verdict["failures"]
    assert verdict["checks"]["provenance"]["passed"] is True
    assert verdict["checks"]["reference_reproducible"]["passed"] is True
    row = next(r for r in verdict["checks"]["tolerance"]["rows"] if r["quantity"] == "force_amplitude_ka_1.00")
    assert row["error_pct"] > 5.0


@pytest.mark.parametrize(
    "mutate, code",
    [
        (lambda r: r["reference_method"].__setitem__("id", "trust_me"), "reference_method_unknown"),
        (lambda r: r["quantities"][0].__setitem__("kind", "whatever"), "quantity_kind_unknown"),
    ],
)
def test_gate_fails_closed_on_records_it_cannot_regrade(anchor: dict, computed_32: dict, mutate, code) -> None:
    """An unregistered reference method or quantity kind must verdict FAIL, not
    raise: a gate that crashes is a gate an operator can ignore."""
    import copy

    record = copy.deepcopy(anchor)
    mutate(record)
    verdict = gate_mod.gate(record, computed_32)  # must not raise
    assert verdict["verdict"] == "FAIL"
    assert code in _codes(verdict), _codes(verdict)


def test_gate_does_not_raise_on_an_empty_record(computed_32: dict) -> None:
    verdict = gate_mod.gate({}, computed_32)
    assert verdict["verdict"] == "FAIL"
    assert "missing_source" in _codes(verdict)


def test_missing_computed_value_is_a_failure_not_a_skip(anchor: dict, computed_32: dict) -> None:
    partial = dict(computed_32)
    del partial["force_amplitude_ka_0.10"]
    verdict = gate_mod.gate(anchor, partial)
    assert verdict["verdict"] == "FAIL"
    row = next(r for r in verdict["checks"]["tolerance"]["rows"] if r["quantity"] == "force_amplitude_ka_0.10")
    assert row["computed"] is None
    assert row["pass_5pct"] is False