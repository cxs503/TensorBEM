"""Gates for the numeric evidence layer.

The structural capability audit can say ``REAL`` for a capability whose number
misses its own criterion; ``tensorbem.numeric_evidence`` exists to make that
visible and to keep it visible.  These tests therefore do two things:

* re-derive every committed item's error **with an independent pointer walker
  written here**, and require the module's verdict to agree with that number --
  a table that is merely executed is not evidence;
* build synthetic registries in ``tmp_path`` where the artifact and the claim
  disagree (overstated error field, tampered digest, tightened tolerance,
  missing artifact, unregistered gap) and require the grader to fail each one
  closed.  Without that negative control a green gate would only prove the gate
  is agreeable.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from tensorbem import numeric_evidence as ne


# ---------------------------------------------------------------------------
# an independent resolver: deliberately not ne.resolve_value
# ---------------------------------------------------------------------------


def _walk(document, pointer: str):
    node = document
    for token in [t for t in pointer.split("/") if t]:
        node = node[int(token)] if isinstance(node, list) else node[token]
    return node


def _pick(document, selector):
    if isinstance(selector, str):
        return _walk(document, selector)
    rows = _walk(document, selector["array"])
    hits = [r for r in rows if all(r.get(k) == v for k, v in selector.get("match", {}).items())]
    assert len(hits) == 1, selector
    value = hits[0]
    for token in [t for t in selector.get("field", "").split("/") if t]:
        value = value[token]
    return value


def _recompute(root: pathlib.Path, item: dict) -> float:
    document = json.loads((root / item["artifact"]).read_text(encoding="utf-8"))
    computed = float(_pick(document, item["computed"]))
    reference = (
        float(_pick(document, item["reference"]))
        if "reference" in item
        else float(item["reference_value"])
    )
    return abs(computed - reference) / abs(reference) * 100.0


@pytest.fixture(scope="module")
def root() -> pathlib.Path:
    return ne.repo_root()


@pytest.fixture(scope="module")
def registry(root: pathlib.Path) -> dict:
    return json.loads((root / ne.REGISTRY).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# gates on the real tree
# ---------------------------------------------------------------------------


def test_gate_is_green(root: pathlib.Path) -> None:
    assert ne.check(root) == []


def test_every_item_regrades_to_the_independently_recomputed_number(
    root: pathlib.Path, registry: dict
) -> None:
    rows = {row.item_id: row for row in ne.audit(root)}
    checked = 0
    for item in registry["items"]:
        row = rows[item["id"]]
        if row.error_pct is None:  # artifact missing: nothing to recompute
            break
        independent = _recompute(root, item)
        assert row.error_pct == pytest.approx(independent, rel=1e-9, abs=0.0), item["id"]
        checked += 1
    assert checked >= 8


def test_no_real_item_hides_an_overrun(root: pathlib.Path, registry: dict) -> None:
    items = {item["id"]: item for item in registry["items"]}
    for row in ne.audit(root):
        if row.level != ne.REAL:
            continue
        independent = _recompute(root, items[row.item_id])
        assert independent <= items[row.item_id]["tolerance_pct"], row.item_id


def test_honest_gaps_are_registered_and_visible(root: pathlib.Path) -> None:
    rows = ne.audit(root)
    gaps = [row for row in rows if row.level == ne.WITHHELD]
    assert gaps, "the KP505 tolerance gap must not vanish from the record"
    for row in gaps:
        assert row.expected_level == ne.WITHHELD
        assert row.gap_note
    text = (root / ne.DOCUMENT).read_text(encoding="utf-8")
    assert "Accepted gaps" in text
    for row in gaps:
        assert row.item_id in text


def test_structural_and_numeric_layers_are_crossed(root: pathlib.Path) -> None:
    rows = ne.audit(root)
    crossing = ne.contradictions(rows)
    keys = {entry["capability"] for entry in crossing}
    # propeller-open-water is structurally REAL and numerically WITHHELD: that
    # disagreement is the reason this layer exists, so it must be reported.
    assert "propeller-open-water" in keys
    for entry in crossing:
        assert entry["structural_level"] != entry["numeric_level"]


def test_published_matrix_matches_the_registry(root: pathlib.Path) -> None:
    assert ne.document_drift(root) is None


def test_registry_capabilities_exist_in_the_structural_layer(
    root: pathlib.Path, registry: dict
) -> None:
    known = set(ne.structural_capabilities())
    unknown = [item["capability"] for item in registry["items"] if item["capability"] not in known]
    assert unknown == []


# ---------------------------------------------------------------------------
# negative controls on synthetic registries
# ---------------------------------------------------------------------------


def _synthetic_root(tmp_path: pathlib.Path, *, computed: float, reference: float,
                    tolerance: float, reported: float | None = None,
                    capability: str = "hydrostatics", expected: str | None = "REAL") -> pathlib.Path:
    fake = tmp_path / "repo"
    (fake / "src" / "tensorbem").mkdir(parents=True)
    (fake / "pyproject.toml").write_text("[project]\nname='tensorbem'\n", encoding="utf-8")
    artifact = fake / "artifacts" / "case.json"
    artifact.parent.mkdir(parents=True)
    payload = {"measured": {"value": computed}, "analytic": reference}
    if reported is not None:
        payload["reported_relative_error"] = reported
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    item = {
        "id": "synthetic-item",
        "capability": capability,
        "criterion": "synthetic criterion for the negative control",
        "tolerance_pct": tolerance,
        "artifact": "artifacts/case.json",
        "computed": "/measured/value",
        "reference": "/analytic",
    }
    if reported is not None:
        item["reported_error_ratio"] = "/reported_relative_error"
    if expected is not None:
        item["expected_level"] = expected
        item["gap_note"] = "synthetic gap note"
    (fake / ne.REGISTRY).parent.mkdir(parents=True, exist_ok=True)
    (fake / ne.REGISTRY).write_text(
        json.dumps({"schema_version": ne.SCHEMA, "items": [item]}), encoding="utf-8"
    )
    return fake


def test_within_tolerance_is_real(tmp_path: pathlib.Path) -> None:
    fake = _synthetic_root(tmp_path, computed=100.5, reference=100.0, tolerance=1.0)
    rows = ne.audit(fake, capabilities=["hydrostatics"])
    assert rows[0].level == ne.REAL
    assert rows[0].error_pct == pytest.approx(0.5, rel=1e-12)


def test_overrun_is_withheld_not_passed(tmp_path: pathlib.Path) -> None:
    fake = _synthetic_root(tmp_path, computed=110.0, reference=100.0, tolerance=1.0,
                           expected=ne.WITHHELD)
    rows = ne.audit(fake, capabilities=["hydrostatics"])
    assert rows[0].level == ne.WITHHELD
    assert "exceeds the declared tolerance" in rows[0].reasons[0]


def test_tightening_the_tolerance_flips_a_green_item(tmp_path: pathlib.Path) -> None:
    fake = _synthetic_root(tmp_path, computed=100.5, reference=100.0, tolerance=1.0)
    assert ne.audit(fake, capabilities=["hydrostatics"])[0].level == ne.REAL
    registry = json.loads((fake / ne.REGISTRY).read_text(encoding="utf-8"))
    registry["items"][0]["tolerance_pct"] = 0.1
    registry["items"][0]["expected_level"] = ne.WITHHELD
    (fake / ne.REGISTRY).write_text(json.dumps(registry), encoding="utf-8")
    row = ne.audit(fake, capabilities=["hydrostatics"])[0]
    assert row.level == ne.WITHHELD, "the grader must read the tolerance, not the artifact"


def test_artifact_error_field_that_contradicts_its_own_numbers_is_rejected(
    tmp_path: pathlib.Path,
) -> None:
    # raw numbers say 10%, the artifact's own field claims 0.0
    fake = _synthetic_root(tmp_path, computed=110.0, reference=100.0, tolerance=5.0,
                           reported=0.0, expected=ne.WITHHELD)
    row = ne.audit(fake, capabilities=["hydrostatics"])[0]
    assert row.level == ne.WITHHELD
    assert "disagrees" in row.reasons[0]


def test_agreeing_artifact_error_field_does_not_change_the_verdict(
    tmp_path: pathlib.Path,
) -> None:
    fake = _synthetic_root(tmp_path, computed=110.0, reference=100.0, tolerance=15.0,
                           reported=0.10)
    row = ne.audit(fake, capabilities=["hydrostatics"])[0]
    assert row.level == ne.REAL
    assert row.reported_error_pct == pytest.approx(10.0, rel=1e-9)


def test_tampered_artifact_is_stale(tmp_path: pathlib.Path) -> None:
    fake = _synthetic_root(tmp_path, computed=100.0, reference=100.0, tolerance=1.0)
    registry = json.loads((fake / ne.REGISTRY).read_text(encoding="utf-8"))
    registry["items"][0]["sha256"] = ne.sha256_file(fake / "artifacts" / "case.json")
    (fake / ne.REGISTRY).write_text(json.dumps(registry), encoding="utf-8")
    assert ne.audit(fake, capabilities=["hydrostatics"])[0].level == ne.REAL
    (fake / "artifacts" / "case.json").write_text(
        json.dumps({"measured": {"value": 100.0}, "analytic": 100.0, "note": "edited"}),
        encoding="utf-8",
    )
    row = ne.audit(fake, capabilities=["hydrostatics"])[0]
    assert row.level == ne.STALE
    assert "digest" in row.reasons[0]


def test_missing_artifact_is_missing(tmp_path: pathlib.Path) -> None:
    fake = _synthetic_root(tmp_path, computed=100.0, reference=100.0, tolerance=1.0)
    (fake / "artifacts" / "case.json").unlink()
    assert ne.audit(fake, capabilities=["hydrostatics"])[0].level == ne.MISSING


def test_unknown_capability_is_refused(tmp_path: pathlib.Path) -> None:
    fake = _synthetic_root(tmp_path, computed=100.0, reference=100.0, tolerance=1.0,
                           capability="not-a-published-capability")
    row = ne.audit(fake, capabilities=["hydrostatics"])[0]
    assert row.level == ne.WITHHELD
    assert "not published" in row.reasons[0]


def test_unregistered_gap_makes_the_gate_red(tmp_path: pathlib.Path) -> None:
    fake = _synthetic_root(tmp_path, computed=110.0, reference=100.0, tolerance=1.0,
                           expected=None)
    row = ne.audit(fake, capabilities=["hydrostatics"])[0]
    assert row.level == ne.WITHHELD
    problems = ne.check(fake, registry_path=fake / ne.REGISTRY)
    assert any("no registered expectation" in problem for problem in problems)


def test_registered_real_that_slips_to_withheld_makes_the_gate_red(
    tmp_path: pathlib.Path,
) -> None:
    fake = _synthetic_root(tmp_path, computed=100.5, reference=100.0, tolerance=1.0)
    registry = json.loads((fake / ne.REGISTRY).read_text(encoding="utf-8"))
    registry["items"][0]["tolerance_pct"] = 1e-9  # a regression the record should catch
    (fake / ne.REGISTRY).write_text(json.dumps(registry), encoding="utf-8")
    problems = ne.check(fake, registry_path=fake / ne.REGISTRY)
    assert any("registry expects REAL" in problem for problem in problems)


def test_cli_check_returns_zero_on_the_real_tree() -> None:
    assert ne.main(["--check"]) == 0