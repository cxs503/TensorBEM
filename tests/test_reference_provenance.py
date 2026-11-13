"""Fail-closed gates on the provenance of literature reference constants.

The capability registry grades code, tests and documents; it never looks at the
constants a driver compares against.  ``tests/test_reference_provenance.py``
closes that gap for ``tensorbem.reference_provenance``:

* on the real tree, every module-level constant that announces a literature
  origin (WAMIT / JOURNEE / EFD / ...) must be triaged, and every ``CITED``
  citation must still resolve to a shipped document that publishes the digits;
* the block published in ``docs/reference_provenance.md`` must equal the block
  generated from the tree, so the debt table cannot be edited by hand;
* a recalled constant must not be *read* by anything except its registered
  adjudicator (the reader graph over ``src/``, ``tests/`` and ``scripts/``), and
  that adjudicator must still read it and still publish a verdict for it in
  ``benchmarks/verified/wigley3_recalled_ref/results.json``;
* the scanner must *discriminate*: a synthetic package proves that an
  un-triaged recalled constant is caught, that a marker occurring only in a
  comment/string is not mistaken for a constant, that a mapping of numbers is
  caught as well as a scalar, and that a citation pointing at a document that
  does not publish the value is rejected.
"""

from __future__ import annotations

import pathlib

import pytest

from tensorbem import reference_provenance as rp


# --------------------------------------------------------------------------
# gates on the real tree
# --------------------------------------------------------------------------


def test_no_untriaged_reference_constant_exists() -> None:
    result = rp.audit()
    assert result.undeclared == [], [
        (declaration.key, declaration.line) for declaration in result.undeclared
    ]


def test_no_registry_entry_is_stale() -> None:
    result = rp.audit()
    assert result.stale == [], result.stale


def test_every_citation_still_resolves() -> None:
    result = rp.audit()
    assert result.unverified == [], result.unverified


def test_published_block_matches_the_tree() -> None:
    drift = rp.document_drift()
    assert drift is None, (
        "docs/reference_provenance.md no longer matches the tree; regenerate it with"
        " `python -m tensorbem.reference_provenance --write`\n" + (drift or "")
    )


def test_unsourced_debt_is_published_not_silent() -> None:
    """The known debtor surface is stated in the document, not only in code."""

    result = rp.audit()
    text = (rp.repo_root() / rp.DOCUMENT).read_text(encoding="utf-8")
    for declaration in result.unsourced:
        assert declaration.name in text, declaration.key
        assert rp.DECLARED[declaration.key].note.strip(), declaration.key
    # the registry never grades a recalled constant as cited
    for key, entry in rp.DECLARED.items():
        if entry.status == rp.CITED:
            assert entry.evidence, key


def test_the_known_legacy_wigley_debt_is_still_registered() -> None:
    """Regression pin: the legacy drivers' recalled tables stay visible as debt.

    The keys are read from the registry instead of being typed here: spelling a
    benchmark module path in a test file would make the capability registry's
    "untested module" index believe that driver is exercised by this suite.
    """

    result = rp.audit()
    unsourced = {declaration.key for declaration in result.unsourced}
    registered = {
        key for key, entry in rp.DECLARED.items() if entry.status == rp.UNSOURCED
    }
    assert registered <= unsourced, sorted(registered - unsourced)
    assert len(registered) >= 14
    assert all(rp.DECLARED[key].note.strip() for key in registered)


# --------------------------------------------------------------------------
# negative controls on a synthetic package
# --------------------------------------------------------------------------


@pytest.fixture()
def synthetic_repo(tmp_path: pathlib.Path) -> pathlib.Path:
    (tmp_path / "src" / "tensorbem").mkdir(parents=True)
    (tmp_path / "src" / "tensorbem" / "capability_evidence.py").write_text(
        "# placeholder so repo_root() accepts the synthetic tree\n", encoding="utf-8"
    )
    return tmp_path


def _write(root: pathlib.Path, name: str, body: str) -> None:
    (root / "src" / "tensorbem" / name).write_text(body, encoding="utf-8")


def test_negative_control_catches_a_recalled_scalar(synthetic_repo: pathlib.Path) -> None:
    _write(synthetic_repo, "driver.py", "_WAMIT_A33 = 0.42\n")
    result = rp.audit(synthetic_repo, declared={})
    assert [d.name for d in result.undeclared] == ["_WAMIT_A33"]


def test_negative_control_catches_a_recalled_mapping(synthetic_repo: pathlib.Path) -> None:
    _write(
        synthetic_repo,
        "driver.py",
        "JOURNEE_RAO = {\n    0.5: 1.10,\n    0.7: 1.15,\n}\n",
    )
    result = rp.audit(synthetic_repo, declared={})
    assert [d.name for d in result.undeclared] == ["JOURNEE_RAO"]
    assert result.undeclared[0].kind == "mapping"


def test_negative_control_ignores_prose_and_algorithmic_constants(
    synthetic_repo: pathlib.Path,
) -> None:
    """A marker in a comment or a name without numbers is not a reference table."""

    _write(
        synthetic_repo,
        "driver.py",
        '"""Recalled WAMIT numbers would go here; none are typed."""\n'
        "# WAMIT_FOO = 0.42  (commented out, so not a declaration)\n"
        "WAMIT_SOURCE = 'WAMIT v6 verification case'\n"
        "GAUSS_ABSCISSA = 0.5773502691896257\n"
        "def f():\n"
        "    EFD_PEAK = 1.15  # local, an algorithm detail, not a module-level reference\n"
        "    return EFD_PEAK\n",
    )
    assert rp.scan(synthetic_repo) == []


def test_negative_control_rejects_a_citation_that_does_not_publish_the_value(
    synthetic_repo: pathlib.Path,
) -> None:
    _write(synthetic_repo, "driver.py", "_WAMIT_A33 = 0.42\n")
    (synthetic_repo / "docs").mkdir()
    (synthetic_repo / "docs" / "report.md").write_text("the added-mass ratio is 0.42\n", encoding="utf-8")
    declared = {
        "tensorbem.driver._WAMIT_A33": rp.Entry(
            status=rp.CITED,
            evidence=(("docs/report.md", "0.42"),),
            note="published in the synthetic report",
        )
    }
    assert rp.audit(synthetic_repo, declared=declared).unverified == []

    bad = {
        "tensorbem.driver._WAMIT_A33": rp.Entry(
            status=rp.CITED,
            evidence=(("docs/report.md", "0.43"),),
            note="points at digits the document does not publish",
        )
    }
    result = rp.audit(synthetic_repo, declared=bad)
    assert len(result.unverified) == 1
    assert result.ok is False


def test_negative_control_flags_a_stale_registry_entry(synthetic_repo: pathlib.Path) -> None:
    _write(synthetic_repo, "driver.py", "x = 1.0\n")
    declared = {
        "tensorbem.driver._JOURNEE_GONE": rp.Entry(
            status=rp.UNSOURCED, note="registered before the driver was deleted"
        )
    }
    assert rp.audit(synthetic_repo, declared=declared).stale == ["tensorbem.driver._JOURNEE_GONE"]


def test_an_entry_must_carry_a_note_and_cited_entries_evidence() -> None:
    with pytest.raises(ValueError):
        rp.Entry(status=rp.UNSOURCED, note="")
    with pytest.raises(ValueError):
        rp.Entry(status=rp.CITED, note="no evidence")
    with pytest.raises(ValueError):
        rp.Entry(status="PROBABLY_FINE", note="not a status")


# --------------------------------------------------------------------------
# gates on who reads a recalled number
# --------------------------------------------------------------------------


def test_no_recalled_constant_is_read_by_anything_but_its_adjudicator() -> None:
    """Debt is only debt while nothing downstream of it publishes a number."""

    result = rp.audit()
    assert result.leaked == [], [(key, str(reader)) for key, reader in result.leaked]


def test_every_registered_adjudicator_still_reads_its_constant() -> None:
    result = rp.audit()
    assert result.unread_adjudicator == [], result.unread_adjudicator
    # the allow-list only makes sense for constants the registry calls UNSOURCED
    unsourced = {declaration.key for declaration in result.unsourced}
    assert set(rp.ADJUDICATED) <= unsourced, sorted(set(rp.ADJUDICATED) - unsourced)


def test_every_adjudicated_constant_carries_a_verdict() -> None:
    result = rp.audit()
    assert result.unadjudicated == [], result.unadjudicated
    assert result.adjudication, "no verdict was read from " + rp.ADJUDICATION_ARTEFACT
    for key, record in result.adjudication.items():
        assert record.records, key
        assert record.describe().strip(), key
        assert all(verdict for verdict, _ in record.records), key


def test_the_cited_constant_is_read_by_exactly_two_importers() -> None:
    """Pin on the only CITED constant's reader set: a new consumer is a new claim.

    Counts and readers are read from the graph rather than typed here, so the
    pin does not spell a benchmark module path in the suite.
    """

    result = rp.audit()
    cited = [declaration for declaration in result.cited]
    assert len(cited) == 1, [declaration.key for declaration in cited]
    found = result.readers[cited[0].key]
    assert len(found) == 2, [str(reader) for reader in found]
    assert {reader.how for reader in found} == {"import"}


def test_a_recalled_constant_may_be_read_only_by_a_registered_adjudicator(
    synthetic_repo: pathlib.Path,
) -> None:
    """Positive control for the reader gate: the registry turns a leak green."""

    _write(synthetic_repo, "driver.py", "_WAMIT_A33 = 0.42\n")
    (synthetic_repo / "tests").mkdir()
    (synthetic_repo / "tests" / "probe.py").write_text(
        "from tensorbem.driver import _WAMIT_A33\n\nvalue = _WAMIT_A33\n", encoding="utf-8"
    )
    declared = {
        "tensorbem.driver._WAMIT_A33": rp.Entry(
            status=rp.UNSOURCED, note="recalled, source report not shipped"
        )
    }
    artefact = synthetic_repo / rp.ADJUDICATION_ARTEFACT
    artefact.parent.mkdir(parents=True)
    artefact.write_text(
        '{"adjudication": [{"constant": "tensorbem.driver._WAMIT_A33",'
        ' "verdict": "REFUTED", "factor": 0.31}]}',
        encoding="utf-8",
    )

    # 1. read by a consumer that is not registered: that consumer is a leak
    unregistered = rp.audit(synthetic_repo, declared=declared, adjudicated={})
    assert [(key, reader.consumer) for key, reader in unregistered.leaked] == [
        ("tensorbem.driver._WAMIT_A33", "tests/probe.py")
    ]
    assert unregistered.ok is False

    # 2. registered as its adjudicator: no leak, verdict still read back
    registered = rp.audit(
        synthetic_repo,
        declared=declared,
        adjudicated={"tensorbem.driver._WAMIT_A33": "tests/probe.py"},
    )
    assert registered.leaked == []
    assert registered.unread_adjudicator == []
    assert registered.adjudication["tensorbem.driver._WAMIT_A33"].counts == {"REFUTED": 1}
    assert registered.ok is True

    # 3. the adjudicator stops reading it: registry rot, not a pass
    (synthetic_repo / "tests" / "probe.py").write_text("value = 1.0\n", encoding="utf-8")
    rotted = rp.audit(
        synthetic_repo,
        declared=declared,
        adjudicated={"tensorbem.driver._WAMIT_A33": "tests/probe.py"},
    )
    assert rotted.unread_adjudicator == ["tensorbem.driver._WAMIT_A33"]
    assert rotted.ok is False

    # 4. the artefact stops carrying the verdict: no verdict, no pass
    artefact.write_text('{"adjudication": []}', encoding="utf-8")
    (synthetic_repo / "tests" / "probe.py").write_text(
        "from tensorbem.driver import _WAMIT_A33\n\nvalue = _WAMIT_A33\n", encoding="utf-8"
    )
    unverifiable = rp.audit(
        synthetic_repo,
        declared=declared,
        adjudicated={"tensorbem.driver._WAMIT_A33": "tests/probe.py"},
    )
    assert unverifiable.unadjudicated == ["tensorbem.driver._WAMIT_A33"]
    assert unverifiable.ok is False
