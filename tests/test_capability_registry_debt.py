"""Capability-ledger debt guard for the modules that used to have no row.

Before m16384 twenty-six core physics modules were parked in
``AUXILIARY_MODULES`` with a deferred verdict:

    "physics/geometry module <leaf> is pinned by tests/<file>.py ... it has no
     row of its own yet (debt registered in docs/m14869/report.md), so its
     level is NOT claimed here."

A reader of the published table therefore could not tell what evidence backs a
quarter of the package.  m16384 promoted twenty-five of those modules into
graded rows (twenty-one rows) after re-deriving, from the AST, that each one is
actually called *inside an assertion* by the test file the note cites --
``scripts/audit_module_pins.py`` writes that matrix to
``docs/capability_pin_audit.json``.

This test keeps the debt from silently coming back:

* the deferred-verdict sentence may only survive for modules listed in
  ``DOC_ANCHOR_DEBT`` with a stated reason (today: one module whose note says
  the repository carries no documentation anchor for it at all);
* no module may be graded by a row *and* excused as auxiliary;
* every module named in the pin-audit matrix must have at least one test node
  that calls its public API inside an assertion (otherwise the matrix is
  quoting a smoke-only file and the "pinned by tests" wording is unsupported);
* the registry must still derive a non-failing level for every row.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from tensorbem import capability_evidence as ce

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Modules whose deferred verdict is kept, with the reason it is kept.
DOC_ANCHOR_DEBT: dict[str, str] = {}
"""Empty since m16708: the last deferred module (pekeris_acoustics) got its
document anchor (docs/pekeris_w22_cross_route/report.md) and a graded row."""

DEFERRED_SENTENCE = "no row of its own"


def test_deferred_verdicts_are_all_discharged() -> None:
    offenders = {
        name: note
        for name, note in ce.AUXILIARY_MODULES.items()
        if DEFERRED_SENTENCE in note and name not in DOC_ANCHOR_DEBT
    }
    assert not offenders, (
        "these modules still defer their evidence level; either promote them to a "
        f"graded row or register the blocking reason in DOC_ANCHOR_DEBT: {sorted(offenders)}"
    )


def test_doc_anchor_debt_still_holds() -> None:
    """The one surviving deferral must still be true: no doc anchor exists."""

    for name in DOC_ANCHOR_DEBT:
        assert name in ce.AUXILIARY_MODULES, f"{name} is no longer auxiliary"
        reason = DOC_ANCHOR_DEBT[name]
        assert "documentation anchor" in reason
        assert name in ce.AUXILIARY_MODULES


def test_no_module_is_both_graded_and_auxiliary() -> None:
    graded = {mod for row in ce.CAPABILITIES for mod in row.modules}
    overlap = sorted(graded & set(ce.AUXILIARY_MODULES))
    assert not overlap, f"module graded by a row and excused as auxiliary: {overlap}"


def test_pin_audit_matrix_shows_asserting_coverage() -> None:
    matrix_path = ROOT / "docs" / "capability_pin_audit.json"
    assert matrix_path.is_file(), (
        "run `python scripts/audit_module_pins.py --json docs/capability_pin_audit.json`"
    )
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    smoke_only = sorted(
        leaf for leaf, rec in matrix.items() if not rec["asserting_nodes"]
    )
    assert not smoke_only, (
        "these modules are only called by tests without an assertion, so any "
        f"'pinned by tests' wording is unsupported: {smoke_only}"
    )


def test_every_row_still_derives_an_acceptable_level() -> None:
    failing = [row.capability.key for row in ce.audit() if not row.ok]
    assert not failing, f"rows whose evidence no longer resolves: {failing}"


#: Rows that still derive WITHHELD after promotion, because *every* test that
#: asserts their API is a method inside a ``class Test...`` and the registry's
#: node index (``_test_nodes``) only lists module-level ``def test_*`` nodes.
#: The evidence exists; the registry cannot address it.  Recorded, not hidden.
CLASS_ONLY_TESTS = {
    "figure-integrity": "tests/test_figure_integrity.py::TestCLI::*",
    "finite-depth-green-kernel": "tests/test_finite_depth_green.py::Test*",
    "haskind-2d-excitation": "tests/test_stf_haskind_reg.py::TestHaskindCrossValidation::*",
    "neumann-kelvin-steady": "tests/test_neumann_kelvin.py::Test*",
    "rankine-wave-series60": "tests/test_rankine_wave_resistance.py::Test*",
    "stf-fullspeed-strip": "tests/test_stf_haskind_reg.py::Test*",
}


def test_promoted_rows_are_derived_real() -> None:
    """The m16384 promotions are REAL *because* they resolve, not by declaration."""

    promoted = {
        "cathodic-protection-bem",
        "cylinder-and-farfield-drift",
        "figure-integrity",
        "finite-depth-green-kernel",
        "finite-depth-3d-quadrature-solver",
        "flexural-gravity-dispersion",
        "haskind-2d-excitation",
        "interior-panel-lid",
        "irregular-frequency-sanitisation",
        "neumann-kelvin-steady",
        "nk-panel-3d",
        "quadratic-panel-operator",
        "propeller-open-water",
        "rankine-fs-panel",
        "rankine-wave-series60",
        "reference-provenance",
        "state-space-realization",
        "stf-fullspeed-strip",
        "tank-free-surface",
        "wigley3-hullform",
    }
    by_key = {row.capability.key: row for row in ce.audit()}
    missing = sorted(promoted - set(by_key))
    assert not missing, f"promoted rows disappeared from the registry: {missing}"
    not_real = sorted(key for key in promoted if by_key[key].level != ce.REAL)
    # only the class-only-test rows may stay WITHHELD, and only for that reason
    unexpected = [key for key in not_real if key not in CLASS_ONLY_TESTS]
    assert not unexpected, (
        "these promoted rows no longer derive REAL: "
        f"{ {key: by_key[key].level for key in unexpected} }"
    )
    for key in not_real:
        assert "named test node missing" in " ".join(by_key[key].reasons), (
            f"{key} is WITHHELD for an unexpected reason: {by_key[key].reasons}"
        )
    assert len(promoted) - len(not_real) >= 14


#: Rows whose subject *is* the baseline document itself; citing it is not circular.
META_ROWS = {"benchmark-cli-contract", "capability-evidence"}


def test_rows_have_documentation_anchors_that_are_not_the_registry_itself() -> None:
    """A physics row may not cite the generated baseline block as its own evidence."""

    cyclic = sorted(
        row.key
        for row in ce.CAPABILITIES
        if row.key not in META_ROWS
        for path, _anchor in row.docs
        if path == "docs/audit_baseline.md"
    )
    assert not cyclic, f"rows citing the generated baseline as evidence: {cyclic}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))