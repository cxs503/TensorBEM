"""Fail-closed gates for the audit baseline's capability table.

``docs/audit_baseline.md`` section 4 publishes, capability by capability, what
this package implements and at what evidence level.  Until mission m3336 that
table was hand-written: it still called finite-depth kernels, forward-speed
wave resistance and Kutta lifting wakes ``NOT_SUPPORTED`` long after those
modules shipped with their own suites.  ``tensorbem.capability_evidence``
derives the table from code/test/doc evidence instead; the tests here make the
derivation itself testable:

* the real tree must contain no failing evidence level and no unexplained
  untested module;
* the block published in the document must equal the block generated from the
  tree, so editing the document by hand goes red;
* the grader must *discriminate* -- a synthetic repository is built with a
  stubbed module, a module with unseeded randomness, a constant-returning
  function, a missing module, a documented capability and an undocumented one,
  and each must come out at the expected level.  Without that negative control
  a green gate would only prove the gate is agreeable.
"""

from __future__ import annotations

import pathlib

import pytest

from tensorbem import capability_evidence as ce


# --------------------------------------------------------------------------
# gates on the real tree
# --------------------------------------------------------------------------


def test_no_stale_claim_survives() -> None:
    rows = ce.audit()
    failing = [row for row in rows if row.level in ce.FAILING_LEVELS]
    assert failing == [], [(row.capability.key, row.level, row.reasons) for row in failing]
    stale = [row for row in rows if row.level == ce.STALE_CLAIM]
    assert stale == []


def test_uncapability_baseline_has_no_untested_module_without_a_reason() -> None:
    inventory = ce.untested_modules()
    assert inventory["unexplained"] == [], inventory["unexplained"]
    assert inventory["stale_reasons"] == [], inventory["stale_reasons"]


def test_published_block_matches_the_tree() -> None:
    drift = ce.document_drift()
    assert drift is None, (
        "docs/audit_baseline.md section 4 no longer matches the tree; "
        "regenerate it with `python -m tensorbem.capability_evidence --write`\n" + (drift or "")
    )


def test_document_records_the_capability_corrections() -> None:
    text = (ce.repo_root() / ce.BASELINE_DOC).read_text(encoding="utf-8")
    block = ce.extract_document_block(text)
    for title in (
        "Finite-depth kernels + solver (2-D)",
        "Finite-depth kernels + solver (3-D)",
        "Forward-speed wave resistance (Wigley, Michell/Kochin)",
        "Kutta condition / lifting foil",
    ):
        assert f"**{title}** — was `NOT_SUPPORTED`" in block, title


def test_previously_unsupported_capabilities_are_now_real() -> None:
    rows = {row.capability.key: row for row in ce.audit()}
    for key in ("finite-depth-2d", "finite-depth-3d", "forward-speed-wave-resistance", "kutta-lifting"):
        row = rows[key]
        assert row.capability.published_2026_10_29 == ce.NOT_SUPPORTED
        assert row.level == ce.REAL, (key, row.level, row.reasons)
        assert row.test_nodes >= 2


def test_summary_counts_are_consistent() -> None:
    rows = ce.audit()
    counts = ce.summary(rows)
    assert counts["total"] == len(rows)
    assert counts[ce.REAL] + counts[ce.WITHHELD] + counts[ce.STALE_CLAIM] + counts[ce.MISSING] == len(rows)


def test_registry_keys_are_unique() -> None:
    keys = [capability.key for capability in ce.CAPABILITIES]
    assert len(keys) == len(set(keys))


# --------------------------------------------------------------------------
# negative controls: the grader must be able to fail
# --------------------------------------------------------------------------

_GOOD = '''"""A well-behaved module."""


def compute(x):
    return 2.0 * x
'''

_STUBBY = '''def compute(x):
    raise NotImplementedError("not yet")
'''

_FAKE = '''import numpy as np


def estimate():
    return float(np.random.rand())
'''

_CONSTANT = '''def value():
    return 42
'''


@pytest.fixture()
def synthetic_repo(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """A miniature repository whose modules are graded by the real grader."""

    src = tmp_path / "src" / "toypkg"
    src.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'toy'\n", encoding="utf-8")
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "good.py").write_text(_GOOD, encoding="utf-8")
    (src / "stubby.py").write_text(_STUBBY, encoding="utf-8")
    (src / "fake.py").write_text(_FAKE, encoding="utf-8")
    (src / "constant.py").write_text(_CONSTANT, encoding="utf-8")

    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_good.py").write_text(
        "from toypkg.good import compute\n\n\ndef test_compute_doubles():\n    assert compute(2.0) == 4.0\n",
        encoding="utf-8",
    )
    (tests / "test_stubby.py").write_text(
        "import toypkg.stubby\n\n\ndef test_stub_raises():\n    pass\n", encoding="utf-8"
    )
    (tests / "test_fake.py").write_text(
        "import toypkg.fake\n\n\ndef test_estimate_is_finite():\n    pass\n", encoding="utf-8"
    )
    (tests / "test_constant.py").write_text(
        "import toypkg.constant\n\n\ndef test_value_is_constant():\n    pass\n", encoding="utf-8"
    )
    (tests / "test_unrelated.py").write_text(
        "def test_nothing_to_do_with_the_modules():\n    pass\n", encoding="utf-8"
    )

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "good.md").write_text("# good\n\ncompute is documented here.\n", encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path / "src"))
    return tmp_path


def _row(key: str, modules, symbols, tests, docs, level=ce.REAL) -> ce.Capability:
    return ce.Capability(
        key=key, title=key, claim="synthetic", level=level,
        modules=tuple(modules), symbols=tuple(symbols), tests=tuple(tests), docs=tuple(docs),
    )


def test_negative_control_real_then_stale_claim(synthetic_repo: pathlib.Path) -> None:
    root = synthetic_repo
    nodes = ce._test_nodes(root)
    real = ce.grade(
        _row("good", ["toypkg.good"], ["compute"], ["tests/test_good.py::test_compute_doubles"],
             [("docs/good.md", "documented")]),
        root, nodes,
    )
    assert real.level == ce.REAL, real.reasons

    stale = ce.grade(
        _row("good", ["toypkg.good"], ["compute"], ["tests/test_good.py::test_compute_doubles"],
             [("docs/good.md", "documented")], level=ce.NOT_SUPPORTED),
        root, nodes,
    )
    assert stale.level == ce.STALE_CLAIM


def test_negative_control_stub_mock_constant_and_missing(synthetic_repo: pathlib.Path) -> None:
    root = synthetic_repo
    nodes = ce._test_nodes(root)

    stub = ce.grade(
        _row("stubby", ["toypkg.stubby"], ["compute"], ["tests/test_stubby.py::test_stub_raises"], []),
        root, nodes,
    )
    assert stub.level == ce.STUB, stub.reasons
    assert stub.stub_symbols

    random_mock = ce.grade(
        _row("fake", ["toypkg.fake"], ["estimate"], ["tests/test_fake.py::test_estimate_is_finite"], []),
        root, nodes,
    )
    assert random_mock.level == ce.MOCK, random_mock.reasons

    constant_mock = ce.grade(
        _row("constant", ["toypkg.constant"], ["value"], ["tests/test_constant.py::test_value_is_constant"], []),
        root, nodes,
    )
    assert constant_mock.level == ce.MOCK, constant_mock.reasons

    missing = ce.grade(_row("ghost", ["toypkg.ghost"], ["compute"], [], []), root, nodes)
    assert missing.level == ce.MISSING


def test_negative_control_withheld_when_test_is_absent_or_detached(synthetic_repo: pathlib.Path) -> None:
    root = synthetic_repo
    nodes = ce._test_nodes(root)

    no_test = ce.grade(
        _row("good", ["toypkg.good"], ["compute"], [], [("docs/good.md", "documented")]), root, nodes
    )
    assert no_test.level == ce.WITHHELD

    detached = ce.grade(
        _row("good", ["toypkg.good"], ["compute"],
             ["tests/test_unrelated.py::test_nothing_to_do_with_the_modules"], []),
        root, nodes,
    )
    assert detached.level == ce.WITHHELD
    assert detached.detached_tests

    no_doc = ce.grade(
        _row("good", ["toypkg.good"], ["compute"], ["tests/test_good.py::test_compute_doubles"], []),
        root, nodes,
    )
    assert no_doc.level == ce.WITHHELD
    assert "no documentation anchor for the claim" in no_doc.reasons


def test_seeded_randomness_is_not_treated_as_a_mock(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "src" / "toyseededpkg"
    src.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'toy'\n", encoding="utf-8")
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "seeded.py").write_text(
        "import numpy as np\n\n\ndef estimate():\n    rng = np.random.default_rng(seed=7)\n    return float(rng.random())\n",
        encoding="utf-8",
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_seeded.py").write_text(
        "import toyseededpkg.seeded\n\n\ndef test_estimate():\n    pass\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path / "src"))

    row = ce.grade(_row("seeded", ["toyseededpkg.seeded"], ["estimate"], ["tests/test_seeded.py::test_estimate"], []),
                   tmp_path, ce._test_nodes(tmp_path))
    assert row.level == ce.WITHHELD  # only the doc anchor is missing, not MOCK
    assert row.mock_markers == []