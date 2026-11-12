"""Guards for the mutation harness's shadow-tree isolation (mission m16514).

Background.  ``tensorbem.benchmarks.mutation_audit`` scores a suite by running it
against a mutated copy of the package.  Until this mission the shadow tree was
passed only through ``PYTHONPATH``, while both ``tests/conftest.py`` and the
``-m pytest`` entry point push the *repository* ``src`` in front of it.  The
session therefore imported the unmutated package and reported the mutant as
"survived": re-running the wave-load sweep on the current tree produced 1 kill
out of 30 mutants, including mutants that delete the ``m + added_mass`` term of
the RAO assembly.  Those verdicts were artifacts of the leak, not blind spots of
the suite.

The fix is ``tensorbem.benchmarks.mutation_isolation``: a pytest plugin that puts
a ``sys.meta_path`` finder in front of every other import mechanism, so that a
later ``sys.path.insert(0, <repo>/src)`` (exactly what ``conftest.py`` does)
cannot win.  These tests pin that property without running a real sweep.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

from tensorbem.benchmarks import mutation_isolation


@pytest.fixture()
def shadow_tree(tmp_path: pathlib.Path):
    """A stand-in shadow ``src`` tree plus the repository ``src``."""
    shadow = tmp_path / "shadow" / "src"
    package = shadow / "tensorbem"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("ORIGIN = 'shadow'\n", encoding="utf-8")
    (package / "core.py").write_text("VALUE = 2.0\n", encoding="utf-8")
    repo_src = tmp_path / "repo" / "src"
    (repo_src / "tensorbem").mkdir(parents=True)
    (repo_src / "tensorbem" / "__init__.py").write_text(
        "ORIGIN = 'repo'\n", encoding="utf-8"
    )
    (repo_src / "tensorbem" / "core.py").write_text("VALUE = 1.0\n", encoding="utf-8")
    return shadow, repo_src


@pytest.fixture()
def restore_import_state():
    saved_path = list(sys.path)
    saved_meta = list(sys.meta_path)
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "tensorbem" or name.startswith("tensorbem.")
    }
    for name in saved_modules:
        del sys.modules[name]
    try:
        yield
    finally:
        for name in [n for n in sys.modules if n == "tensorbem" or n.startswith("tensorbem.")]:
            del sys.modules[name]
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path
        sys.meta_path[:] = saved_meta


def test_finder_wins_over_a_later_conftest_style_path_insertion(
    shadow_tree, restore_import_state
) -> None:
    """``conftest.py`` prepending the repository ``src`` must not win.

    This is the exact failure mode: the plugin installs, then a ``conftest``
    inserts the repository ``src`` at ``sys.path[0]``, and the session must
    still see the shadow module.
    """
    shadow, repo_src = shadow_tree
    resolved = mutation_isolation.install(shadow, repo_src)

    assert str(shadow.resolve()) in str(resolved)

    # Simulate tests/conftest.py, which runs *after* plugin installation.
    sys.path.insert(0, str(repo_src))
    for name in [n for n in sys.modules if n == "tensorbem" or n.startswith("tensorbem.")]:
        # keep the package object itself (conftest does not purge it), drop the
        # submodule so the import machinery has to resolve it again
        if name != "tensorbem":
            del sys.modules[name]

    import tensorbem

    assert tensorbem.ORIGIN == "shadow"
    assert str(shadow.resolve()) in str(pathlib.Path(tensorbem.__file__).resolve())

    from tensorbem import core

    assert core.VALUE == 2.0  # the mutated copy, not the repository one


def test_install_reports_the_resolved_package_path(
    shadow_tree, restore_import_state, tmp_path
) -> None:
    """The plugin writes what it imported, so a caller can audit the claim."""
    shadow, repo_src = shadow_tree
    report = tmp_path / "imported.txt"

    resolved = mutation_isolation.install(shadow, repo_src, report)

    written = pathlib.Path(report.read_text(encoding="utf-8").strip())
    assert written == resolved
    assert str(shadow.resolve()) in str(written)


def test_install_rejects_a_shadow_tree_that_does_not_contain_the_package(
    shadow_tree, restore_import_state, tmp_path
) -> None:
    """Fail closed: an empty shadow tree must raise, never fall back silently."""
    _, repo_src = shadow_tree
    empty = tmp_path / "empty" / "src"
    empty.mkdir(parents=True)

    with pytest.raises(RuntimeError):
        mutation_isolation.install(empty, repo_src)


def test_harness_runs_pytest_with_the_isolation_plugin() -> None:
    """The sweep must pass ``-p`` for the plugin, not just a ``PYTHONPATH``."""
    import inspect

    from tensorbem.benchmarks import mutation_audit

    source = inspect.getsource(mutation_audit.run_tests)
    assert "ISOLATION.PLUGIN_NAME" in source
    assert "ISOLATION.SHADOW_ENV" in source
    assert "ISOLATION.REPORT_ENV" in source
    assert "ISOLATION-FAILURE" in inspect.getsource(mutation_audit._classify)