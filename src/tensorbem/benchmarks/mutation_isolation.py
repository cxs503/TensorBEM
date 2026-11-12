"""Pytest plugin that pins the ``tensorbem`` package to a shadow source tree.

Why this exists
---------------
``tensorbem.benchmarks.mutation_audit`` scores a test suite by running it against
a *mutated copy* of the package.  That only means anything if the test session
imports the copy.  Passing ``PYTHONPATH=<shadow>/src`` is **not** sufficient in
this repository, for two independent reasons:

* ``tests/conftest.py`` prepends ``<repo>/src`` to ``sys.path`` so that the
  checked-out tree wins over any editable install -- which is exactly backwards
  for a mutation run, and
* the shadow directory is only consulted by the path-based finder, i.e. after
  ``sys.meta_path``, so any earlier insertion of ``<repo>/src`` wins.

Measured on this tree (see ``runs/m16514/probe_isolation.py``): a mutant that
replaces ``m + added_mass`` with ``m - added_mass`` in ``tensorbem.rao`` left
``tests/test_rao_verification.py`` fully green, because the session imported the
*unmutated* repository package.  The pre-existing ``_check_isolation`` helper
only checked a bare ``import tensorbem`` in a plain interpreter, where
``conftest.py`` never runs -- so it certified an isolation it did not have.

What this plugin does
---------------------
It is loaded with ``-p tensorbem.benchmarks.mutation_isolation`` and, at
``pytest_configure`` time (before any ``conftest.py`` is imported), it

1. purges every already-imported ``tensorbem`` module,
2. removes the repository ``src`` from ``sys.path``, and
3. puts a :class:`_ShadowFinder` at the front of ``sys.meta_path`` that resolves
   *every* ``tensorbem`` / ``tensorbem.*`` import from the shadow tree only.

Because ``sys.meta_path`` is consulted before ``sys.path``, a later
``sys.path.insert(0, <repo>/src)`` by ``conftest.py`` cannot undo this.

The resolved package path is written to ``$TENSORBEM_MUTATION_ISOLATION_FILE``
so the caller can *verify after the fact* which package the session used; a
session that still resolves elsewhere is reported as an isolation error rather
than as a surviving or killed mutant.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import os
import pathlib
import sys

__all__ = ["ShadowFinder", "install", "PLUGIN_NAME"]

PLUGIN_NAME = "tensorbem.benchmarks.mutation_isolation"

SHADOW_ENV = "TENSORBEM_MUTATION_SHADOW"
REPO_SRC_ENV = "TENSORBEM_MUTATION_REPO_SRC"
REPORT_ENV = "TENSORBEM_MUTATION_ISOLATION_FILE"


class _ShadowFinder(importlib.abc.MetaPathFinder):
    """Resolve ``tensorbem`` and ``tensorbem.*`` from one directory tree."""

    def __init__(self, shadow_src: pathlib.Path) -> None:
        self.shadow_src = pathlib.Path(shadow_src).resolve()
        self.package = self.shadow_src / "tensorbem"

    # -- finder protocol ---------------------------------------------------
    def find_spec(self, fullname, path=None, target=None):  # noqa: D401
        if fullname != "tensorbem" and not fullname.startswith("tensorbem."):
            return None
        parts = fullname.split(".")[1:]
        base = self.package.joinpath(*parts) if parts else self.package
        if base.is_dir() and (base / "__init__.py").is_file():
            return importlib.util.spec_from_file_location(
                fullname,
                str(base / "__init__.py"),
                submodule_search_locations=[str(base)],
            )
        module = base.with_suffix(".py")
        if module.is_file():
            return importlib.util.spec_from_file_location(fullname, str(module))
        return None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ShadowFinder({self.package})"


def _purge(prefix: str = "tensorbem") -> None:
    for name in [n for n in sys.modules if n == prefix or n.startswith(prefix + ".")]:
        del sys.modules[name]


def _drop_repo_src(repo_src: pathlib.Path | None) -> None:
    if repo_src is None:
        return
    resolved = repo_src.resolve()
    kept = []
    for entry in sys.path:
        try:
            candidate = pathlib.Path(entry or ".").resolve()
        except OSError:  # pragma: no cover - exotic path entries
            kept.append(entry)
            continue
        if candidate != resolved:
            kept.append(entry)
    sys.path[:] = kept


def install(
    shadow_src: pathlib.Path,
    repo_src: pathlib.Path | None = None,
    report_path: pathlib.Path | None = None,
) -> pathlib.Path:
    """Pin ``tensorbem`` to ``shadow_src``; return the resolved package path."""
    shadow_src = pathlib.Path(shadow_src).resolve()
    _purge()
    _drop_repo_src(repo_src)
    sys.meta_path.insert(0, _ShadowFinder(shadow_src))
    sys.path.insert(0, str(shadow_src))

    import tensorbem  # resolved by the finder, never by sys.path

    resolved = pathlib.Path(tensorbem.__file__).resolve()
    if shadow_src not in resolved.parents:
        raise RuntimeError(
            f"mutation isolation failed: tensorbem resolved to {resolved}, "
            f"not under the shadow tree {shadow_src}"
        )
    if report_path is not None:
        pathlib.Path(report_path).write_text(str(resolved), encoding="utf-8")
    return resolved


def pytest_configure(config):  # noqa: D103 - pytest hook
    shadow = os.environ.get(SHADOW_ENV)
    if not shadow:
        raise RuntimeError(f"{SHADOW_ENV} must be set for a mutation run")
    repo_src = os.environ.get(REPO_SRC_ENV)
    report = os.environ.get(REPORT_ENV)
    resolved = install(
        pathlib.Path(shadow),
        pathlib.Path(repo_src) if repo_src else None,
        pathlib.Path(report) if report else None,
    )
    config.stash  # touch: keeps the hook cheap  # noqa: B018
    print(f"mutation isolation pinned to {resolved}")