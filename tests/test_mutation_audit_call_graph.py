"""AST call-graph family extension for the mutation audit (m3531).

m3414 recorded as a known limitation that the stage-2 family search is empty
for module-level private helpers: no test file names the helper, so a mutant
inside the helper was called a blind spot even when a test of its public
caller would kill it.  The fix (``caller_names`` + the ``extra_names``
argument of ``sibling_tests``) extends the re-check family with tests that
name any *transitive caller* of the mutated function.

These tests pin the fix without running any BEM solve: the call graph and the
family computation are pure text/AST operations.  The hermetic tests point
``REPO_ROOT`` at a temporary tree so the scan can never pick up this very
file (a name defined here would otherwise match this file's own source).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensorbem.benchmarks import mutation_audit as ma  # noqa: E402


SYNTHETIC = """
def _secret_helper(x):
    return x + 1


def public_api(x):
    return _secret_helper(x) * 2


def unrelated(x):
    return x
"""

CHAIN = """
def _leaf(x):
    return x


def mid(x):
    return _leaf(x) + 1


def top(x):
    return mid(x) + 2
"""


def make_tests_root(tmp_path, files):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    for name, text in files.items():
        (tests_dir / name).write_text(text, encoding="utf-8")
    return tmp_path


def test_call_graph_links_private_helper_to_public_caller():
    graph = ma._module_call_graph(SYNTHETIC)
    assert graph["public_api"] == frozenset({"_secret_helper"})
    assert graph["unrelated"] == frozenset()
    assert graph["_secret_helper"] == frozenset()


def test_caller_names_is_transitive():
    assert set(ma.caller_names(CHAIN, "_leaf")) == {"mid", "top"}


def test_caller_names_unknown_or_module_level_is_empty():
    assert ma.caller_names(SYNTHETIC, "no_such_function") == ()
    assert ma.caller_names(SYNTHETIC, "<module>") == ()


def test_sibling_tests_without_extra_names_unchanged(tmp_path, monkeypatch):
    # Backward compatibility: with no extra names the behaviour is exactly the
    # pre-m3531 textual search on the helper's own name.
    root = make_tests_root(tmp_path, {
        "test_public_api.py": "def test_it():\n    assert public_api(1) == 4\n",
    })
    monkeypatch.setattr(ma, "REPO_ROOT", root)
    ma._FAMILY_CACHE.clear()
    assert ma.sibling_tests("_secret_helper") == ()
    ma._FAMILY_CACHE.clear()


def test_sibling_tests_with_caller_names_finds_caller_tests(tmp_path, monkeypatch):
    # A test file that exercises the public caller -- without ever spelling
    # the helper's name -- must enter the helper's re-check family.
    root = make_tests_root(tmp_path, {
        "test_public_api.py": "def test_it():\n    assert public_api(1) == 4\n",
        "test_unrelated.py": "def test_it():\n    assert unrelated(1) == 1\n",
    })
    monkeypatch.setattr(ma, "REPO_ROOT", root)
    extra = ma.caller_names(SYNTHETIC, "_secret_helper")
    assert extra == ("public_api",)
    ma._FAMILY_CACHE.clear()
    old = ma.sibling_tests("_secret_helper")
    new = ma.sibling_tests("_secret_helper", extra_names=extra)
    ma._FAMILY_CACHE.clear()
    assert old == ()
    assert new == ("tests/test_public_api.py",)


def test_real_helper_family_widens_and_stays_bounded():
    src = (SRC / "tensorbem" / "analytic.py").read_text(encoding="utf-8")
    extra = ma.caller_names(src, "_ursell_multipole_system")
    assert "ursell_cylinder_added_mass" in extra
    assert "ursell_cylinder_coefficients" in extra
    # every caller must be a module-level def of analytic.py
    tree = ast.parse(src)
    top = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert set(extra) <= top
    ma._FAMILY_CACHE.clear()
    old = ma.sibling_tests("_ursell_multipole_system")
    new = ma.sibling_tests("_ursell_multipole_system", extra_names=extra)
    ma._FAMILY_CACHE.clear()
    assert set(old) <= set(new), "the extension must only add tests, never remove"
    assert len(new) > len(old), "the extension must actually widen the family"
    assert len(new) <= 8, "the limit must still apply"
    # at least one newly admitted file never names the helper itself: that is
    # precisely the coverage the old search could not see
    widened = [f for f in new if f not in old]
    assert any(
        "_ursell_multipole_system"
        not in (ma.REPO_ROOT / f).read_text(encoding="utf-8", errors="ignore")
        for f in widened
    )
