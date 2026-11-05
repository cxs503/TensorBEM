"""Regression contract for the PR #128 3-way merge (agent/m3181 <- main).

Both parents independently registered the 3-D free-surface tables of
``docs/marine_bem_benchmarks.md`` as executable numbers, so the merged branch
has to carry *both* registration families at once. Everything here is cheap
(no benchmark is solved) and pins the four ways a later re-merge can lose
content silently:

1. a whole registration family disappears from the registry,
2. the union of two key schemes collides on a key,
3. the same top-level name is defined twice (the original conflict was one:
   two different ``_chief_rows`` implementations, of which Python keeps only
   the later one -- a silent loss that breaks the registry at import time),
4. a registered group has no dispatcher branch in ``_run_group``.

The docs pin is textual on purpose: the healed merge must keep main's
measurement-metric correction note *and* this branch's §4.12 registry section.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from tensorbem.benchmarks import declared_numbers
from tensorbem.benchmarks.declared_numbers import DECLARED_NUMBERS

MODULE = Path(declared_numbers.__file__)
DOCS = MODULE.parents[3] / "docs" / "marine_bem_benchmarks.md"

#: Registration families contributed by agent/m3181 (programmatic 3-D Kelvin
#: tables) and by main (inline tables 10-12 rows) respectively.
BRANCH_GROUPS = {"hulme3d", "hulme3d_chief", "wang_linton3d"}
MAIN_GROUPS = {"hulme_hemisphere", "submerged_sphere", "hulme_chief"}

def _top_level_names() -> list[str]:
    tree = ast.parse(MODULE.read_text(encoding="utf-8"), str(MODULE))
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]


def test_both_registration_families_survive_the_union():
    groups = {entry.group for entry in DECLARED_NUMBERS}
    assert BRANCH_GROUPS <= groups, sorted(BRANCH_GROUPS - groups)
    assert MAIN_GROUPS <= groups, sorted(MAIN_GROUPS - groups)


def test_union_keys_stay_unique():
    keys = [entry.key for entry in DECLARED_NUMBERS]
    assert len(keys) == len(set(keys))


def test_no_top_level_symbol_is_defined_twice():
    """A duplicated name means one definition shadows the other (silent loss)."""
    names = _top_level_names()
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert duplicates == [], f"shadowed definitions: {duplicates}"


@pytest.mark.parametrize("group", sorted(BRANCH_GROUPS | MAIN_GROUPS))
def test_every_registered_group_has_a_dispatcher(group: str):
    source = inspect.getsource(declared_numbers._run_group)
    assert f'group == "{group}"' in source, f"no _run_group branch for {group!r}"


def test_both_documented_sections_are_present():
    text = DOCS.read_text(encoding="utf-8")
    # main: the B11' observed-order / rotational-coupling metric correction.
    assert "m3150 复算订正" in text
    # this branch: the §4.12 registry section for tables 10-12.
    assert "4.12 三维自由液面声明数字的注册表复算" in text


def test_main_half_dispatcher_still_flattens_its_study():
    """The renamed helper must still be the one main's branch calls."""
    source = inspect.getsource(declared_numbers._run_group)
    assert "_chief_study_rows(rows)" in source
    # this branch's zero-argument registry builder is untouched.
    assert callable(declared_numbers._chief_rows)
    assert len(inspect.signature(declared_numbers._chief_rows).parameters) == 0


def test_both_parents_module_aliases_stay_importable():
    """Each parent picked its own alias for the same two benchmark modules."""
    for alias in ("hulme_benchmark", "wang_linton_benchmark"):
        assert hasattr(declared_numbers, alias), alias
    for alias in ("hemisphere_benchmark", "sphere_benchmark"):
        assert hasattr(declared_numbers, alias), alias