"""Semantic resolution of the agent/m3181 -> main merge (PR #128, mission 3751).

Conflict resolution is done on *content*, not by overlay: every hunk keeps the
meaning of both parents.  Besides the three textual conflict regions, the merge
produced a silent semantic collision -- both parents added a module-level helper
named ``_chief_rows`` (side A: a registry-row factory taking no argument, side
B: a study flattener taking the study rows).  Git merged them without markers,
so the last definition would have shadowed the first and broken side A's call.
The flattener is therefore renamed to ``_chief_study_rows``.

Run from the repository root *inside an unresolved merge*::

    python docs/m3751/resolve_conflicts.py

The script is idempotent (it errors out if the expected conflict markers are
missing rather than silently doing nothing).
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
DECL = ROOT / "src/tensorbem/benchmarks/declared_numbers.py"
TEST = ROOT / "tests/test_declared_numbers.py"


def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: pathlib.Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def resolve_declared_numbers() -> None:
    text = _read(DECL)

    # ------------------------------------------------------------------
    # Conflict 1: import aliases.  Both parents renamed the same modules;
    # the merged body references *both* spellings (side A's tables live in
    # the ``hulme3d*``/``wang_linton3d`` blocks, side B's in the
    # ``hulme_*``/``submerged_sphere`` blocks), so a single canonical alias
    # set is used and side A's call sites are renamed to match below.
    # ------------------------------------------------------------------
    old = """<<<<<<< HEAD
    floating_hemisphere as hulme_benchmark,
    interior_harmonics as interior_harmonics_benchmark,
    submerged_sphere_wave as wang_linton_benchmark,
=======
    floating_hemisphere as hemisphere_benchmark,
    interior_harmonics as interior_harmonics_benchmark,
    submerged_sphere_wave as sphere_benchmark,
>>>>>>> origin/main
"""
    new = """    floating_hemisphere as hemisphere_benchmark,
    interior_harmonics as interior_harmonics_benchmark,
    submerged_sphere_wave as sphere_benchmark,
"""
    assert text.count(old) == 1, "import conflict not found"
    text = text.replace(old, new)

    # ------------------------------------------------------------------
    # Conflict 2: the registry tuple.  Side B appended its table-10/11/12
    # rows inline and closes the tuple with ``)``; side A keeps the same
    # tables in a generated block (``_free_surface_3d_rows()``).  Union:
    # B's inline rows stay (they are cited to the same document sections
    # and carry their own keys) and A's generator is concatenated.
    # ------------------------------------------------------------------
    old = """<<<<<<< HEAD
) + _free_surface_3d_rows()
=======
"""
    assert text.count(old) == 1, "registry-tuple conflict head not found"
    text = text.replace(old, "")
    old = """<<<<<<< HEAD
    if group == "hulme3d":
"""
    assert text.count(old) == 1, "dispatch conflict head not found"
    text = text.replace(old, """    if group == "hulme3d":
""")

    # Close B's inline block with A's generated block appended.
    marker = """)
>>>>>>> origin/main

_URSELL_RADIUS = 1.0"""
    assert text.count(marker) == 1, "registry-tuple tail marker not found"
    text = text.replace(
        marker,
        """) + _free_surface_3d_rows()

_URSELL_RADIUS = 1.0""",
    )

    # ------------------------------------------------------------------
    # Conflict 3: the benchmark dispatch.  Union keeps A's three groups and
    # B's three groups; they take disjoint ``group`` values.
    # ------------------------------------------------------------------
    marker = """        return values
=======
    if group == "hulme_hemisphere":"""
    assert text.count(marker) == 1, "dispatch split marker not found"
    text = text.replace(marker, """        return values
    if group == "hulme_hemisphere":""")
    marker = """        return _chief_rows(rows)
>>>>>>> origin/main
    raise ValueError"""
    assert text.count(marker) == 1, "dispatch tail marker not found"
    text = text.replace(
        marker,
        """        return _chief_study_rows(rows)
    raise ValueError""",
    )

    # ------------------------------------------------------------------
    # Silent collision: rename B's flattener (A's zero-argument factory of
    # the same name is defined earlier in the file and must keep its name,
    # because ``_free_surface_3d_rows`` calls it).
    # ------------------------------------------------------------------
    old = '''def _chief_rows(rows) -> dict[str, float]:
    """Flatten the irregular-frequency study'''
    assert text.count(old) == 1, "flattener definition not found"
    text = text.replace(
        old,
        '''def _chief_study_rows(rows) -> dict[str, float]:
    """Flatten the irregular-frequency study''',
    )

    # Side A's call sites use the aliases that no longer exist.
    text = text.replace("hulme_benchmark.run_refinement_study(", "hemisphere_benchmark.run_refinement_study(")
    text = text.replace("hulme_benchmark.run_irregular_frequency_study(", "hemisphere_benchmark.run_irregular_frequency_study(")
    text = text.replace("wang_linton_benchmark.run_refinement_study(", "sphere_benchmark.run_refinement_study(")

    assert "<<<<<<<" not in text and ">>>>>>>" not in text
    assert "\n=======\n" not in text
    _write(DECL, text)


def resolve_test() -> None:
    text = _read(TEST)

    # Conflict 1: two phrasings of the same invariant.  Keep both reasons --
    # they document different tables (A: table 10's negative heave damping,
    # B: table 11's CHIEF defect).
    old = """<<<<<<< HEAD
        # A quote may legitimately be non-positive: the unregularized
        # 3-D BIE produces an unphysical negative heave damping, and that
        # sign is exactly the claim the table makes.
=======
        # sign-bearing quantities exist (e.g. the unphysical negative B33' that
        # table 11 quotes as the defect CHIEF removes), so only zero is illegal
>>>>>>> origin/main
"""
    new = """        # A quote may legitimately be non-positive: the unregularized
        # 3-D BIE produces an unphysical negative heave damping (table 10) and
        # an unphysical B33' that table 11 quotes as the defect CHIEF removes;
        # that sign is exactly the claim the tables make, so only zero is
        # illegal.
"""
    assert text.count(old) == 1, "test comment conflict not found"
    text = text.replace(old, new)

    # Conflict 2: the group inventory.  Union of both parents' families.
    old = """<<<<<<< HEAD
        "hulme3d",
        "hulme3d_chief",
=======
        "hulme_chief",
        "hulme_hemisphere",
>>>>>>> origin/main
"""
    new = """        "hulme3d",
        "hulme3d_chief",
        "hulme_chief",
        "hulme_hemisphere",
"""
    assert text.count(old) == 1, "test group-set conflict not found"
    text = text.replace(old, new)

    assert "<<<<<<<" not in text and ">>>>>>>" not in text
    _write(TEST, text)


def main() -> int:
    resolve_declared_numbers()
    resolve_test()
    print("resolved:", DECL.relative_to(ROOT), TEST.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())