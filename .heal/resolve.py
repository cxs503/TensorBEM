"""Resolve the two conflicted files of the agent/m3181 -> main merge, PR #128.

Semantic union, no mechanical overlay: every block that exists on one side and
not the other is kept; the only text removed is a duplicate *name*, and the
colliding symbol is renamed on the side that owns fewer call sites so that both
implementations stay reachable (a silently shadowed def is a silent content
loss).
"""

from __future__ import annotations

import re
from pathlib import Path

WT = Path(__file__).resolve().parent / "wt"

REGISTRY = WT / "src/tensorbem/benchmarks/declared_numbers.py"
TESTS = WT / "tests/test_declared_numbers.py"


def split_conflicts(text: str) -> list[tuple[str, str, str]]:
    """Return [(ours, theirs, full_marker_block)] for every conflict."""
    blocks = []
    pattern = re.compile(
        r"<<<<<<< [^\n]*\n(.*?)\n=======\n(.*?)\n>>>>>>> [^\n]*\n", re.S
    )
    for match in pattern.finditer(text):
        blocks.append((match.group(1), match.group(2), match.group(0)))
    return blocks


def resolve_registry() -> None:
    text = REGISTRY.read_text()
    conflicts = split_conflicts(text)
    assert len(conflicts) == 3, f"expected 3 conflicts, got {len(conflicts)}"

    # ---- conflict 1: the benchmark module aliases of the two sides ----------
    ours, theirs, block = conflicts[0]
    union_imports = """    # Both sides of the divergence imported the same benchmark modules under
    # their own local names; the 3-D group dispatchers below still refer to
    # both spellings, so both aliases are kept (dropping either would silently
    # delete a working registry group).
    floating_hemisphere as hulme_benchmark,
    floating_hemisphere as hemisphere_benchmark,
    interior_harmonics as interior_harmonics_benchmark,
    submerged_sphere_wave as wang_linton_benchmark,
    submerged_sphere_wave as sphere_benchmark,
"""
    text = text.replace(block, union_imports, 1)

    # ---- conflict 2: the DECLARED_NUMBERS tuple -----------------------------
    conflicts = split_conflicts(text)
    ours, theirs, block = conflicts[0]
    assert ours.strip() == ") + _free_surface_3d_rows()", ours
    assert theirs.rstrip().endswith(")"), theirs[-80:]
    # main's rows are kept verbatim; the m3181 extension is re-attached to the
    # closing parenthesis instead of replacing main's rows.
    merged = theirs + " + _free_surface_3d_rows()\n"
    text = text.replace(block, merged, 1)

    # ---- conflict 3: the _run_group dispatch table --------------------------
    conflicts = split_conflicts(text)
    ours, theirs, block = conflicts[0]
    for needle in ("hulme3d", "hulme3d_chief", "wang_linton3d"):
        assert needle in ours, needle
    for needle in ("hulme_hemisphere", "submerged_sphere", "hulme_chief"):
        assert needle in theirs, needle
    text = text.replace(block, ours + "\n" + theirs + "\n", 1)
    assert "<<<<<<<" not in text

    # ---- the one genuine collision: _chief_rows is defined twice -----------
    # m3181: _chief_rows() -> tuple[DeclaredNumber, ...] (registry rows)
    # main : _chief_rows(rows) -> dict[str, float]      (study flattener)
    # Python would keep only the later def, so m3181's rows would vanish while
    # _free_surface_3d_rows() called the wrong signature at run time. Rename the
    # flattener (one definition, one call site) next to its sibling _study_rows.
    assert text.count("def _chief_rows(rows)") == 1
    text = text.replace("def _chief_rows(rows)", "def _chief_study_rows(rows)", 1)
    assert text.count("return _chief_rows(rows)") == 1
    text = text.replace("return _chief_rows(rows)", "return _chief_study_rows(rows)", 1)
    assert text.count("def _chief_rows(") == 1  # m3181's registry row builder
    assert text.count("def _chief_study_rows(") == 1

    REGISTRY.write_text(text)


def resolve_tests() -> None:
    text = TESTS.read_text()
    conflicts = split_conflicts(text)
    assert len(conflicts) == 2, f"expected 2 conflicts, got {len(conflicts)}"

    # ---- conflict 1: the rationale comment above a shared assert -----------
    ours, theirs, block = conflicts[0]
    union_comment = (
        "        # A quote may legitimately be non-positive: the unregularized\n"
        "        # 3-D BIE produces an unphysical negative heave damping, and that\n"
        "        # sign is exactly the claim the table makes. The other sign-bearing\n"
        "        # quantities (e.g. the unphysical negative B33' that table 11\n"
        "        # quotes as the defect CHIEF removes) follow the same rule, so only\n"
        "        # zero is illegal.\n"
    )
    text = text.replace(block, union_comment, 1)

    # ---- conflict 2: the set of groups the registry must cover -------------
    conflicts = split_conflicts(text)
    ours, theirs, block = conflicts[0]
    names = set()
    for chunk in (ours, theirs):
        for name in re.findall(r'"([a-z0-9_]+)"', chunk):
            names.add(name)
    assert names == {"hulme3d", "hulme3d_chief", "hulme_chief", "hulme_hemisphere"}, names
    union_names = "".join(f'        "{name}",\n' for name in sorted(names))
    text = text.replace(block, union_names, 1)
    assert "<<<<<<<" not in text
    TESTS.write_text(text)


if __name__ == "__main__":
    resolve_registry()
    resolve_tests()
    print("resolved")