#!/usr/bin/env python3
"""Resolve the two conflicted files of merge agent/m3181 -> main (mission m3737).

Both sides registered the *same* three-dimensional free-surface tables
(docs/marine_bem_benchmarks.md tables 10-12), but with two different
implementations:

* main (via PR #126 / m3150): aliases ``hemisphere_benchmark`` /
  ``sphere_benchmark``, groups ``hulme_hemisphere`` / ``submerged_sphere`` /
  ``hulme_chief``, measurement helpers ``_study_rows`` + ``_chief_rows(rows)``,
  and one hand-written ``DeclaredNumber`` per quoted table cell inside the
  ``DECLARED_NUMBERS`` tuple literal.
* agent/m3181: aliases ``hulme_benchmark`` / ``wang_linton_benchmark``, groups
  ``hulme3d`` / ``hulme3d_chief`` / ``wang_linton3d``, table-driven row builders
  (``_refinement_rows`` / ``_chief_rows()`` / ``_free_surface_3d_rows``) and
  group runners that flatten the studies inline.

Neither side is a subset of the other (different keys, different quantities,
different failure modes), so the merge is a semantic union: every row, alias,
group branch and test from both sides is kept.

Two hazards this script fixes explicitly:

1. ``_chief_rows`` exists on both sides with *incompatible signatures* -- in a
   plain merge the ``main`` version (``(rows)``) is defined later and therefore
   silently shadows the ``agent/m3181`` version (``()``), which
   ``_free_surface_3d_rows()`` calls at import time.  The m3181 builder is
   renamed ``_free_surface_3d_chief_rows`` so both survive.
2. ``DECLARED_NUMBERS`` must be the concatenation of both sides' rows, i.e.
   m3181's ``) + _free_surface_3d_rows()`` tail is applied to main's literal.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "src/tensorbem/benchmarks/declared_numbers.py"
TESTS = ROOT / "tests/test_declared_numbers.py"

MARK = re.compile(r"^(<<<<<<< |=======$|>>>>>>> )", re.MULTILINE)


def hunks(text: str):
    """Yield (start, end, ours, theirs) for every conflict block."""
    out = []
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        if lines[i].startswith("<<<<<<< "):
            j = i + 1
            ours = []
            while not lines[j].startswith("======="):
                ours.append(lines[j])
                j += 1
            j += 1
            theirs = []
            while not lines[j].startswith(">>>>>>> "):
                theirs.append(lines[j])
                j += 1
            out.append((i, j, ours, theirs))
            i = j + 1
        else:
            i += 1
    return out


def resolve_registry(text: str) -> str:
    block = hunks(text)
    assert len(block) == 3, f"expected 3 conflicts in the registry, got {len(block)}"
    lines = text.splitlines(keepends=True)

    # ---- hunk 1: imports. Keep both alias sets (union, one per side). ------
    start, end, ours, theirs = block[0]
    aliases = ours + [line for line in theirs if line not in ours]
    imports = [
        "    # Both alias sets are kept by the m3181<->main semantic union: the\n",
        "    # m3150 rows call the first pair, the m3181 rows the second.\n",
    ] + aliases
    lines[start : end + 1] = imports

    # ---- hunk 3 (resolve last, offsets): group runners. Keep both. ---------
    text = "".join(lines)
    block = hunks(text)
    assert len(block) == 2, "registry hunks 1+3 should merge to 2 blocks"
    lines = text.splitlines(keepends=True)
    start, end, ours, theirs = block[1]
    lines[start : end + 1] = ours + theirs
    text = "".join(lines)

    # ---- hunk 2: DECLARED_NUMBERS tail. main's literal + m3181's rows. ----
    block = hunks(text)
    assert len(block) == 1, "registry should have one conflict left"
    lines = text.splitlines(keepends=True)
    start, end, ours, theirs = block[0]
    assert ours[-1].strip() == ")", ours[-1]
    assert theirs[0].strip() == ") + _free_surface_3d_rows()", theirs[0]
    # main's literal already closes the tuple with ``)``; m3181 turns that same
    # closing parenthesis into ``) + _free_surface_3d_rows()``.  Union = main's
    # hand-written rows, then the m3181 table-driven rows appended to them.
    lines[start : end + 1] = ours[:-1] + [") + _free_surface_3d_rows()\n"]
    text = "".join(lines)

    # ---- de-collide the two _chief_rows definitions -----------------------
    m3181_def = (
        "def _chief_rows():\n"
        '    """One :class:`DeclaredNumber` per cell of the irregular-frequency table."""\n'
    )
    assert text.count(m3181_def) == 1, "m3181 _chief_rows() block not found verbatim"
    text = text.replace(
        m3181_def,
        "def _free_surface_3d_chief_rows():\n"
        '    """One :class:`DeclaredNumber` per cell of the irregular-frequency\n'
        '    table (m3181 table 11 registry rows; renamed on merge so it does not\n'
        '    collide with ``_chief_rows(rows)``, the measurement flattener)."""\n',
    )
    call = "        + _chief_rows()\n"
    assert text.count(call) == 1
    text = text.replace(call, "        + _free_surface_3d_chief_rows()\n")
    return text


def resolve_tests(text: str) -> str:
    block = hunks(text)
    assert len(block) == 2, f"expected 2 conflicts in the tests, got {len(block)}"
    lines = text.splitlines(keepends=True)

    # ---- hunk 2 first (later in the file): the group-coverage set. --------
    start, end, ours, theirs = block[1]
    lines[start : end + 1] = sorted(ours + theirs)

    text = "".join(lines)
    block = hunks(text)
    assert len(block) == 1
    lines = text.splitlines(keepends=True)

    # ---- hunk 1: both comments say the same thing; keep one merged note. --
    start, end, ours, theirs = block[0]
    lines[start : end + 1] = [
        "        # Sign-bearing quantities are legal (e.g. the unphysical negative\n",
        "        # B33' that table 11 quotes as the defect CHIEF removes): the plain\n",
        "        # 3-D BIE produces that negative heave damping and the sign is the\n",
        "        # claim the table makes, so only an exact zero is illegal.\n",
    ]
    return "".join(lines)


def main() -> int:
    for path, fn in ((REGISTRY, resolve_registry), (TESTS, resolve_tests)):
        text = path.read_text()
        assert MARK.search(text), f"no conflict markers in {path}"
        path.write_text(fn(text))
        assert not MARK.search(path.read_text()), f"markers left in {path}"
        print(f"resolved {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())