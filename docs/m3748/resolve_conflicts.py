#!/usr/bin/env python3
"""Resolve the ``git merge origin/agent/m3181`` conflicts in the m3748 healing
mission (PR #128).

Policy: every hunk is resolved as a **semantic union** — both sides' content is
kept verbatim; nothing on either side is silently dropped (no mechanical
"take ours"/"take theirs" overlay).

Usage:  python3 docs/m3748/resolve_conflicts.py <repo-root>

Idempotent: a file with no conflict markers left is reported and skipped.
Exit code 0 only if every expected hunk was resolved by an explicit rule.
"""
from __future__ import annotations

import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Per-file, per-hunk resolvers.  Each resolver receives (ours, theirs) and
# returns the merged text; it asserts its own pre-conditions so a layout drift
# fails loudly instead of silently merging the wrong thing.
# --------------------------------------------------------------------------- #


def union_sorted_lines(ours: str, theirs: str) -> str:
    """Union two sets of full lines, de-duplicated and sorted."""
    lines = sorted(set(ours.splitlines()) | set(theirs.splitlines()))
    return "".join(line + "\n" for line in lines)


def import_union(ours: str, theirs: str) -> str:
    merged = union_sorted_lines(ours, theirs)
    for required in (
        "floating_hemisphere as hemisphere_benchmark,",   # main side
        "floating_hemisphere as hulme_benchmark,",        # m3181 side
        "interior_harmonics as interior_harmonics_benchmark,",
        "submerged_sphere_wave as sphere_benchmark,",     # main side
        "submerged_sphere_wave as wang_linton_benchmark,",  # m3181 side
    ):
        assert required in merged, required
    return merged


def registry_tuple_tail(ours: str, theirs: str) -> str:
    """main appended rows inside the tuple literal; m3181 closed the literal and
    appended ``+ _free_surface_3d_rows()``.  Keep both: inline rows *and* the
    m3181 generated-rows function."""
    assert ours.endswith("    ),\n)\n"), repr(ours[-40:])
    assert theirs == ") + _free_surface_3d_rows()\n", repr(theirs)
    return ours[: -len(")\n")] + theirs


def measure_groups(ours: str, theirs: str) -> str:
    """Disjoint benchmark-group dispatch arms: keep every arm from both sides."""
    merged = ours + theirs
    for group in ("hulme_hemisphere", "submerged_sphere", "hulme_chief",
                  "hulme3d", "hulme3d_chief", "wang_linton3d"):
        assert f'group == "{group}"' in merged, group
    return merged


def comment_union(ours: str, theirs: str) -> str:
    """Two wordings of the same invariant; keep both sentences."""
    assert "table 11 quotes as the defect CHIEF removes" in ours
    assert "unphysical negative heave damping" in theirs
    return (
        "        # sign-bearing quantities exist (e.g. the unphysical negative B33' that\n"
        "        # table 11 quotes as the defect CHIEF removes), so only zero is illegal.\n"
        "        # A quote may legitimately be non-positive: the unregularized 3-D BIE\n"
        "        # produces an unphysical negative heave damping, and that sign is\n"
        "        # exactly the claim the table makes.\n"
    )


def group_name_union(ours: str, theirs: str) -> str:
    merged = union_sorted_lines(ours, theirs)
    for group in ("hulme3d", "hulme3d_chief", "hulme_chief", "hulme_hemisphere"):
        assert f'"{group}",' in merged, group
    return merged


# file -> list of resolvers in hunk order
PLAN = {
    "src/tensorbem/benchmarks/declared_numbers.py": [
        import_union,
        registry_tuple_tail,
        measure_groups,
    ],
    "tests/test_declared_numbers.py": [
        comment_union,
        group_name_union,
    ],
}


def split_blocks(path: Path):
    lines = path.read_text().splitlines(keepends=True)
    blocks = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("<<<<<<<"):
            start = i
            ours, theirs = [], []
            i += 1
            while not lines[i].startswith("======="):
                ours.append(lines[i])
                i += 1
            i += 1
            while not lines[i].startswith(">>>>>>>"):
                theirs.append(lines[i])
                i += 1
            blocks.append((start, i + 1, "".join(ours), "".join(theirs)))
            i += 1
        else:
            i += 1
    return lines, blocks


def main(root: str) -> int:
    rootp = Path(root)
    rc = 0
    for rel, resolvers in PLAN.items():
        path = rootp / rel
        for _ in range(50):  # resolve one hunk per pass so line numbers stay valid
            lines, blocks = split_blocks(path)
            if not blocks:
                break
            start, end, ours, theirs = blocks[0]
            resolver = resolvers[len(resolvers) - len(blocks)]
            merged = resolver(ours, theirs)
            lines[start:end] = [merged]
            path.write_text("".join(lines))
            print(f"{rel}: hunk @ {start + 1} resolved by {resolver.__name__} "
                  f"(ours {len(ours.splitlines())}L + theirs {len(theirs.splitlines())}L kept)")
        lines, blocks = split_blocks(path)
        if blocks:
            print(f"{rel}: {len(blocks)} unresolved hunk(s) left")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "."))