"""Map the two table 10-12 registries onto each other (merge evidence, #135).

The PR ``agent/m3264`` and ``main`` independently registered the 3-D
free-surface numbers of ``docs/marine_bem_benchmarks.md`` (tables 10-12):
the branch generated the rows from the refinement studies, main listed them as
literals. The conflict resolution keeps main's literal block and re-declares the
cells only the branch had. This script produces the evidence that nothing was
dropped: every row of the branch's generator is matched to a merged row, by
group and quoted value.

Run from the repository root::

    PYTHONPATH=src python3 docs/m3264_merge/coverage_map.py

It writes ``docs/m3264_merge/registry_coverage.json`` and prints a summary; the
tests that read the registry are not touched by it.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BRANCH_REF = "origin/agent/m3264"
BRANCH_PATH = "src/tensorbem/benchmarks/declared_numbers.py"
GROUPS = ("hulme_chief", "hulme_hemisphere", "submerged_sphere")


def _branch_rows() -> list[dict]:
    """Import the branch's registry without disturbing the worktree."""
    blob = subprocess.run(
        ["git", "show", f"{BRANCH_REF}:{BRANCH_PATH}"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(blob)
        path = handle.name
    spec = importlib.util.spec_from_file_location("_branch_registry", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_branch_registry"] = module
    spec.loader.exec_module(module)
    return [
        {
            "key": entry.key,
            "group": entry.group,
            "quantity": entry.quantity,
            "declared": float(entry.declared),
            "kind": entry.kind,
            "tier": entry.tier,
        }
        for entry in module.DECLARED_NUMBERS
        if entry.group in GROUPS
    ]


def _merged_rows() -> list[dict]:
    from tensorbem.benchmarks.declared_numbers import DECLARED_NUMBERS

    return [
        {
            "key": entry.key,
            "group": entry.group,
            "quantity": entry.quantity,
            "declared": float(entry.declared),
            "kind": entry.kind,
            "tier": entry.tier,
        }
        for entry in DECLARED_NUMBERS
        if entry.group in GROUPS
    ]


def _matches(branch_row: dict, merged_row: dict) -> bool:
    if branch_row["group"] != merged_row["group"]:
        return False
    mine, theirs = abs(branch_row["declared"]), abs(merged_row["declared"])
    if mine == theirs:
        return True
    scale = max(mine, theirs, 1.0e-300)
    # Table 12 quotes the rotational coupling to two significant figures, so a
    # match there is only meaningful at the 5 % level; everything else in the
    # tables carries four figures.
    tol = 5.0e-2 if "rotational_added_mass" in branch_row["quantity"] else 5.0e-3
    return abs(mine - theirs) / scale <= tol


def main() -> int:
    branch, merged = _branch_rows(), _merged_rows()
    mapping, unmatched = [], []
    for row in branch:
        hit = next((m for m in merged if _matches(row, m)), None)
        (mapping if hit else unmatched).append(
            {"branch": row, "merged": hit} if hit else row
        )
    merged_unmatched = [
        m
        for m in merged
        if not any(_matches(b, m) for b in branch)
    ]
    report = {
        "branch_rows": len(branch),
        "merged_rows": len(merged),
        "branch_rows_mapped": len(mapping),
        "branch_rows_unmapped": unmatched,
        "merged_rows_unique": merged_unmatched,
        "mapping": mapping,
    }
    out = REPO / "docs" / "m3264_merge" / "registry_coverage.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"branch rows {len(branch)} -> mapped {len(mapping)}, unmapped {len(unmatched)}")
    print(f"merged rows {len(merged)}, unique to the merge {len(merged_unmatched)}")
    for row in merged_unmatched:
        print(f"  merge-only {row['key']} = {row['declared']:.6g} ({row['kind']})")
    for row in unmatched:
        print(f"  UNMAPPED branch {row['key']} = {row['declared']:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())