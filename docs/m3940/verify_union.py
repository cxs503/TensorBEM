"""Prove that the merge kept every registered cell of both parents.

Loads the three versions of ``declared_numbers`` straight out of git (base,
``agent/m3181``, ``main``) plus the merged working tree, maps the branch's cell
labels onto the canonical group/key names now in the registry, and fails loudly
if any cell of either parent has no counterpart in the merge.

Run:  PYTHONPATH=src python3 docs/m3940/verify_union.py
"""

from __future__ import annotations

import collections
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODULE = "src/tensorbem/benchmarks/declared_numbers.py"


def load(origin: str | None, name: str):
    """Import the registry from a git rev (or the working tree when None)."""
    if origin is None:
        path = REPO / MODULE
        source = path.read_text()
    else:
        source = subprocess.run(
            ["git", "show", f"{origin}:{MODULE}"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    tmp = Path(f"/tmp/_registry_{name}.py")
    tmp.write_text(source)
    spec = importlib.util.spec_from_file_location(f"_reg_{name}", tmp)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cells(module) -> dict[tuple[str, str], float]:
    return {(entry.group, entry.key): entry.declared for entry in module.DECLARED_NUMBERS}


"""Normalise a branch quantity label onto the canonical one.

The two branches registered the same physical cells under different labels: the
branch used ``level<n>_`` prefixes and ``ka2p58_`` keys, main uses ``sub<n>_``
and ``ka2.58_``. The *quantity* is the semantic identity of a cell, so the audit
compares quantities (and table family), never the registry key text.
"""
LEVEL = re.compile(r"^(sub|level)(\d)_")


def normalise_quantity(quantity: str) -> str:
    quantity = LEVEL.sub(lambda match: "L" + match.group(2) + "_", quantity)
    return quantity.replace("ka2p58_", "ka2.58_")


GROUP_MAP = {
    "hulme3d": "hulme_hemisphere",
    "hulme3d_chief": "hulme_chief",
    "wang_linton3d": "submerged_sphere",
}

CORRECTED = {
    "L2_observed_order_b11",
    "L3_observed_order_b11",
    "L1_rotational_added_mass_max_ratio",
    "L2_rotational_added_mass_max_ratio",
}


def identity(entry) -> tuple[str, str]:
    return (GROUP_MAP.get(entry.group, entry.group), normalise_quantity(entry.quantity))


def cells(module) -> dict[tuple[str, str], tuple[str, float]]:
    return {
        identity(entry): (entry.key, entry.declared) for entry in module.DECLARED_NUMBERS
    }


def main() -> int:
    base = cells(load("4f5b73e", "base"))
    branch = cells(load("origin/agent/m3181", "branch"))
    upstream = cells(load("origin/main", "main"))
    merged = cells(load(None, "merged"))

    problems: list[str] = []
    for label, side in (("branch", branch), ("main", upstream)):
        for cell in side:
            if cell not in merged:
                problems.append(f"{label} cell lost: {cell}")
    for cell, (key, declared) in branch.items():
        if cell not in upstream:
            continue
        main_value = upstream[cell][1]
        if declared != main_value:
            flag = "corrected" if cell[1] in CORRECTED else "UNEXPECTED"
            print(
                f"  [{flag}] {cell[1]}: branch {declared:g} vs main {main_value:g}"
                f"  (merged keeps {merged[cell][1]:g})"
            )
            if cell[1] not in CORRECTED:
                problems.append(f"unexplained value conflict on {cell}")

    shared = set(branch) & set(upstream)
    union = len(branch) + len(upstream) - len(shared)
    print(f"base cells      : {len(base)}")
    print(f"branch cells    : {len(branch)}")
    print(f"main cells      : {len(upstream)}")
    print(f"shared cells    : {len(shared)}")
    print(f"parents union   : {union}")
    print(f"merged cells    : {len(merged)}")
    if len(merged) != union:
        problems.append(f"merged holds {len(merged)} cells, parents union is {union}")
    for cell in sorted(set(merged) - (set(branch) | set(upstream))):
        problems.append(f"merged cell with no parent: {cell}")
    for cell in sorted(branch.keys() - set(merged)):
        problems.append(f"branch-only cell missing from merge: {cell}")

    groups = collections.Counter(family for family, _ in merged)
    print("\nmerged cells per table:", dict(sorted(groups.items())))
    print(
        "\nbranch-only cells ported into the merge:",
        sorted(merged[cell][0] for cell in set(branch) - set(upstream)),
    )
    print(
        "main-only cells kept:",
        sorted(merged[cell][0] for cell in set(upstream) - set(branch)),
    )

    if problems:
        print("\nFAIL")
        for problem in problems:
            print("  -", problem)
        return 1
    print("\nPASS: every cell of both parents has exactly one merged row")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
