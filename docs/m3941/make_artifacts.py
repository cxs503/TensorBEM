"""Artifacts for the PR #134 (agent/m3236) merge-heal report (m3941).

Produces two JSON files next to this script:

``census.json``
    Number of declared-number rows per documentation group, on each merge
    parent and on the merged tree, parsed straight out of the three versions of
    ``src/tensorbem/benchmarks/declared_numbers.py`` (working tree, ``main``,
    ``agent/m3236``). This is the machine check behind the "no row may be lost
    by the merge" claim: for every group the merged count must equal
    ``max(main, branch)`` and the merged key set must be the union of the two
    parents' key sets.

``registry_merged.json``
    The ``declared_numbers.check()`` report for the eleven ``dean`` rows, run
    **from the merged working tree**. Written only with ``--with-check``
    because it costs one Dean-Ursell-Ogilvie 64-panel solve per parameter set.

Usage
-----
    python docs/m3941/make_artifacts.py              # census only
    python docs/m3941/make_artifacts.py --with-check # census + Dean recompute
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
TARGET = "src/tensorbem/benchmarks/declared_numbers.py"
PARENTS = ("origin/main", "origin/agent/m3236")

KEY_RE = re.compile(r'key="([^"]+)"')
GROUP_RE = re.compile(r'group="([^"]+)"')


def _text(ref: str | None) -> str:
    if ref is None:
        return (REPO / TARGET).read_text(encoding="utf-8")
    return subprocess.run(
        ["git", "show", f"{ref}:{TARGET}"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _entries(text: str) -> dict[str, str]:
    keys = KEY_RE.findall(text)
    groups = GROUP_RE.findall(text)
    if len(keys) != len(groups):
        raise SystemExit(
            f"registry parse mismatch: {len(keys)} keys vs {len(groups)} groups"
        )
    return dict(zip(keys, groups))


def census() -> dict:
    merged = _entries(_text(None))
    per_parent = {ref: _entries(_text(ref)) for ref in PARENTS}
    groups = sorted({g for g in merged.values()} | {g for e in per_parent.values() for g in e.values()})
    table = {}
    ok = True
    for g in groups:
        counts = {ref: sum(1 for v in per_parent[ref].values() if v == g) for ref in PARENTS}
        counts["merged"] = sum(1 for v in merged.values() if v == g)
        union = {
            k for ref in PARENTS for k, v in per_parent[ref].items() if v == g
        }
        merged_keys = {k for k, v in merged.items() if v == g}
        missing = sorted(union - merged_keys)
        table[g] = {**counts, "union_of_parents": len(union), "missing": missing}
        ok &= not missing and counts["merged"] == max(counts[x] for x in PARENTS)
    doc = {
        "target": TARGET,
        "parents": list(PARENTS),
        "groups": table,
        "totals": {
            "main": len(per_parent[PARENTS[0]]),
            "branch": len(per_parent[PARENTS[1]]),
            "merged": len(merged),
            "union_of_parents": len(set(per_parent[PARENTS[0]]) | set(per_parent[PARENTS[1]])),
        },
        "no_row_lost": ok,
        "duplicate_keys": sorted(k for k, n in Counter(merged.values()).items() if n > 1),
        "group_counter_merged": dict(Counter(merged.values())),
    }
    return doc


def dean_check() -> dict:
    sys.path.insert(0, str(REPO / "src"))
    from tensorbem.benchmarks import declared_numbers as dn  # noqa: WPS433

    print("module:", dn.__file__)
    entries = tuple(e for e in dn.DECLARED_NUMBERS if e.group == "dean")
    report = dn.check(tiers=("fast",), entries=entries)
    report["module_file"] = dn.__file__
    report["registry_size"] = len(dn.DECLARED_NUMBERS)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-check", action="store_true")
    args = ap.parse_args()

    doc = census()
    (HERE / "census.json").write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("census:", doc["totals"], "no_row_lost=", doc["no_row_lost"])
    if not doc["no_row_lost"]:
        raise SystemExit("merge lost registry rows")

    if args.with_check:
        report = dean_check()
        (HERE / "registry_merged.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print("dean:", report["summary"])


if __name__ == "__main__":
    main()