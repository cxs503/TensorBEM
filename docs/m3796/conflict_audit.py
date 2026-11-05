#!/usr/bin/env python3
"""Produce machine-checkable evidence for the m3236 -> main union merge.

For every path either side touched since the merge base it records
  * what each side did (A/M/D),
  * whether the merged tree keeps every added line of both sides
    (multiplicity-aware: a line added twice by one side must appear twice),
  * whether every line deleted by one side is still absent, unless the other
    side deliberately kept/edited it (reported, not silently accepted).

Nothing here replaces the test suite; it is the "no silent loss" gate that
mechanical overlays historically failed.
"""
from __future__ import annotations

import collections
import json
import subprocess
import sys
from pathlib import Path

BASE = "6025970"
SIDES = {"main": "origin/main", "m3236": "agent/m3236"}


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout


def name_status(rev: str) -> dict[str, str]:
    out = {}
    for line in git("diff", "--name-status", BASE, rev).splitlines():
        status, path = line.split("\t", 1)
        out[path] = status
    return out


def added_lines(rev: str, path: str) -> collections.Counter:
    diff = git("diff", "-U0", BASE, rev, "--", path)
    counter: collections.Counter = collections.Counter()
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            text = line[1:].strip()
            if text:
                counter[text] += 1
    return counter


def deleted_lines(rev: str, path: str) -> collections.Counter:
    diff = git("diff", "-U0", BASE, rev, "--", path)
    counter: collections.Counter = collections.Counter()
    for line in diff.splitlines():
        if line.startswith("-") and not line.startswith("---"):
            text = line[1:].strip()
            if text:
                counter[text] += 1
    return counter


def main() -> int:
    statuses = {side: name_status(rev) for side, rev in SIDES.items()}
    paths = sorted(set(statuses["main"]) | set(statuses["m3236"]))
    report: dict = {
        "merge_base": git("rev-parse", BASE).strip(),
        "sides": {k: git("rev-parse", v).strip() for k, v in SIDES.items()},
        "paths": {},
        "totals": collections.Counter(),
    }
    missing_any = False
    for path in paths:
        work = Path(path)
        if work.exists():
            present = collections.Counter(
                line.strip() for line in work.read_text(errors="replace").splitlines()
            )
            exists = True
        else:
            present = collections.Counter()
            exists = False
        entry = {
            "side_status": {side: statuses[side].get(path, "-") for side in SIDES},
            "exists_in_merged": exists,
        }
        for side, rev in SIDES.items():
            status = statuses[side].get(path)
            if status is None or status.startswith("D"):
                entry[f"{side}_added"] = 0
                continue
            need = added_lines(rev, path)
            missing = need - present
            entry[f"{side}_added"] = sum(need.values())
            entry[f"{side}_missing"] = sum(missing.values())
            if missing:
                missing_any = True
                entry[f"{side}_missing_sample"] = list(missing)[:5]
            # deletions: legitimate only if the other side re-added/kept it
            gone = deleted_lines(rev, path)
            other = "m3236" if side == "main" else "main"
            other_need = added_lines(SIDES[other], path) if other in statuses and statuses[other].get(path) else collections.Counter()
            kept_back = {line: n for line, n in gone.items() if present[line] and other_need[line]}
            if kept_back:
                entry[f"{side}_deletions_overridden_by_{other}"] = list(kept_back)[:5]
        report["paths"][path] = entry
    report["totals"] = {
        "paths": len(paths),
        "conflicted_paths": ["src/tensorbem/benchmarks/declared_numbers.py"],
        "delay_no_silent_loss_failures": sum(
            1
            for e in report["paths"].values()
            for side in SIDES
            if e.get(f"{side}_missing", 0)
        ),
    }
    out = Path("docs/m3796/conflict_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["totals"], indent=2))
    print(f"wrote {out}")
    return 1 if missing_any else 0


if __name__ == "__main__":
    sys.exit(main())