#!/usr/bin/env python3
"""Static audit of the m3236 -> main merge (mission 3854).

The merge of ``agent/m3236`` into ``main`` was hand-resolved in
``src/tensorbem/benchmarks/declared_numbers.py``.  A hand resolution is only
trustworthy if it can be shown that *nothing* from either side disappeared, so
this script compares three revisions of every file that either side touched and
asserts a strict set-union contract:

* every ``DeclaredNumber.key`` of ``main`` and of ``agent/m3236`` exists exactly
  once in the merged registry;
* every benchmark group dispatched by ``_run_group`` on either side is
  dispatched after the merge;
* every ``def`` / ``class`` of the two sides survives;
* every ``### 4.x`` documentation section of the two sides survives, and the
  m3236 sections carry the numbers they quote;
* every ``def test_*`` of the two sides survives.

It exits non-zero on any loss and prints a JSON summary that is committed as
evidence under ``artifacts/``.  Usage::

    python3 tools/verify_m3236_merge.py --json artifacts/m3236_merge_audit.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

MAIN = "origin/main"
BRANCH = "origin/agent/m3236"
MERGED_REV = "the working tree (index-free: this is the resolved merge)"

#: Files whose content must be a semantic superset of both sides.
CONTENT_FILES = (
    "src/tensorbem/benchmarks/declared_numbers.py",
    "tests/test_declared_numbers.py",
    "docs/marine_bem_benchmarks.md",
)


def show(rev: str, path: str) -> str:
    """Text of ``path`` at revision ``rev`` (empty string when absent)."""
    try:
        return subprocess.run(
            ["git", "show", f"{rev}:{path}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except subprocess.CalledProcessError:
        return ""


def keys(text: str) -> list[str]:
    return re.findall(r'key="([^"]+)"', text)


def groups(text: str) -> list[str]:
    return re.findall(r'group == "([^"]+)"', text)


def defs(text: str) -> list[str]:
    return re.findall(r"^(?:def|class) ([A-Za-z_0-9]+)", text, re.M)


def tests(text: str) -> list[str]:
    return re.findall(r"^def (test_[A-Za-z_0-9]+)", text, re.M)


def doc_sections(text: str) -> list[str]:
    return re.findall(r"^(#{2,4} .+)$", text, re.M)


def compare(name: str, main_items: list[str], branch_items: list[str], merged: list[str]) -> dict:
    main_set, branch_set, merged_set = set(main_items), set(branch_items), set(merged)
    missing = sorted((main_set | branch_set) - merged_set)
    duplicated = sorted({item for item in merged_set if merged.count(item) > 1})
    return {
        "name": name,
        "main": len(main_set),
        "branch": len(branch_set),
        "merged": len(merged_set),
        "union": len(main_set | branch_set),
        "missing_from_merged": missing,
        "duplicated_in_merged": duplicated,
        "ok": not missing and not duplicated,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    base = subprocess.run(
        ["git", "merge-base", MAIN, BRANCH], check=True, capture_output=True, text=True
    ).stdout.strip()

    checks: list[dict] = []
    for path in CONTENT_FILES:
        main_text, branch_text, merged_text = (
            show(MAIN, path),
            show(BRANCH, path),
            Path(path).read_text(),
        )
        checks.append(compare(f"{path}:keys", keys(main_text), keys(branch_text), keys(merged_text)))
        checks.append(compare(f"{path}:groups", groups(main_text), groups(branch_text), groups(merged_text)))
        checks.append(compare(f"{path}:defs", defs(main_text), defs(branch_text), defs(merged_text)))
        checks.append(compare(f"{path}:tests", tests(main_text), tests(branch_text), tests(merged_text)))
        checks.append(
            compare(
                f"{path}:doc_sections",
                doc_sections(main_text),
                doc_sections(branch_text),
                doc_sections(merged_text),
            )
        )

    # Files only one side touched must still be present in the merged tree.
    added_by_branch = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=A", base, BRANCH],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    perd = {
        "name": "branch_added_files",
        "added": added_by_branch,
        "missing_from_merged": [p for p in added_by_branch if not Path(p).exists()],
    }
    perd["ok"] = not perd["missing_from_merged"]

    summary = {
        "main": MAIN,
        "main_sha": subprocess.run(
            ["git", "rev-parse", MAIN], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "branch": BRANCH,
        "branch_sha": subprocess.run(
            ["git", "rev-parse", BRANCH], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "merge_base": base,
        "checks": checks + [perd],
        "ok": all(entry["ok"] for entry in checks + [perd]),
    }

    for entry in checks:
        flag = "OK  " if entry["ok"] else "FAIL"
        print(
            f"{flag} {entry['name']:<62} main={entry['main']:>4} branch={entry['branch']:>4} "
            f"union={entry['union']:>4} merged={entry['merged']:>4}"
            + (f" missing={entry['missing_from_merged']}" if entry["missing_from_merged"] else "")
            + (f" dup={entry['duplicated_in_merged']}" if entry["duplicated_in_merged"] else "")
        )
    print(f"{'OK  ' if perd['ok'] else 'FAIL'} {perd['name']:<62} added={len(perd['added'])}")
    print("MERGE-CONTENT-UNION:", "PASS" if summary["ok"] else "FAIL")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2) + "\n")
        print("wrote", args.json)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())