#!/usr/bin/env python3
"""Semantic-union resolver for the single conflicted file of PR #134.

conflict: src/tensorbem/benchmarks/declared_numbers.py
both sides are purely additive (each appended new DeclaredNumber records /
new benchmark-group branches at the same anchor), so the union is
"HEAD block, then m3236 block", with a closing '),' inserted after the HEAD
block of hunk 1 (the HEAD side stops mid-record because the shared trailing
context supplies the last record's closing paren).

The script is idempotent and fails loudly on anything unexpected: it never
drops a side, never reorders content, and refuses to run when the number of
conflict hunks is not the audited one.
"""
from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

TARGET = Path("src/tensorbem/benchmarks/declared_numbers.py")

# hunk index -> text inserted between the HEAD side and the m3236 side
SEPARATORS = {
    0: "    ),\n",  # close the HEAD side's last DeclaredNumber(...) record
    1: "",         # two independent `if group == ...` branches
}


def resolve(text: str) -> tuple[str, list[dict]]:
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    report: list[dict] = []
    i = 0
    hunk = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("<<<<<<<"):
            ours_start = i + 1
            j = i + 1
            # two-way markers end "ours" at '======='; diff3 ("--diff3") inserts a
            # '|||||||' base section in between, which must not be swallowed.
            while not (lines[j].startswith("=======") or lines[j].startswith("|||||||")):
                j += 1
            if lines[j].startswith("|||||||"):
                while not lines[j].startswith("======="):
                    j += 1
            theirs_start = j + 1
            k = theirs_start
            while not lines[k].startswith(">>>>>>>"):
                k += 1
            ours = lines[ours_start:j]
            theirs = lines[theirs_start:k]
            sep = SEPARATORS.get(hunk)
            if sep is None:
                raise SystemExit(f"unaudited conflict hunk #{hunk} at line {i + 1}")
            out.extend(ours)
            out.append(sep)
            out.extend(theirs)
            report.append(
                {
                    "hunk": hunk,
                    "ours_lines": len(ours),
                    "theirs_lines": len(theirs),
                    "separator": sep,
                    "ours_sha": hashlib.sha1("".join(ours).encode()).hexdigest()[:12],
                    "theirs_sha": hashlib.sha1("".join(theirs).encode()).hexdigest()[:12],
                }
            )
            hunk += 1
            i = k + 1
            continue
        out.append(line)
        i += 1
    if hunk != len(SEPARATORS):
        raise SystemExit(f"expected {len(SEPARATORS)} conflict hunks, found {hunk}")
    return "".join(out), report


def main() -> int:
    original = TARGET.read_text()
    if "<<<<<<<" not in original:
        raise SystemExit("no conflict markers present; nothing to resolve")
    merged, report = resolve(original)
    ast.parse(merged)  # syntax gate before writing
    TARGET.write_text(merged)
    print(f"resolved {len(report)} hunk(s) -> {TARGET}")
    for item in report:
        print("  ", item)
    return 0


if __name__ == "__main__":
    sys.exit(main())