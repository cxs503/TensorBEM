"""One-shot conflict resolver for the m3181 -> main three-way merge.

Semantics: union of both sides.  Every conflicted hunk is resolved by keeping
the complete set of lines of *both* sides (order: HEAD first, then m3181), with
alphabetical de-duplication for the import block only.  Nothing is dropped.
"""

from __future__ import annotations

import sys
from pathlib import Path


def split_conflicts(text: str):
    """Yield ('plain', lines) / ('conflict', head_lines, other_lines) chunks."""
    out = []
    plain: list[str] = []
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("<<<<<<<"):
            if plain:
                out.append(("plain", "".join(plain)))
                plain = []
            head: list[str] = []
            other: list[str] = []
            i += 1
            while not lines[i].startswith("======="):
                head.append(lines[i])
                i += 1
            i += 1
            while not lines[i].startswith(">>>>>>>"):
                other.append(lines[i])
                i += 1
            i += 1
            out.append(("conflict", "".join(head), "".join(other)))
        else:
            plain.append(line)
            i += 1
    if plain:
        out.append(("plain", "".join(plain)))
    return out


def resolve(path: Path, merge_conflict) -> str:
    chunks = split_conflicts(path.read_text())
    parts: list[str] = []
    index = 0
    for chunk in chunks:
        if chunk[0] == "plain":
            parts.append(chunk[1])
            continue
        head, other = chunk[1], chunk[2]
        parts.append(merge_conflict(index, head, other))
        index += 1
    return "".join(parts)


def union_imports(_index: int, head: str, other: str) -> str:
    """Union of the two import-line sets, sorted (deterministic, deduped)."""
    lines = {
        line for line in (head + other).splitlines(keepends=True) if line.strip()
    }
    return "".join(sorted(lines))


def keep_both(index: int, head: str, other: str) -> str:
    """HEAD（main 侧）内容在前，m3181 侧内容紧随其后，两侧皆不丢。"""
    return head + other


SRC_PLAN = {0: union_imports, 1: keep_both, 2: keep_both}
TEST_PLAN = {0: keep_both, 1: keep_both}


def main() -> int:
    for path, plan in (
        (Path("src/tensorbem/benchmarks/declared_numbers.py"), SRC_PLAN),
        (Path("tests/test_declared_numbers.py"), TEST_PLAN),
    ):
        resolved = resolve(path, lambda i, h, o, plan=plan: plan[i](i, h, o))
        assert "<<<<<<<" not in resolved and ">>>>>>>" not in resolved
        path.write_text(resolved)
        print(f"resolved {path} ({len(plan)} conflict hunks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())