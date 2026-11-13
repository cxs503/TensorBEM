"""Corpus extraction: turn repository prose into atomic, hash-bound fact units.

Design rules
------------
* Read-only.  Extraction never writes into the corpus.
* Every unit keeps ``source`` + ``lineno`` + ``sha256`` of the exact line, so a
  brief package can be re-verified against the tree (see ``verify`` in cli.py).
* Classification is curation-semantic (does this line carry a threshold / a
  negative result / a pin?), not domain-semantic: no solver, ship or physics
  vocabulary is hard-coded here.
"""

from __future__ import annotations

import os
import re
from typing import Iterator, Sequence

from .model import FactItem

# --- classification markers -------------------------------------------------
_M_TOLERANCE = re.compile(
    r"(误差|容差|阈值|判据|验收线|tolerance|threshold|criteri|acceptable|<\s*\d+(\.\d+)?\s*%|"
    r"\d+(\.\d+)?\s*%\s*(以内|以下|不可接受))",
    re.I,
)
_M_NEGATIVE = re.compile(
    r"(fail|失败|不可接受|不适用|不支持|禁止|不得|局限|限制|未支持|not[_ ]supported|unsupported|"
    r"反例|反证|不能|无法|欠账|债务)",
    re.I,
)
_M_TRAP = re.compile(r"(坑|陷阱|trap|踩过|踩坑|counterexample|易错)", re.I)
_M_VERDICT = re.compile(
    r"(pass|通过|已核|verified|证据等级|\bREAL\b|\bMOCK\b|\bSTUB\b|verdict|裁定|复现)", re.I
)
_M_PIN = re.compile(
    r"(pin|针值|锚|锁定|frozen|sha256|commit|@\s*[0-9a-f]{7,}|registry|台账|基准值)", re.I
)
_NUM = re.compile(r"\d")

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE = re.compile(r"^\s*(```|~~~)")


def classify(text: str) -> str:
    """Assign one kind per line, by decision-relevance priority."""
    if _M_TOLERANCE.search(text) and _NUM.search(text):
        return "TOLERANCE"
    if _M_NEGATIVE.search(text):
        return "NEGATIVE"
    if _M_TRAP.search(text):
        return "TRAP"
    if _M_VERDICT.search(text):
        return "VERDICT"
    if _M_PIN.search(text):
        return "PIN"
    return "CONTEXT"


def _clean(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^[-*>+]\s+", "", line)          # list / quote markers
    line = re.sub(r"^\d+[.)]\s+", "", line)          # ordered list markers
    line = re.sub(r"\|", " ", line)                  # table pipes -> spaces
    return re.sub(r"\s+", " ", line).strip()


def is_fact_line(text: str, min_chars: int = 24, max_chars: int = 400) -> bool:
    """Keep prose lines that can carry a decision; drop separators/headers."""
    if not (min_chars <= len(text) <= max_chars):
        return False
    if set(text) <= set("-: |\t"):
        return False
    if re.fullmatch(r"[\d\s.%:|/-]+", text):         # bare numeric table row
        return True
    return bool(re.search(r"[A-Za-z\u4e00-\u9fff]", text))


def iter_markdown(path: str) -> Iterator[tuple[int, str, str]]:
    """Yield (lineno, heading_breadcrumb, cleaned_text) for prose lines."""
    stack: list[tuple[int, str]] = []
    in_fence = False
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for lineno, raw in enumerate(fh, 1):
                if _FENCE.match(raw):
                    in_fence = not in_fence
                    continue
                if in_fence:
                    continue
                m = _HEADING.match(raw)
                if m:
                    level = len(m.group(1))
                    stack = [s for s in stack if s[0] < level]
                    stack.append((level, m.group(2).strip()))
                    continue
                text = _clean(raw)
                if not is_fact_line(text):
                    continue
                yield lineno, " / ".join(h for _, h in stack), text
    except (OSError, UnicodeError):
        return


def extract_file(path: str, rel: str | None = None) -> list[FactItem]:
    rel = rel or os.path.basename(path)
    out: list[FactItem] = []
    for lineno, heading, text in iter_markdown(path):
        out.append(FactItem.build(classify(text), text, rel, lineno, heading))
    return out


def extract_corpus(root: str, patterns: Sequence[str]) -> list[FactItem]:
    """Walk ``root`` collecting markdown files matching any fnmatch pattern."""
    import fnmatch

    items: list[FactItem] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if any(p in ("/.git", "/node_modules", "/.venv") for p in ("/" + dirpath,)):
            continue
        for name in sorted(filenames):
            if not name.endswith((".md", ".markdown")):
                continue
            rel_os = os.path.relpath(os.path.join(dirpath, name), root)
            if patterns and not any(fnmatch.fnmatch(rel_os, p) for p in patterns):
                continue
            items.extend(extract_file(os.path.join(dirpath, name), rel_os))
    items.sort(key=lambda i: (i.source, i.lineno))
    return items


def section_of(path: str, lineno: int) -> str:
    """Heading breadcrumb for a line (used to build held-out splits)."""
    best = ""
    for n, heading, _text in iter_markdown(path):
        if n > lineno:
            break
        best = heading
    return best