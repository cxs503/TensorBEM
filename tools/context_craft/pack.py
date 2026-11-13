"""Task brief package assembly ("干货包" packer).

Given a corpus of fact units, a task query and a token budget, select the
smallest set of lines that still carries every class of *decision-critical*
knowledge the task needs: at least one acceptance threshold, at least one
negative result / unsupported claim, at least one recorded pitfall — then fill
the rest of the budget by lexical relevance weighted with kind priority.

Deterministic by construction: same corpus + query + budget + profile yields
byte-identical provenance (hence identical ``package_sha256``).
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .model import (
    FLOOR_KINDS,
    KIND_WEIGHT,
    BriefPackage,
    FactItem,
    dedupe,
    estimate_tokens,
    normalize,
)

_ASCII_WORD = re.compile(r"[a-z0-9_][a-z0-9_+.\-/]{1,}")
_CJK_CHAR = re.compile(r"[\u3400-\u9fff]")


def tokenize(text: str) -> list[str]:
    """Mixed-script tokenizer: ASCII words + CJK character bigrams."""
    low = text.lower()
    toks = _ASCII_WORD.findall(low)
    cjk = _CJK_CHAR.findall(low)
    toks.extend(cjk[k] + cjk[k + 1] for k in range(len(cjk) - 1))
    if len(cjk) == 1:
        toks.append(cjk[0])
    return toks


# --------------------------------------------------------------- profiles
@dataclass(frozen=True)
class TaskProfile:
    """What flavour of task is being briefed (drives kind priority only)."""

    name: str
    kind_boost: dict[str, float] = field(default_factory=dict)
    floor_kinds: tuple[str, ...] = FLOOR_KINDS
    min_floor_items: int = 1
    # Two-stream split: (own-work-package share, cross-document share).
    # Round-1 A/B showed a pure global packer loses to file-local reading, so
    # the protocol reserves the majority of the budget for the work package's
    # own artifacts and keeps a fixed slice for cross-document KO/verdicts.
    local_share: float = 0.70

    def weight(self, kind: str) -> float:
        return KIND_WEIGHT.get(kind, 0.4) * self.kind_boost.get(kind, 1.0)


PROFILES: dict[str, TaskProfile] = {
    "verification": TaskProfile(
        "verification", {"TOLERANCE": 1.20, "PIN": 1.15, "NEGATIVE": 1.05, "CONTEXT": 0.7}
    ),
    "defect_fix": TaskProfile(
        "defect_fix", {"NEGATIVE": 1.20, "TRAP": 1.15, "TOLERANCE": 1.05, "CONTEXT": 0.7}
    ),
    "feature": TaskProfile(
        "feature", {"VERDICT": 1.15, "PIN": 1.10, "TOLERANCE": 1.05, "CONTEXT": 0.8}
    ),
    "audit": TaskProfile(
        "audit", {"NEGATIVE": 1.15, "TOLERANCE": 1.15, "TRAP": 1.05, "CONTEXT": 0.6}
    ),
    "generic": TaskProfile("generic"),
}


def get_profile(name: str) -> TaskProfile:
    return PROFILES.get(name, PROFILES["generic"])


# ----------------------------------------------------------------- BM25
class LexicalIndex:
    """Corpus statistics for a BM25-style relevance score."""

    def __init__(self, items: Sequence[FactItem], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(i.text)) for i in items]
        self.lens = [sum(c.values()) for c in self.docs]
        self.avgdl = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df: Counter[str] = Counter()
        for c in self.docs:
            df.update(c.keys())
        n = len(self.docs) or 1
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.default_idf = math.log(1 + (n + 0.5) / 0.5)
        # inverted index: term -> [(doc_index, term_frequency)]
        postings: dict[str, list[tuple[int, int]]] = {}
        for idx, c in enumerate(self.docs):
            for t, tf in c.items():
                postings.setdefault(t, []).append((idx, tf))
        self.postings = postings

    def score_all(self, query_terms: Sequence[str]) -> list[float]:
        """BM25 score of every document against the query (inverted-index walk).

        Only documents that share at least one query term are touched, so a
        corpus of tens of thousands of lines stays cheap per query.
        """
        acc = [0.0] * len(self.docs)
        k1, b, avgdl = self.k1, self.b, self.avgdl or 1.0
        for t in set(query_terms):
            postings = self.postings.get(t)
            if not postings:
                continue
            idf = self.idf.get(t, self.default_idf)
            for idx, tf in postings:
                dl = self.lens[idx] or 1
                acc[idx] += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
        return acc

    def score(self, idx: int, query_terms: Iterable[str]) -> float:
        c = self.docs[idx]
        dl = self.lens[idx] or 1
        s = 0.0
        for t in query_terms:
            tf = c.get(t, 0)
            if not tf:
                continue
            idf = self.idf.get(t, self.default_idf)
            s += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s


def rank(
    items: Sequence[FactItem],
    query: str,
    profile: TaskProfile | None = None,
    index: "LexicalIndex | None" = None,
) -> list[tuple[FactItem, float]]:
    """Return (item, combined score) sorted deterministically.

    ``index`` may be supplied to reuse corpus statistics across many queries.
    """
    profile = profile or PROFILES["generic"]
    index = index or LexicalIndex(items)
    terms = tokenize(query)
    lex_all = index.score_all(terms)
    scored: list[tuple[FactItem, float]] = [
        (it, lex_all[k] * profile.weight(it.kind)) for k, it in enumerate(items)
    ]
    scored.sort(key=lambda p: (-p[1], p[0].item_id))
    return scored


# ----------------------------------------------------------------- pack
def _shingles(text: str) -> set[str]:
    norm = normalize(text)
    return {norm[k : k + 3] for k in range(max(1, len(norm) - 2))}


def _is_duplicate(it: FactItem, norms: set[str], shs: list[set[str]], threshold: float) -> bool:
    norm = normalize(it.text)
    if norm in norms:
        return True
    sh = _shingles(it.text)
    for other in shs:
        if len(sh & other) / (len(sh | other) or 1) >= threshold:
            return True
    return False


def pack(
    items: Sequence[FactItem],
    query: str,
    budget_tokens: int,
    profile: str | TaskProfile = "generic",
    dedupe_threshold: float = 0.85,
    enforce_floor: bool = True,
    index: "LexicalIndex | None" = None,
) -> BriefPackage:
    """Assemble a brief package under a hard token budget.

    Order of operations is part of the contract (see protocol doc §2):
      1. retention floor - one best-ranked line per floor kind
      2. greedy fill by relevance x kind priority, near-duplicates suppressed
      3. hard budget: an item that does not fit is skipped, never truncated
    """
    prof = profile if isinstance(profile, TaskProfile) else get_profile(profile)
    pkg = BriefPackage(query=query, budget_tokens=int(budget_tokens))

    ranked = rank(items, query, prof, index=index)
    order = {it.item_id: n for n, (it, _s) in enumerate(ranked)}
    chosen: list[FactItem] = []
    taken_ids: set[str] = set()
    accepted_norms: set[str] = set()
    accepted_shingles: list[set[str]] = []
    used = 0

    def accept(it: FactItem) -> None:
        nonlocal used
        chosen.append(it)
        taken_ids.add(it.item_id)
        accepted_norms.add(normalize(it.text))
        accepted_shingles.append(_shingles(it.text))
        used += it.tokens

    def fits(it: FactItem) -> bool:
        return used + it.tokens <= budget_tokens

    # 1) retention floor
    if enforce_floor:
        for kind in prof.floor_kinds:
            got = 0
            for it, _score in ranked:
                if it.kind != kind or it.item_id in taken_ids or not fits(it):
                    continue
                accept(it)
                got += 1
                if got >= prof.min_floor_items:
                    break

    # 2) greedy fill (cheap tests first: fit, then duplicate)
    min_tok = min((i.tokens for i in items), default=1)
    for it, _score in ranked:
        if used + min_tok > budget_tokens:
            break
        if it.item_id in taken_ids:
            continue
        if not fits(it):
            pkg.skipped_over_budget += 1
            continue
        if _is_duplicate(it, accepted_norms, accepted_shingles, dedupe_threshold):
            pkg.dropped_duplicates += 1
            continue
        accept(it)

    pkg.items = sorted(chosen, key=lambda i: order.get(i.item_id, 10 ** 9))
    return pkg


def pack_two_stream(
    items: Sequence[FactItem],
    query: str,
    budget_tokens: int,
    profile: str | TaskProfile = "generic",
    local_sources: Sequence[str] = (),
    local_share: float | None = None,
    index: "LexicalIndex | None" = None,
) -> BriefPackage:
    """Brief package with a work-package / cross-document budget split.

    Stream A (``local_sources``) is the task's own artifacts; stream B is the
    rest of the corpus (prior missions, negative knowledge, pins).  Stream A
    gets ``local_share`` of the budget, stream B the remainder, and the floor
    kinds are enforced globally (a criterion already carried by stream A is not
    bought twice).
    """
    prof = profile if isinstance(profile, TaskProfile) else get_profile(profile)
    share = prof.local_share if local_share is None else local_share
    local_set = {s.replace(os.sep, "/") for s in local_sources}

    def norm_src(it: FactItem) -> str:
        return it.source.replace(os.sep, "/")

    local_items = [i for i in items if norm_src(i) in local_set]
    other_items = [i for i in items if norm_src(i) not in local_set]

    budget_local = int(round(budget_tokens * share))
    budget_other = max(0, budget_tokens - budget_local)

    pkg_local = pack(local_items, query, budget_local, prof, index=index) if local_items else \
        BriefPackage(query=query, budget_tokens=budget_local)
    pkg_other = pack(other_items, query, budget_other, prof,
                     enforce_floor=not local_items, index=index) if other_items else \
        BriefPackage(query=query, budget_tokens=budget_other)

    merged: list[FactItem] = []
    used = 0
    seen_norms: set[str] = set()
    for it in pkg_local.items + pkg_other.items:
        norm = normalize(it.text)
        if norm in seen_norms:
            continue
        if used + it.tokens > budget_tokens:
            continue
        seen_norms.add(norm)
        merged.append(it)
        used += it.tokens

    ranked = rank(items, query, prof, index=index)
    # 3) top-up: an under-filled stream budget must not shrink the package
    for it, _score in ranked:
        if used >= budget_tokens:
            break
        if used + it.tokens > budget_tokens:
            continue
        norm = normalize(it.text)
        if norm in seen_norms:
            continue
        seen_norms.add(norm)
        merged.append(it)
        used += it.tokens

    order = {it.item_id: n for n, (it, _s) in enumerate(ranked)}
    out = BriefPackage(query=query, budget_tokens=int(budget_tokens))
    out.items = sorted(merged, key=lambda i: order.get(i.item_id, 10 ** 9))
    out.dropped_duplicates = pkg_local.dropped_duplicates + pkg_other.dropped_duplicates
    out.skipped_over_budget = pkg_local.skipped_over_budget + pkg_other.skipped_over_budget
    return out


def pack_from_files(
    root: str,
    patterns: Sequence[str],
    query: str,
    budget_tokens: int,
    profile: str = "generic",
) -> BriefPackage:
    from .extract import extract_corpus

    return pack(extract_corpus(root, patterns), query, budget_tokens, profile)


def package_digest(pkg: BriefPackage) -> str:
    """Independent re-derivation of the package hash (self-check helper).

    Recomputes the hash from the provenance sidecar rather than trusting the
    value cached on the package object.
    """
    from .model import canon_json, sha256_text

    return sha256_text(canon_json(pkg.provenance()))


def redundancy_ratio(items: Sequence[FactItem], threshold: float = 0.85) -> float:
    """Fraction of lines that a near-duplicate filter would remove."""
    if not items:
        return 0.0
    _kept, dropped = dedupe(items, threshold)
    return dropped / len(items)


def lexical_overlap(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def fact_text_budget(items: Iterable[FactItem]) -> int:
    return sum(estimate_tokens(normalize(i.text)) for i in items)