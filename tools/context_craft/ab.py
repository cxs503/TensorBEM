"""Offline A/B harness: brief package arm vs bare-context arms.

Held-out construction (no leakage):
    query  <- report title + sections 1-2   (case setup / method)
    targets<- fact units in the "caveats / error verdict" sections (§5 §6 ...)
    corpus <- every markdown fact unit in the repository

Arms
    head_dump : read the task's own file top-down until the budget is full
                (what an agent does with "just read the doc")
    bm25_raw  : pure lexical ranking over the whole corpus, no kind floor,
                no duplicate filter (a "good search" without curation)
    brief     : this package's packer (kind floor + kind weighting + dedupe)

Metric = fraction of the held-out decision facts that reach the context
window under a fixed token budget, plus a "gate" metric: does at least one
acceptance-threshold line make it in.
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
from dataclasses import dataclass, field
from typing import Sequence

from .extract import extract_corpus, extract_file
from .model import FactItem, dedupe
from .pack import LexicalIndex, tokenize

QUERY_HEADINGS = ("案例介绍", "方法", "背景", "问题", "介绍", "how", "case", "method", "context")
TARGET_HEADINGS = ("注意点", "局限", "误差判定", "陷阱", "坑", "结论", "风险", "caveat", "limitation",
                   "verdict", "lesson")

_SECTION = re.compile(r"^(#{2,3})\s+(.*)$")


@dataclass
class ABTask:
    task_id: str
    path: str
    query: str
    query_linenos: list[int] = field(default_factory=list)
    target_linenos: list[int] = field(default_factory=list)
    target_ids: set[str] = field(default_factory=set)
    target_tolerance_ids: set[str] = field(default_factory=set)


def _sections(path: str) -> list[tuple[str, int, int]]:
    """[(heading, first_lineno, last_lineno)] for level-2/3 headings."""
    heads: list[tuple[str, int]] = []
    n_lines = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for n, raw in enumerate(fh, 1):
            n_lines = n
            m = _SECTION.match(raw)
            if m:
                heads.append((m.group(2).strip(), n))
    out = []
    for k, (h, start) in enumerate(heads):
        end = heads[k + 1][1] - 1 if k + 1 < len(heads) else n_lines
        out.append((h, start, end))
    return out


def build_task(path: str, rel: str, max_query_chars: int = 1200) -> ABTask | None:
    sections = _sections(path)
    if not sections:
        return None
    title = ""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            if raw.startswith("# "):
                title = raw[2:].strip()
                break
    q_parts = [title]
    for h, s, e in sections:
        if any(k in h.lower() for k in QUERY_HEADINGS):
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for n, raw in enumerate(fh, 1):
                    if s <= n <= min(e, s + 40):
                        q_parts.append(raw.strip())
    query = "\n".join(p for p in q_parts if p)[:max_query_chars]
    t_ranges = [(s, e) for h, s, e in sections if any(k in h.lower() for k in TARGET_HEADINGS)]
    if not query.strip() or not t_ranges:
        return None
    facts = extract_file(path, rel)
    targets = [f for f in facts if any(s <= f.lineno <= e for s, e in t_ranges)]
    if len(targets) < 3:
        return None
    task = ABTask(
        task_id=os.path.basename(path),
        path=rel,
        query=query,
        target_ids={t.item_id for t in targets},
        target_tolerance_ids={t.item_id for t in targets if t.kind == "TOLERANCE"},
    )
    return task


def build_tasks(root: str, patterns: Sequence[str], limit: int = 40) -> list[ABTask]:
    import fnmatch

    tasks: list[ABTask] = []
    for dirpath, _dn, filenames in os.walk(root):
        for name in sorted(filenames):
            if not name.endswith(".md"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, name), root)
            if patterns and not any(fnmatch.fnmatch(rel, p) for p in patterns):
                continue
            t = build_task(os.path.join(dirpath, name), rel)
            if t:
                tasks.append(t)
    tasks.sort(key=lambda t: t.task_id)
    return tasks[:limit]


# ------------------------------------------------------------------ arms
def arm_head_dump(task: ABTask, corpus: Sequence[FactItem], budget: int,
                  index: LexicalIndex | None = None) -> list[FactItem]:
    own = [i for i in corpus if i.source == task.path]
    out, used = [], 0
    for it in own:
        if used + it.tokens > budget:
            continue
        out.append(it)
        used += it.tokens
    return out


def arm_bm25_raw(task: ABTask, corpus: Sequence[FactItem], budget: int,
                 index: LexicalIndex | None = None) -> list[FactItem]:
    index = index or LexicalIndex(corpus)
    terms = tokenize(task.query)
    scored = [(it, index.score(k, terms)) for k, it in enumerate(corpus)]
    scored.sort(key=lambda p: (-p[1], p[0].item_id))
    out, used = [], 0
    for it, _s in scored:
        if used + it.tokens > budget:
            continue
        out.append(it)
        used += it.tokens
    return out


def arm_brief(task: ABTask, corpus: Sequence[FactItem], budget: int,
              index: LexicalIndex | None = None) -> list[FactItem]:
    from .pack import pack

    return pack(corpus, task.query, budget, profile="verification", index=index).items


def arm_brief_local(task: ABTask, corpus: Sequence[FactItem], budget: int,
                    index: LexicalIndex | None = None) -> list[FactItem]:
    """Brief package with a work-package/cross-document budget split."""
    from .pack import pack_two_stream

    return pack_two_stream(corpus, task.query, budget, profile="verification",
                           local_sources=[task.path], index=index).items


ARMS = {
    "head_dump": arm_head_dump,        # read your own doc top-down
    "bm25_raw": arm_bm25_raw,          # good search, no curation
    "brief_global": arm_brief,         # curation, no locality prior
    "brief_local": arm_brief_local,    # curation + locality prior (protocol default)
}


def _fill(ordered: Sequence[FactItem], budget: int) -> list[FactItem]:
    out, used = [], 0
    for it in ordered:
        if used + it.tokens <= budget:
            out.append(it)
            used += it.tokens
    return out


def _row(task: ABTask, got: Sequence[FactItem]) -> dict:
    ids = {i.item_id for i in got}
    hit = len(ids & task.target_ids)
    tol_hit = len(ids & task.target_tolerance_ids)
    _kept, dropped = dedupe(got)
    return {
        "task": task.task_id,
        "recall": hit / max(1, len(task.target_ids)),
        "tol_recall": tol_hit / max(1, len(task.target_tolerance_ids)),
        "gate": 1.0 if tol_hit else 0.0,
        "tokens": sum(i.tokens for i in got),
        "redundancy": dropped / max(1, len(got)),
        "decision_share": sum(1 for i in got if i.kind != "CONTEXT") / max(1, len(got)),
        "targets": len(task.target_ids),
    }


def evaluate(tasks: Sequence[ABTask], corpus: Sequence[FactItem], budgets: Sequence[int]) -> dict:
    from .pack import pack

    results: dict = {"tasks": len(tasks), "corpus_items": len(corpus), "budgets": list(budgets),
                     "arms": list(ARMS), "per_budget": {}}
    index = LexicalIndex(corpus)
    rows: dict[str, dict[str, list[dict]]] = {str(b): {a: [] for a in ARMS} for b in budgets}
    for task in tasks:
        own = [i for i in corpus if i.source == task.path]
        terms = tokenize(task.query)
        lex = index.score_all(terms)
        bm25_order = [it for it, _s in sorted(
            ((it, lex[k]) for k, it in enumerate(corpus)),
            key=lambda p: (-p[1], p[0].item_id),
        )]
        for budget in budgets:
            got: dict[str, list[FactItem]] = {
                "head_dump": _fill(own, budget),
                "bm25_raw": _fill(bm25_order, budget),
                "brief_global": pack(corpus, task.query, budget, profile="verification",
                                     index=index).items,
                "brief_local": arm_brief_local(task, corpus, budget, index=index),
            }
            for arm in ARMS:
                rows[str(budget)][arm].append(_row(task, got[arm]))

    for budget in budgets:
        summary = {}
        for arm, rs in rows[str(budget)].items():
            summary[arm] = {
                key: _mean_ci([r[key] for r in rs])
                for key in ("recall", "tol_recall", "gate", "tokens", "redundancy",
                            "decision_share")
            }
        results["per_budget"][str(budget)] = {"summary": summary, "rows": rows[str(budget)]}
    return results


def _mean_ci(xs: Sequence[float]) -> dict:
    if not xs:
        return {"mean": 0.0, "ci95": 0.0, "n": 0}
    m = statistics.fmean(xs)
    if len(xs) < 2:
        return {"mean": m, "ci95": 0.0, "n": len(xs)}
    sd = statistics.stdev(xs)
    return {"mean": m, "ci95": 1.96 * sd / math.sqrt(len(xs)), "n": len(xs)}


def run(root: str, patterns: Sequence[str], budgets: Sequence[int], limit: int,
        out_path: str | None = None) -> dict:
    corpus = extract_corpus(root, patterns)
    tasks = build_tasks(root, patterns, limit=limit)
    res = evaluate(tasks, corpus, budgets)
    res["patterns"] = list(patterns)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2, sort_keys=True)
    return res


def to_markdown(res: dict) -> str:
    lines = [
        "# Context-craft offline A/B",
        "",
        f"- tasks: {res['tasks']} · corpus fact units: {res['corpus_items']}",
        f"- arms: {', '.join(res['arms'])}",
        "",
    ]
    for budget, block in res["per_budget"].items():
        lines += [
            f"## budget {budget} tokens",
            "",
            "| arm | recall | 95%CI | tolerance recall | gate | tokens | redundancy | decision share |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for arm, s in block["summary"].items():
            lines.append(
                f"| {arm} | {s['recall']['mean']:.3f} | ±{s['recall']['ci95']:.3f} | "
                f"{s['tol_recall']['mean']:.3f} | {s['gate']['mean']:.3f} | "
                f"{s['tokens']['mean']:.0f} | {s['redundancy']['mean']:.3f} | "
                f"{s['decision_share']['mean']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines)