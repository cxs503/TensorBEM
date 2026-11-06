"""Coverage audit: which documentation numbers are guarded, and which are not.

Background
----------
This package has four separate gate layers that make documented numbers
executable:

===========================  ====================================================
``doc_pins``                 README + ``marine_bem_benchmarks.md`` tables
``doc_claims``               Ursell rows of ``marine_bem_benchmarks.md`` / README
``doc_number_ledger``        MacCamy-Fuchs tables 8 and 9
``declared_numbers``         cylinder / ellipsoid / Ursell / interior harmonics
``three_dimensional_pins``   the four result tables of ``three_dimensional_bem.md``
===========================  ====================================================

Each gate was written *after* a documented number was found to be stale (see the
provenance notes in :mod:`tensorbem.doc_claims` and
``dev_report_tensorbem_declared_numbers.md``).  Nobody, however, has ever asked
the inverse question: **which documents still quote numbers that no gate reads?**

That question is answerable mechanically, and it is the one that predicts the
next incident.  This module answers it.  Every markdown file that states numbers
is classified into one of three tiers:

``PINNED``
    A test reads the document itself (the file name appears in ``tests/``), so
    the printed numbers are parsed and compared.  This is the only tier that
    catches a *document* edit.
``ARTIFACT``
    The document declares its generator (``python -m tensorbem...``), that
    module is imported by a test, and the rendered numbers come from a JSON
    artifact.  The numbers are reproducible, but the document text itself is not
    parsed: an edit that changes a printed digit is not caught.
``UNGUARDED``
    Neither: no test reads the file, no producing module is exercised.  These
    are the documents that can silently rot.

The audit separates *paper-level* documents (specifications, benchmark notes,
verification notes) from *historical* ones (dated dev reports, audits, reviews):
the former are contracts with a reader, the latter are a snapshot of a past run
and are allowed to be frozen.

Usage::

    python -m tensorbem.benchmarks.doc_claim_audit \
        --json docs/doc_claim_audit/audit.json --figures docs/doc_claim_audit

``--check`` exits non-zero when the ``UNGUARDED`` set grew beyond the recorded
baseline, which is what ``tests/test_doc_claim_audit.py`` gates on.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS = REPO_ROOT / "docs"
TESTS = REPO_ROOT / "tests"
SRC = REPO_ROOT / "src"

SCHEMA = "tensorbem.doc_claim_audit/1"

#: A number token in prose: ``8.51e-3``, ``-0.87202``, ``2566.83``, ``1.2%``.
NUM_RE = re.compile(r"(?<![\w.])[-\u2212]?\d+\.\d+(?:[eE][-\u2212]?\d+)?")

#: ``python -m tensorbem.benchmarks.foo`` (also ``python3``, ``PYTHONPATH=src``).
MODULE_RE = re.compile(r"-m\s+(tensorbem(?:\.\w+)+)")

#: Historical files: dated reports, audits, reviews, milestone notes, incident logs.
#: ``^m\d{3,}`` covers the mission-prefixed deliverable form (``m4283_pr186_ci_triage.md``)
#: whose snapshot nature the ``_m\d{3,}`` branch only caught mid-name (m4655, 2026-11-06).
HISTORICAL_RE = re.compile(
    r"(dev_report|report_m\d|report_publish|audit|review|incident|provenance|"
    r"health_audit|_m\d{3,}|_m\d{3,}_zh|discrepancy|charter|^m\d{3,})",
    re.IGNORECASE,
)

#: Documents that are demoted to historical on purpose (English/Chinese singles).
HISTORICAL_EXTRA = frozenset(
    {
        "hydrostatic_verification_m2127_zh.md",
        "finite_depth_radiation.md",  # superseded by finite_depth_envelope.md
    }
)


@dataclass(frozen=True)
class DocRecord:
    doc: str
    tier: str  # PINNED / ARTIFACT / UNGUARDED
    paper_level: bool
    number_tokens: int
    lines: int
    generators: tuple[str, ...]
    reading_tests: tuple[str, ...]
    modules_exercised: tuple[str, ...]
    artifact: str | None = None
    note: str = ""


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _test_sources() -> dict[str, str]:
    return {p.name: _read(p) for p in sorted(TESTS.glob("*.py"))}


def _module_is_exercised(module: str, test_sources: dict[str, str]) -> tuple[str, ...]:
    """Tests that import/mention ``module`` (e.g. ``tensorbem.benchmarks.foo``)."""
    short = module.rsplit(".", 1)[-1]
    patterns = (module, f".benchmarks import {short}", f"import {module}")
    hits = []
    for name, text in test_sources.items():
        if any(pat in text for pat in patterns):
            hits.append(name)
    return tuple(hits)


def _declared_artifact(doc_text: str) -> str | None:
    """First ``docs/.../*.json`` path mentioned by the document, if any."""
    for token in re.findall(r"`?([\w./-]*docs/[\w./-]+\.json)`?", doc_text):
        return token.lstrip("./")
    return None


def is_historical(name: str) -> bool:
    return bool(HISTORICAL_RE.search(name)) or name in HISTORICAL_EXTRA


def classify(doc_path: Path, test_sources: dict[str, str]) -> DocRecord:
    text = _read(doc_path)
    numbers = len(NUM_RE.findall(text))
    generators = tuple(dict.fromkeys(MODULE_RE.findall(text)))
    readers = tuple(n for n, t in test_sources.items() if doc_path.name in t)

    exercised: list[str] = []
    for module in generators:
        for test in _module_is_exercised(module, test_sources):
            exercised.append(test)
    exercised = list(dict.fromkeys(exercised))

    artifact = _declared_artifact(text)

    if readers:
        tier = "PINNED"
        note = f"read by {len(readers)} test file(s)"
    elif generators and exercised:
        tier = "ARTIFACT"
        note = "generator exercised; document text not parsed"
    elif generators:
        tier = "UNGUARDED"
        note = "generator never exercised by a test"
    else:
        tier = "UNGUARDED"
        note = "no declared generator and no reading test"

    return DocRecord(
        doc=doc_path.name,
        tier=tier,
        paper_level=not is_historical(doc_path.name),
        number_tokens=numbers,
        lines=len(text.splitlines()),
        generators=generators,
        reading_tests=readers,
        modules_exercised=tuple(exercised),
        artifact=artifact,
        note=note,
    )


def audit() -> dict:
    test_sources = _test_sources()
    records = [
        classify(p, test_sources)
        for p in sorted(list(DOCS.glob("*.md")) + [REPO_ROOT / "README.md"])
        if p.exists()
    ]
    # docs/<subdir>/*.md are milestone work products, not contracts.
    unguarded = sorted(
        (r for r in records if r.tier == "UNGUARDED"),
        key=lambda r: (-r.number_tokens, r.doc),
    )
    paper_unguarded = [r for r in unguarded if r.paper_level]
    covered = sum(r.number_tokens for r in records if r.tier != "UNGUARDED")
    total = sum(r.number_tokens for r in records)
    return {
        "schema": SCHEMA,
        "counts": {
            "documents": len(records),
            "pinned": sum(r.tier == "PINNED" for r in records),
            "artifact": sum(r.tier == "ARTIFACT" for r in records),
            "unguarded": len(unguarded),
            "unguarded_paper_level": len(paper_unguarded),
            "number_tokens": total,
            "number_tokens_covered": covered,
        },
        "coverage": {
            "guarded_number_fraction": (covered / total) if total else 0.0,
            "documents_guarded_fraction": (
                (len(records) - len(unguarded)) / len(records) if records else 0.0
            ),
        },
        "records": [asdict(r) for r in records],
    }


def check(report: dict, baseline: dict) -> list[str]:
    """Failures against a baseline: the unguarded set may shrink, never grow."""
    failures: list[str] = []
    allowed = set(baseline.get("unguarded", []))
    current = {r["doc"] for r in report["records"] if r["tier"] == "UNGUARDED"}
    for doc in sorted(current - allowed):
        failures.append(f"new unguarded document: {doc}")
    base_paper = set(baseline.get("unguarded_paper_level", []))
    current_paper = {
        r["doc"]
        for r in report["records"]
        if r["tier"] == "UNGUARDED" and r["paper_level"]
    }
    for doc in sorted(current_paper - base_paper):
        failures.append(f"paper-level document lost its guard: {doc}")
    return failures


def render_table(report: dict, limit: int = 20) -> str:
    rows = [
        "| document | tier | numbers | lines | generator |",
        "|---|---|---:|---:|---|",
    ]
    order = {"UNGUARDED": 0, "ARTIFACT": 1, "PINNED": 2}
    recs = sorted(
        report["records"],
        key=lambda r: (not r["paper_level"], order[r["tier"]], -r["number_tokens"]),
    )
    for r in recs[:limit]:
        gen = ", ".join(f"`{g}`" for g in r["generators"]) or "—"
        rows.append(
            f"| `{r['doc']}` | {r['tier']} | {r['number_tokens']} | {r['lines']} | {gen} |"
        )
    return "\n".join(rows)


def make_figure(report: dict, outdir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    recs = [r for r in report["records"] if r["number_tokens"] > 0]
    recs.sort(key=lambda r: ({"UNGUARDED": 0, "ARTIFACT": 1, "PINNED": 2}[r["tier"]],
                             -r["number_tokens"]))
    recs = recs[:22]
    colors = {"UNGUARDED": "#c0392b", "ARTIFACT": "#e67e22", "PINNED": "#27ae60"}
    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    labels = [r["doc"].replace(".md", "") for r in recs][::-1]
    vals = [r["number_tokens"] for r in recs][::-1]
    cols = [colors[r["tier"]] for r in recs][::-1]
    ax.barh(range(len(vals)), vals, color=cols, height=0.72)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("numeric claim tokens in the document")
    ax.set_title(
        "TensorBEM doc-claim audit: numbers per document, coloured by guard tier\n"
        f"(green PINNED={report['counts']['pinned']}, "
        f"orange ARTIFACT={report['counts']['artifact']}, "
        f"red UNGUARDED={report['counts']['unguarded']})",
        fontsize=9,
    )
    for y, v in enumerate(vals):
        ax.text(v + 3, y, str(v), va="center", fontsize=6.5)
    ax.set_xlim(0, max(vals) * 1.12)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = outdir / "doc_claim_audit.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def render_markdown(report: dict) -> str:
    c = report["counts"]
    cov = report["coverage"]
    paper = [r for r in report["records"] if r["tier"] == "UNGUARDED" and r["paper_level"]]
    paper.sort(key=lambda r: -r["number_tokens"])
    lines = [
        "# Doc-claim coverage audit（文档数字守卫覆盖审计）",
        "",
        "本文件由 `python -m tensorbem.benchmarks.doc_claim_audit` 程序化生成，"
        "无手抄数字。",
        "",
        "## 1. 覆盖统计",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
        f"| 被审计文档数 | {c['documents']} |",
        f"| PINNED（测试直接解析文档） | {c['pinned']} |",
        f"| ARTIFACT（生成模块被测、文档文本未解析） | {c['artifact']} |",
        f"| UNGUARDED（无人守） | {c['unguarded']} |",
        f"| 其中论文级 UNGUARDED | {c['unguarded_paper_level']} |",
        f"| 数字 token 总数 | {c['number_tokens']} |",
        f"| 已守数字 token | {c['number_tokens_covered']} |",
        f"| 数字覆盖率 | {cov['guarded_number_fraction']:.1%} |",
        f"| 文档覆盖率 | {cov['documents_guarded_fraction']:.1%} |",
        "",
        "## 2. 论文级未守文档（按数字量降序，前 15）",
        "",
        "| 文档 | 数字 token | 行数 | 声明的生成器 | 现状 |",
        "|---|---:|---:|---|---|",
    ]
    for r in paper[:15]:
        gen = ", ".join(f"`{g}`" for g in r["generators"]) or "—"
        lines.append(
            f"| `{r['doc']}` | {r['number_tokens']} | {r['lines']} | {gen} | {r['note']} |"
        )
    lines += ["", "## 3. 全部文档分层", "", render_table(report, limit=80), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DOCS / "doc_claim_audit" / "audit.json")
    parser.add_argument("--markdown", type=Path, default=DOCS / "doc_claim_audit.md")
    parser.add_argument("--figures", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, default=DOCS / "doc_claim_audit" / "baseline.json")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args(argv)

    report = audit()
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.write_text(render_markdown(report), encoding="utf-8")
    if args.figures:
        make_figure(report, args.figures)

    counts = report["counts"]
    print(
        f"documents={counts['documents']} pinned={counts['pinned']} "
        f"artifact={counts['artifact']} unguarded={counts['unguarded']} "
        f"(paper-level {counts['unguarded_paper_level']}) "
        f"number coverage={report['coverage']['guarded_number_fraction']:.1%}"
    )

    if args.write_baseline:
        baseline = {
            "schema": SCHEMA,
            "unguarded": [
                r["doc"] for r in report["records"] if r["tier"] == "UNGUARDED"
            ],
            "unguarded_paper_level": [
                r["doc"]
                for r in report["records"]
                if r["tier"] == "UNGUARDED" and r["paper_level"]
            ],
        }
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(
            json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"baseline written: {args.baseline}")

    if args.check:
        baseline = (
            json.loads(args.baseline.read_text(encoding="utf-8"))
            if args.baseline.exists()
            else {}
        )
        failures = check(report, baseline)
        for failure in failures:
            print(f"FAIL: {failure}")
        if failures:
            return 1
        print("OK: unguarded set did not grow")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())