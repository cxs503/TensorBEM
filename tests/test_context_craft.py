"""Tests for tools/context_craft (task brief packages + layered memory).

Fixtures are synthetic (alpha/beta/gamma), so the tests carry no domain
vocabulary: the toolkit must work on any corpus.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib                                       # noqa: E402

# NB: the package re-exports the `pack` function, which shadows the module of
# the same name on the package object; importlib hands back the module itself.
cc_ab = importlib.import_module("tools.context_craft.ab")      # noqa: E402
cc_mem = importlib.import_module("tools.context_craft.memory")  # noqa: E402
cc_pack = importlib.import_module("tools.context_craft.pack")   # noqa: E402
from tools.context_craft.extract import classify, extract_file  # noqa: E402
from tools.context_craft.model import (                # noqa: E402
    FLOOR_KINDS,
    FactItem,
    MemoryEntry,
    dedupe,
    estimate_tokens,
    sha256_text,
)


def mk(kind: str, text: str, source: str = "alpha.md", lineno: int = 1) -> FactItem:
    return FactItem.build(kind, text, source, lineno)


# ------------------------------------------------------------------ model
def test_estimate_tokens_monotone_and_mixed_script():
    assert estimate_tokens("") == 0
    a = estimate_tokens("alpha beta gamma delta")
    b = estimate_tokens("alpha beta gamma delta epsilon zeta")
    assert b > a
    # CJK is denser per character than latin
    assert estimate_tokens("对齐验收阈值") > estimate_tokens("abcd")


def test_fact_item_hash_is_line_bound():
    i1 = mk("TOLERANCE", "threshold 3% acceptable", "a.md", 7)
    i2 = mk("TOLERANCE", "threshold 3% acceptable", "a.md", 8)
    assert i1.sha256 == i2.sha256 == sha256_text("threshold 3% acceptable")
    assert i1.item_id != i2.item_id  # identity includes provenance


def test_dedupe_exact_and_near_duplicate():
    a = mk("TOLERANCE", "acceptance threshold must stay below 3 percent of the reference")
    b = mk("TOLERANCE", "acceptance threshold must stay below 3 percent of the reference")
    c = mk("TOLERANCE", "acceptance threshold must stay below 3 percent of the  reference ")
    d = mk("NEGATIVE", "an unrelated negative result about a different quantity entirely")
    kept, dropped = dedupe([a, b, c, d])
    assert [k.item_id for k in kept] == [a.item_id, d.item_id]
    assert dropped == 2


# ---------------------------------------------------------------- extract
@pytest.mark.parametrize(
    "text,expected",
    [
        ("acceptance tolerance is 5% of the analytic value", "TOLERANCE"),
        ("this route is NOT_SUPPORTED for the beta case", "NEGATIVE"),
        ("known trap: the two conventions differ by a factor of 2", "TRAP"),
        ("verdict: PASS on the gamma benchmark", "VERDICT"),
        ("pinned at commit abc1234 with sha256 recorded", "PIN"),
        ("the routine assembles the matrix in blocks", "CONTEXT"),
    ],
)
def test_classify_priority(text, expected):
    assert classify(text) == expected


def test_extract_skips_code_fences_and_captures_provenance(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text(
        "# Title\n\n"
        "## Section A\n\n"
        "- acceptance tolerance is 4% of the reference value\n"
        "```\n"
        "tolerance 99% inside a code fence must be ignored\n"
        "```\n"
        "* a negative result: the beta route is not supported here\n",
        encoding="utf-8",
    )
    items = extract_file(str(p), "doc.md")
    texts = [i.text for i in items]
    assert any("4% of the reference" in t for t in texts)
    assert not any("99%" in t for t in texts)
    assert all(i.source == "doc.md" and i.lineno > 0 for i in items)
    assert any(i.kind == "NEGATIVE" for i in items)


# ------------------------------------------------------------------- pack
@pytest.fixture
def corpus() -> list[FactItem]:
    rows = [
        ("TOLERANCE", "alpha route tolerance is 3% against the analytic reference", 1),
        ("NEGATIVE", "the beta route is NOT_SUPPORTED for thin panels, honest FAIL", 2),
        ("TRAP", "known trap: gamma sign convention flips between versions", 3),
        ("VERDICT", "verdict PASS for the gamma benchmark after the fix", 4),
        ("PIN", "beta reference pinned at commit abc1234", 5),
        ("CONTEXT", "the assembler loops over panels in blocks of eight", 6),
        ("CONTEXT", "the assembler loops over panels in blocks of eight", 7),  # dup
        ("CONTEXT", "a long prose paragraph about unrelated bookkeeping details", 8),
    ]
    return [mk(k, t, "alpha.md", n) for k, t, n in rows]


def test_pack_respects_budget_and_floor(corpus):
    pkg = cc_pack.pack(corpus, "alpha tolerance route", budget_tokens=60, profile="verification")
    assert pkg.tokens_used <= 60
    assert pkg.tokens_used > 0
    kinds = pkg.kinds
    for k in FLOOR_KINDS:
        assert kinds.get(k, 0) >= 1, f"retention floor lost {k}"


def test_pack_is_deterministic(corpus):
    a = cc_pack.pack(corpus, "gamma verdict", 80, "audit")
    b = cc_pack.pack(corpus, "gamma verdict", 80, "audit")
    assert a.package_sha256 == b.package_sha256
    assert cc_pack.package_digest(a) == a.package_sha256


def test_pack_drops_duplicates(corpus):
    pkg = cc_pack.pack(corpus, "assembler panels blocks", 200, "generic")
    texts = [i.text for i in pkg.items]
    assert len(texts) == len(set(texts))
    assert pkg.dropped_duplicates >= 1


def test_pack_provenance_is_rehashable(tmp_path, corpus):
    pkg = cc_pack.pack(corpus, "alpha tolerance", 120, "verification")
    doc = tmp_path / "alpha.md"
    doc.write_text("\n" * 10, encoding="utf-8")
    lines = {}
    for it in corpus:
        lines.setdefault(it.lineno, it.text)
    doc.write_text("\n".join(lines.get(n, "") for n in range(1, max(lines) + 1)), encoding="utf-8")
    for prov in pkg.provenance():
        raw = doc.read_text(encoding="utf-8").splitlines()[prov["lineno"] - 1]
        assert sha256_text(raw.strip()) == prov["sha256"]


def test_two_stream_split_reserves_cross_document_slice():
    local = [mk("TOLERANCE", f"local alpha criterion number {n} is 3% of reference", "own.md", n)
             for n in range(1, 12)]
    other = [mk("NEGATIVE", f"prior alpha mission {n} recorded an unsupported route", "prior.md", n)
             for n in range(1, 12)]
    pkg = cc_pack.pack_two_stream(
        local + other, "alpha criterion", 300, "verification",
        local_sources=["own.md"], local_share=0.5,
    )
    sources = {i.source for i in pkg.items}
    assert sources == {"own.md", "prior.md"}, "cross-document slice must survive"
    assert pkg.tokens_used <= 300


def test_pack_two_stream_without_local_still_fills():
    other = [mk("TOLERANCE", f"criterion {n} is 2% of the reference value", "x.md", n)
             for n in range(1, 20)]
    pkg = cc_pack.pack_two_stream(other, "criterion", 150, "verification", local_sources=["missing.md"])
    assert pkg.tokens_used > 100


# ----------------------------------------------------------------- memory
def _entry(kind: str, text: str, layer: str, mission: int, uses: int = 0,
           stable: bool = False, lineno: int = 1) -> MemoryEntry:
    return MemoryEntry(item=mk(kind, text, f"{layer}.md", lineno), layer=layer,
                       use_count=uses, last_used_mission=mission, stable=stable)


def test_compress_forgets_stale_unused_but_keeps_floor(tmp_path):
    entries = [
        _entry("CONTEXT", "long stale prose about old bookkeeping", "L3", mission=0, uses=0),
        _entry("TOLERANCE", "acceptance threshold 3% must be respected", "L3", mission=0, uses=0),
    ]
    # budget forces eviction of everything but the floor kind
    layers = (cc_mem.LayerConfig("L3", budget_tokens=12, half_life=5.0, min_age_for_forget=1),)
    res = cc_mem.compress(entries, mission_index=20, out_dir=str(tmp_path), layers=layers, force=True)
    kinds = [e.item.kind for e in res.kept["L3"]]
    assert "TOLERANCE" in kinds, "floor kind must never be forgotten"
    assert res.forgotten and res.forgotten[0]["kind"] == "CONTEXT"
    assert os.path.exists(res.log_path)


def test_compress_protects_stable_head(tmp_path):
    entries = [_entry("CONTEXT", "bootstrap head block that must stay", "L1", 0, stable=True)]
    layers = (cc_mem.LayerConfig("L1", budget_tokens=1, half_life=1e9, min_age_for_forget=10**6,
                                 stable_head=True, allow_forget=False),)
    res = cc_mem.compress(entries, 50, str(tmp_path), layers=layers, force=True)
    assert len(res.kept["L1"]) == 1
    assert res.overflow_tokens.get("L1", 0) > 0  # over budget is reported, not hidden


def test_compress_keeps_recently_used_entry(tmp_path):
    entries = [_entry("CONTEXT", "recently reused note about the alpha route", "L2", 10, uses=3),
               _entry("CONTEXT", "old unused note about the beta route", "L2", 0, uses=0)]
    layers = (cc_mem.LayerConfig("L2", budget_tokens=15, half_life=5.0, min_age_for_forget=2),)
    res = cc_mem.compress(entries, 12, str(tmp_path), layers=layers, force=True)
    texts = [e.item.text for e in res.kept["L2"]]
    assert any("alpha" in t for t in texts)


def test_forget_log_is_append_only_and_rehashable(tmp_path):
    e1 = _entry("CONTEXT", "first stale paragraph to be archived away", "L3", 0)
    layers = (cc_mem.LayerConfig("L3", budget_tokens=1, half_life=4.0, min_age_for_forget=1),)
    r1 = cc_mem.compress([e1], 10, str(tmp_path), layers=layers, force=True)
    log = r1.log_path
    n1 = len(open(log, encoding="utf-8").read().splitlines())
    e2 = _entry("CONTEXT", "second stale paragraph to be archived away", "L3", 0, lineno=2)
    r2 = cc_mem.compress([e2], 11, str(tmp_path), layers=layers, force=True)
    n2 = len(open(log, encoding="utf-8").read().splitlines())
    assert n2 == n1 + 1, "forget log must only grow"
    chk = cc_mem.verify_forget_log(log, str(tmp_path / "archive"))
    assert chk["bad"] == 0 and chk["ok"] == n2


def test_cadence_trigger():
    entries = [_entry("CONTEXT", "x" * 200, "L2", 0) for _ in range(1)]
    assert cc_mem.should_compress(entries, mission_index=10, last_compress_mission=2)
    assert not cc_mem.should_compress([], mission_index=3, last_compress_mission=2)


# --------------------------------------------------------------------- ab
def test_ab_task_build_holds_out_caveat_sections(tmp_path):
    p = tmp_path / "report.md"
    p.write_text(
        "# Task title\n\n## 1. 案例介绍\n\nthe alpha panel study uses a fixed reference\n\n"
        "## 2. 方法\n\nwe assemble the operator and solve once\n\n"
        "## 5. 注意点\n\nknown trap: gamma convention differs between routes\n"
        "one must not reuse the beta operator here\n\n"
        "## 6. 误差判定\n\nacceptance tolerance is 3% of the reference value\nthe second quantity stays below 5%\n",
        encoding="utf-8",
    )
    task = cc_ab.build_task(str(p), "report.md")
    assert task is not None
    assert "alpha panel study" in task.query
    assert "gamma convention" not in task.query, "target section must not leak into the query"
    assert task.target_ids and task.target_tolerance_ids


def test_ab_run_end_to_end(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    for k in range(3):
        (docs / f"dev_report_x{k}.md").write_text(
            f"# Report {k}\n\n## 1. 案例介绍\n\nalpha case {k} uses the standard reference\n\n"
            "## 5. 注意点\n\ntrap: the beta convention flips sign in case {0}\n"
            "not supported for thin panels\n\n"
            "## 6. 误差判定\n\nacceptance tolerance is 3% of the reference\n"
            "the residual stays below 5% of the reference\n".format(k),
            encoding="utf-8",
        )
    res = cc_ab.run(str(tmp_path), ["docs/dev_report_*.md"], [80, 160], limit=3,
                    out_path=str(tmp_path / "ab.json"))
    assert res["tasks"] == 3
    for budget in ("80", "160"):
        assert set(res["per_budget"][budget]["summary"]) == set(cc_ab.ARMS)
        for arm in cc_ab.ARMS:
            assert res["per_budget"][budget]["summary"][arm]["recall"]["n"] == 3
    md = cc_ab.to_markdown(res)
    assert "budget 80 tokens" in md and "brief_local" in md


# -------------------------------------------------------------------- cli
def test_cli_pack_and_verify(tmp_path, corpus):
    from tools.context_craft.cli import main

    root = tmp_path / "corpus"
    root.mkdir()
    by_line: dict[int, str] = {}
    for it in corpus:
        by_line[it.lineno] = it.text
    (root / "alpha.md").write_text(
        "\n".join(by_line.get(n, "") for n in range(1, max(by_line) + 1)), encoding="utf-8"
    )
    out = str(tmp_path / "brief")
    assert main(["pack", "--root", str(root), "--pattern", "*.md", "--query",
                 "alpha tolerance route", "--budget", "60", "--out", out]) == 0
    data = json.loads(open(out + ".json", encoding="utf-8").read())
    assert data["tokens_used"] <= 60
    assert main(["verify", "--brief", out + ".json", "--root", str(root)]) == 0
    # tamper: verification must fail (zero-hand-copy gate bites)
    (root / "alpha.md").write_text(
        "\n".join("tampered line here instead of the original text"
                  for _ in range(1, max(by_line) + 1)), encoding="utf-8"
    )
    assert main(["verify", "--brief", out + ".json", "--root", str(root)]) == 1


def test_cli_memory_roundtrip(tmp_path):
    from tools.context_craft.cli import main

    mem = tmp_path / "mem.json"
    entries = [_entry("CONTEXT", "stale note to be forgotten later", "L2", 0)]
    cc_mem.save_memory(entries, str(mem))
    out = tmp_path / "memout"
    assert main(["memory", "--memory", str(mem), "--mission-index", "40", "--out-dir", str(out),
                 "--force"]) == 0
    rep = json.loads((out / "compress_report.json").read_text(encoding="utf-8"))
    assert rep["triggered"] is True
    assert (out / "memory.json").exists()