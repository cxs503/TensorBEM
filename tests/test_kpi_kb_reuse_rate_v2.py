"""KPI-KB-REUSE-RATE v2 单测：以人工算例（ground truth）为参照，误差须为 0。"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "kpi_kb_reuse_rate_v2"))

import measure as M  # noqa: E402


# ---------------------------------------------------------------- 键空间解析
def test_canonical_kb_numeric_id():
    assert M.canonical_ko_id("kb:48", {48: "cap:TensorLBM:x"}, set(), set()) == "kb:cap:TensorLBM:x"
    assert M.canonical_ko_id("kb:999", {48: "cap:TensorLBM:x"}, set(), set()) is None


def test_canonical_negk_both_spellings():
    ids = {"negk:loop:9", "m1-nk-001"}
    assert M.canonical_ko_id("negk:loop:9", {}, set(), ids) == "negk:loop:9"
    assert M.canonical_ko_id("negk:m1-nk-001", {}, set(), ids) == "negk:m1-nk-001"


def test_canonical_unknown_namespace_is_none():
    # anchor: / verdict: 属外部命名空间，不许静默当成 KO
    assert M.canonical_ko_id("anchor:GAP-LBM", {}, set(), set()) is None
    assert M.canonical_ko_id("", {}, set(), set()) is None


def test_alias_and_dealias_collapse_mirror():
    aliases = M.alias_map({"lesson:m9", "cap:x"}, [("negk:loop:9", 9), ("m1-nk-001", 1)])
    assert aliases == {"negk:loop:9": "kb:lesson:m9"}
    u = M.dealiased_universe({"lesson:m9", "cap:x"}, {"negk:loop:9", "m1-nk-001"}, aliases)
    assert u["negk:loop:9"] == "kb:lesson:m9"
    assert u["negk:m1-nk-001"] == "negk:m1-nk-001"
    assert len(set(u.values())) == 3          # 3 个规范知识（不是 4 个成员）


# ---------------------------------------------------------------- 去重
def test_dedup_takes_max_not_sum():
    evs = [M.Event("kb:a", 7, c, "2026-01-01 00:00:00", "r", "loop") for c in (1, 2, 3, 4, 5)]
    agg = M.dedup_events(evs)
    assert len(agg) == 1
    v = agg[("kb:a", 7)]
    assert v["n_rows"] == 5 and v["count"] == 5        # 轮次累计被识别
    assert sum(v["count"] for v in agg.values()) == 5  # 逐行求和会得 15
    assert sum(1 for v in agg.values() if v["counted"]) == 1


def test_dedup_earliest_citation_used_for_subsequence():
    evs = [M.Event("kb:a", 7, 1, "2026-01-02 00:00:00", "r2", "loop"),
           M.Event("kb:a", 7, 1, "2026-01-01 00:00:00", "r1", "loop")]
    assert M.dedup_events(evs)[("kb:a", 7)]["cited_at"] == "2026-01-01 00:00:00"


# ---------------------------------------------------------------- 度量（人工算例）
def _fixture():
    """人工算例：10 个规范 KO，引用情形逐条可手算。

    ko0..ko9；建档时间 2026-01-01。
    引用事件（全部 cited_at 晚于建档，除 ko3）：
      ko0 -> mission 100（有报告）   => L1 L2 L3
      ko1 -> mission 101（有报告）   => L1 L2 L3
      ko2 -> mission 102（无报告）   => L1 L2
      ko3 -> mission 103（有报告）但 cited_at 早于建档 => 只有 L1
      ko4 -> mission 100 ×5 行（轮次累计，去重后仍 1 个引用主体） => L1 L2 L3
    其余 ko5..ko9 无引用。
    手算：L1=5/10 L2=4/10 L3=3/10
    """
    universe = {f"kb:ko{i}": f"kb:ko{i}" for i in range(10)}
    created = {f"kb:ko{i}": "2026-01-01 00:00:00" for i in range(10)}
    t = "2026-01-02 00:00:00"
    evs = [M.Event("kb:ko0", 100, 1, t, "ref", "loop"),
           M.Event("kb:ko1", 101, 1, t, "ref", "loop"),
           M.Event("kb:ko2", 102, 1, t, "ref", "loop"),
           M.Event("kb:ko3", 103, 1, "2025-12-31 00:00:00", "ref", "loop")]
    evs += [M.Event("kb:ko4", 100, c, t, "ref", "loop") for c in (1, 2, 3, 4, 5)]
    return universe, created, evs, {100, 101, 103}


def test_measure_matches_hand_computed_ground_truth():
    universe, created, evs, reports = _fixture()
    res = M.measure(universe, created, evs, reports)
    truth = {"L1_any": 5, "L2_subsequent": 4, "L3_evidenced": 3}
    for gate, exp in truth.items():
        assert res["gates"][gate]["count"] == exp, gate
        assert abs(res["gates"][gate]["rate"] - exp / 10) < 1e-12, gate
    # 分母不因重复注入行而变
    assert res["universe"]["canonical_total"] == 10


def test_measure_alias_off_changes_denominator_and_numerator():
    universe = {"kb:lesson:m9": "kb:lesson:m9", "negk:loop:9": "kb:lesson:m9", "kb:other": "kb:other"}
    created = {k: "2026-01-01 00:00:00" for k in universe}
    evs = [M.Event("negk:loop:9", 5, 1, "2026-01-02 00:00:00", "r", "loop")]
    merged = M.measure(universe, created, evs, {5})
    split = M.measure(universe, created, evs, {5}, merges_off=True)
    assert merged["universe"]["canonical_total"] == 2 and split["universe"]["canonical_total"] == 3
    assert merged["gates"]["L3_evidenced"]["count"] == 1
    assert split["gates"]["L3_evidenced"]["count"] == 1     # 同一事件只挂 negk 一侧
    assert merged["gates"]["L3_evidenced"]["rate"] == pytest.approx(0.5)
    assert split["gates"]["L3_evidenced"]["rate"] == pytest.approx(1 / 3)


def test_measure_ignores_keys_outside_universe():
    universe = {"kb:a": "kb:a"}
    created = {"kb:a": "2026-01-01 00:00:00"}
    evs = [M.Event("?anchor:X", 1, 1, "2026-01-02 00:00:00", "r", "loop")]
    res = M.measure(universe, created, evs, {1})
    assert res["gates"]["L1_any"]["count"] == 0
    assert res["universe"]["canonical_total"] == 1


def test_measure_empty_universe_rate_is_none():
    res = M.measure({}, {}, [], set())
    assert res["gates"]["L3_evidenced"]["rate"] is None


# ---------------------------------------------------------------- 文本口径（v1 复现）
def test_text_scan_respects_token_boundaries():
    keys = {"KB-CLUSTER-FIRST", "KB-CLUSTER-FIRST-2"}
    assert M.text_scan(["见 KB-CLUSTER-FIRST 条款"], keys) == {"KB-CLUSTER-FIRST"}
    assert M.text_scan(["xKB-CLUSTER-FIRST"], keys) == set()
    assert M.text_scan(["KB-CLUSTER-FIRST-2"], keys) == {"KB-CLUSTER-FIRST-2"}


def test_coupling_is_not_promoted_to_citation():
    assert M.merges_citation(0, 5, 3) == "coupling"
    assert M.merges_citation(1, 0, 3) == "direct"
