"""AUDIT-RESTAT 校准工具的不变量测试。

原则：断言**判据**而非跑通流程。三条主线：
  1. 旧代理分档器被原样复刻（否则本审计在打稻草人）；
  2. 两处校准只改它该改的地方（缺陷1 元数据缺失、缺陷2 一次性脚本），
     不得顺手改动其它档位判据；
  3. 聚合表的守恒律（分档之和=群体量、混淆矩阵行和=n、pinned 重算=直算）。

全部用合成文件清单，不需要数据库；运行：pytest tests/test_restat_calibration.py -q
"""
from __future__ import annotations

import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import audit_restat_calibration_20261113 as cal  # noqa: E402

MERGE = "merged"


# --------------------------------------------------------------- 原样复刻 ----
def test_docs_only_is_L0():
    assert cal.classify(["docs/report.md", "docs/fig.png"], "", "", "")[0] == 0


def test_tests_only_is_L1():
    lvl = cal.classify(["tests/test_x.py", "docs/a.md"], "", "", "")
    assert lvl[0] == 1 and lvl[2] == "tests_or_verify_only"


def test_no_files_is_undecidable():
    assert cal.classify([], "", "", "")[0] == -1


def test_impl_without_merge_is_L1_strict_but_L2_loose():
    """旧口径的缺陷复现：实现存在但无合并证据 -> 严格档压成 L1。"""
    strict, loose, reason, _ = cal.classify(["src/pkg/a.py"], "", "", "")
    assert strict == 1 and loose == 2
    assert reason.startswith("impl_without_merge_evidence")


def test_impl_with_merge_is_L2():
    assert cal.classify(["src/pkg/a.py"], MERGE, "", "")[0] == 2


def test_integration_marker_lifts_to_L3():
    assert cal.classify(["src/api/handlers.py"], MERGE, "", "")[0] == 3


def test_bench_keyword_plus_run_artifact_lifts_to_L4():
    assert cal.classify(["src/pkg/a.py", "artifacts/run1.csv"], MERGE, "",
                        "benchmark vs 解析解")[0] == 4


def test_worktree_prefix_is_normalised():
    """绝对工作树路径必须先归一，否则 co_repos/集成标记判断会被前缀污染。"""
    assert cal.strip_prefix("/root/tp-agent/work/m1234/src/x.py") == "src/x.py"
    assert cal.strip_prefix("src/x.py") == "src/x.py"


# ------------------------------------------------------------- 两处校准 ----
def test_calibration_fixes_metadata_absence():
    """缺陷1：无 pr_gate 键 != 未合并 -> 校准档应升到 L2。"""
    strict, loose, _, _ = cal.classify(["src/pkg/a.py"], "", "", "")
    lvl, reason = cal.classify_calibrated(["src/pkg/a.py"], "", False, "", "")
    assert (strict, loose, lvl) == (1, 2, 2)
    assert reason.startswith("merged_")


def test_calibration_respects_recorded_non_merge():
    """有键且明确非 merged = 真未入库，校准口径**不得**放行。"""
    lvl, reason = cal.classify_calibrated(["src/pkg/a.py"], "open", True, "", "")
    assert lvl == 1 and reason.startswith("impl_without_merge_evidence")


def test_calibration_rejects_scratch_only_impl():
    """缺陷2：实现文件全是根目录随手脚本 -> 不算独立功能实现。"""
    files = ["_scratch_patch.py", "post_report_m123.py", "make_report_fig.py"]
    assert cal.classify(files, MERGE, "", "")[0] == 2          # 旧口径误判为 L2
    lvl, reason = cal.classify_calibrated(files, MERGE, True, "", "")
    assert lvl == 1 and reason == "scratch_only_impl"           # 校准后 L1


def test_calibration_keeps_real_package_impl():
    """库内实现（含目录）不得被纯度过滤误伤——只滤根目录脚本。"""
    files = ["scripts/solve_case.py", "src/pkg/core.py"]
    lib, scratch = cal.split_impl_purity(files)
    assert scratch == [] and len(lib) == 2


def test_purity_does_not_touch_nested_scratch_named_file():
    """同名但位于子目录的脚本不是根目录随手脚本（保守：不误降档）。"""
    lib, scratch = cal.split_impl_purity(["scripts/_scratch_x.py"])
    assert scratch == []


# ------------------------------------------------------------- 不变量 ----
def _rows(levels):
    return [{"l_strict": s, "l_loose": l, "l_cal": c, "has_pr_gate_key": True,
             "pr_state": "", "repo": "R", "v_level": v, "v_gate": g,
             "cal_reason": "", "week": "2026-11-09"} for s, l, c, v, g in levels]


def test_tier_table_conserves_population():
    rows = _rows([(1, 2, 2, 2, 1), (0, 0, 0, None, None), (2, 2, 2, 3, 2)])
    t = cal.tier_table(rows, "l_loose")
    assert sum(t.values()) == len(rows)


@pytest.mark.parametrize("key", ["l_strict", "l_loose", "l_cal"])
def test_confusion_matrix_rows_sum_to_n(key):
    rows = _rows([(1, 2, 2, 2, 1), (2, 2, 2, 3, 2), (0, 0, 0, 0, 0), (1, 1, 1, None, None)])
    c = cal.confusion(rows, key, "v_gate")
    assert sum(sum(v.values()) for v in c["matrix"].values()) == c["n"]
    assert c["n"] == 3          # 只有三行带 v_gate


def test_sensitivity_is_monotone_in_q():
    rows = _rows([(1, 2, 2, None, None), (2, 2, 2, None, None)])
    sens = cal.sensitivity_merge_imputation(rows, qs=(0.0, 0.5, 1.0))
    vals = [x["expected_L2plus"] for x in sens]
    assert vals == sorted(vals), "L2+ 期望值必须随 q 单调不减"
    assert sens[0]["expected_L2plus"] == 1.0   # q=0 退化为旧严格口径


def test_sensitivity_candidates_require_missing_pr_key():
    """候选集判据：只有「宽松 L2+、严格不足、且无 pr_gate 键」的单才被插补。

    有键的单一律不进候选——否则校准会把「有键且明确未合并」也放行（那是另一条缺陷，
    由 test_calibration_respects_recorded_non_merge 把守）。
    """
    keyed = _rows([(1, 2, 2, None, None)])          # has_pr_gate_key=True -> 非候选
    assert cal.sensitivity_merge_imputation(keyed, qs=(1.0,))[0]["candidate_missions"] == 0

    cand = [{"l_strict": 1, "l_loose": 2, "l_cal": 2, "has_pr_gate_key": False,
             "pr_state": "", "repo": "R", "v_level": None, "v_gate": None,
             "cal_reason": "merged_impl+bogus", "week": "2026-11-09"},
            {"l_strict": 2, "l_loose": 2, "l_cal": 2, "has_pr_gate_key": False,
             "pr_state": "", "repo": "R", "v_level": None, "v_gate": None,
             "cal_reason": "merged_impl", "week": "2026-11-09"}]
    sens = cal.sensitivity_merge_imputation(cand, qs=(1.0,))
    # 基线 = 严格 L2+ 单数(1)；候选 = 1 -> q=1 时期望 2.0
    assert sens[0]["candidate_missions"] == 1
    assert sens[0]["expected_L2plus"] == 2.0


def test_pinned_recompute_detects_mismatch(tmp_path):
    """pinned 重算必须能**发现**不一致：故意改一个档位，match 应为 False。"""
    d = tmp_path / cal.OUT_JSON_DIR
    d.mkdir(parents=True)
    rows = _rows([(1, 2, 2, None, None), (2, 2, 2, None, None)])
    tables = {"tiers_old_strict": cal.tier_table(rows, "l_strict"),
              "tiers_old_loose": cal.tier_table(rows, "l_loose"),
              "tiers_calibrated": cal.tier_table(rows, "l_cal"),
              "L2plus": {"old_strict": 1, "old_loose": 2, "calibrated": 2},
              "confusion_loose_vs_gate_level": cal.confusion(rows, "l_loose", "v_gate")}
    cols = ["id", "repo", "week", "l_strict", "l_loose", "l_cal", "reason", "cal_reason",
            "n_files", "pr_state", "has_pr_gate_key", "v_level", "v_gate", "v_auditors"]
    with open(d / "missions_calibrated.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i, r in enumerate(rows):
            w.writerow([i, r["repo"], r["week"], r["l_strict"], r["l_loose"], r["l_cal"],
                        "", "", 1, r["pr_state"], r["has_pr_gate_key"], "", "", ""])
    assert cal.pinned_recompute(str(tmp_path), tables)["all_match"] is True

    # 篡改：把第一行严格档从 L1 改成 L2 -> 必须被检出
    with open(d / "missions_calibrated.csv") as fh:
        data = list(csv.reader(fh))
    data[1][3] = "2"
    with open(d / "missions_calibrated.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(data)
    assert cal.pinned_recompute(str(tmp_path), tables)["all_match"] is False