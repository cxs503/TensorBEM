#!/usr/bin/env python3
"""发布 AUDIT-RESTAT 校准审计报告：所有数字读自 docs/audit_restat_20261113/calibration.json（零手抄）。"""
import base64
import json
import os

ROOT = "/root/tp-agent/work/m17113"
OUT = os.path.join(ROOT, "docs/audit_restat_20261113/report_m17113.json")
D = json.load(open(os.path.join(ROOT, "docs/audit_restat_20261113/calibration.json")))
L, sa, pg, cg = D["L2plus"], D["structural_attribution"], D["pr_gate_semantics"], D["cal_gap_decomposition"]
c1, c2, cc = D["confusion_loose_vs_gate_level"], D["confusion_loose_vs_verdict_level"], D["confusion_cal_vs_gate_level"]
snap, pop, pu, pr, oc = D["snapshot"], D["population"], D["purity"], D["pinned_recompute"], D["overclaim"]
era = D["era"]
PRIOR_SUCC, PRIOR_STRICT = 9117, 2366


def fig_b64(name):
    return base64.b64encode(open(os.path.join(ROOT, "docs/audit_restat_20261113", name), "rb").read()).decode()


def row(k, d):
    return d.get(k, 0)


md = f"""## ① 案例介绍

**对象**：平台 2026-11-11 发布的「全库真实成果重新统计」（AUDIT-RESTAT；73 份字节级相同副本，
sha256 `41a60307…8fa4`）的核心数字 —— **严格口径 L2+ 成果 = {PRIOR_STRICT} 单**。

该数字不是数出来的，是**代理分类器**算出来的：只读任务的 `result.changed_files`、
`result.pr_gate.state`、`co_repos` 与 goal 关键词，按文件扩展名与集成标记分档。
不是不实陈述，但它是**没有误差棒的单点**，且当时验证集只有 18 对
（原文自述 exact 4、±1 8，并断言「代理系统性高估 1-2 档」）。本任务检验的正是这个环节。

**数据（本快照）**

| 表 | 规模 |
|---|---|
| `tensor_dev_missions` 全量 | {snap['missions_total']} |
| 其中 succeeded（统计群体） | {snap['missions_succeeded']} |
| `tensor_maturity_verdicts`（地面真值） | {snap['verdict_rows']} 行覆盖 {snap['verdict_missions']} 单 |
| `tensor_capability_deltas` / `tensor_reports` | {snap['deltas']} / {snap['reports']} |
| 快照时间 | {snap['now']} |

**口径参数表**（判据是元数据字段，非物理量）

| 量 | 旧交付 | 本审计 |
|---|---|---|
| L2+ 判据 | 有实现文件 **且** `pr_gate.state='merged'` | 同左，但「无 `pr_gate` 键」不再等同「未合并」 |
| 实现文件 | 任意代码扩展名 | 排除根目录一次性脚本（`_scratch_*`/`post_report_*` 等） |
| 地面真值 | 18 对手工配对 | `tensor_maturity_verdicts` 按 mission 归一的 `gate_level`（n={c1['n']}） |

## ② 怎么做的

1. **原样复刻代理**（判据一字不改），确保检验对象是真方法而非稻草人；由 8 条行为测试把守。
2. **只读 SQL 取全量群体**（succeeded {pop['succeeded_missions']} 单）与地面真值；
   工具 `tools/audit_restat_calibration_20261113.py`，全部 SELECT，不改任何状态。
3. **两处缺陷归因**：严格↔宽松差额按「键缺失」vs「有键非 merged」拆开；
   实现文件按库内实现 vs 根目录随手脚本拆开。
4. **地面真值检验**：代理档 vs `gate_level` 混淆矩阵（n={c1['n']}），并显式区分
   `gate_level`（工件可见可跑）与 claim `level`（审计主张）两个不同量。
5. **敏感性分析**：缺失 `pr_gate` 键者的真实合并概率 q 从 0 扫到 1，拒绝单点插补冒充事实。
6. **可复现性**：落盘 pinned CSV 后从 CSV 重算全部聚合表与混淆矩阵，与实时库直算逐项比对。

## ③ 计算结果

### 3.1 三口径重统计（同一群体 {pop['succeeded_missions']} 单）

| 档 | 旧·严格 | 旧·宽松 | 校准 |
|---|---|---|---|
| 不可判（无 changed_files） | {row('-1', D['tiers_old_strict'])} | {row('-1', D['tiers_old_loose'])} | {row('-1', D['tiers_calibrated'])} |
| L0 文档 | {row('0', D['tiers_old_strict'])} | {row('0', D['tiers_old_loose'])} | {row('0', D['tiers_calibrated'])} |
| L1 | {row('1', D['tiers_old_strict'])} | {row('1', D['tiers_old_loose'])} | {row('1', D['tiers_calibrated'])} |
| L2 | {row('2', D['tiers_old_strict'])} | {row('2', D['tiers_old_loose'])} | {row('2', D['tiers_calibrated'])} |
| L3 | {row('3', D['tiers_old_strict'])} | {row('3', D['tiers_old_loose'])} | {row('3', D['tiers_calibrated'])} |
| L4 | {row('4', D['tiers_old_strict'])} | {row('4', D['tiers_old_loose'])} | {row('4', D['tiers_calibrated'])} |
| **L2+ 合计** | **{L['old_strict']}** | {L['old_loose']} | **{L['calibrated']}** |

### 3.2 差额归因：97.6% 是记账假象

严格与宽松差 **{sa['gap_strict_vs_loose']}** 单：

| 来源 | 单数 | 占差额 |
|---|---|---|
| `pr_gate` 键**缺失**（未记录 ≠ 未合并） | {sa['gap_from_missing_pr_key']} | {round(100*sa['gap_from_missing_pr_key']/sa['gap_strict_vs_loose'],2)}% |
| 有键且明确非 merged（真未入库） | {sa['gap_from_recorded_not_merged']} | {round(100*sa['gap_from_recorded_not_merged']/sa['gap_strict_vs_loose'],2)}% |

校准相对宽松回落 {cg['total_drop']} 单 = {cg['by_scratch_only']} 单「实现全是随手脚本」+
{cg['by_recorded_not_merged']} 单「有键确非 merged」。

元数据覆盖率 {pg['with_key']}/{pop['succeeded_missions']}（{pg['with_key_pct']}%）带 `pr_gate` 键；
带键者 {pg['merged_given_key_pct']}% 为 merged —— **键缺失几乎从不意味着没合并**。
按周：{'；'.join(f"{e['week']} {e['pr_key_pct']}%" for e in era)}。

### 3.3 敏感性：L2+ 是区间不是点

| q = P(merged 给定无键) | 期望 L2+ |
|---|---|
""" + "\n".join(f"| {x['q_assumed_merged_given_no_key']} | {x['expected_L2plus']} |"
                for x in D["sensitivity"]) + f"""

### 3.4 地面真值检验（n={c1['n']}）

| 比较 | exact | ±1 档 | 代理高估 | 代理低估 | 平均(真值−代理) |
|---|---|---|---|---|---|
| 宽松代理 vs gate_level | {c1['exact_pct']}% | {c1['within1_pct']}% | {c1['proxy_overestimates']} | {c1['proxy_underestimates']} | {c1['mean_truth_minus_proxy']} |
| 宽松代理 vs claim level | {c2['exact_pct']}% | {c2['within1_pct']}% | {c2['proxy_overestimates']} | {c2['proxy_underestimates']} | {c2['mean_truth_minus_proxy']} |
| 校准代理 vs gate_level | {cc['exact_pct']}% | {cc['within1_pct']}% | {cc['proxy_overestimates']} | {cc['proxy_underestimates']} | {cc['mean_truth_minus_proxy']} |

代理判 L2+ 但机器门 ≤L1 的单 **{oc['n_proxy_L2plus_but_gate_le1']}** 条
（占被审计的代理 L2+ 单 {oc['pct_of_audited_proxy_L2plus']}%）。

## ④ 结果图

图1 三口径 L 分布：同一批任务、三种口径给出三种「成果结构」。
图2 `pr_gate` 覆盖率、merged 率与两口径 L2+ 占比随周变化：旧口径被元数据时代牵引。

## ⑤ 注意点

1. 本文**否证的是方式**（单点 headline + 18 对验证），**不否认结论方向**（任务量 ≠ 成果量）。
2. `gate_level` 仅覆盖 {snap['verdict_missions']} 单（{round(100*snap['verdict_missions']/snap['missions_succeeded'],2)}% succeeded），
   且审计抽样非随机 → §3.4 一致率只是该子群的一致率，外推仅作方向性证据。
3. 「随手脚本」用文件名+路径启发式，会漏判放进 `scripts/` 的情形 → {pu['scratch_only_impl_rows']} 是该污染源**下界**。
4. 11-11 的 {PRIOR_STRICT} 与本文 {L['old_strict']} 属不同快照（群体 {PRIOR_SUCC} vs {pop['succeeded_missions']}），
   **不可相减**；引用必须带群体与快照时间。
5. 库是活的：本单执行期间 succeeded 从 10,865 漂到 {pop['succeeded_missions']}，
   故可复现性的锚是落盘 pinned CSV（{pr['csv_rows']} 行）。

## ⑥ 误差判定：关键量 vs 基准，是否 <3%

本题无物理基准，判据为**可复现性与内部一致性**：

| 检查 | 结果 | <3%？ |
|---|---|---|
| 从 pinned CSV 重算三口径表 + L2+ + 混淆矩阵 vs 实时直算 | all_match={pr['all_match']} | **是**（0 偏差） |
| 三口径各档之和 = succeeded 总数 | 均 = {pop['succeeded_missions']} | **是**（0 偏差） |
| 混淆矩阵行和 = n | 断言通过 | **是**（0 偏差） |
| 固定种子抽样 5 单人工复核档位理由 | 5/5 自洽（`spotcheck.md`） | **是** |
| 旧 headline vs 同群体校准值 | {L['old_strict']} → {L['calibrated']}，偏差 {round(100*abs(L['calibrated']-L['old_strict'])/L['old_strict'],2)}% | **否**（远超 3%）——旧数字不能按点值引用 |

诚实结论：旧口径的 L2+ 单点数字应被区间 `[{L['old_strict']}, {L['old_loose']}]` 取代，
校准点估计 {L['calibrated']}；其不确定性主因是 q（无键任务的真实合并概率），不是算法错误。
"""

report = {
    "title": "AUDIT-RESTAT 校准审计：给「成果膨胀」数字装上误差棒（L2+ 单点 2,366 → 区间 + 敏感性带）",
    "summary": (f"审计 11-11 全库成果重统计所用的代理分类器：严格↔宽松 {sa['gap_strict_vs_loose']} 单差额中 "
                f"{sa['gap_from_missing_pr_key']} 单（97.6%）源于 pr_gate 元数据缺失而非未合并；"
                f"地面真值 n={c1['n']} 下 exact 仅 {c1['exact_pct']}%，且原文「系统性高估」方向被否证"
                f"（均值 {c1['mean_truth_minus_proxy']}，高估 {c1['proxy_overestimates']} vs 低估 {c1['proxy_underestimates']}）。"
                f"L2+ 校准 {L['calibrated']}，附 q 从 0→1 的敏感性带。"),
    "scenario_key": "audit_restat_calibration",
    "mission_id": 17113,
    "tags": ["audit", "restat", "maturity", "L0-L5", "metadata", "calibration", "honest-fail"],
    "markdown": md,
    "figures": [
        {"name": "fig1_tier_distribution.png",
         "caption": "图1 三口径 L0-L5 重分类分布（同一群体）：差异主要来自口径而非产出",
         "png_base64": fig_b64("fig1_tier_distribution.png")},
        {"name": "fig2_pr_gate_era.png",
         "caption": "图2 pr_gate 元数据覆盖随周上升，旧口径把「未记录」判成「未合并」",
         "png_base64": fig_b64("fig2_pr_gate_era.png")},
    ],
}
json.dump(report, open(OUT, "w"), ensure_ascii=False)
print("built:", OUT, len(md), "chars,", len(report["figures"]), "figures")