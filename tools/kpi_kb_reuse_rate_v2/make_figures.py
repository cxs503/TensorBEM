#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""由 measure_*.json / crosscheck_sql.json 出图（零手抄：所有数字来自 JSON）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("runs/kpi_kb_reuse_rate_v2")


def main() -> int:
    merged = json.loads((OUT / "measure_alias_merged.json").read_text())
    split = json.loads((OUT / "measure_alias_split.json").read_text())
    cc = json.loads((OUT / "crosscheck_sql.json").read_text())
    g = merged["gates"]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6))
    ax = axes[0][0]
    names = ["L0_text_only", "L1_any", "L2_subsequent", "L3_evidenced", "L4_direct_text"]
    vals = [g[n]["rate"] * 100 for n in names]
    cols = ["#b0b0b0", "#4c72b0", "#55a868", "#c44e52", "#8172b2"]
    ax.bar(range(len(names)), vals, color=cols)
    for i, (n, v) in enumerate(zip(names, vals)):
        ax.text(i, v + max(vals) * 0.02, f"{g[n]['count']} KO\n{v:.2f}%", ha="center", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.split("_")[0] for n in names])
    ax.set_ylabel("reuse_rate (%)")
    ax.set_title("(A) 口径阶梯：v1(L0)=0 是度量盲区，不是无复用")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[0][1]
    labels = ["kb only\n(874)", "unified\n(1228)", "naive alias-split\n(1860)"]
    l1 = [g["L1_any"]["count"] / merged["run"]["n_kb"] * 100,
          g["L1_any"]["rate"] * 100,
          split["gates"]["L1_any"]["rate"] * 100]
    l3 = [g["L3_evidenced"]["count"] / merged["run"]["n_kb"] * 100,
          g["L3_evidenced"]["rate"] * 100,
          split["gates"]["L3_evidenced"]["rate"] * 100]
    x = range(3)
    ax.bar([i - 0.18 for i in x], l1, width=0.36, label="L1 any", color="#4c72b0")
    ax.bar([i + 0.18 for i in x], l3, width=0.36, label="L3 evidenced", color="#c44e52")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("reuse_rate (%)")
    ax.set_title("(B) 分母口径敏感性：别名折叠与否差 ~1.5×")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    ax = axes[1][0]
    det = merged["cited_evidenced_detail"]
    top = sorted(det.items(), key=lambda kv: -len(kv[1]))[:10]
    ax.barh(range(len(top)), [len(v) for _, v in top], color="#55a868")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([k.replace("kb:", "") for k, _ in top], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("引用该 KO 的 mission 数（去重后）")
    ax.set_title("(C) L3 分子被少数 lesson 卡的大面积词面耦合主导")
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1][1]
    ax.axis("off")
    ca = merged["counter_audit"]
    rows = [
        ["metric", "value"],
        ["KO 宇宙（规范）", merged["universe"]["canonical_total"]],
        ["折叠别名对", merged["universe"]["alias_pairs_collapsed"]],
        ["台账注入行", merged["citation_events"]["raw_rows"]],
        ["去重 (KO,consumer) 对", ca["dedup_pairs"]],
        ["逐行求和 citation_count", ca["naive_row_sum"]],
        ["计数虚增倍数", f"{ca['inflation_factor']:.4f}"],
        ["最坏一对 count/rows", f"{ca['worst_pair']['max_count']}/{ca['worst_pair']['rows']}"],
        ["SQL 独立复算 L1", cc["L1_any"]],
        ["SQL 独立复算 L3", cc["L3_evidenced"]],
        ["Python L1 / L3", f"{g['L1_any']['count']} / {g['L3_evidenced']['count']}"],
        ["未解析注入行", merged["unresolved_injections"]],
    ]
    t = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="left")
    t.scale(1, 1.35)
    ax.set_title("(D) 计数与双路线对账")

    fig.suptitle("KPI-KB-REUSE-RATE v2：口径缺陷定位与复用率区间（数据: tensorplatform@%s UTC）"
                 % merged["run"]["fetched_at_utc"])
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png = OUT / "kpi_kb_reuse_rate_v2.png"
    fig.savefig(png, dpi=130)
    print("wrote", png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
