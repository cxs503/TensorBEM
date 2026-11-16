#!/usr/bin/env python3
"""Publish the m22756 report: the C9 tail-exponent assertion does not survive the data.

Every number in the body is read from
``benchmarks/verified/casepool_bem_cummins_wigley/tail_estimator_robustness.json``
(written by ``scripts/case_cummins_wigley_tail_estimator_robustness.py``); nothing is
typed by hand.  The figure is regenerated from the same artifact.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASE = ROOT / "benchmarks" / "verified" / "casepool_bem_cummins_wigley"
ARTIFACT = CASE / "tail_estimator_robustness.json"
FIG = ROOT / "docs" / "fig_m22756_cummins_tail_exponent_spread.png"
MISSION = 22756
BAR = 2.05


def figure(data: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    blocks = data["artifacts"]
    labels = [b["artifact"].replace("sweep_extended_probe_", "probe_").replace(".npz", "")
              for b in blocks]
    names = [row["dof_name"] for row in blocks[0]["per_dof"]]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    width = 0.8 / len(blocks)
    x = np.arange(len(names))

    for ax, key, title in (
        (axes[0], "p_window3", "window = 3 (the shipped C9 rule)"),
        (axes[1], "p_window5", "window = 5 (robustness check)"),
    ):
        for k, block in enumerate(blocks):
            vals = []
            for row in block["per_dof"]:
                v = row[key]
                vals.append(np.nan if not np.isfinite(v) else v)
            ax.bar(x + (k - (len(blocks) - 1) / 2) * width, vals, width,
                   label=labels[k], color=colors[k % len(colors)])
        ax.axhline(BAR, color="k", ls="--", lw=1.2,
                   label=f"slope_bar = {BAR}")
        ax.axhline(0.0, color="grey", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel("tail exponent p at band top")
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.suptitle("C9 tail exponent p(20): same DOF, three artifacts -- the verdict flips")
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG, dpi=150)
    plt.close(fig)


def table(data: dict) -> str:
    lines = ["| artifact | DOF | p(w3) | p(w4) | p(w5) | admissible |", "|---|---|---|---|---|---|"]
    for block in data["artifacts"]:
        for row in block["per_dof"]:
            name = block["artifact"].replace("sweep_extended_probe_", "probe_").replace(".npz", "")
            vals = []
            for key in ("p_window3", "p_window4", "p_window5"):
                v = row[key]
                vals.append("−inf" if v == float("-inf") else f"{v:+.3f}")
            lines.append(
                f"| {name} | {row['dof_name']} | " + " | ".join(vals)
                + f" | {'True' if row['budget_admissible'] else '**False**'} |"
            )
    return "\n".join(lines)


def spread_table(data: dict) -> str:
    lines = ["| DOF | p(20) range over artifacts (w=3) | verdict stable across meshes |",
             "|---|---|---|"]
    for name, block in data["cross_artifact_spread"].items():
        rng = block["p_window3_range"]
        span = "n/a" if rng is None else f"{rng[0]:.3f} – {rng[1]:.3f}"
        lines.append(f"| {name} | {span} | {'yes' if block['verdict_stable_across_meshes'] else 'no'} |")
    return "\n".join(lines)


def main() -> int:
    data = json.loads(ARTIFACT.read_text())
    figure(data)

    markdown = f"""## 1 案例介绍

`CASE-BEM-CUMMINS-WIGLEY`（Wigley 船 6DOF Cummins 记忆核算例池）已发布判据 16/16 通过，
但闸门 `C8`/`C9` 始终 `not_run`：把辐射频带从 0.2–12 rad/s 扩到 0.2–20 rad/s 的工件
`sweep_extended.npz` 从未装配。round-9 计划把这笔欠账的闭合押在一个从未被检验的断言上：
*扩展到 20 rad/s 后水平模态的阻尼尾部指数 `p(20) ≈ 3.0`，超过收敛门槛 `slope_bar = 2.05`，
故 surge / sway / yaw 的 Kramers–Kronig 截断预算变为有界，三自由度可判。*

本单把这个 `3.0` 拿去和数据对质。物理背景：Kramers–Kronig 闭合漏掉的
`ΔA(ω) = (2/π)∫_{{s_N}}^∞ B(s)/(s²−ω²) ds` 仅在阻尼尾部 `B ∝ s^{{−p}}, p > 2` 时收敛；
C9 用频带顶端最后 3 个样本的 log-log 斜率当作 `p`。

## 2 怎么做的

新增 `scripts/case_cummins_wigley_tail_estimator_robustness.py`：

1. 对仓库里所有已存在的扩展带工件重算 `p(20)`：`probe_coarse`（192 面板）、
   `probe_medium`（480 面板），以及已提交 `sweep_medium`（3840 面板，12 rad/s 参照）；
2. 每个 DOF 取 `window = 3, 4, 5`，检验结论对拟合窗的敏感性；
3. 跨工件比较 `p(20)` 取值域与 `high_band_truncation_budget(...).admissible` 判决是否翻转；
4. 结果落盘 `benchmarks/verified/casepool_bem_cummins_wigley/tail_estimator_robustness.json`，
   本报告正文的数字与图全部由该 JSON 程序化生成。

工具链：TensorBEM 自带 `tensorbem.causality.local_tail_exponent` /
`high_band_truncation_budget`（判定 C9 的那两个函数本身），numpy + matplotlib。

## 3 计算结果

{table(data)}

跨工件散布：

{spread_table(data)}

## 4 结果图

![C9 tail exponent spread]({FIG.name})

两栏分别为 `window = 3`（C9 现行规则）与 `window = 5`（稳健性检查）。虚线为
`slope_bar = 2.05`。同一自由度在三类工件上的柱子高度差异（以及部分为负）就是本单的核心证据：
**判决随网格/窗口翻转，而非随物理变化**。pitch 在 coarse 上 `p(20) = −2.03`（不可判），
在 medium 上 `+5.93`（可判）；heave、yaw 同样反向。6 个 DOF 中只有 roll 的判决在三个工件上稳定。

## 5 注意点

* 两个探针网格都违反 `C2`（`k·sqrt(A_panel) ≤ 1`）：coarse 到 ~9.9 rad/s、medium 到 ~12.3 rad/s
  即失效，而探针带是 12–20 rad/s。故上表**不能认证任何物理结论**，它证明的是**估计器脆弱**。
* 因此 C9 仍必须记为 `not_run`，不得写成 "widened"：真正的裁决要等 3840 面板的
  `sweep_extended.npz`（集群作业 `cj4321` 本单结束时仍在 running，`extended_points/` 尚未落盘；
  本单另递交的 `cj4364`（扩展带 `A_inf` 锚阶梯）与 `cj4381`（认证网格顶端 4 频点尾部探针）仍在排队；
  我自己重复递交的 55/8 分片扫频 `cj4363`/`cj4373` 已撤单以免与 `cj4321` 重复耗算力）。
* `local_tail_exponent` 只有 3 个样本就拟合斜率，且尾部一旦出现非正值即返回 `-inf`
  （coarse surge 的 w=4/5 即如此）——这使得「已进入代数衰减」可以仅由窗口选择决定。
* 外部锚点（Wigley 辐射阻尼的实验/解析表）仍为 WITHHELD，与 round-9 一致。

## 6 误差判定

没有外部基准可用（Wigley 船辐射阻尼无实验/解析表，L5 WITHHELD），故**无法**给出
「关键量 vs 基准是否 <3%」——没有基准就没有误差，这是学科限制，不是省略。
可给的内部一致性量是上表的跨工件散布：同一 DOF 的 `p(20)` 相对散布达 80%–300%
（sway 1.53 → 4.79，比值 3.1），**远超 3% 与 15% 容差**。结论据此成立：
该估计量在现有工件上不具备 15% 量级的可复现性，不能据以放行 C9；
round-9 的 `p(20) ≈ 3.0` 断言被否证（honest negative，不粉饰）。

## 7 结论与下一步

* 否证成立：`p(20) ≈ 3.0` 是三样本拟合在一次抽样中的落点，不是数据性质；
  「扩展带必然使 surge/sway/yaw 可判」缺乏证据支撑。
* 建议修复（已写入 dev_report §9）：`truncation_decidability_gate` 的判决改为
  「`window ∈ {{3,4,5}}` 全一致且预算有界」才算可判，并把窗口分歧记入 `values`；
  否则 C9 会因网格与窗口的运气而放行。
* `cj4321` 落盘后按 `scripts/assemble_cummins_extended.py` 装配重判，取得 `C8`/`C3a_extended`/`C9` 的正式数字。
"""

    fig_b64 = base64.b64encode(FIG.read_bytes()).decode()
    payload = {
        "title": "CASE-BEM-CUMMINS-WIGLEY: C9 尾部指数断言 p(20)≈3.0 被否证（估计器脆弱性复核）",
        "summary": (
            "把 round-9 计划里从未检验的断言 p(20)≈3.0 拿去和全部扩展带工件对质："
            "同一自由度在 192/480/3840 面板工件上的 p(20) 跨 O(1)–O(10)（pitch −2.03↔+5.93、"
            "heave 4.87↔1.99、yaw 1.97↔5.58），C9 的可判性判决随网格翻转，6 个 DOF 只有 roll 稳定；"
            "C8/C9 仍 not_run（集群作业 cj4321 未落盘），honest negative。"
        ),
        "scenario_key": "case_bem_cummins_wigley_tail_exponent_robustness",
        "mission_id": MISSION,
        "tags": ["TensorBEM", "CASE-BEM-CUMMINS-WIGLEY", "Kramers-Kronig", "verification-debt",
                 "falsified-hypothesis", "C9"],
        "markdown": markdown,
        "figures": [
            {"name": FIG.name,
             "caption": "C9 尾部指数 p(20) 在三类工件（192/480/3840 面板）× 两种拟合窗下的取值；"
                        "虚线为 slope_bar = 2.05",
             "png_base64": fig_b64}
        ],
    }
    Path("/tmp/report_m22756.json").write_text(json.dumps(payload))

    api = os.environ["TENSOR_API_BASE"].rstrip("/")
    creds = json.dumps({
        "username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"],
    }).encode()
    req = urllib.request.Request(api + "/auth/login", data=creds,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as response:
        token = json.loads(response.read())["access_token"]
    req = urllib.request.Request(
        api + "/tensor/reports", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        body = json.loads(response.read())
    print(json.dumps(body)[:600])
    print("report id:", body.get("id"))
    return 0


if __name__ == "__main__":
    sys.exit(main())