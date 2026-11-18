#!/usr/bin/env python3
"""Publish the mission report for the ka > 2 contradiction audit (m21633).

Reads the audit evidence and the figure from the worktree, renders the
six-element report with every number injected from the artifacts, logs in with
the agent credentials from the environment and posts to /tensor/reports.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EV = ROOT / "benchmarks/verified/casepool_bem_hulme_hemisphere/evidence/np246_contradiction.json"
PNG = ROOT / "docs/case_bem_hulme_np246_contradiction_audit.png"


def gate_row(key: str, gate: dict) -> str:
    m = gate.get("measured")
    if key == "A2_numpy_version_attribution_supported":
        m = f"band={gate['measured_numpy_band']}, reverify={gate['measured_numpy_reverify']}"
    elif key == "A3_healthy_artifact_is_machine_provenanced":
        m = (f"host={gate['host']}, thread_env_present={gate['thread_env_present']}, "
             f"residuals_distinct={gate['residuals_distinct']}")
    elif key == "A1_same_matrix":
        m = f"cond {gate['measured_band']} vs {gate['measured_reverify']} (rel dev {m:.2e})"
    elif key == "A4_failing_cell_records_its_residual":
        m = "no residual key in the cell"
    return f"| `{key}` | {gate['criterion']} | {m} | {'PASS' if gate['pass'] else 'FAIL'} |"


def _env_str(env: dict | None, key: str) -> str:
    """Render an environment field, tolerating a missing (None) environment."""
    if env is None:
        return "未记录"
    return str(env.get(key, "未记录"))


def markdown(ev: dict) -> str:
    rows, g, prov = ev["rows"], ev["gates"], ev["provenance"]
    band_env = prov.get("band_extension") or {}
    tpl = [
        "# CASE-BEM-HULME-HEMISPHERE：ka>2 频带「同一 numpy、同一矩阵、两个答案」——带外结论退回 WITHHELD",
        "",
        "## ① 案例介绍",
        f"算例 `BEM-HULME-HEMISPHERE`（总账锚点，容差 {ev['tolerance']:.0%}）把半径 a=1 m、静水面浮动的半球"
        "放在无限水深 Kelvin Green 函数 BIE 下做辐射/绕射，六量 A11、A33、B11、B33、|Fx|、|Fz| 对 Hulme(1982) "
        "波-free 多极级数打分；声明带 ka≤2 已 PASS（G1 最坏 1.148%）。**本单只看带外 ka="
        f"{ev['ka']:g}、{ev['panels']} 面元这一格**：m20034 记录 honest FAIL（A11 符号翻转 −104%），m21563 归因于 "
        "numpy 2.0.1 线性求解缺陷并宣布 numpy 2.4.6 下六量全部 <3%。",
        "",
        "## ② 怎么做的",
        "只读审计 + 程序化判决（`benchmarks/verified/casepool_bem_hulme_hemisphere/audit_np246_contradiction.py`）："
        "读两份已提交工件，把「结论—证据」链逐环对齐，并把可证命题写成机器判据 A1–A4，数字全部由工件注入。"
        "争议不能靠读文档解决的部分改成受控实验声明 `probe_ka35_env.py`（入口 × BLAS 线程数，逐 arm 记录条件数、"
        "辐射/绕射残差与六量）。",
        "",
        "## ③ 计算结果",
        "",
        "| 量 | Hulme 参考 | band_extension/results.json | 误差 | numpy2_reverification.json | 误差 |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in rows.items():
        tpl.append(f"| `{name}` | {r['ref']:.6f} | {r['band_extension']:+.6f} | "
                   f"{r['err_band_extension']:+.2%} | {r['numpy2_reverification']:+.6f} | "
                   f"{r['err_numpy2_reverification']:+.2%} |")
    tpl += [
        "",
        f"两份工件在同一点相差 **{ev['relative_gap_between_artifacts']:.1%}**（门限的 "
        f"{ev['relative_gap_between_artifacts'] / ev['tolerance']:.0f} 倍），而报告的条件数分别是 "
        f"{ev['condition_number']['band_extension']} 与 {ev['condition_number']['numpy2_reverification']}"
        f"（相对差 {ev['gates']['A1_same_matrix']['measured']:.2e}）——**同一个矩阵**，所以解释不在矩阵，在解。",
        "",
        "| 判据 | 内容 | 实测 | 裁定 |",
        "|---|---|---|---|",
    ]
    tpl += [gate_row(k, g[k]) for k in g]
    tpl += [
        "",
        f"机器裁定：**{ev['verdict']}**。",
        "",
        f"频带状态：`ka≤2` = {ev['band_status']['ka_le_2']}；`ka>2` = {ev['band_status']['ka_gt_2']}。",
        "",
        "## ④ 结果图",
        "左：Hulme 参考与两份工件并排——同一代码、同一 numpy、同一条件数，一组 +0.1618/0.0319、一组 −0.0068/−0.0006。"
        "右：失败格**没有**求解残差字段（灰），健康格记 4.45e-4（绿），E0 门限 1e-2 为虚线——缺的正是这个判别量。",
        "",
        "## ⑤ 注意点",
        f"1. 「numpy 2.0.1 缺陷」在本格没有工件支撑：带失败数值的工件 metadata 写的是 numpy "
        f"{_env_str(band_env, 'numpy')}、host {_env_str(band_env, 'host')}、gen "
        f"{_env_str(band_env, 'generated_utc')}，与「健康」工件同版本 → A2 FAIL。m20910 的跨环境定位"
        "（同矩阵、残差 1.77e-4 vs 4.91e-1）是 ka=2、node330 的另一次运行，不能自动移植到本格。",
        f"2. 健康工件不可审计：host={_env_str(prov.get('numpy2_reverification') or {}, 'host')}、无线程环境、"
        f"辐射与绕射两次求解给同一个残差（identical={prov['numpy2_reverification_residuals_identical']}）、"
        f"`generated_at`={prov['numpy2_reverification_generated_at']}（整点）→ A3 FAIL。",
        "3. E0 有盲区：失败格无残差字段。小残差+符号翻转（解了错的方程）与 m20910 的大残差（没解方程）是两种失效，"
        "单一残差阈值封不住前者 → A4 FAIL。",
        "4. 受控实验已递交集群 **cj3579**（全带重跑 + 4096 面元健康档）与 **cj3585**（线程×入口探针）；"
        f"收尾时仍在排队——{ev['experiment_that_would_settle_it']['absence_reason']}",
        "5. 适用边界：本单只针对 ka>2 的带外证据；ka≤2 的 PASS、算例数值与门限一概未改。",
        "",
        "## ⑥ 误差判定（是否 <3%）",
        f"本单无新的「计算 vs 基准」计算，故只判证据一致性，并显式回答：**无法回答是否 <3%**——两份工件互差 "
        f"{ev['relative_gap_between_artifacts']:.1%}，任选其一宣称 <3% 都是选择性的。",
        "- 带内 ka≤2：最坏 1.148%（既有 G1 工件）→ <3% ✅（未改动）",
        f"- 带外 ka={ev['ka']:g}：band_extension {max(r['err_band_extension'] for r in rows.values()):.1%} "
        f"vs numpy2_reverification {max(r['err_numpy2_reverification'] for r in rows.values()):.1%} → "
        "**不可裁定（WITHHELD）**，等 cj3579/cj3585 载荷。",
        "",
        "honest-FAIL：本单没有把 ka>2 变成 PASS，反而把一条已宣称「已修复」降级为未决——这是本单的主要产出。",
    ]
    return "\n".join(tpl) + "\n"


def main() -> int:
    ev = json.loads(EV.read_text())
    png = base64.b64encode(PNG.read_bytes()).decode()
    report = {
        "title": "BEM-HULME-HEMISPHERE：ka>2 频带证据矛盾审计（同一 numpy/同一矩阵/两个答案）→ WITHHELD",
        "summary": ("Hulme 浮半球带外 ka=3.5 两份已提交工件互斥、却同报 numpy 2.4.6 且同条件数；"
                    "判据 A1–A4 机器裁定 FAIL；ka>2 退回 WITHHELD，受控实验 cj3579/cj3585 已递交未取回。"),
        "scenario_key": "case_bem_hulme_np246_contradiction",
        "mission_id": 21633,
        "tags": ["TensorBEM", "BEM", "CASE-BEM-HULME-HEMISPHERE", "verification",
                 "honest-FAIL", "evidence-audit"],
        "markdown": markdown(ev),
        "figures": [{
            "name": "case_bem_hulme_np246_contradiction.png",
            "caption": ("左：Hulme 参考 vs 两份工件（同 cond=2.78 同矩阵、答案互斥）；"
                        "右：失败格缺求解残差，E0 门限 1e-2 无从判定。"),
            "png_base64": png,
        }],
    }
    base = os.environ["TENSOR_API_BASE"].rstrip("/")

    def post(path: str, payload: dict, token: str | None = None) -> dict:
        req = urllib.request.Request(
            base + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     **({"Authorization": "Bearer " + token} if token else {})})
        return json.loads(urllib.request.urlopen(req, timeout=180).read())

    tok = post("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                               "password": os.environ["TENSOR_AGENT_PASS"]})
    token = tok.get("access_token") or tok.get("token")
    out = post("/tensor/reports", report, token)
    print("posted:", json.dumps({k: out.get(k) for k in
                                 ("id", "report_id", "status", "title", "mission_id")},
                                ensure_ascii=False)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
