#!/usr/bin/env python3
"""Publish the m17339 board-audit report (six-element, figure embedded as base64)."""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def api(method, path, payload=None, token=None):
    req = urllib.request.Request(BASE + path, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, data=json.dumps(payload).encode() if payload else None, timeout=120) as r:
        return json.loads(r.read() or b"{}")


MARKDOWN = """## ⑥ 误差判定（先答问题）
- 主判据（`tensor_benchmark_cases.id=43`，<5 %）：800 面板 BEM 远场 vs 闭式 **2.028 / 1.068 / 1.317 %** → **全部 <3 %，是**。
- 长波律 @ka=0.02：**0.0116824 %** → 是；斜率 2.9964（3±0.05）→ 是；近-远场闭式互证 2.84e-15 → 是。
- 唯一未过：800 面板**近场** ka=0.5 **6.08 %**（>5 %，如实披露；1776 面板 2.12 % 合格，属离散误差）；同网格 ka=1.0/2.0 为 4.12 %/4.85 % 合格。

## ① 案例介绍
锚点 `CASE-BEM-CYL-DRIFT-ASYMP`：深水直立圆柱（a=1 m，A=1 m，ρ=1025 kg/m³，g=9.81 m/s²，固定不浮动，kd=5 截断）二阶时间平均漂移力；ref = MacCamy-Fuchs (1954) 绕射势 + Maruo (1960) 远场动量定理闭式，长波极限 `F_x=(5π²/16)ρgA²a(ka)³ = 3.0842514 ρgA²a(ka)³`。被审对象 = mission **m17296**（返修轮1，anchor 的解析半交付）；同案前史 m17204（真实 BEM 算例）与 m17263（首轮审计 L4，被 gate reopened 至 L1：指针不可机器核）。

## ② 怎么做的（交差核验，非重做）
1. 机器裁判先行：读 `tensor_benchmark_cases` row 43（status=passed）逐条取判据，不看报告措辞。
2. 三件套取证：m17296 工作树 `artifacts/cylinder_drift_asymptotic/{study.json,benchmark.json,fig_*.png}` 存在性/大小/sha256；`benchmark.json` ref-vs-got；`repro_cmd` 实跑。
3. 授权复现：①`repro_cmd` 拷到 /tmp 重跑（3.39 s）→ 产物逐字节/逐 sha256 相同；②**只用 main 树代码**独立复算最便宜 BEM 档（208 面板 @ka=0.5，8.9 s）：`drift_far_field_3d_study.run_case` → Kochin → Maruo，与交付值 |Δ|=1.7e-16。
4. 判级 L0–L5 按功能实质；工件固化进本审计分支（`docs/board_audit/m17339/snapshots/`，sha256 在案）。

## ③ 计算结果（ref vs got，程序化取值）

| 量 | ref | got | 误差 | 判据 | 结论 |
|---|---|---|---|---|---|
| BEM 远场 800 面板 ka=0.5 | 2875.6570 N | 2817.3272 N | 2.0284 % | <5 %/<3 % | ✅/✅ |
| 同上 ka=1.0 | 6685.9935 N | 6614.5977 N | 1.0678 % | <5 %/<3 % | ✅/✅ |
| 同上 ka=2.0 | 6306.3981 N | 6223.3267 N | 1.3173 % | <5 %/<3 % | ✅/✅ |
| 长波律 ka=0.02 | 0.24810335 N | 0.24807436 N | 0.0116824 % | <5 % | ✅ |
| 斜率 ka≤0.1 | 3.0 | 2.99643 | 0.12 % | 3±0.05 | ✅ |
| 近-远场互证（8 ka） | — | — | 2.84e-15 | <1e-6 | ✅ |
| 幅值² 标度 | 4.0 | 4.000000 | 0 | 1e-9 | ✅ |
| BEM 阶梯远场 208/456/800/1776 | 闭式 | 7.396/3.179/2.028/1.406 % | 单调，阶≈2.15 | 收敛 | ✅ |
| BEM 阶梯近场 同上 | 闭式 | 30.202/13.350/**6.082**/2.123 % | 单调 | <5 % | ⚠️ 800 档 ka=0.5 超门 |

## ④ 结果图
左：208/456/800/1776 面板阶梯（远/近场误差 vs 面板数，5 % 门与 3 % 生态线；空心点=本审计用 main 树代码独立复算的 208 档）。右：`F/(ρgA²a(ka)³)` vs ka 与 5π²/16 平台（±3 %/±5 % 带）。数值全部读快照，零手抄。

## ⑤ 注意点 / 局限
- m17296 的 `repro_cmd` **只重放解析半**（0 次 BEM 求解），与主判据（BEM 800 面板）不同源；BEM 半由 m17204 工件（sha256 `ac14e4e5…`）+ 本审计 208 档复算背书。
- `artifacts/` 被 `.gitignore` 忽略、`cc3c087` 只提交 2 脚本、分支未并入 main → 工件原本无处持久；本审计固化快照（含 sha256）并留返修建议：case driver（sha256 `7cd5cfd9…`）与窄化 results 提交到 `benchmarks/verified/cylinder_drift_asymptotic/`。
- 未做：750 s 全案重跑（只复算 208 档）；原文 PDF 页码核验；PNG 像素内容（只核 sha256 与确定性生成）。
- 登记文本与工件有 1.6e-5 四舍五入差（row 43 写 0.011684 %，工件 0.0116824 %）。

## 判级与处置
**L4**（L2 代码/入口/工件真、L3 主求解器验证面调用真、L4 完整算例+逐项误差+可逐位复现）。非 L5：外部锚为解析闭式，无实验/印刷数表锚，`5π²/16` 无逐字页码（WITHHELD，m17263 独立重推 7.7e-9）。Verdict 落表 `tensor_maturity_verdicts.id=536`（scope `m17296 CASE-BEM-CYL-DRIFT-ASYMP`，verdict_by board-audit），maturity_gate 机器复核 gate_level=4（本地 dry-run 一致）。处置：approve with notes。
"""


def main() -> int:
    tok = api("POST", "/auth/login", {"username": USER, "password": PASS})["access_token"]
    fig = HERE / "fig_m17339_benchmark_vs_ref.png"
    payload = {
        "title": "交差审计 m17339: CASE-BEM-CYL-DRIFT-ASYMP (对象 m17296) — 机器裁判逐项复核, verdict L4",
        "summary": ("基准裁判 row43 逐条复核：800 面板 BEM 远场 2.028/1.068/1.317 % <3 %、长波 0.0116824 %、"
                    "斜率 2.9964 均过；唯一未过为 800 面板近场 6.08 %（已披露，1776 档 2.12 %）。"
                    "独立复现：repro 3.39 s 逐字节一致；main 树代码复算 208 档 |Δ|=1.7e-16。判级 L4，工件 sha256 固化入审计分支。"),
        "scenario_key": "tensorbem_cylinder_drift_asymptotic",
        "mission_id": 17339,
        "tags": ["board-audit", "benchmark", "TensorBEM", "CASE-BEM-CYL-DRIFT-ASYMP", "maturity-verdict", "L4"],
        "markdown": MARKDOWN,
        "figures": [{
            "name": "benchmark_vs_ref",
            "caption": "左：BEM 面板阶梯远/近场误差 vs 5 % 门（空心点=本次审计 main 树代码独立复算 208 档）；右：长波律 vs 5π²/16 平台。",
            "png_base64": base64.b64encode(fig.read_bytes()).decode(),
        }],
    }
    out = api("POST", "/tensor/reports", payload, tok)
    print(json.dumps({"report_id": out.get("id"), "status": out.get("status")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())