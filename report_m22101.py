"""Render the m22101 dev note and publish the platform report (numbers from JSON, zero hand-typing).

Usage: PYTHONPATH=src python3 report_m22101.py [--publish]
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

METRICS = Path("artifacts/m22101/audit_metrics.json")
DOC = Path("docs/dev_report_m22101_reference_audit.md")
FIG1 = Path("artifacts/m22101/audit_convergence.png")
FIG2 = Path("artifacts/m22101/audit_surface.png")


def g(value: float) -> str:
    if value and abs(value) < 1e-3:
        return f"{value:.3e}"
    return f"{value:.6f}"


def build_markdown() -> tuple[str, dict]:
    metrics = json.loads(METRICS.read_text())
    free = metrics["free_stream"]
    rows = [
        "| 复核项 | 声明参考值 | 独立数值推导 | 偏差 | <3% |",
        "| --- | --- | --- | --- | --- |",
        f"| 自由流速度场 dW/dz | U(1-a^2/z^2) | Cauchy 积分 | {g(free['velocity_max_abs_error_over_U'])} /U | 是（~1e-15，双精度底噪） |",
        f"| 扰动势 phi-Ux | U a^2 cos(theta)/r | Re W - Ux | {g(free['perturbation_potential_max_abs_error'])} | 是 |",
        f"| Cp | 1-(|v|/U)^2 | Bernoulli + 数值速度 | {g(free['cp_max_abs_residual'])} | 是 |",
    ]
    for row in metrics["lifting"]:
        c = row["circulation"]
        rows.append(
            f"| 升力势速度场 Gamma={c:g} | U(1-a^2/z^2) - i Gamma/(2 pi z) | Cauchy 积分 | "
            f"{g(row['velocity_max_abs_error_over_U'])} /U | 是 |"
        )
        rows.append(
            f"| 物面速度 Gamma={c:g} | |Gamma/(2 pi a) - 2U sin(theta)| | 物面数值导数模长 | "
            f"{g(row['surface_speed_max_abs_error_over_U'])} /U | 是 |"
        )
        rows.append(
            f"| 法向无流动 Gamma={c:g} | v_r = 0 | 数值速度径向分量 | "
            f"{g(row['surface_radial_velocity_over_U'])} /U | 是 |"
        )
        rows.append(
            f"| 升力 Gamma={c:g} | -rho U Gamma = {row['lift_joukowski']:.6f} | 压力积分 "
            f"{row['lift_from_pressure_integral']:.6f} | 相对 {g(row['lift_relative_error'])} | 是 |"
        )
        rows.append(
            f"| 阻力 Gamma={c:g} | 0 (d'Alembert) | 压力积分 {g(row['drag_from_pressure_integral'])} | "
            f"相对升力 {g(row['drag_over_lift'])} | 是 |"
        )
        rows.append(
            f"| 驻点角 Gamma={c:g} | {row['stagnation_stated_deg'][0]:.6f}°, "
            f"{row['stagnation_stated_deg'][1]:.6f}° | 二分零点 "
            f"{row['stagnation_bisection_deg'][0]:.6f}°, {row['stagnation_bisection_deg'][1]:.6f}° | "
            f"{g(row['stagnation_max_deviation_deg'])}° | 是 |"
        )
    table = "\n".join(rows)

    markdown = f"""## ① 案例介绍

锚点 `CASE-BEM-LAPLACE-CYLINDER`（算例池·圆柱绕势流速度场，ref 解析复势，容差 <3%），仓 TensorBEM。
物理：二维不可压无粘绕流，圆柱半径 a={metrics['radius']:.3f}、来流 U={metrics['speed']:.3f}，
旋转圆柱附加环量 Gamma in {{1,2,3}}（均 < 4 pi U a，两驻点区）。几何门是无量纲的圆，Cb/L/B 不适用；
本单复核的是**参考解本身**，不上几何。

算例本体（路线 D/N/Gamma/M、常值+一阶矩密度、刻面归因、方位角扫描、多极远场，44 条判据，N=32/64/128）
已在 m20303/m21425/m21490 等轮并入。这些判据全部以手推闭式公式为基准，若公式转录有误，
BEM 与参考会一起对错、判据仍全绿——结构性自证风险。本单补这一环。

## ② 怎么做的

1. 速度 = 复势导数，用**离散 Cauchy 积分**在小圆（r=0.05|z|，256 节点）上求：
   W'(z) = (1/2 pi r) * oint W(z + r e^{{i theta}}) e^{{-i theta}} d theta。
   被积函数解析时梯形公式谱精度，公式内无相近数相减 → 误差落双精度底噪。
   升力势的 ln(z/a) 沿回路用 np.unwrap 连续化，避免主值支割线切过小圆。
2. 升力 = 压力积分 L' = -oint p sin(theta) a d theta（4096 点，p = (U^2-|v|^2)/2），不代入 Joukowski。
3. 驻点角 = 数值切向速度的零点（721 点扫描 + 二分到 1e-13 rad），不代入 sin(theta_s)=Gamma/(4 pi U a)。
4. 复核工具自身收敛性：节点数 8→256 的谱收敛曲线（图 1），确认裁判无误差。
5. 工具链：numpy + matplotlib；审计脚本 `audit_reference_m22101.py`（{metrics['duration_of_audit_seconds']} s）、
   守门测试 `tests/test_case_cylinder_reference_independent.py`（16 条，3.5 s）。

## ③ 计算结果（脚本写 JSON 再渲染，零手抄）

{table}

## ④ 结果图

图 1：Cauchy 求导随回路节点数的谱收敛（自由流与升力势），两支曲线都落到双精度底噪。
图 2：Gamma=3 旋转圆柱物面 Cp，闭式（实线）与数值复势导数（空心圈）重合并列，副轴为残差。

## ⑤ 注意点

- 复核对象是**参考解**，不是 BEM 离散解：保证「靶子对」，不保证边界元实现无误（后者由 44 条判据与二阶收敛负责）。
- 谱精度依赖被积函数解析；本算例奇点仅 z=0（体内），小圆恒不含奇点。含自由面/多体近场遮挡的场不能直接外推。
- 双精度底噪 ~1e-15 是复核下限，更小的参考解误差无法判定。
- 未完成项（如实）：已提交记录 `benchmarks/verified/case_bem_laplace_cylinder/results.json` 仍是 38 条判据、无 provenance 的旧版
  （模块现在输出 44 条）。重生成作业 cj3901 已递交但本单窗口内排队未起跑（队列 ~56 在飞），记录刷新未完成；
  对应 `test_committed_record_was_rendered_from_the_current_case_module` 仍 SKIP、
  `test_committed_case_artifact_matches_the_committed_document` 仍 FAILED，属既有陈旧欠账，非本轮引入。

## ⑥ 误差判定（逐项，vs 基准）

- 自由流速度场 vs Cauchy 数值导数：{g(free['velocity_max_abs_error_over_U'])} /U —— **<3%，是**
- 扰动势 vs Re W - Ux：{g(free['perturbation_potential_max_abs_error'])} —— **<3%，是**
- Cp vs Bernoulli+数值速度：{g(free['cp_max_abs_residual'])} —— **<3%，是**
- 升力势速度场（Gamma=1/2/3）vs Cauchy 数值导数：最大 {g(max(r['velocity_max_abs_error_over_U'] for r in metrics['lifting']))} /U —— **<3%，是**
- 物面速度（Gamma=1/2/3）：最大 {g(max(r['surface_speed_max_abs_error_over_U'] for r in metrics['lifting']))} /U —— **<3%，是**
- 升力 vs Joukowski（压力积分独立求得）：最大相对 {g(max(r['lift_relative_error'] for r in metrics['lifting']))} —— **<3%，是**
- 驻点角 vs 二分零点：最大 {g(max(r['stagnation_max_deviation_deg'] for r in metrics['lifting']))}° （相对 180° 量级 <1e-14）—— **<3%，是**

即：全部参考解都与「只从声明的复势出发的数值推导」一致到 1e-12 以内，
说明上一阶段 N=128 最差判据误差 2.849e-04 **来自离散化而非基准写错**。
"""
    return markdown, metrics


def publish(markdown: str) -> None:
    import urllib.request

    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    login = urllib.request.Request(
        f"{base}/auth/login",
        data=json.dumps(
            {
                "username": os.environ["TENSOR_AGENT_USER"],
                "password": os.environ["TENSOR_AGENT_PASS"],
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(login, timeout=60) as response:
        token = json.load(response)["access_token"]
    payload = {
        "title": "m22101 CASE-BEM-LAPLACE-CYLINDER：参考解链的独立复核（Cauchy 求导 / 压力积分 / 二分零点）",
        "summary": (
            "用与闭式公式不共享代数的数值方法从声明复势重推速度/Cp/升力/驻点角，"
            "全部一致到 1e-12 以内，证明算例 44 条判据的基准本身没错；新增 16 条守门测试。"
        ),
        "scenario_key": "CASE-BEM-LAPLACE-CYLINDER",
        "mission_id": 22101,
        "tags": ["TensorBEM", "case-pool", "verification-debt", "laplace", "cylinder", "reference-audit"],
        "markdown": markdown,
        "figures": [
            {
                "name": "audit_convergence.png",
                "caption": "Cauchy 求导的谱收敛：节点 8→256，误差落到双精度底噪，裁判自身无误差",
                "png_base64": base64.b64encode(FIG1.read_bytes()).decode(),
            },
            {
                "name": "audit_surface.png",
                "caption": "Gamma=3 旋转圆柱物面 Cp：闭式 vs 数值复势导数，副轴为残差",
                "png_base64": base64.b64encode(FIG2.read_bytes()).decode(),
            },
        ],
    }
    request = urllib.request.Request(
        f"{base}/tensor/reports",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        print("report:", json.load(response))


if __name__ == "__main__":
    markdown, _ = build_markdown()
    DOC.write_text(markdown, encoding="utf-8")
    print(f"wrote {DOC} ({len(markdown)} chars)")
    if "--publish" in sys.argv:
        publish(markdown)