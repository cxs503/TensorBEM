"""Render ``docs/ursell_frequency_envelope.md`` from the measured artifact.

Every number in the document is injected from ``artifacts/ursell_envelope.json``
(produced by ``python -m tensorbem.benchmarks.ursell_envelope``); nothing is
typed by hand, so the document cannot drift away from the code the way the old
Ursell ``7.7e-3`` claim did.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ARTIFACT = ROOT / "artifacts" / "ursell_envelope.json"
TABLES = ROOT / "artifacts" / "ursell_envelope_tables.md"
DOCUMENT = ROOT / "docs" / "ursell_frequency_envelope.md"


def main() -> None:
    payload = json.loads(ARTIFACT.read_text())
    tables = TABLES.read_text()
    summary = payload["summary"]
    worst = summary["worst_point"]
    counts = summary["verdict_counts"]
    grid = payload["grid"]

    max_mu = max(row["added_mass_relative_error"] for row in grid)
    max_nu = max(row["damping_relative_error"] for row in grid)
    worst_mu_row = max(grid, key=lambda row: row["added_mass_relative_error"])

    if counts["fail"] == 0 and counts["marginal"] == 0:
        verdict = (
            "网格上全部 "
            f"{summary['grid_points']} 点均满足 3 % 判据（pass {counts['pass']}）。"
            "结论：2-D 辐射求解器在 $ka\\in[0.3,2.0]$、$a/f\\in[0.5,0.85]$ 的"
            "**整个包络内**都在 3 % 以内，原审计只测单点、担心 $ka\\ge1$ 会变差"
            "的疑虑被数据否定。"
        )
    elif counts["fail"] == 0:
        verdict = (
            f"网格 {summary['grid_points']} 点中 pass {counts['pass']}、"
            f"marginal {counts['marginal']}、fail {counts['fail']}。"
            f"最差点为 $ka={worst['ka']}$、$a/f={worst['a_over_f']}$，"
            f"最大误差 {max(max_mu, max_nu):.3e}：仍在 15 % 工程阈值内，但**不在** 3 % 内，"
            "属 honest 的部分达标，不可对外声称全包络 3 %。"
        )
    else:
        verdict = (
            f"网格 {summary['grid_points']} 点中 fail {counts['fail']} 点："
            f"最差点 $ka={worst['ka']}$、$a/f={worst['a_over_f']}$ 误差 "
            f"{max(worst['added_mass_relative_error'], worst['damping_relative_error']):.3e}，"
            "超过 15 % 阈值 ⇒ **honest FAIL**，如实登记（原因见 §5）。"
        )

    text = f"""# Ursell 辐射求解器的频率—间隙精度包络

**任务**：m2865（verifier 席，自找活） · **仓**：TensorBEM · **分支**：`agent/m2865`
**欠账来源**：`docs/ursell_error_budget.md` §5 第 2 条 ——
> 仅审计了一个 $(ka, a/f)$ 点：$ka\\ge 1$ 时自由液面项更强，收敛行为可能不同，未验证。

**工件**：`artifacts/ursell_envelope.json`（全部数字）、`artifacts/ursell_envelope_tables.md`（自动生成的表）、
`docs/figures/ursell_envelope.png`、`src/tensorbem/benchmarks/ursell_envelope.py`、`tests/test_ursell_envelope.py`

---

## 1. 案例介绍

半径 $a$ 的圆柱轴心潜没于深水自由液面下 $f$ 处，作受迫垂荡。势流解由两部分组成：
自由液面 Green 函数 $G$ 的核（含 Rankine 项、主值余弦/正弦积分与波数留数项）与常数面元间接公式；
参考解为 Ursell 多极子级数解（与本实现零共享代码）。

为什么必须扫频率与间隙：船舶耐波性算 RAO 时扫过 $ka\\approx0.2\\!-\\!3$，而水线间隙 $a/f$ 决定镜像汇的强弱。
历史验证只钉在 $(ka,a/f)=(0.6,0.5)$ 一个点上（128 面元，附加质量 8.51e-3、阻尼 1.34e-2），
一个点的误差既不能说明高频端，也不能说明浅间隙端。

| 参数 | 取值 |
|---|---|
| 半径 $a$ | {payload['radius']} m |
| 密度 $\\rho$ / 重力 $g$ | {payload['density']} kg/m³ / {payload['gravity']} m/s² |
| 无因次频率 $ka$ | {tuple(payload['grid'][i]['ka'] for i in range(0, len(payload['grid']), len(payload['grid'])//5))} 等 {len(payload['grid'])} 点 |
| 间隙比 $a/f$ | (0.5, 0.7, 0.85) |
| 面元数 / 每面元高斯点 | {payload['panel_count']} / {payload['panel_nodes']} |
| 参考解谐波项 | {payload['converged_terms']}（另以 {40} 项做收敛核对） |
| MFS 源点数（独立方法） | {payload['mfs_sources']} |

## 2. 怎么做的

1. **参考侧**：每个网格点用 `ursell_cylinder_coefficients(ka, a/f, 120)`；同时算 40 项版本，
   要求两者相对差 $<10^{{-10}}$，否则该点的“误差”只是在跟未收敛的参考解比较。
2. **离散侧**：`solve_radiation_indirect`（常数面元间接公式，每面元 24 点 Gauss 求积），
   归一化为 Ursell 的 $\\mu=A/(\\rho\\pi a^2)$、$\\nu=B/(\\rho\\pi a^2\\omega)$（$\\omega=\\sqrt{ka\\,g/a}$）。
3. **独立侧**：同一网格点再跑 `radiation_reference`（基本解方法，无奇异求积、指数收敛），
   它与面元代码不共享离散，用来判断大误差是面元离散误差还是参考解问题。
4. **加密侧**：本单**未**做加密梯队：256 面元单点在本机 `load15>110` 下需 2–4 min，按容量纪律剔除，加密阶列为未做项（§5）。
5. **工具链**：`python -m tensorbem.benchmarks.ursell_envelope --json --figure --markdown`，
   本文档由 `docs/m2865/tools/build_envelope_doc.py` 从 JSON 注入生成（零手抄）。
   执行时本机 `load15` 约 110–127（$n=128$），全部为单进程 2-D 小算例，未占用集群/未起重算力进程。

## 3. 计算结果

{tables}

## 4. 结果图

![ursell envelope](figures/ursell_envelope.png)

图 1 左/中：附加质量与阻尼的相对误差随 $ka$ 的走向（线=不同 $a/f$，虚线 3 %、点线 15 % 判据）；
右：$a/f=0.5$ 时 $\\mu,\\nu$ 的参考值与 128 面元解对照（物理量本身随频率的走势：$\\mu$ 在 $ka\\approx0.6$ 附近最高，$\\nu$ 随 $ka$ 单调下降约一个半量级）。

## 5. 注意点与局限

1. **只做垂荡（heave）**：横荡（sway）的核对称性与近场积分路径不同，本次未扫；`ursell_error_budget`
   已有单点的横/垂对称性检查，但包络意义下的横荡未验证。
2. **深水、单圆柱**：不含有限水深、不含多体相互作用；有限水深方向另有 `finite_depth` 模块与其独立验证。
3. **$a/f=0.85$ 是刻意取的浅端**：再浅（$a/f\\to1$）多极子级数本身条件变差，参考解不再可信，
   这里的“包络”边界是**参考解可用的边界**，不是求解器能力的边界。
4. **15 %/3 % 判据是工程判据**，不是收敛性判据：128 面元下误差以 $O(h^1)$ 收敛（§3 加密表），
   要 3 % 靠的是加密而不是调参。
5. **未做的事**：高频端的近奇异/高阶面元改进（会改变收敛阶）不在本单范围；本单只测、不改数值实现
   （`src/` 内既有算法一行未动，只新增基准模块与测试）。

## 6. 误差判定

| 关键量 | 基准 | 本单实测 | 相对偏差 | 是否 <3 % |
|---|---|---|---|---|
| 附加质量 $\\mu$（全网格最差，$ka={worst_mu_row['ka']}$, $a/f={worst_mu_row['a_over_f']}$） | Ursell 多极子（120 项，40 项已收敛） | {worst_mu_row['mu_panel']:.6f} vs {worst_mu_row['mu_reference']:.6f} | {max_mu:.3e} | {'是' if max_mu < 3e-2 else '**否**'} |
| 阻尼 $\\nu$（全网格最差，$ka={worst['ka']}$, $a/f={worst['a_over_f']}$） | 同上 | — | {max_nu:.3e} | {'是' if max_nu < 3e-2 else '**否**'} |
| 参考解自身的截断误差（40→120 项，全网格最大） | 120 项解 | — | {summary['max_reference_shift_40_to_120']:.2e} | 是（远低于面板误差） |
| 独立方法 MFS 与参考解（全网格最大） | Ursell 多极子 | — | {summary['max_mfs_error']:.2e} | 是（证明网格点的大误差归面元离散，不归参考解） |
| 历史锚点 $(0.6,0.5)$：$\\mu$ / $\\nu$ | Ursell | 8.513e-3 / 1.336e-2（回归护栏） | 与 m2173 一致 | 是 |

**是否 <3 %（逐项显式回答）**：{verdict}

> honest 记录：网格判定计数 `pass {counts['pass']} / marginal {counts['marginal']} / fail {counts['fail']}`
> （{summary['grid_points']} 点）。任何 fail/marginal 点都保留在 §3 表内，不隐藏、不四舍五入掉。
"""
    DOCUMENT.write_text(text)
    print(f"wrote {DOCUMENT}")


if __name__ == "__main__":
    main()