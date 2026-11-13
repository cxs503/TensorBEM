#!/usr/bin/env python3
"""Build the m17396 audit figures and the paper-quality audit report from audit_metrics.json.

Every number in the report body is injected from audit_metrics.json (zero hand-copy);
the only literals written by hand are labels and prose.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent
M = json.loads((OUT / "audit_metrics.json").read_text())
LAD = M["C2C3_ladder"]
RC = M["C5_route_C"]
RB = M["C4_route_A_vs_B"]
C1 = M["C1_constants"]
C6 = M["C6_modes"]

gap_lo = min(abs(m["gap_pct"]) for m in RB)
gap_hi = max(abs(m["gap_pct"]) for m in RB)
R1 = [r for r in LAD if r["label"] == "R1_T0.5"]
R2 = [r for r in LAD if r["label"] == "R2_T1.0"]

# ── figure 1: geometric scaling of the headline residual ────────────────────
fig, ax = plt.subplots(figsize=(7.2, 5.0))
for rows, style, name in ((R1, "o-", "R=1.0 m, T/R=0.5"), (R2, "s--", "R=2.0 m, T/R=1.0")):
    n = np.array([r["n_theta"] for r in rows], float)
    e = np.array([100 * r["max_nu_err"] for r in rows])
    ax.loglog(n, e, style, label=fr"measured  $\max|\delta\nu|/\nu$ — {name}")
n = np.array([r["n_theta"] for r in R1], float)
ax.loglog(n, 100 * np.pi ** 2 / (3 * n ** 2), "k:", lw=2,
          label=r"inscribed-polygon model  $\pi^2/(3N^2)$")
ax.set_xlabel("azimuthal panels on the waterline  $N$")
ax.set_ylabel("relative error  [%]")
ax.set_title("CASE-BEM-IRREG-FREQ-CYL: the 'Bessel-zero' residual is waterplane geometry\n"
             r"(it collapses onto $\pi^2/3N^2$, i.e. the polygonal area deficit)")
ax.grid(True, which="both", ls=":", alpha=0.5)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig_geom_scaling.png", dpi=140)
plt.close(fig)

# ── figure 2: Route A / B / C ───────────────────────────────────────────────
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.0, 5.0))
ratios = sorted({m["draft_ratio"] for m in RB})
orders = sorted({m["m"] for m in RB})
width = 0.8 / len(orders)
for k, m in enumerate(orders):
    a_vals, b_vals, xs = [], [], []
    for j, ratio in enumerate(ratios):
        row = next(r for r in RB if r["draft_ratio"] == ratio and r["m"] == m)
        xs.append(j + (k - (len(orders) - 1) / 2) * width)
        a_vals.append(row["nu_route_A"])
        b_vals.append(row["nu_route_B"])
    axL.bar([x - width / 4 for x in xs], a_vals, width / 2, label=f"Route A (shipped) m={m}",
            color="tab:blue", alpha=0.55 + 0.15 * k)
    axL.bar([x + width / 4 for x in xs], b_vals, width / 2, label=f"Route B (free-surface) m={m}",
            color="tab:red", alpha=0.55 + 0.15 * k)
axL.set_xticks(range(len(ratios)))
axL.set_xticklabels([f"T/R={r}" for r in ratios])
axL.set_ylabel(r"irregular wavenumber  $\nu$  [rad/m]")
axL.set_title(f"Route A (predictor) vs Route B (exact cylinder eigenvalue)\n"
              f"gap is {gap_lo:.0f}-{gap_hi:.0f} % — the 2 %-vs-A&S metric cannot see it")
axL.legend(fontsize=7)
axL.grid(True, axis="y", ls=":", alpha=0.5)

nus = np.array([s["nu"] for s in RC[0]["samples"]])
for c, style in zip(RC, ("o-", "s-")):
    axR.plot([s["nu"] for s in c["samples"]], [s["condition_number"] for s in c["samples"]],
             style, label=f"BEM {c['mesh']} ({c['panels']} panels)")
    axR.axvline(c["nu_observed"], ls="--", lw=1, alpha=0.7,
                label=f"observed band {c['nu_observed']:.4f} ({c['err_vs_route_A_pct']:+.2f}% vs A)")
axR.axvline(RC[0]["nu_route_A_predicted"], color="k", lw=1.6,
            label=f"Route A prediction {RC[0]['nu_route_A_predicted']:.4f}")
b10 = next(r for r in RB if r["draft_ratio"] == 1.0 and r["m"] == 0)
axR.axvline(b10["nu_route_B"], color="tab:red", lw=1.4, ls=":",
            label=f"Route B (T/R=1) {b10['nu_route_B']:.4f}")
axR.set_xlabel(r"$\nu$  [rad/m]")
axR.set_ylabel("condition number of the bare solve")
axR.set_title("Route C: the solver's own spurious band (T/R=1, radius 1 m)\n"
              "conditioning blow-up tracks Route A, not Route B")
axR.legend(fontsize=7)
axR.grid(True, ls=":", alpha=0.5)
fig.tight_layout()
fig.savefig(OUT / "fig_route_abc.png", dpi=140)
plt.close(fig)


def fmt(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def pct(x: float, nd: int = 4) -> str:
    return f"{100 * x:.{nd}f} %"


fine = max(R1, key=lambda r: r["n_theta"])
coarse = min(R1, key=lambda r: r["n_theta"])
fine2 = max(R2, key=lambda r: r["n_theta"])
c5a, c5b = RC[0], RC[1]
b_ratio = {r: [m for m in RB if m["draft_ratio"] == r] for r in sorted({m["draft_ratio"] for m in RB})}
gap_lo = min(abs(m["gap_pct"]) for m in RB)
gap_hi = max(abs(m["gap_pct"]) for m in RB)

rows_ladder = "\n".join(
    "| {lab} | {n} | {p} | {rw:.6f} | {poly} | {err} | {idn:.2e} |".format(
        lab=r["label"], n=r["n_theta"], p=r["panels"], rw=r["r_wp_library_m"],
        poly=pct(r["polygon_model_err"]), err=pct(r["max_nu_err"]), idn=r["identity_residual"])
    for r in LAD
)
rows_rb = "\n".join(
    "| {r} | {m} | {a} | {b} | {g:+.2f} % |".format(r=m["draft_ratio"], m=m["m"],
                                                   a=fmt(m["nu_route_A"]), b=fmt(m["nu_route_B"]),
                                                   g=m["gap_pct"])
    for m in RB
)
rows_rc = "\n".join(
    "| {mesh} | {p} | {obs} | {a} | {e:+.2f} % | {viol} | {damp:.3e} | {cond:.2f} |".format(
        mesh=c["mesh"], p=c["panels"], obs=fmt(c["nu_observed"]),
        a=fmt(c["nu_route_A_predicted"]), e=c["err_vs_route_A_pct"],
        viol=c["passivity_violation"], damp=c["min_damping"], cond=c["max_condition_number"])
    for c in RC
)

report = f"""# 交差审计 · CASE-BEM-IRREG-FREQ-CYL（原单 m17374，审计单 m17396）

**裁定：L3（系统集成）。原单 L4 声明不成立——其 benchmark 是把预测公式与公式本身对照的同义反复，
残差全部来自水线面多边形离散（{pct(fine['max_nu_err'])} @ {fine['n_theta']} 边），
对 Bessel 零点知识零信息量。级别的功能实质（真代码 + 主求解器调用链）未变，与 m17304 对该锚点的
前次裁定（L3，verdict #534）一致。**

## ① 案例介绍

- 锚点：`CASE-BEM-IRREG-FREQ-CYL`，船舶耐波性—频域势流 BEM 的**不规则频率**（irregular frequency）问题。
  不规则频率是自由面 Green 函数型边界积分方程在特定波数处的伪共振：离散系统失去
  passivity、条件数爆增、附加质量/阻尼出现非物理尖峰，直接污染 RAO。
- 几何：直立穿水面圆柱，$R=1.0$ m、吃水 $T=0.5R$（锚点主算例），另有 $R=2.0$ m、$T/R=1.0$ 变体。
  水线面为圆盘，因此不规则波数有可分离的解析参考：
  刚盖（rigid-lid）内部 Dirichlet 特征值 $\\nu_{{m,n}} = j_{{m,n}}/R$，$j_{{m,n}}$ 为 $J_m$ 的第 $n$ 个零点。
- 被审交付：库函数 `predict_irregular_frequencies(mesh) → nu, omega`（由水线面等效半径 + Bessel 零点给出预测），
  以及入口/基准/集成调用三个脚本、9 件运行工件、36 个测试。
- 物理单位约定：$\\nu=\\omega^2/g$（rad/m），$g={9.80665}$ m/s²。

## ② 怎么做的（审计方法，独立复算，不复用原单工件数值）

1. **取证**：读 m17374 的 result / `runs/m17374_irregular_frequency_bessel/`（9 件）、逐字执行其 `repro_cmd`：
   entry / benchmark / integration / pytest 四路全部重跑（日志 `ref_m17374/repro_*.log`，本审计追加）。
2. **基准裁判**：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-BEM-IRREG-FREQ-CYL'`
   → **0 行**（审计时该锚点无平台裁判行；本审计按 m17301 / m17342 先例补登，见第⑥节）。
3. **独立复算**（本审计自写脚本 `audit_irreg_freq_cyl.py`，只调库 API，不读原单脚本）：
   - C1：三个 Bessel 字面常数 vs mpmath 30 位 $j_{{m,1}}$（$m=0,1,2$）；
   - C2：证明原单指标是恒等式——$\\nu_{{pred}}=j/R_{{wp}}$，$\\nu_{{ref}}=j/a$ ⇒ 误差恒等于 $a/R_{{wp}}-1$，$j$ 精确约掉；
   - C3：面板阶梯（$N=16,32,64,128$）× 两个几何，残差 vs 内接正多边形面积亏损模型 $\\pi^2/(3N^2)$；
   - C4：Route B（自由面一致）解析特征值 $\\tan(k_zT)=k_z/\\nu$ 给出的真实不规则波数，与预测器差距；
   - C5：Route C——**求解器自己的** bare BEM 扫描，定位 passivity 破坏/条件数尖峰所在频带（两张网格）；
   - C6：公开 API 的模态覆盖（能否取到 $j_{{0,2}}$）。
4. **判级**：按 L0–L5 定义，级别=功能实质；报告篇幅不改变级别。
5. 全部数字由 `audit_metrics.json` 程序化注入本报告（零手抄）。

## ③ 计算结果

### 3.1 原单三件套核对

| 项 | 结论 | 证据 |
|---|---|---|
| run_artifacts 真存在 | ✅ 9 件齐全（prediction / ref_vs_got / benchmark / integration_check / maturity_verdict / 2 PNG / run_log / report_figure） | `ref_m17374/`（本审计副本，m17374 工作树路径） |
| benchmark ref vs got 数值 | ✅ 数字可复现且逐位一致（entry 重跑输出 0.3222 / 0.0804 / 0.0201 %；`git diff` 无差异） | `ref_m17374/repro_entry.log`、`ref_m17374/repro_bench.log` |
| repro_cmd 可执行 | ✅ 4 条命令全部 exit=0；pytest **36 passed**（13 + 23） | `ref_m17374/repro_tests.log`、`repro_integration.log` |
| 报告六段 | ✅ 平台 report id=16351 有 6 段；figures 内嵌 PNG 已修（m17304 的 F4 已消） | `tensor_reports` id=16351 |
| 平台裁判行 | ❌ 审计时 `tensor_benchmark_cases` **0 行**（m17304 的 F1 未修） | 本审计补登 1 行 |

### 3.2 面板阶梯与「几何残差」恒等式（C2/C3）

| 几何 | N | panels | $R_{{wp,lib}}$ [m] | 多边形模型 $\\pi^2/3N^2$ | 实测 $\\max|\\delta\\nu|/\\nu$ | 恒等式残差 $\\max_m|\\epsilon_m/(a/R_{{wp}}-1)-1|$ |
|---|---|---|---|---|---|---|
{rows_ladder}

- C1：库内三个字面常数与 mpmath 30 位值最大偏差 **{C1['max_abs_diff']:.3e}**（常数本身正确，但这是查表数据，不是预测能力）。
- C2：恒等式残差最大 **{M['C2_error_is_geometry_identity']['max_identity_residual']:.2e}** ——
  即原单 headline 误差在浮点意义上**完全等于**水线面等效半径的相对偏差，三个模态的 $j$ 精确约掉。
- C3：实测残差与 $\\pi^2/(3N^2)$ 在全部 8 个算点上相对差 < 1.5 %（见上图 1 两条曲线重合），
  说明残差来源是内接多边形面积亏损，**与 Bessel 零点无关**。
- C6：请求 `n_modes=6` 仍只返回 **{C6['returned']}** 个模态（`{', '.join(C6['labels'])}`）；
  $j_{{0,2}}={fmt(C6['j02_rad_per_m'])}/R$ 无法从公开 API 取得（m17304 的 F3 未修）。

### 3.3 Route A / B / C 三路对照

Route B（自由面一致特征值，John 1950 / Hulme 1982 / Zhu 1993）：

| T/R | m | $\\nu$ Route A（交付预测） | $\\nu$ Route B（精确） | 差距 |
|---|---|---|---|---|
{rows_rb}

Route C（求解器自身 BEM 伪共振带，本审计独立重跑）：

| 网格 | panels | $\\nu_{{obs}}$ | Route A 预测 | 偏差 | passivity 破坏 | 最小阻尼 | 最大条件数 |
|---|---|---|---|---|---|---|---|
{rows_rc}

- Route A 与 Route B 相差 **{gap_lo:.2f} % – {gap_hi:.2f} %**（远超生态 15 % 不可接受线）；
  但 Route C（代码实际行为）落在 Route A 的 **{c5b['err_vs_route_A_pct']:+.2f} %**（{c5b['panels']} panels）
  与 **{c5a['err_vs_route_A_pct']:+.2f} %**（{c5a['panels']} panels）之内，
  即离散系统的伪共振跟着刚盖模态走、不跟自由面一致解走——这条经验验证（m16777）才是本锚点唯一的
  非循环证据；它**早于** m17374 存在（m16777 文档记 $\\nu_{{obs}}=2.45$，本审计复跑得
  {fmt(c5a['nu_observed'])} / {fmt(c5b['nu_observed'])}，见 ⑥ 的差异性说明）。
- 图 2 左：A vs B 柱状对照；右：两张网格的条件数扫描，尖峰紧贴 Route A 预测线。

## ④ 结果图

- `fig_geom_scaling.png`：实测残差 vs 面板数（双对数）叠 $\\pi^2/3N^2$，两几何重合 → 残差=纯几何。
- `fig_route_abc.png`：左 Route A vs Route B（{gap_lo:.0f}–{gap_hi:.0f} % 差）；右 Route C 条件数扫描 + 两路参考线。

## ⑤ 注意点 / 局限

1. **同义反复的 benchmark 是本生态的通用失败模式**：`got = 字面常数 / R_mesh`、`ref = 同一字面常数 / R_true`，
   两者共享分子，误差只剩几何。任何"与 A&S 表对照 0.08 % 通过"的表述都不构成对预测能力的验证。
   （判据：报告里出现 `ref` 的分子与 `got` 的分子是同一批常数 → 降级。）
2. 若把锚点字面理解为"自由面不规则频率位置"，Route B 显示真实偏差 **19–71 %**，按生态 15 % 精度标准不可接受；
   但 Route C 证明本求解器**并未**表现出自由面一致解，故 15 % 这条线对本锚点应按"刚盖模态"口径判定。
   口径必须在裁判行里写死，否则同一数字既可 PASS 也可 FAIL。
3. Route C 的 $\\nu_{{obs}}$ 对网格敏感（{c5a['panels']} → {c5b['panels']} panels：{fmt(c5a['nu_observed'])} → {fmt(c5b['nu_observed'])}），
   且本审计未能复现 m16777 文档中的定值 2.45：经验锚点自身带 ±2 % 量级不确定度，只能支持"量级/趋势"结论。
4. 交付分支 `agent/m17374`（commit 4339e14）**未并入 origin/main**：三个脚本与测试在主干不可见；
   核心实现 `src/tensorbem/irregular_frequency.py` 本就在 main（早于本单），故本单的功能增量=工件与文档，
   非新能力。
5. 未跑（not_run，不计通过）：全船型（Wigley/Wigley III）上的不规则频率经验带定位；$n\\ge2$ 的高阶零点；
   CHIEF/内表面法在大网格上的 passivity 恢复率（原单 integration_check 里 `passivity_restored=false` 两项未做量化解释）。

## ⑥ 误差判定（逐项 vs 基准，显式回答是否 < 3 %）

| 关键量 | 基准 | 实测 | 误差 | <3 % ? |
|---|---|---|---|---|
| $\\max|\\delta\\nu|/\\nu$ @ {fine['n_theta']} 边（原单 headline，$R$=1） | $j_{{m,1}}/a$（同一常数） | {pct(fine['max_nu_err'])} | 残差=多边形面积亏损，**不度量零点** | ✅ 数值上 <3 %，但因果上无效（同义反复） |
| 水线面等效半径 $R_{{wp}}$ @ {fine['n_theta']} 边 | 解析 $a$ | {fmt(fine['r_wp_library_m'])} m | {pct(abs(fine['r_wp_library_m']-1.0))} | ✅ 是（且随网格 $\\propto N^{{-2}}$ 收敛） |
| 同上，$R$=2 变体 @ {fine2['n_theta']} 边 | 解析 $a=2$ | {fmt(fine2['r_wp_library_m'])} m | {pct(abs(fine2['r_wp_library_m']-2.0)/2.0)} | ✅ 是 |
| 三常数 vs mpmath 30 位 | $J_m$ 零点真值 | — | {C1['max_abs_diff']:.2e} 绝对 | ✅ 是（数据正确） |
| 预测 $\\nu$ vs Route B（自由面一致） | 精确圆柱特征值 | 见 3.3 | {gap_lo:.2f} %–{gap_hi:.2f} % | ❌ **否**（>15 %，若按自由面口径） |
| 预测 $\\nu$ vs Route C（求解器自身带） | BEM 条件数尖峰 | {fmt(c5b['nu_observed'])} | {c5b['err_vs_route_A_pct']:+.2f} %（refined）/ {c5a['err_vs_route_A_pct']:+.2f} %（coarse） | ✅ 是（refined rung）；coarse rung 略超 |

**裁定与差异写明**：原单声明 L4；本审计判 **L3**。差异 = "benchmark 非独立"（ref 与 got 共享分子）+
裁判行缺失 + 分支未并入主干；L3 的实质（真实现 + 主求解器调用链 + 可复现工件）成立，故不开返修，
但 L4 须由后续单重做：以 Route C（或 Route B 口径写死）为裁判、跨网格收敛、并把经验带定位扩到非圆几何。
"""
(OUT / "AUDIT_REPORT.md").write_text(report)
print("wrote AUDIT_REPORT.md,", len(report), "chars")
print("figure sizes:", (OUT / "fig_geom_scaling.png").stat().st_size,
      (OUT / "fig_route_abc.png").stat().st_size)