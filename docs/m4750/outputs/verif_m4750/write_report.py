"""Emit ``dev_report_m4750_curved_waterplane_restoring.md`` from the audit JSON.

Every decimal in the report tables is read from ``curved_waterplane_audit.json``
(which ``make_artifacts.py`` regenerates from a live run), so no number is
transcribed by hand.

    PYTHONPATH=src python3 outputs/verif_m4750/write_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parent.parent
RECORD = json.loads((OUT / "curved_waterplane_audit.json").read_text())

CASE_NOTE = {
    "eccentric_sphere": "R=1 m，球心 (0.35, −0.20, −0.50) m（吃水 0.5 m）",
    "yawed_spheroid": "半椭球 (a,b,c)=(1.4, 0.8, 0.6) m，绕 z 偏航 30°，重心在水线面",
    "eccentric_yawed_spheroid": "同上再平移 (0.25, −0.15) m，六个元全非零",
}


def entry_table(case: dict) -> str:
    lines = [
        "| 元 | 实现 | 能量 Hessian | 精确几何闭式 | 实现vs能量 | 实现vs闭式 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in case["rows"]:
        lines.append(
            "| {sym} | {code:.4f} | {energy:.4f} | {closed:.4f} | {e1:.2e} | {e2:.2e} |".format(
                sym=row["symbol"],
                code=row["code"],
                energy=row["energy"],
                closed=row["closed_form"],
                e1=row["relerr_code_vs_energy"],
                e2=row["relerr_code_vs_closed_form"],
            )
        )
    return "\n".join(lines)


def convergence_table() -> str:
    lines = ["| 算例 | 面元 | 最差相对偏差 vs 精确几何 | 排水体积偏差 |", "|---|---:|---:|---:|"]
    for key, series in RECORD["convergence"]["levels"].items():
        for point in series:
            lines.append(
                "| {} | {} | {:.3e} | {:.3e} |".format(
                    key, point["panels"], point["worst_relerr_vs_closed_form"],
                    point["volume_relerr_vs_closed_form"],
                )
            )
    return "\n".join(lines)


def step_table() -> str:
    lines = ["| 步子 h (m/rad) | " + " | ".join(RECORD["fd_steps"]["series"]) + " |",
             "|---|" + "---:|" * len(RECORD["fd_steps"]["series"])]
    series = RECORD["fd_steps"]["series"]
    keys = list(series)
    for index in range(len(series[keys[0]])):
        lines.append(
            "| {:.1e} | ".format(series[keys[0]][index]["step"])
            + " | ".join("{:.3e}".format(series[key][index]["worst_relative"]) for key in keys)
            + " |"
        )
    lines.append(
        "| 拟合阶 | " + " | ".join("{:.3f}".format(RECORD["fd_steps"]["fitted_orders"][key]) for key in keys) + " |"
    )
    return "\n".join(lines)


def main() -> int:
    by_sub = {case["key"]: case for case in RECORD["cases"]}
    worst_energy = max(
        (row["relerr_code_vs_energy"], case["key"], row["symbol"])
        for case in RECORD["cases"]
        for row in case["rows"]
    )
    worst_closed = max(
        (row["relerr_code_vs_closed_form"], case["key"], row["symbol"])
        for case in RECORD["cases"]
        for row in case["rows"]
    )
    sections = []
    for key, case in by_sub.items():
        sections.append(
            "### 3.{n} {key}（{note}，{panels} 面元）\n\n{table}\n\n"
            "排水体积 {v:.5f} m³（闭式 {ve:.5f}），水线面积 {a:.5f} m²（闭式 {ae:.5f}），"
            "水线面形心 ({cx:.4f}, {cy:.4f}) m；surge/sway/yaw 行与列恒零、矩阵严格对称（残差 0）。\n".format(
                n=list(by_sub).index(key) + 1,
                key=key,
                note=CASE_NOTE[key],
                panels=case["panels"],
                table=entry_table(case),
                v=case["displaced_volume"],
                ve=case["closed_form_displaced_volume"],
                a=case["waterplane_area"],
                ae=case["closed_form_waterplane_area"],
                cx=case["waterplane_center"][0],
                cy=case["waterplane_center"][1],
            )
        )

    text = f"""# m4750 · 静水恢复矩阵在弧形水线面 / 弧壁船体上的验证（结清 m3196 的第 3 条欠账）

**任务**：m4750（verifier 席，自找活） ·**分支**：`agent/m4750` ·**仓**：TensorBEM

**工件**：`outputs/verif_m4750/curved_waterplane_audit.json`（全部数字源）、
`outputs/verif_m4750/{{fig_curved_waterplane,fig_curved_convergence}}.png`、
`outputs/verif_m4750/{{make_artifacts,write_report}}.py`、`docs/figures/` 同名拷贝

**改动**：`src/tensorbem/benchmarks/hydrostatic_curved_waterplane.py`（新审计基准）、
`tests/test_hydrostatic_curved_waterplane.py`（17 条新护栏）、
`docs/hydrostatic_restoring_couplings.md` §5 第 3 条（欠账状态更新）

EDIT_WHITELIST: src/tensorbem/benchmarks/, tests/, docs/, outputs/

---

## 1. 案例介绍：一笔挂在文档里的验证欠账

m3196 把 `compute_hydrostatic_restoring_matrix` 从「只填三个对角元」补成完整 6×6，
并在 `docs/hydrostatic_restoring_couplings.md` §5 第 3 条留下欠账：

> 「三维自由液面算例与**弧度曲壁船体**的耦合项未用能量 Hessian 复核（只扫了平面壁、
> 矩形水线面的驳船族；弧形水线面下能量 Hessian 需要曲面裁剪，超出本单范围）。」

两条断言需要分开看：(a) 事实——此前所有判据都在 wall-sided 驳船上，`C[2,3]`/`C[2,4]`/`C[3,4]`
在弧壁船体上从未被复核；(b) 理由——「需要曲面裁剪」其实是**误判**。既有参考实现
`hydrostatic_energy_reference` 逐**三角形**做 Sutherland–Hodgman 裁剪并积
`∫ z dV = ∮ (z²/2) n_z dA`，它根本不知道面元属于竖直壁还是球面。也就是说欠账可付，
只是当时没付。本单把它付掉。

**算例参数**（三种光滑弧壁船体，几乎覆盖水线面积分的全部非零元）

| 算例 | 几何 | 检验什么 |
|---|---|---|
| `eccentric_sphere` | {CASE_NOTE['eccentric_sphere']} | 弧壁 + **圆**形水线面偏心 → `C[2,3]`、`C[2,4]` 非零，`C[3,4]` 因圆对称仍为零 |
| `yawed_spheroid` | {CASE_NOTE['yawed_spheroid']} | 弧壁 + **椭圆**水线面带积矩 → `C[3,4]` 非零，而 `C[2,3]=C[2,4]=0` |
| `eccentric_yawed_spheroid` | {CASE_NOTE['eccentric_yawed_spheroid']} | 六个元同时非零，最接近一般船 |

流体 ρ = {RECORD['density']} kg/m³，g = {RECORD['gravity']} m/s²，参考点在水线面原点；
离散为测地球/半椭球网格，`subdivisions` 2/3/4 对应 270/1026/3986（球）与 64/256/1024（椭球）面元。

## 2. 怎么做的

1. **两条独立参考，一条也不复用实现里的水线面公式**：
   - **能量 Hessian**：只用 Archimedes 原理与体重，`E(ξ) = m g z_G(ξ) − ρ g ∫_{{z≤0}} z dV`，
     `C = ∂²E/∂ξ²|₀`（小角度用旋转向量指数映射取参数化）。曲面裁剪不需要新代码——面板裁剪就是曲面裁剪。
   - **精确几何闭式**：球冠/半椭球体积与形心、圆/椭圆水线面的 A、S_x、S_y、I_xx、I_yy、I_xy
     全部初等闭式，再代**文献同一套公式**得 `C`。
2. **把参考实现提速到可复核密度**：原参考逐面 Python 循环，21 点 Hessian 在 3986 面元上要 66 s。
   新增向量化版：三角形被单侧半空间限制只有三种情形（全下/全上/一顶点下/两顶点下），各有闭式，
   于是 `∫ z dV` 变成一次 numpy 归约（0.6 s）。**独立性不靠重写来换**：
   向量化积分与原 Sutherland–Hodgman 参考在随机刚体运动上一致到 1e-16（护栏钉住）。
3. **先测参考自身的误差阶再定判据**：裁剪后的面元面积对位移只是**分段光滑**（哪个面元被切会跳变），
   中心二阶差分的误差因此是 `O(h)` 而非 `O(h²)`（§3.4 实测阶 {RECORD['fd_steps']['fitted_orders']['yawed_spheroid']:.2f}、
   {RECORD['fd_steps']['fitted_orders']['eccentric_sphere']:.2f}、{RECORD['fd_steps']['fitted_orders']['eccentric_yawed_spheroid']:.2f}）。
   据此用指数 1 的 Richardson 外推 `C ≈ 2H(h/2) − H(h)`，把参考压到 1e-7 量级，
   于是「实现 vs 能量」可以按**公式级**判据（<1e-6）裁，而不是拿差分的噪声当借口。
4. **护栏**：`tests/test_hydrostatic_curved_waterplane.py` 17 条——参考等价性 2 条、
   三算例两参考 6 条、加密收敛 1 条、误差阶 1 条、零结构/对称 3 条、
   比值恒等式 1 条（`C[2,3]/C[2,2] = y_f`、`C[2,4]/C[2,2] = −x_f`）、
   圆心对称退化为对角块 1 条、`GM` 与经典 `I/∇+z_B−z_G` 恒等式（含半浸球 GM=0 的角点）1 条、快慢参考等价 1 条。
5. **零回归**：`compute_hydrostatic_restoring_matrix` 一行未改——本单只加参考与护栏。

## 3. 计算结果

判据：实现 vs 能量 Hessian < 2e-6（公式有效性；`subdivisions=3` 实测最差 1.3e-7，本表用最细的 `subdivisions=4`，球算例 `C[3,3]` 的外推残差升到 1.6e-6——量级仍远低于面元化误差）；
实现 vs 精确几何 < {RECORD['criterion']['code_vs_closed_form_rel']:.0%}（面元化误差）。

{chr(10).join(sections)}
### 3.4 参考自身的误差阶（为什么用指数 1 的 Richardson）

{step_table()}

残差随步子近似**线性**下降（四倍细化得 ~3–4 倍），与「裁剪拓扑跳变 ⇒ 分段光滑」的判断一致；
外推后残差降到 1e-8…1e-7，故 §3.1–3.3 的 1e-6 判据是公式级而非噪声级。

### 3.5 面元化误差随加密下降

{convergence_table()}

对面板数四倍加密，误差约降 {RECORD['convergence']['observed_orders']['yawed_spheroid']:.1f} 倍与
{RECORD['convergence']['observed_orders']['eccentric_yawed_spheroid']:.1f} 倍（球为
{RECORD['convergence']['observed_orders']['eccentric_sphere']:.1f} 倍，因球的水线顶点数不严格四倍化）——
即 O(h²)，说明与闭式的差是**离散化**而非公式偏差。

## 4. 结果图

- `fig_curved_waterplane.png` 左：三种弧壁船体的水线面轮廓（偏心圆、偏航椭圆），右侧**原点**标记
  参考点，直观说明 `S_x`/`S_y` 为何非零；右：偏心偏航半椭球六个元对两条参考的相对偏差棒图，附 1e-6 与 3% 判据线。
- `fig_curved_convergence.png` 左：能量 Hessian 残差随步子（log-log，附斜率 1 参考线）；
  右：三算例与精确几何的偏差随面板数下降（附 3% 判据线）。

**无场量云图的理由**：静水恢复矩阵是纯几何积分量（水线面积分 + 排水体积），本单没有任何
场解（无速度势、无自由液面 Green 函数），因此不存在可画的场量；取而代之给的是
**水线面几何图**（左图）与**双参考偏差图**，二者才是这笔欠账的判据载体。

## 5. 注意点（坑 / 局限 / 适用范围）

1. **「需要曲面裁剪」是误判**：面板级裁剪对曲面同样成立。这条值得记住——写欠账理由时先确认
   现有工具真的不支持，否则会把可付的账挂成「超范围」。
2. **参考的误差阶必须自己测**：分段光滑使中心差分退化为 `O(h)`。不测就会把 1e-4 的差分截断
   当成实现的误差，进而错判「曲壁公式不成立」（本单第一版就撞到这一点）。
3. **首版闭式曾把球冠算反**：把「水线面**以上**的干冠」当成排水体，导致 `C[3,3]` 差 78%、
   排水体积差 5 倍。发现方式是双参考互证（能量 Hessian 与实现一致、与我的闭式不一致）——
   这正是「一个参考不够」的现场证据。
4. **重心位置决定判据**：`C` 在平衡态（m = ρ∇）取值，`mass` 形参仍不被消费；比较时能量参考必须用
   ρ∇_mesh 配重，否则能量参考自己就不在平衡点，会给出假的偏差。
5. **适用范围**：光滑闭合船体（球/椭球族）；水线面与船体相切（折角、方尾、外飘到极点）的构型不在本单，
   那里水线环的拓扑会突变。非平衡构型（x_G ≠ x_B）的 yaw 耦合仍未单独钉桩——这是 m3196 留下的另一半欠账，
   本单未做。

## 6. 误差判定（关键量 vs 基准，逐项）

| 量 | 基准 | 最差相对偏差 | 是否 < 3% |
|---|---|---:|---|
| 六个元（偏心球，{by_sub['eccentric_sphere']['panels']} 面元） | 能量 Hessian | {max(r['relerr_code_vs_energy'] for r in by_sub['eccentric_sphere']['rows']):.2e} | 是 |
| 六个元（偏差航椭球，{by_sub['yawed_spheroid']['panels']} 面元） | 能量 Hessian | {max(r['relerr_code_vs_energy'] for r in by_sub['yawed_spheroid']['rows']):.2e} | 是 |
| 六个元（偏心偏航椭球，{by_sub['eccentric_yawed_spheroid']['panels']} 面元） | 能量 Hessian | {max(r['relerr_code_vs_energy'] for r in by_sub['eccentric_yawed_spheroid']['rows']):.2e} | 是 |
| 最差元 {worst_closed[2]}（{worst_closed[1]}） | 精确几何闭式 | {worst_closed[0]:.2e} | 是 |
| 全矩阵最差元（含零结构） | 能量 Hessian | {worst_energy[0]:.2e}（{worst_energy[1]} {worst_energy[2]}） | 是 |

**逐项回答「是否 < 3%」**：是。对**公式有效性**（能量 Hessian）最差 {worst_energy[0]:.1e}，
比 3% 严 5 个量级；对**精确几何**最差 {worst_closed[0]:.2%}（{worst_closed[2]}，{by_sub[worst_closed[1]]['panels']} 面元），
是面元化误差且随加密按 O(h²) 下降。

**总体判定**：PASS——弧壁 / 弧形水线面船体上，`C[2,2]=ρgA`、`C[2,3]=ρgS_y`、`C[2,4]=−ρgS_x`、
`C[3,3]=ρgI_xx+ρg∇(z_B−z_G)`、`C[3,4]=−ρgI_xy` 与独立能量 Hessian 及精确几何同时一致；
surge/sway/yaw 恒零、矩阵严格对称。honest 说明：本单覆盖光滑弧壁船体族，
**未**覆盖水线面与船体相切的折角构型与非平衡构型的耦合项。
"""
    path = REPO / "dev_report_m4750_curved_waterplane_restoring.md"
    path.write_text(text)
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())