#!/usr/bin/env python3
"""Figures + platform report for XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN (m21082)."""
import base64, io, json, os, subprocess
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CASE = Path("benchmarks/verified/bem_finite_depth_capytaine_xval")
d = json.loads((CASE / "xval_results.json").read_text())
rows = [r for r in d["comparison"]["comparison"] if "am_rel_error" in r]
g = d["comparison"]["gates"]
ana = d["analytic_infinite_depth"]

stations = [3.0, 5.0, 12.0, 20.0]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

col = {3.0: "#1f77b4", 5.0: "#2ca02c", 12.0: "#d62728", 20.0: "#9467bd"}
for s in stations:
    rs = sorted([r for r in rows if r["depth_ratio"] == s], key=lambda r: r["ka"])
    ks = [r["ka"] for r in rs]
    ax[0].plot(ks, [r["tbem_am_heave"] for r in rs], "o-", color=col[s], label=f"TensorBEM h/R={s:g}")
    ax[0].plot(ks, [r["cpt_am_heave"] for r in rs], "x--", color=col[s], alpha=0.6,
               label=f"capytaine h/R={s:g}")
ax[0].plot([k for k in (0.3, 0.6, 1.0, 1.5, 2.0)],
           [ana[str(k)]["added_mass_heave"] for k in (0.3, 0.6, 1.0, 1.5, 2.0)],
           "k:", lw=2, label="analytic multipole ($h=\\infty$)")
ax[0].set_xlabel("ka"); ax[0].set_ylabel("$A_{33}$ [kg]")
ax[0].set_title("Heave added mass, 320-panel icosphere"); ax[0].legend(fontsize=6); ax[0].grid(alpha=.3)

for s in stations:
    rs = sorted([r for r in rows if r["depth_ratio"] == s], key=lambda r: r["ka"])
    ax[1].plot([r["ka"] for r in rs], [100 * r["am_rel_error"] for r in rs], "o-",
               color=col[s], label=f"h/R={s:g}")
ax[1].axhline(10, color="r", ls="--", label="G1 gate 10%")
ax[1].set_xlabel("ka"); ax[1].set_ylabel("$|A_{33}^{TBEM}-A_{33}^{cpt}|/A_{33}^{cpt}$ [%]")
ax[1].set_title("Cross-code error (sign flips at h/R=12)"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)

modes = [(48, 9786.4), (64, 463426.8), (96, -2175.0)]
ax[2].bar([str(m) for m, _ in modes], [v for _, v in modes], color=["#2ca02c", "#d62728", "#9467bd"])
ax[2].set_yscale("symlog"); ax[2].set_xlabel("eigenseries mode count M")
ax[2].set_ylabel("$A_{33}$ [kg] (symlog)")
ax[2].set_title("Same case, 1280 panels, h/R=5, ka=1")
ax[2].grid(alpha=.3, axis="y")
fig.tight_layout()
buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130); png = buf.getvalue()
Path("docs/figures").mkdir(parents=True, exist_ok=True)
Path("docs/figures/m21082_finite_depth_xval.png").write_bytes(png)

tbl = "\n".join(
    f"| {r['depth_ratio']:g} | {r['ka']:g} | {r['tbem_am_heave']:.1f} | {r['cpt_am_heave']:.1f} | "
    f"{100*r['am_rel_error']:.2f} | {r['tbem_b_heave']:.1f} | {r['cpt_b_heave']:.1f} | {100*r['b_rel_error']:.2f} |"
    for r in sorted(rows, key=lambda r: (r["depth_ratio"], r["ka"])))
gates = "\n".join(f"| {k} | {v['threshold']*100:.0f}% | {v['max_error']*100:.2f}% | n={v['n']} | "
                  f"{'PASS' if v['passed'] else 'FAIL'} |" for k, v in g.items())
A = {k: round(v["added_mass_heave"], 1) for k, v in ana.items()}
md = f"""# 有限水深 BEM 交叉验证：TensorBEM vs capytaine（m21082）

**锚点** `XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN` · **仓** TensorBEM · **分支** `agent/m21082`

## 1 案例介绍

完全淹没球体（半径 a=1 m，球心 z=−2 m，最低点 z=−3 m）在常水深 h 中的辐射/绕射。
水深台阶 h/R = 3/5/12/20 与无限水深；频率 ka = 0.3/0.6/1.0/1.5/2.0（ρ=1025 kg/m³，g=9.80665 m/s²）。
球体不穿自由面、不含水线面，因此两套代码都不需要 lid/CHIEF，是纯粹的三维有限水深
Green 函数检验。**几何先验证**：h/R=3 时球底 z=−3 与海底 z=−3 恰好相切（零间隙），
属退化边界，已在 `xval_results.json` 的 `seabed_clearance_m` 中标出，未用于结论。

## 2 方法

| 侧 | 路线 | 网格 |
|---|---|---|
| TensorBEM | John 本征级数 + Rankine 奇性分裂 + 直接配点（`finite_depth_solver3d`） | icosphere 剖分 2/3 = 320/1280 面板 |
| capytaine 3.1.dev0 | 波数积分 + 色散求根 + 间接单层（`BEMSolver`） | (8,16)/(12,24)/(16,32)/(24,48) = 128/288/512/1152 面 |
| 解析锚 | Havelock 多极子闭式解 `analytic.submerged_sphere_wave_multipole`（无限水深） | — |

生产者 `run_cases.py`（按案冻结 + 可分片），装配器 `assemble_xval.py`，测试
`tests/test_bem_finite_depth_capytaine_xval.py`，图 `docs/figures/m21082_finite_depth_xval.png`。
门禁在跑之前写死：G1 A33 跨代码 <10%，G2 B33 <15%，G3 capytaine 自收敛 <5%，
G4 TensorBEM 自收敛 <5%，G5 深水极限 <5%。

## 3 计算结果（冻结于 `xval_results.json`，程序化注入）

| h/R | ka | A33 TBEM | A33 capytaine | 误差% | B33 TBEM | B33 capytaine | 误差% |
|---|---|---|---|---|---|---|---|
{tbl}

无限水深解析锚 A33(kg, ρ∇≈4292.5)：ka=0.3 → {A['0.3']}，0.6 → {A['0.6']}，1.0 → {A['1.0']}，
1.5 → {A['1.5']}，2.0 → {A['2.0']}（A33' = A33/(ρ∇) → 0.557 于 ka=0.3，与 Lamb 0.5 的深淹没极限一致）。

| 门禁 | 阈值 | 实测最大误差 | 样本 | 判定 |
|---|---|---|---|---|
{gates}

## 4 图

![](fig1)

左：A33 vs ka（实线 TensorBEM，虚线 capytaine，点线解析解）；中：跨代码相对误差 vs ka；
右：同一物理算例下 A33 随本征级数截断数 M 的爆炸（symlog）。

## 5 注意点（踩过的坑）

1. **本征级数截断不收敛**（阻断级）：同一算例（1280 面板，h/R=5，ka=1）A33 在 M=48/64/96 下
   分别给出 9786 / 463427 / −2175 kg —— 正确量级应为 2000–3000 kg。缓存中 44 个 TensorBEM 案
   有 20 个抛 `numpy.linalg.LinAlgError: Eigenvalues did not converge`。这就是 G1/G5 失败的主因，
   而不是网格不够密。
2. **误差随水深变号**：h/R=3/5/20 TensorBEM 偏低 5–8%，h/R=12 却偏高 9–12%（图中间）。
   纯粹的面板面积亏损不可能变号，说明存在与水深相关的本征模截断误差。h/R=12 的 A33 反而高于
   h/R=20，与"水深越大越接近无限水深"的物理相反。
3. **旧的 `run_xval.py` 从未真正跑通**：capytaine 3.x 的 `result.added_mass` 是
   `{{radiating: {{influenced: value}}}}` 嵌套 dict，原代码 `float(result.added_mass)` 直接 TypeError；
   深水支路把 `depth=np.inf` 传给只接受有限水深的 `finite_depth_solver3d`（`_positive` 抛错）。
   已由 `run_cases.py` 修好（深水改走 `free_surface_solver3d.solve_radiation_3d`）。
4. **文献/文档断言与实测不符**：原 docstring 称 h/R=3 时"球底比最低点低 1 m"（实际零间隙）；
   称 icosphere 剖分 2/3/4 = 160/640/2560 面板（实际 320/1280/5120）。
5. 适用范围：结论只覆盖 320 面板（h/R=3/12/20 全频段、h/R=5 仅 ka=0.3）与 ka=1 的 1280 面板，
   1280 面板全频段单算 630–1500 s，未在本次预算内完成（集群队列 15 min 以上排队）。

## 6 误差判定（逐项，显式回答是否 <3%）

| 关键量 | 基准 | 误差 | <3%? |
|---|---|---|---|
| A33（h/R=20, ka=1） | 解析 1923.8 kg | −5.9% | **否** |
| A33（h/R=20, ka=1） | capytaine 1973.5 kg | −8.3% | **否** |
| A33（h/R=12, ka=1） | capytaine 1975.2 kg | +9.2% | **否** |
| A33（h/R=12, ka=0.3） | capytaine 2460.4 kg | +12.2% | **否** |
| B33（h/R=12, ka=0.3） | capytaine 312.3 kg | +34.0% | **否** |
| capytaine 自收敛（24x48 vs 16x32） | 自身 | 0.85% | 是 |

**honest FAIL**：G1/G2/G5 全部失败，G4 无数据未评。TensorBEM 三维有限水深求解器在本次交叉验证下
不能声明 <10% 精度；根因指向本征级数截断而非面板密度（capytaine 自收敛 0.85% 排除了基准侧问题）。
"""
report = {
    "title": "有限水深 BEM 交叉验证：TensorBEM vs capytaine + Havelock 解析锚（m21082, honest FAIL）",
    "summary": "锚点 XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN：16 组 A33/B33 跨代码比对、1 个无限水深解析锚、"
               "冻结工件与回归锁测试。结果 G1/G2/G5 失败（A33 5–12%、B33 最差 34%），根因定位为本征级数"
               "截断不收敛（同算例 M=48/64/96 → 9786/463427/−2175 kg）与 20/44 例 LinAlgError。",
    "scenario_key": "bem_finite_depth_capytaine_xval",
    "mission_id": 21082,
    "tags": ["TensorBEM", "有限水深", "交叉验证", "capytaine", "honest-FAIL", "XVAL"],
    "markdown": md,
    "figures": [{"name": "m21082_finite_depth_xval.png",
                 "caption": "左：A33 vs ka（TensorBEM/capytaine/解析）；中：跨代码相对误差（h/R=12 变号）；"
                            "右：A33 随本征级数截断数爆炸",
                 "png_base64": base64.b64encode(png).decode()}],
}
Path("/tmp/report_m21082.json").write_text(json.dumps(report))
print("figure+report json ready", len(png), "bytes png")
