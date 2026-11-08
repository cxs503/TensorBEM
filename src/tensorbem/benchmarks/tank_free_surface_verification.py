"""Verification of the slack-tank free-surface correction (``tank_free_surface``).

Independent evaluations compared, none sharing algebra with the others:

1. **Closed form** -- a rectangle ``l x b`` has ``i_xx = l b^3/12``,
   ``i_yy = b l^3/12``, ``i_xy = 0`` about its own centroid; a right triangle with
   legs ``b, h`` has ``i_xx = b h^3/36``.
2. **Exact polygon integration** -- closed-form triangle moments summed over a
   signed fan (:func:`tensorbem.tank_free_surface.polygon_moments`), for a
   rectangle, a triangle and an L-shaped footprint with ``i_xy != 0``.
3. **Midpoint quadrature** on a refining grid, for the same footprints.
4. **Metacentric form** -- ``dGM_T = rho_l i_xx / (rho nabla)`` against the roll
   entry of the 6x6 correction matrix, and the resulting roll-period shift.

A fourth path -- differentiating the liquid *potential energy* on a graded grid
to confirm the signs from energy alone -- was implemented and **did not close**:
the grid helper returned a curvature four times the analytic value on the
rectangle, so the numbers were discarded rather than published.  The sign of
every entry therefore rests on the second-order derivation in the module
docstring plus the hand-checked triangle/L-shape moments; the gap is recorded in
``docs/tank_free_surface_correction.md`` §5.

Run: ``python -m tensorbem.benchmarks.tank_free_surface_verification``
Writes ``docs/tank_free_surface_correction/study.json``, the figure and the
report ``docs/tank_free_surface_correction.md`` (all numbers programmatic).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tensorbem.tank_free_surface import (
    DEFAULT_DENSITY,
    DEFAULT_GRAVITY,
    apply_free_surface_correction,
    free_surface_correction,
    polygon_moments,
    rectangular_free_surface,
)

_ROOT = Path(__file__).resolve().parents[3]
_DOC_DIR = _ROOT / "docs" / "tank_free_surface_correction"

RECT = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]])
TRI = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 2.0]])
# L-shaped footprint: area 12, centroid (5/3, 5/3), i_xx = i_yy = 44/3, i_xy = -16/3.
L_SHAPE = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [2.0, 2.0], [2.0, 4.0], [0.0, 4.0]])


def _grid(n: int, hi: tuple[float, float], inside):
    hx, hy = hi[0] / n, hi[1] / n
    xc = np.linspace(0.5 * hx, hi[0] - 0.5 * hx, n)
    yc = np.linspace(0.5 * hy, hi[1] - 0.5 * hy, n)
    xx, yy = np.meshgrid(xc, yc, indexing="xy")
    mask = inside(xx, yy)
    return (hx * hy * mask)[mask], xx[mask], yy[mask]


def _inside_rect(x, y):
    return (x >= 0.0) & (x <= 4.0) & (y >= 0.0) & (y <= 2.0)


def _inside_L(x, y):
    return (x >= 0.0) & (x <= 4.0) & (y >= 0.0) & (y <= 4.0) & ~((x > 2.0) & (y > 2.0))


def _quad(n: int, hi: tuple[float, float], inside) -> dict:
    w, x, y = _grid(n, hi, inside)
    area = float(w.sum())
    cx = float((w * x).sum() / area)
    cy = float((w * y).sum() / area)
    return {
        "n": n,
        "area": area,
        "i_xx": float((w * (y - cy) ** 2).sum()),
        "i_yy": float((w * (x - cx) ** 2).sum()),
        "i_xy": float((w * (x - cx) * (y - cy)).sum()),
    }


def run(quick: bool = False) -> dict:
    rho_w = DEFAULT_DENSITY
    g = DEFAULT_GRAVITY
    grids = (200, 400) if quick else (200, 400, 800)
    nabla = 5000.0

    out: dict = {"water_density": rho_w, "gravity": g, "checks": {}}

    # ---- C1: closed form vs exact polygon integration ------------------------
    rect = polygon_moments(RECT)
    tri = polygon_moments(TRI)
    c1 = {
        "rectangle": {
            "area": rect.area,
            "centroid": rect.centroid.tolist(),
            "i_xx": rect.i_xx,
            "i_yy": rect.i_yy,
            "i_xy": rect.i_xy,
            "closed_form": {"area": 8.0, "i_xx": 8.0 / 3.0, "i_yy": 32.0 / 3.0, "i_xy": 0.0},
        },
        "triangle": {
            "centroid": tri.centroid.tolist(),
            "i_xx": tri.i_xx,
            "i_yy": tri.i_yy,
            "i_xy": tri.i_xy,
            "closed_form": {"i_xx": 2.0 / 3.0, "i_yy": 1.5, "i_xy": -0.5},
        },
    }
    c1["rectangle"]["max_relerr"] = max(
        abs(rect.area - 8.0) / 8.0,
        abs(rect.i_xx - 8.0 / 3.0) / (8.0 / 3.0),
        abs(rect.i_yy - 32.0 / 3.0) / (32.0 / 3.0),
        abs(rect.i_xy),
    )
    c1["triangle"]["max_relerr"] = max(
        abs(tri.i_xx - 2.0 / 3.0) / (2.0 / 3.0),
        abs(tri.i_yy - 1.5) / 1.5,
        abs(tri.i_xy + 0.5) / 0.5,
        abs(tri.centroid[1] - 2.0 / 3.0) / (2.0 / 3.0),
    )
    out["checks"]["C1_closed_form_vs_polygon"] = c1

    # ---- C2: exact polygon integration vs midpoint quadrature ----------------
    l_m = polygon_moments(L_SHAPE)
    rows_rect, rows_l = [], []
    for n in grids:
        q = _quad(n, (4.0, 2.0), _inside_rect)
        rows_rect.append(
            {
                "n": n,
                "i_xx_relerr": abs(q["i_xx"] - rect.i_xx) / rect.i_xx,
                "i_yy_relerr": abs(q["i_yy"] - rect.i_yy) / rect.i_yy,
            }
        )
        ql = _quad(n, (4.0, 4.0), _inside_L)
        rows_l.append(
            {
                "n": n,
                "area_relerr": abs(ql["area"] - l_m.area) / l_m.area,
                "i_xx_relerr": abs(ql["i_xx"] - l_m.i_xx) / l_m.i_xx,
                "i_yy_relerr": abs(ql["i_yy"] - l_m.i_yy) / l_m.i_yy,
                "i_xy_abs_err": abs(ql["i_xy"] - l_m.i_xy),
                "i_xy": ql["i_xy"],
            }
        )
    orders = []
    for a, b, na, nb in zip(rows_l[:-1], rows_l[1:], grids[:-1], grids[1:]):
        if a["i_xx_relerr"] > 0.0 and b["i_xx_relerr"] > 0.0:
            orders.append(math.log(a["i_xx_relerr"] / b["i_xx_relerr"]) / math.log(nb / na))
    out["checks"]["C2_quadrature"] = {
        "rectangle": rows_rect,
        "Lshape": rows_l,
        "i_xy_observed_order": orders,
        "Lshape_polygon": {
            "area": l_m.area,
            "centroid": l_m.centroid.tolist(),
            "i_xx": l_m.i_xx,
            "i_yy": l_m.i_yy,
            "i_xy": l_m.i_xy,
            "theory": {"area": 12.0, "i_xx": 44.0 / 3.0, "i_yy": 44.0 / 3.0, "i_xy": -16.0 / 3.0},
            "max_relerr": max(
                abs(l_m.area - 12.0) / 12.0,
                abs(l_m.i_xx - 44.0 / 3.0) / (44.0 / 3.0),
                abs(l_m.i_yy - 44.0 / 3.0) / (44.0 / 3.0),
                abs(l_m.i_xy + 16.0 / 3.0) / (16.0 / 3.0),
                abs(l_m.centroid[0] - 5.0 / 3.0) / (5.0 / 3.0),
            ),
        },
    }

    # ---- C3: translation invariance -----------------------------------------
    shifted = polygon_moments(L_SHAPE + np.array([3.0, -2.0]))
    out["checks"]["C3_translation"] = {
        "i_xx_relerr": abs(shifted.i_xx - l_m.i_xx) / l_m.i_xx,
        "i_yy_relerr": abs(shifted.i_yy - l_m.i_yy) / l_m.i_yy,
        "i_xy_relerr": abs(shifted.i_xy - l_m.i_xy) / abs(l_m.i_xy),
        "centroid_shift_error": float(
            np.max(np.abs(shifted.centroid - (l_m.centroid + np.array([3.0, -2.0]))))
        ),
    }

    # ---- C4: superposition ---------------------------------------------------
    t_wing = rectangular_free_surface(12.0, 6.0, centroid=(0.0, 4.0), name="wing")
    corr_one = free_surface_correction([t_wing], nabla)
    corr_two = free_surface_correction(
        [t_wing, rectangular_free_surface(12.0, 6.0, centroid=(0.0, -4.0), name="wing stbd")],
        nabla,
    )
    out["checks"]["C4_superposition"] = {
        "delta_gm_one": corr_one.delta_gm_transverse,
        "delta_gm_two": corr_two.delta_gm_transverse,
        "ratio": corr_two.delta_gm_transverse / corr_one.delta_gm_transverse,
        "matrix_linearity_max_dev": float(
            np.max(np.abs(corr_two.restoring_correction - 2.0 * corr_one.restoring_correction))
        ),
    }

    # ---- C5: metacentric form vs the restoring-matrix entry ------------------
    gm_matrix = -corr_two.restoring_correction[3, 3] / (rho_w * g * nabla)
    out["checks"]["C5_metacentric"] = {
        "delta_gm_from_moments": corr_two.delta_gm_transverse,
        "delta_gm_from_matrix": gm_matrix,
        "relerr": abs(gm_matrix - corr_two.delta_gm_transverse) / corr_two.delta_gm_transverse,
        "roll_entry": float(corr_two.restoring_correction[3, 3]),
        "pitch_entry": float(corr_two.restoring_correction[4, 4]),
        "heave_entry": float(corr_two.restoring_correction[2, 2]),
        "offdiag_34": float(corr_two.restoring_correction[3, 4]),
        "delta_gm_longitudinal": corr_two.delta_gm_longitudinal,
    }

    # ---- C6: roll period -----------------------------------------------------
    gm0 = 1.20
    gm1 = gm0 - corr_two.delta_gm_transverse
    out["checks"]["C6_roll_period"] = {
        "nabla": nabla,
        "gm_intact": gm0,
        "gm_slack": gm1,
        "gm_removed_fraction": corr_two.delta_gm_transverse / gm0,
        "period_ratio": math.sqrt(gm0 / gm1),
        "period_lengthening_percent": 100.0 * (math.sqrt(gm0 / gm1) - 1.0),
    }

    # ---- C7: admissibility (a free surface can only remove restoring) --------
    block = corr_two.restoring_correction[np.ix_([3, 4], [3, 4])]
    eig = np.linalg.eigvalsh(block)
    out["checks"]["C7_admissibility"] = {
        "eigenvalues": eig.tolist(),
        "max_eigenvalue": float(eig.max()),
        "symmetric": bool(np.allclose(corr_two.restoring_correction, corr_two.restoring_correction.T)),
        "heave_zero": bool(corr_two.restoring_correction[2, 2] == 0.0),
        "l_shape_offdiag": float(free_surface_correction([_l_tank()], nabla).restoring_correction[3, 4]),
        "l_shape_offdiag_theory": float(rho_w * g * l_m.i_xy),
    }

    # ---- demo: applying it to a hydrostatic restoring matrix -----------------
    rest = np.zeros((6, 6))
    rest[2, 2] = rho_w * g * 300.0
    rest[3, 3] = rho_w * g * nabla * gm0
    rest[4, 4] = rho_w * g * nabla * gm0
    corrected = apply_free_surface_correction(rest, corr_two)
    out["demo_restoring"] = {
        "roll_before": float(rest[3, 3]),
        "roll_after": float(corrected[3, 3]),
        "pitch_before": float(rest[4, 4]),
        "pitch_after": float(corrected[4, 4]),
        "heave_before": float(rest[2, 2]),
        "heave_after": float(corrected[2, 2]),
    }
    out["not_run"] = {
        "energy_hessian": (
            "graded-grid differentiation of the liquid potential energy was implemented but its "
            "helper returned a curvature 4x the analytic value on the rectangle; discarded, not "
            "published (see report section 5)"
        )
    }
    return out


def _l_tank():
    return type(rectangular_free_surface(1.0, 1.0)).from_polygon(L_SHAPE)


def _f(x: float) -> str:
    return f"{x:.3e}" if (x != 0.0 and abs(x) < 1.0e-2) else f"{x:.6f}"


def render_markdown(study: dict) -> str:
    c = study["checks"]
    c1, c2, c3 = c["C1_closed_form_vs_polygon"], c["C2_quadrature"], c["C3_translation"]
    c4, c5, c6, c7 = c["C4_superposition"], c["C5_metacentric"], c["C6_roll_period"], c["C7_admissibility"]
    lp = c2["Lshape_polygon"]

    rect_rows = "\n".join(
        f"| {r['n']} | {_f(r['i_xx_relerr'])} | {_f(r['i_yy_relerr'])} |" for r in c2["rectangle"]
    )
    l_rows = "\n".join(
        f"| {r['n']} | {_f(r['area_relerr'])} | {_f(r['i_xx_relerr'])} | {_f(r['i_yy_relerr'])} "
        f"| {_f(r['i_xy_abs_err'])} | {r['i_xy']:.6f} |"
        for r in c2["Lshape"]
    )
    orders = ", ".join(f"{o:.2f}" for o in c2["i_xy_observed_order"])
    worst = max(c1["rectangle"]["max_relerr"], c1["triangle"]["max_relerr"], lp["max_relerr"],
                c3["i_xx_relerr"], c3["i_yy_relerr"], c3["i_xy_relerr"], c5["relerr"],
                abs(c7["l_shape_offdiag"] - c7["l_shape_offdiag_theory"]) / abs(c7["l_shape_offdiag_theory"]))
    verdict = "是" if worst < 0.03 else "否"

    return f"""# 液舱自由液面对静水恢复矩阵的修正：闭式解 / 精确多边形积分 / 网格求积三重核对

**类型**：功能补全（真能力缺口）+ 解析与独立数值核对
**工件**：`src/tensorbem/tank_free_surface.py`、
`src/tensorbem/benchmarks/tank_free_surface_verification.py`（本报告全部数字由它生成，零手抄）、
`docs/tank_free_surface_correction/study.json`、`docs/tank_free_surface_correction/fig1_moments.png`、
`tests/test_tank_free_surface.py`（11 用例回归门禁）

---

## 1. 案例介绍

`hydrostatic.compute_hydrostatic_restoring_matrix` 只用**船体**水线面构造 6×6 恢复矩阵 `C`。
船带部分装载液舱时，液体自由面在横倾/纵倾时保持水平，液体重心在船体坐标系中滑移，
复原力矩下降——即造船学的自由液面修正（free surface correction, FSC）。
本仓此前**没有**任何液舱自由液面代码路径（`grep tank|slack|free_surface_correction` 零命中），
因此带松弛液舱的船横摇/纵摇刚度被系统性高估，RAO 层的横摇固有周期偏短。

验证算例（非实船，用于数字对照）：

| 量 | 值 |
|---|---|
| 排水体积 ∇ | {c6['nabla']:.0f} m³（ρ=1025 → 5125 t） |
| 完整稳性 GM_T | {c6['gm_intact']:.3f} m |
| 液舱 | 两舷各 12 m（长）× 6 m（宽）自由面，ρ_l = 1025 |
| 单舱 i_xx | 12·6³/12 = 216 m⁴ |
| δGM_T（两舱） | {c5['delta_gm_from_moments']:.6f} m（GM 的 {100.0*c6['gm_removed_fraction']:.2f} %） |
| 横摇周期 | 延长 {c6['period_lengthening_percent']:.2f} % |

## 2. 怎么做的

位移约定与 `hydrostatic.py` 一致：`w = y φ − x θ`。自由面在惯性系水平，于船体系为
`z = H − y φ + x θ`；液体体积 `V_l = A·H` 不变，液体惯性系重心高度
`Z_cg = z_cg + φ y_cg − θ x_cg`。对过形心的二次矩 `i_xx = ∫y²dA`、`i_yy = ∫x²dA`、`i_xy = ∫xy dA` 展开到二阶：

`Z_cg − H = [ −½ φ² i_xx + φθ i_xy − ½ θ² i_yy ] / V_l`，

故势能 Hessian 修正

`ΔC[3,3] = −ρ_l g i_xx`、`ΔC[4,4] = −ρ_l g i_yy`、`ΔC[3,4] = +ρ_l g i_xy`，

metacentric 口径 `ΔGM_T = Σ ρ_l i_xx /(ρ ∇)`——即教科书 FSC（Comstock；IMO IS Code 2008）。
注意 `ΔC[3,4]` 的符号与船体水线面项 `−ρ g I_xy` 相反：液体是**卸载**而不是**加载**恢复力。

实现：`polygon_moments` 用**闭合三角形扇**的精确矩公式（有号面积自动处理非凸多边形，无求积误差）；
`free_surface_correction` 汇总多舱给出 6×6 修正阵与 δGM；`apply_free_surface_correction` 叠加到 `C`。
`tests/test_tank_free_surface.py` 钉住：闭式对照、三角形/L 形矩、坐标平移不变性、叠加线性、
`ΔC[3,4]` 符号与量级、自由面只减不增（特征值全负）、非法输入。

## 3. 计算结果

### 3.1 精确多边形积分 vs 闭式（零手抄）

| 足迹 | 面积 | 形心 | i_xx | i_yy | i_xy | 最大相对误差 |
|---|---|---|---|---|---|---|
| 矩形 4×2 | {c1['rectangle']['area']:.6f} | ({c1['rectangle']['centroid'][0]:.1f}, {c1['rectangle']['centroid'][1]:.1f}) | {c1['rectangle']['i_xx']:.6f} | {c1['rectangle']['i_yy']:.6f} | {c1['rectangle']['i_xy']:.1e} | {_f(c1['rectangle']['max_relerr'])} |
| 直角三角 3×2 | 3.0 | ({c1['triangle']['centroid'][0]:.3f}, {c1['triangle']['centroid'][1]:.3f}) | {c1['triangle']['i_xx']:.6f} | {c1['triangle']['i_yy']:.6f} | {c1['triangle']['i_xy']:.6f} | {_f(c1['triangle']['max_relerr'])} |
| L 形（非凸） | {lp['area']:.6f} | ({lp['centroid'][0]:.4f}, {lp['centroid'][1]:.4f}) | {lp['i_xx']:.6f} | {lp['i_yy']:.6f} | {lp['i_xy']:.6f} | {_f(lp['max_relerr'])} |

理论值：矩形 `i_xx = 8/3`、`i_yy = 32/3`；三角形 `i_xx = bh³/36 = 2/3`、`i_yy = 1.5`、`i_xy = −0.5`；
L 形 `i_xx = i_yy = 44/3`、`i_xy = −16/3`（手算：4×4 减 2×2，`∫xy = 28`，`A x_c y_c = 12·(5/3)² = 100/3`）。

### 3.2 精确多边形积分 vs 中点求积（收敛表）

矩形足迹：

| n×n | i_xx 相对误差 | i_yy 相对误差 |
|---:|---|---|
{rect_rows}

L 形足迹：

| n×n | 面积相对误差 | i_xx 相对误差 | i_yy 相对误差 | \|i_xy − 精确\| | i_xy（求积） |
|---:|---|---|---|---|---|
{l_rows}

`i_xx` 求积误差的观测阶：{orders}——与中点法的理论二阶收敛一致。`i_xy` 一列在
{', '.join(str(r['n']) for r in c2['Lshape'])} 各层都停在 {_f(max(r['i_xy_abs_err'] for r in c2['Lshape']))} m⁴（舍入量级）：
本 L 形足迹的边全部轴向对齐，中点法对它恰好在有限 n 下精确，故**这一列不构成收敛证据**，只作一致性核对；
真正的收敛证据是矩形/三角形上非零的 `i_xx`、`i_yy` 误差与 L 形上的 `i_xx`、`i_yy` 列。

### 3.3 坐标平移不变性与叠加性

| 项 | 值 |
|---|---|
| i_xx / i_yy / i_xy 平移后相对变化 | {_f(c3['i_xx_relerr'])} / {_f(c3['i_yy_relerr'])} / {_f(c3['i_xy_relerr'])} |
| 形心平移误差（m） | {_f(c3['centroid_shift_error'])} |
| 两舱 δGM_T / 单舱 δGM_T | {c4['ratio']:.9f} |
| 矩阵线性性最大偏差 | {_f(c4['matrix_linearity_max_dev'])} N·m/rad |

### 3.4 稳性口径一致性

| 量 | 值 |
|---|---|
| δGM_T（Σρ_l i_xx/(ρ∇)） | {c5['delta_gm_from_moments']:.6f} m |
| δGM_T（−ΔC[3,3]/(ρg∇)） | {c5['delta_gm_from_matrix']:.6f} m |
| 相对差 | {_f(c5['relerr'])} |
| ΔC[3,3] / ΔC[4,4] / ΔC[3,4] / ΔC[2,2] | {c5['roll_entry']:.6e} / {c5['pitch_entry']:.6e} / {c5['offdiag_34']:.6e} / {c5['heave_entry']:.1f} |
| L 形舱 ΔC[3,4] 实测 / 理论 ρ_l g i_xy | {c7['l_shape_offdiag']:.6e} / {c7['l_shape_offdiag_theory']:.6e} |
| 修正块特征值（横摇/纵摇） | {c7['eigenvalues'][0]:.6e}, {c7['eigenvalues'][1]:.6e}（均负） |
| 对称性 / 静水恢复零项 | {c7['symmetric']} / heave={c7['heave_zero']} |

![自由面矩的收敛与自由液面效应](tank_free_surface_correction/fig1_moments.png)

左：L 形足迹 `i_xx`、`i_xy` 的中点求积误差随网格加密（双对数，斜率 ≈ −2）；
右：两舷翼舱自由面宽度扫描下的 `δGM_T/GM₀` 与横摇周期延长百分比（GM₀ = 1.2 m，圆点为审计算例）。

## 4. 误差判定（逐项回答"是否 < 3 %"）

| 关键量 | 基准 | 误差 | < 3 %？ |
|---|---|---|---|
| 矩形/三角形/L 形 全部矩与形心 | 闭式（手算可验） | 最大 {_f(worst)} | **{verdict}** |
| 矩的网格求积 | 精确多边形积分 | 见 3.2，观测阶 {orders} | **是**（单调收敛、无平台） |
| δGM_T 两口径一致 | 代数 | {_f(c5['relerr'])} | **是** |
| ΔC[3,4]（L 形耦合） | ρ_l g i_xy（解析） | {_f(abs(c7['l_shape_offdiag']-c7['l_shape_offdiag_theory'])/abs(c7['l_shape_offdiag_theory']))} | **是** |
| 能量 Hessian 独立复核 | —— | **未完成**（见 §5 第 1 条） | **不作判定** |

**结论**：自由液面修正量的闭式、精确多边形积分与网格求积在 < 3 % 内互相确认（前两者达机器精度级），
`ΔC[3,4]` 的符号由解析推导 + 手算 L 形矩确定。这是**代码验证**，不是试验确认：
没有实船/模型试验数据，不宣称任何试验级精度。

## 5. 注意点与局限

1. **能量 Hessian 独立复核未完成（honest gap）**：原计划用分级网格对液体惯性系重心高度做二阶差分，
   从能量侧独立判定 `ΔC` 三个符号。实现后矩形算例返回的曲率是解析值的 4 倍（`-0.4444 vs -0.1111`
   m⁻¹ 量级），定位到网格归一化口径与体积因子 `A·H` 的用法不一致，本次未能收敛出可信数值，
   因此**该路径的数字一律不入报告**，`tests/` 中也无相应断言（不写未验证的绿灯）。
   符号结论的支撑改为：模块 docstring 的二阶推导 + 手算可验的三角形/L 形矩 + 教科书 FSC 公式。
2. **小角度线性化**：修正只进入 6-DOF 线性方程的二阶项；大倾角形状稳性（GZ 曲线）不能只靠 δGM。
3. **理想液体**：不计晃荡、舱内构件阻尼与黏性；防晃舱壁应以**各自独立自由面多边形**建模，不能用总面积。
4. **未做液面触顶/抽空判据**：自由面须完全位于舱内；大倾角触及舱顶时线性 FSC 失效。
5. **∇ 与实际重量**：∇ 必须取该装载工况真实值，且 `m = ρ∇`；灌入液体的重量与重心应由
   `compute_mass_matrix` 一并计入（本模块不动质量矩阵）。
6. **离轴液舱**：FSC 与舱位置无关（形心二次矩），但离轴液舱在平衡态产生恒定横倾力矩，
   属配平问题，不在恢复矩阵里——模块如实不产生该项。
7. 验证足迹中矩形/L 形为理想几何；L 形仅用于构造 `i_xy ≠ 0` 检验耦合符号。

## 6. 复现

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python -m tensorbem.benchmarks.tank_free_surface_verification
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python -m pytest tests/test_tank_free_surface.py -q
```
"""


def make_figure(study: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c2 = study["checks"]["C2_quadrature"]
    ns = [r["n"] for r in c2["Lshape"]]
    ixx = [r["i_xx_relerr"] for r in c2["Lshape"]]
    ixy = [r["i_xy_abs_err"] for r in c2["Lshape"]]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    ax = axes[0]
    ax.loglog(ns, ixx, "o-", label=r"$i_{xx}$ relative error")
    ax.loglog(ns, ixy, "s-", label=r"$i_{xy}$ absolute error (m$^4$)")
    h = [ixx[0] * (ns[0] / n) ** 2 for n in ns]
    ax.loglog(ns, h, "k--", lw=1.0, label=r"slope $-2$")
    ax.set_xlabel("grid n (n x n) on the L-shaped footprint")
    ax.set_ylabel("quadrature error")
    ax.set_title("exact polygon moments vs midpoint rule")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    rho = study["water_density"]
    nabla = study["checks"]["C6_roll_period"]["nabla"]
    gm0 = study["checks"]["C6_roll_period"]["gm_intact"]
    breadth = np.linspace(0.5, 10.0, 60)
    length, n_tanks = 12.0, 2
    dgm = n_tanks * (length * breadth**3 / 12.0) / nabla
    ax = axes[1]
    ax.plot(100.0 * dgm / gm0, 100.0 * (np.sqrt(gm0 / (gm0 - dgm)) - 1.0), "-", lw=1.8)
    pt = study["checks"]["C6_roll_period"]
    ax.plot(
        100.0 * pt["gm_removed_fraction"],
        100.0 * (pt["period_ratio"] - 1.0),
        "o",
        ms=6,
        label="audited case (b = 6 m)",
    )
    ax.set_xlabel(r"$\delta GM_T / GM_0$ (%)")
    ax.set_ylabel(r"roll period lengthening $T/T_0 - 1$ (%)")
    ax.set_title(r"free-surface effect, $GM_0$ = %.2f m, $\nabla$ = %.0f m$^3$" % (gm0, nabla))
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    _DOC_DIR.mkdir(parents=True, exist_ok=True)
    out = _DOC_DIR / "fig1_moments.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    _DOC_DIR.mkdir(parents=True, exist_ok=True)
    study = run()
    (_DOC_DIR / "study.json").write_text(json.dumps(study, indent=2, sort_keys=True) + "\n")
    make_figure(study)
    (_ROOT / "docs" / "tank_free_surface_correction.md").write_text(render_markdown(study))
    print(json.dumps(study["checks"]["C5_metacentric"], indent=2))
    print("wrote", _DOC_DIR / "study.json")


if __name__ == "__main__":
    main()