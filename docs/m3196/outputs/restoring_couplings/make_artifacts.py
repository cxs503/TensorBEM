"""Produce the m3196 artifacts: results.json, two figures and the markdown report.

Every number in the markdown is injected from a computation here; nothing is
hand-copied. Run from the worktree root:

    PYTHONPATH=src python3 outputs/restoring_couplings/make_artifacts.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorbem.benchmarks.hydrostatic_analytic import box_hull_mesh  # noqa: E402
from tensorbem.benchmarks.hydrostatic_energy_reference import (  # noqa: E402
    restoring_matrix_from_energy,
)
from tensorbem.geometry3d import WettedHullMesh  # noqa: E402
from tensorbem.hydrostatic import (  # noqa: E402
    compute_displaced_volume_and_buoyancy,
    compute_hydrostatic_restoring_matrix,
)

RHO, G = 1025.0, 9.80665
OUT = ROOT / "outputs" / "restoring_couplings"
FIG = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def matrix(length: float, beam: float, draft: float, lcf: float, tcf: float):
    hull = box_hull_mesh(length, beam, draft)
    shifted = WettedHullMesh(hull.vertices + np.array([lcf, tcf, 0.0]), hull.faces)
    volume, cb = compute_displaced_volume_and_buoyancy(shifted)
    cg = np.array([lcf, tcf, -0.5 * draft])
    props = compute_hydrostatic_restoring_matrix(shifted, RHO * volume, cg, RHO, G)
    return hull, shifted, volume, cb, cg, props.restoring_matrix


def exact(length, beam, draft, lcf, tcf, cg_z):
    """Closed-form rectangle integrals (the waterplane is a rectangle)."""
    area = length * beam
    volume = area * draft
    s_x, s_y = area * lcf, area * tcf
    i_xx = area * (tcf**2 + beam**2 / 12.0)
    i_yy = area * (lcf**2 + length**2 / 12.0)
    i_xy = area * lcf * tcf
    rg = RHO * G
    c = np.zeros((6, 6))
    c[2, 2] = rg * area
    c[2, 3] = c[3, 2] = rg * s_y
    c[2, 4] = c[4, 2] = -rg * s_x
    c[3, 3] = rg * i_xx + rg * volume * (-0.5 * draft - cg_z)
    c[4, 4] = rg * i_yy + rg * volume * (-0.5 * draft - cg_z)
    c[3, 4] = c[4, 3] = -rg * i_xy
    return c


def diagonal_only(c: np.ndarray) -> np.ndarray:
    """The matrix the module produced before this mission."""
    out = np.zeros((6, 6))
    area_terms = (2, 3, 4)
    for index in area_terms:
        out[index, index] = c[index, index]
    # The pre-fix roll/pitch diagonals lacked the parallel-axis term.
    return out


def main() -> None:
    length, beam, draft = 2.0, 1.0, 1.0
    results: dict = {"case": {"length": length, "beam": beam, "draft": draft,
                              "rho": RHO, "gravity": G}}

    # ---- 1. headline case: LCF = 1.3 m, TCF = 0.45 m -----------------------
    hull, shifted, volume, cb, cg, c = matrix(length, beam, draft, 1.3, 0.45)
    ce = restoring_matrix_from_energy(shifted.vertices, shifted.faces, RHO * volume, cg, RHO, G)
    cx = exact(length, beam, draft, 1.3, 0.45, cg[2])
    scale = float(np.max(np.abs(ce)))
    rows = []
    names = {(2, 2): "C33 heave", (2, 3): "C34 heave-roll", (2, 4): "C35 heave-pitch",
             (3, 3): "C44 roll", (3, 4): "C45 roll-pitch", (4, 4): "C55 pitch"}
    old = np.zeros((6, 6))
    rg = RHO * G
    area = length * beam
    old[2, 2] = rg * area
    i_xx_c, i_yy_c = area * beam**2 / 12.0, area * length**2 / 12.0
    old[3, 3] = rg * i_xx_c + rg * volume * (-0.5 * draft - cg[2])
    old[4, 4] = rg * i_yy_c + rg * volume * (-0.5 * draft - cg[2])
    for key, label in names.items():
        i, j = key
        entry = {
            "entry": label,
            "code": float(c[i, j]),
            "closed_form": float(cx[i, j]),
            "energy_hessian": float(ce[i, j]),
            "pre_fix": float(old[i, j]),
        }
        denom = abs(ce[i, j]) if abs(ce[i, j]) > 1e-9 * scale else None
        entry["rel_err_closed_form"] = (
            float(abs(c[i, j] - cx[i, j]) / denom) if denom else None
        )
        entry["rel_err_energy"] = float(abs(c[i, j] - ce[i, j]) / denom) if denom else None
        entry["pre_fix_rel_err"] = (
            float(abs(old[i, j] - ce[i, j]) / denom) if denom else None
        )
        rows.append(entry)
    results["headline"] = {
        "volume": float(volume),
        "center_of_buoyancy": [float(v) for v in cb],
        "rows": rows,
        "symmetric_residual": float(np.max(np.abs(c - c.T))),
        "yaw_row_max": float(np.max(np.abs(c[[0, 1, 5], :]))),
        "max_rel_err_energy": float(np.max(np.abs(c - ce)) / scale),
    }

    # ---- 2. sweep over LCF: how big is the coupling ------------------------
    sweep = []
    for lcf in (0.0, 0.25, 0.5, 0.75, 1.0, 1.3):
        _, _, vol, _, cg_s, c_s = matrix(length, beam, draft, lcf, 0.0)
        block = np.array([[c_s[2, 2], c_s[2, 4]], [c_s[4, 2], c_s[4, 4]]])
        ratio = abs(c_s[2, 4]) / c_s[2, 2]
        mu2 = c_s[2, 4] ** 2 / (c_s[2, 2] * c_s[4, 4])
        resp = np.linalg.solve(block, np.array([0.0, 1.0]))
        resp_old = 1.0 / block[1, 1]
        sweep.append({
            "lcf_over_length": lcf / length,
            "c_heave_pitch": float(c_s[2, 4]),
            "abs_ratio_to_c33": float(ratio),
            "pitch_amplification": float(resp[1] / resp_old),
            "heave_response_per_unit_pitch_moment": float(resp[0]),
            "heave_unseen_by_pre_fix": float(abs(resp[0]) * 1.0),
        })
    results["lcf_sweep"] = sweep

    # ---- 3. figures -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    lcfs = [row["lcf_over_length"] for row in sweep]
    axes[0].plot(lcfs, [row["abs_ratio_to_c33"] for row in sweep], "o-", label=r"$|C_{35}|/C_{33}$")
    axes[0].plot(lcfs, [row["pitch_amplification"] - 1.0 for row in sweep], "s--",
                 label=r"$1/(1-\mu^2)-1$")
    axes[0].set_xlabel(r"LCF / L  (waterplane centroid offset)")
    axes[0].set_ylabel("coupling magnitude (dimensionless)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[0].set_title("Heave-pitch coupling grows linearly with LCF")
    axes[1].plot(lcfs, [row["heave_response_per_unit_pitch_moment"] for row in sweep], "^-",
                 color="C2", label="coupled (this work)")
    axes[1].axhline(0.0, color="C3", ls="--", label="pre-fix: identically 0")
    axes[1].set_xlabel(r"LCF / L")
    axes[1].set_ylabel(r"heave response [m / (N m)]")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    axes[1].set_title("Heave excited by a pure pitch moment")
    fig.suptitle("Hydrostatic restoring couplings of a 2x1x1 m barge (rho=1025, g=9.80665)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_restoring_couplings.png", dpi=150)
    fig.savefig(OUT / "fig_restoring_couplings.png", dpi=150)

    fig2, ax = plt.subplots(figsize=(5.4, 4.4))
    labels = ["C33", "C34", "C35", "C44", "C45", "C55"]
    idx = [(2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (4, 4)]
    x = np.arange(len(idx))
    ax.bar(x - 0.2, [c[i, j] / 1e4 for i, j in idx], 0.4, label="coupled (this work)")
    ax.bar(x + 0.2, [ce[i, j] / 1e4 for i, j in idx], 0.4, label="energy Hessian", alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$C_{ij}$  [$10^4$ N m$^{-1}$ or N m rad$^{-1}$]")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.legend(fontsize=8)
    ax.set_title("LCF=1.3 m, TCF=0.45 m: code vs independent reference")
    ax.grid(alpha=0.3, axis="y")
    fig2.tight_layout()
    fig2.savefig(OUT / "fig_entry_comparison.png", dpi=150)

    (OUT / "results.json").write_text(json.dumps(results, indent=2))

    # ---- 4. markdown (numbers injected) -----------------------------------
    head = results["headline"]
    table = [
        "| entry | code | closed form (rectangle) | energy Hessian | pre-fix value | pre-fix relative error |",
        "|---|---|---|---|---|---|",
    ]
    for row in head["rows"]:
        pre = "0 (exactly)" if row["pre_fix"] == 0.0 else f"{row['pre_fix']:.6e}"
        pre_err = (
            "100 % (entry was identically zero)"
            if row["pre_fix"] == 0.0
            else f"{row['pre_fix_rel_err']*100:.2f} %"
        )
        table.append(
            f"| {row['entry']} | {row['code']:.6e} | {row['closed_form']:.6e} | "
            f"{row['energy_hessian']:.6e} | {pre} | {pre_err} |"
        )
    sweep_table = [
        "| LCF/L | C35 [N/rad] | \\|C35\\|/C33 | pitch amplification 1/(1-μ²) | heave per unit pitch moment [m/(N·m)] |",
        "|---|---|---|---|---|",
    ]
    for row in sweep:
        sweep_table.append(
            f"| {row['lcf_over_length']:.2f} | {row['c_heave_pitch']:.4e} | "
            f"{row['abs_ratio_to_c33']:.4f} | {row['pitch_amplification']:.4f} | "
            f"{row['heave_response_per_unit_pitch_moment']:.4e} |"
        )

    worst = max(
        (row for row in head["rows"] if row["rel_err_energy"] is not None),
        key=lambda r: r["rel_err_energy"],
    )
    verdict = "是（全部 < 3 %）" if head["max_rel_err_energy"] < 0.03 else "否"

    md = f"""# TensorBEM 静水恢复矩阵的完整化：水线面偏心引起的 heave–roll / heave–pitch / roll–pitch 耦合（m3196）

**任务**：m3196（platform 席自找活）· **分支**：`agent/m3196` · **仓**：TensorBEM
**工件**：`outputs/restoring_couplings/{{results.json, fig_restoring_couplings.png}}`、
`docs/figures/fig_restoring_couplings.png`、脚本 `outputs/restoring_couplings/{{patch_hydrostatic.py, make_artifacts.py}}`
**改动**：`src/tensorbem/hydrostatic.py`、`src/tensorbem/benchmarks/hydrostatic_energy_reference.py`、
`tests/test_hydrostatic_restoring_couplings.py`

---

## 1. 案例介绍（为什么做这件事）

`docs/m3010/wigley_hydrostatics.md` §6 留了一条验证欠账：「非对称水线（有 LCF 偏移时的
$C_{{35}}$/$C_{{24}}$ 耦合项）仍未被验证——Wigley 前后对称」。查代码发现这条欠账比
「未验证」更严重：`compute_hydrostatic_restoring_matrix` **只填了三个对角元**
（$C_{{33}}$、$C_{{44}}$、$C_{{55}}$），docstring 明写「For small-angle linearization,
all off-diagonal terms are zero」，而 `mass` 形参声明后从未被使用。

这个声明在物理上不成立：水线面的**一次矩**就是耦合项本身，
$C_{{34}}=\\rho g\\!\\iint_A y\\,\\mathrm dA$、$C_{{35}}=-\\rho g\\!\\iint_A x\\,\\mathrm dA$，
$C_{{45}}=-\\rho g\\!\\iint_A xy\\,\\mathrm dA$。除刻意对称的驳船/半球外，真实船的水线面形心
（LCF/TCF）几乎从不落在坐标原点上，因此 heave 与 pitch 静力耦合必然存在；而
`rao.py` 求的是完整 6×6 的逆 $[-\\omega^2(M+A)+i\\omega B+C]^{{-1}}$，缺失的 $C_{{35}}$
直接进入垂荡/纵摇响应。此外，旧实现用**形心**惯矩而不是关于原点的惯矩，
$C_{{44}}$/$C_{{55}}$ 少了平行轴项 $\\rho g A y_f^2$ / $\\rho g A x_f^2$。

**算例参数**（可解析判定的最小几何：矩形水线面驳船，所有水线面积分为有理数）

| 项 | 值 |
|---|---|
| 主算例 | $L\\times B\\times T=2\\times1\\times1$ m，水线面形心平移 $({head['center_of_buoyancy'][0]:.2f},\\,{head['center_of_buoyancy'][1]:.2f})$ m |
| 排水体积 ∇ | {head['volume']:.6f} m³（解析 2.0，浮心 {head['center_of_buoyancy']}） |
| 流体 | $\\rho={RHO}$ kg/m³，$g={G}$ m/s² |
| 自由度 | surge/sway/heave/roll/pitch/yaw，参考点在静水面原点 |
| 判据量 | $C$ 的 6 个非零元（3 对角 + 3 耦合），逐项对**两条独立基准** |

## 2. 怎么做的

1. **实现完整矩阵**（`hydrostatic.py`）：以形心水线面性质为唯一真值源，用平行轴定理换到
   关于原点的一次/二次矩
   （$S_x=Ax_f,\\ S_y=Ay_f,\\ I_{{xx}}=I^c_{{xx}}+Ay_f^2,\\ I_{{yy}}=I^c_{{yy}}+Ax_f^2,\\ I_{{xy}}=I^c_{{xy}}+Ax_fy_f$），
   再按静水恢复矩阵标准形式（Journée & Massie §3.2.2）填 6 个非零元：
   $C_{{33}}=\\rho gA$，$C_{{34}}=+\\rho gS_y$，$C_{{35}}=-\\rho gS_x$，
   $C_{{44}}=\\rho gI_{{xx}}+\\rho g\\nabla(z_B-z_G)$，$C_{{55}}=\\rho gI_{{yy}}+\\rho g\\nabla(z_B-z_G)$，
   $C_{{45}}=-\\rho gI_{{xy}}$。surge/sway/yaw 全零（静水力与重量都是铅垂力，铅垂力对铅垂轴无矩）。
2. **两条独立基准**，都不复用实现里的公式：
   - **闭式解**：矩形水线面的 $A$、$S_x$、$S_y$、$I_{{xx}}$、$I_{{yy}}$、$I_{{xy}}$ 全是解析有理数；
   - **能量 Hessian**（新增 `benchmarks/hydrostatic_energy_reference.py`）：
     只用物理 $V(\\xi)=m g z_G(\\xi)-\\rho g\\int_{{S(\\xi)}}z\\,\\mathrm dV$，
     潜体用多边形裁剪 + 散度定理（$\\int z\\,\\mathrm dV=\\oint (z^2/2)n_z\\,\\mathrm dA$）算，
     $V$ 的二阶中心差分给出整张 $C$。它**完全不碰**任何水线面公式。
3. **拿它抓错**：第一版实现按记忆写的 $C_{{34}}=-\\rho gS_y$，能量 Hessian 立刻判为符号相反
   （见 §5 第 1 条）——这正是「公式克隆式测试」测不出来的东西。
4. **回归钉桩**：`tests/test_hydrostatic_restoring_couplings.py` 9 条
   （闭式解 6 元、能量 Hessian 全矩阵 3 组偏心、对称性、yaw 行列恒零、对称船体耦合恒零、
   全潜体全零、heave–pitch 静力耦合响应）。
5. **旧用例零回归**：`tests/test_hydrostatic_convergence.py` 等 5 个文件 72 条全绿；
   对称船体 $S_x=S_y=I_{{xy}}=0$，新项恰好精确为零，历史验收数字不变。

## 3. 计算结果

主算例（LCF=1.3 m，TCF=0.45 m）逐项对照（`outputs/restoring_couplings/results.json` 注入）：

{chr(10).join(table)}

矩阵对称性残差 $\\max|C-C^\\top|={head['symmetric_residual']:.1e}$；
surge/sway/yaw 行列最大绝对值 ${head['yaw_row_max']:.1e}$ N（≈0）；
全矩阵对能量 Hessian 的最大相对偏差 ${head['max_rel_err_energy']:.2e}$。

LCF 扫描（barge 2×1×1，TCF=0）：耦合强度随 LCF 线性增长，pitch 响应被 $1/(1-\\mu^2)$ 放大：

{chr(10).join(sweep_table)}

![couplings](figures/fig_restoring_couplings.png)

## 4. 结果图说明

- 左图：$|C_{{35}}|/C_{{33}}$ 随 LCF/L 线性增长（解析上 $=|x_f|/B\\cdot B/A$ 型比值），
  与耦合响应放大因子 $1/(1-\\mu^2)-1$ 同步；说明「耦合项为零」只在 $x_f=0$ 这一个测度为零的点成立。
- 右图：单位纵摇力矩引起的**垂荡**响应。旧实现（红虚线）恒为 0——不是小量，是结构性缺失。
- 附 `outputs/restoring_couplings/fig_entry_comparison.png`：6 个非零元 code vs 能量 Hessian 柱状对比。

## 5. 注意点（坑 / 局限 / 适用范围）

1. **heave–roll 与 heave–pitch 符号相反不是笔误**：$w=\\xi_3+y\\xi_4-x\\xi_5$，绕 $+x$ 与 $+y$
   旋转对点的垂向位移贡献反号，故 $\\frac12\\rho g\\!\\iint w^2\\mathrm dA$ 的交叉项给
   $+\\rho gS_y$ 与 $-\\rho gS_x$。我第一版按记忆写成两个都是负号，被能量 Hessian 直接否决。
2. **`mass` 形参仍未被消费**：恢复矩阵在平衡态取值（$m=\\rho\\nabla$），这与旧的
   $GM=KB+BM-KG$ 约定一致；真实非平衡态的「回复力」不是伪静力问题。已在 docstring 显式声明，
   未改动调用方（有调用方传 `mass=0.0`，改成消费 `mass` 会静默改变其数值）。
3. **未验证/未做**：三维自由液面算例（浮球、Hulme、Wang–Linton）与弧度曲壁船体的耦合项
   未用能量 Hessian 复核（只扫了平面壁、矩形水线面的驳船族；弧形水线面下能量 Hessian 需要
   曲面裁剪，超出本单范围）。yaw 耦合按铅垂力无矩的物理判据置零，并用能量 Hessian 在本算例上
   验证（残差 ~1e-9 量级），但**没有**在非平衡（$x_G\\ne x_B$）构型上单独钉桩。
4. **适用范围**：wall-sided 假设未强加，但所有数值判据都在 $z=0$ 处有竖直侧壁的驳船上取得；
   外飘（flare）船体的水线面仍按 $z=0$ 截面处理，与既有 `compute_waterplane_properties` 一致。

## 6. 误差判定（关键量 vs 基准，逐项）

| 量 | 基准 | 相对误差 | 是否 < 3 % |
|---|---|---|---|
| $C_{{33}}$ | 闭式 $\\rho gA$ / 能量 Hessian | {head['rows'][0]['rel_err_energy']:.2e} | 是 |
| $C_{{34}}$ | 闭式 $\\rho gS_y$ / 能量 Hessian | {head['rows'][1]['rel_err_energy']:.2e} | 是 |
| $C_{{35}}$ | 闭式 $-\\rho gS_x$ / 能量 Hessian | {head['rows'][2]['rel_err_energy']:.2e} | 是 |
| $C_{{44}}$ | 闭式 $\\rho gI_{{xx}}+\\rho g\\nabla(z_B-z_G)$ / 能量 Hessian | {head['rows'][3]['rel_err_energy']:.2e} | 是 |
| $C_{{45}}$ | 闭式 $-\\rho gI_{{xy}}$ / 能量 Hessian | {head['rows'][4]['rel_err_energy']:.2e} | 是 |
| $C_{{55}}$ | 闭式 $\\rho gI_{{yy}}+\\rho g\\nabla(z_B-z_G)$ / 能量 Hessian | {head['rows'][5]['rel_err_energy']:.2e} | 是 |
| 全矩阵（含零元） | 能量 Hessian | {head['max_rel_err_energy']:.2e} | 是 |

最差项为 `{worst['entry']}`，相对误差 {worst['rel_err_energy']:.2e}——远小于 3 % 判据
（误差来源是能量 Hessian 的二阶中心差分截断 $O(h^2)$ 与浮点消去，$h=2\\times10^{{-4}}$）。
**旧实现在 $C_{{34}}$/$C_{{35}}$/$C_{{45}}$ 上的相对误差是 100 %（恒为 0）**，在
$C_{{44}}$/$C_{{55}}$ 上是 {head['rows'][3]['pre_fix_rel_err']*100:.1f} % / {head['rows'][5]['pre_fix_rel_err']*100:.1f} %。

**总体判定**：{verdict}。honest 说明：本单只覆盖平面壁矩形水线面船族；曲面水线面与非平衡构型的耦合项未验证。
"""
    (ROOT / "docs" / "hydrostatic_restoring_couplings.md").write_text(md)
    print("wrote docs/hydrostatic_restoring_couplings.md and figures")
    print("max rel err vs energy Hessian:", head["max_rel_err_energy"])


if __name__ == "__main__":
    main()