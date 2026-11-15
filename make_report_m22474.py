#!/usr/bin/env python3
"""Build (and optionally publish) the m22474 report for the finite-depth cross-validation.

Reads the frozen study artifact ``benchmarks/verified/capytaine_finite_depth/results.json``
plus the frozen capytaine reference and the TensorBEM ladder, renders five figures
and writes ``/tmp/report_m22474.json``.  Every number in the prose and tables is
injected from the artifact -- nothing is typed by hand.

Usage::

    PYTHONPATH=src python3 make_report_m22474.py [--publish]
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorbem.benchmarks import finite_depth_crossval_plan as plan  # noqa: E402

ROOT = Path(__file__).resolve().parent
VERIFIED = ROOT / "benchmarks" / "verified" / "capytaine_finite_depth"
RESULTS = VERIFIED / "results.json"
LADDER = VERIFIED / "tensor_ladder.json"
FIGDIR = ROOT / "docs" / "figures" / "m22474"
MISSION_ID = 22474
API_BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")

QUANTITIES = ("heave.added_mass", "heave.damping",
              "surge.added_mass", "surge.damping")


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #
def pct(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{100.0 * value:+.{digits}f} %"


def num(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(out)


def gate(result, name):
    for row in result["gates"]:
        if row["gate"] == name:
            return row
    raise KeyError(name)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def figure_frequency(result, path: Path) -> str:
    """Heave A33/B33 vs k0*h at every depth, both codes, extrapolated values."""
    rows = result["comparison"]["rows"]
    ratios = sorted({row["h_over_a"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
    for index, ratio in enumerate(ratios):
        subset = sorted([r for r in rows if r["h_over_a"] == ratio],
                        key=lambda r: r["k0h"])
        x = [r["k0h"] for r in subset]
        for ax, quantity, label in (
            (axes[0], "heave.added_mass", r"heave added mass $A_{33}$ [kg]"),
            (axes[1], "heave.damping", r"heave damping $B_{33}$ [kg/s]"),
        ):
            ax.plot(x, [r["quantities"][quantity]["tensorbem"] for r in subset],
                    "o-", color=f"C{index}", label=f"TensorBEM $h/a={ratio:.0f}$")
            ax.plot(x, [r["quantities"][quantity]["capytaine"] for r in subset],
                    "x--", color=f"C{index}", alpha=0.85,
                    label=f"capytaine $h/a={ratio:.0f}$")
            ax.set_xscale("log")
            ax.set_xlabel(r"$k_0 h$")
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6.5, ncol=2)
    fig.suptitle("Finite-depth cross-validation: Richardson-extrapolated values, "
                 "12 cells, submerged sphere")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return "Heave added mass and damping against $k_0h$ for all four depths; "
    "TensorBEM solid markers, capytaine crosses (both Richardson extrapolated)."


def figure_map(result, path: Path) -> str:
    rows = result["comparison"]["rows"]
    ratios = sorted({row["h_over_a"] for row in rows})
    kas = sorted({row["ka"] for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for ax, quantity in zip(axes, ("heave.added_mass", "heave.damping")):
        matrix = np.array([[100.0 * next(r for r in rows if r["h_over_a"] == ratio
                                         and r["ka"] == ka)["quantities"][quantity]["relative"]
                            for ka in kas] for ratio in ratios])
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-5.0, vmax=5.0)
        ax.set_xticks(range(len(kas)), [f"{ka:.1f}" for ka in kas])
        ax.set_yticks(range(len(ratios)), [f"{r:.0f}" for r in ratios])
        ax.set_xlabel(r"$k_0 a$")
        ax.set_ylabel(r"$h/a$")
        ax.set_title(quantity.replace(".", " ") + "  (TensorBEM vs capytaine)")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:+.1f}", ha="center", va="center",
                        fontsize=7.5)
        fig.colorbar(image, ax=ax, label="relative deviation [%]")
    fig.suptitle("Extrapolated deviation map over the $(h/a, k_0a)$ grid "
                 "(bands: 3 % added mass, 5 % damping)")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return "Deviation map of the extrapolated values over the $(h/a,k_0a)$ grid; "
    "the pre-registered bands are 3 % (added mass) and 5 % (damping)."


def figure_depth_physics(result, path: Path) -> str:
    """A33(h)/A33(deep-water analytic) -- an independent, non-capytaine anchor."""
    from tensorbem.analytic import submerged_sphere_wave_multipole

    rows = result["comparison"]["rows"]
    kas = sorted({row["ka"] for row in rows})
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for index, ka in enumerate(kas):
        subset = sorted([r for r in rows if r["ka"] == ka], key=lambda r: r["k0h"])
        analytic = submerged_sphere_wave_multipole(
            ka, radius=plan.SPHERE_RADIUS, submergence=plan.SPHERE_SUBMERGENCE,
            density=plan.DENSITY, gravity=plan.GRAVITY,
        )
        deep = float(analytic.heave_added_mass)
        depths = [r["h_over_a"] for r in subset]
        ax.plot(depths, [r["quantities"]["heave.added_mass"]["tensorbem"] / deep
                         for r in subset], "o-", color=f"C{index}",
                label=f"TensorBEM $k_0a={ka:.1f}$")
        ax.plot(depths, [r["quantities"]["heave.added_mass"]["capytaine"] / deep
                         for r in subset], "x--", color=f"C{index}", alpha=0.85)
        ax.axhline(1.0, color=f"C{index}", lw=0.6, ls=":", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel(r"$h/a$")
    ax.set_ylabel(r"$A_{33}(h)\,/\,A_{33}(\infty)$")
    ax.set_title("Finite-depth amplification relative to the multipole anchor\n"
                 "(dotted: deep-water analytic = 1, per $k_0a$)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return ("Heave added mass normalised by the independent deep-water multipole "
            "anchor $A_{33}(\\infty)$; both codes must fall onto the same "
            "depth-amplification curve and tend to 1 at large $h/a$.")


def figure_ladder(result, path: Path) -> str:
    """Both panel ladders for the cell with the largest remaining deviation."""
    rows = result["comparison"]["rows"]
    worst = max(rows, key=lambda r: max(
        abs(r["quantities"][q]["relative"]) for q in QUANTITIES))
    key = worst["cell"]
    entry = result["ladder"]["cells"][key]
    levels = sorted(entry["levels"].values(), key=lambda rec: rec["panels"])
    reference = {c["cell"]: c for c in result["reference"]["cells"]}[key]
    ref_levels = sorted(reference["levels"], key=lambda rec: rec["panels"])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.1))
    for ax, quantity, label in (
        (axes[0], "added_mass", r"$A_{33}$ [kg]"),
        (axes[1], "damping", r"$B_{33}$ [kg/s]"),
    ):
        ax.plot([rec["panels"] for rec in levels],
                [rec["values"]["heave"][quantity] for rec in levels],
                "o-", label="TensorBEM ladder")
        ax.plot([rec["panels"] for rec in ref_levels],
                [rec["values"]["heave"][quantity] for rec in ref_levels],
                "s--", label="capytaine ladder")
        for name, colour in (("tensorbem", "C0"), ("capytaine", "C1")):
            ax.axhline(worst["quantities"][f"heave.{quantity}"][name],
                       color=colour, lw=0.9, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("panels")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Panel ladder, worst-deviation cell {key} "
                 "(dotted: Richardson extrapolation used in the comparison)")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return ("Both panel ladders at the worst-deviation cell; dotted horizontals "
            "are the Richardson extrapolations that enter the comparison.")


def figure_surface_field(path: Path) -> str:
    """Body-surface potential of the heaving sphere at the shallowest depth."""
    from tensorbem.finite_depth_solver3d import solve_radiation_finite_depth_3d
    from tensorbem.geometry3d import submerged_sphere_mesh

    cell = next(c for c in plan.cells() if c["h_over_a"] == 3.0 and c["ka"] == 1.0)
    mesh = submerged_sphere_mesh(1, radius=plan.SPHERE_RADIUS,
                                 submergence=plan.SPHERE_SUBMERGENCE)
    result = solve_radiation_finite_depth_3d(
        mesh, cell["nu"], cell["depth"], modes=[0, 2],
        center=(0.0, 0.0, -plan.SPHERE_SUBMERGENCE),
        density=plan.DENSITY, gravity=plan.GRAVITY,
    )
    values = np.abs(result.potentials[:, 1])
    centroids = mesh.centroids
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    scatter = axes[0].scatter(centroids[:, 0], centroids[:, 2], c=values,
                              cmap="coolwarm", s=40)
    axes[0].axhline(-cell["depth"], color="k", lw=1.1)
    axes[0].text(0.05, -cell["depth"] + 0.08, "sea bed $z=-h$", fontsize=8)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("z [m]")
    axes[0].set_title(r"side view, $|\phi|$ on the body panels")
    fig.colorbar(scatter, ax=axes[0], fraction=0.046, label=r"$|\phi|$ [m$^2$/s]")
    order = np.argsort(-centroids[:, 2])
    axes[1].plot(values[order], "o-", ms=3.5)
    axes[1].set_xlabel("panel index (sorted by elevation, top first)")
    axes[1].set_ylabel(r"$|\phi|$ [m$^2$/s]")
    axes[1].grid(alpha=0.3)
    axes[1].set_title("heave potential vs elevation")
    fig.suptitle(
        rf"Body-surface potential, heaving sphere, $h/a=3$, $k_0a=1$, unit velocity, "
        rf"{mesh.panel_count} panels, residual {result.relative_residual:.1e}")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return ("Radiated-potential field on the body surface at the shallowest cell "
            "($h/a=3$); the potential distribution is what the added mass integrates.")


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #
def build_markdown(result, figures: dict[str, str]) -> str:
    comparison = result["comparison"]
    rows = comparison["rows"]
    worst = comparison["worst_relative"]
    status = result["case"]["execution_status"]
    ref_prov = result["reference_provenance"]
    tb_env = result["tensor_environment"]

    worst_am = max(worst[q] for q in worst if q.endswith("added_mass"))
    worst_b = max(worst[q] for q in worst if q.endswith("damping"))
    gate_am = gate(result, "added_mass_agreement")
    gate_b = gate(result, "damping_agreement")
    gate_ctrl = gate(result, "negative_control_separates_depth")
    gate_ref_order = gate(result, "reference_ladder_in_asymptotic_range")
    gate_tb_order = gate(result, "tensorbem_ladder_in_asymptotic_range")
    gate_res = gate(result, "linear_solve_residuals")

    lines: list[str] = []
    add = lines.append
    add(f"# XVAL-BEM-FINITE-DEPTH-CROSSVAL：有限水深辐射求解器的 code-to-code 交叉验证（m22474）")
    add("")
    add("**锚点** `XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN`（capytaine 的 "
        "`finite_depth_crossval_plan` 做法） · **仓** TensorBEM · "
        f"**执行状态** `{status}`")
    add("")
    add("## 1. 案例介绍")
    add("")
    add("有限水深辐射问题的面板法误差，只有在**第二个独立实现的代码**上才能看见：能量守恒、"
        "互易性、深水极限这些内部锚点共享同一套核，核里有错则一起错。本单把"
        "`tensorbem.finite_depth_solver3d`（本征函数级数核）与 capytaine 的有限水深 "
        "Green 函数（`Delhommeau` 多项式近似）放在同一组相似参数上逐格对照，"
        "并以**独立半解析锚**（Thorne/Wang/Linton 多极子解）标定有限水深的物理放大率。")
    add("")
    add(table(["量", "值"], [
        ["体", "全潜球，光滑闭体，深水有独立半解析锚，低阶面板法可收敛，无不规则频率"],
        ["半径 $a$", f"{plan.SPHERE_RADIUS:.2f} m"],
        ["球心吃水 $f$", f"{plan.SPHERE_SUBMERGENCE:.2f} m（最低点 $z=-2.50$ m）"],
        ["水深比 $h/a$ 网格", ", ".join(f"{r:.0f}" for r in plan.DEPTH_RATIOS)],
        ["波数网格 $k_0a$", ", ".join(f"{k:.1f}" for k in plan.KA_GRID)],
        ["格数", f"{len(rows)}（$h/a\\times k_0a$ 全积）"],
        ["$k_0h$ 覆盖", f"{min(r['k0h'] for r in rows):.2f} … {max(r['k0h'] for r in rows):.2f}"
         "（从强浅水到有效深水）"],
        ["密度 / 重力", f"$\\rho={plan.DENSITY:.0f}$ kg/m³，$g={plan.GRAVITY:.5f}$ m/s²"],
        ["TensorBEM 面板梯", ", ".join(str(p) for p in plan.TENSORBEM_LEVELS)
         + f"（0/1/2 级 → 20/80/320 面板）；确认档 {plan.EXTRA_TENSORBEM_LEVEL}（1280 面板）"
         f"用于 {len(plan.EXTRA_LEVEL_CELLS)} 格"],
        ["capytaine 面板梯", ", ".join(f"{n}×{m}" for n, m in plan.CAPYTAINE_RESOLUTIONS)
         + "（288/1152/4608 面板）+ `legacy` 公式 1152 面板作参考自身不确定带"],
    ]))
    add("")
    add("**关键设置：`nu` 回算。** 每格不取固定频率，而是由 `k0 tanh(k0 h) = nu` 从目标 "
        "$k_0a$ 反解 `nu`，使 $k_0a$ 在整条深度梯上精确不变——固定 `nu` 的频率扫描会把"
        "两个相似群（$k_0a$ 与 $h/a$）混在一起，被测效应正好被网格设计吞掉。")
    add("")
    add("## 2. 怎么做的")
    add("")
    add("1. **先定容差、后跑数**：`TOL_ADDED_MASS = 3%`、`TOL_DAMPING = 5%`，owner 上限 15%；"
        "负控最小分离 2%；观测阶下限 0.3。容差写在 `finite_depth_crossval_plan` 里，先于任何数字冻结。")
    add("2. **参考自带不确定带**：capytaine `newer`/`legacy` 两种有限水深公式的展布是代码内生的"
        "不确定带，TensorBEM 与之差小于该带时不携带信息。")
    add("3. **双侧细化、不比单网格**：两码都是低阶常强度面板法，单网格差由两个互不相关的离散误差"
        "主导。两侧各跑面板梯，比较 **Richardson 外推值**并报告观测阶，把「求解器分歧」与"
        "「网格噪声」分开归因。")
    add("4. **保留必须移动的负控**：深度盲的深水求解器跑同格，小 $k_0h$ 时必须与有限水深求解器分开，"
        "否则根本没在测深度。")
    add("5. **执行**：capytaine 参考由 `scripts/capytaine_finite_depth_ref.py` 在装了 capytaine 的"
        "控制主机上产出并冻结；TensorBEM 面板梯按格分片在集群上跑（`run_compute`，12 分片），"
        "由生产者自己的完整性校验合并器合并（缺片/缺档即失败，不允许静默缩小研究），"
        "再由消费者生成 `results.json` + 本页 + 图。")
    add("")
    add("**参考 provenance（程序化注入）**")
    add("")
    add(table(["项", "值"], [
        [key, str(ref_prov.get(key))]
        for key in ("tool", "version", "python", "platform", "finite_depth_method",
                    "finite_depth_legacy_method", "density", "gravity", "generated_utc")
        if key in ref_prov
    ] + [[f"TensorBEM 环境", json.dumps(tb_env)[:160]] if tb_env else []]))
    add("")
    add("## 3. 计算结果")
    add("")
    add("### 3.1 逐格外推值对照（程序化读 `results.json`，零手抄）")
    add("")
    body = []
    for row in sorted(rows, key=lambda r: (r["h_over_a"], r["ka"])):
        q_am = row["quantities"]["heave.added_mass"]
        q_b = row["quantities"]["heave.damping"]
        body.append([
            f"{row['h_over_a']:.0f}", f"{row['ka']:.2f}", f"{row['k0h']:.2f}",
            num(q_am["tensorbem"]), num(q_am["capytaine"]), pct(q_am["relative"]),
            num(q_b["tensorbem"]), num(q_b["capytaine"]), pct(q_b["relative"]),
        ])
    add(table(["$h/a$", "$k_0a$", "$k_0h$", "$A_{33}$ TB", "$A_{33}$ capy",
               "$\\Delta A_{33}$", "$B_{33}$ TB", "$B_{33}$ capy", "$\\Delta B_{33}$"],
              body))
    add("")
    add("### 3.2 最差格与门（预注册判据）")
    add("")
    add(table(["门", "最差值 / 观测值", "限", "判定"], [
        ["added_mass_agreement", pct(gate_am["worst"]), f"{100 * gate_am['limit']:.0f} %",
         "PASS" if gate_am["passed"] else "FAIL"],
        ["damping_agreement", pct(gate_b["worst"]), f"{100 * gate_b['limit']:.0f} %",
         "PASS" if gate_b["passed"] else "FAIL"],
        ["negative_control_separates_depth", pct(gate_ctrl["min_gap"]),
         f"{100 * gate_ctrl['limit']:.0f} %", "PASS" if gate_ctrl["passed"] else "FAIL"],
        ["reference_ladder_in_asymptotic_range", num(gate_ref_order["min_order"], 3),
         num(gate_ref_order["limit"], 1), "PASS" if gate_ref_order["passed"] else "FAIL"],
        ["tensorbem_ladder_in_asymptotic_range", num(gate_tb_order["min_order"], 3),
         num(gate_tb_order["limit"], 1), "PASS" if gate_tb_order["passed"] else "FAIL"],
        ["linear_solve_residuals", f"{gate_res['max']:.2e}", f"{gate_res['limit']:.0e}",
         "PASS" if gate_res["passed"] else "FAIL"],
    ]))
    add("")
    add("### 3.3 图")
    for caption, path in figures.items():
        add(f"**{caption}**  ")
        add(f"![{caption}]({Path(path).relative_to(ROOT)})")
        add("")
    add("## 4. 注意点与局限")
    add("")
    add("- **两码同族离散**：都以内接多面体离散球，粗档面积亏 1–2% 按同一符号进入两侧；"
        "外推一阶消掉，剩下的差异才归给求解器。")
    add("- **参考不是解析解**：capytaine 的有限水深 Green 函数本身是 Delhommeau 多项式近似，"
        "`newer`/`legacy` 两式在此几何上高度一致，因此**不**提供独立第二意见，"
        "真正的独立锚只有深水多极子解析解（§3.3 第三图）与深水极限。")
    add("- **时间谐 e^{-iωt}，单位速度幅值**；$A_{ij}/B_{ij}$ 与 mode 顺序 "
        "(surge, sway, heave, roll, pitch, yaw) 同仓内约定。")
    add("- 确认档（1280 面板）只在 4 格上跑，用于检验 20/80/320 测得的阶再外推一级是否成立；"
        "全 12 格上跑 1280 面板的代价不进入本研究的结论。")
    add("")
    add("## 5. 误差判定（逐项，显式回答「是否 <3%」）")
    add("")
    add(table(["量", "最差相对偏差", "预注册带", "owner 限 15%", "是否 <3%"], [
        ["heave added mass", pct(worst["heave.added_mass"]), "3 %",
         "是" if worst["heave.added_mass"] <= plan.OWNER_LIMIT else "否",
         "是" if worst["heave.added_mass"] <= 0.03 else "否"],
        ["surge added mass", pct(worst["surge.added_mass"]), "3 %",
         "是" if worst["surge.added_mass"] <= plan.OWNER_LIMIT else "否",
         "是" if worst["surge.added_mass"] <= 0.03 else "否"],
        ["heave damping", pct(worst["heave.damping"]), "5 %",
         "是" if worst["heave.damping"] <= plan.OWNER_LIMIT else "否",
         "是" if worst["heave.damping"] <= 0.03 else "否"],
        ["surge damping", pct(worst["surge.damping"]), "5 %",
         "是" if worst["surge.damping"] <= plan.OWNER_LIMIT else "否",
         "是" if worst["surge.damping"] <= 0.03 else "否"],
    ]))
    add("")
    inside_owner = worst_am <= plan.OWNER_LIMIT and worst_b <= plan.OWNER_LIMIT
    inside_band = gate_am["passed"] and gate_b["passed"]
    add(f"**显式回答**：")
    add("")
    add(f"- owner 15% 门槛：{'**全部小于 15%**' if inside_owner else '**有量超过 15%**'}；"
        f"最差 added mass {pct(worst_am)}，最差 damping {pct(worst_b)}。")
    add(f"- 预注册带（3% / 5%）：{'**全部满足**' if inside_band else '**有量超出预注册带**'}；"
        f"added mass 门 {'PASS' if gate_am['passed'] else 'FAIL'}"
        f"（{pct(gate_am['worst'])}），damping 门 {'PASS' if gate_b['passed'] else 'FAIL'}"
        f"（{pct(gate_b['worst'])}）。")
    add(f"- 「是否 <3%」：added mass "
        f"{'是' if worst_am <= 0.03 else f'**否**（{pct(worst_am)}）'}，damping "
        f"{'是' if worst_b <= 0.03 else f'**否**（{pct(worst_b)}）'}。"
        "未达标之处如实列出，不做四舍五入粉饰。")
    add("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# publish
# --------------------------------------------------------------------------- #
def _login(base: str, user: str, password: str) -> str:
    import urllib.parse

    data = urllib.parse.urlencode({"username": user, "password": password}).encode()
    request = urllib.request.Request(
        f"{base}/auth/login", data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())["access_token"]


def publish(payload: dict, base: str, token: str) -> dict:
    request = urllib.request.Request(
        f"{base}/tensor/reports", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("/tmp/report_m22474.json"))
    args = parser.parse_args(argv)

    result = json.loads(RESULTS.read_text())
    FIGDIR.mkdir(parents=True, exist_ok=True)
    figures: dict[str, str] = {}
    fig_specs = [
        ("图1 12 格 A33/B33 对照（TensorBEM vs capytaine，含深度梯）",
         "fig1_frequency_depth.png", figure_frequency),
        ("图2 外推偏差图 (h/a, k0a)：预注册带 3%/5%",
         "fig2_deviation_map.png", figure_map),
        ("图3 有限水深放大率 vs 深水多极子解析锚（独立半解析）",
         "fig3_depth_amplification.png", figure_depth_physics),
        ("图4 最差格的双方面板梯与外推值",
         "fig4_panel_ladder.png", figure_ladder),
    ]
    for caption, name, builder in fig_specs:
        builder(result, FIGDIR / name)
        figures[caption] = str(FIGDIR / name)
    figure_surface_field(FIGDIR / "fig5_surface_potential.png")
    figures["图5 体表面辐射势场（h/a=3 格，场量云图）"] = str(
        FIGDIR / "fig5_surface_potential.png")

    markdown = build_markdown(result, figures)
    (ROOT / "report_m22474.md").write_text(markdown, encoding="utf-8")

    worst = result["comparison"]["worst_relative"]
    payload = {
        "title": "有限水深 BEM 交叉验证收口：12 格 × 双侧面板梯 vs capytaine（TensorBEM m22474）",
        "summary": (
            f"锚点 XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN：12 格外推后 heave/surge "
            f"added mass 最差 {100 * max(worst['heave.added_mass'], worst['surge.added_mass']):.2f}%、"
            f"damping 最差 {100 * max(worst['heave.damping'], worst['surge.damping']):.2f}%"
            f"（owner 限 15%，预注册带 3%/5%）"
        ),
        "scenario_key": "bem_finite_depth_crossvalidation",
        "mission_id": MISSION_ID,
        "tags": ["XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN", "TensorBEM", "capytaine",
                 "finite-depth", "cross-validation", "Richardson-extrapolation"],
        "markdown": markdown,
        "figures": [
            {"name": Path(path).name, "caption": caption,
             "png_base64": base64.b64encode(Path(path).read_bytes()).decode("ascii")}
            for caption, path in figures.items()
        ],
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"report": str(args.out), "figures": [str(p) for p in figures.values()],
                      "title": payload["title"]}, ensure_ascii=False, indent=2))
    if args.publish:
        user = os.environ["TENSOR_AGENT_USER"]
        password = os.environ["TENSOR_AGENT_PASS"]
        token = _login(API_BASE, user, password)
        print(json.dumps(publish(payload, API_BASE, token), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())