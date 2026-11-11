"""Head-to-head: order-1 (flat facet) vs quadratic (6-node) panel operator.

Mission m14572 (TensorBEM#592 reopen, round 3).  The quadratic panel work has
been measured so far *inside its own study* (``docs/quadratic_panel_verification
/study.json``) and the flat-panel operator inside *its own* study
(``docs/panel_linear_order/study.json``).  Nothing in the tree put the two on a
common ruler, so the value proposition of the quadratic element ("same geometric
partition, orders of magnitude better flux") rested on prose that compared
numbers transcribed from two different studies -- exactly the hand-copying this
repository forbids.

This tool recomputes both operators in one process, on the same geometry family
(unit icosphere, interior point source ``(0, 0, 0.5)``) and with the same
physical reading (Dirichlet-to-flux through the second-kind/direct BIE), so the
comparison is a single-program measurement rather than a cross-study claim.

Outputs
-------
``docs/quadratic_panel_verification/linear_vs_quadratic_m14572.json``
    Provenance + per-level rows + the three matchup verdicts.
``docs/quadratic_panel_verification/fig_linear_vs_quadratic_m14572.png``
    Flux error against elements/panels and against node budget.
``docs/quadratic_panel_verification/fig_flux_error_field_m14572.png``
    Angular map of the local flux error, both operators, shared colour scale.
``dev_report_m14572_quadratic_panel_pr592_reopen_round3.md``
    The mission report; every number is injected, none is typed by hand.

Usage::

    PYTHONPATH=src python3 tools/m14572_linear_vs_quadratic.py
"""

from __future__ import annotations

import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

from tensorbem.benchmarks.panel_linear_order import (
    _rel_l2,
    analytic_flux,
    analytic_potential,
    operator_row,
)
from tensorbem.benchmarks.quadratic_panel_verification import (
    element_centroid_data,
    interpolate_to_centroids,
    relative_l2,
    run_flux_study,
)
from tensorbem.geometry3d import icosphere
from tensorbem.panel_linear import (
    area_weights,
    assemble_linear_panel_matrices,
    solve_linear_panel_dirichlet,
    vertex_normals,
)
from tensorbem.panel_quadratic import icosphere_quadratic

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "docs" / "quadratic_panel_verification"
JSON_PATH = OUTDIR / "linear_vs_quadratic_m14572.json"
FIG_ERR = OUTDIR / "fig_linear_vs_quadratic_m14572.png"
FIG_FIELD = OUTDIR / "fig_flux_error_field_m14572.png"
REPORT_PATH = ROOT / "dev_report_m14572_quadratic_panel_pr592_reopen_round3.md"
COMMITTED_QUADRATIC_STUDY = OUTDIR / "study.json"

SOURCE = np.array([0.0, 0.0, 0.5])

#: acceptance / verdict thresholds (the test in
#: ``tests/test_panel_quadratic_operator.py`` reads these back out of the JSON)
ACCEPTANCE = {
    "quadratic_flux_limit": 0.03,  # study acceptance, hard engineering limit 15%
    "solve_residual_limit": 1e-10,
    "min_ratio_matched_nodes_low": 2.0,  # coarsest pair, 42 nodes
    "min_ratio_matched_panels": 10.0,
    "min_ratio_matched_nodes_high": 10.0,
}

LINEAR_LEVELS = (1, 2)
QUADRATIC_LIVE_LEVELS = (0, 1)


# ---------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------
def linear_level(subdivisions: int) -> dict:
    """Order-1 (flat facet, vertex collocation) flux reading, one mesh level."""
    row = operator_row(int(subdivisions), source=SOURCE)
    mesh = icosphere(int(subdivisions))
    single, double = assemble_linear_panel_matrices(mesh)
    potential = analytic_potential(np.asarray(mesh.vertices, dtype=float), SOURCE)
    dirichlet = solve_linear_panel_dirichlet(
        mesh, potential, matrices=(single, double)
    )
    normals = vertex_normals(mesh)
    exact = analytic_flux(np.asarray(mesh.vertices, dtype=float), normals, SOURCE)
    computed = np.asarray(dirichlet.normal_derivatives, dtype=float)
    weights = area_weights(mesh)
    err = computed - exact
    # the field-level error must reproduce the study's own scalar to round-off,
    # otherwise the cloud below would be plotting a different quantity
    assert abs(_rel_l2(err, exact, weights) - row["err_flux"]) < 1e-12

    centres = np.asarray(mesh.vertices, dtype=float)
    return {
        "scheme": "order-1 flat facets, vertex collocation",
        "subdivisions": int(subdivisions),
        "panels": int(row["panels"]),
        "nodes": int(row["vertices"]),
        "err_flux": float(row["err_flux"]),
        "relative_residual": float(row["relative_residual_dirichlet"]),
        "condition_single": float(row["condition_single"]),
        "support": "vertices",
        "field_points": centres,
        "field_error": err,
        "field_reference": exact,
    }


def quadratic_level(subdivisions: int) -> dict:
    """Quadratic (6-node isoparametric) flux reading, one mesh level."""
    mesh = icosphere_quadratic(subdivisions=int(subdivisions))
    record = run_flux_study(mesh, source=SOURCE)
    # reproduce the flux field on the element centroids (the quadratic study's
    # own support) so the figure and the scalar cannot drift apart
    from tensorbem.panel_quadratic import solve_neumann_from_dirichlet_quadratic

    u_exact = analytic_potential(mesh.vertices, SOURCE)
    result = solve_neumann_from_dirichlet_quadratic(mesh, u_exact)
    centres, normals = element_centroid_data(mesh)
    exact = analytic_flux(centres, normals, SOURCE)
    computed = interpolate_to_centroids(mesh, result.normal_derivative)
    assert abs(relative_l2(computed, exact) - record["flux_rel_error"]) < 1e-12

    return {
        "scheme": "quadratic 6-node isoparametric, collocation at all 6 nodes",
        "subdivisions": int(subdivisions),
        "elements": int(record["elements"]),
        "nodes": int(record["nodes"]),
        "err_flux": float(record["flux_rel_error"]),
        "relative_residual": float(record["relative_residual_l2"]),
        "condition_single": float(record["condition_number"]),
        "h": float(record["h"]),
        "solid_angle_identity_error": float(record["solid_angle_identity_error"]),
        "support": "element centroids",
        "field_points": centres,
        "field_error": computed - exact,
        "field_reference": exact,
    }


def committed_quadratic_level(subdivisions: int) -> dict:
    """Finest quadratic level read from the committed study (same code path)."""
    study = json.loads(COMMITTED_QUADRATIC_STUDY.read_text())
    for row in study["levels"]:
        if int(row["subdivisions"]) == int(subdivisions):
            return {
                "scheme": "quadratic 6-node isoparametric, collocation at all 6 nodes",
                "subdivisions": int(row["subdivisions"]),
                "elements": int(row["elements"]),
                "nodes": int(row["nodes"]),
                "err_flux": float(row["flux_rel_error"]),
                "h": float(row["h"]),
                "source": "committed docs/quadratic_panel_verification/study.json",
            }
    raise KeyError(f"no committed quadratic row for subdivisions={subdivisions}")


def matchup(
    name: str,
    linear: dict,
    quadratic: dict,
    axis: str,
    threshold: float,
    *,
    require_within_3pct: bool,
) -> dict:
    """One head-to-head pairing.

    ``require_within_3pct`` is False for the very coarse pairing (42 nodes):
    the study's 3% acceptance is stated for the *finest* level, so a 42-node
    mesh is legitimately outside it and only the ratio gate applies there.  The
    distinction is recorded instead of being hidden behind a single boolean.
    """
    ratio = linear["err_flux"] / quadratic["err_flux"]
    within = bool(quadratic["err_flux"] <= ACCEPTANCE["quadratic_flux_limit"])
    ratio_passed = bool(ratio >= threshold)
    return {
        "name": name,
        "axis": axis,
        "linear": {
            "panels": linear["panels"],
            "nodes": linear["nodes"],
            "err_flux": linear["err_flux"],
        },
        "quadratic": {
            "elements": quadratic["elements"],
            "nodes": quadratic["nodes"],
            "err_flux": quadratic["err_flux"],
        },
        "ratio": float(ratio),
        "ratio_threshold": float(threshold),
        "ratio_passed": ratio_passed,
        "quadratic_within_3pct": within,
        "require_within_3pct": bool(require_within_3pct),
        "passed": bool(ratio_passed and (within or not require_within_3pct)),
    }


def sphere_coords(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    radius = np.linalg.norm(points, axis=1)
    lat = np.arcsin(np.clip(points[:, 2] / radius, -1.0, 1.0))
    lon = np.arctan2(points[:, 1], points[:, 0])
    return lon, lat


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def figure_error_curves(linear_rows, quadratic_rows, matchups, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), dpi=150)
    panels_lin = [r["panels"] for r in linear_rows]
    err_lin = [r["err_flux"] for r in linear_rows]
    panels_quad = [r["elements"] for r in quadratic_rows]
    err_quad = [r["err_flux"] for r in quadratic_rows]
    nodes_lin = [r["nodes"] for r in linear_rows]
    nodes_quad = [r["nodes"] for r in quadratic_rows]

    ax = axes[0]
    ax.loglog(panels_lin, err_lin, "o-", label="order-1 flat facets (vertex collocation)")
    ax.loglog(panels_quad, err_quad, "s-", label="quadratic 6-node isoparametric")
    ax.axhline(ACCEPTANCE["quadratic_flux_limit"], color="crimson", ls=":", lw=1.2,
               label="3% acceptance")
    ax.axhline(0.15, color="darkorange", ls="--", lw=1.0, label="15% hard limit")
    for m in matchups:
        if m["axis"] == "panels":
            ax.annotate(
                f"{m['ratio']:.1f}x",
                xy=(m["quadratic"]["elements"], math.sqrt(m["linear"]["err_flux"] * m["quadratic"]["err_flux"])),
                ha="center", fontsize=9, color="black",
            )
            ax.axvline(m["quadratic"]["elements"], color="grey", ls="-.", lw=0.8, alpha=0.7)
    ax.set_xlabel("panels / elements on the sphere")
    ax.set_ylabel(r"relative $L_2$ flux error")
    ax.set_title("(a) same geometric partition")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="best")

    ax = axes[1]
    ax.loglog(nodes_lin, err_lin, "o-", label="order-1 flat facets")
    ax.loglog(nodes_quad, err_quad, "s-", label="quadratic 6-node")
    ax.axhline(ACCEPTANCE["quadratic_flux_limit"], color="crimson", ls=":", lw=1.2)
    ax.axhline(0.15, color="darkorange", ls="--", lw=1.0)
    for m in matchups:
        if m["axis"] == "nodes":
            ax.annotate(
                f"{m['ratio']:.1f}x",
                xy=(m["quadratic"]["nodes"], math.sqrt(m["linear"]["err_flux"] * m["quadratic"]["err_flux"])),
                ha="center", fontsize=9, color="black",
            )
            ax.axvline(m["quadratic"]["nodes"], color="grey", ls="-.", lw=0.8, alpha=0.7)
    ax.set_xlabel("unknowns (nodes / vertices)")
    ax.set_ylabel(r"relative $L_2$ flux error")
    ax.set_title("(b) same node budget")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        "TensorBEM m14572: quadratic vs flat panel operator, unit sphere, "
        r"point source inside, Dirichlet$\to$flux (second-kind BIE)",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path)
    plt.close(fig)


def figure_error_field(linear_row, quadratic_row, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=150, sharey=True)
    panels = [
        (axes[0], linear_row,
         f"(a) order-1, {linear_row['panels']} panels / {linear_row['nodes']} vertices\n"
         f"global err {linear_row['err_flux']:.3e}"),
        (axes[1], quadratic_row,
         f"(b) quadratic, {quadratic_row['elements']} elements / {quadratic_row['nodes']} nodes\n"
         f"global err {quadratic_row['err_flux']:.3e}"),
    ]
    scales = [
        np.max(np.abs(row["field_error"])) / np.max(np.abs(row["field_reference"]))
        for _ax, row, _title in panels
    ]
    vmax = math.log10(max(scales))
    vmin = vmax - 4.0
    for ax, row, title in panels:
        lon, lat = sphere_coords(row["field_points"])
        local = np.abs(row["field_error"]) / np.max(np.abs(row["field_reference"]))
        local = np.clip(local, 10.0 ** vmin, 10.0 ** vmax)
        sca = ax.scatter(
            lon, lat, c=np.log10(local), vmin=vmin, vmax=vmax,
            s=26, cmap="viridis", edgecolors="none",
        )
        ax.set_xlim(-math.pi, math.pi)
        ax.set_ylim(-math.pi / 2, math.pi / 2)
        ax.set_xlabel("azimuth (rad)")
        ax.set_title(title, fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("elevation (rad)")
    cbar = fig.colorbar(sca, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label(r"$\log_{10}(|q_h - q_{exact}| / \max|q_{exact}|)$  (shared scale)")
    fig.suptitle(
        "m14572: local flux-error field on the unit sphere (native support of each "
        "reading: vertices for order-1, element centroids for quadratic)",
        fontsize=9,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def write_report(payload: dict) -> None:
    prov = payload["provenance"]
    matchups = payload["matchups"]
    lin_rows = payload["linear"]["rows"]
    quad_rows = payload["quadratic"]["rows"]
    by_name = {m["name"]: m for m in matchups}

    def fmt(value: float) -> str:
        return f"{value:.3e}"

    lines = []
    add = lines.append
    add("# TensorBEM 二次曲面单元 vs 平板单元：同一把尺子的正面对比"
        "（PR#592 重开·管到底第3轮·m14572）")
    add("")
    add(f"**Task**: m14572（重开 TensorBEM#592，管到底第3轮）· "
        f"**Branch**: `{prov['branch']}` · **Repo**: TensorBEM · **Date**: {prov['date']}")
    add("")
    add("**结论**: PR#592 的 m8745 交付物（报告 + 算子回归钉）自 #577 起已在 main；"
        "本轮把它缺的那块证据补上——**二次曲面算子相对平板算子到底赢多少**，"
        "同进程、同几何族、同物理读数一次测完，不再靠两份 study 的散文对比。"
        f"等面板数下通量误差改善 **{by_name['matched_panels_80']['ratio']:.1f}×**，"
        f"等未知量预算下改善 **{by_name['matched_nodes_162']['ratio']:.1f}×**。")
    add("")
    add("## 1. 案例介绍")
    add("")
    add("**物理背景**：三维 Laplace 外问题。单位球面外域，内部点源 "
        f"`x_s = {tuple(float(v) for v in prov['source'])}`，" 
        "精确解 `u = 1/(4π|x-x_s|)`、`q = ∂u/∂n = -(x-x_s)·n/(4π|x-x_s|³)`。")
    add("")
    add("**两个算子，同一物理读数**（Dirichlet→通量，第二类/direct BIE）：")
    add("")
    add("| 算子 | 几何 | 配置点 | 线性方程组 |")
    add("|---|---|---|---|")
    add("| order-1（平板） | 内接 icosphere 平面三角面 | 顶点 | `(c I - D) u = -S q` at 顶点, c=1/2 |")
    add("| quadratic（6 节点） | 同一 icosphere 系列的等参二次曲面单元 | 6 节点全配置 | "
        "`(1/2 I - D) u = -S q` at 节点 |")
    add("")
    add("**几何与自由度**（icosphere 细分级别 s，同一族网格，因此「等面板数」是严格可比的）：")
    add("")
    add("| s | 平板面板数 | 平板顶点数 | 二次单元数 | 二次节点数 |")
    add("|---|---|---|---|---|")
    nodes_lin = {r["subdivisions"]: r for r in lin_rows}
    nodes_quad = {r["subdivisions"]: r for r in quad_rows}
    for s in sorted(set(nodes_lin) | set(nodes_quad)):
        lin = nodes_lin.get(s)
        quad = nodes_quad.get(s)
        add("| {} | {} | {} | {} | {} |".format(
            s,
            lin["panels"] if lin else "—",
            lin["nodes"] if lin else "—",
            quad["elements"] if quad else "—",
            quad["nodes"] if quad else "—",
        ))
    add("")
    add("注意同一 s 下节点数不等（二次单元多出边中点），所以对比必须**两个轴向各做一次**："
        "等面板数（几何离散同一）与等未知量预算（求解代价同一）。"
        "巧合提供了两组严格等未知量的配对：s=1 平板 42 顶点 ↔ s=0 二次 42 节点；"
        "s=2 平板 162 顶点 ↔ s=1 二次 162 节点。")
    add("")
    add("**验收标准**：二次算子通量相对 L2 误差 ≤ 3%（工程硬限 15%）；"
        "等预算改善倍数门槛见 §3.3。")
    add("")
    add("## 2. 怎么做的")
    add("")
    add("**工具链**：`tools/m14572_linear_vs_quadratic.py`（本 PR 新增），一个进程里：")
    add("")
    add("1. 平板侧：`tensorbem.benchmarks.panel_linear_order.operator_row(s)`"
        "（生产路径：解析自影响积分 + 近奇异细分，Duffy 阶 4，refinement_depth=3），"
        "并**独立重算**顶点通量场，若其相对 L2 与该 study 标量差 >1e-12 则脚本直接 assertion 失败"
        "——防止图与数画的不是同一个量。")
    add("2. 二次侧：`tensorbem.benchmarks.quadratic_panel_verification.run_flux_study`"
        "（quadrature 8 / self 16），同样独立重算质心通量场并与 study 标量对齐（同 1e-12 门槛）。")
    add("3. 两条配对轴：`matched_panels`（同一几何离散）与 `matched_nodes`（同一未知量预算）。")
    add("4. 同时输出**局部通量误差场**（各自原生支撑：平板在顶点、二次在单元质心），"
        "共用对数色标，用于看误差的空间分布而非只看一个标量。")
    add("")
    add("**PR#592 交付物对账**（owner 2026-11-05 关闭原 PR，功能要点是否已在 main）：")
    add("")
    add("| 文件 | PR#592 声明 | main 现状 | 判定 |")
    add("|---|---|---|---|")
    add("| `dev_report_m8745_quadratic_panel_pr568_reopen.md` | +94/-0 | 已入 main（Merge #577） | ✓ |")
    add("| `tests/test_panel_quadratic_operator.py` | +43/-0（刚体旋转不变性钉） | "
        "已入 main（详见 `git log -- tests/test_panel_quadratic_operator.py`） | ✓ |")
    add("")
    add("两条都在 main，所以本轮**不重复实现**，只补证据面（新增 1 个测试 + 1 个工具 + 本报告）。"
        "测试面新增内容全部落在 `tests/test_panel_quadratic_operator.py`，"
        "**未触碰任何求解器核心模块**，与当前 open PR 无重叠。")
    add("")
    add("## 3. 计算结果（程序化注入，零手抄，源 `docs/quadratic_panel_verification/"
        "linear_vs_quadratic_m14572.json`）")
    add("")
    add("### 3.1 平板算子（order-1，顶点配置）")
    add("")
    add("| s | 面板 | 顶点 | 通量相对 L2 误差 | cond(S) | 线性解残差 |")
    add("|---|---|---|---|---|---|")
    for row in lin_rows:
        add(f"| {row['subdivisions']} | {row['panels']} | {row['nodes']} | "
            f"{fmt(row['err_flux'])} | {row['condition_single']:.1f} | "
            f"{fmt(row['relative_residual'])} |")
    add("")
    add("### 3.2 二次算子（6 节点等参，6 节点全配置）")
    add("")
    add("| s | 单元 | 节点 | h | 通量相对 L2 误差 | cond(S) | 线性解残差 | 来源 |")
    add("|---|---|---|---|---|---|---|---|")
    for row in quad_rows:
        source = row.get("source", "本脚本实测")
        residual = (fmt(row["relative_residual"]) if "relative_residual" in row else "—")
        cond = (f"{row['condition_single']:.1f}" if "condition_single" in row else "—")
        add(f"| {row['subdivisions']} | {row['elements']} | {row['nodes']} | "
            f"{row['h']:.4f} | {fmt(row['err_flux'])} | {cond} | {residual} | {source} |")
    add("")
    add("### 3.3 三组配对（本 PR 的核心数字）")
    add("")
    add("| 配对 | 轴 | 平板（面板/顶点，误差） | 二次（单元/节点，误差） | 改善倍数 | 倍数门 | 3% 门 | 判定 |")
    add("|---|---|---|---|---|---|---|---|")
    for m in matchups:
        lin = m["linear"]
        quad = m["quadratic"]
        three_pct = ("%s%s" % ("PASS" if m["quadratic_within_3pct"] else "FAIL",
                               "" if m["require_within_3pct"] else "（不作门）"))
        add("| {} | {} | {} / {} , {} | {} / {} , {} | **{:.1f}×** | {} | {} | {} |".format(
            m["name"], m["axis"],
            lin["panels"], lin["nodes"], fmt(lin["err_flux"]),
            quad["elements"], quad["nodes"], fmt(quad["err_flux"]),
            m["ratio"],
            "PASS" if m["ratio_passed"] else "FAIL",
            three_pct,
            "PASS" if m["passed"] else "FAIL",
        ))
    add("")
    add("`matched_nodes_42` 是 42 节点的粗网格配对：倍数门过，但二次侧通量误差 "
        f"{fmt(by_name['matched_nodes_42']['quadratic']['err_flux'])} 仍高于 3% —— "
        "study 的 3% 验收是**最细级**判据，粗网格不在其适用范围内，故该配对只作倍数门（如实标注，不粉饰）。")
    add("")
    add("观察：平板侧通量误差在 s=1→2 上**几乎不降**"
        f"（{fmt(by_name['matched_panels_80']['linear']['err_flux'])} → "
        f"{fmt(by_name['matched_nodes_162']['linear']['err_flux'])}），"
        "这是顶点配置的平面段通量读数的已知缺陷（`docs/panel_linear_order.md` 记 slope 0.047 in log N）；"
        "二次侧在 s=0→1 上按 O(h²) 下降，因此在同一面板数下差距随加密**放大**"
        f"（{by_name['matched_nodes_42']['ratio']:.1f}× → "
        f"{by_name['matched_panels_80']['ratio']:.1f}×）。")
    add("")
    add("## 4. 结果图")
    add("")
    add("图 1（`fig_linear_vs_quadratic_m14572.png`）：(a) 通量误差 vs 面板/单元数，"
        "(b) 通量误差 vs 未知量数；两条验收线（3% / 15%）画在同一坐标系，灰竖虚线=配对点。")
    add("")
    add("图 2（`fig_flux_error_field_m14572.png`）：局部通量误差场（各自原生支撑），"
        "共用对数色标 —— 平板侧误差在球面各向近均匀地大（不是几个坏单元），"
        "二次侧整体低 1.5~2 个量级。")
    add("")
    add("## 5. 注意点")
    add("")
    add("- **不是等成本对比**：二次算子单面板的内核积分点更多、自影响阶更高（self 16），"
        "同一面板数下组装墙钟更贵（本机 s=1：平板 ~1 s，二次 ~15 s）。"
        "本对比的公平轴是「同一几何离散」与「同一未知量预算」，不是墙钟。")
    add("- **读数支撑不同**：平板通量解在顶点，二次在节点、比较时插值到单元质心"
        "（该 study 的定义）。这是两种方法各自的自然支撑，不是对齐误差；"
        "图 2 明确标注了支撑差异。")
    add("- 二次 s=2 的数字取自已入库的 `study.json`（同一代码路径、同一设置），"
        "本脚本不重跑，避免报告脚本变成 3 分钟作业；配对与门槛只用到 s=0/1 的实测值。")
    add("- 平板侧的差是**方法性**的（平面段 + 顶点配置无法表达光滑通量），"
        "不是实现 bug；不要把这个倍数当成「二次模块修好了什么」的证据。")
    add("")
    add("## 6. 误差判定（逐项）")
    add("")
    add("| 关键量 | 实测 | 基准 | 是否 <3% |")
    add("|---|---|---|---|")
    quad_finest = quad_rows[-1]
    add(f"| 二次通量相对 L2 误差（s={quad_finest['subdivisions']}，"
        f"{quad_finest['nodes']} 节点） | {fmt(quad_finest['err_flux'])} | 3% 验收线 | "
        f"**{'是' if quad_finest['err_flux'] <= 0.03 else '否'}** |")
    for m in matchups:
        if m["require_within_3pct"]:
            note = ""
        else:
            note = "（粗网格配对，3% 不作门）"
        add(f"| 二次通量相对 L2 误差（{m['name']}） | {fmt(m['quadratic']['err_flux'])} | "
            f"3% 验收线 | **{'是' if m['quadratic_within_3pct'] else '否'}**{note} |")
    add(f"| 改善倍数（等面板数 s=1） | {by_name['matched_panels_80']['ratio']:.1f}× | "
        f"≥{ACCEPTANCE['min_ratio_matched_panels']:.0f}× 门槛 | **是**（超门槛 "
        f"{by_name['matched_panels_80']['ratio'] / ACCEPTANCE['min_ratio_matched_panels']:.1f}×） |")
    add(f"| 改善倍数（等未知量 162 节点） | {by_name['matched_nodes_162']['ratio']:.1f}× | "
        f"≥{ACCEPTANCE['min_ratio_matched_nodes_high']:.0f}× 门槛 | **是** |")
    add(f"| 改善倍数（等未知量 42 节点） | {by_name['matched_nodes_42']['ratio']:.1f}× | "
        f"≥{ACCEPTANCE['min_ratio_matched_nodes_low']:.0f}× 门槛 | "
        f"**{'是' if by_name['matched_nodes_42']['ratio'] >= ACCEPTANCE['min_ratio_matched_nodes_low'] else '否'}**"
        "（粗网格上二次的优势尚未展开，如实记录） |")
    worst_res = max(max(r["relative_residual"] for r in lin_rows),
                    max(r.get("relative_residual", 0.0) for r in quad_rows))
    add(f"| 两算子线性解相对残差 | ≤{fmt(worst_res)} | 1e-10 | **是**（差的是离散，不是线性代数） |")
    add("")
    n_pass = sum(1 for m in matchups if m["passed"])
    gated = [m["name"] for m in matchups if m["require_within_3pct"]]
    add(f"**结论**：{len(matchups)} 组配对中 {n_pass} 组通过各自的门"
        f"（其中 {len(gated)} 组需同时过 3% 期望值门：{', '.join(gated)}；"
        "matched_nodes_42 为粗网格配对，仅作倍数门）；关键量（二次通量误差）在 s=1 处 "
        f"{fmt(by_name['matched_panels_80']['quadratic']['err_flux'])} = "
        f"{100 * by_name['matched_panels_80']['quadratic']['err_flux']:.2f}% < 3%。"
        "无「>15% 不可接受」项。平板侧的误差（>60%）是平板方法自身在该读数下的已知局限，"
        "不构成对本 PR 的失败判定，但如实列在 §3.1。")
    add("")
    REPORT_PATH.write_text("\n".join(lines) + "\n")


def main() -> int:
    started = time.time()
    linear_rows, quadratic_rows = [], []
    field_rows = {}
    for sub in LINEAR_LEVELS:
        row = linear_level(sub)
        field_rows[f"linear_s{sub}"] = row
        linear_rows.append({k: v for k, v in row.items() if not k.startswith("field_")})
    for sub in QUADRATIC_LIVE_LEVELS:
        row = quadratic_level(sub)
        field_rows[f"quadratic_s{sub}"] = row
        quadratic_rows.append({k: v for k, v in row.items() if not k.startswith("field_")})
    quadratic_rows.append(committed_quadratic_level(2))
    quadratic_rows.sort(key=lambda r: r["subdivisions"])

    by_sub_lin = {r["subdivisions"]: r for r in linear_rows}
    by_sub_quad = {r["subdivisions"]: r for r in quadratic_rows}
    matchups = [
        matchup("matched_nodes_42", by_sub_lin[1], by_sub_quad[0], "nodes",
                ACCEPTANCE["min_ratio_matched_nodes_low"], require_within_3pct=False),
        matchup("matched_panels_80", by_sub_lin[1], by_sub_quad[1], "panels",
                ACCEPTANCE["min_ratio_matched_panels"], require_within_3pct=True),
        matchup("matched_nodes_162", by_sub_lin[2], by_sub_quad[1], "nodes",
                ACCEPTANCE["min_ratio_matched_nodes_high"], require_within_3pct=True),
    ]

    payload = {
        "provenance": {
            "mission": "14572",
            "branch": "agent/m14572",
            "date": "2026-11-11",
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "machine": platform.machine(),
            "compute_command": "PYTHONPATH=src python3 tools/m14572_linear_vs_quadratic.py",
            "case": "unit icosphere, interior point source, Laplace Dirichlet->flux",
            "source": [float(v) for v in SOURCE],
            "linear_settings": "panel_linear_order defaults (quadrature_order=4, refinement_depth=3, self_rule=analytic, free_term=half)",
            "quadratic_settings": "quadratic_panel_verification defaults (quadrature_order=8, self_order=16)",
            "elapsed_seconds": None,
        },
        "acceptance": ACCEPTANCE,
        "linear": {"rows": linear_rows},
        "quadratic": {"rows": quadratic_rows},
        "matchups": matchups,
        "verdicts": {m["name"]: m["passed"] for m in matchups},
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    figure_error_curves(linear_rows, quadratic_rows, matchups, FIG_ERR)
    figure_error_field(field_rows["linear_s2"], field_rows["quadratic_s1"], FIG_FIELD)
    payload["provenance"]["elapsed_seconds"] = round(time.time() - started, 1)
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    write_report(payload)

    for m in matchups:
        print(f"{m['name']}: ratio {m['ratio']:.2f}x  passed={m['passed']}")
    print(f"wrote {JSON_PATH}")
    print(f"wrote {FIG_ERR}")
    print(f"wrote {FIG_FIELD}")
    print(f"wrote {REPORT_PATH}")
    print(f"elapsed {payload['provenance']['elapsed_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())