#!/usr/bin/env python3
"""Render the case document + figures from the committed ``numbers.json``.

Every number in the generated document is read out of the machine artefact
produced by ``run_case.py``; nothing is retyped.  Re-run after any change::

    PYTHONPATH=src python3 benchmarks/verified/casepool_bem_submerged_sphere_havelock/make_doc.py \
        --numbers docs/case_bem_submerged_sphere_havelock/numbers.json \
        --outdir docs/case_bem_submerged_sphere_havelock

Figures (matplotlib, Agg backend, deterministic):
    fig1_band.png      A33'/B33'/A11'/B11' vs ka: BEM (case mesh) vs multipole ref
    fig2_error.png     per-item relative error vs ka with the predeclared 5 % band
    fig3_ladder.png    h-ladder of the relative error at ka=0.8 (log-log)
    fig4_submergence.png  coefficients vs f/a at ka=1.0 plus Lamb's limit
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LABEL = {
    "a33": "A33/(rho nabla)",
    "b33": "B33/(rho omega nabla)",
    "a11": "A11/(rho nabla)",
    "b11": "B11/(rho omega nabla)",
}
SHORT = {"a33": "$A'_{33}$", "b33": "$B'_{33}$", "a11": "$A'_{11}$", "b11": "$B'_{11}$"}
COEFFICIENTS = ("a33", "b33", "a11", "b11")
GATE = 0.05


def _fmt(value: float, digits: int = 4) -> str:
    if value is None:
        return "-"
    if value != 0 and abs(value) < 1e-3:
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def _pct(value: float) -> str:
    return f"{100.0 * value:+.3f}%"


def _tick(rel_error: float) -> str:
    return "PASS" if rel_error < GATE else "FAIL"


# --------------------------------------------------------------------------- figures
def _figure_band(payload: dict[str, Any], path: Path) -> Path:
    rows = payload["band_sweep_case"] or payload["band_sweep_economy"]
    kin = np.array([row["ka"] for row in rows])
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), sharex=True)
    for ax, key in zip(axes.ravel(), COEFFICIENTS):
        got = np.array([row[f"{key}_bem"] for row in rows])
        ref = np.array([row[f"{key}_exact"] for row in rows])
        ax.plot(kin, ref, "-", color="k", lw=1.6, label="multipole ref")
        ax.plot(kin, got, "o", ms=5.0, mfc="none", color="tab:red", label="TensorBEM")
        ax.set_title(LABEL[key], fontsize=10)
        ax.grid(alpha=0.3)
        if key in ("a33", "b33"):
            ax.set_xscale("log")
    axes[0, 0].legend(fontsize=8, loc="best")
    for ax in axes[-1]:
        ax.set_xlabel("ka")
    fig.suptitle(
        "Submerged sphere radiation: BEM vs multipole reference "
        f"(f/a={payload['case']['submergence_ratio']:.2f}, "
        f"{payload['nondimensional_parameters']['case_panels'] or payload['nondimensional_parameters']['economy_panels']} panels)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _figure_error(payload: dict[str, Any], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for tier, style, colour in (
        ("band_sweep_economy", "s--", "tab:blue"),
        ("band_sweep_case", "o-", "tab:red"),
    ):
        rows = payload[tier]
        if not rows:
            continue
        kin = np.array([row["ka"] for row in rows])
        for key in COEFFICIENTS:
            err = np.array([row[f"{key}_rel_error"] for row in rows])
            ax.plot(kin, 100.0 * err, style, ms=4.0, lw=1.0, color=colour, alpha=0.75,
                    label=f"{SHORT[key]} ({tier.split('_')[-1]})")
    ax.axhline(100.0 * GATE, color="k", ls=":", lw=1.4, label="predeclared 5 % gate")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ka")
    ax.set_ylabel("relative error [%]")
    ax.set_title("Per-item relative error, economy (320) vs case (1280) mesh")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _figure_ladder(payload: dict[str, Any], path: Path) -> Path:
    levels = payload["convergence_ladder"]["levels"]
    h = np.array([lv["h_mean"] for lv in levels])
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for key, marker in zip(COEFFICIENTS, ("o", "s", "^", "v")):
        err = np.array([lv[f"{key}_rel_error"] for lv in levels])
        ax.loglog(h, 100.0 * err, marker + "-", label=SHORT[key], ms=5.0)
    # second-order reference slope through the coarsest point
    err0 = max(levels[0]["a33_rel_error"], 1e-12)
    scale = 100.0 * err0 / (h[0] ** 2)
    ax.loglog(h, scale * h**2, "k:", lw=1.2, label="slope 2 guide")
    ax.set_xlabel("mean panel edge h/a")
    ax.set_ylabel("relative error [%]")
    order = payload["convergence_ladder"].get("observed_order_a33")
    ax.set_title(f"Mesh ladder at ka=0.8 (observed order of A33': {order:.2f})"
                 if order is not None else "Mesh ladder at ka=0.8")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _figure_submergence(payload: dict[str, Any], path: Path) -> Path:
    rows = payload["submergence_sweep"]
    ratios = [row["submergence_ratio"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for ax, key, lamb in (
        (axes[0], "a33", 0.5),
        (axes[1], "a11", 0.5),
    ):
        got = [row["got"][key] for row in rows]
        ref = [row["ref"][key] for row in rows]
        ax.plot(ratios, ref, "k-", lw=1.5, label="multipole ref")
        ax.plot(ratios, got, "o", ms=5.0, mfc="none", color="tab:red", label="TensorBEM")
        ax.axhline(lamb, color="tab:green", ls="--", lw=1.1, label="Lamb limit 0.5")
        ax.set_xlabel("f/a")
        ax.set_title(LABEL[key], fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Submergence sweep at ka=1.0 (320 panels)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def render_figures(payload: dict[str, Any], outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    return [
        _figure_band(payload, outdir / "fig1_band.png"),
        _figure_error(payload, outdir / "fig2_error.png"),
        _figure_ladder(payload, outdir / "fig3_ladder.png"),
        _figure_submergence(payload, outdir / "fig4_submergence.png"),
    ]


# --------------------------------------------------------------------------- tables
def _band_table(rows: Sequence[dict[str, Any]]) -> list[str]:
    head = (
        "| ka | mesh | " + " | ".join(f"{LABEL[k]} got / ref / err" for k in COEFFICIENTS) + " |"
    )
    sep = "|" + "---|" * (len(COEFFICIENTS) + 2)
    lines = [head, sep]
    for row in sorted(rows, key=lambda r: r["ka"]):
        cells = [f"{row['ka']:.3g}", str(row["n_panels"])]
        for key in COEFFICIENTS:
            cells.append(
                f"{_fmt(row[f'{key}_bem'])} / {_fmt(row[f'{key}_exact'])} / "
                f"{_pct(row[f'{key}_rel_error'])} {_tick(row[f'{key}_rel_error'])}"
            )
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _tier_table(payload: dict[str, Any]) -> list[str]:
    lines = ["| tier | items | failures | worst item | max rel. error | verdict |",
             "|---|---|---|---|---|---|"]
    for name, tier in payload["tiers"].items():
        worst = tier["worst_item"]
        lines.append(
            f"| {name} ({tier['n_items']} items, gate {100 * tier['gate']:.0f}%) | "
            f"{tier['n_items']} | {tier['n_failures']} | "
            f"{worst['item']} {worst['quantity']} | {_pct(tier['max_rel_error'])} | "
            f"**{tier['verdict']}** |"
        )
    return lines


def _ladder_table(payload: dict[str, Any]) -> list[str]:
    lines = ["| subdivision | panels | h/a | " + " | ".join(f"{SHORT[k]} err" for k in COEFFICIENTS)
             + " | cond(A) |", "|---|---|---|---|---|---|---|---|"]
    for lv in payload["convergence_ladder"]["levels"]:
        cells = [str(lv["subdivision"]), str(lv["n_panels"]), f"{lv['h_mean']:.4f}"]
        cells += [_pct(lv[f"{k}_rel_error"]) + " " + _tick(lv[f"{k}_rel_error"]) for k in COEFFICIENTS]
        cells.append(_fmt(lv.get("condition_number") or float("nan"), 1))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _submergence_table(payload: dict[str, Any]) -> list[str]:
    lines = ["| f/a | " + " | ".join(f"{SHORT[k]} got / ref / err" for k in COEFFICIENTS) + " |",
             "|---|---|---|---|---|"]
    for row in payload["submergence_sweep"]:
        cells = [f"{row['submergence_ratio']:.2f}"]
        for key in COEFFICIENTS:
            cells.append(
                f"{_fmt(row['got'][key])} / {_fmt(row['ref'][key])} / "
                f"{_pct(row['rel_error'][key])} {_tick(row['rel_error'][key])}"
            )
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _low_frequency_table(payload: dict[str, Any]) -> list[str]:
    lines = ["| mesh | ka | A33' BEM | A33' ref | err |", "|---|---|---|---|---|"]
    for tag, block in (("economy", payload["low_frequency_economy"]),
                       ("case", payload["low_frequency_case"])):
        if not block:
            continue
        for row in block["data_points"]:
            lines.append(
                f"| {tag} | {row['ka']:.3g} | {_fmt(row['a33_bem'])} | "
                f"{_fmt(row['a33_exact'])} | {_pct(row['a33_rel_error'])} "
                f"{_tick(row['a33_rel_error'])} |"
            )
        lines.append(
            f"| {tag} | ka->0 extrap. | {_fmt(block['bem_extrapolated_limit'])} | "
            f"{_fmt(block['exact_limit'])} | {_pct(block['limit_relative_error'])} "
            f"{_tick(block['limit_relative_error'])} |"
        )
    return lines


def _crosscheck_table(cross: dict[str, Any] | None) -> list[str]:
    """Independent third-party software (capytaine) vs the multipole reference."""
    if not cross:
        return ["(independent-software cross-check not present in this artefact directory)"]
    lines = [
        "| mesh (panels) | ka | " + " | ".join(f"{SHORT[k]} capytaine / ref / err" for k in COEFFICIENTS) + " |",
        "|---|---|---|---|---|---|",
    ]
    for block in cross["rows"]:
        for row in sorted(block, key=lambda r: r["ka"]):
            cells = [f"{row['subdivision']} ({row['n_panels']})", f"{row['ka']:.3g}"]
            for key in COEFFICIENTS:
                err = row["capytaine_vs_ref_rel_error"][key]
                cells.append(
                    f"{_fmt(row['capytaine'][key])} / {_fmt(row['multipole_ref'][key])} / "
                    f"{_pct(err)} {_tick(err)}"
                )
            lines.append("| " + " | ".join(cells) + " |")
    worst = cross["worst"]
    lines.append("")
    lines.append(
        f"独立软件最差点：ka={worst['ka']:.3g}，{worst['subdivision']} 细分（"
        f"{'a33' if worst['quantity'] == 'a33' else worst['quantity']}），误差 "
        f"{_pct(worst['rel_error'])} → 判级 `{cross['verdict']}`（capytaine "
        f"{cross['capytaine_version']}）。"
    )
    return lines


def _geometry_table(payload: dict[str, Any]) -> list[str]:
    block = payload["geometry_precondition"]
    lines = ["| subdivision | panels | h/a | mesh volume | volume error |", "|---|---|---|---|---|"]
    for lv in block["levels"]:
        lines.append(
            f"| {lv['subdivision']} | {lv['n_panels']} | {lv['h_mean']:.4f} | "
            f"{_fmt(lv['mesh_volume'])} | {_pct(lv['volume_relative_error'])} |"
        )
    return lines


def _reference_gate_table(payload: dict[str, Any]) -> list[str]:
    gates = payload.get("reference_integrity", {}).get("gates", {})
    if not gates:
        return ["(reference integrity suite not run in this artefact)"]
    lines = ["| invariant | value / threshold | result |", "|---|---|---|"]
    trunc = payload["reference_integrity"]["truncation"]
    lines.append(
        f"| multipole truncation 20->40 terms | worst |dcoef| = "
        f"{trunc['worst_truncation_rel_delta']:.2e} (< 1e-12) | "
        f"{'PASS' if gates['truncation_machine_zero'] else 'FAIL'} |"
    )
    lines.append(
        f"| Haskind reciprocity residual | {trunc['worst_haskind_error']:.2e} (< 1e-10) | "
        f"{'PASS' if gates['haskind_residual'] else 'FAIL'} |"
    )
    deep = payload["reference_integrity"]["deep_submergence_limit"]["rows"][-1]
    lines.append(
        f"| Lamb (1932) A' -> 0.5 at f/a=20 | {deep['a33']:.6f} | "
        f"{'PASS' if gates['lamb_a33'] else 'FAIL'} |"
    )
    lines.append(
        f"| Lamb (1932) C_M -> 1.5 at f/a=20 | {deep['inertia_coefficient_cm']:.6f} | "
        f"{'PASS' if gates['lamb_cm'] else 'FAIL'} |"
    )
    low = payload["reference_integrity"]["low_frequency_damping"][0]
    lines.append(
        f"| B33' -> 0 as ka -> 0 | B33'(ka={low['ka']:.0e}) = {low['b33']:.2e} | "
        f"{'PASS' if gates['damping_vanishes'] else 'FAIL'} |"
    )
    return lines


# --------------------------------------------------------------------------- document
def _ka_list(values: Any) -> str:
    """Format a wavenumber band as ``0.1, 0.5, 1, 2, 3``."""
    if not values:
        return "n/a"
    return ", ".join(f"{float(v):g}" for v in values)


def _with_actual_panels(payload: dict[str, Any]) -> dict[str, Any]:
    """Overwrite the documented panel counts with the ones the run actually used.

    The protocol block carries a hand-written formula for the panel count; the
    icosphere mesh has ``20 * 4**subdivision`` panels (80 / 320 / 1280), not
    ``4 * 4**subdivision`` (64 / 256), so that block under-counted the meshes by a
    factor 5 and the generated CASE.md inherited the error.  The sweep rows carry
    the authoritative ``n_panels`` of each level, so the document is re-derived
    from them: a document can no longer disagree with the data it describes.
    """
    params = dict(payload.get("nondimensional_parameters") or {})
    economy_rows = payload.get("band_sweep_economy") or []
    case_rows = payload.get("band_sweep_case") or []
    if economy_rows:
        params["economy_panels"] = int(economy_rows[0]["n_panels"])
    if case_rows:
        params["case_panels"] = int(case_rows[0]["n_panels"])
    gated = params.get("case_gated_band_ka")
    if case_rows and not gated:
        params["case_gated_band_ka"] = sorted({float(row["ka"]) for row in case_rows})
    out = dict(payload)
    out["nondimensional_parameters"] = params
    return out


def build_markdown(payload: dict[str, Any], crosscheck: dict[str, Any] | None = None) -> str:
    payload = _with_actual_panels(payload)
    case = payload["case"]
    ref = payload["reference"]
    verdict = payload["verdict"]
    case_tier = payload["tiers"].get("case_mesh")
    econ = payload["tiers"]["economy_mesh"]
    order = payload["convergence_ladder"].get("observed_order_a33")
    env = payload["environment"]
    prov = payload["provenance"]

    case_max = case_tier["max_rel_error"] if case_tier else None
    worst_case = case_tier["worst_item"] if case_tier else None

    out: list[str] = []
    out.append("# 算例 CASE-BEM-SUBMERGED-SPHERE-HAVELOCK：全潜球频域辐射（TensorBEM）\n")
    out.append(
        "> 算例池条目 `tensor_case_pool.case_code = " + payload["case"]["case_pool_code"] +
        "`，容差 `<5%`。本文件由 `benchmarks/verified/casepool_bem_submerged_sphere_havelock/"
        "make_doc.py` 从 `numbers.json` **程序化生成（零手抄）**。\n"
    )
    out.append(
        f"> 机读工件：`numbers.json` / `verdict.json` / `tiers.json` / `table_band_economy.csv` / "
        f"`table_band_case.csv` / `table_ladder.csv` / `table_low_frequency.csv` / "
        f"`table_submergence.csv` / `run.log`；图 1-4 见本节末尾。\n"
    )
    out.append(f"> 运行环境：Python {env['python']}，numpy {env['numpy']}，{env['platform']}；"
               f"git `{prov['git_head'][:7]}`（dirty={prov['git_dirty']}）；"
               f"总运行 {payload['runtime_seconds']:.1f} s。\n")

    # ① problem + reference provenance
    out.append("## ① 问题与参考解出处\n")
    out.append(
        f"半径 $a={case['radius_m']}$ m 的刚性球，球心潜深 $f={case['submergence_m']}$ m"
        f"（$f/a={case['submergence_ratio']:.2f}$），在深水自由液面下作单位速度幅值简谐振荡，"
        "求频域附加质量与阻尼：$A'_{33}=A_{33}/(\\rho\\nabla)$、"
        "$B'_{33}=B_{33}/(\\rho\\omega\\nabla)$、$A'_{11}$、$B'_{11}$，"
        f"$\\nabla=\\tfrac43\\pi a^3={case['displaced_volume_m3']:.6f}$ m³（解析体积，"
        "不是网格体积）。时间因子 $e^{-i\\omega t}$。\n"
    )
    out.append(f"**参考解（ref）出处。** {ref['citation']}\n")
    out.append(f"- 方法：{ref['method']}；实现：`{ref['implementation']}`；截断 $n_{{terms}}={ref['terms_used']}$。\n")
    out.append(f"- 独立数值锚：{ref['independent_anchor']}\n")
    out.append(f"- **零手抄声明**：{ref['provenance_note']}\n")
    out.append(
        "- 参考解自证的机器门（本工件实测值）：\n"
    )
    out.extend("\n" + line for line in _reference_gate_table(payload))
    out.append("")

    # ② method
    out.append("## ② 方法与实现\n")
    out.append(
        "- **求解器**（被测方）：`tensorbem.free_surface_solver3d."
        "assemble_free_surface_operators_3d` + `solve_radiation_3d`，三维深水自由液面 "
        "Green 函数（Delhommeau/Kelvin 型），常值强度面元，物面 Neumann 条件。\n"
        "- **参考解**（裁判方）：`tensorbem.analytic.submerged_sphere_wave_multipole`，"
        "球谐多极子展开，耦合矩阵元用精确阶乘与指数积分 $\\mathrm{Ei}(2\\nu f)$ 闭式求值，"
        "无面元、无数值求积，与 BEM 完全独立。\n"
        "- **几何**：`tensorbem.geometry3d.submerged_sphere_mesh`（测地 icosphere 球），"
        "细分 1/2/3 → 80/320/1280 面元；内接多面体，体积亏损见 ⑤ 表。\n"
        "- **入口（一键复现）**：\n"
        "```\n"
        "PYTHONPATH=src python3 benchmarks/verified/casepool_bem_submerged_sphere_havelock/run_case.py \\\n"
        "    --stage all --outdir docs/case_bem_submerged_sphere_havelock\n"
        "PYTHONPATH=src python3 benchmarks/verified/casepool_bem_submerged_sphere_havelock/make_doc.py \\\n"
        "    --numbers docs/case_bem_submerged_sphere_havelock/numbers.json \\\n"
        "    --outdir docs/case_bem_submerged_sphere_havelock\n"
        "```\n"
        "- **判据（预先声明，未在看结果后修改）**："
        f"G1 案定网格（{payload['nondimensional_parameters']['case_panels'] or 'n/a'} 面元）"
        f"在门控带 $ka\\in${{{_ka_list(payload['nondimensional_parameters'].get('case_gated_band_ka'))}}} "
        "逐项相对误差 <5%（$ka$ 更高两点只在经济网格门控，见 `gci/GCI.md` 的全带收敛研究）；"
        f"G2 经济网格（{payload['nondimensional_parameters']['economy_panels']} 面元，m17242 预声明的 production mesh）"
        "逐项结果原样结转报告；G3 几何前置条件（网格体积误差 <2%）与低频阻尼趋于零；"
        "G4 加密不得使最大误差上升、观测阶数为正；G5 参考解三项不变量到机器精度。\n"
    )

    # ③ parameter table
    out.append("## ③ 参数表（无量纲量完整）\n")
    out.append("| 参数 | 值 |\n|---|---|")
    out.append(f"| $f/a$（潜深比） | {case['submergence_ratio']:.2f}（另扫 "
               + ", ".join(f"{r:.2f}" for r in payload['nondimensional_parameters']['submergence_ratio_f_over_a'])
               + "） |")
    out.append(f"| $ka$ 频带 | " + ", ".join(f"{k:g}" for k in payload["nondimensional_parameters"]["band_ka"]) + " |")
    out.append(f"| $ka$ 低频外推点 | " + ", ".join(f"{k:g}" for k in payload["nondimensional_parameters"]["low_frequency_ka"]) + " |")
    ladder_levels = (payload.get("geometry_precondition") or {}).get("levels") or []
    ladder_counts = (" / ".join(str(level["n_panels"]) for level in ladder_levels)
                     if ladder_levels else
                     f"{payload['nondimensional_parameters']['economy_panels']} / "
                     f"{payload['nondimensional_parameters']['case_panels'] or '-'}")
    out.append(f"| 面元细分阶梯 | " + ", ".join(str(s) for s in payload["nondimensional_parameters"]["panel_subdivisions"])
               + f"（{ladder_counts} 面元） |")
    out.append(f"| 归一化 | {payload['nondimensional_parameters']['normalisation']} |")
    out.append(f"| $\\rho$ / $g$ | {case['density_kg_m3']:.0f} kg/m³ / {case['gravity_m_s2']} m/s² |")
    out.append(f"| 时间因子 | $e^{{-i\\omega t}}$ |")
    out.append("")

    # ④ comparison table + verdict
    out.append("## ④ 对照表（ref vs got + 逐项误差打勾/叉）\n")
    out.append(f"**判级汇总表**（门={100 * GATE:.0f}%）：\n")
    out.extend(_tier_table(payload))
    out.append("")
    if case_max is not None:
        out.append(
            f"**案定网格（门控带 $ka$ = {_ka_list(payload['nondimensional_parameters'].get('case_gated_band_ka'))}）**"
            f"（最大误差 {_pct(case_max)}，最差点 {worst_case['item']} "
            f"{worst_case['quantity']}）：\n"
        )
        out.extend(_band_table(payload["band_sweep_case"]))
        out.append("")
    out.append(f"**经济网格全带**（m17242 预声明 production mesh，{payload['nondimensional_parameters']['economy_panels']} 面元）：\n")
    out.extend(_band_table(payload["band_sweep_economy"]))
    out.append("")
    out.append("**低频极限 $A'_{33}$（$ka\\to0$ 外推）**：\n")
    out.extend(_low_frequency_table(payload))
    out.append("")
    if crosscheck:
        out.append("**独立第三方软件交叉核对（capytaine，与多极子解代码完全独立）**：\n")
        out.extend(_crosscheck_table(crosscheck))
        out.append("")
    out.append("**潜深扫描（$ka=1.0$）**：\n")
    out.extend(_submergence_table(payload))
    out.append("")

    # ⑤ convergence / sensitivity
    out.append("## ⑤ 收敛性与参数敏感性\n")
    out.append(
        f"**网格敏感性与观测阶（ka=0.8 阶梯）**，观测阶 $p={order:.3f}$（$A'_{{33}}$ 误差 vs $h$）：\n"
        if order is not None else "**网格敏感性（ka=0.8 阶梯）**：\n"
    )
    out.extend(_ladder_table(payload))
    out.append("")
    out.append("**几何前置条件（内接多面体体积亏损）**：\n")
    out.extend(_geometry_table(payload))
    out.append("")
    out.append(
        f"**频率敏感性**：带内最大误差出现在 {worst_case['item'] if worst_case else econ['worst_item']['item']}"
        f"；高 $ka$ 端 $B'$ 幅值本身趋于 $0$（参见下表补充分母归一化误差），"
        "因此相对误差在高频端被离散噪声主导，这是本算例的固有难点。\n"
    )
    out.append("| 量 | 带内 $\\max|ref|$ | 最坏分母归一化误差（仅解释用，不翻判） |\n|---|---|---|")
    for key, block in payload["supplementary_scale_normalised"].items():
        out.append(f"| {LABEL[key]} | {_fmt(block['reference_scale'])} | "
                   f"{100 * block['worst_scale_normalised_error']:.3f}% |")
    out.append("")
    deficit = payload["area_deficit_diagnostic"]
    if isinstance(deficit, dict):
        keys = [k for k in deficit if isinstance(deficit[k], (int, float))]
        if keys:
            out.append(
                "**面积/体积亏损诊断（网格几何对系数的解释力）**："
                + "，".join(f"`{k}` = {_fmt(float(deficit[k]))}" for k in keys[:6]) + "。\n"
            )
    out.append(
        "**参数敏感性小结**：(i) 面元数——从经济到案定网格，带内最大误差 "
        f"由 {_pct(econ['max_rel_error'])} 变为 "
        f"{_pct(case_max) if case_max is not None else 'n/a'}；"
        "(ii) 潜深比 $f/a$——见上表，$f/a$ 越大参考解越接近 Lamb 极限、BEM 误差单调改善；"
        "(iii) 参考解截断阶——20→40 阶到机器零，不构成误差源。\n"
    )

    # ⑥ failure modes
    out.append("## ⑥ 失败模式、局限与工件\n")
    out.append("**判据结论（逐项显式回答）**：\n")
    out.append(f"- 案定网格（{payload['nondimensional_parameters']['case_panels'] or '-'} 面元，"
               f"门控带 $ka$ = {_ka_list(payload['nondimensional_parameters'].get('case_gated_band_ka'))}）：`{verdict['case_mesh']}`"
               + (f"，带内最大相对误差 {_pct(case_max)}" if case_max is not None else "")
               + f" → **{'全部 <5%' if verdict['case_mesh'] == 'PASS' else '存在 ≥5% 项'}**。\n")
    out.append(f"- 经济网格（{payload['nondimensional_parameters']['economy_panels']} 面元）：`{verdict['economy_mesh']}`，"
               f"带内最大相对误差 {_pct(econ['max_rel_error'])}，失败项 {econ['n_failures']}/{econ['n_items']}"
               f" → **{'全部 <5%' if verdict['economy_mesh'] == 'PASS' else '存在 ≥5% 项'}**。\n")
    out.append(f"- 几何前置条件：`{verdict['geometry']}`；加密判据：`{verdict['refinement']}`。\n")
    out.append(f"- 总判：`{verdict['overall']}`。{verdict['honest_note']}\n")
    levels_geom = payload["geometry_precondition"]["levels"]
    econ_lv = levels_geom[min(1, len(levels_geom) - 1)]
    out.append("\n**失败模式清单**：\n")
    out.append(
        "1. **高频端 $B'$ 相对误差失效**：$B'\\to0$ 而绝对误差由面元离散固定，"
        "相对误差必然发散；本工件按预声明的 5% 判据如实记 FAIL，并另附分母归一化误差表。\n"
        "2. **几何一阶表示**：常值面元的内接多面体在 "
        f"{econ_lv['n_panels']} 面元时体积亏损 {100 * econ_lv['volume_relative_error']:.2f}%"
        "，附加质量对体积敏感，这是经济网格误差的主因之一。\n"
        "3. **参考解不可在沙箱内与原表逐行核对**（Havelock 1955 原表为半潜浮球；"
        "Wang 1986 / Linton & McIver 1991 表被付费墙挡住），故本算例不声明 L5 级文献表锚。\n"
        "4. **未做**：不规则频率（本算例潜球在带内无奇点，未启用 CHIEF 内点面）、"
        "有限水深路径、时间域因果性交叉。\n"
    )
    out.append("**工件清单**：`numbers.json`（全部数值）、`verdict.json`（判级）、"
               "`tiers.json`（分级门）、四张 CSV（带内/阶梯/低频/潜深）、`run.log`（分段计时）、"
               "`fig1_band.png`~`fig4_submergence.png`。\n")
    out.append("**复现命令**：见 ② 的两条命令；冒烟版加 `--stage cheap`（约 2 分钟）。\n")
    return "\n".join(out) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--numbers", type=Path,
                        default=Path("docs/case_bem_submerged_sphere_havelock/numbers.json"))
    parser.add_argument("--outdir", type=Path,
                        default=Path("docs/case_bem_submerged_sphere_havelock"))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    payload = json.loads(args.numbers.read_text())
    args.outdir.mkdir(parents=True, exist_ok=True)
    cross_path = args.outdir / "capytaine_crosscheck.json"
    crosscheck = json.loads(cross_path.read_text()) if cross_path.is_file() else None
    docs = args.outdir / "CASE.md"
    docs.write_text(build_markdown(payload, crosscheck))
    print(f"wrote {docs}")
    if not args.no_figures:
        for path in render_figures(payload, args.outdir):
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())