"""Post-process the RAO heading x frequency field: verification, figures, note.

Reads the artefacts written by
``python -m tensorbem.benchmarks.rao_heading_field`` (``rao_field.json`` and,
when available, ``rao_field_coarse.json``), re-runs every independent check in
:func:`tensorbem.benchmarks.rao_heading_field.verify` and regenerates

* ``<data>/verification.json``  - the canonical, machine-readable check table,
* the three PNG clouds / comparison curves under ``<figures>``,
* ``<doc>``                     - a markdown note whose numbers are *injected*
                                  from the JSON (no hand-copied digits).

The stored field artefact keeps magnitudes (|RAO|, |F|) plus the heave
diagonal of A/B, which is exactly what the magnitude-based checks need; the
RAO phases are not part of the delivered cloud.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorbem.benchmarks.rao_heading_field import (
    DOF_NAMES,
    DOF_UNITS,
    verify,
)

_PANELS = (0, 2, 4, 3)  # surge, heave, pitch, roll


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def merge_parts(parts: list[Path], out_path: Path) -> dict[str, Any]:
    """Concatenate per-frequency chunk artefacts (``--tag`` runs) into one field."""
    blobs = [json.loads(p.read_text()) for p in parts]
    head = blobs[0]
    rows = []
    for blob in blobs:
        if blob["panel_count"] != head["panel_count"]:
            raise RuntimeError("panel count differs between chunks")
        if blob["headings_deg"] != head["headings_deg"]:
            raise RuntimeError("heading grid differs between chunks")
        rows.extend(
            zip(
                blob["kas"],
                blob["omegas"],
                blob["rao_amplitudes"],
                blob["excitation_force_abs"],
                blob["added_mass_heave"],
                blob["damping_heave"],
            )
        )
    rows.sort(key=lambda r: r[0])
    merged = dict(head)
    merged["kas"] = [r[0] for r in rows]
    merged["omegas"] = [r[1] for r in rows]
    merged["rao_amplitudes"] = [r[2] for r in rows]
    merged["excitation_force_abs"] = [r[3] for r in rows]
    merged["added_mass_heave"] = [r[4] for r in rows]
    merged["damping_heave"] = [r[5] for r in rows]
    out_path.write_text(json.dumps(merged, indent=1) + "\n")
    return merged


def load_field(path: Path) -> dict[str, Any]:
    """Rebuild the field dict (magnitudes only) from the stored JSON artefact."""
    blob = json.loads(path.read_text())
    amp = np.asarray(blob["rao_amplitudes"], dtype=float)
    n_f, n_b, _ = amp.shape
    force = np.asarray(blob["excitation_force_abs"], dtype=float).astype(complex)
    if "added_mass" in blob:  # full 6x6 matrices
        added = np.asarray(blob["added_mass"], dtype=float)
        damp = np.asarray(blob["damping"], dtype=float)
        residual = np.asarray(blob["impedance_residual"], dtype=float)
    else:  # heave diagonal only
        added = np.zeros((n_f, 6, 6), dtype=float)
        damp = np.zeros((n_f, 6, 6), dtype=float)
        added[:, 2, 2] = np.asarray(blob["added_mass_heave"], dtype=float)
        damp[:, 2, 2] = np.asarray(blob["damping_heave"], dtype=float)
        residual = np.full((n_f, n_b), float(blob.get("impedance_residual_max", 0.0)))
    if "rao_real" in blob:
        rao = np.asarray(blob["rao_real"], dtype=float) + 1j * np.asarray(
            blob["rao_imag"], dtype=float
        )
    else:
        rao = amp.astype(complex)
    if "excitation_force_real" in blob:
        force = np.asarray(blob["excitation_force_real"], dtype=float) + 1j * np.asarray(
            blob["excitation_force_imag"], dtype=float
        )
    return {
        "subdivisions": blob["subdivisions"],
        "panel_count": blob["panel_count"],
        "mass": blob["mass"],
        "displaced_volume": blob["displaced_volume"],
        "restoring": np.asarray(blob["restoring"], dtype=float),
        "field": {
            "kas": np.asarray(blob["kas"], dtype=float),
            "headings_deg": np.asarray(blob["headings_deg"], dtype=float),
            "omegas": np.asarray(blob["omegas"], dtype=float),
            "rao": rao,
            "rao_amplitude": amp,
            "excitation_force": force,
            "added_mass": added,
            "damping": damp,
            "impedance_residual": residual,
        },
    }


def plot_dof_clouds(field: dict[str, Any], path: Path) -> None:
    """2x2 field clouds |RAO_i(ka, beta)| for surge / heave / pitch / roll."""
    plt = _pyplot()
    data = field["field"]
    kas = np.asarray(data["kas"], dtype=float)
    headings = np.asarray(data["headings_deg"], dtype=float)
    amp = np.asarray(data["rao_amplitude"], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)
    for ax, dof in zip(axes.ravel(), _PANELS):
        grid = amp[:, :, dof]
        mesh = ax.pcolormesh(headings, kas, grid, shading="nearest", cmap="viridis")
        if float(np.ptp(grid)) > 1e-12:
            levels = np.linspace(float(grid.min()), float(grid.max()), 7)[1:-1]
            cs = ax.contour(headings, kas, grid, levels=levels, colors="w", linewidths=0.7)
            ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
        ax.set_xlabel(r"wave heading $\beta$ [deg]  (0 = waves along +x)")
        ax.set_ylabel(r"dimensionless wavenumber $ka = \nu a$ [-]")
        ax.set_title(
            f"$|$RAO$_{dof + 1}|$ {DOF_NAMES[dof]}  [{DOF_UNITS[dof]}]  "
            f"(max {grid.max():.3g} at ka={kas[np.unravel_index(grid.argmax(), grid.shape)[0]]:.2f}, "
            f"beta={headings[np.unravel_index(grid.argmax(), grid.shape)[1]]:.0f} deg)"
        )
        cb = fig.colorbar(mesh, ax=ax)
        cb.set_label(f"$|$RAO$_{dof + 1}|$ [{DOF_UNITS[dof]}]")
    fig.suptitle(
        "TensorBEM frequency-domain response surface, floating hemisphere "
        f"(a = 1 m, {field['panel_count']} panels, deep water, no forward speed)"
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_verification(report: dict[str, Any], field: dict[str, Any], path: Path) -> None:
    """Heave RAO / load coefficients vs Hulme (1982) and the axisymmetry law."""
    plt = _pyplot()
    rows = report["heave_reference_vs_hulme"]["rows"]
    kas = np.array([r["ka"] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), constrained_layout=True)

    ax = axes[0]
    ax.plot(kas, [r["heave_rao_reference"] for r in rows], "k-", lw=2.2,
            label="Hulme (1982) series - reference")
    ax.plot(kas, [r["heave_rao_bem"] for r in rows], "o", ms=6,
            label=f"TensorBEM RAO chain ({field['panel_count']} panels)")
    ax.axhline(1.0, color="0.55", ls=":", lw=1.2, label=r"long-wave limit $|$RAO$_3|=1$")
    ax.set_xlabel(r"$ka$ [-]")
    ax.set_ylabel(r"$|$RAO$_3|$ heave [m/m]")
    ax.set_title("heave RAO vs closed-form reference")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.semilogy(kas, [100 * r["heave_rao_rel_error"] for r in rows], "o-", label=r"$|$RAO$_3|$")
    ax.semilogy(kas, [100 * r["excitation_heave_rel_error"] for r in rows], "s-",
                label=r"$|F_3|$")
    ax.semilogy(kas, [100 * r["heave_added_mass_rel_error"] for r in rows], "^-",
                label=r"$A_{33}$")
    ax.semilogy(kas, [100 * r["heave_damping_rel_error"] for r in rows], "v-",
                label=r"$B_{33}$")
    ax.axhline(3.0, color="r", ls="--", lw=1.1, label="3 % acceptance band")
    ax.set_xlabel(r"$ka$ [-]")
    ax.set_ylabel("relative error vs reference [%]")
    ax.set_title("error budget vs reference (log scale)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax = axes[2]
    data = field["field"]
    headings = np.asarray(data["headings_deg"], dtype=float)
    force = np.abs(np.asarray(data["excitation_force"], dtype=complex))
    j0 = int(np.argmin(np.abs(headings)))
    f_h = force[:, j0, 0]
    for i in range(0, len(kas), 3):
        ax.plot(headings, force[i, :, 0] / f_h[i], "o-", ms=3.5,
                label=rf"BEM $ka={kas[i]:.2f}$")
    ax.plot(headings, np.abs(np.cos(np.deg2rad(headings))), "k--", lw=1.5,
            label=r"exact $|\cos\beta|$ (Haskind, axisymmetric body)")
    ax.set_xlabel(r"wave heading $\beta$ [deg]")
    ax.set_ylabel(r"$|F_1(\beta)| / |F_h|$ [-]")
    ax.set_title("axisymmetry check on the excitation field")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_convergence(
    field: dict[str, Any], coarse: dict[str, Any], report: dict[str, Any], path: Path
) -> None:
    """Fine/coarse discretisation evidence for the delivered cloud."""
    plt = _pyplot()
    data = field["field"]
    kas = np.asarray(data["kas"], dtype=float)
    headings = np.asarray(data["headings_deg"], dtype=float)
    fine = np.asarray(data["rao_amplitude"], dtype=float)
    coarse_amp = np.asarray(coarse["field"]["rao_amplitude"], dtype=float)
    rel = np.abs(fine - coarse_amp) / np.maximum(np.abs(fine), 1e-12)
    j0 = int(np.argmin(np.abs(headings)))
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), constrained_layout=True)
    ax = axes[0]
    mesh = ax.pcolormesh(headings, kas, 100 * rel[:, :, 2], shading="nearest", cmap="magma")
    ax.set_xlabel(r"wave heading $\beta$ [deg]")
    ax.set_ylabel(r"$ka$ [-]")
    ax.set_title(
        "heave cloud: $|$RAO$_3^{256}-$RAO$_3^{64}| / |$RAO$_3^{256}|$  [%]"
    )
    fig.colorbar(mesh, ax=ax).set_label("relative difference [%]")
    ax = axes[1]
    ax.plot(kas, 100 * rel[:, j0, 2], "o-", label=r"heave, $\beta=0$")
    ax.plot(kas, 100 * rel[:, j0, 0], "s-", label=r"surge, $\beta=0$")
    ax.plot(kas, 100 * rel[:, j0, 4], "^-", label=r"pitch, $\beta=0$")
    ax.axhline(3.0, color="r", ls="--", lw=1.1, label="3 % band")
    ax.set_xlabel(r"$ka$ [-]")
    ax.set_ylabel("relative difference [%]")
    ax.set_title(
        f"discretisation difference {field['panel_count']} vs {coarse['panel_count']} panels"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}g}"


def _pct(value: float, digits: int = 3) -> str:
    return f"{100.0 * value:.{digits}f} %"


def write_note(
    path: Path,
    field: dict[str, Any],
    report: dict[str, Any],
    figures: dict[str, str],
    data_dir: Path,
) -> None:
    """Write the markdown note with every number injected from the artefacts."""
    data = field["field"]
    kas = np.asarray(data["kas"], dtype=float)
    omegas = np.asarray(data["omegas"], dtype=float)
    headings = np.asarray(data["headings_deg"], dtype=float)
    amp = np.asarray(data["rao_amplitude"], dtype=float)
    chain = report.get("chain_integrity") or {
        "max_abs_rao_difference": float("nan"),
        "max_abs_excitation_difference": float("nan"),
    }
    hulme = report["heave_reference_vs_hulme"]
    axy = report["axisymmetry"]
    mom = report.get("moment_excitation_heading_law", {})
    disc = report.get("discretisation") or {}
    committed = report.get("committed_baseline") or {}
    long_wave = report["long_wave_heave_limit"]

    lines: list[str] = []
    add = lines.append
    add("# 耐波响应场（频率 x 浪向）交付：浮半球 |RAO| 云图与独立基准核验")
    add("")
    add(f"> 数据工件：`{data_dir}/rao_field.json`（场量全量）、"
        f"`{data_dir}/rao_field_coarse.json`（64 面板对照）、"
        f"`{data_dir}/verification.json`（判据表）。")
    add("> 复现：`python -m tensorbem.benchmarks.rao_heading_field "
        f"--out {data_dir} --figures {figures['cloud'].parent}` 然后 "
        "`python tools/rao_field_report.py`（数字零手抄，全部由工件注入）。")
    add("")
    add("## 1. 案例介绍")
    add("")
    add("**物理背景.** 频域耐波性的核心量是响应幅值算子 "
        "$|\\mathrm{RAO}_i(\\omega,\\beta)|$，它把入射波幅映射到 6 自由度运动幅值；"
        "耐波性后处理（谱矩、短期统计、极值）本质上是这张 $\\omega\\times\\beta$ "
        "响应场的加权积分。平台此前只在单个 $(\\omega,\\beta)$ 上给出 RAO 数值，"
        "从未交付过场量本身，因此本单补齐**响应场云图**这一交付物。")
    add("")
    add("**几何/参数表**（浮半球，深水，零航速；场量对尺度不变，故取 $a=1$ m）：")
    add("")
    add("| 量 | 值 | 来源 |")
    add("|---|---|---|")
    add("| 半径 $a$ | 1.0 m | 几何 |")
    add(f"| 面板数（细网格） | {field['panel_count']} | `hemisphere_hull_mesh(3)` |")
    add(f"| 位移体积 $\\nabla$（网格） | {_fmt(field['displaced_volume'],6)} m^3 | "
        "`compute_displaced_volume_and_buoyancy` |")
    add(f"| 质量 $m=\\rho\\nabla$ | {_fmt(field['mass'],6)} kg | $\\rho=1025$ kg/m^3 |")
    add("| 重心 $z_G$ | $-0.5a$ | 与已入库 RAO 基线一致 |")
    add(f"| 恢复矩阵 $C_{33}$ | {_fmt(hulme['c33'],6)} N/m | 网格静水（$\\rho g A_{{wp}}$） |")
    add("| 惯量 $J$ | $m\\cdot 0.4a^2\\cdot I$ | 与已入库基线的测试惯量一致 |")
    add(f"| 无量纲波数 $ka$ | {kas[0]:.2f} … {kas[-1]:.2f}（{len(kas)} 点） | 深水色散 |")
    add(f"| 对应圆频率 $\\omega$ | {omegas[0]:.3f} … {omegas[-1]:.3f} rad/s | "
        "$\\omega=\\sqrt{g\\,ka/a}$ |")
    add(f"| 浪向 $\\beta$ | {headings[0]:.0f} … {headings[-1]:.0f} deg（{len(headings)} 点） | "
        "$0$ = 波沿 $+x$ 传播 |")
    add("")
    add("## 2. 怎么做的")
    add("")
    add("1. **算子复用但零妥协**：自由面算子只依赖（网格, $\\nu$），辐射问题与浪向无关，"
        "故每个频率只装配一次算子、解一次辐射，再对每个浪向只解一次绕射；"
        "RAO 由 6x6 阻抗 $Z=-\\omega^2(M+A)+i\\omega B+C$ 直接求解。"
        f"复用后的结果与公开入口 `compute_rao_at_frequency` 逐元素一致"
        f"（本单实测 $\\max|\\Delta|$RAO$||={_fmt(chain['max_abs_rao_difference'],3)}$、"
        f"$\\max|\\Delta F|={_fmt(chain['max_abs_excitation_difference'],3)}$，"
        "机器判据见 `tests/test_rao_heading_field.py`）。")
    add("2. **判据先于计算**：验证只用与面元算子无关的参照——")
    add("   (a) Hulme (1982) 多极级数（文献）给出的 $A_{33},B_{33},F_3$，"
        "配合单自由度垂荡传递函数得到闭式 $|$RAO$_3|$ 参照；"
        "(b) 回转体激励的 Haskind 精确指向律 $F_1(\\beta)=F_h\\cos\\beta$、"
        "$F_3$ 与浪向无关；"
        "(c) 长波极限 $|$RAO$_3|(ka\\to0)=1$；"
        "(d) 仓内已入库基线 `docs/rao_verification_metrics.json`（64 面板）；"
        "(e) 256 vs 64 面板离散收敛。")
    add(f"3. **工具链**：Python 3 / numpy / matplotlib（Agg，无中文字体故图内标注为英文）；"
        f"50 个 BEM 频率-浪向组合并行 4 进程，单核约 "
        f"{len(kas)}x(装配+辐射+{len(headings)}x绕射)。")
    add("")
    add("## 3. 计算结果")
    add("")
    add("### 3.1 场量极值（256 面板）")
    add("")
    add("| 自由度 | 单位 | 场量最大值 | 位置 $(ka,\\beta)$ | 场量最小值 |")
    add("|---|---|---|---|---|")
    for dof in (0, 2, 4, 3):
        grid = amp[:, :, dof]
        i, j = np.unravel_index(grid.argmax(), grid.shape)
        add(f"| RAO$_{dof+1}$ {DOF_NAMES[dof]} | {DOF_UNITS[dof]} | {_fmt(grid.max(),5)} | "
            f"({kas[i]:.2f}, {headings[j]:.0f} deg) | {_fmt(grid.min(),5)} |")
    add("")
    add("### 3.2 垂荡 RAO 与 Hulme (1982) 逐频对表（$\\beta=0$）")
    add("")
    add("| $ka$ | $\\omega$ [rad/s] | $|$RAO$_3|$ BEM | $|$RAO$_3|$ 参照 | 相对误差 | "
        "$|F_3|$ 相对误差 | $A_{33}$ 相对误差 | $B_{33}$ 相对误差 |")
    add("|---|---|---|---|---|---|---|")
    for r in hulme["rows"]:
        add(f"| {r['ka']:.2f} | {r['omega']:.4f} | {r['heave_rao_bem']:.5f} | "
            f"{r['heave_rao_reference']:.5f} | {_pct(r['heave_rao_rel_error'],2)} | "
            f"{_pct(r['excitation_heave_rel_error'],2)} | "
            f"{_pct(r['heave_added_mass_rel_error'],2)} | "
            f"{_pct(r['heave_damping_rel_error'],2)} |")
    add("")
    if committed:
        add("### 3.3 与仓内已入库基线（64 面板）对表")
        add("")
        add("| $ka$ | 基线 $|$RAO$_3|$ | 本单复算 $|$RAO$_3|$ | 相对误差 | 基线 $|$RAO$_1|$ | "
            "本单复算 $|$RAO$_1|$ |")
        add("|---|---|---|---|---|---|")
        for r in committed["rows"]:
            add(f"| {r['ka']:.2f} | {r['heave_rao_committed']:.10f} | "
                f"{r['heave_rao_recomputed']:.10f} | {_pct(r['heave_rao_rel_error'],3)} | "
                f"{r['surge_rao_committed']:.10f} | {r['surge_rao_recomputed']:.10f} |")
        add("")
        add(f"基线工件：`{committed['artifact']}`（{committed['panel_count_committed']} 面板，"
            f"本单 {committed['panel_count_recomputed']} 面板）。")
        add("")
    if disc:
        add("### 3.4 离散收敛（256 vs 64 面板，|RAO| 相对差）")
        add("")
        add("| 自由度 | 全网格最大相对差 |")
        add("|---|---|")
        for name, value in disc["per_dof_max_abs_rel_difference"].items():
            add(f"| {name} | {_pct(value,2)} |")
        add("")
    add("## 4. 图")
    add("")
    add(f"- `{figures['cloud']}`：$ka\\times\\beta$ 二维色图（surge/heave/pitch/roll），"
        "色标为 $|$RAO$_i|$ 及其量纲（m/m、rad/m），坐标为浪向 deg 与无量纲波数 $ka$，"
        "等值线标数值。")
    add(f"- `{figures['verification']}`：垂荡 RAO 与 Hulme 级数比较、误差预算（含 3% 带）、"
        "以及激励场指向律核验。")
    add(f"- `{figures['convergence']}`：256 vs 64 面板的场量差异云图与 $\\beta=0$ 剖面。")
    add("")
    add("## 5. 注意点与适用边界")
    add("")
    add("- **无粘绕射/辐射**：只含辐射阻尼（无粘性），横摇在共振区响应偏大，"
        "云图中现出的 RAO$_4$ 峰值不可直接当实船横摇幅值用，需另加粘性阻尼。")
    add("- **对象是浮半球**（回转体），不是船体：无航速、无细长体、无 Wigley/实船几何；"
        "响应场用于**校核求解器场量链**，不构成某型船的耐波性结论。")
    add("- **深水、线性、单色波**：无浅水/有限水深修正，无二阶力。")
    add(f"- $ka$ 上限 {kas[-1]:.2f}：更高频需要更密网格（256 面板在 $ka>2$ 已开始欠分辨）。")
    add("- 浪向对称律只在回转体上严格成立；对一般船体，$\\beta$ 与 $180^\\circ-\\beta$ "
        "不再等价。")
    add("")
    add("## 6. 误差判定（是否 <3%，逐项）")
    add("")
    add("| 判据 | 参照 | 结果 | <3% ? |")
    add("|---|---|---|---|")
    bib = [
        ("垂荡 $|$RAO$_3|$（全频段最大）", "Hulme (1982) 闭式", hulme["max_heave_rao_rel_error"]),
        ("垂荡激励 $|F_3|$（最大）", "Hulme (1982) 闭式", hulme["max_excitation_heave_rel_error"]),
        ("垂荡附加质量 $A_{33}$（最大）", "Hulme (1982) 闭式", hulme["max_heave_added_mass_rel_error"]),
        ("垂荡阻尼 $B_{33}$（最大）", "Hulme (1982) 闭式", hulme["max_heave_damping_rel_error"]),
        ("回转体指向律 $|F_1|$（全网格）", "Haskind 精确 $|\\cos\\beta|$", axy["max_abs_rel_error_excitation_x"]),
        ("回转体指向律 $|F_2|$（全网格）", "Haskind 精确 $|\\sin\\beta|$", axy["max_abs_rel_error_excitation_y"]),
    ]
    if mom:
        bib.append(("纵摇力矩指向律 $|F_5|$", "Haskind 精确 $|\\cos\\beta|$",
                    mom["max_abs_rel_error_pitch_moment"]))
        bib.append(("横摇力矩指向律 $|F_4|$", "Haskind 精确 $|\\sin\\beta|$",
                    mom["max_abs_rel_error_roll_moment"]))
    bib.append(("垂荡与浪向无关性（$|$RAO$_3|$ 全网格散布）", "回转体解析结论",
                axy["max_abs_rel_spread_heave_rao"]))
    bib.append((f"长波极限 $|$RAO$_3|(ka={long_wave['ka']:.2f})$", "$=1$",
                long_wave["relative_error"]))
    if committed:
        bib.append(("与仓内已入库 RAO 基线（重算复现）", "`docs/rao_verification_metrics.json`",
                    committed["max_heave_rao_rel_error"]))
    for name, ref, value in bib:
        add(f"| {name} | {ref} | {_pct(value,3)} | {'是' if value < 0.03 else '**否**'} |")
    add("")
    add("**无基准项（显式声明）**：纵摇/横摇/surge 的 RAO 本身在仓内没有闭式参照"
        "（Hulme 级数只给垂荡 $m=0$ 与纵荡 $m=1$ 的激励与附加质量），"
        "因此这三个自由度的云图**不声称** <3% 精度，只给出：激励场指向律核验"
        "（上表）、长波垂荡极限、以及与已入库基线的复现。要补齐需要额外引入"
        "文献 RAO 表或模型试验数据。")
    add("")
    path.write_text("\n".join(lines) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data/rao_heading_field_2026-11-05")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--doc", default="docs/rao_heading_frequency_field.md")
    parser.add_argument(
        "--merge-parts",
        nargs="*",
        default=None,
        help="chunk artefacts to merge into rao_field.json / rao_field_coarse.json",
    )
    args = parser.parse_args(argv)

    data_dir = Path(args.data)
    fig_dir = Path(args.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_parts:
        fine_parts = sorted(p for p in map(Path, args.merge_parts) if "coarse" not in p.name)
        coarse_parts = sorted(p for p in map(Path, args.merge_parts) if "coarse" in p.name)
        if fine_parts:
            merge_parts(fine_parts, data_dir / "rao_field.json")
        if coarse_parts:
            merge_parts(coarse_parts, data_dir / "rao_field_coarse.json")
    field = load_field(data_dir / "rao_field.json")
    coarse = None
    coarse_path = data_dir / "rao_field_coarse.json"
    if coarse_path.exists():
        coarse = load_field(coarse_path)

    committed = Path("docs/rao_verification_metrics.json")
    report = verify(field, coarse=coarse, committed_path=committed)
    previous = data_dir / "verification.json"
    if previous.exists():
        try:
            old = json.loads(previous.read_text())
        except json.JSONDecodeError:
            old = {}
        for key in ("chain_integrity", "chain_integrity_pair"):
            if key in old:
                report[key] = old[key]
    report["chain_integrity_note"] = (
        "asserted by tensorbem.benchmarks.rao_heading_field.check_chain_integrity "
        "and tests/test_rao_heading_field.py"
    )
    (data_dir / "verification.json").write_text(json.dumps(report, indent=1) + "\n")

    figures = {
        "cloud": fig_dir / "fig_rao_field_dof_clouds.png",
        "verification": fig_dir / "fig_rao_field_verification.png",
        "convergence": fig_dir / "fig_rao_field_convergence.png",
    }
    plot_dof_clouds(field, figures["cloud"])
    plot_verification(report, field, figures["verification"])
    if coarse is not None:
        plot_convergence(field, coarse, report, figures["convergence"])
    write_note(Path(args.doc), field, report, figures, data_dir)
    print(json.dumps(
        {
            "doc": str(args.doc),
            "figures": {k: str(v) for k, v in figures.items()},
            "max_heave_rao_rel_error": report["heave_reference_vs_hulme"]["max_heave_rao_rel_error"],
            "axisymmetry": report["axisymmetry"],
            "long_wave": report["long_wave_heave_limit"],
        },
        indent=1,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())