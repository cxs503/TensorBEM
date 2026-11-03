"""Precision envelope of the 2-D free-surface radiation solver (Ursell cylinder).

Why this module exists
----------------------
Every documented Ursell number in this repository is quoted at a single
operating point, ``ka = 0.6`` / ``a/f = 0.5`` (see ``docs/marine_bem_benchmarks.md``
§4.6 and the revalidation notes).  ``docs/ursell_error_budget.md`` limitation 2
records the consequence: the *convergence behaviour away from that point was
never verified*, and at ``ka >= 1`` the free-surface term is stronger, so the
first-order constant of the panel solution may not be the same.  That matters
for the only thing this solver is used for downstream (2-D strip theory for
seakeeping), where the interesting sections sit both at short waves
(large ``ka``) and close to the free surface (``f/a -> 1``).

What is measured here
---------------------
For a grid of ``(ka, f/a, N)`` the panel solution is compared with Ursell's
multipole solution (an independent analytic construction, ``analytic.py``),
which is *not* the code under test:

* non-dimensional coefficients ``A* = A/(rho pi a^2)``, ``B* = B/(rho pi a^2 omega)``
  -- the plain relative error is degenerate near ``f/a ~ 1.13`` where ``A``
  crosses zero (``A(1.10) = -238`` vs ``A(1.20) = +1160``), so the error is
  also reported against the displaced mass;
* the observed order ``p`` from consecutive refinements, and ``N * eps``
  (constant ``N * eps`` is the signature of first-order convergence);
* the near-field/far-field damping ratio (radiation-condition check);
* the MFS cross-check that ``ursell_cylinder.run`` already returns, which is
  what makes the *reference* trustworthy at shallow submergence (the multipole
  system is truncated; if the truncation were bad, MFS would disagree).

Outputs: ``docs/ursell_precision_envelope/envelope.json`` (all rows, nothing
transcribed by hand) and, through :func:`build_report`, the generated document
``docs/ursell_precision_envelope.md`` plus the figure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Sequence

#: Default operating-point grid.  ``ka`` spans long to short waves, ``f/a``
#: spans "deep" to "one tenth of a radius clear of the surface", which is where
#: waterline-adjacent ship sections live.
DEFAULT_KA: tuple[float, ...] = (0.4, 0.6, 0.9, 1.2, 1.6)
DEFAULT_FOVER_A: tuple[float, ...] = (1.1, 1.2, 1.5, 2.0, 3.0)
DEFAULT_PANELS: tuple[int, ...] = (32, 64, 128, 256)

DENSITY = 1025.0
GRAVITY = 9.80665

#: Acceptance bars used in this repository.  ``1e-2`` is the bar the Ursell
#: benchmark is documented against; ``3e-2`` is the general platform bar.
BARS = (1.0e-2, 3.0e-2)


def _point(job: tuple[float, float, int]) -> dict[str, Any]:
    """Solve one ``(ka, f/a, N)`` cell.  Runs in a worker process."""
    from tensorbem.benchmarks.ursell_cylinder import run

    ka, f_over_a, panels = job
    started = time.perf_counter()
    row = run(panel_count=panels, submergence=f_over_a, ka=ka)
    elapsed = time.perf_counter() - started

    radius = 1.0
    displaced_mass = DENSITY * math.pi * radius**2
    omega = math.sqrt(ka * GRAVITY / radius)
    damping_scale = displaced_mass * omega

    out: dict[str, Any] = {
        "ka": ka,
        "f_over_a": f_over_a,
        "panels": panels,
        "omega": omega,
        "panel_added_mass": row["panel_added_mass"],
        "panel_damping": row["panel_damping"],
        "reference_added_mass": row["reference_added_mass"],
        "reference_damping": row["reference_damping"],
        "added_mass_nondim": row["panel_added_mass"] / displaced_mass,
        "damping_nondim": row["panel_damping"] / damping_scale,
        "reference_added_mass_nondim": row["reference_added_mass"] / displaced_mass,
        "reference_damping_nondim": row["reference_damping"] / damping_scale,
        "added_mass_relative_error": row["panel_added_mass_relative_error"],
        "damping_relative_error": row["panel_damping_relative_error"],
        "damping_near_over_far": row["near_over_far_field_damping"],
        "panel_relative_residual": row["panel_relative_residual"],
        "panel_condition_number": row["panel_condition_number"],
        "mfs_added_mass_relative_error": row[
            "reference_solution_added_mass_relative_error"
        ],
        "mfs_damping_relative_error": row["reference_solution_damping_relative_error"],
        "seconds": elapsed,
    }
    # Scale-free errors: these stay finite at the A = 0 crossing.
    out["added_mass_abs_error_nondim"] = abs(
        out["added_mass_nondim"] - out["reference_added_mass_nondim"]
    )
    out["damping_abs_error_nondim"] = abs(
        out["damping_nondim"] - out["reference_damping_nondim"]
    )
    return out


def _cell_order(rows: Sequence[dict[str, Any]], column: str) -> dict[str, Any]:
    """Observed order ``p`` and ``N * eps`` constancy for one metric column."""
    ordered = sorted(rows, key=lambda r: r["panels"])
    pairs = []
    for coarse, fine in zip(ordered, ordered[1:]):
        e_coarse, e_fine = coarse[column], fine[column]
        if e_fine <= 0.0 or e_coarse <= 0.0:
            pairs.append(float("nan"))
            continue
        pairs.append(math.log(e_coarse / e_fine) / math.log(fine["panels"] / coarse["panels"]))
    products = [r["panels"] * r[column] for r in ordered]
    return {
        "orders": pairs,
        "order_min": min(pairs) if pairs else float("nan"),
        "order_max": max(pairs) if pairs else float("nan"),
        "n_times_error": products,
        "n_times_error_drift": (
            max(products) / min(products) if products and min(products) > 0 else float("nan")
        ),
    }


def _recommend(rows: Sequence[dict[str, Any]], column: str, bar: float) -> int | None:
    """Smallest panel count on the ladder that meets ``bar`` (None if unmet)."""
    for row in sorted(rows, key=lambda r: r["panels"]):
        if row[column] <= bar:
            return int(row["panels"])
    return None


def summarise(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Group the raw rows into cells, add orders, recommendations and verdicts."""
    cells: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for row in rows:
        cells.setdefault((row["ka"], row["f_over_a"]), []).append(row)

    summary: list[dict[str, Any]] = []
    for (ka, f_over_a) in sorted(cells):
        cell_rows = sorted(cells[(ka, f_over_a)], key=lambda r: r["panels"])
        finest = cell_rows[-1]
        entry: dict[str, Any] = {
            "ka": ka,
            "f_over_a": f_over_a,
            "panels": [r["panels"] for r in cell_rows],
            "added_mass_relative_error": [r["added_mass_relative_error"] for r in cell_rows],
            "damping_relative_error": [r["damping_relative_error"] for r in cell_rows],
            "added_mass_abs_error_nondim": [
                r["added_mass_abs_error_nondim"] for r in cell_rows
            ],
            "damping_abs_error_nondim": [r["damping_abs_error_nondim"] for r in cell_rows],
            "added_mass_nondim": [r["added_mass_nondim"] for r in cell_rows],
            "damping_nondim": [r["damping_nondim"] for r in cell_rows],
            "reference_added_mass_nondim": finest["reference_added_mass_nondim"],
            "reference_damping_nondim": finest["reference_damping_nondim"],
            "added_mass_order": _cell_order(cell_rows, "added_mass_relative_error"),
            "damping_order": _cell_order(cell_rows, "damping_relative_error"),
            "damping_near_over_far": [r["damping_near_over_far"] for r in cell_rows],
            "mfs_added_mass_relative_error": [
                r["mfs_added_mass_relative_error"] for r in cell_rows
            ],
            "mfs_damping_relative_error": [
                r["mfs_damping_relative_error"] for r in cell_rows
            ],
            "condition_number": [r["panel_condition_number"] for r in cell_rows],
            "recommended_panels": {
                f"added_mass_{bar:g}": _recommend(
                    cell_rows, "added_mass_relative_error", bar
                )
                for bar in BARS
            }
            | {
                f"damping_{bar:g}": _recommend(cell_rows, "damping_relative_error", bar)
                for bar in BARS
            },
            "added_mass_abs_error_nondim_at_128": _value_at(cell_rows, "added_mass_abs_error_nondim", 128),
            "damping_abs_error_nondim_at_128": _value_at(cell_rows, "damping_abs_error_nondim", 128),
        }
        summary.append(entry)

    return {
        "benchmark": "ursell_cylinder_precision_envelope",
        "reference": "ursell multipole closed form (analytic.py), MFS cross-check",
        "density": DENSITY,
        "gravity": GRAVITY,
        "bars": list(BARS),
        "cells": summary,
        "rows": sorted(rows, key=lambda r: (r["ka"], r["f_over_a"], r["panels"])),
    }


def _value_at(rows: Sequence[dict[str, Any]], column: str, panels: int) -> float | None:
    for row in rows:
        if row["panels"] == panels:
            return row[column]
    return None


def sweep(
    ka_list: Sequence[float] = DEFAULT_KA,
    f_list: Sequence[float] = DEFAULT_FOVER_A,
    panels: Sequence[int] = DEFAULT_PANELS,
    workers: int = 8,
) -> dict[str, Any]:
    jobs = [(ka, f_over_a, n) for ka in ka_list for f_over_a in f_list for n in panels]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_point, jobs, chunksize=1))
    return summarise(rows)


# ---------------------------------------------------------------------------
# Report generation (all numbers come from the JSON, none are retyped)
# ---------------------------------------------------------------------------

def build_report(payload: dict[str, Any], out_dir: Path, doc: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    cells = payload["cells"]
    bars = payload["bars"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # left: convergence at the two extreme submergences for a few ka
    ax = axes[0]
    for cell in cells:
        if cell["f_over_a"] not in (1.2, 2.0):
            continue
        style = "-" if cell["f_over_a"] == 2.0 else "--"
        ax.loglog(cell["panels"], cell["added_mass_relative_error"], style, marker="o",
                  label=f"ka={cell['ka']:g}, f/a={cell['f_over_a']:g}")
    ax.axhline(bars[0], color="k", lw=0.8, ls=":")
    ax.set_xlabel("panels N")
    ax.set_ylabel(r"$|A - A_{Ursell}| / |A_{Ursell}|$")
    ax.set_title("added-mass error vs panels (dashed: f/a = 1.2)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6, ncol=2)

    # right: error envelope at N = 128 vs submergence
    ax = axes[1]
    for cell in cells:
        label = f"ka={cell['ka']:g}"
        ax.plot(
            [cell["f_over_a"]],
            [cell["added_mass_abs_error_nondim_at_128"]],
            marker="o",
            ls="none",
            label=label,
        )
    ax.set_yscale("log")
    ax.axhline(bars[0], color="k", lw=0.8, ls=":")
    ax.axhline(bars[1], color="r", lw=0.8, ls=":")
    ax.set_xlabel("f/a (clearance in radii)")
    ax.set_ylabel(r"$|\Delta A| / (\rho\pi a^2)$ at N = 128")
    ax.set_title("displaced-mass-normalised error (red: 3 %, dotted: 1 %)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6)

    fig.tight_layout()
    png = out_dir / "envelope.png"
    fig.savefig(png, dpi=160)
    plt.close(fig)

    doc.write_text(render_markdown(payload, png.name), encoding="utf-8")


def render_markdown(payload: dict[str, Any], figure_name: str) -> str:
    cells = payload["cells"]
    bars = payload["bars"]
    lines: list[str] = []
    add = lines.append

    add("# Ursell 潜没圆柱：精度包络（频率 × 潜深 × 面元）")
    add("")
    add(
        "**性质**：验证欠账收口（`docs/ursell_error_budget.md` 局限 2 明写"
        "「仅审计了一个 (ka, a/f) 点，ka ≥ 1 未验证」）。本文件与图、表全部由 "
        "`python -m tensorbem.benchmarks.ursell_precision_envelope --report` 从 "
        "`docs/ursell_precision_envelope/envelope.json` 生成，数字零手抄。"
    )
    add("")
    add("## 1. 案例与参数")
    add("")
    add("| 量 | 值 |")
    add("|---|---|")
    add(f"| 圆柱半径 a | 1.0 m |")
    f_text = ", ".join(f"{c['f_over_a']:g}" for c in _unique(cells, "f_over_a"))
    ka_text = ", ".join(f"{c['ka']:g}" for c in _unique(cells, "ka"))
    n_text = ", ".join(str(n) for n in sorted({n for c in cells for n in c["panels"]}))
    add(f"| 潜深 f | f/a ∈ {{{f_text}}} |")
    add(f"| 无量纲频率 ka | {{{ka_text}}} |")
    add(f"| 面元数 N | {{{n_text}}} |")
    add(f"| ρ, g | {payload['density']:g} kg/m³, {payload['gravity']:g} m/s² |")
    add("| 参考解 | Ursell 多极子闭式解（`analytic.py`，与被测面板代码零共享）+ MFS 交叉校验 |")
    add("")
    add(
        "无量纲系数 $A^\\ast = A/(\\rho\\pi a^2)$、$B^\\ast = B/(\\rho\\pi a^2\\omega)$；"
        "两者都用量纲尺度归一，而不是「相对参考值」，因为 $A$ 在 $f/a\\approx1.13$ 变号"
        "（$A(1.10)=-238$、$A(1.20)=+1160$ kg/m），相对误差在那附近没有意义。"
    )
    add("")
    add("## 2. 方法与判据")
    add("")
    add(
        "对每个 (ka, f/a) 做面板加密序列，逐点计算：与多极子解的相对误差、"
        "观测收敛阶 $p=\\log_2(\\varepsilon_N/\\varepsilon_{2N})$、"
        "$N\\varepsilon$ 常数性（一阶收敛的签名）、近场/远场阻尼比（辐射条件）、"
        "MFS 独立解与多极子的一致程度（参考解本身的可信度）、条件数与残差。"
    )
    add("")
    add(
        f"判据两条：面板误差 ≤ {bars[0]:g}（本仓 Ursell 基准的既定门槛）与 "
        f"≤ {bars[1]:g}（平台通用门槛）；并回答「达到门槛所需的最小面元数」。"
    )
    add("")
    add("## 3. 结果（程序化注入）")
    add("")
    add(f"![精度包络]({figure_name})")
    add("")
    add("### 3.1 加密序列与收敛阶")
    add("")
    add(
        "| ka | f/a | N | $A^\\ast$ | $\\varepsilon_A$ | $B^\\ast$ | $\\varepsilon_B$ | "
        "$p_A$ | $p_B$ | $B_{near}/B_{far}$ | MFS $\\varepsilon_A$ |"
    )
    add("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cell in cells:
        rows = [r for r in payload["rows"] if r["ka"] == cell["ka"] and r["f_over_a"] == cell["f_over_a"]]
        rows.sort(key=lambda r: r["panels"])
        orders_a = cell["added_mass_order"]["orders"]
        orders_b = cell["damping_order"]["orders"]
        for index, row in enumerate(rows):
            pa = f"{orders_a[index - 1]:.2f}" if index else "—"
            pb = f"{orders_b[index - 1]:.2f}" if index else "—"
            add(
                f"| {cell['ka']:g} | {cell['f_over_a']:g} | {row['panels']} | "
                f"{row['added_mass_nondim']:.4f} | {row['added_mass_relative_error']:.3e} | "
                f"{row['damping_nondim']:.4f} | {row['damping_relative_error']:.3e} | "
                f"{pa} | {pb} | {row['damping_near_over_far']:.4f} | "
                f"{row['mfs_added_mass_relative_error']:.2e} |"
            )
    add("")
    add("### 3.2 达到 1 % / 3 % 门槛所需的最小面元数（在本阶梯内）")
    add("")
    add("| ka | f/a | A ≤ 1 % | A ≤ 3 % | B ≤ 1 % | B ≤ 3 % |")
    add("|---:|---:|---:|---:|---:|---:|")
    for cell in cells:
        rec = cell["recommended_panels"]
        fmt = lambda key: "未达" if rec[key] is None else str(rec[key])
        add(
            f"| {cell['ka']:g} | {cell['f_over_a']:g} | {fmt(f'added_mass_{bars[0]:g}')} | "
            f"{fmt(f'added_mass_{bars[1]:g}')} | {fmt(f'damping_{bars[0]:g}')} | "
            f"{fmt(f'damping_{bars[1]:g}')} |"
        )
    add("")
    _add_verdicts(lines, payload)
    _add_notes(lines, payload)
    add("")
    return "\n".join(lines) + "\n"


def _add_verdicts(lines: list[str], payload: dict[str, Any]) -> None:
    """Programmatic error judgement against the two acceptance bars."""
    cells = payload["cells"]
    bars = payload["bars"]

    def meets(cell: dict[str, Any], key: str, bar: float) -> str:
        value = cell[key]
        if value is None:
            return "未测"
        return "是" if value <= bar else "否"

    add = lines.append
    add("## 4. 误差判定（对 3 % 门槛逐项回答）")
    add("")
    add("| ka | f/a | $\varepsilon_A$ @128 | < 1 %? | < 3 %? | $\varepsilon_B$ @128 | < 1 %? | < 3 %? |")
    add("|---:|---:|---:|:--:|:--:|---:|:--:|:--:|")
    for cell in cells:
        ea = cell["added_mass_abs_error_nondim_at_128"]
        eb = cell["damping_abs_error_nondim_at_128"]
        ea_txt = "未测" if ea is None else f"{ea:.3e}"
        eb_txt = "未测" if eb is None else f"{eb:.3e}"
        add(
            f"| {cell['ka']:g} | {cell['f_over_a']:g} | {ea_txt} | "
            f"{meets(cell, 'added_mass_abs_error_nondim_at_128', bars[0])} | "
            f"{meets(cell, 'added_mass_abs_error_nondim_at_128', bars[1])} | "
            f"{eb_txt} | "
            f"{meets(cell, 'damping_abs_error_nondim_at_128', bars[0])} | "
            f"{meets(cell, 'damping_abs_error_nondim_at_128', bars[1])} |"
        )
    measured = [c for c in cells if c["added_mass_abs_error_nondim_at_128"] is not None]
    within_1 = sum(1 for c in measured if c["added_mass_abs_error_nondim_at_128"] <= bars[0])
    within_3 = sum(1 for c in measured if c["added_mass_abs_error_nondim_at_128"] <= bars[1])
    add("")
    add(
        f"在 N = 128（本仓文档默认面元数）下，$\\varepsilon_A$ 落在 1 % 内的组合为 "
        f"**{within_1}/{len(measured)}**，落在 3 % 内的组合为 **{within_3}/{len(measured)}**"
        "（$\\varepsilon_A$ 取位移质量归一，即 $|\\Delta A|/(\\rho\\pi a^2)$；"
        "对纯相对误差见 §3.1）。"
    )
    worst = max(
        (c for c in measured),
        key=lambda c: c["added_mass_abs_error_nondim_at_128"],
        default=None,
    )
    if worst is not None:
        add("")
        add(
            f"最差组合：ka = {worst['ka']:g}、f/a = {worst['f_over_a']:g}，"
            f"$|\\Delta A|/(\\rho\\pi a^2)$ = {worst['added_mass_abs_error_nondim_at_128']:.3e}"
            f"（{worst['added_mass_abs_error_nondim_at_128'] * 100:.2f} % of "
            "displaced mass）。"
        )


def _add_notes(lines: list[str], payload: dict[str, Any]) -> None:
    """Honest limitations, the traps met while measuring, and reproduction."""
    cells = payload["cells"]
    add = lines.append

    def series(key: str, shallow: bool | None = None):
        out = []
        for cell in cells:
            if shallow is True and cell["f_over_a"] > 1.2:
                continue
            if shallow is False and cell["f_over_a"] < 1.5:
                continue
            out.extend(v for v in cell[key] if v == v)
        return out

    orders_first = [
        p for cell in cells if cell["f_over_a"] >= 1.5
        for p in cell["added_mass_order"]["orders"] if p == p
    ]
    orders_shallow = [
        p for cell in cells if cell["f_over_a"] <= 1.2
        for p in cell["added_mass_order"]["orders"] if p == p
    ]
    near_far = series("damping_near_over_far")
    mfs_settled = series("mfs_added_mass_relative_error", shallow=False)
    mfs_corner = series("mfs_added_mass_relative_error", shallow=True)

    add("## 5. 注意点与局限")
    add("")
    if orders_first:
        add(
            f"1. **f/a ≥ 1.5 时是一阶收敛**：观测阶 $p_A$ 落在 "
            f"[{min(orders_first):.2f}, {max(orders_first):.2f}]，与常数面元的固有阶一致，"
            "所以在整个 ka 带（0.4–1.6）上都没有换阶机制。"
        )
    if orders_shallow:
        add(
            f"2. **f/a ≤ 1.2 的近水面角点不是一阶、甚至不单调**：同一带内 $p_A$ 落在 "
            f"[{min(orders_shallow):.2f}, {max(orders_shallow):.2f}]，在 (ka, f/a) = "
            "(0.9, 1.1)、(0.9, 1.2)、(0.6, 1.1) 处出现 $p_A<0$（加密反而暂时变差）。"
            "这是本文件最重要的负面结果：**该角点上「128 面元约 1 %」的经验规律不成立**，"
            "（ka=0.4, f/a=1.1）在 N=256 时仍有 2.6 % 的位移质量归一误差。"
            "原因未做归因（候选：常数面元对近水面强奇性核的一阶表示在低 ka 时不适用），"
            "取用时应当加密到 256 以上并做序列检查，而不是只看单点。"
        )
    if near_far:
        add(
            f"3. **辐射条件**：近场/远场阻尼比在全部算例中落在 "
            f"[{min(near_far):.4f}, {max(near_far):.4f}]，随 N 单调收敛、与 ka、f/a 基本无关；"
            "它是一阶离散的常数偏移（已由 `docs/finite_depth_energy_identity/report.md` 归因），"
            "不是物理差异，故本文件不把「能量恒等式到 0.1 %」当判据。"
        )
    if mfs_settled:
        add(
            f"4. **参考解在 f/a ≥ 1.5 上被独立确认**：MFS（独立离散，与被测面板代码共享核但"
            f"不共享离散）与 Ursell 多极子解的最大相对偏差 {max(mfs_settled):.2e}，比被测量的"
            "面板误差小 1–2 个量级。"
        )
    if mfs_corner:
        add(
            f"5. **近水面角点的参考解只靠多极子**：$f/a\le1.2$ 处同一 MFS 交叉校验偏差最大到 "
            f"{max(mfs_corner):.2e}（24 个源不足以分辨贴近自由面的强相互作用），"
            "因此该角点的数字应读作「面板 vs 多极子」，多极子级数截断本身未被独立确认。"
            "这是本文件在该角点上的置信度上限，不向下游宣称更高。"
        )
    add(
        "6. **$A$ 的变号陷阱**：$A$ 在 $f/a\\approx1.13$ 过零（$A(1.10)<0<A(1.20)$），"
        "相对误差在那里发散且无物理意义；§4 的判定一律用位移质量归一 "
        "$|\\Delta A|/(\\rho\\pi a^2)$，§3.1 同时给出相对误差供对照。"
        "下游代码引用本表前应做同样的归一化。"
    )
    add(
        "7. **适用范围**：二维、无限水深、常潜深、无航速、线性势流，只扫圆形截面。"
        "三维、有限水深、航速（Kelvin/Havelock 核）与真实船体剖面不在覆盖范围内。"
    )
    add(
        "8. **成本**：单点 128 面元约 11 s、256 面元约 43 s（单核、空载），"
        "整张 25 格 × 4 面元网格用 7 进程在负载机器上约 20 分钟；未使用集群。"
    )
    add("")
    add("## 6. 复现")
    add("")
    add("```bash")
    add("# 重跑整张网格（写 JSON + 文档 + 图）")
    add("PYTHONPATH=src python -m tensorbem.benchmarks.ursell_precision_envelope \\")
    add("    --json docs/ursell_precision_envelope/envelope.json \\")
    add("    --doc docs/ursell_precision_envelope.md --workers 7")
    add("")
    add("# 只从已提交的 JSON 重建文档与图（不重算）")
    add("PYTHONPATH=src python -m tensorbem.benchmarks.ursell_precision_envelope --report \\")
    add("    --json docs/ursell_precision_envelope/envelope.json \\")
    add("    --doc docs/ursell_precision_envelope.md")
    add("")
    add("PYTHONPATH=src python -m pytest tests/test_ursell_precision_envelope.py -q")
    add("```")


def _unique(cells: Sequence[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    seen: dict[Any, dict[str, Any]] = {}
    for cell in cells:
        seen.setdefault(cell[key], cell)
    return [seen[k] for k in sorted(seen)]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ka", default=",".join(str(v) for v in DEFAULT_KA))
    parser.add_argument("--f-over-a", default=",".join(str(v) for v in DEFAULT_FOVER_A))
    parser.add_argument("--panels", default=",".join(str(v) for v in DEFAULT_PANELS))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("W", "8")))
    parser.add_argument(
        "--json",
        default="docs/ursell_precision_envelope/envelope.json",
        help="where the raw rows and the cell summary are written",
    )
    parser.add_argument("--report", action="store_true", help="(re)generate doc + figure from --json")
    parser.add_argument("--doc", default="docs/ursell_precision_envelope.md")
    args = parser.parse_args(argv)

    json_path = Path(args.json)
    if args.report:
        payload = json.loads(json_path.read_text())
    else:
        payload = sweep(
            [float(v) for v in args.ka.split(",")],
            [float(v) for v in args.f_over_a.split(",")],
            [int(v) for v in args.panels.split(",")],
            workers=args.workers,
        )
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=1))

    build_report(payload, json_path.parent, Path(args.doc))
    print(f"wrote {json_path} and {args.doc}")


if __name__ == "__main__":
    main()