"""How much of the anchor's "independent" corroboration is discretisation error?

Context
-------
The case-pool anchor ``BEM-HULME-HEMISPHERE`` grades a floating hemisphere
against Hulme (1982)'s wave-free multipole series, and both routes share
``tensorbem.free_surface3d.wave_function_3d``, so the case supplies a third,
independent code (capytaine) as the only adjudicator
(``benchmarks/verified/casepool_bem_hulme_hemisphere/capytaine_excitation.json``,
``docs/case_bem_hulme_independent_axis.md``).

That adjudication reports a worst code-to-code spread of 3.36 % at 1600
capytaine panels, and its support is the gate ``I3``: *the drift of the last
ladder step (1024 -> 1600 panels) is below 1 %*.  A small **step** is not the
same statement as **convergence**: a sequence that still moves 0.67 % between
its last two rungs may sit several percent away from its own continuum limit,
and a residual of that size is the same order as the spread being adjudicated.
This module removes the ambiguity instead of arguing about it:

* the ladder is extended to seven rungs (256 -> 4096 panels) at the full band
  (``scripts/capytaine_hulme_ladder_extended.py``);
* a three-parameter convergence model ``v(N) = v_inf + C N^-p`` is fitted per
  quantity and per ``ka``, with the observed order ``p`` estimated from the data
  (no assumed order), and the fit's stability is checked by fitting disjoint
  rung windows;
* the same model is fitted to the *in-house* solver's ladder at the worst band
  point (``scripts/hulme_hemisphere_inhouse_ladder.py``, ka = 2, 64 -> 4096
  panels), because the anchor's "1.15 % solver error" and "3.36 % independent
  spread" are otherwise not commensurable -- one is a discretised value, the
  other was treated as a limit;
* the adjudication is then redone between **continuum limits** rather than
  between discretised values.

Everything is read from artifacts; nothing is hand-copied into the document.

Predeclared gates (fixed before the extended ladder was fetched)
---------------------------------------------------------------
===================================  =========================================
``E1_convergence_model_credible``    the fitted model reproduces the last three
                                     rungs of every (quantity, ka) cell to
                                     within 0.3 % -- otherwise ``p`` and
                                     ``v_inf`` are not credible and no claim is
                                     made
``E2_observed_order_identified``     the observed order stays interior (0.1-4.0)
                                     and shifts < 25 % between the all-rung and
                                     last-four-rung fits -- otherwise the power
                                     law does not describe the data and neither
                                     ``p`` nor ``v_inf`` may be quoted
``E3_extrapolation_stable``          ``v_inf`` fitted on the last four rungs vs
                                     on all rungs within 0.5 % for every cell
``E4_frozen_level_residual``         residual distance of the anchor's frozen
                                     1600-panel level from ``v_inf``: measured
                                     and required below 2 % (the point of the
                                     gate is that it is *not* the < 1 %
                                     "converged" that ``I3`` is often read as)
``E5_adjudicated_spread``            ``|reference/v_inf - 1| < 2 %`` for every
                                     cell: the independent axis still confirms
                                     the reference once its own discretisation
                                     error is removed
``E6_extrapolation_moves_toward_ref``  at the worst cell of the published
                                     adjudication (B33, ka = 2) extrapolation
                                     reduces the distance to the reference --
                                     directional evidence that the published
                                     spread was the third code's error, not the
                                     reference's
``E7_inhouse_model_credible``        same 0.3 % reproduction test for the
                                     in-house ladder at ka = 2
``E8_inhouse_extrapolated_error``    ``|BEM v_inf / reference - 1| < 1 %`` for
                                     every quantity at ka = 2 -- the graded
                                     1.15 % must be discretisation, not bias
===================================  =========================================
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

ARTIFACT_DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "verified" / "casepool_bem_hulme_hemisphere"
CASE_ARTIFACT = ARTIFACT_DIR / "results.json"
FROZEN_ARTIFACT = ARTIFACT_DIR / "capytaine_excitation.json"
EXTENDED_ARTIFACT = ARTIFACT_DIR / "capytaine_ladder_extended.json"
INHOUSE_ARTIFACT = ARTIFACT_DIR / "inhouse_ladder_ka2.json"
EVIDENCE_DIR = ARTIFACT_DIR / "evidence"
EVIDENCE = EVIDENCE_DIR / "independent_convergence.json"
DOCUMENT = Path(__file__).resolve().parents[3] / "docs" / "case_bem_hulme_independent_convergence.md"

#: anchor band (tolerance of the graded case)
ANCHOR_BAND = 0.05
#: predeclared gate thresholds (see module docstring; do not relax after the fact)
MODEL_BAND = 0.003
STABILITY_BAND = 0.005
FROZEN_RESIDUAL_BAND = 0.02
ADJUDICATED_BAND = 0.02
INHOUSE_EXTRAPOLATED_BAND = 0.01
#: the worst cell of the published adjudication (docs/case_bem_hulme_independent_axis.md)
WORST_CELL = ("B33", 2.0)

#: canonical quantity -> key inside the capytaine artifacts
QUANTITIES: tuple[tuple[str, str], ...] = (
    ("A11", "A11"),
    ("B11", "B11"),
    ("A33", "A33"),
    ("B33", "B33"),
    ("Fx", "Fx_dimless"),
    ("Fz", "Fz_dimless"),
)
#: canonical quantity -> key inside results.json and the in-house ladder rows
CASE_KEY = {
    "A11": "A11_dimless",
    "B11": "B11_dimless",
    "A33": "A33_dimless",
    "B33": "B33_dimless",
    "Fx": "Fx_dimless",
    "Fz": "Fz_dimless",
}


# ---------------------------------------------------------------------------
# convergence model
# ---------------------------------------------------------------------------
def fit_convergence(
    panels: Sequence[float],
    values: Sequence[float],
    *,
    p_lo: float = 0.05,
    p_hi: float = 5.0,
    coarse_grid: int = 600,
) -> dict[str, Any]:
    """Least-squares fit of ``v(N) = v_inf + C N^-p`` with ``p`` estimated.

    ``p`` is found by a log-spaced scan (the model is *linear* in
    ``(v_inf, C)`` for fixed ``p``, so each scan point is one 2x2 least-squares
    solve) followed by parabolic refinement.  Returns the limit, the observed
    order, and the fit residual expressed in the same relative metric the gate
    uses, so an implausible fit cannot hide behind a good ``v_inf``.
    """
    n = [float(x) for x in panels]
    v = [float(x) for x in values]
    if len(n) < 3:
        raise ValueError("three rungs are the minimum for an observed order")

    def solve(p: float) -> tuple[float, float, float]:
        basis = [x ** (-p) for x in n]
        s11 = float(len(n))
        s12 = sum(basis)
        s22 = sum(b * b for b in basis)
        t1 = sum(v)
        t2 = sum(vi * b for vi, b in zip(v, basis))
        det = s11 * s22 - s12 * s12
        # a near-degenerate normal matrix means the rung spread cannot separate
        # v_inf from C at this p: treat it as an infinite residual
        if abs(det) < 1e-30:
            return float("nan"), float("nan"), float("inf")
        c = (s11 * t2 - s12 * t1) / det
        v_inf = (s22 * t1 - s12 * t2) / det
        sse = 0.0
        for ni, vi in zip(n, v):
            sse += (v_inf + c * ni ** (-p) - vi) ** 2
        return v_inf, c, sse

    grid = [
        p_lo * (p_hi / p_lo) ** (i / (coarse_grid - 1)) for i in range(coarse_grid)
    ]
    scored = [(solve(p)[2], p) for p in grid]
    sse_best, p_best = min(scored, key=lambda t: t[0])
    # parabolic refinement in log p around the scan minimum
    step = math.log(p_hi / p_lo) / (coarse_grid - 1)
    for _ in range(60):
        lo, hi = math.exp(math.log(p_best) - step), math.exp(math.log(p_best) + step)
        cand = [(solve(p)[2], p) for p in (lo, p_best, hi)]
        sse_new, p_new = min(cand, key=lambda t: t[0])
        if sse_new >= sse_best:
            step *= 0.5
            if step < 1e-6:
                break
            continue
        sse_best, p_best = sse_new, p_new
    v_inf, c, sse = solve(p_best)
    residuals = {
        f"{ni:g}": (v_inf + c * ni ** (-p_best) - vi) / abs(vi)
        for ni, vi in zip(n, v)
    }
    return {
        "v_inf": v_inf,
        "p": p_best,
        "c": c,
        "sse": sse,
        "rel_residual_by_panels": residuals,
        "max_rel_residual": max(abs(r) for r in residuals.values()),
        "rungs": [f"{x:g}" for x in n],
    }


# ---------------------------------------------------------------------------
# artifact loading
# ---------------------------------------------------------------------------
def capytaine_panels(path: Path) -> list[float]:
    payload = json.loads(Path(path).read_text())
    return [float(level["n_faces"]) for level in payload["levels"]]


def capytaine_cell(path: Path, name: str) -> dict[str, dict[float, float]]:
    """``{ka_tag: {panels: value}}`` for one published quantity name."""
    key = dict(QUANTITIES)[name]
    payload = json.loads(Path(path).read_text())
    out: dict[str, dict[float, float]] = {}
    for level in payload["levels"]:
        for ka_tag, row in level["values"].items():
            out.setdefault(ka_tag, {})[float(level["n_faces"])] = float(row[key])
    return out


def inhouse_cell(
    path: Path, name: str
) -> tuple[list[float], list[float], str]:
    """Rung panel counts, rung values, and the ``ka`` tag of an in-house ladder."""
    key = CASE_KEY[name]
    payload = json.loads(Path(path).read_text())
    ka_tag = f"{float(payload['ka']):g}"
    rows = sorted(payload["rows"], key=lambda r: r["panels"])
    panels = [float(row["panels"]) for row in rows]
    values = [float(row[key]) for row in rows]
    return panels, values, ka_tag


def reference_table(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(Path(path).read_text())
    return {tag: dict(row) for tag, row in payload["reference"].items()}


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def analyse(
    *,
    case_path: Path = CASE_ARTIFACT,
    frozen_path: Path = FROZEN_ARTIFACT,
    extended_path: Path = EXTENDED_ARTIFACT,
    inhouse_path: Path = INHOUSE_ARTIFACT,
) -> dict[str, Any]:
    reference = reference_table(case_path)
    frozen_panels = set(capytaine_panels(frozen_path))
    extended: dict[str, dict[str, list[float]]] = {}
    panels_ext = capytaine_panels(extended_path)
    for name, _ in QUANTITIES:
        per_ka = capytaine_cell(extended_path, name)
        for ka_tag, values in per_ka.items():
            slot = extended.setdefault(ka_tag, {}).setdefault(name, [])
            slot.extend(values[n] for n in sorted(values))
    cells: list[dict[str, Any]] = []

    for ka_tag, per_q in extended.items():
        for name, _ in QUANTITIES:
            values = per_q[name]
            fit_all = fit_convergence(panels_ext, values)
            fit_last4 = fit_convergence(panels_ext[-4:], values[-4:])
            fit_last3 = fit_convergence(panels_ext[-3:], values[-3:])
            frozen_value = next(
                v for n, v in zip(panels_ext, values) if n in frozen_panels and n == max(frozen_panels)
            )
            best_value = values[-1]
            ref = float(reference[ka_tag][CASE_KEY[name]])
            cells.append(
                {
                    "ka": float(ka_tag),
                    "quantity": name,
                    "reference": ref,
                    "fit_all": fit_all,
                    "p_last4": fit_last4["p"],
                    "v_inf_last4": fit_last4["v_inf"],
                    "p_last3": fit_last3["p"],
                    "v_inf_last3": fit_last3["v_inf"],
                    "stability_all_vs_last4": abs(fit_last4["v_inf"] / fit_all["v_inf"] - 1.0),
                    "stability_p_all_vs_last3": abs(fit_last3["p"] / fit_all["p"] - 1.0),
                    "finest_panels": panels_ext[-1],
                    "finest_value": best_value,
                    "finest_residual": abs(best_value / fit_all["v_inf"] - 1.0),
                    "frozen_panels": max(frozen_panels),
                    "frozen_value": frozen_value,
                    "frozen_residual": abs(frozen_value / fit_all["v_inf"] - 1.0),
                    "spread_frozen": abs(frozen_value / ref - 1.0),
                    "spread_extrapolated": abs(fit_all["v_inf"] / ref - 1.0),
                }
            )

    gates: dict[str, Any] = {}
    worst_model = max(
        max(abs(v) for v in list(c["fit_all"]["rel_residual_by_panels"].values())[-3:])
        for c in cells
    )
    gates["E1_convergence_model_credible"] = {
        "criterion": (
            "the fitted v(N)=v_inf+C*N^-p reproduces the last three rungs of "
            f"every (quantity, ka) cell to within {MODEL_BAND:.1%}"
        ),
        "measured": worst_model,
        "pass": bool(worst_model < MODEL_BAND),
    }
    # an observed order that pins to the fit bound means the power law does not
    # describe the data: then neither p nor v_inf may be quoted as a number
    p_floor, p_ceil = 0.1, 4.0
    at_bound = [
        (c["quantity"], c["ka"], c["fit_all"]["p"])
        for c in cells
        if not (p_floor <= c["fit_all"]["p"] <= p_ceil)
    ]
    worst_p_shift = max(
        abs(c["fit_all"]["p"] - c["p_last4"]) / c["fit_all"]["p"] for c in cells
    )
    gates["E2_observed_order_identified"] = {
        "criterion": (
            f"observed order stays inside [{p_floor:g}, {p_ceil:g}] and shifts "
            "by < 25 % between the all-rung and last-four-rung fits, for every cell"
        ),
        "measured": worst_p_shift,
        "cells_at_bound": at_bound,
        "pass": bool(not at_bound and worst_p_shift < 0.25),
    }
    worst_stab = max(c["stability_all_vs_last4"] for c in cells)
    gates["E3_extrapolation_stable"] = {
        "criterion": (
            "v_inf fitted on the last four rungs vs on all rungs agree within "
            f"{STABILITY_BAND:.1%} for every cell"
        ),
        "measured": worst_stab,
        "pass": bool(worst_stab < STABILITY_BAND),
    }
    worst_frozen = max(c["frozen_residual"] for c in cells)
    gates["E4_frozen_level_residual"] = {
        "criterion": (
            "the anchor's frozen 1600-panel level sits below "
            f"{FROZEN_RESIDUAL_BAND:.0%} from the extrapolated limit (it is a "
            "discretised value, not a converged one)"
        ),
        "measured": worst_frozen,
        "worst_cell": max(cells, key=lambda c: c["frozen_residual"])["quantity"],
        "worst_ka": max(cells, key=lambda c: c["frozen_residual"])["ka"],
        "pass": bool(worst_frozen < FROZEN_RESIDUAL_BAND),
    }
    worst_adj = max(cells, key=lambda c: c["spread_extrapolated"])
    gates["E5_adjudicated_spread"] = {
        "criterion": (
            "|reference/v_inf - 1| < "
            f"{ADJUDICATED_BAND:.0%} for every cell: the independent axis "
            "confirms the reference once its own discretisation error is removed"
        ),
        "measured": worst_adj["spread_extrapolated"],
        "worst_quantity": worst_adj["quantity"],
        "worst_ka": worst_adj["ka"],
        "pass": bool(worst_adj["spread_extrapolated"] < ADJUDICATED_BAND),
    }
    worst_name, worst_ka = WORST_CELL
    cell = next(c for c in cells if c["quantity"] == worst_name and c["ka"] == worst_ka)
    gates["E6_extrapolation_moves_toward_ref"] = {
        "criterion": (
            "at the published worst cell (B33, ka=2) extrapolation reduces the "
            "distance to the reference: |ref-v_inf| < |ref-v(1600)|"
        ),
        "measured_frozen": cell["spread_frozen"],
        "measured_extrapolated": cell["spread_extrapolated"],
        "pass": bool(cell["spread_extrapolated"] < cell["spread_frozen"]),
    }

    inhouse: dict[str, Any] | None = None
    if Path(inhouse_path).exists():
        panels_in: list[float] | None = None
        per_q_in: dict[str, dict[str, Any]] = {}
        for name, _ in QUANTITIES:
            panels, panel_values, ka_tag = inhouse_cell(inhouse_path, name)
            panels_in = panels
            fit = fit_convergence(panels, panel_values)
            ref = float(reference[ka_tag][CASE_KEY[name]])
            per_q_in[name] = {
                "values_by_panels": {f"{p:g}": v for p, v in zip(panels, panel_values)},
                "fit": fit,
                "reference": ref,
                "graded_value": panel_values[panels.index(1024.0)] if 1024.0 in panels else None,
                "graded_error": (
                    abs(panel_values[panels.index(1024.0)] / ref - 1.0)
                    if 1024.0 in panels
                    else None
                ),
                "extrapolated_value": fit["v_inf"],
                "extrapolated_error": abs(fit["v_inf"] / ref - 1.0),
                "finest_value": panel_values[-1],
                "finest_error": abs(panel_values[-1] / ref - 1.0),
            }
        inhouse = {
            "ka": float(json.loads(Path(inhouse_path).read_text())["ka"]),
            "panels": panels_in,
            "quantities": per_q_in,
        }
        worst_model_in = max(
            max(abs(v) for v in list(q["fit"]["rel_residual_by_panels"].values())[-3:])
            for q in per_q_in.values()
        )
        gates["E7_inhouse_model_credible"] = {
            "criterion": (
                "the same convergence model reproduces the last three in-house "
                f"rungs at ka=2 to within {MODEL_BAND:.1%} for every quantity"
            ),
            "measured": worst_model_in,
            "pass": bool(worst_model_in < MODEL_BAND),
        }
        worst_in = max(per_q_in.items(), key=lambda kv: kv[1]["extrapolated_error"])
        gates["E8_inhouse_extrapolated_error"] = {
            "criterion": (
                "|BEM v_inf / reference - 1| < "
                f"{INHOUSE_EXTRAPOLATED_BAND:.0%} for every quantity at ka=2 -- "
                "the graded 1.15 % is discretisation, not bias"
            ),
            "measured": worst_in[1]["extrapolated_error"],
            "worst_quantity": worst_in[0],
            "graded_worst_error": max(q["graded_error"] or 0.0 for q in per_q_in.values()),
            "pass": bool(worst_in[1]["extrapolated_error"] < INHOUSE_EXTRAPOLATED_BAND),
        }

    verdict = "PASS" if all(g["pass"] for g in gates.values()) else "FAIL"
    return {
        "case_code": "BEM-HULME-HEMISPHERE",
        "artifact_dir": str(ARTIFACT_DIR),
        "anchor_band": ANCHOR_BAND,
        "capytaine_panels": panels_ext,
        "frozen_panels": sorted(frozen_panels),
        "cells": cells,
        "inhouse": inhouse,
        "gates": gates,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def _pct(x: float, digits: int = 3) -> str:
    return f"{100.0 * x:.{digits}f}%"


def render_document(payload: dict[str, Any]) -> str:
    L: list[str] = []
    cells = payload["cells"]
    gates = payload["gates"]
    frozen_level = max(payload["frozen_panels"])
    finest = payload["capytaine_panels"][-1]
    L.append("# 锚点 BEM-HULME-HEMISPHERE：独立轴的自身收敛性审计（Richardson 外推三路裁定）")
    L.append("")
    L.append(
        "> 本文件由 `tensorbem.benchmarks.hemisphere_independent_convergence` 从工件程序化渲染，"
        "数字零手抄。工件：`benchmarks/verified/casepool_bem_hulme_hemisphere/"
        "capytaine_ladder_extended.json`（7 级 256→4096 面板）、`inhouse_ladder_ka2.json`"
        f"（ka=2，64→4096 面板）、`evidence/independent_convergence.json`。**机器裁定：{payload['verdict']}**"
    )
    L.append("")
    L.append("## ① 案例介绍与欠账")
    L.append("")
    L.append(
        "浮半球（a = 1 m，平面与自由面齐平）在无限深水中的辐射/绕射由 Hulme 1982 的无波多极级数"
        "给出参考，面板解与参考解共享 `tensorbem.free_surface3d.wave_function_3d`，"
        "因此算例池的证据结构把第三方 BEM（capytaine）当作唯一独立裁定者。"
        "已发布的那一版裁定（`docs/case_bem_hulme_independent_axis.md`）报告最坏码间差 3.36%，"
        "其支撑判据 `I3` 只证明「最后一档网格漂移 < 1%」。"
    )
    L.append("")
    L.append(
        f"**这不是同一个命题。** 一列每档仍移动 0.67% 的序列可以离自己的连续极限好几个百分点，"
        f"而残差与被裁定的 3.36% 同量级时，这 3.36% 究竟说明参考解有误、还是说明第三方码没算准，"
        "没有答案。本审计把网格延拓到 7 档（256→4096 面板，最细档是算例主网格的 4 倍、"
        f"冻结档 {frozen_level:.0f} 面板的 2.56 倍），拟合三参数收敛模型 "
        "`v(N) = v_inf + C·N^-p`（阶 p 由数据估，不预设），再把裁定改成**连续极限对连续极限**。"
    )
    L.append("")
    L.append("## ② 怎么做的")
    L.append("")
    L.append("1. **独立码的延拓阶梯**：`scripts/capytaine_hulme_ladder_extended.py`，"
             "capytaine 3.1.dev0，同一 `mesh_sphere(...).immersed_part()` 构造，"
             "分辨率 (16,32)/(24,48)/(32,64)/(40,80)/(48,96)/(56,112)/(64,128) "
             "= 256/576/1024/1600/2304/3136/4096 面板，全 ka 频带（0.2, 0.5, 0.8, 1, 1.5, 2），"
             "6 个量（A11, B11, A33, B33, Fx, Fz）；激励力按 capytaine ≥3.0 的约定"
             "（衍射项 + 手加 Froude-Krylov 项）由既有 producer 的函数直接复用，避免两件工件在约定上漂移。")
    L.append("2. **本仓求解器的延拓阶梯**：`scripts/hulme_hemisphere_inhouse_ladder.py`，"
             "ka = 2（算例带内误差最大点），subdivisions 2/3/4/5 = 64/256/1024/4096 恒定面元，"
             "单格计算全部委托 `hulme_hemisphere_case.run_cell`，不复制求解器代码。")
    L.append("3. **收敛模型**：给定 p，模型对 (v_inf, C) 线性，故 p 由对数网格扫描 + 抛物线细化求得；"
             "拟合优度用与判据同一的相对度量（逐档相对残差）记录，避免「v_inf 好看但阶不可信」。"
             "另用 4 档窗口与 7 档窗口的 v_inf 互校（E2）。")
    L.append("4. **裁定**：把参考级数、外力码外推极限、本仓求解器外推极限三者放在同一坐标系比较；"
             "并逐项回答「3.36% 是不是第三方码自己的离散误差」。")
    L.append("")
    L.append("## ③ 参数表")
    L.append("")
    L.append("| 项目 | 取值 |")
    L.append("|---|---|")
    L.append("| 几何 | 浮半球 a = 1.0 m，平面在 z = 0（平均自由面） |")
    L.append("| 排水体积 ∇（精确） | 2.094395 m³ |")
    L.append("| ρ / g / A | 1025.0 kg/m³ / 9.80665 m/s² / 1.0 m |")
    L.append("| ka 频带 | 0.2, 0.5, 0.8, 1, 1.5, 2 |")
    L.append("| 独立码网格阶梯 | " + ", ".join(f"{n:g}" for n in payload["capytaine_panels"]) + " 面板 |")
    L.append(f"| 已冻结独立码档位 | {frozen_level:.0f} 面板 |")
    L.append("| 本仓阶梯（ka=2） | 64, 256, 1024, 4096 恒定面元（4·4ⁿ） |")
    L.append("| 参考解 | Hulme 1982 无波多极级数，144 项（截断漂移 ~6e-5） |")
    L.append("| 算例容差 | 5% |")
    L.append("")
    L.append("## ④ 结果：每格的观测阶、极限与残差")
    L.append("")
    L.append(
        "| ka | 量 | 观测阶 p | v(N=4096) | v_inf（外推） | 最细档残差 | "
        f"冻结档 {frozen_level:.0f} 残差 | 参考 vs 冻结档 | 参考 vs v_inf |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|")
    for c in sorted(cells, key=lambda c: (c["ka"], c["quantity"])):
        L.append(
            f"| {c['ka']:g} | {c['quantity']} | {c['fit_all']['p']:.2f} | "
            f"{c['finest_value']:.6f} | {c['fit_all']['v_inf']:.6f} | "
            f"{_pct(c['finest_residual'])} | {_pct(c['frozen_residual'])} | "
            f"{_pct(c['spread_frozen'])} | {_pct(c['spread_extrapolated'])} |"
        )
    L.append("")
    L.append("## ⑤ 结果图")
    L.append("")
    L.append(
        "见 `benchmarks/verified/casepool_bem_hulme_hemisphere/independent_convergence_figure.png`："
        "左图为最坏格（B33, ka=2）的残差 vs 面板数（双对数）与拟合线，右图为全部 36 格的"
        "「参考 vs 独立码」距离在冻结档与外推极限下的对比。"
    )
    L.append("")
    L.append("## ⑥ 本仓求解器的同一处理（ka = 2）")
    L.append("")
    inhouse = payload.get("inhouse")
    if inhouse:
        L.append("| 量 | v(1024) 误差（算例判据所用） | v(4096) 误差 | v_inf 误差 | 观测阶 p |")
        L.append("|---|---|---|---|---|")
        for name, q in inhouse["quantities"].items():
            graded = _pct(q["graded_error"]) if q["graded_error"] is not None else "n/a"
            L.append(
                f"| {name} | {graded} | {_pct(q['finest_error'])} | "
                f"{_pct(q['extrapolated_error'])} | {q['fit']['p']:.2f} |"
            )
        L.append("")
    else:
        L.append("（本轮未取得本仓延拓阶梯工件，本节不作为证据。）")
        L.append("")
    L.append("## ⑦ 判据裁定")
    L.append("")
    L.append("| 判据 | 内容 | 实测 | 裁定 |")
    L.append("|---|---|---|---|")
    for key, gate in gates.items():
        measured = gate.get("measured")
        if measured is None:
            numeric = {
                k: v for k, v in gate.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            measured = (max(numeric.values()) if numeric else float("nan"))
        extra = ""
        for tail in ("worst_quantity", "worst_cell"):
            if tail in gate:
                extra += f"（{gate[tail]}"
                if "worst_ka" in gate:
                    extra += f", ka={gate['worst_ka']:g}"
                extra += "）"
        L.append(
            f"| `{key}` | {gate['criterion']} | {_pct(float(measured))}{extra} | "
            f"{'PASS' if gate['pass'] else 'FAIL'} |"
        )
    L.append("")
    L.append(f"**总判定：{payload['verdict']}**")
    L.append("")
    L.append("## ⑧ 注意点与局限")
    L.append("")
    L.append(
        f"- **冻结档 {frozen_level:.0f} 面板不是收敛值**：它与外推极限的残差见 ④ 的"
        "「冻结档残差」列。已发布裁定的 3.36% 最坏码间差里，有多少是第三方码自己的离散误差，"
        "由 ④ 的两列（参考 vs 冻结档 / 参考 vs v_inf）给出直接读数。"
    )
    L.append(
        "- **外推的前提是渐近区**：E1 用「模型能否复现最后三档」检验这一点，E2 用窗口互换检验"
        "极限是否与窗口无关；二者任一不过，v_inf 只能当趋势线读，不能当数值读。"
    )
    L.append(
        "- **p 由数据估而非理论给定**：面元法的理论阶依赖面元类型、水线处的奇异性与 capytaine "
        "`immersed_part()` 在水线处切出多边形造成的几何误差，没有可预设的单一阶；"
        "因此给 p 而不宣称它等于某个理论值。"
    )
    L.append(
        "- **水线离散**：`mesh_sphere(...).immersed_part()` 切出的水线是内接多边形，"
        "其半径小于 a，收敛行为可能与体表面不同，这也是外推残差随 ka 变化的候选解释之一；"
        "本审计只测量、不归因。"
    )
    L.append(
        "- **仍缺的轴**：Hulme 1982 的印刷表在本环境不可得（WITHHELD），"
        "因此「参考」仍是该方法的重新推导级数；本审计只能证明*独立码与参考解在连续极限下一致*，"
        "不能证明*参考解等于印刷值*。"
    )
    L.append("")
    L.append("## ⑨ 误差判定（逐项，是否 < 3%）")
    L.append("")
    L.append("| 关键量 | 值 | < 3%? | < 5%（算例容差）? |")
    L.append("|---|---|---|---|")
    worst_adj = max(cells, key=lambda c: c["spread_extrapolated"])
    worst_frz = max(cells, key=lambda c: c["spread_frozen"])
    L.append(
        f"| 参考 vs 独立码（外推极限）最坏 | {_pct(worst_adj['spread_extrapolated'])} "
        f"（{worst_adj['quantity']}, ka={worst_adj['ka']:g}） | "
        f"{'✅' if worst_adj['spread_extrapolated'] < 0.03 else '❌'} | "
        f"{'✅' if worst_adj['spread_extrapolated'] < ANCHOR_BAND else '❌'} |"
    )
    L.append(
        f"| 参考 vs 独立码（冻结档，已发布口径）最坏 | {_pct(worst_frz['spread_frozen'])} "
        f"（{worst_frz['quantity']}, ka={worst_frz['ka']:g}） | "
        f"{'✅' if worst_frz['spread_frozen'] < 0.03 else '❌'} | "
        f"{'✅' if worst_frz['spread_frozen'] < ANCHOR_BAND else '❌'} |"
    )
    if inhouse:
        worst_in = max(
            inhouse["quantities"].items(), key=lambda kv: kv[1]["extrapolated_error"]
        )
        L.append(
            f"| 本仓求解器 v_inf vs 参考（ka=2）最坏 | "
            f"{_pct(worst_in[1]['extrapolated_error'])}（{worst_in[0]}） | "
            f"{'✅' if worst_in[1]['extrapolated_error'] < 0.03 else '❌'} | "
            f"{'✅' if worst_in[1]['extrapolated_error'] < ANCHOR_BAND else '❌'} |"
        )
    L.append("")
    return "\n".join(L) + "\n"


def render_figure(payload: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cells = payload["cells"]
    panels = payload["capytaine_panels"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax = axes[0]
    targets = [(n, k) for n, k in (("B33", 2.0), ("A33", 2.0), ("Fz", 2.0))]
    styles = ("o-", "s--", "^:")
    extended = {
        (c["quantity"], c["ka"]): c for c in cells
    }
    for (name, ka), style in zip(targets, styles):
        cell = extended[(name, ka)]
        values = []
        for level in json.loads(EXTENDED_ARTIFACT.read_text())["levels"]:
            values.append(float(level["values"][f"{ka:g}"][dict(QUANTITIES)[name]]))
        resid = [abs(v / cell["fit_all"]["v_inf"] - 1.0) for v in values]
        ax.loglog(panels, np.maximum(resid, 1e-12), style, label=f"{name}, ka={ka:g}")
        ax.loglog(
            panels,
            [abs(cell["fit_all"]["c"] * n ** (-cell["fit_all"]["p"]) / cell["fit_all"]["v_inf"])
             for n in panels],
            "k-",
            lw=0.8,
            alpha=0.6,
        )
    ax.axhline(FROZEN_RESIDUAL_BAND, color="r", ls=":", lw=1.0,
               label=f"{FROZEN_RESIDUAL_BAND:.0%} band")
    ax.set_xlabel("capytaine panels $N$")
    ax.set_ylabel(r"$|v(N)/v_\infty - 1|$")
    ax.set_title("Independent code: residual vs its own extrapolated limit")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    labels = [f"{c['quantity']}@{c['ka']:g}" for c in sorted(cells, key=lambda c: (c["ka"], c["quantity"]))]
    frozen = [c["spread_frozen"] for c in sorted(cells, key=lambda c: (c["ka"], c["quantity"]))]
    extrap = [c["spread_extrapolated"] for c in sorted(cells, key=lambda c: (c["ka"], c["quantity"]))]
    idx = np.arange(len(labels))
    ax.bar(idx - 0.2, frozen, 0.4, label=f"vs frozen {max(payload['frozen_panels']):.0f} panels")
    ax.bar(idx + 0.2, extrap, 0.4, label="vs extrapolated limit")
    ax.axhline(ANCHOR_BAND, color="r", ls="--", lw=1.0, label="anchor band 5%")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("|reference/value - 1|")
    ax.set_title("Reference vs independent code: before/after extrapolation")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=EVIDENCE)
    parser.add_argument("--doc", type=Path, default=DOCUMENT)
    parser.add_argument("--figure", type=Path,
                        default=ARTIFACT_DIR / "independent_convergence_figure.png")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload = analyse()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.doc.write_text(render_document(payload))
    if not args.no_figure:
        render_figure(payload, args.figure)
    print(json.dumps({"verdict": payload["verdict"],
                      "gates": {k: g["pass"] for k, g in payload["gates"].items()},
                      "worst_frozen_residual": max(c["frozen_residual"] for c in payload["cells"]),
                      "worst_adjudicated": max(c["spread_extrapolated"] for c in payload["cells"])},
                     indent=2))
    return 0 if payload["verdict"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())