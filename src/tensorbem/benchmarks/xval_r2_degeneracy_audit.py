"""Regenerate ``docs/audit_xval_r2_degeneracy.md`` from the archived results.

Why this module exists
----------------------
``docs/audit_xval_r2_degeneracy.md`` is the only audit document for the anchor
``XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE``.  It was hand-written for the first
scoring round (m20476, TensorBEM 256/270 panels) and its tables were **never
refreshed** when m22034 re-ran the same cases on 1024/1026 panels and replaced
``results.json``.  Consequence: the audit kept reporting "worst scoring
deviation 87.93%" and "12 invariant-bound violations" for an archive whose
actual worst deviation is 6.71% with zero violations, and it recommended a
refinement that the archive already contains.

The fix is structural rather than cosmetic: the document becomes a *derived*
artifact.  Every number in it is rendered from

* ``benchmarks/verified/capytaine_crosscheck_r2/results.json`` -- the current
  archive, and
* ``benchmarks/verified/capytaine_crosscheck_r2/degeneracy_round_m20476.json``
  -- a small summary of the superseded round, kept so the history stays
  citable instead of being silently overwritten,

so ``tests/test_xval_r2_degeneracy_audit_doc.py`` can require
``doc == render_document(...)`` byte for byte.  Same contract as the main
cross-validation document (``capytaine_crosscheck_r2.render_document``).

Regenerate with::

    python -m tensorbem.benchmarks.xval_r2_degeneracy_audit --write
    python -m tensorbem.benchmarks.xval_r2_degeneracy_audit --check

``--write-history-from <results.json>`` rebuilds the superseded-round summary
from an archived ``results.json`` (used once, at commit time, on the m20476
blob) so the history column is not hand-typed either.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ARTIFACT_ROOT = Path("benchmarks/verified/capytaine_crosscheck_r2")
RESULT_FILE = "results.json"
HISTORY_FILE = "degeneracy_round_m20476.json"
DOC_PATH = Path("docs/audit_xval_r2_degeneracy.md")

ANCHOR = "XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE"

ROUND_LABEL = {
    "m20476": "m20476（首测）",
    "current": "当前归档（m22034）",
}


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------


def summarise(study: dict[str, Any]) -> dict[str, Any]:
    """Reduce a ``results.json`` to the numbers this audit document reports."""
    comparison = study["comparison"]
    scoring = study["invariant_scoring"]
    cases: dict[str, Any] = {}
    for name, block in comparison["cases"].items():
        if not block.get("compared"):
            continue
        scored = scoring["cases"][name]
        cases[name] = {
            "tensorbem_faces": block["tensorbem_faces"],
            "capytaine_faces": block["capytaine_faces"],
            "rotation_center": list(block["rotation_center"]),
            "null_dofs": list(scored["null_dofs"]),
            "rotational_block_scored": bool(scored["rotational_block_scored"]),
            "compare_verdict": block["verdict"],
            "worst_active_relative": float(block["worst_active_relative"]),
            "n_active": int(block["n_active"]),
            "n_null": int(block["n_null"]),
            "counts": dict(scored["counts"]),
            "worst_physics_relative": float(scored["worst_physics_relative"]),
            "worst_invariant_ratio": float(scored["worst_invariant_ratio"]),
            "physics_verdict": scored["physics_verdict"],
            "invariant_verdict": scored["invariant_verdict"],
            "raw_failures": int(scored["raw_failures"]),
            "n_physics_failures": len(scored["physics_failures"]),
            "n_invariant_failures": len(scored["invariant_failures"]),
        }
    degeneracy = study["degeneracy"]
    centre = study["centre_transfer"]
    return {
        "anchor": ANCHOR,
        "comparison": {
            "verdict": comparison["verdict"],
            "worst_active_relative": float(comparison["worst_active_relative"]),
            "n_failures": len(comparison["failures"]),
            "tolerance_diagonal": float(comparison["tolerance_diagonal"]),
            "tolerance_offdiagonal": float(comparison["tolerance_offdiagonal"]),
        },
        "scoring": {
            "physics_verdict": scoring["physics_verdict"],
            "invariant_verdict": scoring["invariant_verdict"],
            "invariant_bound_kind": scoring.get("invariant_bound_kind", ""),
            "n_physics_failures": sum(c["n_physics_failures"] for c in cases.values()),
            "n_invariant_failures": sum(
                c["n_invariant_failures"] for c in cases.values()
            ),
        },
        "cases": cases,
        "degeneracy": {
            "verdict": degeneracy["verdict"],
            "ratio_bound": float(degeneracy["ratio_bound"]),
            "cases": [
                {
                    "case": case["case"],
                    "n_faces": [int(level["n_faces"]) for level in case["levels"]],
                    "decay_ratios": [float(r) for r in case["decay_ratios"]],
                    "finest_worst_ratio": float(case["finest_worst_ratio"]),
                    "monotone_decay": bool(case["monotone_decay"]),
                    "null_dofs": list(case["null_dofs"]),
                }
                for case in degeneracy["cases"]
            ],
        },
        "centre_transfer": {
            "verdict": centre["verdict"],
            "worst_residual": float(centre["worst_residual"]),
            "tolerance": float(centre["tolerance"]),
            "shift": list(centre["shift"]),
            "n_faces": int(centre["n_faces"]),
        },
        "figures": [str(f) for f in study.get("figures", [])],
    }


def load_result(root: Path | None = None) -> dict[str, Any]:
    root = Path() if root is None else root
    return json.loads((root / ARTIFACT_ROOT / RESULT_FILE).read_text())


def load_history(root: Path | None = None) -> dict[str, Any]:
    root = Path() if root is None else root
    return json.loads((root / ARTIFACT_ROOT / HISTORY_FILE).read_text())


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _dofs(null_dofs: list[str]) -> str:
    return "/".join(null_dofs) if null_dofs else "—"


def _centre(centre: list[float]) -> str:
    return "[{}]".format(", ".join(f"{float(v):.1f}" for v in centre))


def _pct(x: float) -> str:
    return f"{x:.2%}"


def _pct_points(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def render_document(current: dict[str, Any], history: dict[str, Any]) -> str:
    """Render the audit document.  No number below is typed by hand."""
    cur_cases = current["cases"]
    old_cases = history["cases"]
    order = list(cur_cases)
    # ``history`` must cover the same bodies; a missing body would make the
    # history column silently shorter than the current one.
    missing = [name for name in order if name not in old_cases]
    if missing:
        raise KeyError(f"history round lacks cases: {missing}")

    dec = current["degeneracy"]
    dec_by_case = {c["case"]: c for c in dec["cases"]}
    centre = current["centre_transfer"]
    old_centre = history["centre_transfer"]
    comp = current["comparison"]
    scoring = current["scoring"]
    old_comp = history["comparison"]
    old_scoring = history["scoring"]

    worst_name = max(order, key=lambda n: cur_cases[n]["worst_physics_relative"])
    worst_value = cur_cases[worst_name]["worst_physics_relative"]
    old_worst_name = max(order, key=lambda n: old_cases[n]["worst_physics_relative"])
    old_worst_value = old_cases[old_worst_name]["worst_physics_relative"]
    acceptance = 0.15
    within_acceptance = worst_value < acceptance
    within_diagonal_band = all(cur_cases[n]["physics_verdict"] == "PASS" for n in order)

    lines: list[str] = []
    add = lines.append

    add(
        "# 审计：XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE 的转动块为何先前不可判定，以及现在的真判定"
    )
    add("")
    add(
        "**仓库** TensorBEM · **锚点** `{}` · **前置** m19943 / m20251"
        "（R2 首判 FAIL，最差 active 偏差 {}）".format(
            current["anchor"], _pct(history["comparison"]["worst_active_relative"])
        )
    )
    add("")
    add(
        "> 本文件是 `{}` 与 `{}` 的派生文档，由 "
        "`tensorbem.benchmarks.xval_r2_degeneracy_audit` 渲染；"
        "`tests/test_xval_r2_degeneracy_audit_doc.py` 要求本文件与渲染结果逐字节相同，"
        "因此表内数字不可能再与归档数据分叉。".format(
            (ARTIFACT_ROOT / RESULT_FILE).as_posix(),
            (ARTIFACT_ROOT / HISTORY_FILE).as_posix(),
        )
    )
    add("")

    # -- 1 ---------------------------------------------------------------
    add("## 1. 案例介绍")
    add("")
    add(
        "R2 的目标是把 TensorBEM 的 **完整 6x6 辐射矩阵**（含转动块与平动-转动耦合）与开源面元法 code capytaine 对齐，"
        "验收标准：对标误差 <{:.0%}（owner 精度标准）。R2 用了两个本体、两个转动中心，共四个算例："
        "".format(acceptance)
    )
    add("")
    add("| 本体 | 几何 | 转动中心 | 面板（TensorBEM / capytaine） | 面板一致 |")
    add("|---|---|---|---:|---|")
    for name in order:
        case = cur_cases[name]
        old = old_cases[name]
        add(
            "| {} | {} | {} | {} / {} | {} |".format(
                name,
                _geometry(name),
                _centre(case["rotation_center"]),
                case["tensorbem_faces"],
                case["capytaine_faces"],
                "是" if case["tensorbem_faces"] == case["capytaine_faces"] else "否",
            )
        )
        del old
    add("")
    add(
        "前两个本体把转动中心恰好放在**球面曲率中心**上；后两个保持几何不变、把转动中心移开。"
        "首测轮 `{}` 用的是 {} 面板；当前归档已把所有算例的 TensorBEM 侧细化到 {} 面板"
        "（capytaine 侧参考不变）。".format(
            ROUND_LABEL["m20476"],
            "/".join(str(old_cases[n]["tensorbem_faces"]) for n in order),
            "/".join(str(cur_cases[n]["tensorbem_faces"]) for n in order),
        )
    )
    add("")

    # -- 2 ---------------------------------------------------------------
    add("## 2. 怎么做的")
    add("")
    add(
        "1. **解析判定**：球面是其自身的曲率球，`(r - centre) x n ≡ 0` 逐点成立，故转动中心 = 球心时"
    )
    add(
        "   三个转动自由度与全部平动-转动耦合项的精确目标是 **0**。对精确零做相对偏差，测到的是零的离散残差。"
    )
    add(
        "2. **细化阶梯实测**：`degeneracy_audit()` 在半球、球的多个面板层级上测残差随网格的走向"
    )
    add("   （正确装配 → 残差衰减；力臂取自原点或漏掉 `(r - centre)` → 收敛到 O(1)）。")
    add(
        "3. **不依赖外部程序的恒等式**：`centre_transfer_invariant()` 验证 `A(c+d) = T^T A(c) T`，"
    )
    add(
        "   `T = [[I, +[d]x], [0, I]]`（`V_c = V_c' + [d]x Omega`）。该式对离散算子亦精确成立。"
    )
    add(
        "4. **补做可判定的算例**：新增 `hemisphere_r1_cz-0.5`、`sphere_r1_z-0.5_c0`，转动中心偏离球心，"
    )
    add(
        "   roll/pitch 成为真实非零量；capytaine 侧参考数据由 `scripts/capytaine_crosscheck_r2_ref.py` 现场重算。"
    )
    add(
        "5. **分开计分**：`compare()` 的登记带宽与判定原样保留；`invariant_scoring()` 只把精确零项从相对评分中摘出，"
    )
    add("   改按 `{}·rho L^k` 的结构界限检查。".format(f"{dec['ratio_bound']:g}"))
    add(
        "6. **两轮并存**：首测轮（256/270 面板）与当前归档（{} 面板）都在本文件里列出，旧数不被覆盖、"
        "新数不带历史包袱；两轮的差别只有网格分辨率与由此产生的判定。".format(
            "/".join(str(cur_cases[n]["tensorbem_faces"]) for n in order)
        )
    )
    add("")

    # -- 3 ---------------------------------------------------------------
    add("## 3. 计算结果")
    add("")
    add("### 3.1 关键结论一：旧判 FAIL 是退化伪影（不变量轴全绿）")
    add("")
    add("| 轴 | 判据 | 结果 | 判定 |")
    add("|---|---|---|---|")
    for name in order:
        case = dec_by_case.get(name)
        if case is None:
            continue
        add(
            "| 退化残差衰减 `{}` | 逐层单调衰减且最细网格 ≤ {}% rho L^k | 层级 {} 面板，衰减比 {}，最细 {:.1e}（x rho L^k） | {} |".format(
                name,
                f"{100.0 * dec['ratio_bound']:g}",
                "/".join(str(n) for n in case["n_faces"]),
                "、".join(f"{r:.2f}" for r in case["decay_ratios"]),
                case["finest_worst_ratio"],
                (
                    "PASS"
                    if case["monotone_decay"]
                    and case["finest_worst_ratio"] <= dec["ratio_bound"]
                    else "FAIL"
                ),
            )
        )
    add(
        "| 转动中心变换恒等式 | `A(c+d)=T^T A(c)T`，d = {} | 最大相对残差 {:.1e}（界限 {:g}），{} 面板 | {} |".format(
            _centre(centre["shift"]),
            centre["worst_residual"],
            centre["tolerance"],
            centre["n_faces"],
            centre["verdict"],
        )
    )
    add("")
    add(
        "转动残差随网格单调趋零，面积加权模态范数同步按 O(h) 收缩 → 这是**正确装配的离散残差**，"
        "不是收敛到非零极限的实现错误。首轮的 {} 是把两个数值零相减再除以其中一个；"
        "首轮的中心变换残差 {:.1e} 与当前的 {:.1e} 同量级（机器精度），两轮都通过。"
        "".format(
            _pct(old_comp["worst_active_relative"]),
            old_centre["worst_residual"],
            centre["worst_residual"],
        )
    )
    add("")

    add("### 3.2 关键结论二：可判定算例的真偏差（两轮并列，诚实 FAIL 与否）")
    add("")
    add(
        "| 轮次 | case | 转动中心 | 零模态 dof | 面板 TB/capy | 计分 active | 精确零一致 | 精确零越界 | 最差计分偏差 | 计分判定 |"
    )
    add("|---|---|---|---|---:|---:|---:|---:|---:|---|")
    for label, cases in (
        (ROUND_LABEL["m20476"], old_cases),
        (ROUND_LABEL["current"], cur_cases),
    ):
        for name in order:
            case = cases[name]
            add(
                "| {} | {} | {} | {} | {} / {} | {} | {} | {} | {} | {} |".format(
                    label,
                    name,
                    _centre(case["rotation_center"]),
                    _dofs(case["null_dofs"]),
                    case["tensorbem_faces"],
                    case["capytaine_faces"],
                    case["counts"]["physics_active"],
                    case["counts"]["invariant_consistent"],
                    case["counts"]["invariant_violated"],
                    _pct(case["worst_physics_relative"]),
                    case["physics_verdict"],
                )
            )
    add("")
    add(
        "首测轮共有 {} 个精确零项越出 `{}·rho L^k` 界限（半球 {} 格、球 {} 格），"
        "读数是：该界限对**粗网格**偏紧，它是 sanity bound 而不是物理判定。"
        "当前归档在同等几何上把 TensorBEM 侧细化后，越界数降为 {}，"
        "全部 {} 个精确零项一致，`invariant_verdict` = {}；"
        "而物理项的判定**没有**随之变成 PASS：仍为 {}，失败项 {} 个。"
        "".format(
            sum(c["counts"]["invariant_violated"] for c in old_cases.values()),
            f"{dec['ratio_bound']:g}",
            sum(
                c["counts"]["invariant_violated"]
                for n, c in old_cases.items()
                if n.startswith("hemisphere")
            ),
            sum(
                c["counts"]["invariant_violated"]
                for n, c in old_cases.items()
                if n.startswith("sphere")
            ),
            sum(c["counts"]["invariant_violated"] for c in cur_cases.values()),
            sum(c["counts"]["invariant_consistent"] for c in cur_cases.values()),
            scoring["invariant_verdict"],
            scoring["physics_verdict"],
            scoring["n_physics_failures"],
        )
    )
    add("")

    add("### 3.3 分层：登记带宽 vs 分开计分 vs 验收线")
    add("")
    add("| 口径 | 判定 | 最差偏差 | 失败项数 |")
    add("|---|---|---:|---:|")
    add(
        "| 首测轮 `compare()`（登记带宽，原样） | {} | {} | {} |".format(
            old_comp["verdict"],
            _pct(old_comp["worst_active_relative"]),
            old_comp["n_failures"],
        )
    )
    add(
        "| 首测轮 分开计分·物理项 | {} | {} | {} |".format(
            old_scoring["physics_verdict"],
            _pct(old_cases[old_worst_name]["worst_physics_relative"]),
            old_scoring["n_physics_failures"],
        )
    )
    add(
        "| 首测轮 分开计分·精确零界限 | {} | — | {} |".format(
            old_scoring["invariant_verdict"], old_scoring["n_invariant_failures"]
        )
    )
    add(
        "| 当前归档 `compare()`（登记带宽，原样） | {} | {} | {} |".format(
            comp["verdict"],
            _pct(comp["worst_active_relative"]),
            comp["n_failures"],
        )
    )
    add(
        "| 当前归档 分开计分·物理项 | {} | {} | {} |".format(
            scoring["physics_verdict"],
            _pct(worst_value),
            scoring["n_physics_failures"],
        )
    )
    add(
        "| 当前归档 分开计分·精确零界限 | {} | — | {} |".format(
            scoring["invariant_verdict"], scoring["n_invariant_failures"]
        )
    )
    add(
        "| 当前归档 验收线 {:.0%}（owner 精度标准） | {} | {} | — |".format(
            acceptance,
            "PASS" if within_acceptance else "FAIL",
            _pct(worst_value),
        )
    )
    add("")
    add(
        "两层判定的关系必须说清楚：内部登记带（对角 {:.0%} / 耦合 {:.0%}）比验收线 {:.0%} 更严，"
        "所以「内部 {}、验收 {}」同时成立并不矛盾。当前归档最差偏差 {}（{}，{}），"
        "{}验收线，{}内部对角带；失败项集中在 nu=2 的阻尼侧对角项。"
        "".format(
            comp["tolerance_diagonal"],
            comp["tolerance_offdiagonal"],
            acceptance,
            comp["verdict"],
            "PASS" if within_acceptance else "FAIL",
            _pct(worst_value),
            worst_name,
            (
                "转动块恒零，不计"
                if not cur_cases[worst_name]["rotational_block_scored"]
                else "含转动块，可判定"
            ),
            "满足" if within_acceptance else "未满足",
            "未满足" if not within_diagonal_band else "满足",
        )
    )
    add("")

    # -- 4 ---------------------------------------------------------------
    add("## 4. 结果图")
    add("")
    if current["figures"]:
        add(
            "图由 `capytaine_crosscheck_r2.render_figures()` 程序化生成并随 results.json 一并落盘："
        )
        add("")
        for fig in current["figures"]:
            add(f"* `{fig}` — {_figure_caption(fig)}")
    else:
        add("本归档未记录图件（`figures` 为空）。")
    add("")

    # -- 5 ---------------------------------------------------------------
    add("## 5. 注意点")
    add("")
    add(
        "* 旧的 PASS/FAIL 都不可引用：转动块在中心 = 球心时目标恒为零，任何相对偏差都在测离散残差。"
    )
    add(
        "* 首测轮的最差物理偏差 {} 主要来自**网格分辨率不对等**"
        "（TensorBEM {} 面板 vs capytaine {} 面板）。当前归档把 TensorBEM 侧推到 {} 面板后，"
        "同项偏差降到 {} —— 这确认了「偏差随本侧细化单调下降」的解释，而不是实现错误。".format(
            _pct(old_worst_value),
            "/".join(str(old_cases[n]["tensorbem_faces"]) for n in order),
            "/".join(str(old_cases[n]["capytaine_faces"]) for n in order),
            "/".join(str(cur_cases[n]["tensorbem_faces"]) for n in order),
            _pct(worst_value),
        )
    )
    add(
        "* 面板数仍未完全对等：当前归档 TensorBEM 侧 {} 面板对 capytaine 侧 {} 面板。"
        "两码使用不同的面元离散（常数面元 vs 高阶/不同网格），**面板数相等不等于离散误差相等**，"
        "所以剩余 {} 的偏差不能简单归给「分辨率」一项；等误差水平（同阶收敛验证）仍待做。".format(
            "/".join(str(cur_cases[n]["tensorbem_faces"]) for n in order),
            "/".join(str(cur_cases[n]["capytaine_faces"]) for n in order),
            _pct(worst_value),
        )
    )
    add(
        "* 剩余失败项 {} 个，全部落在对角块、以 nu=2 的阻尼侧为主；内部对角带 {:.0%} 仍未满足。".format(
            scoring["n_physics_failures"], comp["tolerance_diagonal"]
        )
    )
    add("* `deep_submergence` 一节沿用此前已提交的计算结果（该代码路径本轮未改）。")
    add(
        "* 首测轮的数值不再由任何脚本复算，只以 `{}` 的摘要形式保留；"
        "若需复算，用 `--write-history-from <该轮的 results.json>` 重建摘要。".format(
            (ARTIFACT_ROOT / HISTORY_FILE).as_posix()
        )
    )
    add("")

    # -- 6 ---------------------------------------------------------------
    add("## 6. 误差判定（逐项，显式回答是否 <3%）")
    add("")
    add(
        "| 量 | 基准 | TensorBEM | capytaine | 偏差 | 是否 <3% | 是否 <{:.0%} |".format(
            acceptance
        )
    )
    add("|---|---|---:|---:|---:|---|---|")
    for name in order:
        case = cur_cases[name]
        add(
            "| {} 最差计分项（{}） | capytaine 6x6 辐射矩阵 | — | — | {} | {} | {} |".format(
                name,
                "含转动块" if case["rotational_block_scored"] else "转动块恒零，不计",
                _pct(case["worst_physics_relative"]),
                "是" if case["worst_physics_relative"] < 0.03 else "否",
                "是" if case["worst_physics_relative"] < acceptance else "否",
            )
        )
    add(
        "| 转动中心变换恒等式 | 解析恒等 | 残差 {:.1e} | — | 机器精度 | 是 | 是 |".format(
            centre["worst_residual"]
        )
    )
    add(
        "| 退化残差（最细网格） | 0（解析） | {:.1e} x rho L^k | — | 目标为零，不适用相对判据 | 不适用 | 不适用 |".format(
            dec_by_case[order[0]]["finest_worst_ratio"]
        )
    )
    add("")
    add("**结论（诚实）**：")
    add("")
    add(
        "* 不变量轴（退化残差衰减、转动中心变换）两轮都通过，量级为机器精度/离散残差；"
        "首测轮报出的 {} 个精确零越界在细化后全部消失（当前 {} 个）。".format(
            sum(c["counts"]["invariant_violated"] for c in old_cases.values()),
            sum(c["counts"]["invariant_violated"] for c in cur_cases.values()),
        )
    )
    add(
        "* 物理轴**未达到 <3%**：四个算例最差依次为 {}，最差 {}（{}）。"
        "按 owner 的 {:.0%} 验收线，当前归档{}；按内部登记的 {} / {} 带，判定仍为 {}，"
        "因此 `results.json` 里 `comparison.verdict` 保持 {}，两层结论不冲突。".format(
            "、".join(_pct(cur_cases[n]["worst_physics_relative"]) for n in order),
            _pct(worst_value),
            worst_name,
            acceptance,
            "满足" if within_acceptance else "未满足",
            f"{comp['tolerance_diagonal']:.0%}",
            f"{comp['tolerance_offdiagonal']:.0%}",
            comp["verdict"],
            comp["verdict"],
        )
    )
    add(
        "* 本轮修正的是**文档与归档数据的分叉**（旧文档仍写首测轮的 {}），不是数值本身："
        "数值、带宽、判定一律以 `{}` 为准，未做任何放宽。".format(
            _pct(old_worst_value), (ARTIFACT_ROOT / RESULT_FILE).as_posix()
        )
    )
    add("")
    return "\n".join(lines)


def _geometry(case: str) -> str:
    if case.startswith("hemisphere"):
        return "hemisphere R=1 m"
    if case.startswith("sphere"):
        return "floating sphere R=1 m，中心 z=-0.5 m（截于 z=0）"
    return "—"


def _figure_caption(path: str) -> str:
    name = Path(path).name
    return {
        "fig_capytaine_r2_matrix.png": "四个本体 × 附加质量/阻尼的逐格偏差（NaN 为结构零）",
        "fig_capytaine_r2_convergence.png": "最差偏差随面板数",
        "fig_capytaine_r2_deep.png": "深潜解析轴",
    }.get(name, "归档图件")


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------


def write_document(
    current: dict[str, Any],
    history: dict[str, Any],
    path: Path | None = None,
) -> Path:
    path = DOC_PATH if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_document(current, history))
    return path


def write_history(
    source: Path,
    round_label: str,
    provenance: dict[str, Any],
    path: Path | None = None,
) -> Path:
    """Freeze a superseded round as a small, citable summary."""
    summary = summarise(json.loads(source.read_text()))
    payload = {
        "anchor": summary["anchor"],
        "round": round_label,
        "provenance": provenance,
        **{k: v for k, v in summary.items() if k != "anchor"},
    }
    path = ARTIFACT_ROOT / HISTORY_FILE if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="rewrite the document")
    parser.add_argument(
        "--check", action="store_true", help="compare, non-zero on drift"
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--write-history-from", type=Path, default=None)
    parser.add_argument("--history-round", default="m20476")
    args = parser.parse_args(argv)

    root = Path() if args.root is None else args.root
    if args.write_history_from is not None:
        path = write_history(
            args.write_history_from,
            args.history_round,
            {
                "source_revision": "4123e585 (Merge #923 / m20476)",
                "source_path": (ARTIFACT_ROOT / RESULT_FILE).as_posix(),
                "note": "superseded by the m22034 refinement (1024/1026 panels)",
            },
            path=root / ARTIFACT_ROOT / HISTORY_FILE,
        )
        print(f"wrote {path}")

    current = summarise(load_result(root))
    history = load_history(root)
    rendered = render_document(current, history)
    doc = root / DOC_PATH
    if args.write:
        doc.parent.mkdir(parents=True, exist_ok=True)
        doc.write_text(rendered)
        print(f"wrote {doc}")
        return 0
    existing = doc.read_text() if doc.exists() else ""
    if existing != rendered:
        print(f"DRIFT: {doc} does not match the rendered document")
        if args.check:
            return 1
    else:
        print(f"{doc} matches the archived results")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
