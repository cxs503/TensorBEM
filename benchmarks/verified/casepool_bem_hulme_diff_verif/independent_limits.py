"""Richardson-extrapolated third-code limit for every quantity the Hulme case scores.

Why this producer exists
------------------------
The anchor ``CASE-BEM-HULME-DIFF-VERIF`` grades TensorBEM's Kelvin BEM against
``tensorbem.analytic.hulme_hemisphere_radiation``.  That reference is a *re-derived*
multipole series (printed Hulme 1982 digits unavailable) and it shares
``tensorbem.free_surface3d.wave_function_3d`` with the solver, so a kernel defect moves
both routes the same way and leaves the 5 %% gate green.  The only adjudicator in the
sandbox is therefore a third code (capytaine), but it was previously quoted as:

* ``casepool_bem_hulme_hemisphere/capytaine_excitation.json`` -- a committed 4-level
  ladder (256/576/1024/1600 faces) of which only the *finest finite mesh* was used
  (2.3 % worst for the excitation); and
* ``tests/data/hulme_independent_limits.json`` -- Richardson limits, but for the four
  *radiation* coefficients at *two* frequencies only, on an earlier ladder.

A single finite third-code mesh cannot separate "reference is off by 2 %" from "third
code is off by 2 %".  This producer removes that ambiguity over the whole case band:

1. reads the committed 4-level capytaine ladder and Richardson-extrapolates the
   *scattered* force and the four radiation coefficients to the zero-panel limit
   (x = 1/panels, order p estimated per cell);
2. adds the **exact** Froude-Krylov force (``run_case.exact_froude_krylov``, a closed
   reduced quadrature, mesh independent) so no third-code quadrature error enters the
   excitation limit;
3. recomputes the analytic reference from ``analytic.py`` (no stored digits) and reports
   reference <-> independent-limit distance per cell, with the complex-relative metric
   and the wave-pressure-normalised companion metric (meaningful where the force cancels);
4. reports extrapolation quality (estimated order, model residual, spread of two
   overlapping 3-point extrapolations); cells above ``SPREAD_LIMIT`` are flagged
   ``limit_usable = false`` instead of being silently graded.

Nothing in ``src/`` imports this module.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorbem.analytic import hulme_hemisphere_radiation  # noqa: E402

LADDER = (
    REPO_ROOT
    / "benchmarks/verified/casepool_bem_hulme_hemisphere/capytaine_excitation.json"
)
RUN_CASE = Path(__file__).with_name("run_case.py")

GATE_RELATIVE = 0.05        # anchor tolerance from the feature ledger
GATE_ABSOLUTE = 0.01        # companion: |dF| / (rho g A a^2)
SPREAD_LIMIT = 0.02         # extrapolation quality gate
REF_N_TERMS = 144
REF_QUAD_ORDER = 512
RADIATION = ("A11", "B11", "A33", "B33")
DOFS = ("Surge", "Heave")


def _load_run_case():
    spec = importlib.util.spec_from_file_location("hulme_diff_run_case", RUN_CASE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _richardson(xs: np.ndarray, ys: np.ndarray) -> tuple[complex, float, float]:
    """3-point Richardson extrapolation ``y = L + c*x**p``, ``x`` decreasing."""
    d1, d2 = complex(ys[0] - ys[1]), complex(ys[1] - ys[2])
    if abs(d1) == 0.0 or abs(d2) == 0.0:
        p = 1.0
    else:
        target = abs(d1) / abs(d2)
        lo, hi = 0.4, 8.0

        def ratio(p_try: float) -> float:
            return (xs[0] ** p_try - xs[1] ** p_try) / (xs[1] ** p_try - xs[2] ** p_try)

        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if ratio(mid) < target:
                lo = mid
            else:
                hi = mid
        p = 0.5 * (lo + hi)
    denom = (xs[1] / xs[2]) ** p - 1.0
    limit = complex(ys[2]) if abs(denom) < 1e-12 else complex(ys[2]) - (
        complex(ys[1]) - complex(ys[2])) / denom
    resid, scale = 0.0, max(abs(limit), 1e-30)
    if abs(ys[2] - limit) > 0.0:
        c = (ys[2] - limit) / (xs[2] ** p)
        for x, y in zip(xs, ys):
            resid = max(resid, abs(limit + c * x**p - y) / scale)
    return limit, float(p), float(resid)


def _extrapolate(levels, ka_key, *path: str) -> dict:
    xs, ys = [], []
    for lv in levels:
        node = lv["values"][ka_key]
        for part in path:
            node = node[part]
        ys.append(complex(*node) if isinstance(node, list) else complex(node))
        xs.append(1.0 / float(lv["n_faces"]))
    xs, ys = np.asarray(xs), np.asarray(ys, dtype=complex)
    order = np.argsort(-xs)  # x = 1/N descending: coarsest first
    xs, ys = xs[order], ys[order]
    top, p_top, resid_top = _richardson(xs[-3:], ys[-3:])
    low, _, _ = _richardson(xs[:3], ys[:3])
    return {
        "limit": top,
        "order_p": p_top,
        "model_residual": resid_top,
        "spread": float(abs(top - low)),
        "spread_relative": float(abs(top - low) / max(abs(top), 1e-30)),
    }


def _phase(value: complex) -> float:
    return math.degrees(math.atan2(value.imag, value.real))


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def build(ladder_path: Path = LADDER) -> dict:
    raw = ladder_path.read_bytes()
    ladder = json.loads(raw)
    run_case = _load_run_case()
    scale = run_case.DENSITY * run_case.GRAVITY * run_case.RADIUS**2
    levels = sorted(ladder["levels"], key=lambda lv: lv["n_faces"])
    ka_keys = sorted(ladder["reference"], key=float)

    cells: dict[str, dict] = {}
    worst = {"relative": (0.0, ""), "absolute": (0.0, ""), "phase_deg": (0.0, "")}
    flagged: list[str] = []
    for ka_key in ka_keys:
        ka = float(ka_key)
        ref = hulme_hemisphere_radiation(ka, n_terms=REF_N_TERMS, quad_order=REF_QUAD_ORDER)
        cell: dict = {"ka": ka, "quantities": {}, "forces": {}}

        for name, field in (
            ("A11", "surge_added_mass_dimless"), ("B11", "surge_damping_dimless"),
            ("A33", "heave_added_mass_dimless"), ("B33", "heave_damping_dimless"),
        ):
            ref_val = float(getattr(ref, field))
            ext = _extrapolate(levels, ka_key, name)
            point = ext["limit"].real
            rel = (ref_val - point) / point if point else float("inf")
            usable = ext["spread_relative"] <= SPREAD_LIMIT
            if not usable:
                flagged.append(f"{name}@ka={ka:g}")
            cell["quantities"][name] = {
                "reference": ref_val,
                "independent_limit": point,
                "finest_mesh": complex(levels[-1]["values"][ka_key][name]).real,
                "relative": rel,
                "extrapolation": dict(order_p=ext["order_p"],
                                      model_residual=ext["model_residual"],
                                      spread=ext["spread"],
                                      spread_relative=ext["spread_relative"],
                                      limit_usable=usable),
            }
            if abs(rel) > worst["relative"][0]:
                worst["relative"] = (abs(rel), f"{name}@ka={ka:g}")

        fk_x, fk_z = run_case.exact_froude_krylov(ka)
        for dof, fk, ref_force in (
            ("Surge", fk_x, complex(ref.excitation_force_x)),
            ("Heave", fk_z, complex(ref.excitation_force_z)),
        ):
            ext = _extrapolate(levels, ka_key, "excitation_complex", "diffraction", dof)
            total = ext["limit"] + fk
            delta = total - ref_force
            rel = abs(delta) / abs(total)
            absolute = abs(delta) / scale
            phase = _wrap(_phase(ref_force) - _phase(total))
            usable = ext["spread_relative"] <= SPREAD_LIMIT
            tag = f"F{'x' if dof == 'Surge' else 'z'}@ka={ka:g}"
            if not usable:
                flagged.append(tag)
            cell["forces"][dof] = {
                "reference": [ref_force.real, ref_force.imag],
                "exact_froude_krylov": [fk.real, fk.imag],
                "independent_limit": [total.real, total.imag],
                "reference_normalised": [ref_force.real / scale, ref_force.imag / scale],
                "independent_limit_normalised": [total.real / scale, total.imag / scale],
                "amplitude_normalised": abs(total) / scale,
                "relative": rel,
                "absolute_normalised": absolute,
                "phase_difference_deg": phase,
                "extrapolation": dict(order_p=ext["order_p"],
                                      model_residual=ext["model_residual"],
                                      spread=ext["spread"],
                                      spread_relative=ext["spread_relative"],
                                      limit_usable=usable),
            }
            if abs(rel) > worst["relative"][0]:
                worst["relative"] = (abs(rel), tag)
            if absolute > worst["absolute"][0]:
                worst["absolute"] = (absolute, tag)
            if abs(phase) > worst["phase_deg"][0]:
                worst["phase_deg"] = (abs(phase), tag)
        cells[f"{ka:g}"] = cell

    gates = {
        "G1_relative": {"tolerance": GATE_RELATIVE, "worst": worst["relative"][0],
                        "worst_cell": worst["relative"][1],
                        "verdict": "PASS" if worst["relative"][0] <= GATE_RELATIVE else "FAIL"},
        "G2_absolute_normalised_forces": {
            "tolerance": GATE_ABSOLUTE, "worst": worst["absolute"][0],
            "worst_cell": worst["absolute"][1],
            "verdict": "PASS" if worst["absolute"][0] <= GATE_ABSOLUTE else "FAIL"},
        "G3_phase": {"tolerance_deg": 3.0, "worst": worst["phase_deg"][0],
                     "worst_cell": worst["phase_deg"][1],
                     "verdict": "PASS" if worst["phase_deg"][0] <= 3.0 else "FAIL"},
        "G4_extrapolation_quality": {"spread_limit": SPREAD_LIMIT,
                                     "flagged_cells": flagged,
                                     "verdict": "PASS" if not flagged else "FAIL"},
    }
    verdict = "PASS" if all(g["verdict"] == "PASS" for g in gates.values()) else "FAIL"
    return {
        "case_code": "CASE-BEM-HULME-DIFF-VERIF",
        "verdict": verdict,
        "tolerance": f"<{100 * GATE_RELATIVE:.0f}%",
        "gates": gates,
        "worst": {"relative": worst["relative"][0], "relative_cell": worst["relative"][1],
                  "absolute_normalised": worst["absolute"][0],
                  "absolute_cell": worst["absolute"][1],
                  "phase_deg": worst["phase_deg"][0], "phase_cell": worst["phase_deg"][1]},
        "counts": {"cells": len(cells) * (len(RADIATION) + 2 * len(DOFS)),
                   "flagged_extrapolation": len(flagged)},
        "scale_newtons": scale,
        "displaced_volume": float(ladder["nabla_exact"]),
        "reference": {"module": "tensorbem.analytic.hulme_hemisphere_radiation",
                      "n_terms": REF_N_TERMS, "quad_order": REF_QUAD_ORDER},
        "independent_code": {
            "name": "capytaine",
            "version": ladder.get("metadata", {}).get("version"),
            "ladder_faces": [lv["n_faces"] for lv in levels],
            "ladder_file": str(ladder_path.relative_to(REPO_ROOT)),
            "ladder_sha256": hashlib.sha256(raw).hexdigest(),
            "extrapolation": "3-point Richardson in x = 1/panels, order p estimated per cell; "
                             "two overlapping 3-point ladders bound the limit",
            "froude_krylov": "exact reduced quadrature (run_case.exact_froude_krylov), "
                             "mesh independent",
        },
        "machine": {"platform": platform.platform(), "python": platform.python_version(),
                    "numpy": np.__version__},
        "cells": cells,
    }


def render_markdown(payload: dict) -> str:
    w, g, ic = payload["worst"], payload["gates"], payload["independent_code"]
    out = [
        "# CASE-BEM-HULME-DIFF-VERIF：六量第三方 Richardson 极限与参考解偏差",
        "",
        "> 锚点 `CASE-BEM-HULME-DIFF-VERIF`（算例池 id 177，TensorBEM）。本文件由 "
        "`benchmarks/verified/casepool_bem_hulme_diff_verif/independent_limits.py` 从已提交的 "
        "capytaine 四级面元阶梯程序化生成，数字零手抄。",
        "",
        "## 1. 案例介绍",
        "",
        "漂浮半球（a = 1 m，平底贴合静水面，无限水深）的三维绕射激励力与垂荡/纵荡辐射系数。"
        "算例判据：TensorBEM 的 Kelvin 自由面 Green 函数 BEM 对多极级数参考解的复相对误差 <5%。",
        "",
        "| 项 | 值 |",
        "|---|---|",
        "| 波压归一化标尺 ρgAa² | %.3f N |" % payload["scale_newtons"],
        "| 排水体积 ∇ | %.6f m³ |" % payload["displaced_volume"],
        "| 参考解 | `%s`（%d 项，quad %d） |" % (payload["reference"]["module"],
                                             payload["reference"]["n_terms"],
                                             payload["reference"]["quad_order"]),
        "| 第三方码 | capytaine %s，面元 %s |" % (ic["version"],
                                            "/".join(str(n) for n in ic["ladder_faces"])),
        "| 阶梯工件 sha256 | `%s` |" % ic["ladder_sha256"][:16],
        "| 外推 | %s |" % ic["extrapolation"],
        "| Froude-Krylov | %s |" % ic["froude_krylov"],
        "",
        "## 2. 怎么做的",
        "",
        "参考解与求解器共用 `tensorbem.free_surface3d.wave_function_3d`；核内缺陷会让两条路线同向"
        "漂移而门禁照绿。第三方码此前只被用在**单一有限网格**上（1600 面，激励力最差 2.3%），"
        "有限网格分不清「参考解偏 2%」与「第三方码偏 2%」。本工件对全程 ka∈[0.2,2.0]、全部六量做"
        "零面元外推：",
        "",
        "1. 逐量在 x = 1/N 上做三点 Richardson 外推，阶 p 每格单独估计；",
        "2. 激励力极限 = 外推散射力 + 精确 Froude-Krylov（与网格无关），第三方码的 FK 求积误差不进入极限；",
        "3. 参考解由 `analytic.py` 现场重算，不读任何存储数字；",
        "4. 每格记录外推质量：估计阶 p、模型残差、两条重叠三点外推的离散度。",
        "",
        "判据（动手前固定）：G1 复相对 ≤%.0f%%（台账容差）；G2 力波压归一化绝对误差 ≤%.0f%%"
        "（相消区伴随度量）；G3 相位差 ≤3°；G4 外推离散度 ≤%.0f%%。"
        % (100 * g["G1_relative"]["tolerance"],
           100 * g["G2_absolute_normalised_forces"]["tolerance"],
           100 * g["G4_extrapolation_quality"]["spread_limit"]),
        "",
        "## 3. 计算结果",
        "",
        "### 3.1 辐射系数：参考解 vs 第三方外推极限",
        "",
        "| ka | A11 | Δ | B11 | Δ | A33 | Δ | B33 | Δ |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ka_key, cell in sorted(payload["cells"].items(), key=lambda kv: float(kv[0])):
        row = [f"{float(ka_key):.2f}"]
        for name in RADIATION:
            q = cell["quantities"][name]
            row += [f"{q['reference']:.5f}", f"{100 * q['relative']:+.2f}%"]
        out.append("| " + " | ".join(row) + " |")
    out += [
        "",
        "### 3.2 激励力：参考解 vs（外推散射 + 精确 FK）",
        "",
        "| ka | 自由度 | |F|/ρgAa² 极限 | 复相对 | 波压归一化绝对 | 相位差 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for ka_key, cell in sorted(payload["cells"].items(), key=lambda kv: float(kv[0])):
        for dof, label in (("Surge", "纵荡"), ("Heave", "垂荡")):
            f = cell["forces"][dof]
            out.append(f"| {float(ka_key):.2f} | {label} | {f['amplitude_normalised']:.4f} | "
                       f"{100 * f['relative']:.2f}% | {100 * f['absolute_normalised']:.2f}% | "
                       f"{f['phase_difference_deg']:+.2f}° |")
    out += [
        "",
        "### 3.3 外推质量（每格独立估计）",
        "",
        "| ka | 量 | 阶 p | 模型残差 | 外推离散度 | 可用 |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for ka_key, cell in sorted(payload["cells"].items(), key=lambda kv: float(kv[0])):
        for name in RADIATION + ("Surge", "Heave"):
            q = (cell["quantities"][name]["extrapolation"] if name in RADIATION
                 else cell["forces"][name]["extrapolation"])
            out.append(f"| {float(ka_key):.2f} | {name} | {q['order_p']:.2f} | "
                       f"{q['model_residual']:.2e} | {100 * q['spread_relative']:.2f}% | "
                       f"{'是' if q['limit_usable'] else '否'} |")
    out += [
        "",
        "## 4. 误差判定",
        "",
        "- 最差复相对误差：**%.2f%%**（%s），门限 %.0f%% → %s；"
        % (100 * w["relative"], w["relative_cell"], 100 * g["G1_relative"]["tolerance"],
           g["G1_relative"]["verdict"]),
        "- 最差波压归一化绝对误差：**%.2f%%**（%s），门限 %.0f%% → %s；"
        % (100 * w["absolute_normalised"], w["absolute_cell"],
           100 * g["G2_absolute_normalised_forces"]["tolerance"],
           g["G2_absolute_normalised_forces"]["verdict"]),
        "- 最差相位差：**%.2f°**（%s），门限 3° → %s；"
        % (w["phase_deg"], w["phase_cell"], g["G3_phase"]["verdict"]),
        "- 外推质量：%d 格超离散度门限 → %s。"
        % (len(g["G4_extrapolation_quality"]["flagged_cells"]),
           g["G4_extrapolation_quality"]["verdict"]),
        "",
        "**总判定：%s**（%d 格）。" % (payload["verdict"], payload["counts"]["cells"]),
        "",
        "## 5. 注意点与局限",
        "",
        "- capytaine 在 1600 面处相邻差比约 0.6，仍未完全收敛；外推极限带约 1% 量级不确定度，"
        "因此本工件给出的是**带不确定度的界**，不是把参考解钉到 0.1%。",
        "- 参考解与求解器共用自由面核；本工件能发现「核 + 参考解同向偏离第三方」，不能单独定位核内缺陷。",
        "- Hulme (1982) 原文表格数字在沙箱内不可得（无网络、无原文），本工件**不声称**与原文逐位比较。",
        "- 只覆盖零速、深水、线性势流；不含粘性、非线性、有限水深。",
        "",
        "## 6. 复现",
        "",
        "```bash",
        "PYTHONPATH=src python3 benchmarks/verified/casepool_bem_hulme_diff_verif/independent_limits.py \\",
        "    --out benchmarks/verified/casepool_bem_hulme_diff_verif/independent_limits.json \\",
        "    --markdown docs/case_bem_hulme_diff_verif_independent_limits.md",
        "```",
        "",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--markdown", type=Path, default=None)
    ap.add_argument("--ladder", type=Path, default=LADDER)
    args = ap.parse_args(argv)
    payload = build(args.ladder)
    text = json.dumps(payload, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    else:
        print(text)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(payload))
    print(json.dumps({k: v["verdict"] for k, v in payload["gates"].items()}, indent=2))
    print("worst relative: %.3f%% (%s)" % (100 * payload["worst"]["relative"],
                                           payload["worst"]["relative_cell"]))
    print("worst absolute: %.3f%% (%s)" % (100 * payload["worst"]["absolute_normalised"],
                                           payload["worst"]["absolute_cell"]))
    print("verdict:", payload["verdict"])
    return 0 if payload["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())