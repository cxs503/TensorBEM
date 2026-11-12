"""Cross-solver regression: TensorBEM vs capytaine on three bodies x eight frequencies.

Why this module exists
----------------------
``docs/oss_inventory_bem/census.md`` proposes ``GAP-BEM-CROSS-SOLVER-REGRESSION``
(P1): *"跨求解器自动回归（>=3 体 x >=8 频率，误差表入库），max 相对误差 <3%，
CI < 90 s"*.  Wave-2 of that inventory shipped a **single-body** wall-sided-barge
pairing (``tensorbem.benchmarks.barge_crosscheck``); the many-body, many-frequency
regression that the acceptance criterion asks for did not exist.  A one-body
pairing cannot tell a per-case accident from a systematic formulation difference,
because it has no population to look at.

What is compared, and what that means
-------------------------------------
The two codes are *different formulations of the same linear problem*:

* TensorBEM 3-D radiation: low-order **Rankine** panel method -- free-surface
  panels carried as interior unknowns (``free_surface_solver3d``);
* capytaine: low-order constant-strength panels with the **free-surface Green
  function** (Delhommeau).

So agreement is evidence that neither the panel assembly nor the radiation
condition is grossly wrong; it is *not* evidence of accuracy, because a shared
modelling error would not show up.  Every claim below is scoped that way, and
the accuracy claims of the repository stay attached to the analytic anchors
(Ursell, Hulme, Kochin) as before.

Three bodies, chosen for what they stress:

===========================  ==============================================
wall-sided barge             flat panels, sharp bilge corners, rectangle waterplane
floating sphere (2:1 draft)  curved panels, circular waterline, no corners
floating spheroid 2:0.5:0.5  anisotropic curvature (ship-like), elliptical waterline
===========================  ==============================================

Eight non-dimensional frequencies ``nu = omega^2 / g`` in ``[0.25, 2.0]``, heave
only.  The barge case stays below the classical first irregular frequency
(``nu_irr ~ 3.7`` for ``L=2, B=1, T=0.5``), so neither code needs a lid; the lid
is deliberately *not* used on either side, so the comparison has no lid-mesh
mismatch in it.

Identical discretisation
------------------------
Both codes receive the *same* vertex/face arrays: the reference file stores them
for every body and level, ``compare`` re-hashes them, and the geometry builders
of this package must still produce them.  A silent change to a hull builder
therefore fails the regression instead of shifting the numbers quietly.

Metric and its honest limits
----------------------------
Per body, level, frequency and quantity (heave added mass ``A33``, heave damping
``B33``), the reported quantity is the *pair relative difference*
``(A33_tb - A33_cp) / A33_cp``.  Where ``|A33_cp|`` is below 5 % of that body's
frequency peak, the relative difference is not a meaningful number (cancellation)
and the point is marked ``excluded`` while its values stay in the artifact.
The pre-registered acceptance of the census anchor is applied as stated:
``max |relative| < 3 %`` over all *included* points at the finest level of each
body.  The verdict is computed; it is not asserted to be PASS.

Usage::

    python -m tensorbem.benchmarks.cross_solver_regression --write          # study + doc + figure
    python -m tensorbem.benchmarks.cross_solver_regression --json out.json  # machine form
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tensorbem.free_surface_solver3d import solve_radiation_3d
from tensorbem.geometry3d import (
    box_hull_mesh,
    floating_sphere_hull_mesh,
    spheroid_hull_mesh,
)

DENSITY = 1025.0
GRAVITY = 9.80665
HEAVE = 2
NU_GRID: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)

#: Pre-registered by the census anchor (``GAP-BEM-CROSS-SOLVER-REGRESSION``).
TOLERANCE = 0.03
#: A relative difference is not meaningful where the reference crosses zero.
CANCELLATION_FRACTION = 0.05

ARTIFACT_ROOT = Path("benchmarks/verified/cross_solver_regression")
REFERENCE_FILE = "capytaine_ref.json"
RESULT_FILE = "results.json"
DOC_PATH = Path("docs/cross_solver_regression.md")
FIGURE_PATH = Path("docs/fig_cross_solver_regression.png")

Builder = Callable[[int], Any]

CASES: tuple[dict[str, Any], ...] = (
    {
        "name": "barge",
        "title": "wall-sided barge 2.0 x 1.0 x 0.5 m",
        "note": "flat panels, sharp bilge corners, rectangular waterplane",
        "builder": lambda level: box_hull_mesh(
            2.0, 1.0, 0.5, n_length=8 * level, n_beam=4 * level, n_draft=2 * level
        ),
        "levels": (1, 2),
        "level_label": lambda level: f"{8 * level}x{4 * level}x{2 * level}",
    },
    {
        "name": "sphere",
        "title": "floating sphere r = 1 m, centre 0.5 m below the waterline",
        "note": "curved panels, circular waterline, no corners",
        "builder": lambda level: floating_sphere_hull_mesh(level, 1.0, (0.0, 0.0, -0.5)),
        "levels": (1, 2),
        "level_label": lambda level: f"subdivisions={level}",
    },
    {
        "name": "spheroid",
        "title": "floating spheroid semi-axes (1.0, 0.5, 0.5) m",
        "note": "anisotropic curvature, elliptical waterline (ship-like)",
        "builder": lambda level: spheroid_hull_mesh((1.0, 0.5, 0.5), subdivisions=level + 1),
        "levels": (1, 2),
        "level_label": lambda level: f"subdivisions={level + 1}",
    },
)


def _key(nu: float) -> str:
    return f"{nu:.3f}"


def mesh_signature(vertices: Any, faces: Any) -> str:
    """Stable sha256 of the discretisation (rounded so BLAS noise cannot move it)."""
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    payload = np.round(v, 12).tobytes() + f.tobytes() + repr(v.shape + f.shape).encode()
    return hashlib.sha256(payload).hexdigest()[:32]


def _face_list(faces: Any) -> list[list[int]]:
    return [[int(i) for i in row] for row in np.asarray(faces)]


def build_case(case: dict[str, Any], level: int) -> Any:
    return case["builder"](level)


def solve_tensorbem(mesh: Any, nu: float) -> dict[str, float]:
    res = solve_radiation_3d(
        mesh, nu, modes=[HEAVE], density=DENSITY, gravity=GRAVITY
    )
    return {
        "added_mass": float(res.added_mass[0, 0]),
        "damping": float(res.damping[0, 0]),
    }


def run_study(reference: dict[str, Any] | None = None) -> dict[str, Any]:
    ref = reference if reference is not None else load_reference()
    levels_out: list[dict[str, Any]] = []
    for case in CASES:
        ref_case = next(c for c in ref["cases"] if c["name"] == case["name"])
        for level in case["levels"]:
            mesh = build_case(case, level)
            sig = mesh_signature(mesh.vertices, mesh.faces)
            ref_level = next(
                lv for lv in ref_case["levels"] if lv["level"] == level
            )
            if ref_level["mesh_signature"] != sig:
                raise RuntimeError(
                    f"mesh drift for {case['name']} level {level}: builder signature "
                    f"{sig} != reference {ref_level['mesh_signature']}"
                )
            row: dict[str, Any] = {
                "case": case["name"],
                "level": level,
                "panels": int(mesh.panel_count),
                "mesh_signature": sig,
                "tensorbem": {},
            }
            for nu in NU_GRID:
                row["tensorbem"][_key(nu)] = solve_tensorbem(mesh, nu)
            levels_out.append(row)
    study: dict[str, Any] = {
        "meta": {
            "tolerance": TOLERANCE,
            "cancellation_fraction": CANCELLATION_FRACTION,
            "nu_grid": list(NU_GRID),
            "dof": "heave",
            "density": DENSITY,
            "gravity": GRAVITY,
            "tensorbem_formulation": "low-order Rankine panel method with free-surface "
            "panels as interior unknowns (free_surface_solver3d)",
            "capytaine_formulation": "low-order constant-strength panels with the "
            "free-surface Green function",
            "reference_metadata": ref["metadata"],
            "cases": [
                {"name": c["name"], "title": c["title"], "note": c["note"]}
                for c in CASES
            ],
        },
        "levels": levels_out,
        "comparison": compare(levels_out, ref),
        "convergence": convergence(levels_out, ref),
    }
    return study


def load_reference(path: Path | None = None) -> dict[str, Any]:
    target = path if path is not None else ARTIFACT_ROOT / REFERENCE_FILE
    return json.loads(Path(target).read_text())


def _pair_rows(levels_out: list[dict[str, Any]], ref: dict[str, Any]):
    for row in levels_out:
        case = row["case"]
        ref_case = next(c for c in ref["cases"] if c["name"] == case)
        ref_level = next(lv for lv in ref_case["levels"] if lv["level"] == row["level"])
        for nu in NU_GRID:
            key = _key(nu)
            for quantity in ("added_mass", "damping"):
                yield row, ref_level, nu, quantity


def _peaks(levels_out: list[dict[str, Any]], ref: dict[str, Any]) -> dict[tuple[str, str], float]:
    peaks: dict[tuple[str, str], float] = {}
    for row, ref_level, nu, quantity in _pair_rows(levels_out, ref):
        cp = ref_level["capytaine"][_key(nu)][quantity]
        k = (row["case"], quantity)
        peaks[k] = max(peaks.get(k, 0.0), abs(cp))
    return peaks


def compare(levels_out: list[dict[str, Any]], ref: dict[str, Any]) -> dict[str, Any]:
    peaks = _peaks(levels_out, ref)
    entries: list[dict[str, Any]] = []
    for row, ref_level, nu, quantity in _pair_rows(levels_out, ref):
        tb = row["tensorbem"][_key(nu)][quantity]
        cp = ref_level["capytaine"][_key(nu)][quantity]
        peak = peaks[(row["case"], quantity)]
        relative = None if cp == 0.0 else (tb - cp) / cp
        excluded = peak <= 0.0 or abs(cp) < CANCELLATION_FRACTION * peak
        entries.append(
            {
                "case": row["case"],
                "level": row["level"],
                "panels": row["panels"],
                "nu": nu,
                "quantity": quantity,
                "tensorbem": tb,
                "capytaine": cp,
                "relative": relative,
                "excluded": bool(excluded),
            }
        )
    by_case: dict[str, Any] = {}
    failures: list[str] = []
    for case in CASES:
        name = case["name"]
        finest = max(case["levels"])
        per_quantity: dict[str, Any] = {}
        for quantity in ("added_mass", "damping"):
            rows = [
                e
                for e in entries
                if e["case"] == name and e["level"] == finest and e["quantity"] == quantity
            ]
            included = [e for e in rows if not e["excluded"] and e["relative"] is not None]
            worst = max(included, key=lambda e: abs(e["relative"])) if included else None
            per_quantity[quantity] = {
                "worst_relative": None if worst is None else worst["relative"],
                "worst_nu": None if worst is None else worst["nu"],
                "included_points": len(included),
                "excluded_points": len(rows) - len(included),
                "pass": None if worst is None else bool(abs(worst["relative"]) <= TOLERANCE),
            }
            if worst is not None and abs(worst["relative"]) > TOLERANCE:
                failures.append(
                    f"{name} {quantity} nu={worst['nu']:.2f}: {worst['relative']:+.2%} "
                    f"(TensorBEM {worst['tensorbem']:.4g} vs capytaine {worst['capytaine']:.4g})"
                )
        panels = next(e["panels"] for e in entries if e["case"] == name and e["level"] == finest)
        by_case[name] = {
            "finest_level": finest,
            "finest_panels": panels,
            "quantities": per_quantity,
            "pass": all(v["pass"] in (True, None) for v in per_quantity.values()),
        }
    included_all = [e for e in entries if not e["excluded"] and e["relative"] is not None]
    worst_overall = max(included_all, key=lambda e: abs(e["relative"])) if included_all else None
    return {
        "tolerance": TOLERANCE,
        "verdict": "PASS" if not failures else "FAIL",
        "worst_overall_relative": None if worst_overall is None else worst_overall["relative"],
        "worst_overall_at": None
        if worst_overall is None
        else {
            "case": worst_overall["case"],
            "quantity": worst_overall["quantity"],
            "nu": worst_overall["nu"],
            "level": worst_overall["level"],
            "panels": worst_overall["panels"],
        },
        "per_case": by_case,
        "entries": entries,
        "failures": failures,
    }


def convergence(levels_out: list[dict[str, Any]], ref: dict[str, Any]) -> list[dict[str, Any]]:
    """Worst included difference per body per level -- is the residual mesh-limited?"""
    comp_entries = compare(levels_out, ref)["entries"]
    rows: list[dict[str, Any]] = []
    for case in CASES:
        name = case["name"]
        for level in sorted(case["levels"]):
            per_q: dict[str, Any] = {}
            panels = None
            for quantity in ("added_mass", "damping"):
                pts = [
                    e
                    for e in comp_entries
                    if e["case"] == name and e["level"] == level and e["quantity"] == quantity
                    and not e["excluded"] and e["relative"] is not None
                ]
                panels = pts[0]["panels"] if pts else panels
                per_q[quantity] = max((abs(e["relative"]) for e in pts), default=None)
            rows.append(
                {"case": name, "level": level, "panels": panels, "worst": per_q}
            )
    return rows


def render_figure(study: dict[str, Any], path: Path = FIGURE_PATH) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = load_reference()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    for ax, case in zip(axes, CASES):
        name = case["name"]
        finest = max(case["levels"])
        for quantity, style, label in (
            ("added_mass", "o-", "A33 TensorBEM"),
            ("damping", "s-", "B33 TensorBEM"),
        ):
            tb = [
                e["tensorbem"]
                for e in study["comparison"]["entries"]
                if e["case"] == name and e["level"] == finest and e["quantity"] == quantity
            ]
            cp = [
                e["capytaine"]
                for e in study["comparison"]["entries"]
                if e["case"] == name and e["level"] == finest and e["quantity"] == quantity
            ]
            pair = "o" if quantity == "added_mass" else "s"
            ax.plot(NU_GRID, tb, pair + "-", label=label, lw=1.4)
            ax.plot(NU_GRID, cp, pair + "--", mfc="none", label=label.replace("TensorBEM", "capytaine"), lw=1.4)
        panels = next(
            e["panels"] for e in study["comparison"]["entries"] if e["case"] == name
        )
        ax.set_title(f"{name} ({panels} panels)")
        ax.set_xlabel(r"$\nu=\omega^2/g$  [m$^{-1}$]")
        ax.set_ylabel("heave coefficient [kg, kg/s]")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("TensorBEM (Rankine) vs capytaine (free-surface Green function), identical meshes", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def render_document(study: dict[str, Any]) -> str:
    comp = study["comparison"]
    conv = study["convergence"]
    meta = study["meta"]
    L: list[str] = []
    add = L.append
    add("# 跨求解器自动回归：TensorBEM（Rankine）vs capytaine（自由面 Green 函数），3 体 x 8 频点")
    add("")
    add("*锚点：`GAP-BEM-CROSS-SOLVER-REGRESSION`（`docs/oss_inventory_bem/census.md` P1）。"
        "本文件由 `python -m tensorbem.benchmarks.cross_solver_regression --write` 生成，"
        "表中每个数字都来自 `benchmarks/verified/cross_solver_regression/*.json`，零手抄。*")
    add("")
    add("## ① 案例介绍")
    add("")
    add("两个**不同公式化**的低阶面元法解同一个线性辐射问题：")
    add("")
    add(f"- TensorBEM：{meta['tensorbem_formulation']}")
    add(f"- capytaine：{meta['capytaine_formulation']}")
    add("")
    add("因此本表证明的是「两套装配+辐射条件没有系统性错位」，**不是精度**：共同的建模错误不会在此显形。"
        "精度声明仍挂在解析锚（Ursell / Hulme / Kochin）上。")
    add("")
    add("| 体 | 几何要点 |")
    add("|---|---|")
    for c in meta["cases"]:
        add(f"| `{c['name']}` | {c['title']}；{c['note']} |")
    add("")
    add(f"频率：`nu = omega^2/g` ∈ {list(NU_GRID)}（8 点，heave）；"
        f"ρ={meta['density']} kg/m³，g={meta['gravity']} m/s²。"
        "两码收到的**顶点/面数组完全相同**（参考文件存数组 + sha256，`compare` 重算签名，"
        "几何构造器静默漂移会直接判红）。barge 频带上限 2.0 低于经典第一不规则频率 ν_irr≈3.7，"
        "故两码均**不加盖**（无盖子网格失配进入比较）。")
    add("")
    add("## ② 怎么做的")
    add("")
    add("1. `scripts/cross_solver_regression_ref.py`：在**同一份** TensorBEM 网格上跑 capytaine"
        f"（{meta['reference_metadata'].get('tool')} {meta['reference_metadata'].get('version')}），"
        "把数值、版本、平台、网格数组与 sha256 冻结进 `capytaine_ref.json`；测试期不需要安装 capytaine。")
    add("2. `tensorbem.benchmarks.cross_solver_regression`：重建同一网格 → 校验 sha256 → 跑 TensorBEM → "
        "逐点取相对差 `(tb-cp)/cp`。")
    add("3. 消去点规则（**先于运行固定**）：`|cp| < 5% * max_nu|cp|` 处相对差无数值意义，"
        "标 `excluded` 但数值全部留在工件里，可复核。")
    add(f"4. 验收判据：census 锚点原文 `max |relative| < {TOLERANCE:.0%}`，"
        "在每体最细网格、所有**纳入**点上取最大。判词是算出来的，不是断言的。")
    add("")
    add("## ③ 计算结果（数字表）")
    add("")
    add("### 3.1 每体最细网格的最坏相对差")
    add("")
    add("| 体 | 面元 | A33 最坏相对差 | @nu | B33 最坏相对差 | @nu | 纳入点 | 判定 |")
    add("|---|---|---|---|---|---|---|---|")
    for name, row in comp["per_case"].items():
        am = row["quantities"]["added_mass"]
        bd = row["quantities"]["damping"]
        def fmt(q):
            v = q["worst_relative"]
            return "—" if v is None else f"{v:+.2%}"
        add(f"| `{name}` | {row['finest_panels']} | {fmt(am)} | {am['worst_nu']} | "
            f"{fmt(bd)} | {bd['worst_nu']} | {am['included_points']}+{bd['included_points']} | "
            f"{'PASS' if row['pass'] else 'FAIL'} |")
    add("")
    w = comp["worst_overall_at"]
    add(f"全体最坏相对差：**{comp['worst_overall_relative']:+.2%}**（`{w['case']}` "
        f"{w['quantity']} @ nu={w['nu']}, {w['panels']} 面元）→ 锚点判词 **{comp['verdict']}**"
        f"（阈值 {comp['tolerance']:.0%}）。")
    add("")
    add("### 3.2 网格趋势（最坏纳入相对差 vs 面元数）")
    add("")
    add("| 体 | 面元 | A33 最坏 | B33 最坏 |")
    add("|---|---|---|---|")
    for row in conv:
        def f2(v):
            return "—" if v is None else f"{v:.2%}"
        add(f"| `{row['case']}` | {row['panels']} | {f2(row['worst']['added_mass'])} | "
            f"{f2(row['worst']['damping'])} |")
    add("")
    add("![cross-solver](fig_cross_solver_regression.png)")
    add("")
    add("## ④ 结果图")
    add("")
    add("`docs/fig_cross_solver_regression.png`：三体最细网格上 A33/B33 随频率的两码曲线（实线 TensorBEM，"
        "虚线 capytaine）。曲线重合度即上表相对差的图形形式。")
    add("")
    add("## ⑤ 注意点")
    add("")
    add("- 本回归测的是**互校带**，不是真值误差；两码同源错误（如都忽略某粘性效应）不可见。")
    add("- 比较在 heave 单自由度上做：加质量/阻尼的对角项对装配误差最敏感，但纵摇/横摇的力矩积分路径未覆盖。")
    add("- 尖角几何（barge）是两码差异的主要来源：角点附近面元自影响系数与近奇异积分在 Rankine 与 Green 函数"
        "两条路径上收敛速度不同；圆滑体（sphere/spheroid）给出的是纯公式化差异的下界。")
    add("- `excluded` 点（消去区）不是被「洗掉」的失败点：它们在工件里带原值，门槛 5% 峰值写在 meta 中。")
    add("")
    add("## ⑥ 误差判定")
    add("")
    verdict_rows = []
    for name, row in comp["per_case"].items():
        for quantity, label in (("added_mass", "A33"), ("damping", "B33")):
            q = row["quantities"][quantity]
            v = q["worst_relative"]
            verdict_rows.append(
                (label, name, v, None if v is None else abs(v) <= TOLERANCE)
            )
    add("| 量 | 体 | 最坏相对差 | 是否 <3% |")
    add("|---|---|---|---|")
    for label, name, v, ok in verdict_rows:
        add(f"| {label} | `{name}` | {'—' if v is None else f'{v:+.2%}'} | "
            f"{'是' if ok else ('否' if ok is not None else '—')} |")
    add("")
    add(f"结论：`worst |relative| = {comp['worst_overall_relative']:+.2%}`，"
        f"锚点 `max < {TOLERANCE:.0%}` 的判词为 **{comp['verdict']}**。"
        + (" 未达标项与其面元数一起列于上表，未被隐藏。" if comp["verdict"] == "FAIL" else ""))
    add("")
    return "\n".join(L) + "\n"


def write_artifacts(study: dict[str, Any]) -> dict[str, Path]:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    out = ARTIFACT_ROOT / RESULT_FILE
    out.write_text(json.dumps(study, indent=1, sort_keys=True) + "\n")
    doc = DOC_PATH
    doc.write_text(render_document(study))
    fig = render_figure(study)
    return {"results": out, "doc": doc, "figure": fig}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, help="write the machine form here")
    ap.add_argument("--write", action="store_true", help="write results.json + doc + figure")
    args = ap.parse_args(argv)
    study = run_study()
    if args.json:
        args.json.write_text(json.dumps(study, indent=1, sort_keys=True) + "\n")
    if args.write:
        paths = write_artifacts(study)
        for k, v in paths.items():
            print(f"wrote {k}: {v}")
    comp = study["comparison"]
    print(
        f"verdict={comp['verdict']} worst={comp['worst_overall_relative']:+.4%} "
        f"tolerance={comp['tolerance']:.0%}"
    )
    for f in comp["failures"]:
        print("  FAIL", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
