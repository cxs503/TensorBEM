#!/usr/bin/env python3
"""Render ``docs/finite_depth_capytaine_xval.md`` from the measured JSON.

Why this exists
---------------
The report for the finite-depth TensorBEM/capytaine cross-validation shipped as
a *skeleton*: every result sat behind an ``<!-- 注入 -->`` comment that no script
ever filled, so the document announced an error table it did not contain while
the measured data was already frozen in ``xval_results.json``.  A report whose
numbers are typed by hand also cannot be kept honest -- when the cache is
re-assembled the prose silently goes stale.

This renderer is the only writer of the document.  It reads
``xval_results.json`` and emits every table in it, including the verdicts, so
the document states exactly what was measured:

* the geometry/parameter block and the mesh ladders (from the cache, not prose);
* the pre-registered gate thresholds *and* the measured per-gate verdicts;
* the full station table (every TensorBEM/capytaine pair that exists);
* the coverage ledger (measured / missing stations) and the blocking defects;
* an explicit, per-gate ``<3% / <15%`` answer.

The verdict is not softened: the study is ``partial``, gates G1/G2/G4/G5 fail,
and the finite-depth capability stays ``WITHHELD``.  ``tests/
test_bem_finite_depth_capytaine_xval.py::TestDocIntegrity`` re-renders this file
and asserts the committed document matches it byte for byte, so a stale or
hand-edited report fails CI.

Usage
-----
    PYTHONPATH=src python benchmarks/verified/bem_finite_depth_capytaine_xval/make_doc.py
    PYTHONPATH=src python benchmarks/verified/bem_finite_depth_capytaine_xval/make_doc.py --check
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
RESULT_FILE = HERE / "xval_results.json"
MODE_REQ_FILE = HERE / "mode_requirement.json"
DOC_FILE = REPO_ROOT / "docs" / "finite_depth_capytaine_xval.md"

#: ``NOT <3%`` on purpose: the study's own gates are looser (10%/15%), and the
#: ecosystem-wide acceptance bar is 15% for a *validated* capability.  Both
#: verdicts are printed per gate so neither is ever implied.
STRICT_TOL = 0.03

#: The pre-registered study grid, copied verbatim from ``run_cases.py``
#: (``DEPTH_RATIOS`` + the ``inf`` leg) x ``KA_GRID``.  The cache only contains
#: the stations that were actually solved; the plan is what coverage is judged
#: against, so it is stated here rather than inferred from the cache.
PLANNED_DEPTH_RATIOS: tuple[object, ...] = (4.0, 5.0, 8.0, 12.0, 20.0, "inf")


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100.0 * x:.2f}%"


def _num(x: float | None, digits: int = 2) -> str:
    return "—" if x is None else f"{x:,.{digits}f}"


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and v == v and abs(v) != float("inf")


def _depth_label(v: Any) -> str:
    if isinstance(v, str):
        return v
    return f"{float(v):g}"


#: Tail target ``mode_count_for`` aims at (``_MODE_DECAY`` in the solver module).
_MODE_DECAY_NOTE = "1e-12"


def mode_requirement_rows() -> list[dict[str, Any]]:
    """Rows of ``mode_requirement.json`` (frozen) -- empty if it is absent."""
    if not MODE_REQ_FILE.exists():
        return []
    return json.loads(MODE_REQ_FILE.read_text())["rows"]


def write_mode_requirement() -> int:
    """Regenerate ``mode_requirement.json`` from the solver (needs ``src`` on path)."""
    from tensorbem.finite_depth_solver3d import mode_count_for
    from tensorbem.geometry3d import submerged_sphere_mesh

    data = json.loads(RESULT_FILE.read_text())
    pinned = sorted({int(r["mode_count"]) for r in data["tbem_study"]["rows"]})
    pinned_modes = pinned[0]
    levels = sorted({int(s) for s in data["parameters"]["tbem_panels"]})
    depths = sorted({float(r["depth_ratio"]) for r in data["tbem_study"]["rows"]})
    rows = []
    for sub in levels:
        mesh = submerged_sphere_mesh(sub, radius=1.0, submergence=2.0)
        for depth_ratio in depths:
            rows.append(
                {
                    "panel_level": f"icosphere sub={sub}",
                    "subdivisions": sub,
                    "panels": int(mesh.panel_count),
                    "depth_ratio": depth_ratio,
                    "required_modes": int(mode_count_for(mesh, depth_ratio)),
                    "pinned_modes": pinned_modes,
                }
            )
    MODE_REQ_FILE.write_text(
        json.dumps(
            {
                "how": "make_doc.py --write-mode-requirement",
                "note": (
                    "mode_count_for() = evanescent modes needed for an exp(-1e-12*mu*r) tail "
                    "at the nearest target/quadrature separation; the cache pinned every "
                    "TensorBEM case to the same mode_count, which is the recorded defect."
                ),
                "pinned_modes_in_cache": pinned,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"wrote {MODE_REQ_FILE} ({len(rows)} rows)")
    return 0


def _gate_rows(data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    order = [
        "G1_tbem_cpt_am_heave",
        "G2_tbem_cpt_b_heave",
        "G3_cpt_self_conv_am",
        "G4_tbem_self_conv_am",
        "G5_dw_limit_am",
    ]
    gates = data["comparison"]["gates"]
    return [(name, gates[name]) for name in order if name in gates]


def _gate_verdict(gate: dict[str, Any]) -> tuple[str, str]:
    """``(study-gate verdict, strict-3% verdict)`` for one gate."""
    if gate.get("n", 0) == 0:
        return "未评估", "未评估"
    err = gate["max_error"]
    study = "PASS" if gate.get("passed") else "FAIL"
    strict = "是" if err < STRICT_TOL else "否"
    return f"{study}（{_pct(err)} vs 阈值 {_pct(gate['threshold'])}）", strict


def _comparison_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [r for r in data["comparison"]["comparison"] if "am_rel_error" in r]
    rows.sort(key=lambda r: (float(r["depth_ratio"]), float(r["ka"])))
    return rows


def render_doc(data: dict[str, Any]) -> str:
    """Return the full markdown report for ``data`` (deterministic)."""
    geom = data["geometry"]
    par = data["parameters"]
    status = data["study_status"]
    analytic = data["analytic_infinite_depth"]
    rows = _comparison_rows(data)
    gates = dict(_gate_rows(data))

    tbem_rows = data["tbem_study"]["rows"]
    n_tbem = int(data["tbem_study"]["n"])
    n_cpt = int(data["cpt_study"]["n"])
    ka_grid = par["ka_grid"]
    # The plan plus any station the cache holds that the plan does not (h/R = 3,
    # the degenerate zero-clearance case, was solved as an extra datapoint).
    measured_depths = sorted(
        {float(r["depth_ratio"]) for r in data["comparison"]["comparison"]
         if _finite(r["depth_ratio"])}
    )
    ladder = sorted(
        {*PLANNED_DEPTH_RATIOS, *measured_depths},
        key=lambda v: (not _finite(v), float(v) if _finite(v) else 0.0),
    )
    n_planned = len([d for d in PLANNED_DEPTH_RATIOS]) * len(ka_grid)
    passed = sum(1 for g in gates.values() if g.get("passed"))
    strict_all = all(
        (g["max_error"] < STRICT_TOL)
        for g in gates.values()
        if g.get("n", 0) > 0
    ) and any(g.get("n", 0) > 0 for g in gates.values())

    L: list[str] = []
    add = L.append

    add("# TensorBEM 有限水深 capytaine 交叉验证（XVAL-BEM-FINITE-DEPTH-CROSSVAL）")
    add("")
    add(f"> **判定：{status['state'].upper()} — 证据等级 WITHHELD。** 本文档由")
    add("> `benchmarks/verified/bem_finite_depth_capytaine_xval/make_doc.py` 从")
    add("> `xval_results.json` 程序化渲染，零手抄；`tests/test_bem_finite_depth_capytaine_xval.py::TestDocIntegrity`")
    add("> 断言「文档 == 渲染结果」，陈旧或手改的报告会直接判红。")
    add(">")
    add(f"> 预注册站位 {n_planned} 个，其中 TensorBEM 与 capytaine 双侧都有实测的只有 **{len(rows)}** 个；")
    add("> **不得**据本文宣称「全网格误差 <15%」。")
    add("")
    add("**类型**：第三方代码交叉验证（code-to-code）· **算例**：全潜没刚性球垂荡/横荡/纵荡辐射")
    add("")
    add("**工件**")
    add("- `benchmarks/verified/bem_finite_depth_capytaine_xval/run_cases.py`（逐案例求解 + 缓存）")
    add("- `benchmarks/verified/bem_finite_depth_capytaine_xval/assemble_xval.py`（缓存 → `xval_results.json`）")
    add("- `benchmarks/verified/bem_finite_depth_capytaine_xval/xval_results.json`（实测数据，唯一事实源）")
    add("- `benchmarks/verified/bem_finite_depth_capytaine_xval/make_figures.py`（结果图）")
    add("- `benchmarks/verified/bem_finite_depth_capytaine_xval/make_doc.py`（本报告渲染器）")
    add("- `tests/test_bem_finite_depth_capytaine_xval.py`（门禁锁 + 文档新鲜度锁）")
    add("")
    add("**复现**：`PYTHONPATH=src python benchmarks/verified/bem_finite_depth_capytaine_xval/run_cases.py --which all`")
    add("→ `assemble_xval.py` → `make_doc.py` → `make_figures.py`。")
    add("")

    # ------------------------------------------------------------------ 1
    add("## 1. 案例介绍")
    add("")
    add("### 1.1 物理背景")
    add("有限水深改变自由面 Green 函数的本征结构：`kh` 不大时，海底边界条件")
    add("`∂G/∂z = 0`（`z = -h`）引入的衰减模态显著改变附加质量与阻尼。TensorBEM 的")
    add("`finite_depth_solver3d` 用 John 本征级数（传播根 + 衰减根 + 模态范数），")
    add("capytaine（Nemoh 后继者）用独立的 Rankine 面板 + 预计算表实现；两者数学等价、")
    add("实现独立，因此 code-to-code 交叉验证是有限水深能力的外部证据源。")
    add("")
    add("### 1.2 几何与流体参数")
    add("")
    add("| 参数 | 值 |")
    add("|---|---|")
    add(f"| 几何 | 全潜没球（`{geom['type']}`） |")
    add(f"| 半径 R | {geom['radius_m']:.3f} m |")
    add(f"| 球心 z | {geom['centre_z_m']:.3f} m |")
    add(f"| 球底 z | {geom['bottom_z_m']:.3f} m |")
    add(f"| 密度 ρ | {par['density']:g} kg/m³ |")
    add(f"| 重力 g | {par['gravity']:g} m/s² |")
    vol = next(iter(analytic.values()))["displaced_volume"]
    add(f"| 排水体积 ∇ | {vol:.6f} m³ |")
    add("")
    add(f"几何先行校验：球体积 4πR³/3 = {vol:.6f} m³，与缓存一致；球底 z 与海底间隙见 1.3，"
        f"`h/R = 3` 站位的间隙为 {geom['seabed_clearance_m']['3.0']:g} m，属退化布置。")
    add("")
    add("### 1.3 预注册水深梯级与频率网格")
    add("")
    add("| h/R | 水深 h (m) | 海底间隙 (m) | 双侧已测 ka 数 | 备注 |")
    add("|---:|---:|---:|---:|---|")
    for dr in ladder:
        label = _depth_label(dr)
        if not _finite(dr):
            add(f"| {label} | ∞ | ∞ | 0/{len(ka_grid)} | 计划内未跑（需走无限深求解器，接口不同） |")
            continue
        # clearance = depth + bottom z; the frozen dict is cross-checked, not trusted
        clearance = float(dr) + geom["bottom_z_m"]
        frozen = geom["seabed_clearance_m"].get(f"{float(dr):g}")
        if frozen is not None and abs(frozen - clearance) > 1e-9:
            raise ValueError(f"seabed clearance mismatch at h/R={dr}: {frozen} vs {clearance}")
        n_here = len({r["ka"] for r in rows if float(r["depth_ratio"]) == float(dr)})
        measured_any = any(v["depth_ratio"] == float(dr) for v in tbem_rows)
        cells = "无布站" if not measured_any else f"{clearance:.2f}"
        if float(dr) not in PLANNED_DEPTH_RATIOS:
            note = "计划外：球底触底（退化 BVP），仅作极限参考"
        elif n_here == 0:
            note = "计划内未跑"
        elif n_here < len(ka_grid):
            note = "计划内仅部分 ka"
        else:
            note = "计划内已测"
        add(f"| {label} | {float(dr):.1f} | {cells} | {n_here}/{len(ka_grid)} | {note} |")
    add("")
    add(f"频率网格 `ka = {', '.join(f'{k:g}' for k in ka_grid)}`（`nu = ka/R`，R = 1 m）。")
    add("")
    add("### 1.4 网格阶梯")
    add("")
    add("| 工具 | 级别 | 面板数 |")
    add("|---|---|---:|")
    for sub, n in par["tbem_panels"].items():
        add(f"| TensorBEM | icosphere sub={sub} | {n} |")
    for res, n in par["cpt_faces"].items():
        add(f"| capytaine | ({res.replace('x', ', ')}) | {n} |")
    add("")
    add(f"实际参跑的 TensorBEM 案例 {n_tbem} 个、capytaine 案例 {n_cpt} 个。"
        f"TensorBEM 侧：{data['tbem_study']['note']}。")
    add("")

    # ------------------------------------------------------------------ 2
    add("## 2. 怎么做的")
    add("")
    add("1. **TensorBEM 侧**：`solve_radiation_finite_depth_3d` 解垂荡/横荡/纵荡辐射，"
        "记录 `A33/B33`、条件数、相对残差、模态数、面板数。")
    add("2. **capytaine 侧**：`cpt.BEMSolver().solve(cpt.RadiationProblem(..., water_depth=h))`，"
        "四级网格 (8,16)→(24,48)。")
    add("3. **配对规则**：同一 `(h/R, ka)` 上取两侧最细网格（TensorBEM sub=2/sub=3、"
        "capytaine 24×48）做相对误差；capytaine 自收敛用最细 vs 次细。")
    add("4. **门禁预注册**：阈值写在 `assemble_xval.py` 里，运行后不放宽。")
    add("")
    add("| 门禁 | 描述 | 阈值 |")
    add("|---|---|---:|")
    desc = {
        "G1_tbem_cpt_am_heave": "TensorBEM vs capytaine 附加质量相对误差",
        "G2_tbem_cpt_b_heave": "TensorBEM vs capytaine 阻尼相对误差",
        "G3_cpt_self_conv_am": "capytaine 自收敛（最细 vs 次细）",
        "G4_tbem_self_conv_am": "TensorBEM 自收敛（sub=3 vs sub=2）",
        "G5_dw_limit_am": "深水极限（h/R=20）附加质量误差",
    }
    for name, gate in _gate_rows(data):
        add(f"| {name.split('_')[0]} | {desc.get(name, name)} | {_pct(gate['threshold'])} |")
    add("")
    add("**工具链**：TensorBEM `finite_depth_solver3d`（John 本征级数，模态数默认 48）· "
        "capytaine（独立 BEM）· Python + NumPy。")
    add("")

    # ------------------------------------------------------------------ 3
    add("## 3. 计算结果")
    add("")
    add("### 3.1 门禁实测状态")
    add("")
    add("| 门禁 | 实测最大误差 | 样本数 | 阈值 | 结论 |")
    add("|---|---:|---:|---:|---|")
    for name, gate in _gate_rows(data):
        verdict = "PASS" if gate.get("passed") else ("未评估" if gate.get("n", 0) == 0 else "FAIL")
        add(f"| {name.split('_')[0]} | {_pct(gate['max_error']) if gate.get('n', 0) else '—'} "
            f"| {gate.get('n', 0)} | {_pct(gate['threshold'])} | {verdict} |")
    add("")
    add(f"{len(gates)} 个门禁中通过 {passed} 个。`G4` 无配对样本（sub=3 只在 ka=1 上跑过）。")
    add("")
    add("### 3.2 站位表（全部双侧实测配对）")
    add("")
    add("| h/R | ka | TensorBEM A33 (kg) | capytaine A33 (kg) | ΔA33 | TensorBEM B33 | capytaine B33 | ΔB33 | 面板 | 模态数 | 条件数 |")
    add("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    cond_by_case = {r["case_id"]: r for r in tbem_rows}
    for r in rows:
        dr = float(r["depth_ratio"])
        cand = [
            v for k, v in cond_by_case.items()
            if v["depth_ratio"] == dr and abs(v["ka"] - r["ka"]) < 1e-12
        ]
        cond = f"{cand[0]['condition_number']:,.0f}" if cand else "—"
        add(f"| {dr:g} | {r['ka']:g} | {_num(r['tbem_am_heave'])} | {_num(r['cpt_am_heave'])} "
            f"| {_pct(r['am_rel_error'])} | {_num(r['tbem_b_heave'])} | {_num(r['cpt_b_heave'])} "
            f"| {_pct(r['b_rel_error'])} | {r['tbem_n_panels']} | {r['tbem_mode_count']} | {cond} |")
    add("")
    am_err = [r["am_rel_error"] for r in rows]
    b_err = [r["b_rel_error"] for r in rows]
    add(f"ΔA33 区间 {_pct(min(am_err))} – {_pct(max(am_err))}；"
        f"ΔB33 区间 {_pct(min(b_err))} – {_pct(max(b_err))}。")
    add("")
    add("### 3.3 解析锚（无限深多极解）")
    add("")
    add("| ka | 解析 A33 (kg) | A33/ρ∇ | capytaine ∞ (kg) | 解析 vs capytaine |")
    add("|---:|---:|---:|---:|---:|")
    for ka in ka_grid:
        a = analytic.get(f"{ka:g}") or analytic.get(str(ka))
        if a is None:
            continue
        ratio = a["added_mass_heave"] / (a["displaced_volume"] * par["density"])
        add(f"| {ka:g} | {_num(a['added_mass_heave'])} | {ratio:.4f} | "
            f"{_num(a['cpt_inf_am_heave']) if a['cpt_inf_am_heave'] is not None else '—'} | "
            f"{_pct(a['am_rel_error_dimful_vs_cpt_inf']) if a['am_rel_error_dimful_vs_cpt_inf'] is not None else '—'} |")
    add("")
    a03 = analytic["0.3"]
    add(f"`ka→0` 时 A33 趋于 Lamb 的 0.5ρ∇：`ka=0.3` 处 A33/ρ∇ = "
        f"{a03['added_mass_heave'] / (a03['displaced_volume'] * par['density']):.4f}，"
        "闭合解与 capytaine 的两条独立路径相互印证，故这张表可以做锚。")
    add("")
    add("### 3.4 覆盖账本")
    add("")
    add(f"- 已测站位：{len(rows)} / {n_planned}（预注册）。")
    for m in status["missing"]:
        add(f"- **缺**：{m}")
    add("")
    add("### 3.5 模态需求 vs 实际钉住的模态数")
    add("")
    req_rows = mode_requirement_rows()
    if req_rows:
        add("`mode_count_for()` 给出「按最近目标/求积点分离把衰减模态尾部压到 1e-12」所需的模态数，"
            "与缓存里实际使用的 `mode_count` 对比：")
        add("")
        add("| 网格 | 面板数 | h/R | 需要的模态数 | 实际钉住 | 低估倍数 |")
        add("|---|---:|---:|---:|---:|---:|")
        for r in req_rows:
            add(f"| {r['panel_level']} | {r['panels']} | {r['depth_ratio']:g} | {r['required_modes']} "
                f"| {r['pinned_modes']} | {r['required_modes'] / max(r['pinned_modes'], 1):.0f}× |")
        add("")
        cap = max(r["required_modes"] for r in req_rows)
        n_at_cap = sum(1 for r in req_rows if r["required_modes"] == cap)
        if n_at_cap > 1:
            add(f"注意：{n_at_cap}/{len(req_rows)} 行同时停在 {cap}，即求解器的模态**上限**"
                "（`_MODES_CAP`）——这些网格/水深组合已经超出默认本征核的适用范围，"
                "把 `mode_count` 调大也到不了需要的尾部精度，必须换核（`kernel='quadrature'`）"
                "或加解析尾部修正。")
            add("")
        add("这张表是 `mode_requirement.json` 的渲染结果；`tests/test_bem_finite_depth_capytaine_xval.py` "
            "会用求解器现场复算并断言该 JSON 没有过期。")
        add("")
    else:
        add("（缺 `mode_requirement.json`：运行 `make_doc.py --write-mode-requirement` 生成）")
        add("")
    add("### 3.6 阻塞缺陷")
    add("")
    for i, d in enumerate(status["blocking_defects"], 1):
        add(f"{i}. {d}")
    add("")

    # ------------------------------------------------------------------ 4
    add("## 4. 结果图")
    add("")
    figs = [
        ("xval_fd_am_heave_vs_ka.png", "附加质量 A33(ka)：各水深下 TensorBEM（实线/圆点）vs capytaine（虚线/方块）"),
        ("xval_fd_b_heave_vs_ka.png", "阻尼 B33(ka)：同上配对"),
        ("xval_fd_error_heatmap.png", "(h/R, ka) 上的 A33 相对误差热图，直接给出偏置的符号翻转"),
        ("xval_fd_convergence.png", "capytaine 网格自收敛与 TensorBEM 可用阶梯"),
    ]
    have = [f for f in figs if (REPO_ROOT / "docs" / "figures" / f[0]).exists()]
    if len(have) == len(figs):
        for name, cap in figs:
            add(f"![{cap}](figures/{name})")
            add("")
            add(f"*{cap}*")
            add("")
    else:
        add("图由 `make_figures.py` 生成到 `docs/figures/xval_fd_*.png`：")
        add("")
        for name, cap in figs:
            add(f"- `{name}` — {cap}")
        add("")
    add("")

    # ------------------------------------------------------------------ 5
    add("## 5. 注意点（坑与局限）")
    add("")
    add("- **球底接触海底**：`h/R = 3` 时球底（z=-3）恰好落在海底，海底间隙 0，属退化布置；"
        "这一行不是物理基准，只作极限参考。")
    req = mode_requirement_rows()
    if req:
        worst = max(req, key=lambda r: r["required_modes"] / max(r["pinned_modes"], 1))
        add(f"- **模态截断是主缺陷，不是网格问题**：`mode_count` 全部钉在 "
            f"{worst['pinned_modes']}，而同一网格上 `mode_count_for()` 为达到 "
            f"`exp(-{_MODE_DECAY_NOTE})` 尾部要求的是 "
            f"**{worst['required_modes']}** 个衰减模态（{worst['panel_level']}，"
            f"h/R={worst['depth_ratio']:g}，即被低估 "
            f"{worst['required_modes'] / max(worst['pinned_modes'], 1):.0f} 倍）；"
            "级数被硬截断后附加质量对模态数极不收敛（见 3.5 的截断研究）。")
    conds: dict[float, list[float]] = {}
    for r in tbem_rows:
        if r["depth_ratio"] != "inf":
            conds.setdefault(float(r["depth_ratio"]), []).append(float(r["condition_number"]))
    if conds:
        worst_depth = max(conds, key=lambda d: max(conds[d]))
        others = sorted(d for d in conds if d != worst_depth)
        if others:
            add(f"- **条件数异常**：h/R={worst_depth:g} 的离散系统条件数 "
                f"{min(conds[worst_depth]):,.0f}–{max(conds[worst_depth]):,.0f}，"
                f"其余已测水深只有 "
                f"{min(min(conds[d]) for d in others):,.0f}–"
                f"{max(max(conds[d]) for d in others):,.0f}；"
                "A33 偏置换号恰好发生在条件数高一到两个量级的水深上，"
                "指向截断核的伪本征模态（truncated Green function 不再严格满足自由面条件），"
                "而不是单纯的网格面积误差。")
    sign_rows = [r for r in rows if r["tbem_am_heave"] > r["cpt_am_heave"]]
    low_rows = [r for r in rows if r["tbem_am_heave"] < r["cpt_am_heave"]]
    if sign_rows and low_rows:
        hi_depths = sorted({r["depth_ratio"] for r in sign_rows})
        lo_depths = sorted({r["depth_ratio"] for r in low_rows})
        add(f"- **误差随水深非单调**：TensorBEM 在 h/R = "
            f"{', '.join(f'{float(d):g}' for d in lo_depths)} 上偏低 "
            f"{_pct(min(r['am_rel_error'] for r in low_rows))}–"
            f"{_pct(max(r['am_rel_error'] for r in low_rows))}，在 h/R = "
            f"{', '.join(f'{float(d):g}' for d in hi_depths)} 上偏高 "
            f"{_pct(min(r['am_rel_error'] for r in sign_rows))}–"
            f"{_pct(max(r['am_rel_error'] for r in sign_rows))}；"
            "纯面积/离散缺陷不可能换号，所以这是与水深耦合的截断误差。")
    add("- **未覆盖**：非零航速核、多体相互作用、不规则频率抑制（CHIEF）、"
        "有航速/水面穿透船体的有限水深辐射。")
    add("- **capytaine 版本**：缓存里记为 `unknown`，因此本文只做 code-to-code 相对比较，"
        "不做绝对认证。")
    add("")

    # ------------------------------------------------------------------ 6
    add("## 6. 误差判定")
    add("")
    add("| 关键量 | 实测 | 本文阈值 | 是否 <15%（验收） | 是否 <3%（严格） |")
    add("|---|---|---:|---|---|")
    for name, gate in _gate_rows(data):
        if gate.get("n", 0) == 0:
            add(f"| {name.split('_')[0]} | 未评估 | {_pct(gate['threshold'])} | 无样本 | 无样本 |")
            continue
        err = gate["max_error"]
        add(f"| {name.split('_')[0]} | {_pct(err)} | {_pct(gate['threshold'])} "
            f"| {'是' if err < 0.15 else '**否**'} | {'是' if err < STRICT_TOL else '**否**'} |")
    add("")
    worst_am = gates["G1_tbem_cpt_am_heave"]["max_error"]
    worst_b = gates["G2_tbem_cpt_b_heave"]["max_error"]
    add("**逐项回答**")
    add("")
    add(f"- 附加质量 A33：最大 {_pct(worst_am)}，**<15% 成立、<3% 不成立**（超 3% 门槛 "
        f"{worst_am / STRICT_TOL:.1f} 倍），G1 判 FAIL。")
    add(f"- 阻尼 B33：最大 {_pct(worst_b)}，**<15% 不成立**，G2 判 FAIL。")
    add(f"- capytaine 自收敛：{_pct(gates['G3_cpt_self_conv_am']['max_error'])}，"
        f"**<3% 成立** → 基准侧可信，误差不来自 capytaine。")
    add("- TensorBEM 自收敛：**无配对样本**（sub=3 只在 ka=1 跑过），无法判定，G4 记 FAIL。")
    add(f"- 深水极限：{_pct(gates['G5_dw_limit_am']['max_error'])}，**<15% 成立、<3% 不成立**。")
    add("")
    add(f"**总结论**：关键量未同时满足 <3%（严格）与全部 <15%（验收），"
        f"有限水深能力维持 **WITHHELD**；结论为 honest-FAIL，不粉饰。")
    add("")

    # ------------------------------------------------------------------ 7
    add("## 7. 测试")
    add("")
    add("```bash")
    add("PYTHONPATH=src python -m pytest tests/test_bem_finite_depth_capytaine_xval.py -v")
    add("```")
    add("")
    add("测试锁四件事：(1) 失败门禁的数值被钉住，不许「顺手修绿」；"
        "(2) 偏置的符号结构与水深相关性被钉住；"
        "(3) 本文档必须等于 `make_doc.py` 的渲染结果（陈旧/手改即判红）；"
        "(4) `mode_requirement.json` 必须能由求解器现场复算出来（模态需求不许写死）。")
    add("")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the committed document differs from the render")
    ap.add_argument("--write-mode-requirement", action="store_true",
                    help="regenerate mode_requirement.json from the solver, then render")
    args = ap.parse_args(argv)

    if args.write_mode_requirement:
        write_mode_requirement()

    data = json.loads(RESULT_FILE.read_text())
    text = render_doc(data)
    if args.check:
        current = DOC_FILE.read_text() if DOC_FILE.exists() else ""
        if current != text:
            print(f"{DOC_FILE} is stale; re-run make_doc.py", file=sys.stderr)
            return 1
        print(f"{DOC_FILE} is up to date")
        return 0
    DOC_FILE.write_text(text)
    print(f"wrote {DOC_FILE} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())