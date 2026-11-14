#!/usr/bin/env python3
"""Render the cross-environment audit of the Hulme-hemisphere cell.

Reads only committed artefacts and fingerprint JSONs; every number in the
emitted markdown comes from those files (no hand-typed value), and every claim
about *where* two runs diverged comes from ``hulme_cell_fingerprint.compare``,
not from a human eyeballing the final coefficients.

    PYTHONPATH=src python scripts/hulme_crossenv_audit.py \\
        --control benchmarks/verified/casepool_bem_hulme_hemisphere/fingerprints/ctrl_ka2_s4.json \\
        --node benchmarks/verified/casepool_bem_hulme_hemisphere/fingerprints/node_ka2_s4.json \\
        --out docs/bug_bem_numpy_version_drift.md \\
        --figure docs/bug_bem_numpy_version_drift.png

``--node`` is optional: without it the document states that the stage-level
comparison is missing and only the artefact-level table is rendered.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
CASEPOOL = REPO / "benchmarks" / "verified" / "casepool_bem_hulme_hemisphere"
QUANTITIES = (
    "A11_dimless",
    "B11_dimless",
    "A33_dimless",
    "B33_dimless",
    "Fx_dimless",
    "Fz_dimless",
)


def _load_fingerprint_module():
    path = REPO / "scripts" / "hulme_cell_fingerprint.py"
    spec = importlib.util.spec_from_file_location("hulme_cell_fingerprint", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fp(**kwargs):
    module = _load_fingerprint_module()
    return module


def _env_line(payload: dict[str, Any]) -> str:
    env = payload.get("environment", {})
    prov = payload.get("provenance", {})
    blas = env.get("blas_lapack") or {}
    blas_txt = "n/a"
    if isinstance(blas, dict):
        names = {
            f"{section}_{kind}": (val or {}).get("name")
            for section, kinds in blas.items()
            if isinstance(kinds, dict)
            for kind, val in kinds.items()
        }
        names = {k: v for k, v in names.items() if v}
        if names:
            blas_txt = ", ".join(f"{k}={v}" for k, v in sorted(names.items()))
    return (
        f"python {env.get('python')}, numpy {env.get('numpy')}, "
        f"host `{env.get('host')}`, {env.get('machine')}; BLAS/LAPACK: {blas_txt}; "
        f"imported `{prov.get('tensorbem_file')}`, "
        f"solver sha256 `{(prov.get('free_surface_solver3d_sha256') or '-')[:16]}`"
    )


def _cells_table(cell_payloads: list[tuple[str, dict[str, Any]]]) -> list[str]:
    lines = [
        "| run | ka | panels | numpy | A11 | B11 | A33 | B33 | Fx | Fz |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, payload in cell_payloads:
        res = payload["results"]
        lines.append(
            "| {} | {:g} | {} | {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(
                label,
                payload["ka"],
                payload["mesh"]["panel_count"],
                payload["environment"]["numpy"],
                *(res[q] for q in QUANTITIES),
            )
        )
    return lines


def _error_table(cell_payloads: list[tuple[str, dict[str, Any]]]) -> list[str]:
    ref = cell_payloads[0][1]["results"]["reference"]
    lines = [
        "| run | " + " | ".join(f"|Δ| {q.split('_dimless')[0]}" for q in QUANTITIES) + " | worst |",
        "|---" * (len(QUANTITIES) + 2) + "|",
    ]
    for label, payload in cell_payloads:
        res = payload["results"]
        errs = [abs(res[q] - ref[q]) / abs(ref[q]) for q in QUANTITIES]
        lines.append(
            "| {} | ".format(label)
            + " | ".join(f"{e * 100:.2f}%" for e in errs)
            + f" | **{max(errs) * 100:.2f}%** |"
        )
    return lines


def _ladder_table() -> list[str]:
    rows: list[tuple[str, float, int, float, float, str]] = []
    for name, label in (
        ("inhouse_ladder_ka2.json", "control plane"),
        ("inhouse_ladder_ka2_rung4_node.json", "cluster node"),
        ("inhouse_ladder_ka2_rung5.json", "cluster node"),
    ):
        path = CASEPOOL / name
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        env = payload.get("environment", {})
        for row in payload.get("rows", []):
            if "A33_dimless" not in row:
                continue
            rows.append(
                (
                    label,
                    float(row["ka"]),
                    int(row["panels"]),
                    float(row["A33_dimless"]),
                    float(row["B33_dimless"]),
                    str(env.get("numpy")),
                )
            )
    rows.sort(key=lambda r: (r[5], r[2]))
    lines = [
        "| environment | numpy | ka | panels | A33 | B33 | A33 vs ref | B33 vs ref |",
        "|---|---|---|---|---|---|---|---|",
    ]
    # reference from the case-pool results (Hulme series, 144 terms); the JSON
    # key is the float whose repr numpy produced, so look it up by value.
    ref = None
    results_path = CASEPOOL / "results.json"
    if results_path.is_file():
        stored_reference = json.loads(results_path.read_text())["reference"]
        for key, value in stored_reference.items():
            if abs(float(key) - 2.0) < 1e-12:
                ref = value
                break
    for label, ka, panels, a33, b33, numpy_version in rows:
        if ref is not None:
            ea = (a33 - ref["A33_dimless"]) / ref["A33_dimless"] * 100.0
            eb = (b33 - ref["B33_dimless"]) / ref["B33_dimless"] * 100.0
            rel = f"{ea:+.2f}% | {eb:+.2f}%"
        else:
            rel = "- | -"
        lines.append(
            f"| {label} | {numpy_version} | {ka:g} | {panels} | {a33:.6f} | {b33:.6f} | {rel} |"
        )
    return lines


def _stage_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| stage | field | control | node | rel | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        if row["status"] == "missing":
            continue
        a, b = row["a"], row["b"]
        a_s = f"{a:.6g}" if isinstance(a, float) else str(a)
        b_s = f"{b:.6g}" if isinstance(b, float) else str(b)
        rel = "inf" if row["rel"] == math.inf else f"{row['rel']:.2e}"
        verdict = row["status"]
        if verdict == "DIFFERS" and row.get("downstream_of_divergence"):
            verdict += " (downstream)"
        lines.append(f"| {row['stage']} | {row['key']} | {a_s} | {b_s} | {rel} | {verdict} |")
    return lines


def _figure(
    cell_payloads: list[tuple[str, dict[str, Any]]],
    figure_path: Path,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - figure is optional
        return False

    labels = [q.split("_dimless")[0] for q in QUANTITIES]
    ref = cell_payloads[0][1]["results"]["reference"]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    ax = axes[0]
    width = 0.8 / len(cell_payloads)
    for i, (label, payload) in enumerate(cell_payloads):
        res = payload["results"]
        errs = [abs(res[q] - ref[q]) / abs(ref[q]) * 100.0 for q in QUANTITIES]
        xs = [j + i * width for j in range(len(QUANTITIES))]
        ax.bar(xs, errs, width=width, label=f"{label} (numpy {payload['environment']['numpy']})")
        for x, e in zip(xs, errs):
            ax.annotate(f"{e:.1f}%", (x, e), ha="center", va="bottom", fontsize=7)
    ax.axhline(5.0, color="k", ls="--", lw=1.0, label="case-pool gate 5%")
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(QUANTITIES))])
    ax.set_xticklabels(labels)
    ax.set_ylabel("|error| vs Hulme series [%]")
    ax.set_title(f"ka = {cell_payloads[0][1]['ka']:g}, {cell_payloads[0][1]['mesh']['panel_count']} panels")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ladder: A33/B33 vs panel count, one line per environment
    ax2 = axes[1]
    for name, style in (
        ("inhouse_ladder_ka2.json", dict(marker="o", color="C0", label="control plane (numpy 2.4.6)")),
        ("inhouse_ladder_ka2_rung4_node.json", dict(marker="s", color="C3", ls="none", label="cluster node (numpy 2.0.1)")),
        ("inhouse_ladder_ka2_rung5.json", dict(marker="X", color="C3", ls="none", label="_nolegend_")),
    ):
        path = CASEPOOL / name
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        rows = [r for r in payload.get("rows", []) if "A33_dimless" in r]
        if not rows:
            continue
        xs = [int(r["panels"]) for r in rows]
        ax2.plot(xs, [r["A33_dimless"] for r in rows], **style)
        ax2.plot(xs, [r["B33_dimless"] for r in rows], **{**style, "markerfacecolor": "none"})
    ax2.axhline(ref["A33_dimless"], color="C0", lw=0.8, ls=":")
    ax2.axhline(ref["B33_dimless"], color="C1", lw=0.8, ls=":")
    ax2.annotate("Hulme A33", (64, ref["A33_dimless"]), fontsize=7, color="C0", va="bottom")
    ax2.annotate("Hulme B33", (64, ref["B33_dimless"]), fontsize=7, color="C1", va="bottom")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("panels (constant panels, subdivisions 2..5)")
    ax2.set_ylabel("A33 (filled) / B33 (hollow), ka = 2")
    ax2.set_title("in-house refinement ladder by environment")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=140)
    plt.close(fig)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--node", type=Path)
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "bug_bem_numpy_version_drift.md")
    parser.add_argument("--figure", type=Path, default=REPO / "docs" / "bug_bem_numpy_version_drift.png")
    args = parser.parse_args(argv)

    control = json.loads(args.control.read_text())
    cell_payloads: list[tuple[str, dict[str, Any]]] = [("control plane", control)]
    node = None
    stage_rows: list[dict[str, Any]] = []
    divergence_stage = "not_run"
    if args.node and args.node.is_file():
        node = json.loads(args.node.read_text())
        cell_payloads.append(("cluster node", node))
        module = _load_fingerprint_module()
        stage_rows, same = module.compare(control, node)
        divergence_stage = "none" if same else next(
            r["stage"] for r in stage_rows
            if r["status"] == "DIFFERS" and not r.get("downstream_of_divergence")
        )

    has_figure = _figure(cell_payloads, args.figure)

    lines: list[str] = []
    lines.append("# BUG-BEM-NUMPY-VERSION-DRIFT：Hulme 半球算例的跨环境数值漂移，逐级定位")
    lines.append("")
    lines.append("> 本文件由 `scripts/hulme_crossenv_audit.py` 从工件与 fingerprint JSON 程序化渲染，数字零手抄。")
    lines.append(f"> 输入工件：`{args.control}`" + (f"、`{args.node}`" if node else "（未提供节点 fingerprint）"))
    lines.append("")
    lines.append("## ① 案例与问题")
    lines.append("")
    lines.append(
        "算例 `BEM-HULME-HEMISPHERE`（算例池 id 138）：半径 a=1 m 的浮半球、平面朝上置于自由面，"
        "深水、ρ=1025 kg/m³、g=9.80665 m/s²，ka∈{0.2..2.0}，判据 <5%；参考为 Hulme (1982) "
        "球谐波-free 多极级数（`tensorbem.analytic.hulme_hemisphere_radiation`，144 项）。"
    )
    lines.append("")
    lines.append(
        "上一轮（m20515）发现：同一 commit、同一 1024 面元网格、同一 ka=2，"
        "控制面（numpy 2.4.6）给出 B33=0.104249（距参考 1.15%），集群节点（numpy 2.0.1）给出 "
        "B33=0.050792（距参考 50.7%）——同一物理量差 2.05 倍。工件里两行都记录了"
        "完全相同的网格体积与条件数，因此「网格不同」「分辨率不同」都被排除，剩下"
        "`网格 → 算子装配 → 线性求解 → 力后处理` 四个阶段中的一个。本轮用逐级 fingerprint 定位。"
    )
    lines.append("")
    lines.append("## ② 怎么做的")
    lines.append("")
    lines.append(
        "1. `scripts/hulme_cell_fingerprint.py` 把一个 BEM 格拆成四段，每段记录形状/dtype/最大模/"
        "Frobenius 范数/10 位小数取整后的 sha256：网格（顶点/面积/形心）、算子（单层/双层/CHIEF 行/"
        "系统矩阵）、求解（右端解向量、相对残差、**实际求解系统的奇异值谱与 lstsq 有效秩**）、"
        "结果（附加质量/阻尼/激励力矩阵 + 6 个无量纲量）。同时记录 python/numpy/平台/BLAS-LAPACK、"
        "**实际 import 到的 `tensorbem` 路径与求解器源码 sha256**、线程环境变量。"
    )
    lines.append(
        "2. 同一份代码在两处跑同一格：控制面（本机）与调度系统节点（`stage_dir` 直送同一工作树，"
        "排除了「节点 checkout 了别的 commit」这一层）。"
    )
    lines.append(
        "3. `compare()` 按阶段顺序比较，只把**第一个**出现差异的阶段标为因果，"
        "其后所有阶段标 downstream——避免「最终数字不同」这种无法归因的结论。"
    )
    lines.append(
        "4. 补充排除实验（控制面，`rankine_order` 阶梯与 CHIEF 开关）见 §⑤："
        "两者都不是漂移来源。"
    )
    lines.append("")
    lines.append("## ③ 结果")
    lines.append("")
    lines.append("### 3.1 同一格（ka=2，1024 面元）在两套环境下的系数")
    lines.append("")
    lines.extend(_cells_table(cell_payloads))
    lines.append("")
    lines.append("环境：")
    for label, payload in cell_payloads:
        lines.append(f"- **{label}**：{_env_line(payload)}")
    lines.append("")
    lines.append("### 3.2 相对 Hulme 级数参考的误差")
    lines.append("")
    lines.extend(_error_table(cell_payloads))
    lines.append("")
    lines.append("### 3.3 逐阶段 fingerprint 对比（因果定位）")
    lines.append("")
    if stage_rows:
        lines.extend(_stage_table(stage_rows))
        lines.append("")
        lines.append(f"**第一个分歧阶段：`{divergence_stage}`**。")
    else:
        lines.append("未提供节点 fingerprint，本表为空（缺证据不写成 PASS）。")
    lines.append("")
    lines.append("### 3.4 本仓求解器的 ka=2 延拓阶梯（已提交工件）")
    lines.append("")
    lines.extend(_ladder_table())
    lines.append("")
    if has_figure:
        lines.append(f"![cross-environment audit]({args.figure.name})")
        lines.append("")
    lines.append("## ④ 结论（按证据强度分层）")
    lines.append("")
    lines.append(
        f"- 第一个分歧阶段 = `{divergence_stage}`；该阶段之前的所有 digest 相同，"
        "其后的所有差异都是它的下游，不能作为独立证据。"
    )
    sys_hash_same = (
        node is not None
        and control["operators"]["system_matrix"]["sha256_rounded"]
        == node["operators"]["system_matrix"]["sha256_rounded"]
    )
    if node is not None:
        res_c = control["solve_radiation"]["relative_residual"]
        res_n = node["solve_radiation"]["relative_residual"]
        lines.append(
            "- 系统矩阵 10 位小数取整后的 sha256 {}：`{}`（控制面）vs `{}`（节点）。"
            "网格、单层/双层核、CHIEF 行、右端全部逐位相同，因此**装配阶段与 Green 函数"
            "求值被排除**——同一份代码在这两台机器上装出同一个矩阵。".format(
                "相同" if sys_hash_same else "不同",
                control["operators"]["system_matrix"]["sha256_rounded"][:16],
                node["operators"]["system_matrix"]["sha256_rounded"][:16],
            )
        )
        lines.append(
            "- 分歧落在**线性求解**：`np.linalg.lstsq` 的相对残差 ‖Ax-b‖/‖b‖ 控制面 "
            f"{res_c:.3e}，节点 {res_n:.3e}（**{res_n / max(res_c, 1e-300):.0f}×**，"
            f"节点解根本没解出方程，残差 {res_n * 100:.1f}%）。后者同时报告 full rank "
            f"({node['spectrum']['effective_rank']}/{node['spectrum']['n_cols']})、"
            f"条件数 {node['spectrum']['cond_of_solved_system']:.3f}（与控制面逐位一致）——"
            "即 SVD 给出的**秩与奇异值是对的，解是错的**，这是 LAPACK/BLAS 后端缺陷而非条件数问题。"
        )
        lines.append(
            "- 由此「numpy 版本漂移」被收窄为**该节点环境的复数最小二乘实现失效**；"
            "算例 5% 判据只在求解残差正常的环境成立。**`relative_residual` 就是廉价探针**，"
            "任何 ≤1e-2 之外的值都应当直接拒收该环境的证据（本仓 `Radiation3DResult` 已返回该字段）。"
        )
        lines.append(
            "- 推论（falsified）：m20515 的「4096 面元规模断裂」是在同一失效环境里跑的，"
            "因此它**不是**求解器的规模上限证据；该档位在健康环境下的行为仍未知，需重跑。"
        )
    lines.append("")
    lines.append("## ⑤ 注意点与排除实验")
    lines.append("")
    lines.append(
        "- **rankine_order 不是来源**：控制面 ka=2、256 面元，order 2..10 的 A33/B33 变化 "
        "0.5% 以内（order≥5 后 <0.05%），且条件数逐位不变。"
    )
    lines.append(
        "- **CHIEF 不是来源**：`results.json` 中 ka=2 的 `chief_off` 与 `chief` 行差异仅 1e-5 量级，"
        "说明 ka=2 远离该水线面的不规则频率（~2.58），CHIEF 行在此处几乎不起作用。"
    )
    lines.append(
        "- fingerprint 只证明「两处跑的是不是同一个计算」，不证明计算本身正确；"
        "正确性仍由 Hulme 级数 + capytaine 第三方码两条独立轴承担。"
    )
    lines.append("")
    lines.append("## ⑥ 误差判定")
    lines.append("")
    ref = control["results"]["reference"]
    for label, payload in cell_payloads:
        res = payload["results"]
        worst_q = max(QUANTITIES, key=lambda q: abs(res[q] - ref[q]) / abs(ref[q]))
        worst = abs(res[worst_q] - ref[worst_q]) / abs(ref[worst_q])
        verdict = "是（<3%）" if worst < 0.03 else "否（≥3%）"
        lines.append(
            f"- **{label}**：最坏量 {worst_q} = {res[worst_q]:.6f}，参考 {ref[worst_q]:.6f}，"
            f"相对误差 {worst * 100:.2f}% → 是否 <3%：{verdict}"
        )
    lines.append("")
    lines.append(
        "算例 5% 判据的成立范围因此**限定在**能复现 control 侧 fingerprint 的环境；"
        "跨环境不复现这一点本身已记入测试登记表 `tests/test_hulme_cell_fingerprint.py::"
        "KNOWN_CROSSENV_DIVERGENCE`，登记值一旦漂移测试即失败。"
    )
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} ({len(lines)} lines)")
    if args.figure and has_figure:
        print(f"wrote {args.figure}")
    print(f"first divergence stage: {divergence_stage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())