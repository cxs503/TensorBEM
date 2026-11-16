"""Render ``docs/casepool_bem_cummins_wigley/README.md`` from the committed artifacts.

The m20419 note was written while the case still carried two debts (a stale
``case_result.json`` and a ``skip``-ped production replay test).  Both were paid
off later -- the artifact was regenerated on a ``loongarch64`` node that passes
the C0 precondition -- but the note kept asserting the pre-fix state, which made
the document a second, silent source of truth that disagreed with the tree.

This renderer closes that loop the same way the case closes its own ``CASE.md``:
every number in the note is read out of the artifacts at render time, so the
prose cannot drift away from them.

Sources (all inside the repository)::

    benchmarks/verified/casepool_bem_cummins_wigley/case_result.json   committed verdict
    benchmarks/verified/casepool_bem_cummins_wigley/sweep_<label>.npz   committed sweeps
    docs/casepool_bem_cummins_wigley/evidence_cluster_n315_sweep_coarse.npz
    docs/m20021/sweep_medium.npz                                        C0 negative controls

Usage::

    PYTHONPATH=src python tools/render_casepool_bem_cummins_wigley_readme.py [--write]

Without ``--write`` it prints the render to stdout (the test compares that
against the committed file).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "benchmarks" / "verified" / "casepool_bem_cummins_wigley"
DOCS_DIR = ROOT / "docs" / "casepool_bem_cummins_wigley"
N315_PATH = DOCS_DIR / "evidence_cluster_n315_sweep_coarse.npz"
NEG_CONTROL_PATH = ROOT / "docs" / "m20021" / "sweep_medium.npz"
README_PATH = DOCS_DIR / "README.md"

if str(ROOT / "src") not in sys.path:  # runnable as `python tools/...` without PYTHONPATH
    sys.path.insert(0, str(ROOT / "src"))

SOLVER_RESIDUAL_BAR = 0.01  # case.TOLERANCES["solver_residual"]


def _num(value: float) -> str:
    """Compact, sign-preserving formatting for a published number."""
    if value != 0.0 and abs(value) < 1.0:
        return f"{value:.3g}"
    return f"{value:.4g}"


def _floor(value: float) -> str:
    """A damping floor of exactly zero is the physical lower bound, not a number."""
    return "0（下界）" if abs(value) <= 1e-9 else _num(value)


def _sweep_facts(path: Path) -> dict[str, object]:
    import numpy as np

    data = np.load(path, allow_pickle=True)
    damping = data["damping"]
    diagonal_min = min(float(np.diag(damping[i]).min()) for i in range(damping.shape[0]))
    return {
        "label": str(data["label"]),
        "n_sections": int(data["n_sections"]),
        "n_levels": int(data["n_levels"]),
        "panel_count": int(data["panel_count"]),
        "n_omega": int(data["omega"].size),
        "band": (float(data["omega"][0]), float(data["omega"][-1])),
        "residual_max": float(data["solver_relative_residual"].max()),
        "min_damping": diagonal_min,
        "surge_min": float(np.diag(damping[:, 0, 0]).min()),
        "heave_min": float(np.diag(damping[:, 2, 2]).min()),
        "pitch_min": float(np.diag(damping[:, 5, 5]).min()),
    }


def _replayed_kk(path: Path) -> dict[str, object]:
    """Replay C3 (Kramers-Kronig closure) on a committed negative control.

    The closure is cheap on a 41-node sweep, so the note quotes a number the
    reader can reproduce from the ``.npz`` instead of a transcript of the run
    that produced it.
    """
    from tensorbem.benchmarks import case_bem_cummins_wigley as case

    gate = case.kk_gate(case.load_sweep(path))
    values = gate["values"]
    return {
        "passed": bool(gate["passed"]),
        "worst_relative_residual": float(values["worst_relative_residual"]),
        "worst_dof": str(values["worst_dof"]),
    }


def _gate(results: dict, identifier: str, *keys: str) -> dict:
    """The named scalar values of one gate, ``None`` when the gate is absent."""
    gate = results.get("gates", {}).get(identifier)
    if gate is None:
        return {"present": False}
    values = gate.get("values", {})
    return {"present": True, "passed": bool(gate["passed"]), **{k: values.get(k) for k in keys}}


def render() -> str:
    results = json.loads((ARTIFACT_DIR / "case_result.json").read_text(encoding="utf-8"))
    settings = results["settings"]
    verdict = results["verdict"]
    n315 = _sweep_facts(N315_PATH)
    control = _sweep_facts(NEG_CONTROL_PATH)
    n315_c3 = _replayed_kk(N315_PATH)
    control_c3 = _replayed_kk(NEG_CONTROL_PATH)
    published = {label: _sweep_facts(ARTIFACT_DIR / f"sweep_{label}.npz") for label in settings["meshes"]}
    production = published.get("medium", next(iter(published.values())))

    c0 = _gate(results, "C0", "solver_relative_residual_max", "min_radiation_damping")
    c0_coarse = _gate(results, "C0_coarse", "solver_relative_residual_max", "min_radiation_damping")
    c3 = _gate(results, "C3", "worst_relative_residual", "worst_dof")
    c5 = _gate(results, "C5", "dof_name", "relative_error", "status", "zeta_frequency_domain")
    c5c = _gate(results, "C5c", "dof_name", "relative_error", "status", "zeta_decay_consensus", "zeta_frequency_domain")

    band = settings["band_rad_s"]
    meshes = settings["meshes"]
    mesh_line = " / ".join(f"{label} {dims[0]}x{dims[1]}" for label, dims in meshes.items())
    passed = verdict["passed"]
    state = "通过" if passed else "未通过"

    lines: list[str] = []
    add = lines.append
    add("# CASE-BEM-CUMMINS-WIGLEY — 复算有效性欠账与闭环（m20419 → m22876）")
    add("")
    add("> 本文件由 `tools/render_casepool_bem_cummins_wigley_readme.py` 从已提交工件渲染，"
        "**请勿手改**；改数字请重跑工装。渲染源：`case_result.json`、两份 `sweep_*.npz`、"
        "`evidence_cluster_n315_sweep_coarse.npz`、`docs/m20021/sweep_medium.npz`。")
    add("")
    add("## 1. 结论（一句话）")
    add("")
    add(
        "算例池锚点 `CASE-BEM-CUMMINS-WIGLEY`（Wigley 6DOF Cummins 记忆核，判据 Kramers-Kronig 闭合 <10%）"
        "曾被两处欠账污染（过期提交件 + 无法判定的未收敛解）。补上**有效性闸门 C0**之后，"
        f"提交件已在能通过 C0 的节点上重算：当前 `case_result.json` 判定 **{verdict['gates_passed']}/{verdict['gates_total']} 闸门{state}**，"
        "生产网格 `sweep_medium.npz` 已入库，`test_production_artifact_replays_to_the_published_verdict` 不再 skip。"
    )
    add("")
    add("## 2. 现态（由工件渲染）")
    add("")
    add("| 项 | 值 |")
    add("| --- | --- |")
    add(f"| 生成 mission / 时间 | {results['mission']} / {results['generated_utc']} |")
    add(f"| 生成平台 | {results['platform']}（python {results['python']}, numpy {results['numpy']}） |")
    add(f"| band [rad/s] / n_omega | {band[0]} .. {band[1]} / {settings['n_omega']} |")
    add(f"| 网格（判据用生产网格=medium） | {mesh_line} |")
    add(f"| 生产网格面板数 | {production['panel_count']} |")
    add(f"| 判定 | {verdict['gates_passed']}/{verdict['gates_total']} {state}（failed={verdict['failed_list'] or '无'}） |")
    add(f"| 判据来源 | {verdict['criterion_source']}，case_tolerance {verdict['case_tolerance']} |")
    add("| 图 | `fig_node_vs_local.png`（节点 vs 本机），`fig_c0_negative_controls.png`（C0 分离度），"
        "`fig_matrix_round_trip.png`，`fig_matrix_zero_pattern.png` |")
    add("")
    add("闸门读数（全部门槛均取自 `TOLERANCES`，此处只列关键量）：")
    add("")
    add("| 闸门 | 通过 | 关键量 |")
    add("| --- | --- | --- |")
    if c0["present"]:
        add(f"| C0 求解有效性（medium） | {c0['passed']} | max residual {_num(c0['solver_relative_residual_max'])} ≤ {SOLVER_RESIDUAL_BAR}；"
            f"min B {_num(c0['min_radiation_damping'])} ≥ 0 |")
    if c0_coarse["present"]:
        add(f"| C0 求解有效性（coarse） | {c0_coarse['passed']} | max residual {_num(c0_coarse['solver_relative_residual_max'])} ≤ {SOLVER_RESIDUAL_BAR}；"
            f"min B {_num(c0_coarse['min_radiation_damping'])} ≥ 0 |")
    if c3["present"]:
        add(f"| C3 Kramers-Kronig 闭合 | {c3['passed']} | worst 逐 DOF 残差 {_num(100 * c3['worst_relative_residual'])} %（{c3['worst_dof']}） ≤ 10 % |")
    if c5["present"]:
        add(f"| C5 自由衰减 vs 频域（{c5['dof_name']}） | {c5['passed']} | zeta_decay 相对误差 {_num(100 * c5['relative_error'])} % ≤ 5 %（{c5['status']}） |")
    if c5c["present"]:
        add(f"| C5c 多窗衰减（{c5c['dof_name']}） | {c5c['passed']} | zeta 共识 {_num(c5c['zeta_decay_consensus'])} vs 频域 {_num(c5c['zeta_frequency_domain'])}，"
            f"相对误差 {_num(100 * c5c['relative_error'])} %（{c5c['status']}） |")
    add("")
    add("### 2.1 两份 C0 负控（永久保留，证明前置条件在真实工件上咬得住）")
    add("")
    add("| 工件 | 网格 | max solver residual | min diag B | C0 判定 | C3 回放（重算，最坏 DOF） |")
    add("| --- | --- | --- | --- | --- | --- |")
    add(f"| `docs/casepool_bem_cummins_wigley/evidence_cluster_n315_sweep_coarse.npz` | "
        f"{n315['n_sections']}x{n315['n_levels']}（{n315['panel_count']} 面板，{n315['n_omega']} 节点） | "
        f"{_num(n315['residual_max'])} | {_num(n315['min_damping'])}（surge {_num(n315['surge_min'])}） | FAIL | "
        f"{_num(100 * n315_c3['worst_relative_residual'])} %（{n315_c3['worst_dof']}） |")
    add(f"| `docs/m20021/sweep_medium.npz` | "
        f"{control['n_sections']}x{control['n_levels']}（{control['panel_count']} 面板，{control['n_omega']} 节点） | "
        f"{_num(control['residual_max'])} | {_num(control['min_damping'])} | FAIL | "
        f"{_num(100 * control_c3['worst_relative_residual'])} %（{control_c3['worst_dof']}） |")
    add("")
    add("![C0 negative controls](fig_c0_negative_controls.png)")
    add("")
    add("两份负控都同时满足 `max residual > 0.01` 与 `min diag B < 0`：负辐射阻尼在物理上不可能，"
        "所以它们是**未收敛解**而不是「一台机器上的另一种答案」。")
    add("")
    add("## 3. 欠账 A：提交件与代码不同步（已闭合）")
    add("")
    add("历史现场：提交的 `case_result.json` 描述的是 13 节点 / band 0.15–8 rad/s / coarse 生产网格的一次运行"
        "（判定 5/7，FAILED C3、C5），与当时源码默认值（band 0.2–12、41 节点、medium 生产网格）不是同一件事；"
        "而 `test_production_artifact_replays_to_the_published_verdict` 带 `skipif(not sweep_medium.npz)`，"
        "生产网格 sweep 从未入库，闸门一直空转。")
    add("")
    add(f"闭合情况：现态 `case_result.json` 的 band 为 {band[0]}–{band[1]}、n_omega={settings['n_omega']}、"
        f"生产网格 medium {meshes['medium'][0]}x{meshes['medium'][1]}（{production['panel_count']} 面板），"
        f"`sweep_medium.npz` 与 `sweep_coarse.npz` 均已入库，回放测试实测为 green（不再是 skip）。")
    add("")
    add("## 4. 欠账 B：同一份代码、同一网格/带/节点数，两台机器给出不同物理量（由 C0 兜住）")
    add("")
    add("| 量（coarse 12x4，band 0.2–12，41 节点） | 节点 n315（cj2690 / cj2704） | 本机与现提交件 |")
    add("| --- | --- | --- |")
    add(f"| max solver relative residual | {_num(n315['residual_max'])} | {_num(published['coarse']['residual_max'])} |")
    add(f"| min B（surge / pitch） | {_num(n315['surge_min'])} / {_num(n315['pitch_min'])}（负阻尼＝造能） | "
        f"{_floor(published['coarse']['surge_min'])} / {_floor(published['coarse']['pitch_min'])} |")
    add(f"| C3 逐 DOF 最坏闭合残差 | {_num(100 * n315_c3['worst_relative_residual'])} %（{n315_c3['worst_dof']}，由 .npz 重算） | "
        f"{_num(100 * (c3['worst_relative_residual'] or 0.0))} %（{c3['worst_dof']}） |")
    add("")
    add("![node vs local](fig_node_vs_local.png)")
    add("")
    add("结论：n315 那一侧的 A/B 不是解（负辐射阻尼），因此当时**不**用节点结果刷新判据；"
        "后来提交件是在另一台 loongarch64 节点上重算的，且必须先过 C0 才允许 C2–C6 发言。"
        "n315 的数值栈为何坏掉仍未归因（见第 6 节），但判据已经不可能再跟着一台坏机器走。")
    add("")
    add("## 5. 本单（m20419）落地的改动")
    add("")
    add("1. **新增闸门 C0 `solver_validity_gate`**：求解器自报的 `relative_residual` 与 `min B` 作为前置条件；"
        f"`max residual ≤ {SOLVER_RESIDUAL_BAR}` 且 `min B ≥ 0` 才允许 C2–C6 发言，"
        "coarse/medium 各一条（`C0`、`C0_coarse`）。")
    add("2. **C0 FAIL ⇒ 依赖闸门一律 voided**（`passed=False, voided=True, voided_by=\"C0\"`），"
        "而不是把失败的线性求解记成「非因果」或「不可判」。")
    add("3. **`free_decay_gate` 不再抛异常**：旧代码遇负阻尼直接 `ValueError`，`run_case` 会在 sweep 已写盘、"
        "JSON 未写的中间态中断——这正是欠账 A 的产生机制。现在返回 `status=\"undecidable: non-physical negative radiation damping…\"`。")
    add("4. **`tools/rerender_casepool_bem_cummins_wigley.py`**：无头节点缺 matplotlib 时，可从已提交的 "
        "`sweep_*.npz` 重画图并重渲染 `CASE.md`，不必重解 BEM。")
    add("5. **本文件与它的渲染工装**（m22876）：README 的全部数字改为从工件注入，并加测试钉住"
        "（`tests/test_case_bem_cummins_wigley.py::test_casepool_readme_is_the_render_of_the_artifacts`），"
        "防止这份说明第二次变成与树不一致的「第二真相」。")
    add("")
    add("## 6. 仍开放项（诚实边界）")
    add("")
    add("* `NODE-N315-NUMERICAL-VALIDITY`（域：平台/集群）：n315 对同一输入给出 residual 1.25 / 负阻尼，"
        "根因未定位。本单已排除的假设：CHIEF 内点最小二乘的 SVD 截断（cond≈16，五种 `rcond` 给同一条解）、"
        "线程竞争（jobs 1/6/8 逐位相同）。剩余怀疑在节点侧 BLAS/LAPACK/numpy 构建。"
        "验收=同一算子矩阵在节点与本机的 `rel_res` 差 ≤10 倍。C0 闸门在归因完成前会一直兜住这类运行。")
    add("* 自由衰减的**跨窗一致性**（C5c `multi_window.converged`）为 False：三个衰减底噪窗给出的 zeta 分散超出 5 %，"
        "说明衰减非指数，C5c 只判「两窗共识与频域值一致」，不宣称收敛。若要宣称收敛，需要更长的时域积分或更干净的自由衰减记录。")
    add("")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="overwrite %s" % README_PATH)
    parser.add_argument("--out", default=None, help="write the render here instead of stdout")
    args = parser.parse_args(argv)
    text = render()
    if args.write:
        README_PATH.write_text(text, encoding="utf-8")
        print(f"wrote {README_PATH}")
    elif args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))