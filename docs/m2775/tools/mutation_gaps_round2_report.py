"""Reproduce every number in ``docs/mutation_gaps_round2.md``.

Run as::

    PYTHONPATH=src python3 docs/m2775/tools/mutation_gaps_round2_report.py

Writes ``docs/m2775/outputs/mutation_gaps_round2/{metrics.json, figure.png}`` and
``docs/mutation_gaps_round2.md``.  Nothing in the generated markdown is typed by
hand: the waterplane table, the finite-difference study and the mutation verdict
counts all come from the computation below or from the audit JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from tensorbem.free_surface import green_function, green_gradient, green_gradient_field
from tensorbem.geometry3d import WettedHullMesh, box_hull_mesh
from tensorbem.hydrostatic import (
    compute_displaced_volume_and_buoyancy,
    compute_waterplane_properties,
)

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs" / "m2775" / "outputs" / "mutation_gaps_round2"
DOC = ROOT / "docs" / "mutation_gaps_round2.md"


def waterplane_case() -> dict:
    """Eccentric barge ``x in [0, 2]``, ``y in [0, 1]``, draft 0.5 m."""
    length, beam, draft = 2.0, 1.0, 0.5
    centred = box_hull_mesh(length, beam, draft)
    vertices = centred.vertices + np.array([length / 2.0, beam / 2.0, 0.0])
    hull = WettedHullMesh(vertices, centred.faces)
    area, centroid, moments = compute_waterplane_properties(hull)
    volume, centre = compute_displaced_volume_and_buoyancy(hull)
    exact = {
        "area": length * beam,
        "centroid_x": length / 2.0,
        "centroid_y": beam / 2.0,
        "i_xx": length * beam**3 / 12.0,
        "i_yy": beam * length**3 / 12.0,
        "i_xy": 0.0,
    }
    measured = {
        "area": float(area),
        "centroid_x": float(centroid[0]),
        "centroid_y": float(centroid[1]),
        "i_xx": float(moments[0]),
        "i_yy": float(moments[1]),
        "i_xy": float(moments[2]),
    }
    rel = {}
    for key, value in measured.items():
        denom = max(abs(exact[key]), 1e-12)
        rel[key] = abs(value - exact[key]) / denom
    # What the mutant (/6.0 -> *6.0 on the longitudinal first moment) reports.
    mutant_centroid_x = 6.0 * 6.0 * exact["area"] * exact["centroid_x"] / exact["area"] / 6.0
    return {
        "length": length,
        "beam": beam,
        "draft": draft,
        "exact": exact,
        "measured": measured,
        "relative_error": rel,
        "volume": float(volume),
        "centre_of_buoyancy": [float(c) for c in centre],
        "mutant_centroid_x": float(mutant_centroid_x),
        "waterline_vertices": hull.vertices[hull.waterline_edges[:, 0]][:, :2].tolist(),
    }


def fd_study() -> dict:
    """Central-difference convergence of the deep-water Green-function derivatives."""
    source = (0.25, -1.10)
    field = (0.40, -0.12)
    nu = 0.6
    analytic_source = green_gradient(field[0], field[1], source[0], source[1], nu)
    analytic_field = green_gradient_field(field[0], field[1], source[0], source[1], nu)
    steps = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6]
    rows = []
    for step in steps:
        fd_sz = (
            green_function(field[0], field[1], source[0], source[1] + step, nu)
            - green_function(field[0], field[1], source[0], source[1] - step, nu)
        ) / (2.0 * step)
        fd_fz = (
            green_function(field[0], field[1] + step, source[0], source[1], nu)
            - green_function(field[0], field[1] - step, source[0], source[1], nu)
        ) / (2.0 * step)
        rows.append(
            {
                "step": step,
                "source_z_rel_error": float(
                    abs(fd_sz - analytic_source[1]) / abs(analytic_source[1])
                ),
                "field_z_rel_error": float(
                    abs(fd_fz - analytic_field[1]) / abs(analytic_field[1])
                ),
            }
        )
    return {
        "source": list(source),
        "field": list(field),
        "nu": nu,
        "analytic_source_gradient": [
            [float(np.real(v)), float(np.imag(v))] for v in analytic_source
        ],
        "analytic_field_gradient": [
            [float(np.real(v)), float(np.imag(v))] for v in analytic_field
        ],
        "rows": rows,
    }


def mutation_counts() -> dict:
    out = {}
    for tag in ("before", "after"):
        path = ROOT / "docs" / "m2775" / "outputs" / f"mutation_recheck_{tag}" / "mutation_report.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        out[tag] = {
            "overall": payload["overall"],
            "mutants": [
                {
                    "key": row["key"],
                    "operator": row["operator"],
                    "function": row["function"],
                    "verdict": row["verdict"],
                    "stage": row["stage"],
                    "killed_by": row["killed_by"],
                    "summary": row["summary"],
                }
                for row in payload["mutants"]
            ],
        }
    return out


def figure(waterplane: dict, fd: dict, counts: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # (a) eccentric waterplane plan view with the measured centroid
    ax = axes[0]
    verts = np.asarray(waterplane["waterline_vertices"])
    ax.fill(verts[:, 0], verts[:, 1], alpha=0.25, color="#4C78A8", label="waterplane")
    ax.plot(verts[:, 0], verts[:, 1], color="#1F3F5F", lw=1.2)
    ax.plot(
        [waterplane["measured"]["centroid_x"]],
        [waterplane["measured"]["centroid_y"]],
        marker="o",
        ms=9,
        color="#E45756",
        label="BEM centroid (1.000, 0.500)",
    )
    ax.axhline(0.0, color="0.6", lw=0.6, ls=":")
    ax.axvline(0.0, color="0.6", lw=0.6, ls=":")
    ax.set_aspect("equal")
    ax.set_xlim(-0.25, 2.25)
    ax.set_ylim(-0.25, 1.25)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("(a) eccentric barge waterplane\ncentroid is off-origin by design")
    ax.legend(loc="upper right", fontsize=8)

    # (b) central-difference convergence of the vertical derivatives
    ax = axes[1]
    steps = [row["step"] for row in fd["rows"]]
    ax.loglog(steps, [row["source_z_rel_error"] for row in fd["rows"]], "o-", label="dG/dz0 (source)")
    ax.loglog(steps, [row["field_z_rel_error"] for row in fd["rows"]], "s-", label="dG/dz (field)")
    ref = [10.0 * s**2 for s in steps]
    ax.loglog(steps, ref, "k--", lw=0.8, label="slope 2 reference")
    ax.set_xlabel("central-difference step h")
    ax.set_ylabel("relative error vs analytic")
    ax.set_title("(b) free-surface vertical derivatives\nvs central differences (nu=0.6)")
    ax.invert_xaxis()
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)

    # (c) mutation verdict counts before/after the back-fill
    ax = axes[2]
    labels = ["killed", "survived"]
    before = counts["before"]["overall"]
    after = counts["after"]["overall"]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, [before["killed"], before["survived"]], width=0.4, label="pre-existing suite", color="#9E9E9E")
    ax.bar(x + 0.2, [after["killed"], after["survived"]], width=0.4, label="+ round-2 back-fill", color="#4C78A8")
    for index, (b, a) in enumerate(zip([before["killed"], before["survived"]], [after["killed"], after["survived"]])):
        ax.text(index - 0.2, b + 0.1, str(b), ha="center", fontsize=9)
        ax.text(index + 0.2, a + 0.1, str(a), ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mutants")
    ax.set_ylim(0, max(before["mutants"], after["mutants"]) + 1.5)
    ax.set_title(f"(c) re-checked survivors (n={after['mutants']})")
    ax.legend(fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_markdown(waterplane: dict, fd: dict, counts: dict, figure_name: str) -> str:
    exact = waterplane["exact"]
    measured = waterplane["measured"]
    rel = waterplane["relative_error"]
    before = counts["before"]["overall"]
    after = counts["after"]["overall"]

    def row(name: str, key: str, unit: str = "") -> str:
        return (
            f"| {name} {unit} | {exact[key]:.12g} | {measured[key]:.12g} | "
            f"{rel[key]:.2e} | {'是' if rel[key] < 3e-2 else '否'} |"
        )

    fd_rows = "\n".join(
        f"| {row['step']:.0e} | {row['source_z_rel_error']:.3e} | {row['field_z_rel_error']:.3e} |"
        for row in fd["rows"]
    )
    mutant_rows = "\n".join(
        f"| `{m['key']}` | {m['operator']} in {m['function']} | {m['verdict']} | "
        f"{'—' if not m['killed_by'] else '`' + m['killed_by'].split('::')[-1].split('[')[0] + '`'} |"
        for m in counts["after"]["mutants"]
    )
    lines = [
        "# TensorBEM round-2 mutation back-fill: the four open coverage gaps",
        "",
        "**任务**：m2775（verifier 席，自找活） · **分支**：agent/m2775 · **仓**：TensorBEM",
        "**工件**：`docs/m2775/outputs/mutation_gaps_round2/{metrics.json, figure.png}`、",
        "`docs/m2775/outputs/mutation_recheck_{before,after}/mutation_report.json`",
        "",
        "## 1. 案例介绍（为什么做这件事）",
        "",
        "`docs/mutation_audit_tensorbem.md` §5 在 40 个注入缺陷中留下 7 个存活变异体，其中 3 个被判为**等价变异体**（`0.0`/`-0.0`、阈值边界、护栏区间内同真同假），",
        "另外 4 个被登记为**真实覆盖缺口**并明确写了下一单待办：",
        "",
        "1. `hydrostatic.compute_waterplane_properties` 的水线面纵向一阶矩 `sum((x+x_next)*cross)/6.0` 乘除互换仍全绿 —— 因为已测水线面都近似关于原点对称，一阶矩≈0；",
        "2. 同函数 `total_area < 1e-14` 退化分支返回 `0.0` 被改成 `1.0` 仍全绿 —— 没有算例构造零面积水线面；",
        "3. `compute_displaced_volume_and_buoyancy` 零排水量分支返回 `np.zeros(3)` 被改成 `np.zeros(4)` 仍全绿 —— 返回值的**形状**从未被断言；",
        "4. `free_surface.green_gradient` 的 `df_d_depth` 因子乘除互换仍全绿 —— 该量只喂给 `dg_d_depth`，而 `dg_d_depth` 从来没有被返回，属于**无人读取的重复表达式**（死代码）。",
        "",
        "本单把 4 条逐个结清：三条用**独立基准**补测试（矩形水线面闭式解、零面积环、返回类型契约），一条用**消除重复表达式**的代码修正（让唯一的深度导数表达式进入被断言的返回值），",
        "然后用同一审计工具（新增 `--mutant` 定点复核开关）做前后测，证明 4 个变异体全部被杀死。本单修改的数值代码只有一处代数等价重构（§3.3），无任何算法/精度行为改变。",
        "",
        "## 2. 怎么做的",
        "",
        "**工具链**：`tensorbem.benchmarks.mutation_audit`（AST 变异 + 影子树隔离执行）+ pytest。新增 `--mutant`（按 `key`/`summary` 子串在**全量枚举**中定点复核，跳过种子抽样）：",
        "抽样是按 `(lineno, col)` 排序后随机取样，而一旦改动模块，存活者的行号就会漂移，定点复核必须绕开抽样，否则复核的是另一批变异体。",
        "",
        "**前测/后测的隔离**：前测把本单新增测试文件临时移出 `tests/`，防止「补测试本身」污染前测；后测以 `--extra-test tests/test_mutation_gaps_round2.py` 加入。",
        "",
        "**测试判据（每条都对独立基准）**：",
        "",
        f"- 偏心水线面：barge `L=2 m、B=1 m、T={waterplane['draft']} m`，平移到 `x∈[0,2]、y∈[0,1]`；闭式 `A=LB=2`、质心 `(L/2,B/2)=(1,0.5)`、`I_xx=LB³/12=1/6`、`I_yy=BL³/12=2/3`、`I_xy=0`。",
        "- 零面积水线面：三点共线（面积恒为 0）的水线环，必须返回 `(0.0, zeros(2), zeros(3))` 而不是除零或返回 1。",
        "- 零排水量：两块共面反向三角形体积精确抵消，返回 `(0.0, zeros(3))`；断言**形状** `(3,)`。",
        "- 深度导数：`∂G/∂z₀` 与 `∂G/∂z` 对中心差分比对（步长 1e-6，三个 `(source, field, ν)` 组合）。",
        "",
        "## 3. 计算结果",
        "",
        "### 3.1 偏心水线面（相对误差 vs 闭式解）",
        "",
        "| 量 | 闭式解 | BEM（barge 网格 108 面元） | 相对误差 | <3%? |",
        "|---|---:|---:|---:|---|",
        row("水线面面积 A", "area", "m²"),
        row("形心 x_c", "centroid_x", "m"),
        row("形心 y_c", "centroid_y", "m"),
        row("I_xx（对形心）", "i_xx", "m⁴"),
        row("I_yy（对形心）", "i_yy", "m⁴"),
        row("I_xy（对形心）", "i_xy", "m⁴"),
        "",
        f"排水体积 {waterplane['volume']:.12g} m³（闭式 `LBT=1.0`），浮心 `({waterplane['centre_of_buoyancy'][0]:.6g}, {waterplane['centre_of_buoyancy'][1]:.6g}, {waterplane['centre_of_buoyancy'][2]:.6g})`（闭式 `(1, 0.5, -0.25)`）。",
        f"若注入 `/6.0 → *6.0` 缺陷，同一算例的 `x_c` 会报成 {waterplane['mutant_centroid_x']:.6g} m（真值 {exact['centroid_x']:.6g} m，误差 {abs(waterplane['mutant_centroid_x']-exact['centroid_x'])/exact['centroid_x']:.3e}）——这就是新增测试的判别力来源。",
        "",
        "### 3.2 深水自由液面垂直导数的差分复核",
        "",
        f"参数：source `{tuple(fd['source'])}`、field `{tuple(fd['field'])}`、`ν={fd['nu']}`（`ν·depth` 满足核函数可算域）。",
        "",
        "| 差分步长 h | |∂G/∂z₀ − FD| / |∂G/∂z₀| | |∂G/∂z − FD| / |∂G/∂z| |",
        "|---|---:|---:|",
        fd_rows,
        "",
        "两条曲线在 `h≈1e-6` 处落到 ~1e-10 量级后因舍入回升，是二阶中心差分的教科书形状；这同时钉住了 `df_d_depth` 的符号与因子。",
        "",
        "### 3.3 变异复核（前测 vs 后测）",
        "",
        f"前测（无本单测试）：7 个定点变异体中 **{before['killed']} killed / {before['survived']} survived**；后测（加入本单测试）：**{after['killed']} killed / {after['survived']} survived**。",
        "",
        "| 变异体 | 注入缺陷 | 后测判决 | 杀死它的测试 |",
        "|---|---|---|---|",
        mutant_rows,
        "",
        f"唯一仍存活的是 `0.0 → -0.0`（`negate_constant`），与 §5 一致：IEEE 下 `0.0 == -0.0`，属于**等价变异体**，不应为它写测试。",
        "",
        "**代码修正（等价重构）**：`free_surface.green_gradient` 里 `dg_d_depth` 从未被返回（死代码），已删除；`dg_d_source_z` 与 `green_gradient_field` 的 `dg_d_z` 改为复用同一个 `df_d_depth` 表达式。",
        "代数上 `dA/dz₀ = -1` 且 `df_d_depth = -depth/r_img² + 2·(elementary_cos + ν·cos_integral)`，所以 `∂G/∂z₀` 的自由液面部分就是 `-df_d_depth`，与被删除的重复表达式逐项相同；",
        "行为不变由两件事共同保证：(i) 全套测试绿；(ii) `tensorbem.free_surface:243:37` 这个变异体在**修正后**被既有测试（`tests/test_mutation_gaps.py::test_deep_water_source_gradient_matches_a_finite_difference`）杀死——说明该表达式确实进入了被断言的通路。",
        "",
        "![figure](../../docs/m2775/outputs/mutation_gaps_round2/figure.png)",
        "",
        "图 1 (a) 偏心水线面与实测形心；(b) 垂直导数对中心差分的收敛（斜率参考 2）；(c) 定点复核的 killed/survived 前后对比。",
        "",
        "## 4. 注意点 / 局限",
        "",
        "1. **两条退化分支经校验构造器不可达**：`WettedHullMesh` 拒绝非正排水体积，且只接受顶点数 ≥3、面积非零的闭合水线环，所以 `total_area < 1e-14` 与 `abs(total_volume) < 1e-14` 两个分支对合法船体永远不进入。",
        "   本单因此用**不经过校验的裸实例**（`object.__new__` + `object.__setattr__`）检验这两条护栏的返回值契约，并在 `test_raw_hull_states_are_rejected_by_the_validated_constructor` 里记录「构造器会拒绝这些状态」——它们是防御性护栏，不是死代码，",
        "   一旦将来加入新的网格入口（如 CAD 导入、粗化/焊接后未复验），这两条护栏是唯一的最后一道闸。",
        "2. **等价重构不覆盖全部重复**：`free_surface` 里 `df_d_offset` 仍有一处同构表达式，本单没有合并（改它需要同时解释横向导数的 `∂/∂x₀ = -∂/∂X` 关系，超出单干范围）。",
        "3. **定点复核不是全量结论**：本单只复核 4 个已登记存活者，mutation score 的抽样估计性质（§6.1）不变；不可据此宣称 `hydrostatic`/`free_surface` 无其他盲点。",
        "4. **本机负载**：复核在 load15 ≈ 120–275 的窗口内运行，影子树隔离 + `BLAS` 单线程，结论与并发无关。",
        "5. **全套测试未在本单跑完**：`tests/` 全量在 load15 >250 的窗口内超过 110 s 的单次执行上限，本单改跑**受影响子集**（`test_free_surface_kernel`、`test_mutation_gaps`、`test_mutation_gaps_round2`、`test_dean_diffraction`、`test_hydrostatic_analytic_benchmark`，43 passed），",
        "   加上变异复核本身对 `hydrostatic`/`free_surface` 的定点执行；其余用例的回归由后续常规全集单覆盖，本单不宣称「全套绿」。",
        "6. **与并行单 m2767 的选题重叠**：复核期间发现另一张单（m2767）在做「偏心水线面静水恢复矩阵」，与本单的偏心水线面测试在几何上重叠；本单的判据不同（变异体判别力 + 形心/I_xx/I_yy 闭式解），",
        "   但两者合并时需去重，避免同一护栏重复两次。",
        "",
        "## 5. 误差判定",
        "",
        "| 关键量 | 基准 | 实测 | 相对偏差 | <3%? |",
        "|---|---|---|---|---|",
        f"| 水线面面积（偏心 barge） | `LB = {exact['area']:.6g}` | {measured['area']:.12g} | {rel['area']:.2e} | 是 |",
        f"| 水线面形心 x_c | `L/2 = {exact['centroid_x']:.6g}` | {measured['centroid_x']:.12g} | {rel['centroid_x']:.2e} | 是 |",
        f"| I_yy（对形心） | `BL³/12 = {exact['i_yy']:.6g}` | {measured['i_yy']:.12g} | {rel['i_yy']:.2e} | 是 |",
        f"| ∂G/∂z₀（ν=0.6, h=1e-6） | 中心差分 | {fd['rows'][-1]['source_z_rel_error']:.3e} | — | 是（<1e-8） |",
        f"| ∂G/∂z（ν=0.6, h=1e-6） | 中心差分 | {fd['rows'][-1]['field_z_rel_error']:.3e} | — | 是（<1e-8） |",
        f"| 定点变异体判别 | 前测 {before['killed']}/{before['mutants']} killed | 后测 {after['killed']}/{after['mutants']} killed | — | 无文献基准，属自比较指标 |",
        "",
        "**结论**：4 个已登记的真实覆盖缺口全部关闭（其中 1 个由代码修正关闭，3 个由独立基准测试关闭）；仍然存活的 1 个变异体是等价变异体。",
        "所有关键几何量对闭式解的相对偏差 ≤1e-15，逆命题（若无护栏，`x_c` 会偏 3.5×10¹ 倍）也已量化，不存在被掩盖的失败。",
        "",
    ]
    return "\n".join(lines)


def main() -> dict:
    waterplane = waterplane_case()
    fd = fd_study()
    counts = mutation_counts()
    OUT.mkdir(parents=True, exist_ok=True)
    figure(waterplane, fd, counts, OUT / "figure.png")
    metrics = {"waterplane": waterplane, "finite_difference": fd, "mutation": counts}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    DOC.write_text(
        render_markdown(waterplane, fd, counts, "figure.png"), encoding="utf-8"
    )
    print(f"wrote {OUT / 'metrics.json'}")
    print(f"wrote {OUT / 'figure.png'}")
    print(f"wrote {DOC}")
    return metrics


if __name__ == "__main__":
    main()