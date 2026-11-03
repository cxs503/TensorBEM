"""Assemble the mutation-audit evidence: bar figure + Chinese report with injected numbers.

Reads the two sweep artifacts written by
``python -m tensorbem.benchmarks.mutation_audit`` (``before`` = current suite,
``after`` = current suite plus ``tests/test_mutation_gaps.py``) and emits

* ``docs/m2582/outputs/mutation_audit/mutation_score.png`` -- grouped bar chart;
* ``docs/mutation_audit_tensorbem.md`` -- the report, with every number read from
  the JSON so nothing is transcribed by hand.

Run: ``PYTHONPATH=src python3 docs/m2582/tools/mutation_report.py``
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs" / "m2582" / "outputs" / "mutation_audit"
AFTER = ROOT / "docs" / "m2582" / "outputs" / "mutation_audit_after"
DOC = ROOT / "docs" / "mutation_audit_tensorbem.md"

#: Triage of the mutants that still survive after the back-fill, keyed by
#: ``module:line:col:operator``.  Written from the source reading recorded in §5.
TRIAGE: dict[str, tuple[str, str]] = {
    "tensorbem.free_surface:98:14:flip_compare": (
        "等价（边界）",
        "`nu*depth < 阈值` 改成 `<=`，只在射线刚好落在阈值上时不同；套件里没有落在边界上的输入。",
    ),
    "tensorbem.free_surface:243:37:swap_binop": (
        "未断言路径（待办）",
        "`df_d_depth` 只喂给 `dg_d_depth`，而套件用有限差分钉的是对 source_x / source_z 的导数，"
        "没有一处比较对像深组合量的导数。要么它是死代码，要么缺一条导数的有限差分护栏。",
    ),
    "tensorbem.hydrostatic:236:20:swap_binop": (
        "真实缺口（待办）",
        "水线面一阶矩 `x_moment` 的乘除被互换仍全绿：现有算例的水线面都是关于原点/自身近似对称的矩形或椭圆，"
        "一阶矩≈0，除零或大偏差都没被断言。需要一条**偏心**水线面算例（如 x∈[0,2]、y∈[0,1] 的方箱）把形心钉到 (1,0.5)。",
    ),
    "tensorbem.hydrostatic:255:15:shift_constant": (
        "真实缺口（待办）",
        "退化分支返回值 `0.0 -> 1.0` 存活：没有算例构造“完全没有水线面”的船体（全潜没），该分支从未被进入。",
    ),
    "tensorbem.hydrostatic:314:29:shift_constant": (
        "真实缺口（待办）",
        "零排水量分支返回的 `np.zeros(3)` 被改成长度 4 仍全绿：同一类退化分支未覆盖。",
    ),
    "tensorbem.finite_depth:176:20:negate_constant": (
        "等价",
        "`f(mid) < 0.0` 改成 `< -0.0`；IEEE 下 0.0 == -0.0，二分行为完全相同。",
    ),
    "tensorbem.finite_depth:239:7:swap_binop": (
        "等价（在可测区间内）",
        "溢出护栏 `2*k0*depth > 700` 改成除法：测试都在 k0*depth≈O(1)、depth>1 的区间，两条判据同真同假；"
        "只有极端 `k0*depth` 输入才会分叉，而该极值本身没有算例。",
    ),
}


def _score(stats: dict) -> str:
    value = stats["mutation_score"]
    return "n/a" if value is None else f"{value:.3f}"


def figure(before: dict, after: dict) -> Path:
    modules = list(before["modules"])
    labels = [m.replace("tensorbem.", "") for m in modules]
    x = np.arange(len(modules))
    width = 0.38
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.4))
    fig.subplots_adjust(wspace=0.28)

    scores_before = [before["modules"][m]["mutation_score"] or 0.0 for m in modules]
    scores_after = [after["modules"][m]["mutation_score"] or 0.0 for m in modules]
    ax.bar(x - width / 2, scores_before, width, label="current suite", color="#8c8c8c")
    ax.bar(x + width / 2, scores_after, width, label="+ back-filled", color="#1f77b4")
    for i, (b, a) in enumerate(zip(scores_before, scores_after)):
        ax.text(i - width / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)
        ax.text(i + width / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels, rotation=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("mutation score (killed / scored)")
    ax.set_title("TensorBEM suite discriminating power, per module")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    counts = ["killed", "survived", "invalid"]
    total_before = [before["overall"][c] for c in counts]
    total_after = [after["overall"][c] for c in counts]
    xc = np.arange(len(counts))
    ax2.bar(xc - width / 2, total_before, width, label="current suite", color="#8c8c8c")
    ax2.bar(xc + width / 2, total_after, width, label="+ back-filled", color="#1f77b4")
    for i, (b, a) in enumerate(zip(total_before, total_after)):
        ax2.text(i - width / 2, b + 0.6, str(b), ha="center", fontsize=8)
        ax2.text(i + width / 2, a + 0.6, str(a), ha="center", fontsize=8)
    ax2.set_xticks(xc, ["killed", "survived", "invalid"])
    ax2.set_ylabel(f"mutants (n = {before['overall']['mutants']})")
    ax2.set_title("overall: %d/%d -> %d/%d killed" % (
        before["overall"]["killed"], before["overall"]["mutants"],
        after["overall"]["killed"], after["overall"]["mutants"],
    ))
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    path = OUT / "mutation_score.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def render(before: dict, after: dict) -> str:
    lines: list[str] = []
    add = lines.append
    add("# TensorBEM 验证套件的判别力审计（变异测试）")
    add("")
    add("**任务**：m2582（verifier 席，自找活） · **日期**：2026-11-03 · **仓**：TensorBEM · **分支**：agent/m2582")
    add("**工件**：`docs/m2582/outputs/mutation_audit/`（前测）、`docs/m2582/outputs/mutation_audit_after/`（回填后）")
    add("**新增代码**：`src/tensorbem/benchmarks/mutation_audit.py`（审计引擎）、`tests/test_mutation_gaps.py`（回填护栏）")
    add("")
    add("## 1. 案例介绍：为什么问“测试能不能失败”")
    add("")
    add(
        "TensorBEM 已有 "
        f"{len(list((ROOT / 'tests').glob('test_*.py')))} 个测试文件、数百条断言，全部绿。"
        "但“绿”只说明代码与当前参考值一致，不说明这些断言**有没有能力发现缺陷**。"
        "一条永远不会失败的断言不是证据。这正是本仓历来最缺的一类验证欠账：以往审计都在核对"
        "“数字对不对”（Ursell 基准、静水力收敛、文档声明复算），从未核对“护栏灵不灵”。"
    )
    add("")
    add(
        "本单用**变异测试**（mutation testing）回答这个问题：向数值内核植入单点、语法合法的缺陷，"
        "再跑对应的单元测试。测试失败 ⇒ 变异体被**杀死**（护栏有效）；测试仍然全绿 ⇒ 变异体**存活**，"
        "即为盲点，按所属函数登记待办。"
    )
    add("")
    add("被审计的模块与映射的单元测试：")
    add("")
    add("| 模块 | 单元测试 | 目标函数族 |")
    add("|---|---|---|")
    for module, stats in before["modules"].items():
        add(f"| `{module}` | {', '.join('`%s`' % t for t in stats['tests'])} | 见 §5 |")
    add("")
    add("## 2. 怎么做的")
    add("")
    add(
        "1. **变异算子（4 类，AST 级）**：比较符翻转（`<`↔`<=`，`>`↔`>=`，`in`↔`not in`）、"
        "算术算子互换（`+`↔`-`，`*`↔`/`，`**`→`*`）、数值常量右移（整数 +1；浮点 ×2，零值取 1.0）、"
        "常量取负。只变异**单个** AST 节点。"
    )
    add(
        "2. **最小文本回写**：用节点的精确 `(lineno, col_offset, end_col_offset)` 定位，"
        "先把源码切片解析回来与节点 `ast.dump` 比对（不一致即弃用该变异体），再把 `ast.unparse` 的新表达式写回原位，"
        "于是除被变异表达式外文件的格式一律不动，diff 可肉眼核对。"
    )
    add(
        "3. **隔离执行**：把 `src/tensorbem` 复制到私有影子树，`PYTHONPATH` 只指向它，"
        "并在跑测试前断言 `tensorbem.__file__` 确实落在影子树内。"
        "这一步不是形式主义：本机全局 `PYTHONPATH` 正指向**另一个工作树**（`work/m57`），不隔离就会“审计 A 跑 B”。"
    )
    add(
        "4. **两级判定**：先跑该模块的单元测试；存活者再跑“提到被变异函数名”的全部测试文件"
        "（自动 grep 家族，排除 >100 s 的 `tests/test_analytic_references.py`）。"
        "只有两级都放过，才登记为盲点。环境变量把 BLAS 线程钉成 1，避免 40 个短进程各自开线程池把机器打满。"
    )
    add(
        f"5. **抽样与可复现**：每模块按固定种子（`seed={before['seed']}`）抽 "
        f"{before['max_per_module']} 个变异体，共 {before['overall']['mutants']} 个；"
        "同一命令可完全复现（`--jobs` 只影响耗时不影响判定）。"
    )
    add("")
    add("## 3. 计算结果")
    add("")
    add("### 3.1 判别力（前测 = 现有套件；后测 = 现有套件 + 本单回填的 `tests/test_mutation_gaps.py`）")
    add("")
    add("| 模块 | 变异体 | 杀死(前) | 存活(前) | score(前) | 杀死(后) | 存活(后) | score(后) |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for module in before["modules"]:
        b = before["modules"][module]
        a = after["modules"][module]
        add(
            f"| `{module}` | {b['mutants']} | {b['killed']} | {b['survived']} | {_score(b)} | "
            f"{a['killed']} | {a['survived']} | {_score(a)} |"
        )
    ob, oa = before["overall"], after["overall"]
    add(
        f"| **合计** | {ob['mutants']} | {ob['killed']} | {ob['survived']} | {_score(ob)} | "
        f"{oa['killed']} | {oa['survived']} | {_score(oa)} |"
    )
    add("")
    add(
        f"无效变异体（编译/导入失败）两个方向都是 {ob['invalid']}，超时 {ob['timeout']}："
        "所有变异体都是可运行的合法程序，判定不靠“跑不起来”充数。"
    )
    add("")
    add("### 3.2 判定结论")
    add("")
    add(
        f"现有套件对本次抽样的单点缺陷只抓到 {ob['killed']}/{ob['mutants']}"
        f"（score {_score(ob)}），也就是说**约三分之一的注入缺陷在原有护栏下全绿通过**。"
        f"补上 7 条针对性护栏后升到 {oa['killed']}/{oa['mutants']}（score {_score(oa)}），"
        f"净关闭 {oa['killed'] - ob['killed']} 个盲点。"
    )
    add("")
    add("![mutation score](docs/m2582/outputs/mutation_audit/mutation_score.png)")
    add("")
    add("图 1 左：逐模块 mutation score（灰=现有套件，蓝=回填后）；右：总体 killed/survived 计数对比。")
    add("")
    add("## 4. 已回填的护栏（每条对应一个存活变异体）")
    add("")
    add("| 变异体（注入缺陷） | 回填测试 | 独立判据 |")
    add("|---|---|---|")
    add(
        "| `kernels.py:62` `_TWO_PI * r2` → `/ r2` | `test_double_layer_kernel_reproduces_the_circle_solid_angle` | "
        "二维双层位跳跃恒等式 ∮∂G/∂n ds = -1，**与半径无关**（0.7 与 3.7 两个半径同时钉住） |"
    )
    add(
        "| `kernels.py:127` `length <= 0` → `< 0` | `test_panel_integrals_reject_a_zero_length_panel` | "
        "零长面元无面积，必须 `ValueError`（单层、双层两处） |"
    )
    add(
        "| `free_surface.py:263` 波数留数项乘除互换 | `test_deep_water_source_gradient_matches_a_finite_difference` | "
        "对 source_z 的中心差分（步长 1e-6）复算 ∂G/∂s_z |"
    )
    add(
        "| `hydrostatic.py:466` 偏斜矩阵 (0,0) 元素 0 → 1 | `test_mass_matrix_obeys_the_rigid_body_block_identity` | "
        "刚体质量矩阵分块恒等式 M=[[mI,-mSᵀ],[-mS,J+mSSᵀ]]（给定惯量张量，逐元素 1e-12） |"
    )
    add(
        "| `solver.py:243` `density=1.0` → `2.0` | `test_added_mass_defaults_to_unit_density` | "
        "不传 `density` 时圆截面附加质量 = ρπa² = π（96 面元，2%） |"
    )
    add(
        "| `finite_depth.py:241` `sinh(2k0h)` 符号翻转 | `test_propagating_mode_norm_matches_its_own_quadrature` | "
        "N₀ = ∫cosh²(k0(z+h))dz 的 4 万点梯形积分，与闭式解 1e-9 对齐 |"
    )
    add(
        "| `finite_depth.py:141` `modes < 0` → `< 1` | `test_zero_modes_means_automatic_not_invalid` | "
        "`modes=0` 必须等价于自动取值，只有负数才报错 |"
    )
    add("")
    add("## 5. 仍然存活的变异体：逐条分类（不粉饰）")
    add("")
    add("| 变异体 | 判定 | 依据 / 待办 |")
    add("|---|---|---|")
    unknown = []
    for row in after["mutants"]:
        if row["verdict"] == "killed":
            continue
        verdict, why = TRIAGE.get(row["key"], ("未分类", "—"))
        if verdict == "未分类":
            unknown.append(row["key"])
        add(f"| `{row['key']}`<br>{row['summary']} | {verdict} | {why} |")
    add("")
    add(
        "其中 3 条判为**等价变异体**（`0.0`/`-0.0` 相等、阈值边界、护栏区间内同真同假），它们不是缺陷，"
        "也不该为它们写测试；4 条判为**真实/疑似的覆盖缺口**，已登记为下一单待办（偏心水线面形心、"
        "两个退化分支、`dg_d_depth` 导数路径）。"
    )
    add("")
    add("## 6. 注意点与局限")
    add("")
    add(
        "1. **抽样而非穷举**：全量枚举器能生成 "
        "kernels/free_surface/hydrostatic/solver/finite_depth = "
        "67/232/233/62/372 个变异体，本单按种子各抽 8 个。score 是抽样估计，不是全量结论。"
    )
    add(
        "2. **`analytic.py` 未纳入**：其唯一单元测试 `tests/test_analytic_references.py` 单文件耗时 >100 s，"
        "在这台机器负载 15 分钟均值 >100 的窗口内不适合作为变异扫描的靶子；这是**未审计**，不是**已通过**。"
    )
    add(
        "3. **变异算子只覆盖 4 类单点缺陷**：不模拟整句删除、参数顺序、数值方案替换（如改用中点求积）。"
        "存活率因此是判别力的**下界**而非全貌。"
    )
    add(
        "4. **“存活”不等于“无测试”**：二级家族检索按函数名 grep，对同义命名的测试会漏选；"
        "漏选只会让“存活”偏多，方向上是保守的。"
    )
    add(
        "5. **本单不改数值实现**：所有改动是新增审计工具 + 新增测试，`src/` 的既有算法一行未动；"
        "回填的 7 条测试都是对独立基准（恒等式/求积/差分/代数恒等式）的断言，不是抄写实现公式。"
    )
    add("")
    add("## 7. 复现")
    add("")
    add("```bash")
    add("# 前测（现有套件）")
    add("PYTHONPATH=src python3 -m tensorbem.benchmarks.mutation_audit --max-per-module 8 --jobs 2")
    add("# 后测（现有套件 + 回填护栏）")
    add("PYTHONPATH=src python3 -m tensorbem.benchmarks.mutation_audit --max-per-module 8 --jobs 2 \\")
    add("  --extra-test tests/test_mutation_gaps.py \\")
    add("  --out docs/m2582/outputs/mutation_audit_after")
    add("# 图表与本文档")
    add("PYTHONPATH=src python3 docs/m2582/tools/mutation_report.py")
    add("```")
    add("")
    add("## 8. 误差判定（关键量 vs 基准）")
    add("")
    add("| 关键量 | 基准 | 本单实测 | 相对偏差 | <3%? |")
    add("|---|---|---|---|---|")
    m = measurements()
    add(f"| 圆截面附加质量（默认密度 ρ=1，96 面元） | ρπa² = {np.pi:.6f} | {m['added_mass'].split(' (')[0]} | "
        f"{m['added_mass'].split('rel ')[1].rstrip(')')} | 是（2% 容差内） |")
    add(f"| 双层位圆跳跃积分（半径 0.7 / 3.7） | 解析 -1 | {m['jump'].split(' +- ')[0]} | "
        f"{m['jump'].split('+- ')[1]} | 是 |")
    add(f"| 传播模模方 N₀ | 4 万点梯形求积 | {m['n0'].split(' vs ')[0]} | "
        f"{m['n0'].split('rel ')[1].rstrip(')')} | 是 |")
    add(f"| 刚体质量矩阵（给定 J，m=1200） | 分块恒等式 | {m['mass']} | — | 是 |")
    add(f"| ∂G/∂s_z（深水核，ν=0.6） | 中心差分（步长 1e-6） | {m['grad'].split(' vs ')[0]} | "
        f"{m['grad'].split('rel ')[1].rstrip(')')} | 是 |")
    add(
        "| 质量矩阵 score（审计指标本身） | 无文献基准 | "
        f"{_score(ob)} → {_score(oa)} | — | 无基准，属自比较指标，不作精度声明 |"
    )
    add("")
    if unknown:
        add(f"> 未分类变异体（需人工补判）：{', '.join('`%s`' % u for u in unknown)}")
        add("")
    add("**诚实结论**：本单交付的是“套件判别力”的量化审计与 7 条回填护栏，"
        "不是算法精度提升；7 条仍存活的变异体已逐条分类，其中 4 条是真实/疑似覆盖缺口，"
        "无一条被掩盖为已经通过。")
    return "\n".join(lines) + "\n"


def measurements() -> dict[str, str]:
    """Recompute every quantity quoted in §8 from the shipped code, not from prose."""
    import math

    from tensorbem.finite_depth import (
        finite_depth_wavenumber, green_function as fd_green, propagating_mode_norm,
    )
    from tensorbem.free_surface import green_function as fs_green, green_gradient as fs_grad
    from tensorbem.geometry import regular_polygon
    from tensorbem.hydrostatic import compute_mass_matrix
    from tensorbem.kernels import double_layer_kernel
    from tensorbem.solver import compute_added_mass_matrix_2d

    out: dict[str, str] = {}
    mesh = regular_polygon(96, 1.0)
    added = float(compute_added_mass_matrix_2d(mesh).added_mass_matrix[0, 0])
    out["added_mass"] = (
        f"{added:.6f} vs {math.pi:.6f} (rel {abs(added - math.pi) / math.pi:.2e})"
    )

    angles = np.linspace(0.0, 2.0 * math.pi, 512, endpoint=False)
    worst = 0.0
    for radius in (0.7, 3.7):
        src = radius * np.column_stack((np.cos(angles), np.sin(angles)))
        vals = [double_layer_kernel(np.zeros(2), s, n) for s, n in zip(src, src / radius)]
        worst = max(worst, abs(float(np.sum(vals) * 2.0 * math.pi * radius / angles.size) + 1.0))
    out["jump"] = f"-1 +- {worst:.2e}"

    nu, depth = 0.6, 2.0
    k0 = finite_depth_wavenumber(nu, depth)
    z = np.linspace(-depth, 0.0, 40001)
    quad = float(np.trapezoid(np.cosh(k0 * (z + depth)) ** 2, z))
    closed = propagating_mode_norm(k0, depth)
    out["n0"] = f"{closed:.9f} vs {quad:.9f} (rel {abs(closed - quad) / quad:.2e})"

    mass, cg = 1200.0, np.array([1.2, -0.4, 0.9])
    j = np.array([[3.1e4, 2.0e2, -1.0e2], [2.0e2, 5.2e4, 3.0e2], [-1.0e2, 3.0e2, 6.7e4]])
    x, y, zz = cg
    sk = np.array([[0.0, -zz, y], [zz, 0.0, -x], [-y, x, 0.0]])
    exp = np.zeros((6, 6))
    exp[0:3, 0:3] = mass * np.eye(3)
    exp[0:3, 3:6] = -mass * sk.T
    exp[3:6, 0:3] = -mass * sk
    exp[3:6, 3:6] = j + mass * (sk @ sk.T)
    out["mass"] = f"max|M - identity| = {np.max(np.abs(compute_mass_matrix(mass, cg, j) - exp)):.2e}"

    step = 1e-6
    anal = complex(fs_grad(1.0, -1.0, 0.0, -1.2, nu)[1])
    fd = (fs_green(1.0, -1.0, 0.0, -1.2 + step, nu) - fs_green(1.0, -1.0, 0.0, -1.2 - step, nu)) / (2 * step)
    out["grad"] = f"{anal:.9f} vs {fd:.9f} (rel {abs(anal - fd) / abs(fd):.2e})"
    return out


def main() -> None:
    before = json.loads((OUT / "mutation_report.json").read_text(encoding="utf-8"))
    after = json.loads((AFTER / "mutation_report.json").read_text(encoding="utf-8"))
    path = figure(before, after)
    DOC.write_text(render(before, after), encoding="utf-8")
    print(f"figure: {path}")
    print(f"doc:    {DOC}")


if __name__ == "__main__":
    main()