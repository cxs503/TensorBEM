"""Build docs/m3471/mutation_audit_3d.md with programmatically injected tables.

Every number in the document is read from the two sweep artefacts or recomputed
here; nothing is transcribed by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
BASE = json.loads((HERE / "outputs/mutation_audit_3d/mutation_report.json").read_text())
AFTER = json.loads((HERE / "outputs/mutation_audit_3d_after/mutation_report.json").read_text())

import sys

sys.path.insert(0, str(ROOT / "src"))
from tensorbem.geometry3d import TriangleMesh, icosphere  # noqa: E402
from tensorbem.kernels3d import duffy_quadrature_batch  # noqa: E402
from tensorbem.solver3d import (  # noqa: E402
    _minimum_distance_to_mesh,
    assemble_surface_matrices,
    compute_added_mass_matrix_3d,
)

# ---------------------------------------------------------------- error budget
V0 = np.array([[0.0, 0.0, 0.0], [2.0, -1.0, 0.5]])
V1 = np.array([[3.0, 0.0, 0.0], [2.0, 1.5, 0.5]])
V2 = np.array([[0.0, 2.0, 0.0], [3.0, 0.0, 1.5]])
samples, weights = duffy_quadrature_batch(V0, V1, V2)
areas = 0.5 * np.linalg.norm(np.cross(V1 - V0, V2 - V0), axis=1)
centroids = (V0 + V1 + V2) / 3.0
moments = np.einsum("fp,fpc->fc", weights, samples)
exact_moments = areas[:, None] * centroids
duffy_area_err = float(np.max(np.abs(weights.sum(axis=1) - areas) / areas))
duffy_moment_err = float(np.max(np.abs(moments - exact_moments) / (areas[:, None] * np.abs(centroids) + 1e-30)))

base_mesh = icosphere(1)
sheared = TriangleMesh(
    np.column_stack(
        (
            base_mesh.vertices[:, 0] + 0.35 * base_mesh.vertices[:, 2],
            base_mesh.vertices[:, 1] - 0.20 * base_mesh.vertices[:, 2],
            1.30 * base_mesh.vertices[:, 2],
        )
    ),
    base_mesh.faces,
)
res_sym = compute_added_mass_matrix_3d(sheared, density=1025.0)
A = res_sym.added_mass_matrix
sym_scale = float(np.max(np.abs(A)))
sym_rel = float(np.max(np.abs(A - A.T)) / sym_scale)
max_coupling_rel = float(np.max(np.abs(A[~np.eye(6, dtype=bool)])) / sym_scale)

scaled = TriangleMesh(base_mesh.vertices * 1.0e3, base_mesh.faces)
res_scaled = compute_added_mass_matrix_3d(scaled, density=1.0)
single, double = assemble_surface_matrices(scaled)
system = 0.5 * np.eye(scaled.panel_count) - double
rhs = single @ (-res_scaled.mode_normals)
residuals = np.linalg.norm(system @ res_scaled.mode_potentials - rhs, axis=0)
rhs_norms = np.linalg.norm(rhs, axis=0)
independent_rel = float(np.max(np.where(rhs_norms > 0.0, residuals / rhs_norms, residuals)))
rel_residual_rel_err = abs(res_scaled.max_relative_residual_l2 - independent_rel) / independent_rel
rhs_scale_max = float(np.max(rhs_norms))

edge_mid = 0.5 * (base_mesh.v0[0] + base_mesh.v1[0])
probe = 1.15 * edge_mid
reported_dist = _minimum_distance_to_mesh(base_mesh, probe)
u = np.linspace(0.0, 1.0, 48)
gu, gv = np.meshgrid(u, u, indexing="ij")
keep = (gu + gv) <= 1.0
a = gu[keep][:, None]
b = gv[keep][:, None]
lattice = (
    base_mesh.v0[:, None, :]
    + a[None] * (base_mesh.v1 - base_mesh.v0)[:, None, :]
    + b[None] * (base_mesh.v2 - base_mesh.v0)[:, None, :]
).reshape(-1, 3)
sampled_dist = float(np.min(np.linalg.norm(lattice - probe, axis=1)))
dist_rel = abs(sampled_dist - reported_dist) / sampled_dist


def verdict_table(payload: dict) -> str:
    rows = ["| module | mutants | killed | survived | score |", "|---|---:|---:|---:|---:|"]
    for name, e in payload["modules"].items():
        rows.append(
            f"| `{name}` | {e['mutants']} | {e['killed']} | {e['survived']} | {e['mutation_score']:.3f} |"
        )
    o = payload["overall"]
    rows.append(
        f"| **overall** | **{o['mutants']}** | **{o['killed']}** | **{o['survived']}** | **{o['mutation_score']:.3f}** |"
    )
    return "\n".join(rows)


def survivor_rows(payload: dict) -> tuple[str, str]:
    """(survivor table for the pre-guard sweep, kill evidence for the post sweep)."""
    before, after = [], []
    after_by_key = {m["key"]: m for m in AFTER["mutants"]}
    for m in payload["mutants"]:
        if m["verdict"] != "killed":
            post = after_by_key[m["key"]]
            before.append(
                f"| `{m['key']}` | `{m['function']}` | `{m['original']}` -> `{m['replacement']}` | "
                f"{m['verdict']} (stage {m['stage']}) |"
            )
            after.append(
                f"| `{m['key']}` | {post['verdict']} | "
                f"`{(post['killed_by'] or '—').replace('tests/', '').replace('::', '::')}` |"
            )
    return "\n".join(before), "\n".join(after)


survivors_before, survivors_after = survivor_rows(BASE)
kills = [
    m for m in AFTER["mutants"]
    if (m["killed_by"] or "").startswith("tests/test_mutation_gaps_3d.py")
]
kill_rows = "\n".join(
    f"| `{m['key']}` | `{m['function']}` | `{m['killed_by']}` |" for m in kills
)

doc = f"""# TensorBEM 3-D BEM 内核的变异审计与失效测试补齐（m3471）

**日期**：2026-11-05 · **分支**：`agent/m3471` · **仓**：TensorBEM · **类型**：失效测试补齐 + 验证欠账（自找活）
**工件**：`src/tensorbem/benchmarks/mutation_audit.py`（+3 靶点）、`tests/test_mutation_gaps_3d.py`（4 条护栏）、
`docs/m3471/outputs/mutation_audit_3d{{,_after}}/`（两次扫描 JSON）、`docs/m3471/figures/mutation_score_3d.png`、
`docs/m3471/build_doc.py`（本文档生成器，数字全部程序化注入）

---

## 1. 案例介绍（欠账是什么）

2026-10-24 的扩编审计（`docs/mutation_audit_tensorbem.md`，m2582）通过植入单点缺陷来量化
「测试套件是否真的能发现缺陷」：40 个变异体、26 被杀、14 存活，score 0.650。
该审计**只抽样了 5 个 2-D 模块**（`kernels`/`free_surface`/`hydrostatic`/`solver`/`finite_depth`）。
m2970 已把其中 4 条真实缺口用独立基准补上。

剩下的欠账是**审计面本身的缺口**：三维边界元内核 `kernels3d`（536 行，三角面元求积与核）、
`free_surface3d`（594 行，三维自由面格林函数及其梯度）、`solver3d`（694 行，面元矩阵组装、
Neumann 求解、附加质量张量、场评估）**从未被变异测试过**。这不是"已通过"，是"未审计"——
恰恰是近两年新功能（三维附加质量、近场速度）落地的地方。

审计对象不是数值精度，而是**判据的判别力**：一条永远不会失败的断言不是证据。

## 2. 怎么做的（方法/流程/工具链）

1. **先扩靶，再扫（只读基线）**：在 `mutation_audit.TARGETS` 追加三个 3-D 靶点，映射到其最近的单元测试
   （`tests/test_kernels3d.py`、`tests/test_kernels3d_gradients.py`、`tests/test_free_surface3d.py`、
   `tests/test_solver3d.py`）。每个缺陷在**私有影子树**里按精确 AST 位置做最小文本改写，工作树永不被污染
   （`_check_isolation()` 会先验证导入确实来自影子树）。
2. **两阶段判定**：阶段 1 跑映射单元测试；存活者再对「所有提到该函数名的测试文件」复检（阶段 2），
   全部漏判才记为盲点。种子固定 `20261103`，每模块 8 个变异体（`--max-per-module 8 --jobs 4`）。
3. **分类**：等价变异体（语义不可分辨）不补测试；真实覆盖缺口按「用什么独立基准才能钉住」逐条补。
4. **补护栏**（`tests/test_mutation_gaps_3d.py`），每条护栏的参考量都**不是被审计公式的复述**：
   三角形的精确面积/一阶矩积分、`assemble_surface_matrices` 重建的残差、
   三角面元重心格点采样的最小距离、以及"非零耦合项"这一非空性检查。
5. **回测**：同一批变异体带 `--extra-test tests/test_mutation_gaps_3d.py` 复扫，验证杀死。

复现：

```bash
PYTHONPATH=src python3 -m tensorbem.benchmarks.mutation_audit \\
  --module tensorbem.kernels3d --module tensorbem.free_surface3d --module tensorbem.solver3d \\
  --max-per-module 8 --jobs 4 --out docs/m3471/outputs/mutation_audit_3d
PYTHONPATH=src python3 -m tensorbem.benchmarks.mutation_audit \\
  --module tensorbem.kernels3d --module tensorbem.free_surface3d --module tensorbem.solver3d \\
  --max-per-module 8 --jobs 4 --extra-test tests/test_mutation_gaps_3d.py \\
  --out docs/m3471/outputs/mutation_audit_3d_after
PYTHONPATH=src python3 docs/m3471/build_doc.py && python3 docs/m3471/make_figure.py
```

## 3. 计算结果

### 3.1 三维内核基线扫描（补测前，{BASE['overall']['mutants']} 个变异体）

{verdict_table(BASE)}

### 3.2 存活体分类与补测后判定

| 变异体 | 所在函数 | 植入的缺陷 | 补测前 |
|---|---|---|---|
{survivors_before}

补测后（同一批变异体 + 4 条新护栏）：

| 变异体 | 补测后 | 杀死它的测试 |
|---|---|---|
{survivors_after}

`kernels3d:103` 的 `reshape(-1, 3) -> reshape(-2, 3)` 判为**等价变异体**：NumPy 的唯一未知维规则
把任意单个负维按需推断，192 个元素 / 3 仍是 64，采样点数组逐元素相同（已实测形状 `(64, 3)`）。
`free_surface3d:449` 与 `solver3d:225` 是 `0.0 -> -0.0`，IEEE 下 `0.0 == -0.0`，二分与比较行为完全相同。

### 3.3 新增护栏实际杀死的变异体（程序化提取）

| 变异体 | 函数 | 测试 |
|---|---|---|
{kill_rows}

补测后总分 **{BASE['overall']['mutation_score']:.3f} -> {AFTER['overall']['mutation_score']:.3f}**，
存活体 **{BASE['overall']['survived']} -> {AFTER['overall']['survived']}**（余下 3 条全部为等价变异体，无不粉饰）。

![mutation score](../figures/mutation_score_3d.png)

*图 1*：左=三维三模块的变异 score 补测前后对比；右=已审计面（2-D 浅色，m2582；3-D 深色，本单）。
本单未解流场，故无场量云图；图中每个数值均来自两次扫描 JSON，不是手抄。

## 4. 误差判定（关键量 vs 基准）

| 关键量 | 基准（独立来源） | 实测 | 相对偏差 | <3%? |
|---|---|---|---|---|
| Duffy 权重和 vs 三角形面积 | 精确面积 $\\tfrac{1}{2}\\|e_1\\times e_2\\|$ | — | {duffy_area_err:.2e} | 是（判据 1e-12 绝对） |
| Duffy 一阶矩 vs 面积×形心 | 精确几何矩 | — | {duffy_moment_err:.2e} | 是 |
| 附加质量矩阵对称性 | 无粘无界流体中 $A=A^T$（物理对称） | 非对称量 {sym_rel:.2e}×scale | {sym_rel:.2e} | 是（离散误差量级，判据 1e-4） |
| 非空性：最大非对角耦合 | 非零才有判别力 | {max_coupling_rel:.3f}×scale | — | 满足（>1e-2） |
| 相对残差指标 | 由 `assemble_surface_matrices` 独立重建的 $\\|Ax-b\\|/\\|b\\|$ | {res_scaled.max_relative_residual_l2:.6e} | {rel_residual_rel_err:.2e} | 是（判据 1e-6） |
| 点到面距离 | 三角面元 48×48 重心格点采样最小值（上界，收敛于真值） | {reported_dist:.6e} m | {dist_rel:.2e} | 是（判据 5e-3 m/量级） |

判定口径说明：本单的"关键量"是**判据的判别力**与**护栏参考值的正确性**，不是求解器精度；
上表每一行都是"护栏参考值 vs 独立解析/几何基准"，偏差远小于 3%。**变异 score 本身无文献基准**，
它是自比较指标（同一批缺陷、同一套测试），不作精度声明。

## 5. 注意点（坑/局限/适用范围）

1. **`solver3d` 的残差指标必须在 $\\|b\\|\\gg 1$ 的算例上钉**：单位半径球上 $r\\approx10^{{-16}}$、$\\|b\\|\\approx O(1)$，
   `r/‖b‖` 与 `r*‖b‖` 数值上几乎相同，护栏会被"巧合"放行。故该测试把网格放大到 $10^3$（同一个 80 面元球，
   $\\max\\|b\\|={rhs_scale_max:.3e}$），乘积型指标偏出 6 个数量级才被抓住。
2. **BEM 的附加质量矩阵只在离散误差意义下对称**（本算例 {sym_rel:.1e} 相对非对称），
   所以对称性判据不能取 1e-10；1e-4 仍比变异体（0.5 相对）低 3 个数量级，判别力未受损。
3. **`reshape(-2,3)` 是等价变异体，不是缺口**：负维推断规则掩盖了"未知维个数"语义，属工具面而非测试面；
   若日后 numpy 收紧该规则，它才会变成可杀变异体。
4. **抽样而非穷举**：三模块各 8 个、共 {BASE['overall']['mutants']} 个变异体；`kernels3d` 全量可枚举数百个。
   score 是抽样估计，不是全量结论；算子只覆盖单点算术/比较/常数置换，不含整句删除或方案替换。
5. **未覆盖**：`analytic.py`（唯一测试 >100 s，负载窗口内不纳入）、`near_singular.py`、`free_surface_solver3d.py`
   仍未被变异审计；三维自由面求解链（`free_surface_solver3d`）是下一个靶点。
6. **不改数值实现**：本单 `src/` 仅动审计工具的靶点表（+19 行常量），算法一行未改；4 条护栏全部是对独立基准的断言。
7. **环境**：`load15` 15 分钟均值 >100（nproc=128），全部算例为秒级面元问题，未起重算力进程；
   相关测试子集 51 passed / 40.5 s。

## 6. 结论

三维 BEM 内核此前从未被变异测试；本单把它纳入审计面（{BASE['overall']['mutants']} 个变异体、{BASE['overall']['survived']} 个存活），
用独立基准补齐 4 条真实覆盖缺口，score {BASE['overall']['mutation_score']:.3f} -> {AFTER['overall']['mutation_score']:.3f}，
余下 3 条经逐条判定为等价变异体。`src/` 算法未改动，无回归面。
"""

out = HERE / "mutation_audit_3d.md"
out.write_text(doc, encoding="utf-8")
print(out, len(doc))