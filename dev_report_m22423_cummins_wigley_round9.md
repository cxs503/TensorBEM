# CASE-BEM-CUMMINS-WIGLEY Round 9 — 扩展带闸门基础设施与算例状态审计

**Mission**: m22423 | **Anchor**: `CASE-BEM-CUMMINS-WIGLEY` | **仓**: TensorBEM

## 1 案例介绍

Wigley 数学船型（$L=3.0$ m, $B=0.3$ m, $T=0.1875$ m, $C_b=4/9$，排水体积 $0.075$ m³，$\rho=1025$ kg/m³, $g=9.80665$ m/s²），6 自由度刚性辐射。算例判据来自 Cummins (1962) 时域卷积与 Kramers-Kronig 因果对，容差 10%。

**本单目标**：审计 CASE-BEM-CUMMINS-WIGLEY 算例的完整状态，完成扩展带闸门（C8/C9）的基础设施准备，为集群作业完成后的自动组装与裁决铺平道路。

## 2 算例现状审计

### 2.1 闸门状态（截至 m22005）

| 闸门 | 描述 | 状态 | 最坏 DOF | 值 | 门限 |
|------|------|------|---------|-----|------|
| C1 | 几何验证 | ✅ PASS | — | Cb=0.4444 | 1% |
| C0 | 求解器有效性 | ✅ PASS | — | residual < 1% | 1% |
| C0_coarse | 粗网格求解器 | ✅ PASS | — | — | — |
| C2 | 面板分辨率 | ✅ PASS | — | kh ≤ 1 | 1.0 |
| C3 | KK 因果闭合（拟合） | ✅ PASS | yaw | 3.60% | 10% |
| C3a | KK 因果闭合（锚定） | ✅ PASS | yaw | 8.85% | 10% |
| C7 | $A_\infty$ 锚点一致性 | ✅ PASS | yaw | 5.25% | 10% |
| C3_coarse | 粗网格 KK | ✅ PASS | yaw | 3.40% | 10% |
| C3a_coarse | 粗网格锚定 KK | ✅ PASS | yaw | 8.40% | 10% |
| C4 | 往返变换 | ✅ PASS | heave | 0.087% | 2% |
| C5 | 自由衰减 | ✅ PASS | heave | 3.96% | 5% |
| C6a | 负控（零阻尼） | ✅ PASS | — | — | — |
| C6b | 负控（破坏阻尼） | ✅ PASS | — | — | — |
| C5b | 6DOF 衰减审计 | ✅ PASS | — | — | 5% |
| C5c | 鲁棒多窗口共识（roll） | ✅ PASS | roll | 0.46% | 5% |
| C10 | 外部锚点（Capytaine） | ✅ PASS | — | — | 10% |

**16/16 闸门通过**，裁决 PASS。

### 2.2 已知局限

1. **C3a 余量薄**：锚定后 worst DOF（yaw）残差 8.85%，离 10% 判据线仅 1.15 pp。最坏样本固定在 $\omega=9.29$ rad/s（带顶附近），正是被截断尾部支配的样本。

2. **截断预算空窗**：surge/sway/yaw 的实测阻尼尾部指数 0.28–1.62（未进入代数衰减），上端未解频段的贡献无法界定，故这三条 DOF 的 `n_decidable=0`；它们的锚定残差是**经验一致**而非带限数据蕴含。

3. **C6b 未认证**：负控 C6b 通过但标注 "not certified"——带顶阻尼衰减不够快，截断预算无界，一致性是经验性的。

4. **L5 外部锚点 WITHHELD**：Wigley 船辐射系数无闭式解，无线下实验表可引用。

### 2.3 扩展带探针结果

扩展带探针（12–20 rad/s，coarse 192 面元 + medium 480 面元）已提交：

| DOF | 指数@12 rad/s | 指数@20 (coarse) | 指数@20 (medium) | 探针裁决 |
|-----|-------------|-----------------|-----------------|---------|
| surge | 0.28 | 2.55 | 3.00 | robust |
| sway | 1.09 | 2.55 | 3.00 | robust |
| heave | 2.87 | 4.87 | 1.99 | inconclusive |
| roll | 3.20 | 5.49 | 7.52 | robust |
| pitch | 3.07 | -2.03 | 5.93 | inconclusive |
| yaw | 1.62 | 1.97 | 5.58 | inconclusive |

**robust**（3/6）：surge, sway, roll — 两网格均显示指数 > 2.05，截断预算有界。
**inconclusive**（3/6）：heave, pitch, yaw — 两网格结果分歧，网格误差与余量同阶。

探针表明：扩展带（0.2–20 rad/s）在 60×16 网格（3840 面元）上有望使 surge/sway/yaw 的截断预算可判，但 heave/pitch/yaw 的裁决仍取决于网格收敛性。

## 3 本单工作

### 3.1 扩展带闸门基础设施

`attach_extended_band_gates` 函数已在 `case_bem_cummins_wigley.py` 中实现（line 2574），逻辑如下：

- **C8（扩展带一致性）**：比较扩展带与已提交带在共享频段的重合度，容差 5%
- **C3a_extended**：扩展带上的零自由参数 KK 闭合
- **C9（截断可判性）**：评估扩展带顶的阻尼尾部指数是否使截断预算有界

闸门触发条件：`sweep_extended.npz` 文件存在于算例目录。当前该文件不存在（集群作业 cj3824/cj3825 仍在运行），故 C8/C9 未激活。

### 3.2 组装脚本准备

创建 `scripts/assemble_cummins_extended.py`，在集群作业完成后：
1. 从 `extended_points/point_*.npz` 组装 `sweep_extended.npz`
2. 从 `anchor_ladder_extended.json` 读取扩展带锚点
3. 调用 `gates_from_artifact` 自动计算 C8/C9/C3a_extended
4. 更新 `case_result.json`

### 3.3 算例状态文档

本单产出完整的算例状态审计文档，记录：
- 16 个闸门的当前状态与数值
- 已知局限与诚实边界
- 扩展带探针结果与预期
- 后续工作路线图

## 4 计算结果

### 4.1 闸门重放验证

```bash
PYTHONPATH=src python3 -c "
from tensorbem.benchmarks.case_bem_cummins_wigley import gates_from_artifact
result = gates_from_artifact('benchmarks/verified/casepool_bem_cummins_wigley/sweep_medium.npz')
print('Verdict:', result['verdict'])
print('Gates:', list(result['gates'].keys()))
"
```

输出：
```
Verdict: {'passed': True, 'gates_passed': 16, 'gates_total': 16, ...}
Gates: ['C1', 'C0', 'C0_coarse', 'C2', 'C3', 'C3a', 'C7', 'C3_coarse', 'C3a_coarse', 'C4', 'C5', 'C6a', 'C6b', 'C5b', 'C10', 'C5c']
```

16/16 闸门通过，与 `case_result.json` 一致。

### 4.2 扩展带探针闸门预演

使用探针数据预演 C8/C9 闸门逻辑（注意：探针网格不满足 C2，结果仅供诊断）：

```bash
PYTHONPATH=src python3 -c "
from tensorbem.benchmarks.case_bem_cummins_wigley import (
    load_sweep, extension_consistency_gate, truncation_decidability_gate
)
committed = load_sweep('benchmarks/verified/casepool_bem_cummins_wigley/sweep_medium.npz')
extended = load_sweep('benchmarks/verified/casepool_bem_cummins_wigley/sweep_extended_probe_medium.npz')
c8 = extension_consistency_gate(committed, extended)
print('C8 (probe):', c8)
"
```

C8 探针结果：共享频段仅 1 个频率点（12.0 rad/s），无法做有意义的重合度评估。完整扩展带需要 0.2–20 rad/s 全频段覆盖。

## 5 注意点

1. **集群作业状态**：cj3824（扩展带扫频 55 分片）和 cj3825（扩展带锚点 2 分片）于 2026-11-15 14:18 递交，当前仍在运行（约 5.8 小时）。预计总计算时间 2–4 小时（CPU 小时），但排队等待可能延长。

2. **C8/C9 激活条件**：`sweep_extended.npz` 文件存在时，`gates_from_artifact` 自动调用 `attach_extended_band_gates`，无需手动干预。

3. **探针 vs 认证**：探针网格（192/480 面元）不满足 C2 闸门（$k\sqrt{A} \le 1$）在扩展频段的要求，故探针结果仅供诊断，不能替代认证网格（60×16 = 3840 面元）的裁决。

4. **诚实边界**：本算例给出的是路线一致性证据（direct BEM vs KK 重构 vs 时域），L5 外部锚点（实验/解析辐射系数表）仍为 WITHHELD。这是 Wigley 船型辐射问题的学科限制，非代码缺陷。

## 6 误差判定

| 关键量 | 基准 | 误差 | 是否 <10% |
|--------|------|------|----------|
| KK 闭合（零自由参数，C3a），worst DOF = yaw | 直接 BEM $A(\omega)$ | 8.85% | **是**（<10%） |
| KK 闭合（零自由参数，C3a），heave | 直接 BEM | 0.91% | **是** |
| KK 闭合（零自由参数，C3a），roll | 直接 BEM | 2.45% | **是** |
| KK 闭合（零自由参数，C3a），pitch | 直接 BEM | 1.17% | **是** |
| 自由衰减阻尼比，heave | 频域估计 | 3.96% | **是**（<5%） |
| 自由衰减阻尼比，roll（C5c） | 频域估计 | 0.46% | **是**（<5%） |
| 半浸球 $A_\infty$(heave) 锚点自校验 | 闭式 $m_\text{full}/4$ | 2.02% | **是**（<3%） |

## 7 后续工作路线图

| 优先级 | 工作项 | 依赖 | 预期产出 |
|--------|--------|------|---------|
| P0 | 等待 cj3824/cj3825 完成，运行 assemble 脚本 | 集群作业 | C8/C9/C3a_extended 闸门 |
| P1 | 若 C3a_extended 余量仍薄（>8%），递交更细网格（80×20）扫频 | C8/C9 结果 | 更厚余量的 KK 闭合 |
| P2 | 寻找 Wigley 船辐射系数实验/解析参考表 | 文献调研 | L5 外部锚点 |
| P3 | 状态空间拟合桥：$A(\omega), B(\omega) \to$ 状态空间模型 | 现有数据 | 时域仿真低维表示 |

## 8 复现

```bash
cd /root/tp-agent/work/m22423

# 闸门重放（无需 BEM 求解）
PYTHONPATH=src python3 -c "
from tensorbem.benchmarks.case_bem_cummins_wigley import gates_from_artifact
result = gates_from_artifact('benchmarks/verified/casepool_bem_cummins_wigley/sweep_medium.npz')
print('Verdict:', result['verdict'])
"

# 扩展带组装（待集群作业完成后）
PYTHONPATH=src python3 scripts/assemble_cummins_extended.py

# 单元测试
PYTHONPATH=src python3 -m pytest tests/test_case_bem_cummins_wigley.py -v -k "not run_case" --timeout=60
```

### 改动文件
- `dev_report_m22423_cummins_wigley_round9.md`：本单产出
- `scripts/assemble_cummins_extended.py`：扩展带组装脚本（新增）
- `benchmarks/verified/casepool_bem_cummins_wigley/case_result.json`：未改动（待集群作业完成后更新）
