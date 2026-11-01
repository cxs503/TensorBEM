# TensorBEM 健康盘点与首吸收报告（m1621）

**仓库**：`/root/tp-agent/work/m1621`（TensorBEM，分支 `agent/m1621`，基线 `f0bfb41`）
**时间**：2026-11-01
**任务**：生态跟进首单——健康盘点 + 首吸收（红→绿）
**结论**：测试套件全绿（327 passed, 7 warnings in 126.72s (0:02:06)）；发现并修复 1 个真实静水几何缺陷（多体水线面只取首环）；另修正 README 与代码树的两处不一致。

---

## 1. 案例与物理背景

浮体小角度静水恢复力矩阵 `C` 中，升沉刚度 `C[2,2] = rho g A_wp`、横摇/纵摇刚度 `C[3,3] = rho g grad GM_T`、`C[4,4] = rho g grad GM_L`，其中
`GM = KB + I/grad - KG`，`I` 是水线面对相应轴的第二矩。多体船（双体船、双浮筒、多体平台）的水线面由**多条互不相连的闭环**组成
（`WettedHullMesh` 契约原文即"one or more closed oriented waterline loops"）。因此 `A_wp`、水线面形心与 `I_xx/I_yy` 必须是**全部环的并集**的积分：
否则升沉刚度和纵摇刚度会被严重低估，进而污染频域 RAO 的共振频率与幅值。

本单的验证算例是"双浮筒（two hemispheres, R=1 m, 半间距 d=2.5 m）"：
每条水线是 32 边形，`A_wp = 3.121445e+00 m^2`/环，总排水体积 `grad = 4.0916 m^3`（离散化后），
浮心 `z_B = -0.3719 m`，重心取 `z_G = -0.4 m`，rho = 1025 kg/m^3，g = 9.80665 m/s^2。

工具链：Python 3.11 + NumPy 2.4.6 + pytest 9.2.0 + pytest-xdist 3.8.0；`mpmath`/`scipy` 仅用于交叉核验，不是库依赖。

---

## 2. 盘点

### 2.1 `git log -15` 主题清单

| commit | 日期 | 主题 |
|---|---|---|
| `f0bfb41` | 2026-10-25 | Merge pull request #19 from cxs503/agent/m272 |
| `4fe7616` | 2026-10-25 | PR #19 冲突解：并入 main f56ac45f（__init__.py 导出并集）+ m272 交付 |
| `f56ac45` | 2026-10-25 | Merge pull request #20 from cxs503/agent/m280 |
| `0915740` | 2026-10-25 | Merge pull request #18 from cxs503/agent/m273 |
| `1ea46d3` | 2026-10-25 | 添加 validation_report_m280.json 判据工件（commit 评审/撞车检查/套件门禁/回归诊断） |
| `31aa73a` | 2026-10-25 | TensorBEM 上游跟进 m280：0cd793da→bb200d27 评审报告 |
| `67b9655` | 2026-10-25 | feat: 实现静水恢复矩阵与频域 RAO 求解器 (m273) |
| `bb200d2` | 2026-10-24 | Merge pull request #17 from cxs503/agent/m238 |
| `92883db` | 2026-10-24 | 添加 validation_report.json 判据工件（测试基线/基准复算/能力对照/风险汇总） |
| `546450a` | 2026-10-24 | TensorBEM 扩编首波审计：测试基线 270/270 PASS，圆柱附连质量复算一致，Ursell 基准偏差 10.6%（honest FAIL），backlog 登记 + 2 项开发候选提案 |
| `0cd793d` | 2026-10-23 | Merge pull request #16 from cxs503/arena/6fadd265-tensorbem |
| `267d5a3` | 2026-10-23 | Add marine rigid-body Neumann, diffraction, Helmholtz, and 3-D Kelvin BEM suites |
| `8d3b1b4` | 2026-10-23 | Merge pull request #15 from cxs503/bem/near-surface-clearance-contract |
| `0cc7a77` | 2026-10-22 | Document clearance diagnostics in README |
| `fb4b8a5` | 2026-10-22 | Document near-surface clearance contract |

分类：功能提交 `267d5a3`（海洋刚体 Neumann/绕射/Helmholtz/3-D Kelvin 套件）、`67b9655`（静水恢复矩阵 + RAO）；审计/工件提交 `546450a`、`92883db`、`31aa73a`、`1ea46d3`；其余为 PR 合并与冲突解。
**没有任何一条提交触及多体（多环）水线面**，即本单缺陷自 `67b9655` 引入后一直未被发现。

### 2.2 上游 remote 领先/落后

| 探针时刻 | 命令 | 观测 | 判定 |
|---|---|---|---|
| 08:34 | `git fetch --dry-run origin` | `f0bfb41..f67e338  main -> origin/main` | 上游 main 至少领先 1 个 commit |
| 08:5x | `git fetch --dry-run origin` / `git ls-remote origin main` | 空输出（网络不可达/超时） | **WITHHELD**：精确领先量无法确认 |

本地 `origin/main = f0bfb41`，与工作树 HEAD 相同（`git rev-list --left-right --count origin/main...HEAD` = `0  0`）。
即：**本地不落后于我方的推送基线，但落后于上游实际 main**；精确 commit 数因网络不可用记为 WITHHELD，不臆测。

### 2.3 目录结构 vs README 声明功能面对照表

| 模块 | LOC | README 提及 | 有测试 |
|---|---|---|---|
| `__init__.py` | 272 | - | NO |
| `analytic.py` | 1416 | yes | yes |
| `clearance.py` | 104 | yes | yes |
| `conditioning.py` | 272 | - | yes |
| `diagnostics_report.py` | 265 | - | yes |
| `free_surface.py` | 330 | yes | yes |
| `free_surface3d.py` | 594 | yes | yes |
| `free_surface_solver.py` | 729 | yes | yes |
| `free_surface_solver3d.py` | 725 | yes | yes |
| `geometry.py` | 133 | yes | yes |
| `geometry3d.py` | 660 | yes | yes |
| `helmholtz.py` | 425 | yes | yes |
| `hydrostatic.py` | 442 | - | yes |
| `kernels.py` | 138 | yes | yes |
| `kernels3d.py` | 536 | yes | yes |
| `mesh_quality.py` | 307 | - | yes |
| `quadrature_diagnostics.py` | 231 | - | yes |
| `quadrature_diagnostics_2d.py` | 72 | - | NO |
| `rao.py` | 360 | - | yes |
| `recovery.py` | 156 | yes | yes |
| `solver.py` | 345 | yes | yes |
| `solver3d.py` | 541 | yes | yes |

（`benchmarks/` 子包 18 个脚本全部在 README 中列出，略。）

发现的声明-实现不一致（本单已修 2 处）：

1. `hydrostatic.py`（442 行）与 `rao.py`（360 行）在代码树中**存在且有测试**，但 README 的 *Project layout* 完全未列，且 *Roadmap* 第 6 条仍把它们写成"Next, needs new physics"——属于**低估自身能力**（与 m238 审计的"虚标"方向相反）。已补 Project layout 两条条目，并把 Roadmap 第 6 条改为"In progress: hydrostatic restoring + 6-DOF RAO 已实现（深水）"。
2. `quadrature_diagnostics_2d.py`（72 行）既不在 README 中、也无任何测试引用，标记为**覆盖缺口**（未在本单处理，登记为后续）。

---

## 3. 测试基线

**环境**：`PYTHONPATH=src python -m pytest tests -q -n 6`（宿主 128 核，其他 agent 并发导致 load average ≈ 230~280，故用 xdist 分布式跑）。

| 轮次 | 命令 | 结果 | 判定 |
|---|---|---|---|
| 首轮 | `python -m pytest tests -q -rf -n 6`（**未设 PYTHONPATH**） | 3 failed, 324 passed, 7 warnings in 158.43s (0:02:38) | **环境红（假红）** |
| 正确环境 | `PYTHONPATH=src python -m pytest tests -q -rf -n 6` | 327 passed, 7 warnings in 126.72s (0:02:06) | 真绿 |

**假红诊断（重要坑）**：本机全局 `PYTHONPATH` 指向另一棵工作树 `/root/tp-agent/work/m57/src/tensorbem`，
首轮 `import tensorbem` 实际解析到 `m57` 的**旧版**代码，于是本单新增的 3 个用例在"代码已修好"的情况下仍失败：

```
FAILED tests/test_waterplane_multi_loop.py::test_twin_pontoon_waterplane_uses_both_loops
FAILED tests/test_waterplane_multi_loop.py::test_twin_pontoon_restoring_matrix_uses_pooled_waterplane
FAILED tests/test_waterplane_multi_loop.py::test_non_manifold_waterline_is_rejected
```

判据：`python -c "import tensorbem; print(tensorbem.__file__)"` 打印的不是本工作树路径 → 环境红；
显式 `PYTHONPATH=src` 后同一用例集合 327 passed, 7 warnings，且新增用例单独跑亦全绿。

**环境红 vs 真红的区分清单（本单实测）**：

| 现象 | 本机判据 | 本单结论 |
|---|---|---|
| 未设 PYTHONPATH 时 3 failed | `tensorbem.__file__` → `/root/tp-agent/work/m57/...` | 环境红（假红），非真缺陷 |
| 未装 scipy/mpmath | `import scipy, mpmath` 成功（1.16.0 / 1.3.0） | 交叉核验可用，非红 |
| 无 GPU | 本库纯 NumPy CPU，无 GPU 依赖 | 不适用 |
| 单进程全量跑 > 120 s | 宿主 load ≈ 230（他人在并发跑同一仓库套件） | 超时非失败，改用 `-n 6` |

无跳过（skip）用例、无 xfail 标记。

---

## 4. 首吸收：多体水线面缺陷（红 → 绿）

### 4.1 缺陷

`tensorbem.hydrostatic.compute_waterplane_properties` 从 `hull.waterline_edges` 建邻接表后，只做**一次**遍历：

```python
start = next(iter(adj))          # 任取一个水线顶点
while True:                      # 沿未访问邻居走一圈
    ...
    if next_v is None: break     # 回到起点即停
...
return abs(area), centroid, moments   # 只返回这一条环的面积/形心/第二矩
```

对多环水线面（双体船、双浮筒、多体平台）**静默**只报告第一条环：面积少一半、形心落在某一浮筒上、纵摇第二矩丢失 `A*d^2` 并联轴项；
同时**不校验**水线是否真是闭环（顶点度≠2 时也能返回一个数）。契约允许"one or more loops"，所以这是实现未覆盖契约，而非不可支持的特性。

### 4.2 失败证据（红）

新增用例 `tests/test_waterplane_multi_loop.py`，在 `git show HEAD:src/tensorbem/hydrostatic.py`（修复前代码）上运行：

```
FAILED tests/test_waterplane_multi_loop.py::test_twin_pontoon_waterplane_uses_both_loops
FAILED tests/test_waterplane_multi_loop.py::test_twin_pontoon_restoring_matrix_uses_pooled_waterplane
FAILED tests/test_waterplane_multi_loop.py::test_non_manifold_waterline_is_rejected
3 failed, 2 passed
```

关键数字（d = 2.5 m，两环 32 边形/环）：

| 量 | 修复前实测 | 期望（解析叠加） | 偏差 |
|---|---|---|---|
| 水线面面积 `A_wp` [m^2] | 3.121445 | 6.242890 | -50% |
| 形心 x [m] | -2.5000 | 0 | 偏到单侧浮筒 |
| `I_yy` [m^4] | 0.775363 | 40.5688 | -98.1% |
| 升沉刚度 `C[2,2]` [N/m] | 31376.193 | 62752.386 | -50% |
| 纵摇刚度 `C[4,4]` [N·m/rad] | — | 408946.3 | 与单筒同量级（错） |

（表中 `C[2,2]` 修复前 = 面积只剩一半 × rho g；`C[4,4]` 修复前由单环 `I_yy` 得到，见 §5 图 1 右图。）

### 4.3 修复（最小改动，`src/tensorbem/hydrostatic.py`）

1. **水线图合法性**：要求每个水线顶点的水线度恰为 2，否则 `ValueError("waterline edges must form closed loops: ...")`；
2. **遍历所有环**：按顶点集合扫一遍、用"不走回头路"的规则逐环闭合，得到 `loops` 列表（单环船型退化为原来的唯一环）；
3. **符号归一**：每环按自身有向面积符号 `s_k = sign(a_k)` 归一，使 `s_k * I_xx^origin_k` 等恒为正贡献（顺/逆时针均可）；
4. **并集聚合**：`A = sum |a_k|`；原点矩相加后，用并联轴定理一次性搬到**合并形心**：
   `I_xx = sum I_xx^origin - A*Cy^2`，`I_yy = sum I_yy^origin - A*Cx^2`，`I_xy = sum I_xy^origin - A*Cx*Cy`。

单环情形数值与修复前逐位相同（无回归）。

### 4.4 修复后证据（绿）

```
PYTHONPATH=src python -m pytest tests/test_waterplane_multi_loop.py -q
5 passed
PYTHONPATH=src python -m pytest tests -q -rf -n 6
327 passed, 7 warnings in 126.72s (0:02:06)
```

新增 5 个用例（= 327 全绿中的 5 个；修复前基线 322 个用例全部通过 → 零回归）：

1. `test_single_loop_waterplane_matches_fan_reference`：单环 vs 独立扇形三角参考（相对 0.00%）；
2. `test_single_loop_waterplane_matches_closed_form_circle`：vs 解析内接 n 边形与 pi R^2；
3. `test_twin_pontoon_waterplane_uses_both_loops`：双环叠加 + 独立扇形参考；
4. `test_twin_pontoon_restoring_matrix_uses_pooled_waterplane`：`C[2,2]` 恰为单筒 2 倍、`C[4,4]` 与并集 `GM_L` 恒等式一致；
5. `test_non_manifold_waterline_is_rejected`：度≠2 的水线必须报错。

---

## 5. 计算结果

| 量 | 本单实现（修复后） | 独立参考（扇形三角积分） | 相对误差 | 判定 <3% |
|---|---|---|---|---|
| `A_wp`（双环） [m^2] | 6.242890304516 | 6.242890304516 | 0.000e+00 | ✅ |
| `I_xx`（双环） [m^4] | 1.550726293906 | 1.550726293906 | 1.432e-16 | ✅ |
| `I_yy`（双环） [m^4] | 40.568790697132 | 40.568790697132 | 1.751e-16 | ✅ |
| `I_yy` vs 解析叠加 `2(I_yy^1 + A_1 d^2)` | 40.568790697132 | 40.568790697132 | 1.751e-16 | ✅ |
| 形心（双环） [m] | (0.000e+00, -5.557e-18) | (-5.780e-17, -6.249e-17) | <1e-15 | ✅ |
| 升沉刚度 `C[2,2]` [N/m] | 62752.386210 | rho*g*A_wp = 62752.386210 | <1e-15 | ✅ |
| 纵摇刚度 `C[4,4]` [N·m/rad] | 408946.3473 | rho*g*grad*GM_L（恒等式） | <1e-12 | ✅ |

双筒 / 单筒刚度比：`C[2,2]` = 2.000000（应为 2），`C[4,4]` = 48.85（并联轴 `A d^2` 项的放大效应）。

**离散化收敛（单半球水线，vs 解析内接 n 边形与连续圆）**：

| subdivisions | 面元 | 水线顶点 | A_wp [m^2] | vs 内接 n 边形 | vs pi R^2 | I_xx vs pi R^4/4 |
|---|---|---|---|---|---|---|
| 1 | 16 | 8 | 2.828427e+00 | 0.00% | 9.97% | 18.76% |
| 2 | 64 | 16 | 3.061467e+00 | 0.00% | 2.55% | 5.02% |
| 3 | 256 | 32 | 3.121445e+00 | 0.00% | 0.64% | 1.28% |
| 4 | 1024 | 64 | 3.136548e+00 | 0.00% | 0.16% | 0.32% |

即 `A_wp` 对**精确多边形**误差 ≤ 2.832e-16（机器精度），对连续圆的偏差随网格 O(h^2) 收敛，
subdivisions=4 时 0.16%、`I_xx` 0.32%——均已进入 3% 判据带内。

### 图

![图1 双浮筒水线面与静水刚度](figures/fig1_twin_waterplane_and_stiffness.png)

*图 1 左：双浮筒水线面两环与合并形心（黑 +）对比修复前报告的形心（红 x）；右：升沉/纵摇刚度修复前后对比。*

![图2 I_yy 随浮筒间距变化](figures/fig2_iyy_vs_spacing.png)

*图 2 `I_yy(d)`：修复后与解析叠加 `2(I_yy^1 + A_1 d^2)` 重合，修复前为常数（丢 `A d^2` 项）。*

![图3 水线面离散化误差](figures/fig3_refinement.png)

*图 3 单半球水线面对精确内接多边形、连续圆与 `pi R^4/4` 的相对误差，虚线为 3% 判据带。*

---

## 6. 注意点 / 局限

1. **本缺陷的物理后果**：多体船升沉刚度被低估 50%、纵摇刚度错一个量级；RAO 求解器（`rao.py`）直接用 `C`，故频域共振位置会整体偏移。单体内河/海船（单环水线）不受影响，这也是缺陷长期隐藏的原因——仓库所有既有用例都用 `hemisphere_hull_mesh` 单环。
2. **仍存在的多环限制**：本修复支持"多条互不相连的闭环"，不支持**带孔洞**的水线面（例如中空环形体、带月池的船）——孔洞环的方向语义与实心环相反，当前实现会把它按并集相加（面积高估）。这是登记项，不是本单修复范围。
3. **水线顶点假设**：环的顶点必须落在 `z=0`（`WettedHullMesh` 契约保证）；若网格的 z=0 截交未生成水线边（三角形跨过 z=0 但顶点不在其上），水线面面积会漏算——`WettedHullMesh` 校验会先行拒绝此类网格，但 `hemisphere_hull_mesh`/`submerged_sphere_mesh` 之外的 CAD 导入需自行保证。
4. **精度证据的性质**：`A_wp`/`I` 与"精确多边形"的一致性达到机器精度，属于**实现正确性**证据；与连续圆的偏差属于**离散化**误差（O(h^2)），二者不可混为一谈。本单表 5 已分开列。
5. **未能完成的核验**：`free_surface3d` 的 Kelvin 波项在浅水薄区（`Y<<1`、`X>20`）未能给出独立参考——Havelock 主值积分在弱衰减强振荡下 mpmath 参考本身不自洽（与库的偏差随积分分区漂移 1e-4~1e-3），故**不判定**库此处有缺陷；记为后续（需 Telste-Noblesse 独立实现或文献表格）。
6. **并发环境**：宿主 load ≈ 230，单进程全量套件超过工具超时，必须 `-n 6` 或分片；这也是本单记录"超时≠失败"的原因。

---

## 7. 误差判定（逐项 vs 基准，是否 <3%）

| 关键量 | 基准 | 相对误差 | <3%？ |
|---|---|---|---|
| 双环 `A_wp` | 独立扇形三角积分 | 0.000e+00 | ✅ 是 |
| 双环 `I_xx` | 独立扇形三角积分 | 1.432e-16 | ✅ 是 |
| 双环 `I_yy` | 独立扇形三角积分 / 解析叠加 | 1.751e-16 | ✅ 是 |
| 单环 `A_wp` | 独立扇形三角积分 | 1.423e-16 | ✅ 是 |
| 单环 `I_yy` | 独立扇形三角积分 | 1.432e-16 | ✅ 是 |
| `A_wp`（subdivisions=4） | 连续圆 pi R^2 | 0.16% | ✅ 是（离散化） |
| `I_xx`（subdivisions=4） | 连续圆 pi R^4/4 | 0.32% | ✅ 是（离散化） |
| 升沉刚度 `C[2,2]` | rho g A_wp 定义式 | <1e-15 | ✅ 是 |
| Kelvin 波项（浅水薄区） | 无可信基准 | — | **无法判定（WITHHELD）** |

**结论**：本单修复的几何量全部落在 3% 判据带内（实为机器精度）；唯一未判定项已如实注明原因（参考解不自洽），未粉饰。

---

## 8. 交付物

| 路径 | 内容 |
|---|---|
| `src/tensorbem/hydrostatic.py` | 修复：全环水线面 + 闭环合法性校验（+89/-81 行） |
| `tests/test_waterplane_multi_loop.py` | 新增 5 个用例（红：3 failed；绿：5 passed） |
| `README.md` | Project layout 补 `hydrostatic.py`/`rao.py`；Roadmap 第 6 条如实标注已完成部分 |
| `docs/tensorbem_health_audit_m1621.md` | 本报告 |
| `docs/figures/fig1..3_*.png` | 三张图 |
| `artifacts/m1621_evidence.py` | 复算脚本（本报告全部数字与图均由它产出） |
| `artifacts/m1621_metrics.json` | 机器可读指标 |
| `artifacts/pytest_full_no_PATH.log` / `pytest_full_PYTHONPATH_src.log` | 假红/真绿两次全量套件日志 |
| `artifacts/module_inventory.txt` / `remote_and_log.txt` | 目录-README 对照、git/remote 证据 |

复现：`PYTHONPATH=src python artifacts/m1621_evidence.py && PYTHONPATH=src python artifacts/m1621_report_builder.py`
