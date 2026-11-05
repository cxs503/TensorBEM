# m4529 · PR #283（agent/m4315）CI 红复现与修复

## 1. 案例介绍
TensorBEM PR #283（分支 `agent/m4315`，head `c9fd8f0`）在 GitHub Actions `tests` 工作流上 `pytest` 步
`failure`。该 PR 已修掉它继承的三处主线红（`finite_depth3d_pins` 舍入钉值、skip-gate 登记、doc-claim
棘轮基线），并在自己的 `docs/m4315/ci_fix_report.md` §5 明确留了一条未修项：
`tests/test_multicomponent.py` 仍有 `3 failed + 5 errors`，根因
`AttributeError: 'MultiComponentMesh' object has no attribute 'boundary_clearance'`。
本单即针对这条残余红。

| 参数 | 值 |
|---|---|
| 仓库 / 分支 | TensorBEM / `agent/m4315`（PR #283），基线 `main = efec0d7` |
| 复现环境 | CPython 3.11.6，numpy 2.4.6，scipy 1.16，pytest 9.2.0 |
| 复现命令 | `PYTHONPATH=src python -m pytest tests/test_multicomponent.py -q -rf` |
| 失败规模（本机） | `3 failed, 12 passed, 5 errors in 27.35 s` |
| 环域算例 | `annulus_mesh(r_in=0.5 m, r_out=1.0 m)`，内/外环各 N 块面板 |

## 2. 怎么做的
1. 定位调用链：`tensorbem.solver.evaluate_interior_field`（内场 Green 表示的域守卫）在求值前用
   `mesh.boundary_clearance(targets)` 做「离边界太近」筛除、用 `mesh.contains_point(targets)` 做
   「在域外」筛除。`PanelMesh` 两者都有，`MultiComponentMesh`（外环 + 孔环）两者都没有 →
   任何环域内场求值都在守卫处抛 `AttributeError`。
2. 最小修复（单文件 `src/tensorbem/multicomponent.py`，+32 行，无其它模块改动）：
   - `contains_point(points)`：委托 `fluid_contains`（外环内、所有孔外的环奇偶判据），只作为
     守卫所需协议的别名，不复制几何逻辑。
   - `boundary_clearance(points)`：对**所有环的所有面板**取最小点-线段距离；先校验形状
     `(M, 2)`。物理含义：一个点只和它最近的那块面板一样安全，孔壁同样要筛。
3. 验证工具链：`docs/m4529/guard_evidence.py` 用**不共享实现**的闭式参照标定这两个方法——
   内接多边形环域上，数值距离对解析距离 `min(|rho-r_in|, |r_out-rho|)` 的偏差必须随面板尺寸
   二阶收敛；包含判据对解析指示函数 `r_in < rho < r_out` 逐点比对。

## 3. 计算结果
| 项 | 修复前 | 修复后 |
|---|---|---|
| `tests/test_multicomponent.py` | 3 failed + 5 errors（`AttributeError`） | **20 passed**（23.5 s） |
| `tests/test_interior_field_domain_guard.py` | 通过（只走 `PanelMesh`，未覆盖孔环） | 通过 |
| 两模块合计 | — | **32 passed, 0 failed, 0 error**（40.2 s） |

`boundary_clearance` vs 解析环域距离（`docs/m4529/guard_evidence.json`，三个探测环
`rho = 0.75 / r_in+0.05 / r_out-0.05`，每环 720 点）：

| N（面板/环） | 最大绝对偏差 (m) | 最大相对偏差 | 是否 < 3% |
|---|---|---|---|
| 32 | 4.813e-03 | 9.63e-02 | 否（9.6%） |
| 64 | 1.204e-03 | 2.41e-02 | 是（2.41%） |
| 128 | 3.011e-04 | 6.02e-03 | 是（0.60%） |
| 256 | 7.527e-05 | 1.51e-03 | 是（0.15%） |

实测收敛阶 `p = 1.9996`（理论 2，内接弦高 ~ h²/8R）；包含判据与解析指示函数逐点一致率
`1.000`（全部四个 N）。

## 4. 结果图
`docs/m4529/guard_clearance.png`（脚本生成）：左＝`MultiComponentMesh.boundary_clearance` 在环域上
的距离云图（白色为面板、红色虚线为 `0.05*h_max` 守卫带），右＝数值-解析偏差随面板数的双对数
收敛曲线（参照斜率 `N^-2`）。

## 5. 注意点
- 该 bug 是**协议缺失**而非算法错误：守卫方法在 `PanelMesh` 与 `MultiComponentMesh` 上的语义必须
  一致，否则内场求值要么抛 `AttributeError`，要么（去掉守卫后）在孔内返回有限但无意义的数。
- 面板数 N=32 时解析对比偏差 9.6% > 3%：这是内接多边形几何近似误差（弦高），不是守卫 bug；
  守卫本身按面板几何精确计算距离。
- `boundary_clearance` 对含 `nan/inf` 的目标点会经 BLAS 抛出 `RuntimeWarning: invalid value
  encountered in matmul`；`evaluate_interior_field` 在调用前已校验有限性，故不影响求值路径，
  本单未加额外屏蔽（保持与 `PanelMesh` 同行为）。
- **未完成 / 局限**：整仓 `pytest` 的 CI 级全绿未能在本机取得——本机负载 ~300（多智能体并发），
  `pytest -n 8` 的 worker 在 ~44% 处被外部信号打断；集群 n315 无 `scipy`（11 个模块收集失败）且
  numpy 为 2.0.1，整仓跑出 90 failed（含本机全绿的
  `test_interior_field_domain_guard::test_boundary_clearance_matches_the_closed_form_box_distance`），
  属环境不一致，不能作为代码状态的判据，故只把**与本单改动相关**的模块判据记为已证。

## 6. 误差判定
| 关键量 | 基准 | 修复后偏差 | 是否 < 3% |
|---|---|---|---|
| `boundary_clearance`，N=256 环域 | 解析 `min(|rho-r_in|,|r_out-rho|)` | 0.15% | 是 |
| `boundary_clearance`，N=128 | 同上 | 0.60% | 是 |
| `boundary_clearance`，N=64 | 同上 | 2.41% | 是 |
| `boundary_clearance`，N=32 | 同上 | 9.63% | 否（内接弦高，非 bug） |
| 收敛阶 | 2.0 | 1.9996 | 是（<1% 偏差） |
| `test_multicomponent.py` | 0 failed | 0 failed / 20 passed | 是 |
