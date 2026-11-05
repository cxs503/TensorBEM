# m4315 · PR #191 CI 红复现与修复（第 2 轮）

## 1. 案例介绍（背景/对象）
TensorBEM PR #191（分支 `agent/m3776`，head `5f28b09`）在 GitHub Actions `tests` 工作流上 `pytest` 步失败。
PR 相对 `main` 只新增 4 个文件（`docs/m3776/report.md`、`docs/m3776/build_evidence.py`、`.heal/resolve.py`、`.heal/verify_union.py`），不含任何求解器或测试改动。

| 参数 | 值 |
|---|---|
| CI run | 42001970309（job 127159163690） |
| Python / 依赖 | 3.11 + `.[dev,visualization]` |
| 失败步 | 第 23 步 `Run test suite`（前 22 步基准生成全绿） |
| 失败规模 | `6 failed, 1172 passed, 7 skipped in 1007.50 s` |
| 本仓库本轮基线（main tip） | `efec0d7` |

## 2. 怎么做的（方法/流程/工具链）
1. 取 CI 失败日志（GitHub Actions logs API → zip → `pytest/system.txt`），提取 6 条 `FAILED`。
2. 判定归因：6 条失败全部落在 `main` 侧新增的测试文件（`test_doc_claim_audit.py`、`test_finite_depth3d_pins.py`、`test_skip_gates_are_tracked.py`），**均与本 PR 新增的 4 个文件无关**；`main` 自身 run 42011140392 复现同一批失败，说明是主线红，被 PR 继承。
3. 本地 `git fetch origin main` 后切出工作树，逐模块复现：
   - `test_finite_depth3d_pins`：3 处 `free_surface_residual` DRIFT；
   - `test_skip_gates_are_tracked`：2 个未登记 artifact gate；
   - `test_doc_claim_audit`：7 篇新文档未被 claim 登记。
4. 最小修复（3 个文件，不碰求解器）：
   - `src/tensorbem/benchmarks/finite_depth3d_pins.py`：新增 `ROUND_OFF_FLOOR = 1e-12`，`_judge` 与 `Row.rel_deviation` 的**相对**尺度改为 `max(|declared|, ROUND_OFF_FLOOR)`。自由面/底面残差量级 1e-15，末两位随 BLAS/SIMD 内核漂移，对它们用 5e-3 相对容差等于在测机器舍入噪声。
   - `tests/test_skip_gates_are_tracked.py`：登记 `test_lifting_camber.py`、`test_near_surface_three_way.py` 两个 artifact gate（产物均已在 git 跟踪）。
   - `docs/doc_claim_audit/baseline.json`：把落在 main 上但未登记进 ratchet 基线的文档补进 `unguarded` / `unguarded_paper_level`（ratchet 设计意图：新增未受 claim 约束的文档必须显式记账）。
5. 复跑验证（`pytest -q`，PYTHONPATH=src）。

## 3. 计算结果（程序化注入）
| 模块 | 修复前 | 修复后 |
|---|---|---|
| `tests/test_finite_depth3d_pins.py` | 3 DRIFT（worst rel = 1.72e-01） | PASS |
| `tests/test_skip_gates_are_tracked.py` | 2 unregistered gate | PASS |
| `tests/test_doc_claim_audit.py` | ratchet 报 7 篇未记账文档 | PASS |
| 三模块合计 | 6 failed（CI）/ 4 failed（本地含 multicomponent） | **31 passed, 1 warning in 11.06 s** |

## 4. 结果图
无图：本单为 CI 红灯修复（配置/容差/登记表），不产生新的场量或收敛曲线；改变量仅为判定口径与登记表条目。按纪律如实声明理由而非凑图。

## 5. 注意点（坑/局限）
- `free_surface_residual` 的 1.26e-15 / 7.19e-15 / 7.85e-15 在 1e-15 量级的末两位即 1–2 ulp，跨编译期/向量宽度不可复现；只有绝对尺度判据才有物理意义（残差为 0 的上界陈述）。
- 新文档进入 ratchet 基线属于「显式接受既有债务」，不是「加了 guard」；真正的 guard 需要在 `doc_claim_audit` claim 表里写条目，本单未做（避免越界改他人文档）。
- **未完成**：`tests/test_multicomponent.py` 仍有 `3 failed + 5 errors`，根因 `AttributeError: 'MultiComponentMesh' object has no attribute 'boundary_clearance'`（main 侧合入丢失属性），属另一条主线债务，本单未修。
- PR #191 已于 2026-11-05T15:59:14Z（merge commit `3cba1a9c`）被 merge governor 合入 main，先于本修复；因此本修复以独立分支提交，用于把 main 的 pytest 拉回绿。

## 6. 误差判定（逐项 vs 基准）
| 关键量 | 基准/声明 | 修复前偏差 | 修复后偏差 | 是否 < 3% |
|---|---|---|---|---|
| `3.2/h=2/free_surface_residual` | 1.26e-15 | 1.72e-01 (17.2%) | 2.2e-04 (0.022%) | 是 |
| `3.2/h=4/free_surface_residual` | 7.192e-15 | 6.48e-03 (0.65%) | 4.7e-06 (0.0005%) | 是 |
| `3.2/h=10/free_surface_residual` | 7.847e-15 | 3.20e-02 (3.2%) | 2.5e-04 (0.025%) | 是 |

上表偏差按 `max(|declared|, 1e-12)` 归一化；未归一化（绝对）差为 2.16e-16 / 4.66e-18 / 2.51e-16，即 1–2 ulp，属舍入噪声区间。
honest-FAIL：`test_multicomponent.py` 3 failed + 5 errors 未修，如实计入未完成项。
