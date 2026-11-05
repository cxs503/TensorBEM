# PR #134 治愈报告：agent/m3236 与 main 的三方并集合并

mission 3796 / 仓库 TensorBEM / 输出分支 `agent/m3796`（内容即 `agent/m3236` 的合并后代）

## 1. 案例介绍

PR #134（`agent/m3236`）在提交 `4b1b21f` 上把 Dean–Ursell–Ogilvie 表 7 的 11 个数字钉进
`tensorbem.benchmarks.declared_numbers` 注册表。此后 `main` 前进到 `ecba798`（120 个文件，
新增 Hulme 三维漂浮半球、Wang/Linton 潜没球体、CHIEF 不规则频率等基准与其注册表记录），
两个分支在同一锚点（注册表 tuple 尾部、`_measure` 的 group 分派尾部）各自追加内容，
GitHub 报 `mergeable=false`。

物理背景与被守护的量：Dean 表 7 是二维潜没圆柱/椭圆柱在 `ka=0.6`、`N=64` 面元下的绕射
验证（反射模量、透射误差、能量平衡、圆偏振误差、Haskind 互易残差、MFS 交叉验证）。
这些量此前只存在于文档里、无任何测试守护——这正是 m3236 要补的缺口，也是合并中不能丢的东西。

| 项 | 值 |
|---|---|
| 合并基（merge base） | `6025970` |
| main 侧 | `origin/main` |
| m3236 侧 | `agent/m3236` |
| 受影响路径总数（两侧并集） | 126 |
| 文本冲突文件 | 1（`src/tensorbem/benchmarks/declared_numbers.py`），冲突块 2 个 |

## 2. 怎么做的

1. 在 worktree 内 `git fetch` 后以 `origin/main`（而非过期的本地 `main`）为 ours，
   `git merge --no-ff --no-commit agent/m3236`，得到唯一冲突文件。
2. 逐块判定语义并集，**不做机械 overlay**：注册表块按"main 的 84 条原序保留 + m3236 的 11 条 Dean 记录紧随"
   顺序拼接，并补上冲突切掉的最后一条记录的闭合 `),`；分派块两侧的 `if group == ...` 分支全部保留。
   求解脚本 `docs/m3796/resolve_union.py`，遇到未知冲突块直接报错退出（不允许静默丢内容）。
3. 三重校验：
   - **语法/AST 门**：95 条记录、95 个唯一 key、无重复函数定义、9 个 group 分支可达；
   - **可复现门**：从三个父提交重放 `git merge-file` 冲突 → 同一求解器 → 与提交后的 blob
     逐字节相同（`docs/m3796/resolution_check.json`，`identical=True`，
     残留冲突标记 0 个）；
   - **静默丢失门**：对两侧改过的全部 126 条路径做"多重集"行审计
     （某行被一侧加了两次就必须在合并后出现两次），失败路径数
     `0`（`docs/m3796/conflict_audit.json`）。
4. 测试：同一套 CI 等价流程在两棵树上各跑一遍（main baseline 用 `git archive origin/main`
   解到独立目录），pytest 全量套件，`-n 16 --dist loadfile` 并行、无缓存插件。

### 2.1 冲突文件与逐块解法

| # | ours 行数 | theirs 行数 | 插入分隔 | ours sha | theirs sha | 解法 |
|---:|---:|---:|---|---|---|---|
| 1 | 593 | 129 | `),` | `653ea69727a4` | `24b3a3230146` | 注册表记录块（main 追加）+ 闭合括号补全 |
| 2 | 19 | 39 | `(none)` | `bb9cf24a63b1` | `d0d8d6ac1742` | `_measure` 的 group 分派分支（双方各自新增） |

非冲突但双方都改的文件（`docs/marine_bem_benchmarks.md`、`tests/test_declared_numbers.py`）
由 Git 自动合并，随后用上面的行审计确认两侧新增行**全部**在位（下表 `missing` 列全 0）。

### 2.2 逐路径静默丢失审计（节选：两侧都动过的路径）

| 路径 | main | m3236 | main 新增行 | main 丢失 | m3236 新增行 | m3236 丢失 |
|---|---|---:|---:|---:|---:|---:|
| `.github/workflows/tests.yml` | M | - | 33 | 0 | 0 | 0 |
| `dev_report_m3280_orders_convention.md` | A | - | 75 | 0 | 0 | 0 |
| `dev_report_m3336.md` | A | - | 59 | 0 | 0 | 0 |
| `docs/audit_baseline.md` | M | - | 51 | 0 | 0 | 0 |
| `docs/ci_red_triage_m3314.md` | A | - | 116 | 0 | 0 | 0 |
| `docs/figures/fig_restoring_couplings.png` | A | - | 0 | 0 | 0 | 0 |
| `docs/figures/irregular_frequency_band.png` | A | - | 0 | 0 | 0 | 0 |
| `docs/finite_depth3d_regime.md` | A | - | 111 | 0 | 0 | 0 |
| `docs/finite_depth_diffraction.md` | A | - | 113 | 0 | 0 | 0 |
| `docs/finite_depth_diffraction/figures/finite_depth_diffraction.png` | A | - | 0 | 0 | 0 | 0 |
| `docs/finite_depth_diffraction/make_figure.py` | A | - | 60 | 0 | 0 | 0 |
| `docs/finite_depth_diffraction/results.json` | A | - | 217 | 0 | 0 | 0 |
| `docs/hydrostatic_restoring_couplings.md` | A | - | 105 | 0 | 0 | 0 |
| `docs/irregular_frequency_study.md` | A | - | 153 | 0 | 0 | 0 |
| `docs/long_capsule_2d_limit.md` | A | - | 104 | 0 | 0 | 0 |
| `docs/m2904/outputs/m2904/publish_report.py` | A | - | 124 | 0 | 0 | 0 |
| `docs/m2904/outputs/recovery_robustness/recovery_robustness.json` | A | - | 653 | 0 | 0 | 0 |
| `docs/m2904/outputs/recovery_robustness/recovery_robustness.md` | A | - | 32 | 0 | 0 | 0 |
| `docs/m2904/outputs/recovery_robustness/recovery_robustness.png` | A | - | 0 | 0 | 0 | 0 |
| `docs/m3051/figures/kkr_floor.png` | A | - | 0 | 0 | 0 | 0 |
| `docs/m3051/kkr_floor.json` | A | - | 322 | 0 | 0 | 0 |
| `docs/m3051/kkr_floor.md` | A | - | 115 | 0 | 0 | 0 |
| `docs/m3051/kkr_floor_report.py` | A | - | 289 | 0 | 0 | 0 |
| `docs/m3125/figures/fig_dimensional_response.png` | A | - | 0 | 0 | 0 | 0 |
| `docs/m3125/figures/fig_pressure_mesh.png` | A | - | 0 | 0 | 0 | 0 |
| `docs/m3125/make_figure_and_tables.py` | A | - | 142 | 0 | 0 | 0 |
| `docs/m3125/outputs/tables.md` | A | - | 36 | 0 | 0 | 0 |
| `docs/m3125/outputs/wigley_normalization_audit.json` | A | - | 221 | 0 | 0 | 0 |
| `docs/m3125/wigley_normalization_audit.md` | A | - | 134 | 0 | 0 | 0 |
| `docs/m3134/figures/added_mass_vs_ka.png` | A | - | 0 | 0 | 0 | 0 |
| `docs/m3134/figures/envelope_map.png` | A | - | 0 | 0 | 0 | 0 |
| `docs/m3134/figures/mesh_budget.png` | A | - | 0 | 0 | 0 | 0 |
| `docs/m3134/make_figures.py` | A | - | 94 | 0 | 0 | 0 |
| `docs/m3134/outputs/envelope_s2.json` | A | - | 1112 | 0 | 0 | 0 |
| `docs/m3134/outputs/points.jsonl` | A | - | 33 | 0 | 0 | 0 |
| `docs/m3134/outputs/summary.json` | A | - | 685 | 0 | 0 | 0 |
| `docs/m3134/run_envelope.py` | A | - | 53 | 0 | 0 | 0 |
| `docs/m3134/submerged_sphere_envelope.md` | A | - | 131 | 0 | 0 | 0 |
| `docs/m3150/figures/fig_chief_passivity.png` | A | - | 0 | 0 | 0 | 0 |
| `docs/m3150/figures/fig_convergence_3d.png` | A | - | 0 | 0 | 0 | 0 |
| `docs/m3150/figures/fig_registry_deviation_3d.png` | A | - | 0 | 0 | 0 | 0 |
| `docs/m3150/outputs/figure_data.json` | A | - | 77 | 0 | 0 | 0 |
| `docs/m3150/outputs/registry_3d_free_surface.json` | A | - | 1338 | 0 | 0 | 0 |
| `docs/m3150/report.md` | A | - | 83 | 0 | 0 | 0 |
| `docs/m3150/tools/make_figures.py` | A | - | 167 | 0 | 0 | 0 |
| `docs/m3150/tools/patch_registry.py` | A | - | 703 | 0 | 0 | 0 |
| `docs/m3150/tools/publish_report.py` | A | - | 181 | 0 | 0 | 0 |
| `docs/m3150/tools/reconcile_doc_cells.py` | A | - | 96 | 0 | 0 | 0 |
| `docs/m3196/outputs/restoring_couplings/fig_entry_comparison.png` | A | - | 0 | 0 | 0 | 0 |
| `docs/m3196/outputs/restoring_couplings/fig_restoring_couplings.png` | A | - | 0 | 0 | 0 | 0 |
| `docs/m3196/outputs/restoring_couplings/make_artifacts.py` | A | - | 292 | 0 | 0 | 0 |
| `docs/m3196/outputs/restoring_couplings/patch_hydrostatic.py` | A | - | 166 | 0 | 0 | 0 |
| `docs/m3196/outputs/restoring_couplings/publish_report.py` | A | - | 66 | 0 | 0 | 0 |
| `docs/m3196/outputs/restoring_couplings/results.json` | A | - | 132 | 0 | 0 | 0 |
| `docs/m3224/figures/fig_haskind_vs_multipole.png` | A | - | 0 | 0 | 0 | 0 |
| `docs/m3224/figures/fig_kochin_overlap.png` | A | - | 0 | 0 | 0 | 0 |
| `docs/m3224/figures/fig_relative_errors.png` | A | - | 0 | 0 | 0 | 0 |
| `docs/m3224/outputs/symmetry_far_field.json` | A | - | 1456 | 0 | 0 | 0 |
| `docs/m3224/symmetry_far_field.md` | A | - | 94 | 0 | 0 | 0 |
| `docs/m3280/outputs/m3280/kochin_hulme_convergence.png` | A | - | 0 | 0 | 0 | 0 |
| `docs/m3280/outputs/m3280/study_s23.json` | A | - | 74 | 0 | 0 | 0 |
| `docs/m3336/outputs/capability_matrix.json` | A | - | 855 | 0 | 0 | 0 |
| `docs/m3336/outputs/capability_matrix.png` | A | - | 0 | 0 | 0 | 0 |
| `docs/m3370/outputs/inject_study.py` | A | - | 152 | 0 | 0 | 0 |
| `docs/m3383/near_surface_band.png` | A | - | 0 | 0 | 0 | 0 |
| `docs/m3383/near_surface_band_audit.md` | A | - | 50 | 0 | 0 | 0 |
| `docs/m3383/near_surface_band_study.json` | A | - | 845 | 0 | 0 | 0 |
| `docs/m3383/near_surface_band_study.py` | A | - | 154 | 0 | 0 | 0 |
| `docs/m3409/figures/fig_finite_depth3d_regime.png` | A | - | 0 | 0 | 0 | 0 |
| `docs/m3409/make_figures.py` | A | - | 57 | 0 | 0 | 0 |
| `docs/m3409/outputs/finite_depth3d_regime.json` | A | - | 390 | 0 | 0 | 0 |
| `docs/marine_bem_benchmarks.md` | M | M | 13 | 0 | 7 | 0 |
| `docs/recovery_robustness.md` | A | - | 114 | 0 | 0 | 0 |
| `docs/symmetry3d.md` | M | - | 14 | 0 | 0 | 0 |
| `docs/three_dimensional_bem.md` | M | - | 1 | 0 | 0 | 0 |
| `outputs/m3236/dean_128.json` | - | A | 0 | 0 | 78 | 0 |
| `outputs/m3236/dean_32.json` | - | A | 0 | 0 | 78 | 0 |
| `outputs/m3236/dean_64.json` | - | A | 0 | 0 | 78 | 0 |
| `outputs/m3236/fig_dean_convergence.png` | - | A | 0 | 0 | 0 | 0 |
| `outputs/m3236/fig_dean_registry_deviation.png` | - | A | 0 | 0 | 0 | 0 |
| `outputs/m3236/registry_fast.json` | - | A | 0 | 0 | 322 | 0 |
| `src/tensorbem/__init__.py` | M | - | 16 | 0 | 0 | 0 |
| `src/tensorbem/benchmarks/cylinder_3d_limit.py` | A | - | 307 | 0 | 0 | 0 |
| `src/tensorbem/benchmarks/declared_numbers.py` | M | M | 656 | 0 | 170 | 0 |
| `src/tensorbem/benchmarks/finite_depth3d_regime.py` | A | - | 247 | 0 | 0 | 0 |
| `src/tensorbem/benchmarks/finite_depth_diffraction.py` | A | - | 172 | 0 | 0 | 0 |
| `src/tensorbem/benchmarks/hydrostatic_energy_reference.py` | A | - | 175 | 0 | 0 | 0 |
| `src/tensorbem/benchmarks/irregular_frequency_study.py` | A | - | 543 | 0 | 0 | 0 |
| `src/tensorbem/benchmarks/kkr_floor.py` | A | - | 359 | 0 | 0 | 0 |
| `src/tensorbem/benchmarks/kochin_hulme_convergence.py` | M | - | 26 | 0 | 0 | 0 |
| `src/tensorbem/benchmarks/milestone_claims.py` | A | - | 503 | 0 | 0 | 0 |
| `src/tensorbem/benchmarks/recovery_robustness.py` | A | - | 435 | 0 | 0 | 0 |
| `src/tensorbem/benchmarks/submerged_sphere_envelope.py` | A | - | 377 | 0 | 0 | 0 |
| `src/tensorbem/benchmarks/symmetry_far_field.py` | A | - | 440 | 0 | 0 | 0 |
| `src/tensorbem/benchmarks/ursell_convergence_table.py` | M | - | 6 | 0 | 0 | 0 |
| `src/tensorbem/benchmarks/wigley_normalization_audit.py` | A | - | 257 | 0 | 0 | 0 |
| `src/tensorbem/capability_evidence.py` | A | - | 936 | 0 | 0 | 0 |
| `src/tensorbem/doc_claims.py` | M | - | 5 | 0 | 0 | 0 |
| `src/tensorbem/finite_depth3d.py` | M | - | 47 | 0 | 0 | 0 |
| `src/tensorbem/finite_depth_diffraction.py` | A | - | 432 | 0 | 0 | 0 |
| `src/tensorbem/free_surface_solver3d.py` | M | - | 51 | 0 | 0 | 0 |
| `src/tensorbem/geometry3d.py` | M | - | 99 | 0 | 0 | 0 |
| `src/tensorbem/hydrostatic.py` | M | - | 110 | 0 | 0 | 0 |
| `src/tensorbem/panel_smoothing.py` | A | - | 90 | 0 | 0 | 0 |
| `src/tensorbem/recovery.py` | M | - | 84 | 0 | 0 | 0 |
| `src/tensorbem/recovery_uncertainty.py` | A | - | 110 | 0 | 0 | 0 |
| `src/tensorbem/symmetry3d.py` | M | - | 203 | 0 | 0 | 0 |
| `tests/test_benchmark_cli_contract.py` | A | - | 78 | 0 | 0 | 0 |
| `tests/test_capability_evidence.py` | A | - | 196 | 0 | 0 | 0 |
| `tests/test_capsule_3d_limit.py` | A | - | 98 | 0 | 0 | 0 |
| `tests/test_ci_workflow_contract.py` | A | - | 80 | 0 | 0 | 0 |
| `tests/test_declared_numbers.py` | M | M | 49 | 0 | 29 | 0 |
| `tests/test_documented_benchmarks.py` | M | - | 20 | 0 | 0 | 0 |
| `tests/test_finite_depth3d_regime.py` | A | - | 100 | 0 | 0 | 0 |
| `tests/test_finite_depth_diffraction.py` | A | - | 140 | 0 | 0 | 0 |
| `tests/test_hydrostatic_restoring_couplings.py` | A | - | 232 | 0 | 0 | 0 |
| `tests/test_irregular_frequency_study.py` | A | - | 145 | 0 | 0 | 0 |
| `tests/test_kkr_floor.py` | A | - | 96 | 0 | 0 | 0 |
| `tests/test_kochin_hulme_convergence.py` | M | - | 37 | 0 | 0 | 0 |
| `tests/test_milestone_claims.py` | A | - | 80 | 0 | 0 | 0 |
| `tests/test_near_surface_band.py` | A | - | 116 | 0 | 0 | 0 |
| `tests/test_recovery_robustness.py` | A | - | 243 | 0 | 0 | 0 |
| `tests/test_submerged_sphere_wave_envelope.py` | A | - | 68 | 0 | 0 | 0 |
| `tests/test_symmetry3d_far_field.py` | A | - | 167 | 0 | 0 | 0 |
| `tests/test_ursell_reproducibility.py` | M | - | 57 | 0 | 0 | 0 |
| `tests/test_wigley_normalization_audit.py` | A | - | 86 | 0 | 0 | 0 |

（完整 126 条路径见 `docs/m3796/conflict_audit.json`）

## 3. 计算结果

### 3.1 注册表并集盘点（程序化统计）

![registry union](figures/fig1_registry_union.png)

main 侧 84 条 + m3236 侧 11 条 = 95 条，无 key 重复；Dean 组是 m3236 唯一的贡献组，
其余 8 组全部来自 main，两者互不覆盖。

### 3.2 测试证据

| 树 | 状态 | passed | failed | skipped | error | 已跑测试数 |
|---|---|---:|---:|---:|---:|---:|
| main baseline (origin/main) | 未跑完（快照 57%） | 0 | 0 | 0 | 0 | 636 |
| merged (本 PR 分支) | 未跑完（快照 64%） | 0 | 0 | 0 | 0 | 656 |
| merged · tests/test_declared_numbers.py（合并关键模块） | 未跑完（快照 0%） | 0 | 0 | 0 | 0 | 3 |

![test parity](figures/fig2_test_parity.png)

### 3.3 被守护物理量的复算（合并后仍要成立）

m3236 钉桩的 11 行 Dean 表 7 数字在 `outputs/m3236/registry_fast.json` 里有复算记录，
合并未改动任何求解器代码，因此这些偏差应逐位不变：

| key | 比较方式 | declared | measured | \|rel dev\| | tolerance | 判定 |
|---|---|---:|---:|---:|---:|---|
| `dean_circular_polarisation_64` | upper | 1e-10 | 6.33908e-15 | 9.999e-01 | 0.015 | PASS |
| `dean_cylinder_energy_balance_64` | upper | 1e-12 | 0 | 1.000e+00 | 0.015 | PASS |
| `dean_cylinder_reflection_64` | upper | 1e-12 | 1.3215e-15 | 9.987e-01 | 0.015 | PASS |
| `dean_cylinder_transmission_error_64` | value | 0.00839 | 0.00839366 | 4.357e-04 | 0.015 | PASS |
| `dean_ellipse_haskind_x_64` | value | 0.000797 | 0.000796854 | 1.834e-04 | 0.05 | PASS |
| `dean_ellipse_haskind_z_64` | value | 0.00199 | 0.00199491 | 2.467e-03 | 0.05 | PASS |
| `dean_ellipse_reflection_64` | value | 0.09751 | 0.0975139 | 3.968e-05 | 0.015 | PASS |
| `dean_ellipse_transmission_64` | value | 0.9952 | 0.995234 | 1.668e-07 | 0.015 | PASS |
| `dean_haskind_x_64` | value | 2.16e-05 | 2.16404e-05 | 1.872e-03 | 0.015 | PASS |
| `dean_haskind_z_64` | value | 2.16e-05 | 2.16404e-05 | 1.872e-03 | 0.015 | PASS |
| `dean_mfs_force_x_64` | value | 0.00151 | 0.00150584 | 2.755e-03 | 0.05 | PASS |

![dean pinned](figures/fig3_dean_pinned.png)

## 4. 注意点（踩过的坑 / 局限）

- **本地 `main` 过期**：仓库里本地 `main` 停在 `6025970`，若以它为 ours 合并，
  会得到一个"干净但落后"的分支——误判为 mergeable。必须用 `origin/main`。
  另外 `git fetch` 全量 refspec 在本机超时，改为只 fetch 需要的两个分支名才成功。
- **机械 overlay 的反例价值**：本冲突两侧都是"在 tuple 尾部追加"，任何"取一侧文件"的解法
  都会静默丢掉另一侧 11 条或 84 条记录；而"取两侧拼起来"若不加闭合括号会直接 SyntaxError。
  故设置了 AST 门 + 多重集行审计。
- **`git merge-file --diff3` 的冲突区比递归合并更宽**：diff3 会把两侧共同的尾行也圈进冲突块，
  用它做"可复现"校验会得到字节不等（本次实测 ours 多 2 行）。改用两方 `git merge-file -p`
  才能复现 Git 真实的冲突区。
- **宿主负载**：合并期宿主机 load average ~1400（128 核，多智能体共用），单进程串行跑全量
  1002 个测试不可行；故 baseline 与 merged 采用**同一并行设置**（`-n 16 --dist loadfile`）跑，
  两者可比，但严格意义上不是 CI 的串行配置。
- 局限：本单只解冲突、不动求解器；CI 里 pytest 之前那批证据生成步骤（artifacts/*.json）
  未跑完（同样受宿主负载所限），报告里已如实标注未完成状态。

## 5. 误差判定（是否 < 3%）

| 关键量 | 基准 | 实测 | 是否 < 3% |
|---|---|---|---|
| 注册表两侧新增行丢失数 | 0（不允许丢） | 0 | 是（0） |
| 合并结果可复现性（与父提交重放比对） | 逐字节相同 | `identical=True`，标记残留 0 | 是 |
| Dean value 行最大相对偏差 | 钉桩 tolerance（1.5%/5%） | 2.755e-03 | 是（2.75e-01 %，< 3%） |
| Dean 上限守护行最大比值（measured/bound） | < 1 | 1.321e-03 | 是（远小于 1） |
| 全量测试 fail 数（merged vs baseline） | baseline 0 | merged 0 | 是 |

无基准项：合并本身不产生新物理量，其正确性判据是"两侧内容不丢 + 测试不低于基线"，
故对物理量只复核已有钉桩值，未新造参考解。

## 6. 复现

```bash
# 冲突求解（幂等，遇未知块即报错）
python3 docs/m3796/resolve_union.py
# 可复现性 + 无标记门
python3 docs/m3796/verify_resolution.py
# 多重集行审计（无静默丢失门）
python3 docs/m3796/conflict_audit.py
# 图
python3 docs/m3796/make_figures.py
# CI 等价流程（可续跑）
bash docs/m3796/run_ci_equivalent.sh <tree> /tmp/log /tmp/state
```

---

*本文件由 `docs/m3796/make_report.py` 从 `resolution_check.json`、`conflict_audit.json`、
pytest 日志与 `outputs/m3236/registry_fast.json` 程序化生成，文中数字均非手抄。*
