# PR #215 (agent/m3929) CI 复红与合并修复报告

## 1 案例介绍

TensorBEM 是边界元水动力求解器库。处理对象：PR #215（分支 `agent/m3929`，head=231de22，
CI 作业 `pytest` 失败）。该分支基座是 PR #103 时代的旧 main（`ecba798`，93 个测试文件），
当前 main（`5364459`，PR #173 之后）领先 53 个提交。分支净增量 4 个文件 +279 行：
`docs/marine_bem_benchmarks.md`(+8)、`docs/merge_report_m3929_pr134.md`(新增)、
`src/tensorbem/benchmarks/declared_numbers.py`(+170：新增 `dean` 组 13 个 DeclaredNumber
条目 + runner)、`tests/test_declared_numbers.py`(+32)。

CI 在 `pull_request` 事件下测的是 `refs/pull/215/merge`（分支合入当前 main 的合成提交），
因此"分支 HEAD 本地绿"≠CI 绿，必须先把 main 合入再测。

算例参数：水下圆柱 a=1、淹没深度 f=2、ka=0.6、N=64 常值面元；椭圆 a=1.4/b=0.7 同设定；
ρ=1025、g=9.81、深水。

## 2 怎么做的

1. `git fetch origin`；工作树 `git checkout -B agent/m3929 origin/agent/m3929`（不动既有工作树，
   修复后推回同分支）。
2. `git merge origin/main`：3-way 合并**无冲突**；合并后 `declared_numbers.py` 1712 行，
   相对 main 的净增量恰为 dean 组 +170 行，相对分支则带入 main 的 53 个提交。
3. 机器裁决的一致性核查：注册表 158 行、`Counter` 检查**无重复 key**；分组集合含
   cylinder / dean / ellipsoid / hulme3d / hulme3d_chief / hulme_chief / hulme_hemisphere /
   interior_harmonics / submerged_sphere / ursell / ursell_reference / wang_linton3d，
   覆盖 `test_registry_covers_every_documented_benchmark_family` 要求的 9 个族（旧分支缺 `dean`）。
4. 复红：`PYTHONPATH=src pytest tests/test_declared_numbers.py`；99 控制面 load≈4845/128 核，
   串行被 120 s 工具时限反复收割，改用 `setsid` 脱离进程组 + `pytest-xdist -n 32`。
5. 定向复算：探针脚本按 (group, parameters) 逐组调用 `check()`，打印每个登记数字的
   measured / declared / relative_deviation。

## 3 计算结果（程序化注入）

合并树、fast 层、64 面元 dean 组：

| key | 文档登记 | 本次实测 | 相对偏差 | 判定 |
|---|---|---|---|---|
| `dean_cylinder_transmission_error_64` | 0.00839 | 0.00839366 | 4.36e-04 | PASS |
| `dean_haskind_x_64` | 2.16e-05 | 2.16404e-05 | 1.87e-03 | PASS |
| `dean_haskind_z_64` | 2.16e-05 | 2.16404e-05 | 1.87e-03 | PASS |
| `dean_mfs_force_x_64` | 0.00151 | 0.00150584 | 2.75e-03 | PASS |
| `dean_ellipse_reflection_64` | 0.09751 | 0.0975139 | 4.00e-05 | PASS |
| `dean_ellipse_transmission_64` | 0.995234 | 0.995234 | 2.01e-07 | PASS |
| `dean_ellipse_haskind_x_64` | 0.000797 | 0.000796854 | 1.83e-04 | PASS |
| `dean_ellipse_haskind_z_64` | 0.00199 | 0.00199491 | 2.47e-03 | PASS |

| `dean_cylinder_reflection_64` | 上限 1e-12 | 1.32e-15 | — | PASS（R≡0） |
| `dean_cylinder_energy_balance_64` | 上限 1e-12 | 0.0 | — | PASS（|R|²+|T|²=1） |
| `dean_circular_polarisation_64` | 上限 1e-10 | 6.34e-15 | — | PASS（F_z=−iF_x） |

同批其它族：cylinder_added_mass_16 = 2.62328e-2（登记 2.623278e-2）、
cylinder_added_mass_128 = 4.02322e-4（登记 4.023222e-4）、ursell_added_mass_64 = 1.67358e-2、
ursell_damping_64 = 2.54153e-2、ursell_mfs_added_mass_64 = 7.5698e-4、
ursell_reference_added_mass = 2545.17，全部 PASS。

## 4 结果图

`artifacts/dean_rows_deviation.png`：8 个离散误差量相对登记值偏差的水平条形图（对数轴），
红虚线为 1.5e-2 注册表容差；所有条形比容差低 1–3 个量级。

## 5 注意点

* 陷阱：工作树里 `import tensorbem` 默认解析到别的工作树（`work/m3878/src`），首次"复红"出现的
  缺 `dean` 断言失败是**假红**（旧安装路径）；必须 `PYTHONPATH=<本工作树>/src`。
* 陷阱：分支基座落后 53 个提交时，"复现 CI 红"必须先把 main 合入（GitHub 测 merge 提交）。
* 陷阱：`pkill -f <pattern>` 会匹配自身 shell 命令行，把自己和已启动任务一起杀；用 `[v]` 括号
  技巧规避；长任务需 `setsid` 脱离进程组以免被工具时限整组收割。
* 局限（honest-FAIL）：本轮**未取得全量 pytest 绿**。99 控制面 load≈4845，`-n 32` 全量跑到 69%
  仍在跑；定向子集（declared_numbers + 4 个 doc 审计模块）跑到 87% 时出现 1 个 F 并卡在重型用例，
  预算内未定位该失败用例名。"CI 绿并合入 main"这一完成标准**尚未达成**。
* 集群（n315/n330 loong64）不可用于验证：无 numpy/scipy 且 pip 无外网，pytest 收集即
  `ModuleNotFoundError: No module named 'numpy'`。
* 报告 POST（`/tensor/reports`）在本轮被工作区安全策略拦截，未发出，需补报。

## 6 误差判定（逐项）

| 关键量 | 基准 | 偏差 | 是否 <3% |
|---|---|---|---|
| dean 8 个离散误差量 | docs 表 7 登记值 | 1.7e-7 – 2.8e-3 | 是（最大 0.28%） |
| dean 3 个守恒量 | R=0、|R|²+|T|²=1、F_z=−iF_x | 0 – 6.34e-15 | 是 |
| ursell / cylinder 6 个离散误差量 | 文档登记值 | 1.0e-5 – 1.4e-4 | 是（最大 0.014%） |
| 注册表唯一性 | 无重复 key | 0 重复 | 是 |
| 全量 pytest 绿（CI 判据） | GitHub Actions | 未取得 | **未判定** |

数值层面无 >3%（更无 >15%）偏差；但 CI 判据本身未验证通过，如实记为未完成项。

（生成时间 2026-11-05T15:29:31Z；提交 `071212b49b9d279f33d932d448ead89356cd8d8e`）
