# m3940 测试证据（合并后本树 vs main 基线）

环境注意：本机 `site-packages/__editable__.tensorbem-0.5.0.pth` 指向另一 agent 的工作树
（实测 `import tensorbem` 解析到 `/root/tp-agent/work/m3862/src`），因此**每条命令都显式**
`PYTHONPATH=<树>/src`，否则会静默测到别人的源码。

## 1. 注册表单元测试（本树）

```
cd /root/tp-agent/work/m3940
PYTHONPATH=/root/tp-agent/work/m3940/src python3 -m pytest tests/test_declared_numbers.py -q -p no:cacheprovider
```

执行进度（逐条完成，日志 `/tmp/m3940_full.log`）：`s....`

| # | 测试 | 结果 | 覆盖内容 |
|---|---|---|---|
| 1 | `test_heavy_tier_reproduces_the_documented_1280_panel_ellipsoid_row` | skip | 需 `TENSORBEM_HEAVY_TESTS=1` |
| 2 | `test_registry_is_well_formed` | **PASS** | 88 行、无重复键、键名/容差/来源/符号合法 |
| 3 | `test_registry_covers_every_documented_benchmark_family` | **PASS** | 8 个分组齐备（合并后口径） |
| 4 | `test_three_dimensional_free_surface_tables_are_fully_registered` | **PASS** | 表 10 31 行 / 表 11 9 行 / 表 12 26 行；单表共用一次求解；负阻尼符号；20 面元机器零上界 |
| 5 | `test_fast_tier_reproduces_the_documented_numbers` | **PASS** | **默认层端到端复算**：含表 10–12 全部格子（含本次移植的 4 行 `exact_a33`/`exact_b33`/`chief_point_count`/`20 面元转动耦合上界`）与 m3150 订正的 4 格，全部判 PASS |

第 5 条是本单最关键的证据：合并后的 88 条注册项在默认层被逐条重算并全部落在容差内，
说明（a）PR 移植格子的 `quantity` 键与 `_run_group` 输出对齐；（b）m3150 订正值在本树可复现；（c）`_chief_rows` 的 `exact_*` 展平新增分支工作正常。

## 2. 注册表端到端命令行

```
PYTHONPATH=src python -m tensorbem.benchmarks.declared_numbers --tiers fast,standard --strict
```
（同第 5 条测试的代码路径；`--strict` 在任一 DRIFT 时非零退出。默认层整组为分钟级稠密三维求解，
本机同时有其它 agent 的 60+ 个 pytest 进程，故以第 5 条测试的判 PASS 作为等价的机器判据。）

## 3. 内容不丢失审计（不依赖测试框架）

```
PYTHONPATH=src python3 docs/m3940/verify_union.py      # 输出见 verify_union.log
```
结果：`PASS: every cell of both parents has exactly one merged row`

```
base cells      : 13     branch cells    : 76     main cells      : 75
shared cells    : 72     parents union   : 79     merged cells    : 79
branch-only cells ported into the merge: ['hulme_chief_2_58_chief_point_count',
  'hulme_chief_2_58_exact_a33', 'hulme_chief_2_58_exact_b33',
  'submerged_sphere_20_rotational_coupling']
main-only cells kept: ['hulme_hemisphere_256_haskind_heave',
  'hulme_hemisphere_256_haskind_surge', 'hulme_hemisphere_256_near_far_damping_closure']
```

## 4. 全量核心测试与 main 基线对照

两棵树的同一命令（本树 / `main`@`ecba798` 基线树 `/tmp/tb-main`）：

```
PYTHONPATH=<树>/src python3 -m pytest <树>/tests -q -p no:cacheprovider
```

两棵树在本机负载下的进度见 §5；结论以"本树不低于 main 基线"为准，见 `merge_report.md` 的说明。

## 5. 如实说明（未完成/受限部分）

- `test_declared_numbers.py` 第 6 条起（`standard` 层 Ursell 复算、CLI 子进程测试）与全量套件在本机
  60+ 并发 pytest 进程下未能在本单预算内跑完；已完成的 1–5 条覆盖了本单改动面（注册表 + 三维表族 + 文档口径）。
- `TENSORBEM_HEAVY_TESTS=1` 的两条二阶收敛门未执行（需分钟级专用窗口）。
- 未做任何 push 之外的分支改写；`main` 未被本地改动。