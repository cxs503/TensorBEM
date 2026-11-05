# 合并验证证据（m3812 · PR #128）

机器：128 vCPU，运行期 load average ≈ 2.2×10³（同一时刻多个 worktree 在跑同族 3D 自由液面测试），
单测耗时约为空载的 10–20 倍；以下记录的命令与输出为准。

## 1. 合并前基线（origin/main）

```bash
git rev-parse origin/main            # ecba798  (合并基 4f5b73e)
git merge-base HEAD origin/agent/m3181   # 4f5b73e
git merge-tree 4f5b73e HEAD origin/agent/m3181   # 冲突：declared_numbers.py / test_declared_numbers.py
```

`origin/main` 侧 core 测试基线由 m3150 已入库的 `tests/test_declared_numbers.py` 定义（含
`test_standard_tier_reproduces_the_documented_ursell_numbers`、`test_3d_free_surface_rows_are_guarded_by_the_default_tiers`）。
合入后这些测试一行未删，只做增量。

## 2. 合并后结构面测试（本轮直接证据，0.53 s）

```bash
PYTHONPATH=src python3 -m pytest tests/test_declared_numbers.py -q \
  -k "well_formed or covers_every_documented or three_dimensional_free_surface_tables_are_fully_registered"
# 3 passed, 17 deselected in 0.53s
```

覆盖：注册表键唯一且非零、文档基准族齐全（含 `hulme_hemisphere`/`hulme_chief`/`submerged_sphere`）、
表 10/11/12 格子数 31/9/26、同表共用一次求解、CHIEF 阻尼符号、20 面元机器零格为上界。
即 `duplicate registry keys` 这一最易被机械 overlay 触发的门在合并后是绿的。

## 3. 合并后新增 4 格的数值复算（direct check）

```python
check(tiers=("standard",), entries=<4 new rows>)
hulme_chief_2_58_exact_a33         0.4007  0.400737  +9.33e-05  PASS
hulme_chief_2_58_exact_b33         0.0631  0.063122  +3.48e-04  PASS
hulme_chief_2_58_chief_point_count 5       5         +0.00e+00  PASS
# submerged_sphere 组（20/80/320 面元）在该机负载下 110 s 内未收敛完成，单独重跑待补
```

声明值来自文档表 11 表头（0.4007 / 0.0631）与 5 点 CHIEF 约束，测量值由合并后源码给出。

## 4. 未完成项（honest）

* `pytest tests` 全量套件与 `test_standard_tier_reproduces_the_documented_ursell_numbers` 的整轮数值回执
  在本轮预算内未跑完（机器负载 2.2×10³，单文件已跑 >20 min 仅完成 13/24 项，无明显失败）。
  已启动的运行见 `docs/m3812/outputs/`（若存在）与 `/tmp/m3812_all2.log`、`/tmp/dn2.log`。
* `submerged_sphere` 组（含新增的 20 面元上界格）的数值回执因此仍为 PENDING；
  结构断言（`kind == "upper"`、格子数 26）已绿。
