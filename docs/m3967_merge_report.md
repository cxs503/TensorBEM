# PR #128 (agent/m3181) 合并治愈报告

## 任务概述
PR #128（分支 agent/m3181）对 main 变脏（mergeable=false），需三方合并解冲。

## 冲突文件清单（2 文件，5 处冲突）

### 1. src/tensorbem/benchmarks/declared_numbers.py（3 处冲突）

| 冲突行 | HEAD (main) 内容 | agent/m3181 内容 | 解法 |
|--------|-----------------|-----------------|------|
| 49-57 | `floating_hemisphere as hemisphere_benchmark`, `submerged_sphere_wave as sphere_benchmark` | `floating_hemisphere as hulme_benchmark`, `submerged_sphere_wave as wang_linton_benchmark` | 取 agent/m3181（命名更精确，引用原始文献作者） |
| 540-1136 | 60+ 条独立 DeclaredNumber 条目（逐行硬编码） | `) + _free_surface_3d_rows()`（程序化生成） | 取 agent/m3181（可维护性更高，逻辑等价） |
| 1245-1318 | `run_group` 中 `hulme_hemisphere`/`submerged_sphere`/`hulme_chief` 分支 | `run_group` 中 `hulme3d`/`wang_linton3d`/`hulme3d_chief` 分支 | 取 agent/m3181（与新组名一致） |

### 2. tests/test_declared_numbers.py（2 处冲突）

| 冲突行 | HEAD (main) 内容 | agent/m3181 内容 | 解法 |
|--------|-----------------|-----------------|------|
| 81-88 | 简短注释 | 更详细的注释说明负阻尼的物理意义 | 取 agent/m3181 |
| 98-104 | 期望组名 `hulme_chief`, `hulme_hemisphere` | 期望组名 `hulme3d`, `hulme3d_chief`, `wang_linton3d` | 取 agent/m3181（与实际实现一致） |

## 解冲策略
采用 **agent/m3181 侧为基线** 的语义并集策略：
- agent/m3181 的代码是 main 侧的**重构升级版**（程序化生成替代硬编码，命名更精确）
- 所有 main 侧的声明数值均通过 `_free_surface_3d_rows()` 函数等价保留
- 无内容丢失：63 条 3-D 自由面声明数 + 22 条其他基准 = 85 条完整注册

## SHA 记录
- agent/m3181 旧 SHA: `00f53e1013e6cc6bd87b9ea554cb7f7a1e3ddc4f`
- agent/m3967 新 SHA: `fe4e9bbf0433385a256ad1f710e1999c6e30eead`

## 测试证据

### 测试环境
- Python 3.11.6, pytest 9.2.0
- tensorbem-0.5.0 (pip install -e .)

### 测试结果
```
12 passed, 4 skipped (heavy tests require TENSORBEM_HEAVY_TESTS=1), 0 failures
```

| 测试用例 | 结果 |
|----------|------|
| test_registry_is_well_formed | ✅ PASS |
| test_registry_covers_every_documented_benchmark_family | ✅ PASS |
| test_three_dimensional_free_surface_tables_are_fully_registered | ✅ PASS |
| test_fast_tier_reproduces_the_documented_numbers | ✅ PASS |
| test_report_is_strict_json | ✅ PASS |
| test_a_perturbed_measurement_is_reported_as_drift | ✅ PASS |
| test_upper_bound_claims_only_fail_when_exceeded | ✅ PASS |
| test_unknown_tier_is_rejected | ✅ PASS |
| test_repeated_quantities_share_one_solve | ✅ PASS |
| test_richardson_limit_recovers_a_first_order_family | ✅ PASS |
| test_richardson_limit_rejects_a_non_converging_family | ✅ PASS |
| test_cli_writes_strict_json_and_passes | ✅ PASS |

### 语法验证
- `python3 -m py_compile src/tensorbem/benchmarks/declared_numbers.py` ✅
- `python3 -m py_compile tests/test_declared_numbers.py` ✅

## 注意事项
1. 推送被安全策略拦截，平台将自动推送分支到 GitHub
2. 重型测试（heavy tier）需设置 `TENSORBEM_HEAVY_TESTS=1` 才运行，本次未执行（耗时数分钟）
3. 合并后 DECLARED_NUMBERS 共 85 条，覆盖 8 个基准族
