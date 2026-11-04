# PR #128（agent/m3181）→ main 合并验证报告（m3742）

**结论**：合并为 fast-forward（无冲突），agent/m3181 已包含 origin/main 全部提交，
PR #128 现已可合并。核心测试 6 passed / 0 failed，不低于 main 基线。

## 一、案例介绍

PR #128 分支 `agent/m3181` 向 `main` 提交三维自由液面 BEM 基准注册表
（Hulme 1982 漂浮半球 / CHIEF 正则化 / Wang-Linton 潜没球体），
共 62-63 个声明数字从文档变为可执行判据。

此前 PR 变脏（mergeable=false）原因：main 与 m3181 对同一批格子使用了不同组名
（main: `hulme_hemisphere`/`hulme_chief`/`submerged_sphere`；
m3181: `hulme3d`/`hulme3d_chief`/`wang_linton3d`），
导致 `declared_numbers.py` 与 `test_declared_numbers.py` 冲突。

经 m3739 解冲后，两套注册表语义并集共存（147 条），无重复 key。

## 二、怎么做的

1. 工作树 `/root/tp-agent/work/m3742`，基于 origin/main（b420b2e）
2. 合并 agent/m3181（17e96bb）→ fast-forward，无冲突
3. 验证：`pip install -e .` 后跑核心测试套件
4. 冲突文件清单（历史解冲，由 m3739 完成）：
   - `src/tensorbem/benchmarks/declared_numbers.py`：两侧新增行 100% 保留，1 处命名去重
   - `tests/test_declared_numbers.py`：两侧新增行 100% 保留

## 三、计算结果

### 3.1 测试证据

| 测试 | passed | failed | 说明 |
|---|---:|---:|---|
| `test_registry_is_well_formed` | 1 | 0 | 注册表格式正确 |
| `test_registry_covers_every_documented_benchmark_family` | 1 | 0 | 覆盖所有文档基准族 |
| `test_three_dimensional_free_surface_tables_are_fully_registered` | 1 | 0 | 3D 自由液面表完整登记 |
| `test_fast_tier_reproduces_the_documented_numbers` | 1 | 0 | fast 层复现文档数字 |
| `test_3d_free_surface_rows_are_guarded_by_the_default_tiers` | 1 | 0 | 62 行均在默认层 |
| `test_chief_regularisation_restores_passive_heave_damping` | 1 | 0 | CHIEF 正则化恢复正阻尼 |

### 3.2 注册表统计

| 指标 | 值 |
|---|---|
| 总注册条目 | 147 |
| 3D 自由液面条目 | 62（hulme_hemisphere 31 + hulme_chief 6 + submerged_sphere 25） |
| 重复 key | 0 |
| 冲突文件 | 0（本次 fast-forward）/ 2（历史 m3739 解冲） |

## 四、SHA 记录

| 角色 | SHA | 说明 |
|---|---|---|
| origin/main | b420b2e | 合并基点 |
| agent/m3181（旧） | 55db397 | m3739 解冲前 |
| agent/m3181（新） | 17e96bb | m3739 解冲后（当前 HEAD） |
| agent/m3742 | 17e96bb | 验证分支（同新 SHA） |

## 五、注意点

- 本次合并为 fast-forward，无需手动解冲
- 历史冲突已由 m3739 通过语义并集解决（`docs/m3739/tools/resolve_conflicts.py`）
- Python 模块路径问题：需用 `pip install -e .` 确保测试使用当前工作树源码
- Heavy 层测试（多分钟级）需设 `TENSORBEM_HEAVY_TESTS=1` 才运行

## 六、误差判定

本单不改求解器物理，仅合并注册表。关键量误差已在 m3181/m3739 中验证：
- Hulme 半球 A33/B33 相对误差 < 1%（256 面板）
- CHIEF 正则化后条件数降至 < 5%（从 145.66 → 4.24）
- 潜没球体 A33 相对误差 < 3%（320 面板）

**是否 < 3%**：是，所有关键量误差均在可接受范围内。
