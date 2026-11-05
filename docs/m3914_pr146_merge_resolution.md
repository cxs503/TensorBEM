# PR #146 合并治愈报告（m3914）

## 案例介绍
PR #146（分支 agent/m3365）对 main 变脏（mergeable=false），需要三方合并解冲。
- **物理背景**：TensorBEM 声明数字注册表（declared numbers registry）用于锁定文档中引用的基准测试数值，防止代码漂移。
- **冲突根源**：agent/m3365 与 main 在 `src/tensorbem/benchmarks/declared_numbers.py` 和 `tests/test_declared_numbers.py` 上各自添加了不同的 DeclaredNumber 条目和辅助函数，命名约定不同。

## 怎么做的
### 合并策略
1. **三方合并**：`git merge origin/main` 到 agent/m3365 基础
2. **冲突文件**：仅 2 个文件有冲突
   - `src/tensorbem/benchmarks/declared_numbers.py`（3 处冲突）
   - `tests/test_declared_numbers.py`（2 处冲突）
3. **语义并集**：
   - DeclaredNumber 条目：89（HEAD）+ 62（main 独有）= **151 条**
   - 导入别名：保留双方命名（`floating_hemisphere_benchmark` / `hemisphere_benchmark`）
   - 辅助函数：保留双方（`_level_quantities` / `_study_rows` / `_chief_rows` / `_floating_hemisphere_quantities`）
   - 测试组名：并集（`floating_hemisphere`, `hulme_chief`, `hulme_hemisphere`）

### 关键修复
- **submerged_sphere 处理器**：main 独有条目使用 `sub<level>_` 前缀命名（如 `sub0_condition_number`），而 HEAD 的 `_level_quantities` 产生 `@panel_count` 后缀。更新 `_run_group` 对 `submerged_sphere` 组使用 `_study_rows` 以匹配 main 风格键名。

## 计算结果
### 合并统计
| 指标 | 数值 |
|------|------|
| HEAD 独有条目 | 67 |
| main 独有条目 | 62 |
| 公共条目 | 22 |
| **合并后总条目** | **151** |
| 冲突文件数 | 2 |
| 解冲后新增行数 | ~800 |

### 测试证据
```bash
# 注册表结构验证
$ pytest tests/test_declared_numbers.py::test_registry_is_well_formed -v
PASSED

# 基准族覆盖验证
$ pytest tests/test_declared_numbers.py::test_registry_covers_every_documented_benchmark_family -v
PASSED

# 快速层数值复现
$ pytest tests/test_declared_numbers.py::test_fast_tier_reproduces_the_documented_numbers -v
PASSED (86.56s)
```

所有核心测试通过，注册表包含全部 9 个基准族：
- cylinder, ellipsoid, floating_hemisphere, hulme_chief, hulme_hemisphere
- interior_harmonics, submerged_sphere, ursell, ursell_reference

## 结果图
纯合并/后端任务，无数值计算结果图。测试通过即为验收证据。

## 注意点
1. **命名约定差异**：HEAD 和 main 对同一基准使用不同组名和键名格式，合并后需确保 `_run_group` 能处理所有变体。
2. **导入别名**：保留双别名（`floating_hemisphere_benchmark` / `hemisphere_benchmark`）以避免大规模重构。
3. **submerged_sphere 处理器**：必须使用 `_study_rows` 而非 `_level_quantities`，否则 main 独有条目的键名无法匹配。
4. **机械 overlay 风险**：本次合并手动解冲，未使用自动工具，确保双方内容无静默丢失。

## 误差判定
- **注册表完整性**：151 条 = 89（HEAD）+ 62（main 独有），零丢失 ✓
- **测试通过率**：3/3 核心测试通过 ✓
- **是否 <3%**：不适用（合并任务，无数值误差）

## 结论
PR #146 合并冲突已治愈，分支 agent/m3365 现可与 main 合并。等待 merge-governor 复验。
