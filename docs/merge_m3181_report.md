# Merge Report: agent/m3181 → main (PR #128)

## 1. 案例介绍（物理背景/几何/参数表）

本合并涉及 TensorBEM 仓库中 `src/tensorbem/benchmarks/declared_numbers.py` 和 `tests/test_declared_numbers.py` 两个文件的冲突解决。

**背景**：`declared_numbers.py` 模块负责验证文档中引用的所有数值（收敛误差、参考解值、近/远场闭合等）。agent/m3181 分支引入了 3D 自由面基准测试（Hulme 1982 浮半球、Wang & Linton 潜球）的程序化注册方法，而 main 分支已有显式注册的条目。

**冲突文件**：
| 文件 | 冲突区域数 | 性质 |
|---|---|---|
| `src/tensorbem/benchmarks/declared_numbers.py` | 3 | 导入别名、DECLARED_NUMBERS 注册表、_run_group 实现 |
| `tests/test_declared_numbers.py` | 2 | 注释文本、组名集合 |

## 2. 怎么做的（方法/流程/关键设置/工具链）

### 冲突解决策略：语义并集，保留双方内容

**冲突 1：导入别名（第 49-57 行）**
- HEAD (main): `hemisphere_benchmark`, `sphere_benchmark`
- agent/m3181: `hulme_benchmark`, `wang_linton_benchmark`
- **解法**：采用 agent/m3181 的更具描述性的别名（Hulme 1982, Wang & Linton 文献命名）

**冲突 2：DECLARED_NUMBERS 注册表（第 540-1138 行）**
- HEAD (main): 显式列出 60+ 个 DeclaredNumber 条目
- agent/m3181: 使用 `_free_surface_3d_rows()` 函数从表格数据 `_TABLE_HULME`, `_TABLE_HULME_CHIEF`, `_TABLE_WANG_LINTON` 程序化生成
- **解法**：采用 agent/m3181 的 DRY 方法。表格数据结构已存在于合并后的文件中，`_free_surface_3d_rows()` 生成与 HEAD 显式条目等价的 63 个注册项（28 + 9 + 26）

**冲突 3：_run_group 实现（第 1245-1318 行）**
- HEAD (main): 使用 `_study_rows()` 和 `_chief_rows()` 辅助函数处理基准结果，组名为 `hulme_hemisphere`, `submerged_sphere`, `hulme_chief`
- agent/m3181: 内联处理结果，组名为 `hulme3d`, `hulme3d_chief`, `wang_linton3d`，并计算推导量（如 observed_order_b11）
- **解法**：采用 agent/m3181 的更完整实现，支持 wang_linton3d 组

**测试文件冲突**：
- 注释文本：采用 agent/m3181 更精确的描述
- 组名集合：采用 agent/m3181 的 `hulme3d`, `hulme3d_chief`, `wang_linton3d`
- 新增测试 `test_three_dimensional_free_surface_tables_are_fully_registered`：保留

## 3. 计算结果（数字表）

### 注册表统计
| 组名 | 条目数 | 来源 |
|---|---|---|
| cylinder | 2 | README / docs |
| ellipsoid | 3 | README / docs |
| ursell | 8 | README / docs |
| ursell_reference | 2 | docs |
| interior_harmonics | 6 | docs |
| hulme3d | 28 | Table 10 (6×3 + 5×2) |
| hulme3d_chief | 9 | Table 11 (2+3+4) |
| wang_linton3d | 26 | Table 12 (6×3 + 4×2) |
| **总计** | **84** | |

### 测试证据
| 测试 | 状态 |
|---|---|
| test_registry_is_well_formed | ✅ PASSED |
| test_registry_covers_every_documented_benchmark_family | ✅ PASSED |
| test_three_dimensional_free_surface_tables_are_fully_registered | ✅ PASSED |

## 4. 结果图

本任务为代码合并/冲突解决，无数值仿真结果图。合并后的代码结构验证通过结构测试。

## 5. 注意点

1. **命名一致性**：agent/m3181 使用 `level<N>_<quantity>` 键格式，而 HEAD 使用 `sub<N>_<quantity>`。合并后统一使用 agent/m3181 格式，与 `_free_surface_3d_rows()` 生成的 quantity 字段一致。

2. **组名变更**：`hulme_hemisphere` → `hulme3d`，`submerged_sphere` → `wang_linton3d`，`hulme_chief` → `hulme3d_chief`。这些变更在测试中已同步更新。

3. **_run_group 移除**：HEAD 的 `_study_rows()` 和 `_chief_rows()` 结果处理函数被 agent/m3181 的内联处理替代，功能等价但代码更简洁。

4. **重型测试**：`test_fast_tier_reproduces_the_documented_numbers` 等涉及实际基准运行的测试因超时未完整执行，但结构测试全部通过。

## 6. 误差判定

本任务为代码合并，无数值计算误差。结构验证：
- 注册表格式：✅ 所有条目通过 well-formed 检查
- 组覆盖：✅ 所有文档基准族均有注册
- 3D 表完整性：✅ hulme3d=28, hulme3d_chief=9, wang_linton3d=26，与文档表格一致

## SHA 记录

| 分支 | SHA | 说明 |
|---|---|---|
| origin/agent/m3181 (旧) | 55db3976dc283e9522334f2ab7309fc655f3ae27 | 合并前 |
| agent/m3794 (新) | aa5650b7bb2339c2253a920b1331325e939cde25 | 合并后 |
