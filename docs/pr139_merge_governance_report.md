# TensorBEM PR #139 合并治理报告

**任务**：m3904 · **仓**：TensorBEM · **分支**：`agent/m3904`  
**PR**：#139 (`agent/m3306` → `main`)  
**旧 SHA**：`06397b42c1d3b81c244e12ffabb9d866a792e856`（agent/m3306 HEAD）  
**新 SHA**：同上（合并为 fast-forward，无新提交）

---

## 1. 案例介绍

**物理背景**：三维自由液面边界元法（BEM）基准验证，涵盖 Hulme (1982) 半球绕射、Ka=2.58 不规则频率 CHIEF 正则化、Wang/Linton 潜没球绕射三大经典案例。

**冲突根源**：PR #139 将 `docs/marine_bem_benchmarks.md` 表 10-12 的数字登记入 `tensorbem.benchmarks.declared_numbers`。PR 等待期间，main 上由 m3150/m3280 独立长出了同名不同键、数值更细的注册表。冲突发生在同一个 `DECLARED_NUMBERS` 元组的相邻位置，机械 overlay 会导致另一侧的行静默消失。

## 2. 解冲方法（前序 agent/m3306 完成）

前序 agent/m3306 在提交 `06397b4` 已完成三方合并解冲：

| 项 | 数值 |
|---|---:|
| PR 侧注册键 | 55 |
| 其中 PR 新增 | 33 |
| 数值映射 | 32 |
| 符号型断言 | 1 |
| 合并后注册键总数 | 86 |
| 声明值最大相对偏差 | 1.85% |
| 判定 | **PASS** |

解冲策略：取 main 的更完整注册表结构 + 保留 PR 的独有断言（表 11 解析锚点 exact_a33/exact_b33）。PR 的一次性改写脚本（patch_registry.py 等）被清理，其功能已由 merge_audit.py 以只读方式覆盖。

## 3. 本任务验证结果

### 3.1 合并状态

- 合并类型：**fast-forward**（agent/m3306 已包含 origin/main 全部提交）
- 冲突文件：**0**（前序解冲已处理）
- 变更文件：20 个（+2389 行，-7 行）

### 3.2 核心测试

| 测试 | 结果 |
|---|---:|
| test_registry_is_well_formed | ✅ PASS |
| test_registry_covers_every_documented_benchmark_family | ✅ PASS |
| test_capability_evidence (11 tests) | ✅ PASS |
| test_doc_claims (14 tests) | ✅ PASS |

### 3.3 注册表完整性

- 注册表条目数：86
- 必需组全部覆盖：cylinder, ellipsoid, hulme_chief, hulme_hemisphere, interior_harmonics, submerged_sphere, ursell, ursell_reference
- 关键键存在：hulme_hemisphere_256_heave_added_mass_error, hulme_chief_2_58_exact_a33, submerged_sphere_320_heave_added_mass_error

## 4. 冲突文件清单与解法

| 文件 | 冲突类型 | 解法 |
|---|---|---|
| src/tensorbem/benchmarks/declared_numbers.py | 同元组内键名冲突 | 取 main 的键名体系，保留 PR 的 exact_a33/exact_b33 解析锚点 |
| tests/test_declared_numbers.py | 测试断言更新 | 合并双方测试，添加 standard_report fixture 共享标准层扫描 |
| docs/marine_bem_benchmarks.md | 文档段落重叠 | 取 main 的更完整版本，PR 的澄清段落被同义覆盖 |

## 5. 误差判定

| 关键量 | PR 声明值 | 合并树声明值 | 相对偏差 | 是否<3% |
|---|---:|---:|---:|---:|
| hulme_order_b11_256 | 1.94 | 1.9377 | 0.12% | ✅ |
| submerged_sphere_rotational_coupling_320 | 2.6e-5 | 2.6481e-5 | 1.85% | ✅ |
| hulme_chief_2_58_exact_a33 | 0.4007 | 0.4007 | 0.00% | ✅ |
| hulme_chief_2_58_exact_b33 | 0.0631 | 0.0631 | 0.00% | ✅ |

所有关键量偏差均 <3%，合并验证通过。

## 6. 局限

- 本验证基于本地 git 状态（origin/main @ ecba798），未远程 fetch 最新 main（网络超时）
- 重型测试（TENSORBEM_HEAVY_TESTS）因超时未运行，仅验证了注册表结构与快速测试
- PR 合并状态以 GitHub 为准，本地验证仅确认合并树完整性
