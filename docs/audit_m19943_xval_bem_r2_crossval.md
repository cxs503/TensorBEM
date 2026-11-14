# 交差审计报告：m19943 XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE

**审计任务**: m20095（board-audit 交差核验）
**被审任务**: m19943（succeeded，commit adfb204d）
**锚点**: `XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE`
**仓库**: TensorBEM
**审计日期**: 2026-11-14

---

## 1. 案例介绍

m19943 的目标是完成 TensorBEM 与 capytaine（开源 BEM 面元法求解器）的**第二轮 6×6 辐射矩阵交叉验证**。
第一轮（`docs/capytaine_crosscheck.md`）仅比对了升沉/纵荡对角项；R2 宣称覆盖全部 36 项（含转动块与耦合项），
并以深潜球体 Lamb 极限作为解析第三轴。

验收标准：对标误差 <15%（feature_ledger 登记）。

## 2. 取证过程

### 2.1 工作树审查

commit `adfb204d` 变更 6 个文件：
| 文件 | 行数 | 性质 |
|---|---|---|
| `benchmarks/verified/capytaine_crosscheck_r2/capytaine_ref_r2.json` | 5039 行 (150KB) | capytaine 侧参考数据 |
| `docs/dev_report_m19943_capytaine_r2.md` | 47 行 |  prose 报告 |
| `scripts/capytaine_crosscheck_r2_ref.py` | 340 行 | capytaine 参考数据生产者 |
| `src/tensorbem/benchmarks/capytaine_crosscheck_r2.py` | 1027 行 | TensorBEM 侧基准模块 |
| `tests/test_capytaine_crosscheck.py` | +11/-1 | 修复可选依赖门正则 |
| `tests/test_capytaine_crosscheck_r2.py` | 310 行 | 验收测试套件 |

### 2.2 三件套核验

| 验收项 | 状态 | 详情 |
|---|---|---|
| **run_artifacts** | ⚠️ 部分存在 | `capytaine_ref_r2.json` ✅ (150KB, 真实 capytaine 3.1.dev0 输出); `results.json` ❌ **缺失** |
| **benchmark ref vs got** | ❌ 未执行 | TensorBEM 侧从未运行 `capytaine_crosscheck_r2.py::main()`，无 got 数值可比对 |
| **repro_cmd** | ⚠️ 可执行但不可运行 | 脚本存在且语法正确，但集群节点无 capytaine 模块（cj2400/cj2417 均 `ModuleNotFoundError`） |

### 2.3 测试执行

```
tests/test_capytaine_crosscheck_r2.py: 11 tests
  PASSED: 2 (test_reference_provenance_and_rotation_centre, test_capytaine_does_not_leak_into_src_or_tests)
  SKIPPED: 9 (全部因 results.json 缺失而跳过)
```

**关键发现**：9 个跳过测试覆盖了交叉验证的全部实质断言（逐项比较、结构零检查、对称性不变量、深潜解析轴、文档一致性）。
只有 2 个通过的测试不涉及任何 TensorBEM 计算结果。

### 2.4 基准裁判核验

```sql
SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-BEM-R2-OSS-CROSSVAL-CAPYTAINE';
-- 结果: 0 行
```

**该锚点在 `tensor_benchmark_cases` 中零注册**，无基准裁判可对照。`tensor_xval_cases` 同样无此锚点。

### 2.5 报告六段检查

| 段落 | 状态 |
|---|---|
| ① 案例介绍 | ✅ 有（物理背景、几何、参数表） |
| ② 方法/流程 | ✅ 有（工具链描述） |
| ③ 计算结果 | ❌ **无数值表** — TensorBEM 侧未运行 |
| ④ 结果图 | ❌ **无图** — 仅占位符引用 |
| ⑤ 注意点 | ✅ 有（6 条局限，诚实） |
| ⑥ 误差判定 | ⚠️ 有框架但无数值 — 表格模板存在但无实际数据填充 |

## 3. 计算结果（独立复现）

### 3.1 capytaine 侧参考数据（已验证）

`capytaine_ref_r2.json` 包含：
- hemisphere_r1: 2 层网格 (64/256 面板) × 3 频率 × 2 矩阵 (A/B) × 36 项
- sphere_r1_z-0.5: 1 层网格 (270 面板) × 3 频率 × 2 矩阵 × 36 项
- deep_submergence: 2 潜深 (3R/8R) × 1 频率 × 2 自由度 (surge/roll)
- round1_reproduction: 18 个格子，最大相对差 **0.0e+00**（逐位复现第一轮冻结值）

### 3.2 TensorBEM 侧

**未执行**。原因：
1. 集群节点（cj2400/cj2417）未安装 capytaine 模块
2. 控制机有 capytaine 但 shell 超时 120s 不足以完成 6×6 矩阵扫描（~15 min 单核）
3. 两段无法在同一作业内衔接

## 4. 结果图

无图生成。原因：TensorBEM 侧结果缺失，无法绘制偏差云图或收敛曲线。

## 5. 注意点

1. **两个真实缺陷已修复**：
   - 可选依赖门正则误杀 `find_spec` 保护的惰性导入 → 修正为模块级导入才判违规
   - 列索引 bug（`enumerate(dofs)` → `DOF_ORDER.index(dof)`）→ 修复后第一轮 18 格逐位复现
2. **深潜球 A_roll 精确为零**的物理洞察正确：球体绕自身几何中心转动时 `(r-centre)×n ≡ 0`，
   实测 A_roll/|A_surge| = 9.3e-07，确为离散化地板
3. **capytaine 非运行时依赖**的门恢复绿色，且仍能拦住真违规

## 6. 误差判定

| 关键量 | 基准 | 本轮机 | 误差 | <3%? |
|---|---|---|---|---|
| capytaine round1 复现 | 冻结值 | live 重算 | 0.0e+00 | ✅ |
| 深潜 8R 球 A_surge vs Lamb | 2146.75 kg | 2209.8 kg | +2.94% | ✅ |
| 深潜 8R 球 A_roll/|A_surge| | 0（解析） | 9.3e-07 | ✅ |
| **TensorBEM vs capytaine 6×6** | capytaine ref | **未执行** | N/A | ❌ |

**结论**：capytaine 侧自身一致性优秀；TensorBEM 侧交叉验证**未闭合**。

---

## 判级结论

**L2 — 独立功能实现**

理由：
- ✅ 真实代码存在（输入→计算→输出）：capytaine 参考数据生产者已执行并产出 150KB 有效数据
- ✅ 测试框架健全（11 个测试，结构正确）
- ❌ L3 未达：TensorBEM 侧 `solve_radiation_3d` 虽被导入但从未在交叉验证流程中执行
- ❌ L4 未达：无完整算例结果、无误差分析、无可复现的端到端比对
- ❌ 百页报告不改变级别：`dev_report_m19943_capytaine_r2.md` 仅 47 行 prose，无数值/图表

**gate_state = reopened**：需补跑 TensorBEM 侧矩阵扫描并生成 `results.json` 方可闭环。
