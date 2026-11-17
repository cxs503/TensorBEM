# 有限水深交叉验证收口：12 格全网格回填 + 状态语义修正（m24149）

**锚点**: `XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN`（capytaine 的 `finite_depth_crossval_plan` 做法）
**仓**: TensorBEM · **分支**: `agent/m24149`
**日期**: 2026-11-16
**状态**: 跨代码判定 COMPLETE（12 格 × 3 档齐）；确认档（第 4 档）与参考侧渐近性仍为诚实缺口

## 1. 干了什么

前序单（m22103/m22843/m23018/m23110/m23512）把计划、参考冻结、负控腿、无限水深脊都
做完了，但 12 格 TensorBEM 梯长期停在 PARTIAL（仅 2 格有 3 档）。本单做三件实质增量：

1. **回填集群作业 cj4594 的完整梯**：12 格 × 3 档（20/80/320 面板）全部落盘
   （`benchmarks/verified/capytaine_finite_depth/tensor_ladder.json`，全精度，替换
   m21322 的 3 位小数舍入版），用 `scripts/rebuild_finite_depth_verdict.py` 从冻结工件
   重判（不调求解器）。
2. **状态语义修正**：`execution_status` 现在只反映*主比较*覆盖（12 格 × 3 档）；
   确认档（4 个确认格的 1280 面板）拆到新的 `confirmation_status`，不再把主判定降级为
   PARTIAL。确认档缺失只影响 `extrapolation_self_consistent` 门（报 `worst: null`，
   不通过），不影响跨代码判定——这与计划里确认档的用途（测外推带）一致。
3. **测试语义修正**：`test_extrapolation_is_actually_used` 原来断言「参考在渐近区」
   （实测阶数 0.24 < 0.3，是假不变量）；改为断言*正确不变量*——被测侧（TensorBEM）
   必须外推（≥75%），参考侧阶数低于预登记下限时必须被**阻断外推**并记录截断界。
   另修 `test_crossval_decidability` 的预存导入失败（`scripts/` 不在 sys.path，改用
   importlib 按路径加载）。

## 2. 计算结果（程序化注入，零手抄）

跨代码最坏偏差（TensorBEM 外推 vs capytaine 最细档/截断界）：

| 量 | 最坏相对差 | 预登记带 | owner 线 |
|---|---:|---:|---:|
| heave added mass | +1.40 % | 3 % | 15 % |
| surge added mass | +1.66 % | 3 % | 15 % |
| heave damping | +3.38 % | 5 % | 15 % |
| surge damping | +2.15 % | 5 % | 15 % |

逐格（heave/surge × added mass/damping，%）：

| h/a | k0a | k0h | A33 | B33 | A11 | B11 |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.50 | 5.00 | -0.29 | +3.15 | -0.90 | +2.15 |
| 10 | 1.00 | 10.00 | -1.14 | +2.33 | +1.53 | +1.99 |
| 10 | 2.00 | 20.00 | -1.40 | +0.48 | +0.98 | +0.61 |
| 20 | 0.50 | 10.00 | -0.29 | +3.14 | -0.90 | +2.15 |
| 20 | 1.00 | 20.00 | -1.15 | +2.32 | +1.53 | +1.99 |
| 20 | 2.00 | 40.00 | -1.40 | +0.48 | +0.98 | +0.61 |
| 3 | 0.50 | 1.50 | -0.03 | +3.38 | -0.82 | +2.13 |
| 3 | 1.00 | 3.00 | -0.78 | +2.66 | +1.66 | +1.95 |
| 3 | 2.00 | 6.00 | -1.20 | +0.51 | +1.19 | +0.55 |
| 5 | 0.50 | 2.50 | -0.22 | +3.24 | -0.91 | +2.14 |
| 5 | 1.00 | 5.00 | -1.12 | +2.36 | +1.53 | +1.99 |
| 5 | 2.00 | 10.00 | -1.39 | +0.49 | +0.99 | +0.61 |

## 3. 误差判定（逐项，显式回答「是否 <3%」）

- **跨代码（本锚验收口径）**：最坏 +3.38 %（heave damping），
  **<15% owner 线：是**；<5% 阻尼带：是；<3% 附加质量带：是。
- **<3% 显式回答**：4 个量里 added mass 两量（+1.40 %/
  +1.66 %）<3%；damping 两量（+3.38 %/
  +2.15 %）超 3% 但在 5% 预登记带内。**整体 <3%：否（damping 超）**，
  但全部在预登记带与 owner 线内。
- **参考侧（诚实缺口，不粉饰）**：`reference_ladder_in_asymptotic_range` FAIL
  （capytaine 常数面板族实测阶数 0.24 < 0.3）、`reference_truncation_within_share`
  FAIL（末档变化 1.81% > 1.67% 份额）、`extrapolation_self_consistent` FAIL
  （无 4 档，外推带未知≠0）。这是**参考自身**的收敛局限，不是 TensorBEM 缺陷；
  研究侧已按预登记规则阻断参考外推、用截断界比较。

## 4. 门禁终态

| 门 | 判定 |
|---|---|
| added_mass_agreement | PASS |
| damping_agreement | PASS |
| negative_control_separates_depth | PASS |
| tensorbem_ladder_in_asymptotic_range | PASS |
| linear_solve_residuals | PASS |
| reference_is_finite_depth | PASS |
| reference_ladder_in_asymptotic_range | **FAIL**（参考非渐近，0.24） |
| reference_truncation_within_share | **FAIL**（1.81% > 1.67%） |
| extrapolation_self_consistent | **FAIL**（无 4 档，带未知） |

## 5. 注意点与局限

1. **确认档（1280 面板 × 4 格）未跑**：`EXTRA_LEVEL=1` 需数小时，本单未起；
   `confirmation_status` 如实报 PARTIAL，`extrapolation_self_consistent` 保持 FAIL。
2. **参考侧收口路径**（m23018 已指认）：capytaine 常数面板族阶数 0.24，加第 4 档
   需 ~5 万面板（稠密 LU 不可行）；cj4526 探针因集群节点无 capytaine 而失败
   （`ModuleNotFoundError`），需在控制主机重跑或改用有限水深多极子解析锚。
3. **本单未动求解器**：只动状态语义、测试语义与冻结工件回填。

## 6. 复现

```bash
PYTHONPATH=src python3 scripts/rebuild_finite_depth_verdict.py
PYTHONPATH=src python3 -m pytest tests/test_capytaine_finite_depth_crossval.py \
    tests/test_finite_depth_crossval_m23038.py tests/test_crossval_decidability.py \
    tests/test_xval_bem_finite_depth_reference_grid.py -q   # -> 63 passed
```

## 7. 给下一次自找活的线索

1. **有限水深多极子解析锚**（本锚真正收口路径）：仓内只有深水多极子
   （`tensorbem.analytic.submerged_sphere_wave_multipole`，仅 h/a=20 近似深水）；
   实现有限水深 John 本征级数球体多极子解，可把参考侧 3 个 FAIL 门换成对精确锚的判定。
2. **确认档回填**：`EXTRA_LEVEL=1 bash scripts/run_capytaine_finite_depth_xval.sh`
   起 4 格 1280 面板，回填后 `extrapolation_self_consistent` 可判。
3. **cj4526 探针在控制主机重跑**（集群无 capytaine）：直接测量 (96,192) 阶数是否回升。
