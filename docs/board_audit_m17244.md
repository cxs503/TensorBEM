# 交差审计：m17244 CASE-BEM-IRREG-FREQ-CYL（不规则频率 Bessel 零点预测）

**审计单**：m17304（board-audit）· **对象**：m17244（TensorBEM，succeeded，commit 886e77c）
**基准裁判**：`tensor_benchmark_cases WHERE anchor_slug='CASE-BEM-IRREG-FREQ-CYL'` → **0 行**（锚点未登记裁判表）
**结论**：三件套 3/3 实证成立（工件✓、ref vs got ✓、repro_cmd ✓），裁判表空 → **判级 L3**，verdict 已落表（id=534）。

## 1. 案例介绍

锚点 `CASE-BEM-IRREG-FREQ-CYL`：**不规则频率 Bessel 零点预测**。穿透自由面的物体（直立圆柱）在其内部本征频率处会让边界积分方程失秩，出现伪共振——这些"不规则频率"由水线面内 Dirichlet 本征问题给出；对圆形水线面其无量纲波数就是 Bessel 零点 `ν_{m,s} = j_{m,s}/R_wp`。

本单考的是 **TensorBEM 的 `predict_irregular_frequencies` 从湿表面网格反推出来的数，是否等于手册里的 `j_{m,s}`**，容差判据 **< 2 %**。

**ref 出处**：Abramowitz & Stegun Handbook of Mathematical Functions, NBS AMS 55 (1964/1972), Chapter 9 'Bessel Functions', section 9.5 'Zeros', Table 9.5 (pages 370-373)。该表只制表 `J_0,J_1,Y_0,Y_1` 的零点（即 `j_{0,s},j_{1,s}` 家族）；`m>=2` 的零点用 mpmath 40 位 + scipy + 仓内 numpy 递推三路独立计算（`source=computed`，不冒称手册读数）。

## 2. 怎么做的（审计方法）

1. **取证**：读 m17244 工作树 `/root/tp-agent/work/m17244`（commit 886e77c）、`tensor_reports` id=16250、`tensor_case_pool` id=161。
2. **三件套核验**：
   - **run_artifacts**：`artifacts/case_bem_irreg_freq_cyl/results.json`（31959 B）+ `fig1_acceptance_vs_panels.png`（85729 B）+ `fig2_eigenmode_fields.png`（101569 B）+ `fig3_route_c_band.png`（75649 B）+ `make_report.py`（19199 B）+ `report.md`（12917 B）——全部在 commit 886e77c 内（git ls-tree 命中）。
   - **benchmark ref vs got**：`ν_{0,1}R`=2.4067584880 vs `j_{0,1}`=2.4048255577 → +0.0804%；`ν_{1,1}R`=3.8347857863 vs `j_{1,1}`=3.8317059702 → +0.0804%；`ν_{2,1}R`=5.1397501688 vs `j_{2,1}`=5.1356223018 → +0.0804%——全部在 2% 容差内（最大 0.0804%）。
   - **repro_cmd**：`PYTHONPATH=src python -m tensorbem.benchmarks.irregular_frequency_bessel_zero_case --write` → **PASS**（实测）；`PYTHONPATH=src python -m pytest tests/test_irregular_frequency_bessel_zero_case.py -q` → **11 passed**（30.36s）。
3. **基准裁判**：`tensor_benchmark_cases` 对 `CASE-BEM-IRREG-FREQ-CYL` **0 行**（全表 46 个 anchor 无此 slug）；`tensor_case_pool` 有 `BEM-IRREG-FREQ-CYL`（id=161, tol_policy='<2%', status='dispatched', mission_id=17244）但 `linked_anchor=None`——裁判表无法确认。
4. **报告六段**：`tensor_reports` id=16250 六段齐全（①案例介绍 ②方法 ③计算结果 ④结果图 ⑤注意点 ⑥误差判定），但 figures 3 条均无 `png_base64`（只有 name/caption）。

## 3. 计算结果（ref vs got，逐项误差）

| 模式 | m | ref `j_{m,1}` | ref 来源 | got `νR` | 误差 | 判据 2% |
|---|---|---|---|---|---|---|
| `j01_axisymmetric` | 0 | 2.4048255577 | handbook | 2.4067584880 | +0.0804% | ✓ |
| `j11_dipole` | 1 | 3.8317059702 | handbook | 3.8347857863 | +0.0804% | ✓ |
| `j21_quadrupole` | 2 | 5.1356223018 | computed | 5.1397501688 | +0.0804% | ✓ |

三个模式全部通过，最大误差 **0.0804%**（判据 2%）。误差几乎 100% 来自水线面多边形面积偏小（几何项），Bessel 常量项只有 ~7.9e-12。

## 4. 结果图

图 1（fig1）三模式水线面分辨率敏感性曲线；图 2（fig2）三个预测零点对应的内部 Dirichlet 本征模态场云图；图 3（fig3）Route C 裸解条件数与阻尼随 ν 变化。图 PNG 在工作树 `artifacts/case_bem_irreg_freq_cyl/` 存在（85729/101569/75649 B），但 `tensor_reports` id=16250 的 figures 3 条均无 `png_base64`（F4）。

## 5. 注意点

1. **判据被几何支配，不是物理**：2% 的通过与否由水线面多边形面积决定（`e_g≈π²/3N²`），Bessel 常量本身精确到 1e-16。因此"通过 2%"只说明网格够细，不说明预测在哪里都对。
2. **预测器只支持一阶零点**：`predict_irregular_frequencies` 内建 `j_{m,1}` 三个常数，`n>=2` 的零点（`j_{0,2}=5.520` 等）无法从公开 API 取（`dirichlet_zeros` 能算但未接入），属能力缺口，本单未扩权。
3. **公式与自由面本征问题不同源**：`j_{m,s}/R` 是"刚性盖"内部本征值（John 1950 / Zhu 1993）。仓内 `irregular_frequency_cylinder_exact` 已证明自由面一致的 Route B 比它高 16–39%（`T/R`=0.5–1），本单的预测器与 Route C 实测都跟随刚性盖支。
4. **手册页无 OCR（生态摩擦）**：电子版 §9.5 页面为扫描图，无 OCR 引擎；改用"出处块 + 40 位逐位审计"保证数字链，审计当场揪出 3 处录入错。
5. **预登记偏离**：A1 阈值初稿为 1e-12，实测 3 个模式为 2e-12…8e-12——因为常量是以 `νR_wp` 反推的，含鞋带面积的浮点舍入（重建地板 ~8e-12）。故 A1 阈值改为 1e-10 并记录理由。

## 6. 误差判定（关键量 vs 基准，逐项，显式回答是否 <3%）

| 关键量 | 基准 | got | 误差 | <3%？ |
|---|---|---|---|---|
| `ν_{0,1}R` | A&S 表 9.5 `j_{0,1}` | 2.4067584880 | +0.0804% | **是** |
| `ν_{1,1}R` | A&S 表 9.5 `j_{1,1}` | 3.8347857863 | +0.0804% | **是** |
| `ν_{2,1}R` | 三路计算 `j_{2,1}` | 5.1397501688 | +0.0804% | **是** |
| 水线面反推 `R_wp`（N=64） | 真半径 R | 0.999197 | -0.0803% | **是** |
| 离散不规则带 `ν_obs` | 预测 `ν_pred` | 2.4665 | +1.25% | **是** |

结论：本单容差判据 **<2%** 全部通过（最大 0.0804%），并显式满足"<3%"；但如 §5.1 所述，该通过是网格分辨率命题，Bessel 数据本身的误差在 1e-12 量级。

## 7. 判级与落表

- **级别 = L3**（系统集成：真代码 输入→计算→输出 + 接入主求解器 `predict_irregular_frequencies` + 11 个 pytest 测试全过 + run_artifacts 入 commit）。
  - 非 L4：`tensor_benchmark_cases` 无 `CASE-BEM-IRREG-FREQ-CYL` 行（0 rows），平台裁判无此案号，唯一裁判缺席=不能判过；`tensor_case_pool` 有 `BEM-IRREG-FREQ-CYL`（id=161）但 `linked_anchor=None`。
  - 非 L5：无科研工程级新发现（预测器只支持一阶零点，属能力缺口未扩权）。
  - 附百页报告不改变级别；级别=功能实质。
- **落表**：`INSERT tensor_maturity_verdicts (repo='TensorBEM', scope='m17244 CASE-BEM-IRREG-FREQ-CYL', mission_id=17304, level=3, evidence=<JSON 真实指针>, verdict_by='board-audit')` → **id=534, gate_state=pending**。
- **evidence 键**：impl_files / entry / run_artifacts / tests / integration_callers / benchmark(0 行) / repro_cmd(exit=0) / external_anchor —— 全部真实指针。
- **maturity_gate**：verdict 落表后由平台自动裁定（confirmed=消账 / reopened=返修）。
