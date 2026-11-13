# 交差审计：m17248（CASE-BEM-RIGID-BODY-MASS·刚体质量矩阵·TensorBEM）

**Mission**: m17268（交差核验单）
**Date**: 2026-11-13
**Branch**: `agent/m17268`
**对象**: m17248（CASE-BEM-RIGID-BODY-MASS·算例池·刚体质量矩阵·TensorBEM，已 succeeded）
**Status**: ✅ 审计完成 — **verdict L0 / reopened**

## 1. 案例介绍（审计对象与背景）

本单是 **交差核验单**（非重做）：对 m17248（CASE-BEM-RIGID-BODY-MASS·算例池·刚体质量矩阵·TensorBEM）的
「已 succeeded」状态做独立取证与判级。m17248 被派发去跑 **刚体质量矩阵** 基准（ref=刚体质量矩阵闭式块形式，
容差 <3%），目标是产出可复现的 benchmark 验证（run_artifacts + ref vs got + repro_cmd）。

## 2. 怎么做的（方法/流程）

1. **取证**：查询 `tensor_dev_missions`（id=17248）的完整 result；查询 `tensor_benchmark_cases`
   （anchor_slug='CASE-BEM-RIGID-BODY-MASS'）；查询 `tensor_maturity_verdicts`（mission_id=17248）；
   查询 `tensor_reports`（mission_id=17248）；查询 `tensor_case_pool`（case_code='BEM-RIGID-BODY-MASS'）；
   检查 git 分支 `agent/m17248` 是否有 m17248 专属 commit。
2. **判级**：按 L0-L5 功能实质分级。
3. **落表**：`INSERT INTO tensor_maturity_verdicts`（board-audit）。

## 3. 计算结果（取证数字表，程序化注入）

| 取证项 | 结果 | 判定 |
|---|---|---|
| `tensor_dev_missions` id=17248 status | `succeeded`（但 **auto-approve**，`gate_valid=false`） | ⚠️ 假成功 |
| m17248 headline | `honest-FAIL（预算耗尽，两轮法接棒）` | ❌ 未完成 |
| m17248 fail_stage | `no_progress+forced_finish` | ❌ 无进展 |
| m17248 head_commit | `null`（无 commit） | ❌ 无提交 |
| `tensor_benchmark_cases` CASE-BEM-RIGID-BODY-MASS | **count=0**（不存在） | ❌ 无基准 |
| `tensor_maturity_verdicts` mission_id=17248 | **无行** | ❌ 无判定 |
| `tensor_reports` mission_id=17248 | **无行** | ❌ 无报告 |
| `tensor_case_pool` BEM-RIGID-BODY-MASS | `dispatched`→17248，`linked_anchor` 空 | ⚠️ 未链接 |
| m17248 changed_files | `_scratch_three_checks.py` + `run_m17248_rigid_body_mass.py`（未提交） | ❌ 无实质 |
| 底层能力 `compute_mass_matrix`（src/tensorbem/hydrostatic.py:497） | **REAL**（实测 max diff=0.0 vs 闭式块形式） | ✅ 真实（但非 m17248 交付） |

## 4. 结果图

纯审计/数据核验任务，无场量云图可绘——本单是**交差核验**（数据库行取证 + 判级落表），
非数值求解，无物理场量。理由：增量是「m17248 交付真伪判定」，不是「某算例的数值结果」。

## 5. 注意点（踩过的坑/局限）

- **auto-approve 陷阱**：m17248 被 `TP_AUTO_APPROVE_DEV_ACCEPTANCE` 自动批准为 succeeded，
  但 `server_quality_audit` 明确 `gate_valid=false, key_present=false, gate_missing_warning=true`——
  这是**假成功**，不是真验收。
- **honest-FAIL 与 succeeded 矛盾**：m17248 的 headline 是 honest-FAIL（预算耗尽），
  但 status 被标 succeeded——两轮法接棒后未真正交付。
- **无 commit 即无交付**：`head_commit=null`，分支 `agent/m17248` 指向 main（无 m17248 专属 commit），
  所有 claimed 工件（runs/m17248_rigid_body_mass/）均未提交、不可验证。
- **底层能力存在但非本单交付**：`compute_mass_matrix`（hydrostatic.py:497）是真实实现（REAL），
  但由更早任务（m273/m14564）交付，**不是** m17248 的成果——不能因底层能力存在而给 m17248 加分。

## 6. 误差判定

- **基准裁判**：`tensor_benchmark_cases` 中 **无** `CASE-BEM-RIGID-BODY-MASS` 锚（count=0），
  无 ref/got 数值可核，无容差可判——**基准未建立**，m17248 未完成 benchmark 验证。
- **报告六段**：`tensor_reports` 中 **无** m17248 报告——六段不齐全。
- **判级**：**L0**（无实质功能交付；m17248 未产出任何可验证的代码/工件/基准/报告）。
- **裁定**：**reopened**（m17248 的 succeeded 状态为假，需返修重跑基准）。
