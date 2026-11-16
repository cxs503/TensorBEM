# XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN — 有限水深 capytaine 交叉验证收口（mission 22843）

**锚点**: `XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN`（feature_ledger status=open）
**目标**: 把 `tensorbem.finite_depth_solver3d` 的有限水深解对 **capytaine 有限水深 Green 函数** 做全量 12 格 code-to-code 交叉验证，并给出可判 <3% 的外推后偏差。

## 1. 案例背景（物理/几何/参数）

- 几何：全潜球 `a = 1.0 m`，球心 `z = -1.5 m`（最低点 `z = -2.5 m`）。球是唯一有独立半解析锚（Thorne/Wang/Linton 多极子）的光滑闭体，且对每个 `h/a >= 3` 都保持正离底。
- 网格：`h/a ∈ {3, 5, 10, 20}` × `k0 a ∈ {0.5, 1.0, 2.0}` → **12 格**，`k0 h ∈ [1.5, 40]`（从强浅水到有效深水）。
- 频率：`nu = k0 tanh(k0 h)` 逐格反解，保证 `k0 a` 在整条深度梯上精确（`tests/test_capytaine_finite_depth_crossval.py::test_plan_back_computes_nu_from_the_finite_depth_dispersion`）。
- 双码细化：TensorBEM 20/80/320（+4 格 1280 确认档）面板；capytaine (12,24)/(24,48)/(48,96) 面板。
- 预注册容差：added_mass 3%、damping 5%、owner 15%；外推带 ≤ 1/3 容差。

## 2. 方法/流程/工具链

- 计划模块：`tensorbem.benchmarks.finite_depth_crossval_plan`（网格/容差/负控/外推带全部预注册）。
- 参考侧：`scripts/capytaine_finite_depth_ref.py`（capytaine 3.1.dev0，`finite_depth_method=newer`，12 格 × 3 档 + legacy 公式，已分片合并）。
- 求解侧：`tensorbem.benchmarks.capytaine_finite_depth_crossval`（`--ladder-only --shard/--shards` 分片 → `--merge-caches` 合并 → `run_study` 出 verdict + gates + 文档）。
- 外推：Richardson 三档外推，报告观测阶与外推带；负控 = 深水盲解在同一网格上必须与小 `k0 h` 的有限水深解分离。

## 3. 当前状态（honest）

- **capytaine 参考**：COMPLETE（12 格全档，`capytaine_ref.json`，provenance 完整）。
- **TensorBEM 梯**：`PARTIAL: 1 of 12`（`results.json` 的 `execution_status`）。仅 `h/a=3.0|ka=0.5` 有 3 档，`h/a=3.0|ka=1.0` 有 2 档；其余 10 格空。
- **全量 12 格 TensorBEM 梯**：已递交集群作业 **cj4422**（6 分片 × 8 核，`--ladder-only --shard/--shards 6`，`--jobs 4`），排队中（queue_position≈58）。该作业完成后需 `--merge-caches` 合并 6 分片 → `run_study --write-doc` 出全量 verdict。
- **单格预判**（`h/a=3.0|ka=0.5`，320 面板）：TB A33=2862.87 vs capy A33=2999.92 → **-4.57%**（面板内接多面体面积亏缺，外推应一阶移除）；B33=679.86 vs 740.90 → **-8.24%**。均 <15% owner 限，但未外推不能判 <3%。

## 4. 本单已做的验证（pipeline 校验）

- 计划 dispersion 恒等式：`nu = k0 tanh(k0 h)` 对全部 12 格反解到 `k0 a` 精确（`resid < 1e-9`）。
- 参考 dispersion 恒等式：`capytaine_ref.json` 每格 `k0 tanh(k0 h) - nu` 残差 `<= 1e-9`。
- 参考覆盖：`tests/test_xval_bem_finite_depth_reference_grid.py` 断言 12 格全档 + legacy 公式逐格携带（已绿）。
- 参考双公式：`newer` vs `legacy` 在球上一致到 round-off（`rel < 5e-3`），故参考不确定度 = 面板梯（这正是逐格报梯的原因）。

## 5. 注意点/局限

- 两码都把球离散为内接多面体，粗档共享 ~1-2% 面积亏缺；外推一阶移除，观测阶说明残差中多少是离散化。
- 全潜闭体：无吃水线，不触发不规则频率处理；吃水线有限水深情形仍在 `docs/finite_depth_solver3d.md` 范围外。
- 外推带是自洽度量非真值界：两码共享内接多面体几何偏差，公共几何误差会落在每个带内不可见；带排除的是「TensorBEM 梯离散化误差被误判为求解器分歧」。

## 6. 误差判定

- 关键量（外推后 TB vs capy 偏差）**尚未可判**：全量 12 格 TensorBEM 梯在集群 cj4422 在飞，未外推前不能判 <3%（`tests/test_capytaine_finite_depth_crossval.py` 对 partial 执行显式 `xfail`，不粉饰）。
- 单格粗档（320 面板）A33 -4.57%、B33 -8.24%，均 <15% owner 限，但这是单网格差非外推 verdict。
- 无基准须声明：本单基准 = capytaine 有限水深 Green 函数（第三方独立 BEM 码），非解析解；解析锚（多极子）仅作深水端 `h -> inf` 参考。

---
*本文件由 study driver 从冻结工件程序化生成/校验；`results.json` 的 `execution_status` 由 `execution_status()` 从梯本身计算，非手写。*
