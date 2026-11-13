# 交差审计 m17205 — CASE-BEM-TWO-CYL-MUTUAL（双圆柱互附加质量，TensorBEM）

**审计单**: m17237（board-audit）· **被审单**: m17205（succeeded，auto-approve dev_acceptance）
**锚点**: CASE-BEM-TWO-CYL-MUTUAL · **算例池**: tensor_case_pool id=141 (BEM-TWO-CYL-MUTUAL, tol_policy `<5%`)
**日期**: 2026-11-13 · **审计人**: agent/m17237

## 0. 结论

**判级 L4（基准验证）**，三件套全部真存在且独立复核通过：

| 验收件 | 状态 | 证据 |
|---|---|---|
| run_artifacts | ✅ 真存在 | `/root/tp-agent/work/m17205/artifacts/two_cylinder_added_mass/`：benchmark.json (12536 B) + figures_digest.json + 3 PNG，sha256 见 §4 |
| benchmark ref vs got | ✅ 容差内 | 96 面板/体 max rel err = 5.51e-4 (0.055%) < 3% < 5%；48→64→96 单调 2.20e-3→1.24e-3→5.51e-4，观测阶 ~2.0 |
| repro_cmd 可执行 | ✅ 独立重跑 | 审计重跑 282/282 标量键与原件一致，**benchmark.json sha256 逐字节相同**（68e3568c…） |
| 测试 | ✅ | `PYTHONPATH=src pytest tests/test_two_cylinder_added_mass.py -q` → **18 passed** (49.93 s)，与被审单声明一致 |
| 报告六段 | ✅ 齐全 | tensor_reports id=16206：①案例介绍 ②方法与实现 ③计算结果 ④结果图(3 图) ⑤注意点 ⑥误差判定（逐项 <3% 显式回答） |

## 1. 基准裁判（唯一裁判=基准判据，非报告好看）

任务令指定裁判行 `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-BEM-TWO-CYL-MUTUAL'`——**查得 0 行**（系统缺口，同 verdict#490 S2 发现）。容差判据回退到算例池行（id=141, tol_policy `<5%`）+ 被审单 goal（`<5%`）+ 生态线 `<3%`。本审计按 verdict#488 F3 先例**补登记裁判行**（status=passed，数值见 §3）。

逐项核 got vs 容差（96 面板/体，取自 benchmark.json，零手抄）：

| 关键量 | ref（多极级数闭式） | got (BEM 96) | rel err | <3% | <5% |
|---|---|---|---|---|---|
| 自附加质量 A[0,0] | 3322.936921 | 3322.841406 | 2.87e-5 | ✅ | ✅ |
| 互耦（纵向）A[0,2] | -730.506107 | -730.103748 | 5.51e-4 | ✅ | ✅ |
| 互耦（横向）A[1,3] | 730.506107 | 730.103748 | 5.51e-4 | ✅ | ✅ |
| 同相垂荡 A[1,1]+A[1,3] | 4053.443028 | 4052.945155 | 1.23e-4 | ✅ | ✅ |

收敛/敏感性（≥2 点）：① 网格 48/64/96 单调下降，observed_orders=[1.9998, 2.0004]；② 间距 d/a=3/5/10/20/40，互耦×d²→6440.27（(a/d)² 律），自附加质量→ρπa²（1+7.8e-7）。

## 2. 代码与集成（L2/L3 核）

- 实现：`src/tensorbem/benchmarks/two_cylinder_added_mass.py`（413 行，非占位；main() :363，CLI --panels/--separation-ratio/--terms/--inject/--json/--figures）。
- 参考解：`src/tensorbem/analytic.py:1678 two_cylinder_added_mass`（截断多极级数，N=24，512 点梯形；**不调 solver 任何算子**，与 BEM 侧完全独立）。
- 集成：benchmark 模块 `:66 from ..solver import assemble_matrices`（:106 调用）、`:65 MultiLoopMesh` 双环 hole 网格——走主求解器层矩阵，无新核。
- 代码入库状态：模块/测试/文档均已在 main（m4721 交付，PR #315，commit 1259bb5）；m17205 按 goal「实现或复用现有能力」复用该模块跑池例。

## 3. 独立复核动作（审计人执行，非转述）

1. **重跑**：`cd /root/tp-agent/work/m17205 && PYTHONPATH=src python3 -m tensorbem.benchmarks.two_cylinder_added_mass --panels 48 64 96 --json <out>` → 输出与原件 282 个标量键 0 差异，sha256 相同（68e3568c26cfe7b24e65ab88fd95f629c0f488fdf3b87599229065948709b32b）。
2. **测试**：18 passed（含负控 `test_injected_double_layer_sign_error_turns_the_checks_red`、`test_decoupled_single_body_treatment_is_rejected`）。
3. **报告数字溯源**：§3 表与 §6 表全部数字与 benchmark.json 一致（零手抄成立）。
4. **负控语义**：`--inject layer_sign` 使互耦项塌陷（自附加质量仅偏 0.3–3.6%）——单体基准发现不了该错误，本案例价值所在；由树内测试固化。

## 4. 工件 sha256 钉（m17205 worktree）

```
68e3568c26cfe7b24e65ab88fd95f629c0f488fdf3b87599229065948709b32b  benchmark.json
a59408db1b9035aa4875ca1780ed9e7e041afaabf60e1aaac8280017e9a5cf64  figures_digest.json
91785758dea2396ea0f232e8c8e0525ba093a6b6de4c89cbea63cdf563ab5094  field_and_boundary.png
59d9038bf3528bcd7a1ab8295b8745b6a1869e7e6194dd94eb71b9a081d0c95d  interaction.png
39fcdce20cf36f34fa365bfd385b8a004d34d298ce5d9d0f8f0539d4616fbfaa  refinement.png
```

## 5. 发现（severity 排序）

- **F1 medium（系统缺口）**：tensor_benchmark_cases 无 CASE-BEM-TWO-CYL-MUTUAL 行 → 本审计补登记（先例 verdict#488 F3）。
- **F2 medium**：m17205 分支无独有 commit（head_commit=null），artifacts 仅存于 m17205 worktree 磁盘、未入 git；实质由审计重跑恢复（逐字节一致）。代码本体在 main（m4721/PR#315），故不降级。
- **F3 low**：case_pool id=141 仍 'dispatched'、feature_ledger 锚仍 'open'——按机制链（gate confirmed → taskboard 消账 → case_feeder 销池）自动流转，审计不手改。
- **F4 info**：外部锚=仓内闭式重实现（非印刷表/实验值）→ 按 verdict#490 先例计 L4 参考，**不计 L5**（WITHHELD）。
- **F5 info**：被审单 changed_files 仅 `_post_m17205.py`（未跟踪）——m17205 增量=运行+工件+报告+负控，非新代码。

## 6. 判级理由

L1 tests 非空 ✅；L2 impl/entry/run_artifacts 真 ✅；L3 integration_callers 主求解器调用 grep 复核 ✅；L4 benchmark 数值对照 + repro_cmd 独立重跑逐字节复现 ✅；L5 无印刷表/实验锚 → **L4 封顶**。附百页报告不改变级别；级别=功能实质。

## 7. 未做检查（如实）

- 未做第二套独立 BEM 实现交叉验证（参考解为独立解析路线，已足够）。
- 未跑 d/a<3 近距区（案例登记范围 d/a≥3）。
- 未核 3 张 PNG 的像素级内容（figures_digest.json 与测试背书）。
