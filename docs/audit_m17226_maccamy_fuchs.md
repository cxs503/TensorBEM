# 交差核验 m17226 · MacCamy-Fuchs 垂直圆柱绕射（TensorBEM）→ L4

- **审计单**: m17300（board-audit，任务榜令 2026-11-11）
- **被审单**: m17226（返修轮1，原单 m17180，锚点 CASE-BEM-MACCAMY-FUCHS-DIFF）
- **被审仓**: TensorBEM @ HEAD `20d66f8`（m17226 工作树 `/root/tp-agent/work/m17226`）
- **基准裁判**: `tensor_benchmark_cases` id=36（anchor_slug=CASE-BEM-MACCAMY-FUCHS-DIFF，status=passed）
- **裁定**: **L4 confirmed**（完整算例 + 误差分析 + 可复现；无新物理 → 非 L5）

## 0. 结论速览

| 验收三件套 | 判定 | 证据 |
|---|---|---|
| run_artifacts 文件真存在 | ✅ | results.json / gates.json / report.md 均在 main 已提交，两 gate 根均可解析 |
| benchmark ref vs got 在容差内 | ✅ | got=0.000132071 (0.01321%) < 3% 生态 / < 5% 任务；DB 裁判 status=passed |
| repro_cmd 可执行 | ✅ | 独立重跑 `build_maccamy_fuchs_anchor` 工件字节级复现（sha256 与 pin 一致），6 门全 PASS；pytest 32 passed |

## 1. 案例介绍（物理背景 / 几何 / 参数）

MacCamy & Fuchs (1954, USACE TM 69) 给出**底置垂直圆柱**在规则入射波下的绕射闭式解
（Bessel/Hankel 函数）。本算例用 TensorBEM 的 2-D 外域 Helmholtz BEM
（`tensorbem.helmholtz.solve_vertical_cylinder_diffraction`，常值面板 + Schenck CHIEF
内域零场正则化）对标该闭式解。

| 参数 | 值 |
|---|---|
| 半径 a | 1.0 m |
| 水深 h | 3.0 m（h/a=3.0） |
| 频率 ka | 1.0（设计点）；扫描 0.1–3.0 |
| 流体密度 ρ | 1025 kg/m³ |
| 重力 g | 9.80665 m/s² |
| 波幅 A | 1.0 m |
| 面板 | 16/32/64/128 正多边形平板 |

## 2. 怎么做的（方法 / 流程 / 工具链）

审计流程（非重做）：
1. **取证**：读 m17226 result（`tensor_dev_missions.result`）/ 工作树 / `evidence.json`。
2. **逐项核验收三件套**：
   - `impl_files` 5 个源码文件存在且非占位（`maccamy_fuchs.py` 339 行真实现）；
   - `entry` 入口 `python -m tensorbem.benchmarks.maccamy_fuchs --panels 128` 独立重跑，
     复现 `force_x_relative_error=0.00013207074898697037`（与存储值逐位一致）；
   - `run_artifacts` 三件在 m17226 工作树与 `/root/tensor-ecosystem/TensorBEM` 均存在；
   - `repro_cmd` 全量重跑：`build_maccamy_fuchs_anchor` 重建 4 级 h 收敛 + 6 点频扫 +
     2 共振 CHIEF 研究，`results.json`/`gates.json` 与提交工件**字节级一致**
     （sha256 `2a436541…`/`ca65d377…` 与 evidence pin 一致），G1–G6 全 PASS。
3. **基准裁判**：对照 `tensor_benchmark_cases`（唯一裁判）逐项核 got 在容差内。
4. **gate 逻辑复核**：读 `maturity_gate.py` 的 `judge()`/`files_exist()`/`callers_verified()`，
   确认本 verdict 的 evidence 指针（干净文件路径，无 `:行号` 后缀）可在
   `/root/tensor-ecosystem` 与 `/root/tp-agent/work/m17300` 双根解析，
   首个 integration_caller `tensorbem.benchmarks.maccamy_fuchs` 在生态仓 grep 命中 3 文件。

## 3. 计算结果（数字表，程序化注入零手抄）

### 3.1 设计点（ka=1.0, h/a=3.0, 128 面板）

| 量 | BEM (got) | 闭式 (ref) | 相对误差 | <3%? |
|---|---|---|---|---|
| F_x 幅值 (N) | 43101.93895697914 | 43099.663722184494 | 1.3207e-04 (0.01321%) | ✅ |
| 表面 φ L2 | — | — | 1.8171e-04 | ✅ |
| 条件数 κ | 2.4342 | — | — | — |

### 3.2 频率扫描（64 面板，max 误差）

| ka | C_M (BEM) | C_M (exact) | 力相对误差 |
|---|---|---|---|
| 0.1 | 2.018387 | 2.019181 | 3.932e-04 |
| 1.0 | 1.371912 | 1.371616 | 5.305e-04 |
| 3.0 | 0.307836 | 0.307734 | 2.578e-03 |

扫描最大力误差 2.578e-03 (0.258%) < 3% ✅

### 3.3 验收门（gates.json，all_gates_passed=true）

| 门 | 实测 | 阈值 | 判定 |
|---|---|---|---|
| G1 力收敛@128 | 1.3207e-04 | 5e-04 | PASS |
| G2 观测收敛阶 | 2.0069 | 1.9 | PASS |
| G3 表面势@128 | 1.8171e-04 | 2e-04 | PASS |
| G4 频扫最大误差 | 2.5779e-03 | 5e-03 | PASS |
| G5 CHIEF κ 降 | 198.95 | 100 | PASS |
| G6 Morison 长波限 | 1.8387e-02 | 2e-02 | PASS |

## 4. 结果图

`benchmarks/verified/bem_maccamy_fuchs/figures/maccamy_fuchs_convergence.png`
（2081×651 PNG，m17226 生成）：(a) 水平力 h 收敛 + O(h²) 参考线；(b) C_M(ka) 从
Morison 长波限到短波散射；(c) J₀/J₁ 不规则频率处 CHIEF 抑制。

## 5. 注意点（坑 / 局限 / 适用范围）

1. **m17226 新交付未提交**（F1, medium）：`evidence.json`、
   `tools/emit_maccamy_fuchs_evidence.py`、`figures/` 在 m17226 工作树为 `??` 未跟踪，
   `report.md`/`build_maccamy_fuchs_anchor.py`/`test_helmholtz_maccamy_fuchs.py` 为 `M` 未提交。
   **核心 L4 证据全部指向已提交文件**，故 gate 可解析；但 emit 工具未入库，
   含该步的 repro_cmd 在全新检出上会失败——本 verdict 的 repro_cmd 已改用仅含已提交组件的命令。
2. **gate 指针解析陷阱**（F2, info）：前 verdict#488（m17201 审 m17180）被 reopen 到 L1，
   根因是 evidence 指针带 `:行号`/描述文字，`files_exist` 按整串当路径 → 解析失败。
   本 verdict 全部用干净文件路径规避。
3. **2-D Helmholtz 路径**：本锚点只验证深度均匀圆柱路径（精确垂直分离 + 2-D BEM），
   不覆盖 3-D Kelvin 自由面 Green 函数（由 floating_hemisphere / submerged_sphere_wave 覆盖）。
4. **外部锚点为闭式重实现**（F6, info）：无印刷表（TM69/Newman）数值转录，
   印刷表级交叉核对 WITHHELD。

## 6. 误差判定（关键量 vs 基准，逐项）

| 关键量 | got | 基准容差 | 是否 <3% |
|---|---|---|---|
| F_x 相对误差 @ka=1.0/128 | 0.000132071 | <3% 生态 / <5% 任务 | ✅ 是（0.01321%） |
| 表面 φ L2 @128 | 1.8171e-04 | <3% | ✅ 是 |
| 频扫最大力误差 | 2.5779e-03 | <3% | ✅ 是（0.258%） |
| 观测收敛阶 | 2.0069 | ≥1.9 | ✅ 是 |

**显式回答：所有关键量误差均 < 3%。** 设计点力误差 0.01321%，比 3% 限低约两个数量级。

## 7. 判级理由（L0–L5）

- **L2 独立功能** ✅：真代码 输入→计算→输出（`solve_vertical_cylinder_diffraction`）。
- **L3 系统集成** ✅：主求解器 `solve_vertical_cylinder_diffraction` 被基准模块调用，
  并经 `tensorbem/__init__.py` 导出、`doc_number_ledger.py:78` 平台侧 subprocess 调用。
- **L4 基准验证** ✅：完整算例 + 逐点误差表 + O(h²) 收敛 + 频扫/CHIEF 敏感性 +
  一键复现（工件字节级复现 + sha256 pin）。
- **非 L5**：无新物理，属对既有求解器的基准验证（与 m17201 前判一致）。

## 8. 未跑检查（如实）

- ka>4 扩展扫描（DB metric 提及 0.05–4.0 14 点，属 m17180 扩展件，本单未独立重跑）；
- 第二独立 BEM 实现交叉核对；
- 印刷表（TM69/Newman）数值锚点。
