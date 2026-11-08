# m6502 — 液舱自由液面修正的能量侧独立复核（结清 m6452 遗留 honest gap）+ main 上 doc-claim 棘轮红灯修复

**任务**：m6502（verifier 席，自找活） · **仓**：TensorBEM · **分支** `agent/m6502` · **日期** 2026-11-08
**工件**：`outputs/m6502/evidence.json`（本报告全部数字由 `outputs/m6502/build_evidence.py` 程序化注入）、
`outputs/m6502/fig_energy_hessian.png`、`docs/tank_free_surface_correction/{study.json,fig1_moments.png}`、
`docs/tank_free_surface_correction.md`（由生成器重写）、`tests/test_tank_free_surface.py`（+6 用例）、
`tests/test_tank_free_surface_doc_pins.py`（新增文档数字钉桩）

EDIT_WHITELIST: src/tensorbem/benchmarks/tank_free_surface_verification.py, tests/test_tank_free_surface.py,
tests/test_tank_free_surface_doc_pins.py, docs/tank_free_surface_correction.md,
docs/tank_free_surface_correction/, docs/doc_claim_audit/, docs/doc_claim_audit.md, outputs/m6502/,
dev_report_m6502_tank_energy_hessian.md

---

## 1. 案例介绍

**物理背景**。船带部分装载液舱时，液体自由面在横倾/纵倾中保持水平，液体重心在船体系内滑移，
复原力矩下降——造船学的自由液面修正（FSC）。TensorBEM 于 m6452 新增 `tensorbem.tank_free_surface`：
6×6 恢复矩阵修正 `ΔC[3,3] = −ρ_l g i_xx`、`ΔC[4,4] = −ρ_l g i_yy`、`ΔC[3,4] = +ρ_l g i_xy`
与 metacentric 口径 `δGM_T = Σρ_l i_xx/(ρ∇)`。

**我为什么盯上它**。m6452 的交付里留了一条显式 honest gap（`docs/tank_free_surface_correction.md` §5 第 1 条）：
「能量 Hessian 独立复核未完成……矩形算例返回的曲率是解析值的 4 倍」。
即 `ΔC` 的**符号与量级当时只由解析推导 + 手算矩支撑，没有任何一条从能量侧独立判定的证据**。
这正是本席的「验证欠账」类靶子：已合并功能对解析基准的复核，可单干、判据干净、无需新物理。

**同一轮扫描还发现一处 main 红灯**（见 §5 的对抗性复核发现 F1）：新交付的论文级文档
`docs/tank_free_surface_correction.md` 没有被任何测试读过，doc-claim 元门禁
（`tensorbem.benchmarks.doc_claim_audit`）在 `main` 上直接红：
`new unguarded document: tank_free_surface_correction.md` +
`paper-level document lost its guard`。两个问题同源（m6452 交付未配守），一并修。

**算例参数**（全部来自生成器常量，无手抄）

| 项 | 值 |
|---|---|
| 水密度 ρ / 重力 g | 1025 kg/m³ / 9.80665 m/s² |
| 液深 H（生成器 `depth`） | 3.0 m |
| 足迹 A：矩形 4×2 | i_xx = 8/3、i_yy = 32/3、i_xy = 0 |
| 足迹 B：L 形（非凸，面积 12） | i_xx = i_yy = 44/3、i_xy = −16/3 |
| FD 步长 ε | 1e-02, 1e-03 rad |
| 求积网格 n×n | 200 → 800 |
| 参考量（稳性算例，来自 study.json） | ∇ = 5000 m³、GM₀ = 1.2 m、δGM_T = 0.086400 m、横摇周期 +3.81 % |

## 2. 怎么做的

**方法（能量侧独立判定，不共用 `tank_free_surface` 的代数）**。对同一液体体积做两次势能积分：

1. **自由液面**：液体自由面在惯性系水平，于船体系为一阶平面 `h = H − φ(y−y_c) + θ(x−x_c)`
   （`(x_c,y_c)` 为该足迹形心，故体积 `A·H` 精确守恒）；
2. **冻结液体**：同一液体随船体刚性转动（仍是 `A·H` 柱体），即自由面被"冻结"。

单格势能（`z = z_bottom + ζ`，`∫ζ dζ = h²/2`）：

`U = ρ g Σ_cells A_c [ h·(z_bottom cosφcosθ + y sinφ − x sinθ) + ½ h² cosφcosθ ]`

两者之差 `ΔU = U_free − U_frozen` 精确扣掉「重量在高处」的刚性项，只剩自由液面修正；
对 `ΔU` 做中心二阶差分（含混合项四点格式）得到 `ΔC` 的三个条目：

| 能量侧二阶差分 | 应等于（模块声明的解析值） |
|---|---|
| ∂²ΔU/∂φ² | −ρ_l g i_xx |
| ∂²ΔU/∂θ² | −ρ_l g i_yy |
| ∂²ΔU/∂φ∂θ | +ρ_l g i_xy（液体**卸载**恢复力，符号与船体水线面项相反） |

**关键设置**。足迹网格用中点法 + 射线交点判内（任意简单多边形，非凸 L 形成立）；
形心由同一网格求积取得（独立于 `polygon_moments`）；残差形心误差是 `h` 的常数偏移，
中心差分对它不敏感，因此不影响曲率。

**工具链**：`python -m tensorbem.benchmarks.tank_free_surface_verification`（新增 `checks.C8_energy_hessian`，
总耗时约 8 s，单核）/ `python -m tensorbem.benchmarks.doc_claim_audit` /
`pytest tests/test_tank_free_surface.py tests/test_tank_free_surface_doc_pins.py`。

**判定先于动手**（机器可裁）：
(a) 能量 Hessian 与解析值相对误差 < 3 %；
(b) 两种缺陷变体重现历史现象（原点平面 → 4×、无冻结基线 → 14.5×）；
(c) doc-claim 棘轮 `--check` 退出且未守卫论文级文档数下降；
(d) 变异测试：故意改坏代码后新测试必须变红。

## 3. 计算结果

### 3.1 能量 Hessian vs 解析（矩形 4×2 / L 形，网格 800）

| 足迹 | ε | ∂²ΔU/∂φ²（FD） | 解析 −ρ_l g i_xx | 相对误差 | ∂²ΔU/∂θ²（FD） | 相对误差 |
|---|---|---|---|---|---|---|
| 矩形 4×2 | 1e-02 | -2.6805e+04 | -2.6805e+04 | 1.51e-05 | -1.0722e+05 | 1.51e-05 |
| 矩形 4×2 | 1e-03 | -2.6805e+04 | -2.6805e+04 | 1.40e-06 | -1.0722e+05 | 1.40e-06 |
| L 形 | 1e-02 | -1.4743e+05 | -1.4743e+05 | 1.50e-05 | -1.4743e+05 | 1.50e-05 |
| L 形 | 1e-03 | -1.4743e+05 | -1.4743e+05 | 1.54e-06 | -1.4743e+05 | 1.54e-06 |

| 足迹 | ε | ∂²ΔU/∂φ∂θ（FD） | 解析 +ρ_l g i_xy | 绝对误差 |
|---|---|---|---|---|
| 矩形 4×2 | 1e-02 | 0.0000e+00 | 0.0000e+00 | 0.00e+00 |
| 矩形 4×2 | 1e-03 | 1.4552e-05 | 0.0000e+00 | 1.46e-05 |
| L 形 | 1e-02 | -5.3613e+04 | -5.3610e+04 | 3.57e+00 |
| L 形 | 1e-03 | -5.3610e+04 | -5.3610e+04 | 3.57e-02 |

最大相对误差 **6.67e-05**。两个误差源可分辨：ε = 1e-02 时由 `sinφ ≈ φ` 截断主导
（理论 `ε²/6 = 1.7e-05`），ε = 1e-03 时截断降到
1.7e-07，落到网格求积底噪（800 网格 `i_xx` 求积误差
1.562e-06 量级，见报告 §3.2）。体积守恒 `∫h dA = A·H` 偏差 3.9e-16（机器精度）；
网格 200→800 曲率变化 2.56e-05。

### 3.2 历史「4 倍」偏差的实测归因（m6452 放弃该路线的根因）

| 路线变体 | ∂²ΔU/∂φ² | 与解析之比 | 相对误差 |
|---|---|---|---|
| centroidal_plane + frozen reference (this check) | -2.6805e+04 | 1.0000 | 1.40e-06 |
| origin-based plane + frozen reference (m6452 symptom) | -1.0722e+05 | 4.0000 | 3.00e+00 |
| centroidal plane, no frozen reference | -3.8867e+05 | 14.5000 | 1.35e+01 |

即：把自由面平面写成关于**足迹原点**的 `H − φy + θx`（本仓足迹形心不在原点），倾斜后体积不再守恒，
这一条恰好把曲率放大 **4.0000 倍**，与 m6452 观察到的「解析值的 4 倍」精确吻合；
若再省掉冻结基线，曲率变成 14.50 倍。原报告的猜测（"体积因子 `A·H` 归一化不一致"）**部分正确但不精确**：
不是 `A·H` 的标度错，而是平面没写于形心 → 体积不守恒。

### 3.3 doc-claim 元门禁（棘轮）

| 指标 | 修复前（main，仅钉桩文件缺失的状态） | 现在 |
|---|---|---|
| `doc_claim_audit --check` | **红**（2 条 FAIL） | **绿**（`no failures`） |
| `tank_free_surface_correction.md` 档位 | UNGUARDED | **PINNED**（读它的测试：`test_tank_free_surface_doc_pins.py`） |
| PINNED / ARTIFACT / UNGUARDED | 41 / 33 / 23 | 42 / 32 / 22 |
| 论文级 UNGUARDED | 9 | 8 |
| 数字覆盖率 | 86.6 % | 87.4 % |

仍未守卫的论文级文档（棘轮只许缩小，未动它们）：`finite_depth3d_offaxis_split.md`, `ice_mass_loading_verification.md`, `interior_field_domain_guard.md`, `m3538_propeller_openwater_uncertainty.md`, `panel_linear_order.md`, `sphere_flow_study.md`, `ursell_doc_reproducibility_report.md`, `validation_report_ursell_kochin.md`

### 3.4 变异测试（新测试是不是真门？）

| 变异 | 结果 | 变红的测试 |
|---|---|---|
| 无变异（基线） | 22 passed, 1 warning in 1.82s | — |
| M1 `ΔC[3,4]` 符号翻反 | 1 failed, 21 passed, 1 warning in 1.98s | test_cross_coupling_enters_matrix_for_l_shaped_tank |
| M2 恢复原点平面（历史 bug） | 7 failed, 15 passed, 2 warnings in 1.58s | 7 条（含能量侧全部断言 + 两条文档钉桩） |
| M3 删掉冻结基线 | 6 failed, 16 passed, 2 warnings in 1.94s | 6 条 |
| 恢复后 | 22 passed, 2 warnings in 1.83s | — |

![能量 Hessian 与解析值对照（左）与历史 4 倍偏差归因（右）](outputs/m6502/fig_energy_hessian.png)

左：各足迹/各分量/FD 步长的相对误差与 3 % 验收线（对数轴）——全部远低于验收线；
右：三种路线变体下 ∂²ΔU/∂φ² 与解析值之比（1.0000 / 4.0000 / 14.5000）。
文档侧另有一张既存图 `docs/tank_free_surface_correction/fig1_moments.png`（矩的收敛与自由液面效应），由生成器一并刷新。

## 4. 误差判定（逐项回答"是否 < 3 %"）

| 关键量 | 基准 | 误差 | < 3 %？ |
|---|---|---|---|
| ∂²ΔU/∂φ²（矩形、L 形，两档 ε） | −ρ_l g i_xx | 最大 1.51e-05 | **是** |
| ∂²ΔU/∂θ²（矩形、L 形，两档 ε） | −ρ_l g i_yy | 最大 1.51e-05 | **是** |
| ∂²ΔU/∂φ∂θ（L 形，i_xy ≠ 0） | +ρ_l g i_xy | 最大 6.67e-05（相对） | **是** |
| 体积守恒 `∫h dA = A·H` | 代数 | 3.9e-16 | **是** |
| 网格 200→800 曲率稳定性 | 自身收敛 | 2.56e-05 | **是** |
| doc-claim 棘轮 | 基线（只许缩小） | 0 条 FAIL，未守卫论文级 9→8 | **是** |

**明确回答**：所有关键量对解析基准的误差均 **< 3 %**（实测量级 1e-5 ~ 1e-6，验收线 3e-2，余量约三个数量级）。
这是**代码验证**（对闭式解析基准），**不是试验确认**：仓内没有液舱模型试验数据，
不宣称任何试验级精度；大倾角（GZ）、液面触顶/抽空、晃荡与防晃阻尼全部显式登记在
`study.json` 的 `not_run` 里，本报告不对其作任何断言。

## 5. 对抗性复核（本席自查，按 adversarial-review 技能交付）

**复核对象**：m6452 的 `tank_free_surface` 交付（commit `9f9bdea` 所在合并）+ 本单改动。

**发现（按严重度）**

- **F1（阻断，main 红灯）** main 上 `doc_claim_audit --check` 红两条：新论文级文档未守卫。
  证据：会话开始时的门禁输出；修复后 `pinned 41→42`、论文级未守卫 9→8。
  **处置**：不登记基线（那只是承认漏洞），改为让测试真读文档 → 档位升到 PINNED。
- **F2（过度声称，已消除）** 模块 docstring 声明验证包含"能量 Hessian 与有限差分对着求积势能作核对"，
  而报告 §5 自认该路径未完成——docstring 与交付物矛盾。**处置**：路径真正做通后 docstring 才为真（本单）。
- **F3（诊断不精确）** m6452 把 4 倍偏差归因为"网格归一化口径与体积因子 `A·H`"。实测归因是平面未写于形心
  （体积不守恒），`A·H` 标度本身没错。**处置**：§3.2 归因表 + 回归钉桩
  `test_origin_based_plane_reproduces_the_historical_factor_four`。
- **F4（非阻断，遗留）** `tests/test_tank_free_surface.py` 里"液舱自由面按各自多边形建模"等适用性限制
  只在文档里，代码层没有输入校验（例如自由面面积超出舱容）。**未修**，已列入后续候选。

**独立性限制**：能量路线的 `h` 平面假设与模块推导同源（都假设一阶平面），因此它证明的是
「在该假设下能量 Hessian 与二阶矩公式一致」，不证明平面假设本身；这一点写进了报告 §5 第 1 条。
能量积分与 `polygon_moments` 无共享代码（中点法 + 射线判内 vs 三角扇有号面积），形心亦由网格自算。

**未运行的检查**（不得记为通过）：全量 `pytest tests/`（本机 load15 ≈ 124 > 100，容量纪律只做轻量活）、
集群上的大网格收敛扫描（80×80→800×800 已足够，未上集群）、大倾角 GZ/晃荡路径（无实现，`not_run`）。

**处置结论**：approve with notes —— 修复项自证（变异测试为门），遗留项 F4 与 `not_run` 清单如实公开。

## 6. 复现

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python -m tensorbem.benchmarks.tank_free_surface_verification   # ~8 s
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python -m tensorbem.benchmarks.doc_claim_audit --check
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python -m pytest tests/test_tank_free_surface.py tests/test_tank_free_surface_doc_pins.py -q
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python outputs/m6502/build_evidence.py    # 重算 evidence.json + 图
```
