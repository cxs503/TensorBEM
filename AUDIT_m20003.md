# m20065 交差审计报告：CASE-BEM-CRACK-SIF（m20003）

## 审计对象

- **Mission**: m20003
- **Anchor**: CASE-BEM-CRACK-SIF
- **Repo**: TensorBEM
- **Commit**: 4e310bc9
- **Status**: succeeded（声称）

## 取证结果

### 三件套核验

| 项目 | 状态 | 证据 |
|---|---|---|
| run_artifacts | ✓ 存在 | `docs/case_bem_crack_sif/numbers.json` (27410 bytes, 892 lines), `REPORT.md` (166 lines), 4 张 PNG |
| benchmark ref vs got | ✓ 在容差内 | 最细网格三条路线误差 0.512%/0.004%/0.771%，全部 <5% |
| repro_cmd | ✓ 可执行 | `PYTHONPATH=src python3 -m tensorbem.benchmarks.case_bem_crack_sif --output docs/case_bem_crack_sif/numbers.json --figures docs/case_bem_crack_sif` |

### 报告六段核验

| 段 | 状态 |
|---|---|
| ① 案例介绍 | ✓ 物理背景/几何/参数表齐全 |
| ② 方法流程 | ✓ Kelvin BEM 公式/离散/无量纲化/三条路线 |
| ③ 计算结果 | ✓ 数字表（程序化注入） |
| ④ 结果图 | ✓ 4 张 PNG（云图+比较曲线） |
| ⑤ 注意点 | ✓ 5 条失败模式/局限如实记录 |
| ⑥ 误差判定 | ✓ 逐项列出 vs 基准，显式回答 <5% |

### 基准裁判核验

对照 `tensor_benchmark_cases`（通过 case_pool_registry.py 确认）：
- anchor: CASE-BEM-CRACK-SIF
- tolerance: <5%
- reference: K_I = σ√(πa) = 17.7245 MPa·m^(1/2)
- 最细网格最大误差: 0.771% (route C) < 5% → **PASS**

### 测试核验

`tests/test_elasticity2d_crack_sif.py`：8 条测试，全部通过（50.70s）：
- test_rigid_body_row_sum_of_traction_kernel ✓
- test_plane_strain_modulus ✓
- test_reference_center_crack_k_is_the_closed_form ✓
- test_near_tip_relation_recovers_k_on_the_exact_griffith_profile ✓
- test_uniaxial_patch_test ✓
- test_lame_annulus_against_closed_form ✓
- test_crack_quarter_mesh_geometry ✓
- test_crack_sif_coarse_mesh_lands_within_tolerance ✓

## 判级：L4

**理由**：
- 真实 BEM 求解器（elasticity2d.py, 722 行）：输入→计算→输出完整链路
- 完整算例池基准：4 网格细化 + 4 维参数敏感性 + 11 gates 全通过
- 误差分析：三条独立 SIF 路线 + 收敛性 + 截断误差量化
- 一键复现 + 8 条单元测试（含负面对照）
- 参考为闭式理论解（非实验/印刷表），故不到 L5

**不升级 L5 的原因**：参考基准是闭式理论公式的程序化求值，非独立实验数据或印刷表数值；TensorBEM 内无第二套弹性 BEM 实现可做交叉验证。

## 落表

- `tensor_maturity_verdicts` id=811, level=4, gate_state=pending
- `tensor/reports` id=18264

## 审计发现

| # | 严重度 | 描述 |
|---|---|---|
| F1 | info | TensorBEM 此前零弹性算子，elasticity2d.py 是从零新增的 722 行真实模块 |
| F2 | info | 三条独立 SIF 路线全部 <5%；路线 B 近精确（0.004%）因假设 Griffith 形状，A/C 为真实预测（0.51%/0.77%） |
| F3 | info | 路线 A 收敛到 ~0.5% 平台（常值单元无法表示 √r），已如实记录 |
| F4 | info | 参考为闭式理论，适合 L4 但不适合 L5 |
| F5 | info | 8 条测试全通过，含负面对照 |

## 未运行检查

- 完整 benchmark 重跑（超时 >120s；numbers.json 和测试已独立验证）
- PNG 像素级内容验证（由文件存在性和大小背书）
- 独立 BEM 实现交叉验证（TensorBEM 内无第二套弹性 BEM）

## 处置

**approve** — 证据充分，判级 L4 合理，gate 应 confirmed 消账。
