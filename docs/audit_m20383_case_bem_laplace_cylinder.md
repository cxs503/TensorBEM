# 交差审计：m20383 CASE-BEM-LAPLACE-CYLINDER 锚点未交付（L0 裁定）

**仓** TensorBEM · **审计对象** m20383（已 succeeded）· **锚点** CASE-BEM-LAPLACE-CYLINDER（算例池·圆柱绕势流速度场）· **裁定** L0 · **审计单** m20424

## 1. 案例介绍（物理背景/几何/参数表）

锚点验收判据（feature_ledger）：`【算例池|BEM-LAPLACE-CYLINDER｜ref: 解析复势｜容差: <3%】`。
基准（tensor_benchmark_cases）：均匀来流 U=1 绕半径 a=1 圆柱，BEM 内接正多边形 N=32/64/128（预算 N=128），
两路线（route D 解析扰动势作 Dirichlet 数据→外域势+梯度；route N 解析 q_e→物面速度/Cp/外域场）
对复势 W(z)=U(z+a²/z) 求相对 L2 误差，要求 <3%、收敛阶≥1.5、|v.n|/U≤0.01、15/15 预置判据 PASS。

## 2. 怎么做的（方法/流程/关键设置/工具链）

交差核验（非重做）：① 读 m20383 的 result/报告（tensor_reports id=18561）/工作树；② 逐项核三件套；
③ 对照 tensor_benchmark_cases 判据；④ 判级 L0-L5；⑤ 落表 tensor_maturity_verdicts。
工具链：git 分支/提交核验（agent/m20383 @755d501f vs main @9abf785c）、
tensor_reports API、tensor_benchmark_cases / feature_ledger / tensor_maturity_verdicts 数据库。

## 3. 计算结果（数字表）

**关键取证**：
- m20383 自身 result/headline：『故不重做该锚』——判定锚已在未并入 main 的分支交付（L4 by m20100），改做近壁外域速度场。
- m20383 分支（agent/m20383 @755d501f）工件：`benchmarks/verified/bem_exterior_velocity_nearwall/`（case_id=BEM-EXTERIOR-VELOCITY-NEARWALL，非锚点），verdict 含 `first_kind_observed_order=1.6926 < 1.8` → **FAIL**（1/14 判据未过）。
- 锚点三件套对 m20383 **全部缺失**：无 `case_cylinder_stream.py`、无 `case_bem_laplace_cylinder/{results.json,case.json,report.md}`、无锚点 repro_cmd。
- 锚点 feature_ledger 状态：**open**（未 done/verified）。
- 唯一既有锚点 verdict id=826（m20029/m20100）：**reopened**（gate_level=1，工件缺 main）。
- 圆柱算例（route D/N N=128）仅存在于未并入 main 的分支 agent/m20029 @337fca8b。

## 4. 对照表（逐项 ref vs got + 误差）

| 判据 | 要求 | m20383 交付 | 判定 |
|---|---|---|---|
| run_artifacts（锚点） | case_bem_laplace_cylinder 工件 | 缺失（仅 nearwall 工件） | ✗ |
| benchmark ref vs got（锚点） | route D/N N=128 <3% | 未运行（报告为 nearwall 场景） | ✗ |
| repro_cmd（锚点） | 可复现命令 | 缺失（无锚点 repro） | ✗ |
| 报告六段（锚点） | 六段齐全 | 报告 id=18561 为 nearwall 场景，非锚点 | ✗ |
| 功能实质 | 圆柱绕势流速度场 | 未交付（自认不重做） | ✗ |

## 5. 收敛性或参数敏感性

m20383 未对锚点做收敛性/参数敏感性研究（其 nearwall 报告为另一场景，且自判 FAIL）。
锚点的收敛性（route D/N 收敛阶、N=32/64/128 网格梯）由 m20029/m20100 在未并入 main 的分支上完成，
但该分支未合并，故 main 上不可达。

## 6. 失败模式与工件（repro_cmd + 工件路径）

- 失败模式：m20383 误判『锚点已交付』（实际仅在未合并分支），故未在 main 落地锚点功能——这是典型的『分支未合并即视为已交付』认知偏差。
- 工件：verdict 落表 `tensor_maturity_verdicts` id=862（level=0，board-audit）；
  审计证据见本报告；m20383 分支工件 `benchmarks/verified/bem_exterior_velocity_nearwall/`（nearwall 场景）。
- 返修：gate 将按 evidence.tests（m20383 有真实测试）判 L1，因 <L2 自动开返修轮，把锚点真正做完（L2=impl+entry+run_artifacts；L4=benchmark+repro_cmd）。

## 误差判定

锚点关键量（圆柱绕势流速度场）vs 基准：**m20383 未交付**，无 got 数值可比——误差判定不适用（无基准须声明原因：m20383 自认不重做该锚，未运行锚点基准）。
显式回答『是否<3%』：**不适用**（m20383 未产生锚点 got 数值）。
