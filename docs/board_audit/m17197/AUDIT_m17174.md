# 交差审计 · m17174 · CASE-BEM-CYL-ADDED-MASS-2D（TensorBEM）

**审计单** m17197（board-audit） · **受审单** m17174（succeeded，auto-approve） · **审计日期** 2026-11-13
**受审分支** `agent/m17174`（head 与 main 相同 = 96f19f7，**无独有提交**）
**判定** **L4 = 基准验证（完整算例 + 误差分析 + 可复现）**，附两条非阻塞整改项；
**不判 L5**（无科研/工程新意，且算子范围=边界迹，见 §5 S3）。
**处置（disposition）**：approve with notes（L4 confirmed；S1/S2 建议机制化整改，不阻塞原单）。

---

## 1. 审计契约与独立复现范围

| 项 | 审计做法 |
|---|---|
| 契约 | 任务榜令：ref=`ρπa²`（无自由液面闭式）；容差 <3%；需 run_artifacts 真工件 + ref/got 数值对照 + repro_cmd 可执行 + 报告六段 |
| 唯一裁判 | `tensor_benchmark_cases WHERE anchor_slug='CASE-BEM-CYL-ADDED-MASS-2D'` → **0 行**（见 S2），故判据取自 mission goal 文本 `<3%`，并由本审计独立复算 |
| 独立复现 | 在本审计工作树（分支 `agent/m17197`，基于 main 96f19f7，**不复用受审工作树**）以 `PYTHONPATH=src` 重跑 refinement study |
| 复现结果 | `docs/board_audit/m17197/rerun/refinement_cylinder_repro.json` 与受审工件 `refinement_cylinder.json` **逐字节相同**（sha256 `75bce08b98744ac826ad8bf786ee754500332500e016c498538876142a6f2af8`），stdout 亦与之逐字节相同（无随机性/无时间戳） |
| 数值溯源 | 报告 ③④⑤⑦ 全部数字由 JSON 注入核对，未发现手抄偏差（下表即审计复算） |

命令行（本审计实际执行）：

```
PYTHONPATH=src python -m tensorbem.benchmarks.refinement_study \
    --panels 16 32 64 128 256 --output docs/board_audit/m17197/rerun/refinement_cylinder_repro.json
PYTHONPATH=src python -m pytest tests/test_cylinder_benchmark.py -q     # 2 passed, 1 warning, 2.17s
python docs/board_audit/m17197/rerun/make_audit_figure.py                            # 图（数字注入）
```

## 2. 受审工件存在性核验（三件套）

| 三件套 | 受审声明 | 审计实测 |
|---|---|---|
| run_artifacts | `artifacts/refinement_cylinder.json`、`fig_cylinder_added_mass.png` | **存在且非空**（7546 B / 101207 B），位于 `/root/tp-agent/work/m17174/artifacts/`；**未入仓**（`git status` 仍为 `??`，分支无独有提交，生态仓 main 无该文件）→ S1 |
| benchmark ref vs got | `ρπa²=3220.1325` vs 16/32/64/128/256 面元 | **逐项复算一致**（§3），全部 <3%（最粗 16 面元 2.62%），拟合收敛阶 −2.007 |
| repro_cmd | `python -m tensorbem.benchmarks.refinement_study --panels ...` | **可执行**：本审计在工作树外重跑，产物与受审工件**逐字节相同**（需补 `PYTHONPATH=src`，报告未写但仓惯例如此；不影响可复现性） |
| 报告六段 | ①问题+ref ②方法 ③参数表 ④对照表 ⑤收敛/敏感性 ⑥失败模式/工件（+⑦误差判定，超出要求） | `tensor_reports id=16183`（mission_id=17174，figures=1，3319 字符）：六段齐全、编号完整、结论显式（✓） |
| 代码实质 | 复用既有 `cylinder_added_mass` | `src/tensorbem/benchmarks/cylinder_added_mass.py` 真实实现（非占位），引入链 `f5a21e2 Add circular-cylinder added-mass benchmark` + `c18399a Derive added mass from BEM flux`（**均早于本单**，确认「复用不另起炉灶」属实） |

## 3. 数值对照（审计复算 · 数字来自 JSON，零手抄）

ref `ρπa² = 1025 × π × 1² = 3220.132469929538`（机器求值，非手抄）

| N | got `m_A` [kg/m] | 相对误差 | 相邻误差比 | 观测阶 | <3%? |
|---|---|---|---|---|---|
| 16 | 3304.605509 | 2.6233e-2 | — | — | ✓ 2.62% |
| 32 | 3241.000530 | 6.4805e-3 | 4.048 | 2.017 | ✓ 0.65% |
| 64 | 3225.325092 | 1.6125e-3 | 4.019 | 2.007 | ✓ 0.16% |
| 128 | 3221.428001 | 4.0232e-4 | 4.008 | 2.003 | ✓ 0.040% |
| 256 | 3220.456051 | 1.0049e-4 | 4.004 | 2.001 | ✓ 0.010% |

拟合阶（log-log 最小二乘）= **−2.0066**。二阶收敛 + 单调下降是**真离散化误差**的签名，
不是硬编码常数的签名（见 §4 反证）。

## 4. 反证探针（判 MOCK 的硬证据）

| 探针 | 做法 | 结果 | 结论 |
|---|---|---|---|
| 输入反证 | 把边界迹换成错误角向模 `φ=−U·cos2θ`（N=64/128），若实现里硬编码了 `ρπa²` 则结果不变 | m_A = 6465.685 / 6446.674，相对 `ρπa²` 误差 **100.8% / 100.2%** | 输出**强依赖输入**，非硬编码 → 排除 MOCK |
| 速度不变性 | `run(64)` vs `run(64, speed=3.0)` | 3225.3250921803083 vs 3225.325092180308 | `m_A=ρ/U²ΣφqΔs` 的 U 不变性成立（量纲自洽） |
| 报告 §5.2 椭圆声明复核 | `ellipse_added_mass(128, a=1.5,b=0.75)` x/y 向 | 7.5356e-4 / 8.9977e-4（报告写 7.54e-4 / 9.00e-4，且走**外域** Dirichlet 求解器） | 报告该条**如实**，且证明仓内确有外域算子能力 |

## 5. 发现（按严重度，均可复现）

**S1（中，非阻塞）交付未入仓，证据链靠工作树续命。**
`/root/tp-agent/work/m17174` 中 `docs/dev_report_m17174_cylinder_added_mass.md` 与 `artifacts/*`
均为 `??` 未跟踪；`git log main..agent/m17174` **为空**，生态仓 main 无对应文件。
即：run_artifacts 只活在会回收的工作树里，原单 `result.head_commit=null`。
*缓解*：能力代码在 main（96f19f7）且在**另一工作树**重跑得到逐字节相同工件 → 科研数据无损失；
本审计已把受审报告+工件快照存 `docs/board_audit/m17197/refs/`（sha256 见 manifest）以固化审计证据。

**S2（中，非阻塞·系统性）锚点无机器判据行。**
`SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug='CASE-BEM-CYL-ADDED-MASS-2D'` = **0**；
全表 18 行，而 2026-11-12 起 `task_key LIKE 'case-%'` 的单量 **254** → 约 236 个算例池单无注册判据，
回归时无机器基线可比。建议由算例池流程补注册（本审计**不代注册**，避免「审计者自造判据再自评」）：

```sql
INSERT INTO tensor_benchmark_cases (anchor_slug, case_name, ref_source, metric, tolerance, repo, status)
VALUES ('CASE-BEM-CYL-ADDED-MASS-2D','two_dimensional_translating_circular_cylinder_added_mass',
        'closed form m_A=rho*pi*a^2 (Lamb 1932; Newman 1977 §4.10); docs/marine_bem_benchmarks.md:62',
        '|m_A - rho*pi*a^2|/(rho*pi*a^2) at N=64 (monotone in N, 2nd order)','strict < 3% (audit: 0.161% at N=64)','TensorBEM','pending');
```

**S3（低，范围声明）本算例的算子是「有界域原型 + 精确解析边界迹」，不是外域辐射求解。**
`cylinder_added_mass.py:41-46` 施加精确 Dirichlet 迹 `φ=−Ua cosθ`，`solve_laplace_dirichlet` 解内域；
圆的 m=1 模内外同场，故 m_A 的验证在连续极限成立，但**无法暴露外域算子缺陷**。
原报告 §6 已明确声明此限制（未夸大），且仓内 `ellipse_added_mass.py` 用 `solve_laplace_exterior_dirichlet`
补上了外域检验。→ 因此级别上限 L4，不判 L5。

**S4（信息，建议）门禁阈值偏松。** `tests/test_cylinder_benchmark.py` 判 `err < 3.0e-2`，而 N=64 实测
1.61e-3（余量 ~18×）——一个数量级的性能回退仍能过门。建议按实测收至 `5.0e-3`（或按收敛阶设阈值），
并加一条 256 面元的趋势断言。

**S5（信息，工具链）`integration_callers` 的证据格式与 gate 复核语义不兼容。**
`maturity_gate.callers_verified()` 取 `callers[0].split(":")[0][:60]` 后 `grep -F` 仓内 `.py` 内容——
即该字段**必须是能在源码里逐字命中的 import 串**（不能写 `path:line`，冒号会截断成路径、grep 落空）。
本审计初版按「file:line」写法，用
`python docs/board_audit/m17197/simulate_gate.py` 预演得 `declared 4 → gate 2`（理由：`integration_callers(grep 无引用)`），
改为字面 import 串后预演 `gate_level=4, reasons=[]`。建议其它审计单沿用该格式（人读指针另放
`integration_callers_detail`），否则 L4 会被机器降级成 L2 —— 这是格式问题，不是实质不足。

## 6. 判级（L0–L5，级别=功能实质）

| 级 | 判据 | 本单 |
|---|---|---|
| L1 数学/单元验证 | 有测试名 | ✓ `tests/test_cylinder_benchmark.py`（2 passed） |
| L2 独立功能 | 真代码 输入→计算→输出 | ✓ `run(N)`：建网格→单层配点解 Laplace→边界能量积分→m_A |
| L3 系统集成 | 主求解器/平台侧调用 | ✓ 消费主求解器 `solve_laplace_dirichlet`；被 `benchmarks/refinement_study.py:18` 与门禁测试 import |
| **L4 基准验证** | 完整算例+误差分析+可复现 | ✓ 闭式 ref + 5 级网格 + 误差表 + 拟合阶 + 独立重跑逐字节一致 |
| L5 科研工程 | 另有文献/实验外部锚 | ✗ 教科书闭式 + 迹级验证，无新意 → 拒绝 |

**结论：L4**（`tensor_maturity_verdicts`: repo=TensorBEM, scope=`m17174 CASE-BEM-CYL-ADDED-MASS-2D`, mission_id=17197, level=4, verdict_by=board-audit）。

## 7. 独立性局限与被夸大声明

- **局限**：本审计是**同一份 main 代码的重跑**（确定性复现），非独立实现；正确性的最终依托是
  `ρπa²` 闭式本身（经典结果）与仓内椭圆外域算例的一致性，而非交叉实现。
- **被夸大声明**：未发现。mission headline「复用既有 cylinder_added_mass 能力」与代码引入历史
  (f5a21e2/c18399a) 一致；报告 §6 主动声明不覆盖一般外域/自由液面/波浪体，与代码事实相符；
  报告数字与工件逐项相符（含椭圆 7.54e-4/9.00e-4 的复算）。
- **未跑的检查**（状态=not_run，不记作 pass）：仓全量 pytest 套件（规模大，非本单判据所需）；
  loong64 集群节点复算（本机 x86_64，数值为确定性，架构无关但未在集群上验证）；
  自由液面/深水 Ur sell 族算例（不在本单范围）。

## 8. 工件清单

| 文件 | 说明 |
|---|---|
| `docs/board_audit/m17197/rerun/refinement_cylinder_repro.json` | 审计独立重跑输出（与受审工件逐字节相同） |
| `docs/board_audit/m17197/rerun/refinement_cylinder_repro.stdout.txt` | CLI stdout（同内容，证无额外非确定性） |
| `docs/board_audit/m17197/rerun/fig_audit_cylinder_added_mass.png` | 收敛图（左：m_A vs N 与闭式；右：log-log 误差 + N⁻² 参考线 + 3% 线） |
| `docs/board_audit/m17197/rerun/make_audit_figure.py` | 图生成脚本（数字由 JSON 注入） |
| `docs/board_audit/m17197/refs/` | 受审报告 + 受审工件快照 + sha256 manifest |
| `docs/board_audit/m17197/insert_verdict.py` | verdict 落表脚本（幂等，evidence 由计算生成） |
| `docs/board_audit/m17197/simulate_gate.py` | 用平台 gate 的 `judge()` 预演复核（不写库，防格式性降级） |

## 9. 落表与 gate 预演

```
python docs/board_audit/m17197/simulate_gate.py   -> declared 4, gate_level 4, reasons []
python docs/board_audit/m17197/insert_verdict.py  -> verdict id=490, gate_state=pending
SELECT id,repo,scope,mission_id,level,verdict_by,gate_state FROM tensor_maturity_verdicts WHERE id=490;
-- 490 | TensorBEM | m17174 CASE-BEM-CYL-ADDED-MASS-2D | 17197 | 4 | board-audit | pending
```
gate 守护（30 min 周期）复核通过即 `confirmed`（消账）；若被判 `reopened`，请在 `gate_note` 中核对
上表逐项指针而非仅看级别数字。