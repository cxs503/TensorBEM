# 交差核验审计 — m17275 `CASE-BEM-ICE-SPHERE-HEAVE`（TensorBEM）

审计单 **m17312**（board-audit）｜被审单 **m17275**（succeeded，commit `a35c7c0`）｜锚点
`CASE-BEM-ICE-SPHERE-HEAVE`（case pool id 172，`feat-casepool-100`）｜审计性质=**核验单，非重做**。

所有数字由 `scripts/audit/audit_m17312_ice_sphere_heave.py` 从被审工件
`runs/m17312_audit/source_m17275/artifacts/case_bem_ice_sphere_heave.json`（sha256
`9d051609…64a5f2`）程序化重算，零手抄；汇总落 `runs/m17312_audit/audit_summary.json`。

## 1 案例与判据（问题侧）

半径 R=5 m 的刚性球心潜深 d 于冰/水界面之下，受迫做单位速度垂荡；水深 H=1000 m、冰盖按
mass-loading（Robin 条件 dφ/dz = k_i φ，k_i=ω²/(g−μω²)，μ=ρ_i h_i/ρ_w）。参考量 =
Lamb 无限流体垂荡附加质量 A33^Lamb=(2/3)πρR³（Lamb 1932 art. 92；= 排水量的一半）。

判据侧实况（`tensor_benchmark_cases` 与 `tensor_case_pool` 双查）：

| 判据来源 | 内容 | 审计时状态 |
|---|---|---|
| `tensor_case_pool` id 172 | ref_source=「Lamb 无限流体 (2/3)πρR³」；tol_policy=`<5%` | 存在，status=`dispatched`，linked_anchor 空 |
| `tensor_benchmark_cases` (anchor_slug=…) | **0 行** | 缺失（F2）；本审计按池内 ref/容差判，并把该判据登记入表 |

即：判据可判定（池内 ref + <5%），但机器裁判表当时为空——报告再好看也不能替代登记。

## 2 审计方法

1. **取证**：被审工作树 `/root/tp-agent/work/m17275`（HEAD `a35c7c0`）→ 12 个交付件快照入
   `runs/m17312_audit/source_m17275/`（代码/测试/文档/渲染工具/JSON/日志），sha256 留痕。
2. **独立重跑**：`PYTHONPATH=src python -m tensorbem.benchmarks.ice_sphere_heave --workers 8`
   （24 点全矩阵，76 s，日志 `runs/m17312_audit/`），产出 `rerun_full_matrix.json`。
3. **独立重算**：Lamb 闭式从零求值（不调用仓内 helper）；多极参考直接调
   `tensorbem.analytic.submerged_sphere_wave_multipole`；P1–P5 由逐行数值重建；收敛阶自行最小二乘；
   k_i 由色散关系重推（不用代码常量）。
4. **反证探针**：误差分解（物理偏移 vs 离散欠冲）、P5 门阈值松弛度、Lamb 参考在 d=3R 的
   *下界*（P1 能否被离散误差"偶然"满足）、文档敏感性声明的实际跨度、`--quick` 模式的门语义。
5. **落地核验**：`git ls-remote origin refs/heads/agent/m17275`（空）、`git merge-base --is-ancestor
   a35c7c0 HEAD`（NO，main=96f19f7）、`tensor_benchmark_cases` 计数、`tensor_capability_deltas`。

## 3 结果（数字表）

### 3.1 基准判据：ref vs got

| 项 | 值 |
|---|---|
| ref（本审计重算） | A33^Lamb = 2/3·π·1025·5³ = **268344.3724941281 kg**（= 排水量一半，残差 0） |
| 交付 ref 值 | 268344.3724941281 kg（相对差 **0.0**） |
| 主判据点 got（d/R=3, h_i=0.9 m, ω=0.5, 352 panels） | **263979.1242564912 kg** = 2.060324872 ρR³ |
| 相对误差 | **1.6267 % < 5 %** ✓（与交付值差 0.0） |
| 全矩阵 24 点 | 22/24 点 <5 %；最坏 `depth__2R__fine` 6.191 %（d/R=2 近场，文档已声明）；18/24 点 <3 % |

### 3.2 门 P1–P5 由审计重建

| 门 | 审计重算 | 交付 | 一致 |
|---|---|---|---|
| P1 主点 vs Lamb <5 % | True（1.6267 %） | True | ✓ |
| P2 主点 vs 多极 <5 % | True（4.4511 %） | True | ✓ |
| P3a 梯度对多极单调 | True（8.00→4.45→2.01→1.14 %） | True | ✓ |
| P3b 观测阶 ∈[0.3,3] | True（**p=1.8076**，与交付差 1.6e−15） | True | ✓ |
| P4a d/R=12 vs Lamb <5 % | True（4.361 %） | True | ✓ |
| P4b d/R=12 精确偏移 <0.5 % | True（−0.01297 %） | True | ✓ |
| P5 冰控 | True（冰厚 0→0.3 m 改变 8.34e−5） | True | ✓ |

七项全一致；`audit_summary.json.gates.all_agree=true`。

### 3.3 误差分解（关键发现 F3）

| 分量 | 数值 |
|---|---|
| 物理偏移 (A33^exact−A33^Lamb)/A33^Lamb | **+2.9559 %** |
| 离散欠冲 (A33^exact−A33^BEM)/A33^exact | **4.4511 %**（折算到 Lamb 尺度 +4.5826 %） |
| 净误差 (Lamb−A33^BEM)/A33^Lamb | **−1.6267 %** |
| 代数残差 | 0.0 |
| 抵消倍数 | **4.63×**（1.63 % 是两个 3–4.6 % 误差相消的余量） |
| Lamb 参考下界（d=3R） | 误差对 Lamb 不可能低于 +2.96 %；对 Lamb 的梯度 5.28/1.63/0.89/1.79 % **非单调** |

### 3.4 参考链独立锚

| 检查 | 结果 |
|---|---|
| Lamb 闭式 = 排水量一半 | 残差 0 |
| 多极 d/R=20 → Lamb | +0.0058 %（<0.05 % ✓，证明参考不是抄来的） |
| 多极 d/R=3, kR=1e−3 vs 经典镜像项 3/8(a/f)³=1.3889 % | 1.4394 %（相对差 3.6 %，量级/符号一致 ✓） |
| k_i 由色散关系重推（ρ_i=917, h=0.9, ω=0.5） | k_iR=0.13013571059747780，与工件差 **0.0** |
| 冰 vs 开阔水波数 | k_i/k_0=1.021（**仅 +2.1 %**，与文档"冰改变到达极限的速率"一致但作用极小） |

### 3.5 复现性与零手抄

| 检查 | 结果 |
|---|---|
| 审计重跑 vs 交付工件 | **sha256 完全相同**（`9d051609…`），24/24 行 max 相对偏差 **0.0**，门字典逐键相同 |
| 单点 `--quick` 重跑 | 主点误差 1.6267336620716 %（差 2e−16） |
| 测试 | `pytest tests/test_ice_sphere_heave_case.py -q` → **5 passed, 93.6 s** |
| 表格零手抄 | 用工件重跑 `tools/ice_sphere_heave_report.py` → 100 行中 96 行逐字相同，4 行差异仅因审计传入的 `--figdir` 目录名；**无任何数值经人手转录**（散文里也未出现 6 位小数 kg 值） |
| 报告六段 | 7 个 `##` 段齐全（问题/方法/参数/对照/收敛敏感性/失败模式/复现） |

## 4 图

- `docs/board_audit/m17312/fig1_ladder_error.png` — 梯度：对多极单调、对 Lamb 非单调 + 2.96 % 下界。
- `docs/board_audit/m17312/fig2_error_decomposition.png` — 1.63 % = +2.96 % 与 −4.58 % 相消。
- `docs/board_audit/m17312/fig3_submergence.png` — 潜深扫描：误差底线是分辨率不是潜深。
- `docs/board_audit/m17312/fig4_sensitivity.png` — 敏感性实际跨度（频率 1.52 %，非"<1 %"）。

## 5 发现（按严重度）

| # | 严重度 | 发现 | 证据 |
|---|---|---|---|
| F1 | **blocking（落地）** | 交付物**未落地**：origin 无 `agent/m17275` 分支（`git ls-remote` 空），TensorBEM main（96f19f7）不含 `a35c7c0`；工作树 `worktree_recyclable` 一旦回收，能力即蒸发。任务 result 里 `branch: null`、`server_quality_audit.gate_valid=false` 佐证自动推送未触发 | `/root/tensor-ecosystem/TensorBEM` main；`/root/tp-agent/work/m17275` 头 |
| F2 | high（判据） | `tensor_benchmark_cases` 对本锚点 **0 行**，机器裁判表缺登记；`tensor_case_pool` 172 仍 `dispatched`、`linked_anchor` 空。本次审计按池内 ref+`<5%` 判为通过并补登记一行 | SQL 计数；§1 表 |
| F3 | high（读数） | 主点 1.63 % 是 **+2.96 % 物理偏移 − 4.58 % 离散欠冲** 的余量（抵消 4.63×），非收敛精度；BEM 对多极在全部 4 个梯度上**单向偏低**，对 Lamb 的误差存在 +2.96 % 下界，故对 Lamb 梯度非单调。作者已在文档 §4 明示该分解，**未把 1.63 % 说成收敛精度**——属"读数须带分解"，非造假 | §3.3 |
| F4 | medium（门） | P5 阈值 = 主点对多极误差 + 1e−3 = **4.55 %**，观测位移 8.34e−5 → **松弛 546×**；该门只在"冰改变 A33 >4.5 %"时才会失败，实际几乎不可能 FAIL，不构成有效控制 | `gate_weakness.p5_slack_factor` |
| F5 | medium（测试覆盖） | 5 条测试只钉住 P1/P2 与参考链；**P3a/P3b/P4a/P4b/P5 五个门无测试锁定**，后续改动可静默翻转而不红 | `tests/test_ice_sphere_heave_case.py` |
| F6 | low（文档过度） | 文档称频率扫描"A33 flat to a fraction of a percent"，实测跨度 h=0.9 m **1.016 %**、h=2.5 m **1.516 %**，不满足 <1 % | §3.5 表 / fig4 |
| F7 | low（语义） | `--quick` 只跑单点却照常输出 `gate_ladder_order_in_band (P3b): False`、`ladder_observed_order: nan`，易被误读为失败；建议 quick 模式不出 P3 键 | `rerun_quick.json` |
| F8 | info | `kernel_at_ki_identity_relative_difference` 恒 0（同函数自比），是恒等式诊断而非独立校验；文档未声张，不影响结论 | 工件行 `ladder__352` |
| F9 | info（能力边界） | 本算例**不**证明冰致附加质量增量（冰改变 A33 ≤8e−5，低于离散误差）；验证的是冰核 bookkeeping（ν=k_i）+ 深潜极限。文档 §6 已如实声明 | 工件 `ice__*` 行 |

## 6 判级与处置

- **L1 数学/单元**：参考闭式 + 多极极限 + 3/8 镜像项三路互证 ✓
- **L2 独立功能实现**：`src/tensorbem/benchmarks/ice_sphere_heave.py`（562 行，输入→BEM→JSON，无占位）✓
- **L3 系统集成**：调用主求解链 `ice_rao.solve_radiation_3d_ice → free_surface_solver3d.solve_radiation_3d`，复用 `ice_ship_verification.submerged_sphere_mesh` 与 `analytic.submerged_sphere_wave_multipole` ✓
- **L4 基准验证**：完整算例（24 点矩阵+参数表）+ 误差分析（三参收敛阶 p=1.8076、潜深/冰厚/频率敏感性、物理-离散分解）+ 可复现（全矩阵 sha256 一致、5 tests 绿、repro 命令实测）✓
- **L5**：不做（无实验/印版表锚，冰致增量不可测）

**裁定：L4（confirmed 侧），附 F1 blocking 落地缺陷。** 理由：功能实质是"对既有冰区辐射能力的基准验证"，三件套与误差分析齐备且重跑逐位一致；F1 属交付落地管道缺陷（非算法错误），不改变功能级别，但**必须在工作树被回收前把分支推上 GitHub 并合并**，否则该 L4 的证据指针随工作树失效（本审计已将被审件快照+sha256 固化在 `runs/m17312_audit/source_m17275/`）。建议后续单：① 推送 `agent/m17275` 并合并；② 把 `tensor_case_pool` 172 置 `done` 并补 `linked_anchor`；③ 给 P3/P4/P5 加断言（F5）、把 P5 阈值改成分辨率无关量（F4）。

**未做的检查**：未做外部求解器（Capytaine/OpenFOAM）交叉验证（本机无该路由的冰盖核）；未做 p 值随 `n_theta/n_phi` 各向拆分（只按 N^−1/2 单一 h 标度）；未核 1472 面板以上的渐近平台；未核 `surface_potential` 场量物理正确性（只核长度=面板数）；未判定"冰区锚点以 Lamb 无限流体为判据"是否满足 BP 意图（属锚点 owner 口径问题）。