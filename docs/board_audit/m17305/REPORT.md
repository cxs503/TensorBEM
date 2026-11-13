# 交差核验 · m17246 `CASE-BEM-NEUMANN-KELVIN`（Neumann-Kelvin 定常兴波阻力 · TensorBEM）

**审计单** m17305 · **对象** mission 17246（succeeded）· **对象提交** `27ecfe7`
「NK steady wave resistance case vs Michell Cw(Fr): exact Kochin prefactor kappa/(2 pi g)…」
**角色** board-audit（独立复核，非重做）· **独立性说明**：本审计与交付方不共享推理链；
所有数字来自交付工件 `docs/case_bem_neumann_kelvin/numbers.json` + 审计方自写求积
（`scripts/audit/m17305_independent_michell.py`，不 import TensorBEM）+ 审计方重跑（`runs/m17305/`）。

**处置（disposition）**：**approve with notes** → 判级 **L4**（凭证据表见下），
附 6 条严重度分级发现；其中 F1（基准裁判缺行）由本审计现场补登记，
F5（仓内既有 "verified" Michell 基准 ref 偏小 10.3–12.4×）为**新欠账**，需另开返修单。

---

## ① 案例与验收契约

均匀来流 `U` 绕 Wigley III 抛物型船型（`L/B/T = 10/1/0.625 m`，`B/L=0.1`，`Cb=4/9`）深水定常兴波；
锚点 `CASE-BEM-NEUMANN-KELVIN`（算例池 `tensor_case_pool` id=167，
`ref_source = Michell 薄船理论 C_w(Fr)`，`tol_policy = <5%`）。
交付声明的主结果：Fr = 0.20/0.25/0.30/0.35/0.40 五点相对误差 **+0.197 % … +2.309 %（全部 <5 %）**。

三件套核验（是否存在、是否可执行）：

| 项 | 交付声明 | 审计核验 | 结论 |
|---|---|---|---|
| run_artifacts | `docs/case_bem_neumann_kelvin/{numbers.json,table.md,3×png,wave_field.npz}` | 7 个文件全在 worktree（17739 / 550 / 79365 / 75178 / 48597 / 6734 B），`wave_field.npz` 可读（`zeta` 41×25） | **真存在** |
| benchmark ref vs got | 见 ④ 表 | ref 由审计方**独立重推**（自写 Gauss–Legendre 求积）复核，两路 ref 互差 ≤ 0.041 % | **成立** |
| repro_cmd | `PYTHONPATH=src python3 scripts/case_bem_neumann_kelvin_michell.py --stage cheap --only geometry,kernel,route_a,route_b,wave_field` | 在交付提交 `27ecfe7` 的干净副本重跑：`numbers.json` 全部数值叶子**逐一复现**（除 `generated_at_utc`/`git_head`/`python`/`platform`/`seconds`/`blocks_run` 外 0 差异） | **可执行且可复现** |
| 报告六段 | ①–⑥ 齐全（137 行） | ①案例+ref出处 ②方法 ③参数表 ④对照表 ⑤收敛/敏感性 ⑥失败模式+工件+摩擦 | **齐全** |

基准裁判（`tensor_benchmark_cases WHERE anchor_slug='CASE-BEM-NEUMANN-KELVIN'`）：
**审计开始时 0 行**（全表 41 行、TensorBEM 仅 43/44 两行）——见 **F1**。本案按算例池
id=167 的 `tol_policy` 由本审计现场登记（`ref_source`=Michell 薄船 Cw(Fr)，`tolerance`=<5%，
`status`=passed），登记内容与 `runs/m17305/audit_numbers.json` 同源。

## ② 怎么做的（审计流程）

1. **取证**：读 `27ecfe7` 的 diff（`src/tensorbem/nk_panel3d.py` 前因子修改、499 行 case runner、
   97 行 5 项回归测试、报告+3 图），读 `numbers.json` 全部 21 个数值块。
2. **独立重推 ref**：`scripts/audit/m17305_independent_michell.py` —— 不 import TensorBEM，
   按教科书 Michell 式 `R_w = 4ρg²/(πU²)∫₀^{π/2} sec³θ|A|²dθ`，
   `A = ∫∫(∂b/∂x)e^{κsec²θ z}e^{iκsecθ x}dxdz`，中心面 160×480 Gauss–Legendre、θ 2001 点。
3. **重跑验收链**：把 `27ecfe7` 工作树整体复制到 `runs/m17305/repro_tree/`，
   跑 `repro_cmd` → 与交付 `numbers.json` 逐叶比对；`pytest tests/test_case_neumann_kelvin_michell.py`
   与 `tests/test_nk_panel3d.py + tests/test_neumann_kelvin.py`（前因子默认值改动的回归面）。
4. **反演陷阱**：`verify_radiation_direction` 两种辐射模式实调复核报告 3.02 的说法；
   旧出口 `wigley.michell_wave_resistance` 与 `benchmarks/verified/bem_wigley_wave_resistance/results.json`
   的 ref 数值三方对质。
5. **判级**：按 L0–L5 功能实质（不按报告篇幅），evidence 七键逐条给真实指针，`maturity_gate.judge` 机器复核。

## ③ 复核数字表（程序化生成：`docs/board_audit/m17305/table_audit.md`）

| Fr | ref Rw [N] (delivered) | ref Rw [N] (auditor, own quadrature) | ref spread | got Rw [N] | got vs indep ref | tol <5% |
|---:|---:|---:|---:|---:|---:|:--:|
| 0.20 | 26.53927 | 26.55001 | +0.0405% | 27.1520 | +2.2673% | PASS |
| 0.25 | 49.71925 | 49.72591 | +0.0134% | 50.2540 | +1.0620% | PASS |
| 0.30 | 144.13569 | 144.14019 | +0.0031% | 144.8185 | +0.4706% | PASS |
| 0.35 | 114.31457 | 114.31769 | +0.0027% | 114.7902 | +0.4134% | PASS |
| 0.40 | 327.11011 | 327.11249 | +0.0007% | 327.7550 | +0.1964% | PASS |

- 审计方独立 ref 与交付 ref 的最大互差 **0.0405 %**（Fr=0.20）→ 交付 ref 数值**正确**，非自证。
- got vs **审计方独立 ref** 最大 **+2.2673 %**（Fr=0.20）→ 同时满足算例池 `<5 %` 与生态 `<3 %` 线。
- Route A 面板阶梯（交付）：450→1800 面板，Fr=0.20 误差 11.418 %→2.309 %（单调）；θ 节点
  501→8001 误差波动 ≤4e-4；ref λ 截断 20→60 变化 ≤1.2e-4 N。
- 核函数自检（交付 `numbers.json:kernel`）：自由面条件 `max|G_xx+κG_z|/scale = 4.532e-7`（审计方复算同值）。
- Route B（真实船体面元 NK 单层法）：`max_abs_error = 49.5672`（即 +4957 %），`any_passed=false`
  → 交付如实登记为 honest FAIL，本审计**不采纳为证据**。
- 回归面：`test_nk_panel3d.py + test_neumann_kelvin.py` 审计重跑 **64 passed in 18.42 s**；
  case 测试 **5 passed in 5.68 s** → 交付声称的 "38+26 未受影响 / 5 passed" 属实。

## ④ 结果图

`fig_audit_cw_vs_fr.png`（左）got 与两条独立 ref 路线的 `C_w(Fr)`；（右）误差随 Fr 的变化与 5 %/3 % 判据线。
交付的 `fig_cw_vs_froude.png` / `fig_route_a_convergence.png` / `fig_kelvin_wake.png` 已核为
非空 png（79/75/48 kB），其中波形图按交付方声明仅作定性（见 F6）。

## ⑤ 注意点（审计发现，按严重度）

- **F1 · high（平台，系统性）**：`tensor_benchmark_cases` 对锚点 `CASE-BEM-NEUMANN-KELVIN`
  **0 行**（算例池 id=167 只有 `tensor_case_pool` 记录、`status=dispatched`），
  即"基准判据=唯一裁判"在交付时刻**不存在**，机器无从裁。与 verdict #493/#499/#509/#514 报告的
  同一断点（case-pool 派发不写 benchmark 行）。本审计已补登记，但**根因未修**。
- **F2 · medium（溯源）**：交付 `numbers.json` 记 `git_head = 0516776`（**改前父提交**，不含
  `kochin_prefactor`），且 `blocks_run = ["route_b"]` —— 说明该工件是多次分块执行的**合并结果**、
  生成于脏工作树。审计方只能靠"在 `27ecfe7` 重跑数值逐一相同"来建立归因，**无法**仅凭记录的 SHA 归因。
- **F3 · medium（交付面）**：`27ecfe7` 只在 `agent/m17246` 分支，`origin/main`（核验时 `89e5a52`；
  现 `c954f19`）既无 `docs/case_bem_neumann_kelvin/` 也无 `kochin_prefactor` →
  干净 main 检出**不可复现**，须等平台推分支（同 #499/#514 F2 的流程事实，非作者缺陷）。
- **F4 · medium（判级边界，本案核心判断）**：通过路线（Route A，薄船源 `σ = -2U∂b/∂x` + 稳态
  Kochin 远场算子）与 ref（Michell 积分）**结构同源**（`nk_panel3d.wave_resistance_far` 的
  docstring 自行证明二者在 `H = -2UA` 下等价）。故 0.20–2.31 % 的吻合证明的是
  **稳态色散 + 留数常数（`κ/(2πg)`，正是原缺陷处）+ 无量纲归一 + 中心面求积收敛**，
  **不证明**真实船体面元 NK 单层法（Route B，+1399 %…+4957 % 且随加密发散）。
  交付方在报告 §6 已主动声明此边界，本审计复核声明**准确、无粉饰**；但 L4 只能按此缩小后的语义读。
- **F5 · medium（既有欠账，复核确认）**：仓内 "verified" 基准
  `benchmarks/verified/bem_wigley_wave_resistance/results.json` 的 `Rw_reference` 取自被判 DEFECTIVE 的旧出口
  `wigley.michell_wave_resistance`：Fr=0.20/0.30/0.40 → `2.5752 / 12.2926 / 26.3239 N`，
  而正确值 `26.5393 / 144.1357 / 327.1101 N`（**偏小 10.31× / 11.73× / 12.43×，且随 Fr 变**）。
  审计方独立求积与交付 ref 均给出正确值（互差 ≤0.041 %）。交付报告 §6-2 已点出此债（11.7×），
  本审计把它量化到两端并确认**该 "verified" 基准的 ref 必须重出**（新返修单）。
- **F6 · low（工件内部一致性）**：(a) `route_a.richardson_rw_extrapolated/observed_order = NaN`
  （2 层阶梯不足），报告 §5 以文字给出观测阶 ~1.15，未入工件；
  (b) 报告 §2 "辐射后/前比 3.02" 未写入 `numbers.json`（只有 `downstream` 模式的 0.9858 入库）；
  审计方实调 `nk.verify_radiation_direction(1.0, radiation='halfplane', sign=-1.0)` 得 **3.019989** →
  数字**正确**但属"零手抄"承诺的例外；(c) 波形张角测得 74.60° vs 理论 19.47°，交付已声明估计器粗糙、仅定性。

## ⑥ 误差判定（逐项，>3% 即不可接受）

| 关键量 | 基准 | got | 误差 | <3%? |
|---|---|---|---|---|
| `C_w(Fr)`，Fr=0.20–0.40，最坏点 | Michell 薄船（交付 ref） | 见 ③ 表 | **+2.309 %** | 是 |
| `C_w(Fr)`，同，对**审计方独立 ref** | 审计自写求积 | 见 ③ 表 | **+2.267 %** | 是 |
| ref 自身正确性 | 审计自写求积 | 交付 canonical | ≤0.0405 % | 是 |
| 稳态核自由面条件残差 | 0 | 4.53e-7（相对） | — | 是 |
| Route B 面元 NK `C_w` | Michell | +1399 %…+4957 % | **FAIL** | 否（honest FAIL，不入账） |
| 近场压力积分路线 | 远场算子 | 稳定 0.51× | −49 % | 否（交付如实登记缺陷） |

**判级依据（L4）**：impl 真代码 + 库级入口（`nk_panel3d.wave_resistance_far` / `thin_ship_reference`，
调用方 `scripts/nk_panel3d_study.py:199`）+ 用例工件齐 + 5 项回归测试 + 完整算例（5 Fr × 面板/θ/λ 三类敏感性）
+ 误差分析 + 可复现（审计重跑逐叶相同）+ **外部解析锚 Michell 1898 经审计独立重推**。
**不判 L5**：无实验（拖曳水池）锚，且真实船体 BEM 路线 WITHHELD；报告篇幅不改变级别。

## 未跑的检查（记 `not_run`，不记 pass）

- `--stage full` 全量阶梯（本机 120 s 单命令上限；cheap 已逐叶复现交付数字，故不影响判级）。
- 集群侧独立复跑（`run_cluster`）与跨架构（loong64）一致性。
- 波形张角的独立测量器复算（接受交付的"仅定性"声明）。
- 与 Wigley 1934/ITTC 拖曳实验的外部对照（未做 → 故不申 L5）。