# Board audit — m19973 `CASE-BEM-HELMHOLTZ-SPHERE`（交差核验单，非重做）

- **审计对象**：mission 19973（TensorBEM，`task_key=case-bem-helmholtz-sphere`，status=succeeded，`claimed_by=tp-doc-2`，model=qwen3.6-plus，48 tool_calls / 1719 s）
- **被测交付物**：`/root/tp-agent/work/m19973/benchmarks/verified/case_bem_helmholtz_sphere/`（worktree HEAD `635415e1`，目录 **untracked**）
- **审计方**：m19997 / `board-audit`；审计工作树 `/root/tp-agent/work/m19997`（全新 checkout 自 `main=635415e1`）
- **结论**：级别 **L3**（impl/integration/tests/repro/analytic-anchor 全部可核）；**L4 拒绝**——两处交付态硬缺（F1 未入库、F2 池裁判行缺失）。disposition = **request changes（返修）**。
- **证据等级**：ref = **REAL**（三条独立路线复算至 1e-15）；got = **REAL**（14 例全部 bit-identical 复现）；无 MOCK/STUB 发现。

---

## 1. 取证（三件套逐项）

| 项 | 声称 | 实测 | 判定 |
|---|---|---|---|
| run_artifacts 存在 | results.json / report.md / 4×png | 8 个文件确实存在（`results.json` 8299 B、`report.md` 7694 B、4 个 png 共 604 kB、`report.json` 814 kB） | **真**（但见 F1：未入库） |
| ref vs got 数值 | 14 例，13 PASS / 1 FAIL，max 6.119% | 逐行重算一致（§2、§3） | **真** |
| repro_cmd 可执行 | `OMP_NUM_THREADS=1 python3 benchmarks/verified/case_bem_helmholtz_sphere/run_study.py` | 审计方在自己的工作树里以同一脚本重跑：**14/14 行非计时字段 0 差异**（§4） | **真** |
| 报告六段 | 六段 | `tensor_reports.id=18218` 的 markdown 与本地 `report.md` **md5 完全相同**（`8092c919…`），六段标题齐全，4 图内嵌 | **真** |

## 2. 基准裁判（唯一裁判 = 库表）

```sql
SELECT case_name,metric,tolerance FROM tensor_benchmark_cases
 WHERE anchor_slug='CASE-BEM-HELMHOLTZ-SPHERE';      -- >>> 0 rows
```

- 全表 **121 行 / 84 个 anchor**，`anchor_slug ILIKE '%HELMHOLTZ%'` 只有 1 行，且是**另一例** `CASE-BEM-ACOUSTIC-PULSE`（m17210 的时域脉冲算例）。
- 因此本算例的容差判据（任务书 `<6%`）**只存在于任务书文本里，不在裁判表里**：平台侧无法机器核 `got` 是否在容差内。按「基准判据是唯一裁判，报告好看不算数」，L4 基准腿不成立 → **F2**。

## 3. 数值核验（全部程序化生成，零手抄）

`runs/audit_m19973/audited_case/results.json`（m19973 原始载荷）：

| ka | 面板 | σ_BEM [m²] | σ_Mie [m²] | 误差 | <6%? |
|---|---|---|---|---|---|
| 0.25 | 320 | 0.008454 | 0.009005 | **6.119%** | ✗ FAIL |
| 0.50 | 320 | 0.114832 | 0.121716 | 5.656% | ✓ |
| 1.00 | 320 | 0.971037 | 1.010427 | 3.898% | ✓ |
| 2.00 | 320 | 2.408908 | 2.455155 | 1.884% | ✓ |
| 3.00 | 320 | 3.190582 | 3.246837 | 1.733% | ✓ |
| 4.00 | 320 | 3.682558 | 3.748197 | 1.751% | ✓ |
| 5.00 | 320 | 4.051102 | 4.094637 | 1.063% | ✓ |
| 0.25 | 1280 | 0.008864 | 0.009005 | 1.572% | ✓ |
| 0.50 | 1280 | 0.119952 | 0.121716 | 1.449% | ✓ |
| 1.00 | 1280 | 1.000427 | 1.010427 | 0.990% | ✓ |
| 2.00 | 1280 | 2.443080 | 2.455155 | 0.492% | ✓ |
| 3.00 | 1280 | 3.224896 | 3.246837 | 0.676% | ✓ |
| 4.00 | 1280 | 3.727362 | 3.748197 | 0.556% | ✓ |
| 5.00 | 1280 | 4.082951 | 4.094637 | 0.285% | ✓ |

- 声称 «13 PASS / 1 FAIL，max 6.119%» 与载荷**逐项相符**（脚本统计，非人工核对）。
- 关键量是否 **<3%**？→ **不是全部**。1280 面板档：6/7 例 <3%（max 1.572%），但 ka=0.25 为 1.572%（<3% ✓）；实际 >3% 的只有 320 面板档的 ka≤1.0（6.119/5.656/3.898%）与 ka=2.0 的 1.884%（<3% ✓）。诚实结论：**声称的 6% 判据 13/14 通过；3% 判据在 320 面板档 3 例不通过、1280 面板档全通过**。

## 4. ref 腿独立复算（审计方，`runs/audit_m19973/indep_check.py`）

四条互相独立的路线，全部在本审计工作树内运行：

| 路线 | 做法 | 与仓库 ref 的最大相对差 |
|---|---|---|
| A | mpmath 从零实现 Riccati–Bessel（半整数阶 Bessel 拼装，不经 scipy）算 Mie 级数 | 1.14e-15 |
| B | 用 Mie 远场幅值 `f(θ)` 直接角向积分 `∫|f|²dΩ`（不走 Parseval 捷径，GL 240 点） | 9.11e-15 |
| C | 光学定理 `σ = (4π/k)Im f(0)` | 1.09e-15 |
| D | 小 ka 瑞利极限 `σ → (7π/9)(ka)⁴a²`：ka=0.02 差 3.8e-4、ka=0.05 差 2.4e-3、ka=0.10 差 9.5e-3 | 与渐近律一致 |

→ **ref 列是 REAL 解析解，不是同码自证**（虽然 ref 由仓库内 `acoustic3d.rigid_sphere_scattering_cross_section` 算出，但四路线独立确认其到达 1e-15 精度）。产物 `runs/audit_m19973/indep_check.json`。

## 5. got 腿复现（审计方重跑）

- 命令：`cd runs/audit_m19973/repro && OMP_NUM_THREADS=1 python3 run_study.py`（`run_study.py` 为 m19973 原件的逐字节副本，`sha256` 见 `runs/audit_m19973/AUDITED_CASE_SHA256.txt`）
- 结果：`runs/audit_m19973/repro.log`、`repro/results.json`。**14/14 行**的 `sigma_bem / sigma_mie / relative_error / pass / panel_count` **完全一致（max |Δσ| = 0.0）**，仅 `solve_time_s` 变动（0.96–1.51×）。
- 单进程墙钟 ≈ 3 min（1280 面板 8.7–12.7 s/例，与报告 §6 的 "12–19 s" 同量级）。
- 独立算力递交记录：审计方曾递交 cj2430（stage_dir 直送 8 核），因本地重跑已完成而 kill，未占用。

## 6. FAIL 例成因核实（对抗性探针）

任务书要求「>15% 不可接受」，本例 6.12% 已声明 FAIL，但需确认它是**几何离散而非 bug/quadrature 噪声**：

| 探针 | 结果 | 说明 |
|---|---|---|
| 远场求积收敛（64→96→128→192 点） | 相对漂移 **≤ 5.26e-15**（ka=0.25/2.0/5.0，320 与 1280 面板） | σ 的 6.12% 误差**与求积无关** |
| 内接多面体面积亏损 | 320 面板亏损 1.882%，1280 面板 0.476%，比值 **3.95** | 与观测误差比 6.119/1.572 = **3.89** 吻合 |
| 误差符号 | 14/14 全部 σ_BEM < σ_Mie | 内接网格只会低估有效半径（σ∝a⁶ 低频 / ∝a² 光学），系统性负偏差可解释 |

→ 报告 §6 的「flat-panel 几何离散极限」结论**成立**，不是掩盖。

## 7. 发现（按严重度）

- **F1 BLOCKING（交付态）**：`git status` = `?? benchmarks/verified/case_bem_helmholtz_sphere/`；`git log --oneline -- <dir>` **空**；mission result 里 `branch: None`。即：整套交付物**从未进版本库**。证据：`/root/tp-agent/work/m19997`（全新 checkout 自 `main=635415e1`）**不含**该目录，审计方只能从 m19973 工作树拷贝。→ 换机器/工作树回收后不可复现。返修：提交该目录（含 `run_study.py` 与 `results.json`）。
- **F2 BLOCKING（基准腿）**：`tensor_benchmark_cases` 对 `anchor_slug='CASE-BEM-HELMHOLTZ-SPHERE'` **0 行**（§2）。池裁判无法裁定，L4 拒绝。返修：登记 `(anchor_slug, case_name, metric='σ 后向散射截面 vs Mie 级数', tolerance='<6%', ref_source='Mie 级数 / Bowman-Senior-Uslenghi 1969', repo='TensorBEM')`。
- **F3 MEDIUM（可回归性）**：`benchmarks/verified/README.md` 未收录本算例（全仓 `grep HELMHOLTZ-SPHERE` 除本目录外 0 命中）；未新增 pytest 用例，唯一守卫是既有的 `tests/test_acoustic3d.py::test_rigid_sphere_scattering_matches_mie`（5% 阈值、ka∈{0.5,1,2}、1280 面板），**不含 ka=0.25 这一唯一 FAIL 点**，故本算例的失败模式不会被 CI 捕获。
- **F4 LOW（过度陈述）**：report §5.1 正文「Convergence ratio ~3.7–3.9x」与同节表格自相矛盾（表内出现 2.6x、3.2x）；且该量是**误差比**，只有 2 个网格档，无法给出观测阶，不能称收敛阶。
- **F5 LOW（friction 叙述不可机核）**：§6.2「Submitted to cluster but queue was busy (15+ jobs ahead), fell back to local run」——`compute_gate_events` 中 mission 19973 仅有 1 条 `compute_submitted`（2026-11-14 07:23:14），无排队/超时/回退记录，叙述仅部分可证（本地 14 例 ≈3 min，低于本机 10 min 门限，回退本身合规）。
- **F6 INFO（正面核实）**：无 MOCK/STUB；`report.md` 数值表与 `results.json` 一致；`tensor_reports` 18218 的 markdown 与本地 `report.md` md5 相同、六段齐全、4 图内嵌。

## 8. 级别裁定

- **L2（真实现）**：成立。`src/tensorbem/acoustic3d.py:763 solve_rigid_scattering` / `:896 scattering_cross_section` / `:857 far_field_amplitude` / `:1152 rigid_sphere_scattering_cross_section` 均为真实装配+稠密求解+远场积分，非硬编码（审计方重跑复现）。能力本体为既有（`f62f33c6` 引入），本单为复用，任务书明许。
- **L3（集成）**：成立。`acoustic3d` 在包内被 `src/tensorbem/acoustic3d_pulse.py:58`、`src/tensorbem/benchmarks/acoustic3d_pulse_envelope.py:52`、`src/tensorbem/benchmarks/acoustic3d_pulse_ref.py:80` 引用，`tests/test_acoustic3d.py` 有球散射门（审计方实跑 `-k "sphere or mie or neumann"` → **3 passed in 74.91 s**）。
- **L4（基准验证）**：**拒绝**。实质要件（完整算例+误差分析+可复现）本已达成且被审计方 bit-identical 复现，但 F1（0 提交）与 F2（0 裁判行）使「可复现」「可裁定」两条在平台侧不可达 —— 按「级别=功能实质、报告好看不算数、虚标判低级」判 **L3**。
- **L5**：无关（单球、章节级解析锚点，无实验/印刷表实测）。

## 9. 未做的检查（unknown / not_run，不记为 pass）

- 5120 面板（sub=4）档：未跑（稠密 5120² 复数系统 ≈420 MB，O(n³) 超出本机短测门限）。
- 与**独立 BEM 实现**（Capytaine / bempp / 商用码）的交叉验证：未跑（生态内无第二套 Helmholtz 外域 BEM 可直接调）。
- `tests/test_acoustic3d.py` 全模块 17 例：只跑了 sphere/mie/neumann 3 例（74.91 s），其余 14 例 not_run。
- 报告内 4 张 png 的像素级内容核查：未做（只核了存在性与字节数）。
- 「15+ jobs ahead」排队深度：无法回溯，记为 not_verifiable。

## 10. Disposition

**request changes（返修一轮，之后可升 L4）**：
1. 提交 `benchmarks/verified/case_bem_helmholtz_sphere/` 到分支（含 `run_study.py`、`results.json`、`report.md`、4 图）；
2. 在 `tensor_benchmark_cases` 登记本 anchor 的裁判行（metric/tolerance/ref_source）；
3. 把 ka=0.25 / 320 面板的 FAIL 点写进 pytest（期望 `err<6.5%` 或显式 xfail），并在 `benchmarks/verified/README.md` 收录；
4. 修 §5.1 的「收敛比」措辞（改为「误差随网格加密下降 3.89×/3.95× 与内接面积亏损比一致，观测阶未估计」）。

## 11. 审计产物

| 路径 | 内容 |
|---|---|
| `docs/audit_m19973_case_bem_helmholtz_sphere.md` | 本报告 |
| `runs/audit_m19973/indep_check.py` / `.json` | ref 四路线独立复算（mpmath / 直接角积分 / 光学定理 / 瑞利极限）+ 求积敏感性 |
| `runs/audit_m19973/repro/{run_study.py,results.json,*.png}` + `repro.log` | 审计方重跑（14/14 bit-identical） |
| `runs/audit_m19973/audited_case/*` + `AUDITED_CASE_SHA256.txt` | 被审载荷原件副本 + sha256 清单 |
| `runs/audit_m19973/figs/fig1..fig3.png` | 本报告图（BEM vs Mie；误差 vs ka；ref 三路线 + 复现一致性） |
| `runs/audit_m19973/insert_verdict.py` | verdict 落表脚本 |