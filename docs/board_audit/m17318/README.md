# 交差审计 m17318 — 对象 m17262 / CASE-BEM-SPHERE-ADDED-MASS（TensorBEM）

审计性质：**交差核验**（非重做）。基准裁判 = `tensor_benchmark_cases.anchor_slug='CASE-BEM-SPHERE-ADDED-MASS'`。
结论：**level = 3**，`tensor_maturity_verdicts` id=529（verdict_by=`board-audit`，gate_state 待守护裁决）。

## 1. 判级（功能实质，不看报告厚度）

| 级 | 判据 | 实测 | 结论 |
|---|---|---|---|
| L2 独立功能 | impl + entry + run_artifacts 真实 | 真 BEM 外部 Dirichlet 求解 + CLI 入口 + 4 级 JSON（残差 ~1e-15） | **实质满足，但未交付**（见 F1） |
| L3 系统集成 | 生态仓内真调用方 | `three_dimensional_study.py:27/194`、`milestone_claims.py:46/381`、`ice_ship_verification.py:122` | 满足（均为既有调用方） |
| L4 基准验证 | 完整算例 + 误差分析 + **可复现** | 数值 PASS（finest 0.1130%），审计独立复算通过；但 `--ladder` 与全部工件只存在于工作树 | **拒绝**：分支级不可复现 |

数值裁判逐项（DB 原文 vs 工件，零手抄）：

| panels | ref `0.5ρV` [kg] | got [kg] | rel_err（工件实测） | DB 引文 | 容差 3% |
|---|---|---|---|---|---|
| 80 | 2146.754980 | 2007.732789 | 6.475923e-02 | 6.475923e-02 | ✗ |
| 320 | 2146.754980 | 2109.442900 | 1.738069e-02 | 1.738069e-02 | ✓ |
| 1280 | 2146.754980 | 2137.164559 | 4.467404e-03 | 4.467404e-03 | ✓ |
| 5120 | 2146.754980 | 2144.329111 | 1.130017e-03 | 1.130017e-03 | ✓ |

max |Δ(工件, DB 文本)| = **4.10e-09**（DB 文本只留 7 位有效数字，非数值分歧）。

## 2. 审计做了什么（可复跑）

```
python3 docs/board_audit/m17318/insert_verdict.py --dry-run      # 证据自检，打印数值对照
bash    docs/board_audit/m17318/repro/repro_audit.sh            # 审计侧一键复现（PASS，dev=0.0e+00）
python3 docs/board_audit/m17318/make_figures.py                 # figs/fig_convergence.png, fig_parity.png
```

* 在对象工作树内跑对象自带测试：`PYTHONPATH=src pytest tests/test_sphere_added_mass_benchmark.py -q`
  → **3 passed in 42.36s**（对象自报 3 passed/32.7s，一致）。断言是**不变量**（320p<3%、ladder 单调、
  `first_passing_level==320`、解析模块交叉差<1e-9、半径 2× 缩放=8.0 且 rel_err 不变），非"跑通即绿"。
* 审计独立复算 sub1(80p)/sub2(320p) → rel_err 与 DB 引文逐位一致（`artifacts/auditor_rerun_sub{1,2}.json`）。
* 对象工件 sub3/sub4 已快照入 `references/`（含 sha256），并在 `repro_audit.sh` 里做快照-原件一致性校验。

## 3. 发现（按严重度）

**F1（blocking for L4）交付 diff 为空。** 对象收尾时 `git status` = ` M src/tensorbem/benchmarks/sphere_added_mass.py`
+ `?? tests/test_sphere_added_mass_benchmark.py`，`result.evidence.branch=null`，且工件全在 `runs/` 下
（`.gitignore:15`）。即 **m17176 的病灶（runs/ 未入库）没有被修掉，只是换了个轮次重演**。后果：分支检出无法复跑
ladder 基准，证据随工作树回收即失效。

**F2（major）`runs/m17262/repro_cmd.sh` 不能复现它所承诺的东西。** 它调用 `make_artifacts.py`，而该脚本声明产出的
8 个文件里 `refinement_benchmark.json`/`entry.json`/`sensitivity_sub2.json`/`flux_panels_sub2.csv`/2 张 png **全部缺失**；
脚本本身 mode 644（非可执行）。"一键复现"属超额声明。

**F3（minor）5120 面板运行告警**：`src/tensorbem/conditioning.py:69`（`_power_iteration_largest` 的 `A.T @ A`）
报 `RuntimeWarning: invalid value encountered in matmul`（`references/err_sub4.log`）。该路径只用于条件数诊断，
答案未受污染（残差 3.31e-14，rel_err 与 DB 一致），但病态输入未做非有限值守卫，该条件数估计在 sub4 不可信。

**F4（minor）状态反转**：对象行 `succeeded/accepted=True` 同时 `fail_stage='no_progress+forced_finish'`。
机制侧不得把 `accepted=True` 当证据。

**F5（minor）未发论文级报告**：`report_id IS NULL`，六段产物（参数表、图、误差判定）从未生成。

## 4. 机制观察（可复用）

`maturity_gate.judge()` 对**交付性**盲：它只核 evidence 指针**存在且非占位**（根 = 生态仓 + 审计工作树）。
实测 `judge(4, ev, 'TensorBEM', 17318) = (4, [])`、`judge(5, …) = (5, [])` —— 同一份证据，声明级填 4 或 5 也会被机器确认，
即使对象一个 commit 都没产出。故本单**主动下调到 L3** 并在 evidence.detail.delivery_defect 写明差异；
建议守护加一条真判据：`evidence.impl_files` 至少一个指针**位于生态仓 git 跟踪内**（`git ls-files --error-unmatch`）。

## 5. 未跑/局限

* 集群作业 **cj1491**（5120 面板独立复算，`stage_dir`=对象工作树）递交时仍排队，结果未采用；
  sub4 数值结论依赖对象工件与 DB 一致（4.1e-9 级）。
* `make_artifacts.py` 端到端未跑（对象自身也未跑完）；1280 面板未独立复算（工件与 DB 差 4e-10）。
* 独立性限制：审计者与对象处于同一机器、同一 repo 检出体系，仅做了入口级独立复算，未做算法级重写。

## 6. 处置

**request changes（返修，非"消账"）**：commit impl+test 到真分支；工件移出 `runs/`（如
`docs/cases/CASE-BEM-SPHERE-ADDED-MASS/artifacts/`）；`chmod +x repro_cmd.sh` 并补齐其声明的全部产出；
`conditioning._power_iteration_largest` 加非有限守卫；补发六段报告。

## 7. 文件索引

```
insert_verdict.py            判定 + 落表（读 DB 基准行，数值零手抄）
make_figures.py / figs/      fig_convergence.png（收敛+3% 线）、fig_parity.png（ref vs got）
repro/repro_audit.sh         审计侧一键复现（PASS, dev=0.0e+00）
references/                  对象交付快照：impl、diff、测试、make_artifacts、repro_cmd、sub3/sub4 工件、
                             err_sub4.log、REFERENCE.md（Lamb/Newman 出处 + 机器交叉校验）
artifacts/                   审计独立复算工件 sub1/sub2
```
## 8. 已发布报告

`tensor_reports` id=**16321**（六段齐全 + 2 图：`fig_convergence.png`、`fig_parity.png`），由
`post_report.py` 程序化注入数值与图片（零手抄）；verdict 行 529 的 `evidence.detail.report_id` 已回指。

## 9. 审计者自证（同一颗雷）

本仓 `.gitignore` 同时有 `artifacts/`（第 10 行）与 `runs/`（第 15 行）：审计侧复算工件
`artifacts/auditor_rerun_sub{1,2}.json` 首次提交时**也被静默忽略**，用 `git add -f` 才入库。
说明 F1 不是个别疏忽而是仓库级陷阱 —— 凡以 `runs/`/`artifacts/` 命名的证据目录在本仓默认不可交付。
建议：新证据目录改用 `docs/cases/<slug>/` 或对关键工件统一 `git add -f` 并在复现脚本里校验
`git ls-files` 非空。
