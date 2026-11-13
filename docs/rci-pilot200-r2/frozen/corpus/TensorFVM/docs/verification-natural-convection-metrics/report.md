# 自然对流方腔验证证据的算术自洽审计与收敛阶判读 (m6545)

**任务**: m6545 = selfstart 自找活（generalist 席），优先「验证欠账」类
**日期**: 2026-11-08
**性质**: 验证欠账复核（对已合并/已归档的证据做独立复算，轻量分析 + 一次复现算例）
**结论一句话**: 归档方腔梯子 8 份 JSON 的派生场 `psi_error_rel` **全部失效**（用未归一
`|psi|` 而非 `|psi|/α`，报 98.8–99.9%，真实 0.98–37.3%）；且归档梯子**只有 frozen defect
路径可复现**，当前默认 damped 路径在 `n=32, Ra=1e5` 直接发散（NaN/inf）。`Nu` 关键量仍
`<3%` PASS，`|psi|/α` 锚点判 WITHHELD（阶梯恰一阶、n≤96 未收敛），但 `Ra=1e6` 一阶外推
后与 de Vahl Davis 差 **1.05%**，把「14% 偏差」归因为离散而非归一化口径。

---

## 1 案例介绍（物理背景 / 参数表）

二维方腔自然对流（differentially heated square cavity），Boussinesq 近似，左热右冷、
上下绝热，无进出流。无量纲化取 `L = dT = gβ = 1`，`ν = sqrt(Pr/Ra)`，`α = ν/Pr`，
速度尺度 `α/L`（因此流函数锚点必须按 `|ψ|/α` 与文献比较）。

| 参数 | 值 |
|---|---|
| Ra | 1e4 / 1e5 / 1e6 |
| Pr | 0.71（空气） |
| 网格 n | 32 / 64 / 96（归档）；`Ra=1e4` 只有 32/64 |
| 离散 | 交错 SIMPLE + 一阶迎风 + 冻结 van Leer defect；松弛 0.5/0.2/0.7；`momentum_sweeps=5, p_cycles=3`；`tol=1e-7`；multigrid Poisson |
| 基准 | de Vahl Davis (1983)：Nu = 2.238 / 4.509 / 8.817；`|ψ|max = 5.071 / 9.111 / 16.32` |

被审计的证据：`docs/incompressible-baselines/cavity_n0{32,64,96}_ra{1e04,1e05,1e06}.json`
（8 份）。本单不改写任何历史工件，只做**复算 + 判定 + 机器门**。

## 2 怎么做的（方法 / 工具链）

1. `scripts/audit_natural_convection_metrics.py`：逐记录从**同一条记录的原始量**
   （`nusselt`、`nusselt_cold`、`psi_abs_max`、`meta.nu`、`meta.prandtl`）重算所有派生场
   （`nusselt_error_rel`、`energy_imbalance_rel`、`psi_over_alpha`、`psi_error_rel`），
   并**反算旧公式** `(|psi_abs_max| − ref)/ref`，从而把失效场**算术识别**出来，不靠注释。
2. 梯子判读：三网格非等距观测阶 `(h1^p−h2^p)/(h2^p−h3^p) = (f1−f2)/(f2−f3)`（p 二分求解）；
   非单调序列一律返回 `None`（不许拟合鬼数字）。单调序列用拟合阶做 Richardson 外推。
3. 复现性：按归档配方（frozen defect，`dc_relax=1.0`）重跑 `n=32, Ra=1e5`，与归档逐位对照。
   为此给 `scripts/run_fvm_incompressible_baselines.py` 增加 `dc_relax` 参数 /
   `--dc-relax` 开关（默认 0.05 不变），并新增 `scripts/run_cavity_refinement.py`
   （把梯子延伸一级的薄包装）。
4. 机器门：`tests/test_natural_convection_metrics_audit.py`（8 条）——失效场算术识别、
   其余派生场 `rel < 1e-9` 自洽、复现配方、非单调必须 WITHHELD、外推判据、
   提交台账与树不漂移、台账篡改的负向对照。
5. 工具链：纯 NumPy；复现算例 1 次（n=32，1630 次外迭代）；发散探针 2 次走集群
   （n315/n330，本机负载 ~124/128 只做轻量）。

## 3 计算结果

### 3.1 失效场识别（8/8，算术可判）

| 记录 | 存储 `psi_error_rel` | 真实 `||ψ|/α` 误差 | 倍数 |
|---|---|---|---|
| n=32 Ra=1e4 | 98.79 % | 1.86 % | 53× |
| n=32 Ra=1e5 | 99.54 % | 22.57 % | 4.4× |
| n=32 Ra=1e6 | 99.84 % | 37.30 % | 2.7× |
| n=64 Ra=1e4 | 98.80 % | 0.98 % | 101× |
| n=64 Ra=1e5 | 99.57 % | 13.41 % | 7.4× |
| n=64 Ra=1e6 | 99.86 % | 20.10 % | 5.0× |
| n=96 Ra=1e5 | 99.59 % | 10.63 % | 9.4× |
| n=96 Ra=1e6 | 99.87 % | 14.12 % | 7.1× |

判据：`|stored − (|psi_abs_max|−ref)/ref| = 0`（8/8 精确成立）→ 旧公式用的是**量纲 ψ**。
当前 `src/tensorfvm/incompressible_baselines.py:441` 已改为 `|ψ|/(ν/Pr)`，故这是
**归档工件滞后于代码**，非现行求解器缺陷。其余派生场最大相对不自洽 = **0.0**。

### 3.2 复现性：只有 frozen 路径复现（机器精度）

| 量 | 本单复现（`dc_relax=1.0`，multigrid） | 归档 `cavity_n032_ra1e05.json` | 相对差 |
|---|---|---|---|
| Nu | 4.590121081451628 | 4.590121081451628 | **0.0** |
| `|ψ|/α` | 11.166992727842933 | 11.1670 | **0.0** |
| 外迭代 | 1630 | 1630 | 0 |
| 阶段名 | `first_order, deferred_1` | `first_order, deferred_1` | — |

反证（当前默认 `dc_relax=0.05` damped 路径，同一算例）：multigrid → 连续性残差
`NaN`（21370 次迭代，集群作业 `c1794114373_387`/n330）；sparse → `|u|max = inf` →
`_pseudo_inertia` 除零 `ZeroDivisionError`（作业 `c1794114676_394`/n315）。
**结论**：归档梯子不可用今天的默认配方复现，配方（frozen defect）属于证据的一部分，
已写进脚本 docstring 与 `--dc-relax` 开关。

### 3.3 收敛阶与判定（逐项显式回答「是否 <3%」）

| Ra | 网格 | Nu | Nu 误差 | 单调 | Nu 观测阶 | `|ψ|/α` | `|ψ|` 误差 | ψ 观测阶 | Richardson `|ψ|/α` |
|---|---|---|---|---|---|---|---|---|---|
| 1e4 | 32, 64 | 2.2036 → 2.2062 | **1.64 %** | 否 | 不可算（2 级） | 5.1654 → 5.1205 | **0.98 %** | 不可算 | — |
| 1e5 | 32, 64, 96 | 4.5901 → 4.4363 → 4.4418 | **1.71 %** | **否**（+0.0054 回升） | **WITHHELD** | 11.167 → 10.333 → 10.079 | 10.63 % | 1.161 | 9.658（+6.00 %） |
| 1e6 | 32, 64, 96 | 10.6417 → 8.9286 → 8.7352 | **0.74 %** | 是 | 2.813（32 级远离渐近区，仅作参考） | 22.407 → 19.600 → 18.625 | 14.12 % | 0.928 | 16.491（**+1.05 %**） |

回答：**Nu 三个 Ra 全部 <3%（PASS）**；`|ψ|/α` 有限网格误差 0.98 %（Ra=1e4）/ 10.6 % /
14.1 %（**>3%，不达标**），但其阶梯**单调且一阶**（p≈0.93–1.16），`Ra=1e6` 一阶外推后
**1.05 % < 3 %** ⇒ 该锚点的差距是**一阶离散误差**，不是归一化口径错。
故 `|ψ|/α` 在 `Ra=1e5/1e6` 的有限网格值判 **WITHHELD（未收敛）**，不外推不入台账。

## 4 结果图

图 1（`fig_metric_ladder.png`）：左 = `|ψ|/α` 随网格的阶梯（点线 = de Vahl Davis，星号 =
Richardson 外推）；右 = 关键量相对误差随网格下降（红实线 = 3 % 验收线，圆圈 = `|ψ|/α`，
虚线方块 = Nu）。可见 Nu 已在 3 % 内近乎平直，`|ψ|/α` 仍在单调下滑 —— 正是判 WITHHELD 的
图形证据。

![metric ladder](fig_metric_ladder.png)

## 5 注意点（踩过的坑 / 局限）

- **派生场会随代码漂移**：归档 JSON 里的误差场是「当时的代码 + 当时的口径」，复算是最廉价
  的防线；本单把它做成 `--check` 棘轮（失效集合按文件名钉死，重新生成必须显式改测试）。
- **默认配方≠归档配方**：damped defect（`dc_relax=0.05`）在方腔 `n=32, Ra=1e5` 发散，
  multigrid/sparse 两种 Poisson 后端一致复现。发散根因未深挖（本单只确证表型：NaN →
  `|u|max=inf` → `_pseudo_inertia` 除零），列为后续靶子。
- **只有 2–3 级网格**：`Ra=1e4` 仅 32/64 两级，观测阶不可算（本单不拟合）；`Ra=1e6`
  的 2.813 阶是 32 级远离渐近区的表象，**不能当方法阶**引用。
- **非等距三网格观测阶**：公式对非等距 h 成立，但敏感性高（`96` 与 `64` 只差 1.5 倍），
  故外层一律给 Richardson 值 + 判 WITHHELD，不给「PASS」。
- 本单**未**延伸网格（`n=128`）：需要先修 damped 路径或显式用 frozen 配方跑 ~30 min，
  列为下一单（脚本 `scripts/run_cavity_refinement.py` 已就绪）。

## 6 误差判定（逐项，显式）

| 关键量 | 基准 | 误差 | <3%? | 判定 |
|---|---|---|---|---|
| Nu（Ra=1e4, n=64） | 2.243 (contract) / 2.238 (DVD) | 1.64 % | 是 | PASS |
| Nu（Ra=1e5, n=96） | 4.519 / 4.509 | 1.71 % | 是 | PASS |
| Nu（Ra=1e6, n=96） | 8.800 / 8.817 | 0.74 % | 是 | PASS |
| `|ψ|/α`（Ra=1e4, n=64） | 5.071 | 0.98 % | 是 | PASS |
| `|ψ|/α`（Ra=1e5, n=96） | 9.111 | 10.63 %；外推 6.00 % | 否 | **WITHHELD（未收敛）** |
| `|ψ|/α`（Ra=1e6, n=96） | 16.32 | 14.12 %；外推 **1.05 %** | 有限网格否 / 外推是 | **WITHHELD（一阶离散主导）** |
| 复现性（Nu, `|ψ|/α`） | 归档 n=32 Ra=1e5 | **0.0**（机器精度） | 是 | PASS（配方=frozen） |
| 派生场自洽（除失效场） | 同记录原始量 | **0.0** | 是 | PASS |
| 归档 `psi_error_rel` | 真实 `|ψ|/α` 口径 | 2.7–101× | 否 | **FAIL（失效场，已隔离）** |

honest 声明：本单**没有**新增网格层级，`n=96` 上的 `|ψ|/α` 仍**不达 3 %**；
`Ra=1e5` 的 Nu 阶数**判不出来**（非单调），不以外推充数。

## 7 工件与复现

```
scripts/audit_natural_convection_metrics.py        复算 + 阶/外推 + --check 棘轮
scripts/run_cavity_refinement.py                   梯子延伸一级（frozen 配方）
scripts/run_fvm_incompressible_baselines.py        + dc_relax 参数 / --dc-relax
tests/test_natural_convection_metrics_audit.py     8 条机器门（含负向对照）
docs/verification-natural-convection-metrics/audit.json       台账（程序生成）
docs/verification-natural-convection-metrics/reproduced_cavity_n032_ra1e05.json
docs/verification-natural-convection-metrics/fig_metric_ladder.png
```

复现：`python3 scripts/audit_natural_convection_metrics.py --check`（当前 rc=0）；
`python3 -m pytest tests/test_natural_convection_metrics_audit.py -q`（8 passed）。