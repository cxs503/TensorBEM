# 开发报告 · m16042 / m17837：SK 阻尼有理拟合的极点重定位（PR#692 重开第 2 轮）

* 仓库：TensorBEM · 分支 `agent/m17837` · 模块 `tensorbem.sk_fit`
* 工件：`docs/sk_fit_study/results.json`、`results.md`、`sk_fit_verification.png`
  （全部由 `python -m tensorbem.benchmarks.sk_fit_verification --out-root docs/sk_fit_study --write-doc docs/sk_fit_sanathanan_koerner.md` 生成）
* 方法文档：`docs/sk_fit_sanathanan_koerner.md`（§2 步骤、§4 误差判定、§5 坑）
* 证据：`tests/test_sk_fit.py`（21 个节点全绿，48.7 s 本地）

## 1 缘起与现状盘点

TensorBEM#692 于 2026-11-05 被裁定关闭；本单要求“管到底第 2 轮”，把该 PR 的功能要点
（SK 迭代 + 验证研究 + 报告）在**当前 main** 上重新交付并合并。

先做盘点（`git log` / `ls` / `git show --stat`），结论是**功能主体已随 m16021 的合并
（commit 7f8a746，`Merge #691 (m16021) via merge-master`）进入 main**：

| PR#692 声明文件 | 当前 main 状态 | 说明 |
|---|---|---|
| `src/tensorbem/sk_fit.py` (+295/-80) | 已在主干（392 行，含 `_sk_iterate` / `_real_pole_projection` / `fixed_grid_damping_fit`） | 第 1 轮成果 |
| `src/tensorbem/benchmarks/sk_fit_verification.py` (+330/-0) | 已在主干（236 行） | 第 1 轮成果 |
| `tests/test_sk_fit.py` (+165/-0) | 已在主干（160 行 / 10 节点） | 第 1 轮成果 |
| `docs/sk_verification/results.json`、`results.md`、`fig_sk_verification.png` | **路径改名为 `docs/sk_fit_study/`**（`results.json`、`results.md`、`sk_fit_verification.png`） | 合并时改名，内容同源 |
| `docs/dev_report_m16042_sk_iteration.md` (+133/-0) | **缺失** | 本报告补上 |

因此本单**不做重复实现**，而是：(a) 把缺失的开发报告补上；(b) 在第 1 轮留下的
“实测曲线 17–49 % 是模型类下限”这条结论上继续做实验——它是本模块唯一没被证伪过的
精度断言，也是最能决定模块可用性的地方。

## 2 本轮发现：瓶颈是极点落点，不是基的表达能力

第 1 轮的链路是：SK 迭代解出 `A(u)/D(u)` → 把 `D` 的根投影成实衰减率
（实负根精确映射，共轭复根投影到模 `|u|^½`）→ 按实测误差在「投影极点, 固定网格」
里取优。第 1 轮在实测 Hulme 半球曲线 12 阶得到 `1.684e-01`，并把原因归给
“实极点基装不下代数尾”。

但同阶无约束有理模型只要 `1.25e-06`——两个数差 5 个量级，说明**残差主要来自投影步骤
的落点，而不是基的容量**。投影把一对共轭复根压成两个**重合的实极点**（12 阶时 5 对
复根 → 10 个重合极点），SK 的根在复平面上是对的，投影后的横坐标是错的。

## 3 怎么做的（改动清单）

1. `src/tensorbem/sk_fit.py`
   * 新增 `_refine_poles`：对 `log(α_k)` 做有界 Nelder-Mead（`scipy.optimize`，
     无随机状态），目标函数 = `state_space._solve_linear` 已在对残基最小化的**同一个**
     加权残差平方和，因此残基永远线性最小二乘解、`d ≥ 0` 的物理约束不被绕过，
     只有 `n_poles` 个变量在动；两个种子（SK 投影、改前固定网格）各带 1 次热重启；
     搜索区间 `[0.05 ω_min, 50 ω_max]` 外一律 clip。
   * 新增 `_probe_local_optimality`：在返回极点上做单极点 ±5 % 对数探针，
     `refine_probe_gain ≤ 1e-3` 才算局部最优——**不转发 `scipy` 的 `success` 标志**
     （该类平坦目标上它永不满足 simplex-size 判据，转发等于造假）。
   * 候选集从 2 个扩到 4 个（SK 投影 / 固定网格 / 两套重定位极点），仍按**实测**
     `max|ΔB|/max|B|` 取优 → “返回值永不劣于任一候选”是结构性不变量。
   * 预算参数 `refine_maxfev`（默认 1500，`0` 关闭并逐位复现第 1 轮）；
     `SKFitReport` 新增 `refined_pole_rel_error`、`refine_start`、`refine_evaluations`、
     `refine_probe_gain`、`refine_converged`。
2. `src/tensorbem/benchmarks/sk_fit_verification.py`：扫描改成三个估计器
   （改前固定网格 / 第 1 轮 SK / 本轮 SK+重定位）+ 3 % 精度门列 + 增益列 +
   `never_worse_than_unrefined`、`orders_meeting_accuracy_bar` 机检字段；图加一条
   第 1 轮曲线和一条 3 % 门横线。
3. `tests/test_sk_fit.py`：21 节点。新增/改写的 pin：重定位在实测曲线 6/8/10/12 阶
   全部进 3 % 门且比第 1 轮至少 10 倍；`refine_maxfev=0` 逐位复现第 1 轮；
   重定位确定性（两次调用逐位相同）；重定位不打乱可表征曲线的机器零复原；
   收敛是探针测出来的（`fit.converged == report.refine_converged == (probe ≤ 1e-3)`）；
   “返回值 = 候选集最优”（含重定位候选）。
4. `src/tensorbem/capability_evidence.py`：`damping-rational-fit` 行 claim 更新为
   本轮测得的事实（并显式写入“第 1 轮的模型类下限结论已撤回”）。
5. `docs/sk_fit_sanathanan_koerner.md`：§2 增加第 3 步（重定位）、§4 误差判定重写
   （含撤回与两条残留局限）、§5 加 3 条坑（投影落点、单纯形 `success` 不可信、
   重定位不替代阶数）；表由 `--write-doc` 程序化写入。

## 4 计算结果（本表由 `docs/sk_fit_study/results.json` 程序化渲染，零手抄）

| case | order | returned | 第1轮 SK(不重定位) | 改前固定网格 | 增益 | 探针 | 3 % 门 |
|---|---|---|---|---|---|---|---|
| representable | 2 | 5.548e-02 | 7.839e-01 | 8.894e-01 | 14.1x | 0.00e+00 | no |
| representable | 3 | 1.225e-15 | 1.225e-15 | 8.072e-01 | 1.0x | 0.00e+00 | yes |
| representable | 4 | 4.084e-16 | 3.472e-15 | 2.395e-01 | 8.5x | 0.00e+00 | yes |
| representable | 6 | 6.127e-16 | 6.127e-16 | 1.534e-01 | 1.0x | 0.00e+00 | yes |
| representable | 8 | 4.084e-16 | 2.042e-15 | 1.447e-01 | 5.0x | 0.00e+00 | yes |
| representable | 10 | 6.127e-16 | 1.838e-15 | 8.509e-02 | 3.0x | 0.00e+00 | yes |
| representable | 12 | 6.127e-16 | 3.165e-15 | 4.434e-02 | 5.2x | 0.00e+00 | yes |
| hulme_hemisphere | 2 | 9.805e-01 | 9.976e-01 | 9.997e-01 | 1.0x | 0.00e+00 | no |
| hulme_hemisphere | 3 | 6.491e-01 | 9.902e-01 | 9.902e-01 | 1.5x | 0.00e+00 | no |
| hulme_hemisphere | 4 | 3.802e-01 | 9.785e-01 | 9.785e-01 | 2.6x | 0.00e+00 | no |
| hulme_hemisphere | 6 | 1.487e-02 | 9.023e-01 | 9.230e-01 | 60.7x | 1.51e-04 | yes |
| hulme_hemisphere | 8 | 2.418e-03 | 4.921e-01 | 4.921e-01 | 203.5x | 1.00e-04 | yes |
| hulme_hemisphere | 10 | 5.081e-04 | 2.241e-01 | 2.241e-01 | 441.1x | 4.56e-02 | yes |
| hulme_hemisphere | 12 | 2.046e-04 | 1.684e-01 | 1.684e-01 | 822.8x | 1.17e-02 | yes |

* 3 % 门通过的阶次：`{'representable': [3, 4, 6, 8, 10, 12], 'hulme_hemisphere': [6, 8, 10, 12]}`
* 两个不变量：`regression_free = True`（不劣于改前固定网格）、
  `never_worse_than_unrefined = True`（不劣于第 1 轮 SK）。
* 实测曲线（B，Hulme 1982）12 阶：第 1 轮 `1.684e-01` → 本轮 `2.046e-04`（**822.8×**），
  相对峰值误差 0.020 %；同阶无约束有理上限 `1.25e-06`，即剩余差距（163×）才是真正的
  模型类误差。
* 可表征曲线（A，三衰减率合成）3 阶：SK 与本轮都是 `1.225e-15`（机器零），
  改前固定网格 `8.072e-01`——重定位没有破坏解析可解情形。

## 5 验证

| 命令 | 结果 |
|---|---|
| `python -m pytest tests/test_sk_fit.py -q` | **21 passed**，48.7 s（含 study 重跑与工件/表格逐行比对） |
| `python -m tensorbem.benchmarks.sk_fit_verification --out-root docs/sk_fit_study --write-doc docs/sk_fit_sanathanan_koerner.md` | 工件与表重生成（本报告的数字来源） |
| `python -m pytest tests/test_state_space.py tests/test_freq_time_bridge.py -q` | 27 passed（下游链路未受影响） |
| `python -m pytest tests/test_capability_evidence.py -q` | 4 failed / 14 passed —— **与基线（HEAD）逐条相同**，属既有分支漂移（ice_cover / wigley 行与 driver 理由过期），非本单引入 |
| `python -m pytest tests/test_capability_coverage.py -q` | 3 failed —— 同上，基线同样红 |
| 集群全量 `python -m pytest -q`（TensorBEM，loong64） | 与基线对照，见 PR 说明与平台报告 |

## 6 注意点与残留局限

1. **4 阶及以下在实测曲线上仍不达标**（3 阶 65 %、4 阶 38 %）：4 参数装不下跨四个量级
   的曲线，重定位只能改善 1.5–2.6×。达标需要 ≥6 个实衰减率。
2. **10/12 阶未达探针局部最优**（`refine_probe_gain` 仍 4.6e-2 / 1.2e-2）：默认预算
   （1500 评估 × 2 起点 × 2 轮）不足以收敛到驻点，所以 `2.046e-04` 还不是该阶次的下限；
   报告的 `refine_converged` 如实为 `no`。
3. **探针只给必要条件**：单极点移动无改进 ≠ 多极点联合移动无改进，故只当“没说谎的下界”。
4. **成本**：单次 fit 0.4–3 s（阶数相关），批量扫描应上集群或把 `refine_maxfev` 调低；
   由于候选集选择是结构性的，调低预算只会降低精度而不会变差于第 1 轮。
5. **`scipy` 成为运行时依赖**（`pyproject.toml` 已声明 `scipy>=1.10`，非新增依赖）。

## 7 与 PR#692 的关系 / 交付判定

* 功能要点（SK 迭代、验证研究、报告）在 main 已具备，本单不重复实现、只补
  `docs/dev_report_m16042_sk_iteration.md` 并把模块推进到“实测曲线满足 3 % 门”。
* 工件目录沿用主干命名 `docs/sk_fit_study/`（PR#692 的 `docs/sk_verification/` 为合并前
  命名），避免同一研究出现两份可变工件。
* 完成判定按 mission 口径：新 PR 合并进 main，mission 保持 `awaiting_pr_merge` 至合并。
