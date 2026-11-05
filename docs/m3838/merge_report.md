# PR #134 治愈记录：`agent/m3236` ← `main` 三方合并（m3838，2026-11-05）

## 1. 背景与几何

PR #134 的分支 `agent/m3236` 交付物是「把 Dean (1945) 潜没圆柱绕射表 7 的 11 个数字钉入
`tensorbem.benchmarks.declared_numbers` 注册表并加回归门」。分支最后一次 merge `main` 之后
`main` 又前进了 40 个提交（`hulme_hemisphere` / `hulme_chief` / `submerged_sphere` 三组注册项、
`run_study` 观测阶口径订正 m3150 等），于是 PR 变为 `mergeable=false`。

| 项 | SHA / 值 |
|---|---|
| 远端 `agent/m3236`（合并前，旧 SHA） | `1615c100895de9d1e487b5886fd68adea5b0f770` |
| `origin/main`（合并基线） | `ecba798833d6c2429fbc897d2ce515efb70dd682` |
| 三方合并 base | `4f5b73e73c4e731582f2b050328ec5fcf62b4a86` |
| 分支侧领先提交 | 3（`a2a1ed9` 交付物、`f67bbd0` merge main、`1615c10` CI 重跑） |
| 合并方式 | `git merge origin/main --no-ff`，冲突逐文件语义并集解冲 |

物理/几何参数（Dean 基准本身，用于本记录的数字复核）：二维深水自由面，潜没圆柱半径
$a=1\,\mathrm{m}$、潜深 $f=2\,\mathrm{m}$、无量纲波数 $ka=0.6$、$\rho=1025\,\mathrm{kg/m^3}$、
$g=9.80665\,\mathrm{m/s^2}$；面元数 $N=32/64/128$（注册表钉桩用 $N=64$）。

## 2. 方法

1. `git fetch` 取回远端真实 PR 头（本地 `agent/m3236` 停在旧提交 `4b1b21f`，与远端已不共祖，
   已用 `git switch -C agent/m3236 origin/agent/m3236` 对齐；`main` 本地引用是 clone 快照
   `f0bfb41`，全部比较改用 `origin/main`）。
2. 用 `git merge --name-only` 求两侧改动交集，得到**冲突候选 = 3 个文件**，再 `git merge --no-ff`
   触发真实冲突。
3. 逐文件解冲，规则：**语义并集**（两侧新增全部保留，任何一侧都不许静默丢）。禁止机械 overlay。
4. 解冲后做三类自证：
   - 语法：`ast.parse` 通过；
   - 无丢失：注册表 `key=` 集合 与 两侧父提交比较，`ours-theirs` 双向差集为空、无重复 key；
   - 纯增量：`git diff <父提交> -- <文件>` 除文件头外不含 `-` 行。
5. 测试：合并树与 `origin/main` 树跑同一精选命令，比较通过数。

## 3. 冲突文件清单与逐文件解法

| # | 文件 | 冲突类型 | 解法 |
|---|---|---|---|
| 1 | `src/tensorbem/benchmarks/declared_numbers.py` | **内容冲突（2 个 hunk，UU）** | 语义并集：hunk1 = 分支侧 11 条 `dean` 注册项 + main 侧 62 条 `hulme_hemisphere`/`hulme_chief`/`submerged_sphere` 注册项（顺序无关，`DECLARED_NUMBERS` 是元组）；hunk2 = `_run_group` 的 `dean` 分支 + main 的 `hulme_hemisphere`/`submerged_sphere`/`hulme_chief` 分支（group 互斥，串联即并集）。|
| 2 | `docs/marine_bem_benchmarks.md` | 文本自动合并 | 复核双向 diff：main 侧的 4.10/4.11 表 10/12 数字订正（m3150）与分支侧的 4.8「复算与回归钉桩（m3236）」段落**同时存在**，无覆盖。|
| 3 | `tests/test_declared_numbers.py` | 文本自动合并 | 复核双向 diff：main 侧的 `_run_group` 导入、符号型声明值断言放宽（`declared > 0` → `declared != 0`，为表 11 的负 $B_{33}'$ 让路）与分支侧的 `dean` group 断言 + `test_fast_tier_reproduces_the_documented_dean_numbers` 同时存在。|

### 踩坑（本轮唯一实质陷阱）

git 把分支侧 Dean 注册项最后一条的收尾行 `    ),` 判成了**公共后缀**留在 `>>>>>>>` 之后，
于是「两侧块直接串联」少一个右括号。首次并集后 `ast.parse` 报
`DECLARED_NUMBERS: tuple[...] = (` never closed——这条自证把静默语法损坏挡在了提交之前。
修法：在分支侧块尾补回该条注册项的 `    ),`，再次 AST 校验通过。

### 合并语义账本（`docs/m3838/merge_accounting.json`，程序化生成）

| 量 | 值 |
|---|---|
| merge base 注册项数 | 22 |
| `agent/m3236` 侧注册项数 | 33（+11） |
| `origin/main` 侧注册项数 | 84（+62） |
| 合并后注册项数 | **95 = 22 + 11 + 62** |
| 从分支侧丢失的 key | 0 |
| 从 main 侧丢失的 key | 0 |
| 重复 key | 0 |

## 4. 结果与测试证据

- 合并树：`pytest tests/test_declared_numbers.py -k "fast_tier or well_formed or covers_every"`
  → **4 passed, 14 deselected, 1 warning in 66.25 s**（日志 `docs/m3838/core_tests.log`）。
- `origin/main` 基线（同命令）：**3 passed, 14 deselected, 1 warning in 30.77 s**
  （`docs/m3838/core_tests_main_baseline.log`；基线少 1 条正是分支新增的 Dean 门）。
  → 合并后通过数 ≥ 基线，**未低于基线**。
- 注册表 fast 层端到端复算：**19/19 PASS**，`drifted=0`；
  11 条 Dean 行全 PASS（其中零反射/能量平衡/圆偏振三条按上限守护）。
- 全量套件（1002 项）本机未跑完：宿主 load average 峰值 ~2600（同机大量并行求解器测试），
  两次 `-n 6` 全量运行跑到 21%/28% 时已记录 0 failure，随后为腾出 CPU 终止。
  全量对等性**未完成**，如实记录；CI 会在 push 后重跑（PR 触发）。

### 图表（`docs/m3838/figures/`，由 `docs/m3838/make_figures.py` 程序化生成，零手抄）

1. `fig_registry_deviation.png`：fast 层 19 条注册量的 |实测/声明| 与各条漂移门限（红线），
   紫色为上限型守恒量。
2. `fig_dean_convergence.png`：左=Dean 圆柱 $ka=0.6$ 离散误差 vs 精确解随 $N$ 收敛
   （透射误差 1.462e-2 → 8.394e-3 → 4.463e-3，观测阶约 0.85）；右=机器精度不变量
   （$|R_N|$、能量平衡残差、圆偏振误差）与 $10^{-12}$ 守护线。
3. `fig_merge_union.png`：注册表并集账本（22 共有 + 11 + 62 = 95），可视化「无丢失」。

## 5. 误差判定（vs 基准，逐项）

| 关键量 | 基准 | 实测 | 相对偏差 | <3%? |
|---|---|---|---|---|
| 最差离散量：MFS 水平力 $F_x$（$N=64$） | 1.510e-3 | 1.506e-3 | $-2.75\times10^{-3}$ | **是** |
| 面板透射误差（$N=64$） | 8.390e-3 | 8.394e-3 | $+4.36\times10^{-4}$ | **是** |
| Haskind-Newman vs 绕射力（$N=64$） | 2.160e-5 | 2.164e-5 | $+1.87\times10^{-3}$ | **是** |
| 椭圆柱反射/透射模（$N=64$） | 9.751e-2 / 9.952e-1 | 同 | $+3.97\times10^{-5}$ / $+1.67\times10^{-7}$ | **是** |
| 圆柱零反射 $|R|$（声明上式上限 1e-12） | 0（精确） | 1.32e-15 | 绝对量 1.3e-15 | **是** |
| 能量平衡 $||R|^2+|T|^2-1|$ | 0（精确） | 0.0 | 0 | **是** |
| 圆偏振误差 $|F_z+iF_x|$ 相对 | 0（精确） | 6.34e-15 | 6.3e-15 | **是** |
| 注册表覆盖（fast 层） | 19 条 | 19 条 | 0 | **是** |

结论：本单所有可量化关键量相对偏差 $\le 2.75\times10^{-3}$，**全部 < 3%**。
合并操作本身不引入数值变化（合并树与两侧父提交在同一组量上的差异仅为「集合变大 11 条」）。
