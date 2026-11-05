# PR #128 heal — merge `agent/m3181` → `main` (mission m3772)

## 1. 案例背景与几何/参数表

| 项 | 值 |
|---|---|
| 仓库 | TensorBEM（边界元水动力求解器） |
| PR | #128，head = `agent/m3181`，dirty vs `main` |
| merge base | `4f5b73e7` (Merge #120) |
| mainline parent | `ecba7988` (origin/main, 40 commits ahead of base) |
| PR head 归档前 SHA | `55db3976` (origin/agent/m3181) |
| 治愈后 merge commit | `ea12f97`（本工作分支 agent/m3772, commit 后为 HEAD'） |
| 物理对象 | 三维自由液面 BEM 声明数字注册表：Hulme(1982) 漂浮半球表10/11、Wang/Linton 潜没球表12 的 147 条引用数字 |
| 关键参数 | 无量纲 ka=0.8/1.0，ρ=1025 kg·m⁻³，g=9.80665 m·s⁻²，面板加密 subdivisions 0–3，容差 DEFAULT_TOLERANCE=1.5e-2 |

## 2. 做法（方法/流程/工具链）

1. `git merge --no-commit --no-ff origin/agent/m3181`（三方合并，禁用任何 overlay/copy）。
2. 两侧自 base 起都改过的文件仅 3 个：`docs/marine_bem_benchmarks.md`（git 自动合并，双方新增 4.11/4.12 节并存）、`src/tensorbem/benchmarks/declared_numbers.py`（3 处冲突）、`tests/test_declared_numbers.py`（2 处冲突）。逐文件语义并集解冲。
3. 写机器验证器 `docs/tools/verify_pr128_union_m3772.py`，四层反丢内容检查：文件层、行层（每个父提交新增的行必须逐字存活，只有登记在 `EXPECTED_REWRITES` 的白名单改写例外）、符号层（父提交的顶层 def/class/赋值名不得缺失，且合并后不得出现重名遮蔽）、注册表层（`merged == main ∪ branch` 键集精确相等）。
4. 冲突记录机读化：`docs/m3772/outputs/conflicts.json`；验证证据：`docs/m3772/outputs/verify_union.json`。

## 3. 计算结果（程序化注入）

| 量 | 数值 |
|---|---|
| 两侧都改动的文件数 | 3 |
| 文本冲突文件数 / 冲突块数 | 2 / 5 |
| 文件面：main / branch / 并集 / 合并后 | 449 / 386 / **463** / **463** |
| 注册表条目：main / branch | 84 / 85 |
| 双方完全相同的行（同 key 同值同容差同参数） | 22 |
| 仅 main / 仅 branch | 62 / 63 |
| 合并后唯一 key | **147** = 62+63+22 |
| 键集差：父有子无 / 子有父无 | **0 / 0** |
| 行层丢失（未登记换写） | **0** |
| 符号层缺失 / 重名遮蔽 | **0 / 0**（修复前 1，见坑） |

### 逐文件解法

* **declared_numbers.py**
  - 导入块：保留双方别名集（同一模块两次导入，`hemisphere_benchmark`/`hulme_benchmark`、`sphere_benchmark`/`wang_linton_benchmark`），两侧调用点均不改写；
  - `DECLARED_NUMBERS` 字面量尾部：保留 main 的 61 条新行，并把 branch 的 `) + _free_surface_3d_rows()` 追加式并集保留 —— 任一侧 overlay 都会静默丢掉一张表的全部引数；
  - 组分发：保留两个 `if group == ...` 块集合（共 6 个分支，组名互斥）；
  - 未冲突但 **git 静默放过** 的同名冲突：branch 的无参构造器 `_chief_rows()` 与 main 的摊平器 `_chief_rows(rows)` 同名，Python 中后者遮蔽前者（模块可导入、只是注册表变少）→ 将 branch 侧改名为 `_hulme3d_chief_rows()` 并同步唯一调用点，main 代码零改动。
* **test_declared_numbers.py**
  - `assert entry.declared != 0.0` 上方注释：双方各自重写同一断言说明（main 讲 CHIEF 消除的负 B33′，branch 讲未正则化 3-D BIE 的负阻尼）→ 合并为一段，保留两个物理陈述；
  - 必需组集合：取并集 `{hulme3d, hulme3d_chief, hulme_chief, hulme_hemisphere, ...}` —— 合并后的注册表确实含全部 11 个组，只断言一侧会留下另一族未受保护；
  - 测试函数集合 20 = |main(17) ∪ branch(18)|。

## 4. 结果图

本任务为**纯仓库/后端治愈（merge 语义并集 + 注册表键集守恒）**，无场量云图、无解析解/文献对比曲线可画；按报告纪律在此声明无图的理由：被验证的量是**离散的键集与文件集合**（147 条 key、463 个文件），属于集合守恒命题而非连续场，图形化只会是条形图重复上表数字。等价的可复现证据是行级/符号级/注册表级三层机器检查（`verify_union.json`，退出码 0）。

## 5. 注意点（踩坑）

1. **机械 overlay 的隐性证据**：三处冲突都能"取一侧"编译通过；`declared_numbers.py` 尾部那处若取 main 侧，registry 从 147 掉到 84 且 pytest 仍绿（只少数字）。
2. **git 不报的语义冲突**：同名不同义的 `_chief_rows` 被自动合并，这是唯一一处需要重命名才能并集的情形，只有符号层的重名扫描能发现。
3. **环境陷阱**：全局存在指向另一 worktree 的 `tensorbem` editable 安装（`/root/tp-agent/work/m3742/src`），不加 `PYTHONPATH=$PWD/src` 会**静默测另一棵树的代码**；本次所有测试均显式设置 PYTHONPATH。
4. **工具陷阱**：`run_shell` 120 s 收割进程组，长测试需 `setsid`；两次重复启动会写同一日志互相污染（本次已发生并重跑）。

## 6. 误差判定（honest）

| 判据 | 基准 | 结果 | 是否达标 |
|---|---|---|---|
| 注册表键集守恒 | main ∪ branch = 147 | 合并后 147，缺 0 多 0 | **是（0 偏差 <3%）** |
| 文件集合守恒 | 并集 463 | 合并 463 | **是（0 偏差）** |
| 行级不得静默丢失 | 双方新增行（main 661+9、branch 263+109） | 除 6 行登记改写外全部逐字存活 | **是** |
| 测试不低于 main 基线 | main 全量 pytest | **未完成**：两侧已并行跑完全量套件的 ~93%，进度流中未出现 F/E，最重的 1280 面板用例仍在跑时因预算被终止；定向子集复跑亦被 120 s 工具时限中断 | **证据不足（honest-FAIL）** |

诚实结论：**语义并集部分已机器证明通过**（三层零丢失、键集精确等于并集）；**测试基线对比未取到完整计数**，仅能报告 93% 进度内无失败迹象。建议 merge_governor 复验时以 CI（.github/workflows/tests.yml，先产物后 pytest）为准。

## 7. 推送说明

平台纪律禁止智能体自行 push。治愈结果位于本工作树分支 `agent/m3772`（merge commit `ea12f97` + 本报告 commit），其父提交正是 PR #128 的 head `55db3976`，因此把该 commit 推到 `agent/m3181` 为 fast-forward：old SHA = `55db3976`，new SHA = 该分支 HEAD。已存在的其他本地分支（m3744/m3751/m3760/m3762 等历次治愈尝试）未被导入、未被删除。
