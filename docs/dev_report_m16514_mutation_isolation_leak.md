# m16514 — 变异审计引擎的隔离漏洞：一次"全部存活"的假证据，与修复

**仓**：TensorBEM · **分支**：agent/m16514 · **任务**：自找活（generalist 席）· **日期**：2026-11-12
**工件**：`src/tensorbem/benchmarks/mutation_isolation.py`（新）、
`src/tensorbem/benchmarks/mutation_audit.py`（改）、
`tests/test_mutation_audit_isolation.py`（新）、
`docs/m16514/probe_isolation.py`、`docs/m16514/probe_isolation2.py`、
`docs/m16514/baseline.log`（修复前）、`docs/m16514/after_fix*.log`（修复后）

---

## 1. 案例介绍：一条"永远不会失败"的审计

`src/tensorbem/benchmarks/mutation_audit.py` 是本仓判别力审计的引擎（m2582 建，m3108 扩到波浪载荷链）。
它的工作方式：把 `src/tensorbem` 复制成影子树，向其中**单个 AST 节点**植入语义合法的缺陷（`+`↔`-`、
`*`↔`/`、比较符翻转、常量右移/取负），用 `PYTHONPATH=<shadow>/src` 跑映射的单元测试；测试失败=变异体被
杀死（护栏有效），仍然全绿=存活（盲点）。

这个设计有一个前提：**测试会话真的导入影子树**。前提一旦不成立，所有变异体都会"存活"——而"存活"
在报告里的读法是"测试套件有盲点"。也就是说，一个失效的审计会把**引擎自身的故障**伪造成**被测套件的缺陷**。

本次复测（2026-11-12，工作树 `work/m16514`，HEAD `f79b54c`）就是这种状态。

## 2. 怎么做的

1. **复测**：对 m3108 审计过的三个波浪载荷模块（`rao`、`pressure`、`helmholtz`）用**未改动的**引擎
   重跑同一命令（`--max-per-module 10`，`jobs=3`）。命令本身完全可复现。
2. **症状**：记录到的 25 条判定里 **24 条 survive、1 条 killed**。其中包含这样的变异体：
   - `tensorbem.rao:464:21` `m + added_mass` → `m - added_mass`（RAO 组装的运动方程主项！）
   - `tensorbem.pressure:302:43` `0.5` → `-0.5`（动压系数主项）
   - `tensorbem.rao:464:39` `1j * omega * damping` → `1j * omega / damping`
3. **对照**：同一台机器上 m3108 的存档 `docs/m3108/outputs/targeted_closure.json` 把
   `pressure 302:43 (0.5 → -0.5)` 记为 **after = killed**。同一条变异、同一份引擎，一次 killed 一次
   survived —— 二者必有其一是假的。
4. **定位**（`docs/m16514/probe_isolation.py`：照抄 `run_tests` 的 env 与命令行，植入
   `m + added_mass → m - added_mass` 后跑 `tests/test_rao_verification.py`）：
   - 裸解释器 `import tensorbem` → `/tmp/probe-*/src/tensorbem/__init__.py`（**影子树**，看着没问题）；
   - **pytest 会话内** `import tensorbem.rao` → `/root/tp-agent/work/m16514/src/tensorbem/rao.py`
     （**仓库原树**！）；
   - 带变异的 `tests/test_rao_verification.py`：`rc=0, 24 passed`。
5. **根因**（两条独立机制，都在 `sys.path` 上，都排在影子路径之前）：
   - `tests/conftest.py`（m14560/PR #632 引入）为压制 editable install，显式
     `sys.path.insert(0, <repo>/src)` —— 对变异审计而言这是**反的**；
   - `python -m pytest` 以 cwd 为首路径，任何更早的插入都会排在 `PYTHONPATH` 影子目录之前。
   原有的 `_check_isolation` 只做裸解释器 `import tensorbem`，而 **conftest.py 在裸解释器里根本不运行**，
   因此它为一个并不成立的隔离签了字。
6. **修复**（最小、可单测）：
   - 新模块 `tensorbem.benchmarks.mutation_isolation`：一个 pytest 插件（`-p` 加载，`pytest_configure`
     阶段早于任何 conftest），清掉已缓存的 `tensorbem*`、剔除仓库 `src`，并在 **`sys.meta_path` 首位**
     装一个 `_ShadowFinder`，把所有 `tensorbem` / `tensorbem.*` 的解析钉到影子树。`sys.meta_path` 先于
     `sys.path`，所以 conftest 之后的 `insert(0, repo/src)` 不再能翻盘。
   - 插件把**实际导入到的** `tensorbem.__file__` 写进 `$TENSORBEM_MUTATION_ISOLATION_FILE`；
     `run_tests` 事后读它，路径不在影子树内就判定 `ISOLATION-FAILURE`，`_classify` 直接给出
     `isolation_error`（既不算 killed 也不算 survived，`_score` 里单列）。
   - `_check_isolation` 改成**跑一次真实的 pytest 会话**（`--collect-only` 风格的探针测试），不再用裸解释器。
7. **为什么这是"关不掉的假绿"而不是"再补个测试"**：修复后同一批变异体开始按物理必然被杀死（见 §3）。
   被修的对象是**证据链本身**：引擎此前对小改动的沉默，不能作为套件盲点的证据。

## 3. 计算结果

### 3.1 修复前（原样复测，`docs/m16514/baseline.log`）

| 模块 | 抽样变异体 | 记录到的判定 |
|---|---:|---|
| `tensorbem.pressure` | 10 | 9 survived / 1 killed（`_as_points` 之外全部存活，含 `0.5 → -0.5`） |
| `tensorbem.helmholtz` | 10 | 9 survived / 1 killed（`131:32 1 → -1`） |
| `tensorbem.rao` | 10 | 见 log（`m + added_mass → m - added_mass` survived） |
| **合计（记录到的 25 条）** | — | **24 survived / 1 killed，score ≈ 0.04** |

一个把运动方程主项符号翻掉的变异体"存活"，在物理上是不可能的结论 —— 这就是隔离漏洞的指纹。

### 3.2 修复后（`docs/m16514/after_fix.log`、`after_fix_bounded.log`）

| 变异体 | 修复前 | 修复后 |
|---|---|---|
| `pressure:63:44` `_as_points` `3 → -3` | survived | **killed** |
| `pressure:129:16` `PressureLoads.coefficients` `0.5*rho*v**2*A → …` | survived | **killed** |
| `pressure:129:28` `velocity**2 → velocity*2` | survived | **killed** |
| `pressure:78:34` `_positive_finite` `0.0 → -0.0` | survived | survived（**等价变异体**：`x < -0.0` 与 `x < 0.0` 同真同假，杀不掉是正确判定） |

**诚实声明（未完成项）**：修复后的**全量**重测没有跑完。三个模块 × 8 的 `after_fix` 扫描在
"存活者再跑函数名家族测试"（`--broaden`）这一级上耗时远超预算，被我在收尾时中止；`--no-broaden`
的有界扫描同样未跑完（`docs/m16514/after_fix_bounded.log` 为空，进程被我自己的 `pkill -f` 连带杀掉）。
**因此本单不给出"修复后的 mutation score"这个数字**，只给上表这 3 条逐条翻转的证据。重测命令：

```bash
python -m tensorbem.benchmarks.mutation_audit \
  --module tensorbem.rao --module tensorbem.pressure --module tensorbem.helmholtz \
  --out docs/m16514/after_fix --jobs 4 --max-per-module 8
```

### 3.3 修复的有效区（单测证据）

`tests/test_mutation_audit_isolation.py`（4 passed，0.50 s）不跑真扫描，只钉住修复的三条性质：

| 测试 | 钉住的性质 |
|---|---|
| `test_finder_wins_over_a_later_conftest_style_path_insertion` | 插件安装后，模拟 conftest 的 `sys.path.insert(0, repo/src)`，`import tensorbem` 与 `tensorbem.core` 仍解析到影子树（`VALUE = 2.0` 而非仓库的 `1.0`） |
| `test_install_reports_the_resolved_package_path` | 插件把**实际**解析到的路径落盘，事后可审计，不靠"设置了 PYTHONPATH"自证 |
| `test_install_rejects_a_shadow_tree_that_does_not_contain_the_package` | 空影子树 → `RuntimeError`（fail-closed，不回落到仓库树） |
| `test_harness_runs_pytest_with_the_isolation_plugin` | `run_tests` 确实传了 `-p <plugin>` 与三个 env 变量；`_classify` 认 `ISOLATION-FAILURE` |

## 4. 结果图

**本单不出图，理由**：被审计/修复的对象是一个 632 行的证据引擎与其导入隔离机制，结论量是
**分类判定**（killed / survived / isolation_error）与**路径字符串**，不构成随网格/频率/参数连续变化的
场量或曲线；把 4 条分类画成柱状图只会把"没有数字"伪装成"有数字"。可复现的一手证据是三条：
`docs/m16514/probe_isolation.py` 的输出（会话内导入到仓库原树）、`docs/m16514/baseline.log`
（修复前 24/25 存活）、以及 `tests/test_mutation_audit_isolation.py` 的 4 条绿色断言。

## 5. 注意点（踩过的坑 / 局限 / 适用范围）

1. **"证据工具"也要被验证**。此前三轮审计（m2582、m3108 及若干 reopen 单）都在核对"被测代码对不对"，
   没有任何一单核对"审计引擎有没有真的在测它声称测的那份代码"。`_check_isolation` 的存在给了虚假的安心：
   它检查了**机制的一半**（裸导入），漏掉了**实际执行路径**（pytest 会加载 conftest）。
2. **历史结论的有效期**。本仓 `tests/conftest.py` 由 m14560/PR #632 引入。**在含该文件的树上**用旧引擎
   得到的数字（含 `docs/m3108/outputs/mutation_summary.md` 里的 rao 0.250、pressure 0.500）**不可采信，
   需要重测**；更早的树（引擎建立时，2026-11-04）不受此影响，但也应以修复后的引擎复测一次才能引用。
   本单不宣称任何旧数字为假，只指出它们**当前不可复现**。
3. **等价变异体不是盲点**。`0.0 → -0.0`、`>= ` / `>` 的边界、以及只喂给未使用中间变量的算术变异，
   不改变程序语义，杀不掉是**正确**判定。把它们计入"套件盲点"会虚报欠账。
4. **`--broaden` 很贵**。存活者还要跑"提到该函数名的全部测试文件"，在 `tests/` 已有数百用例的树上，
   单条判定可达分钟级。全量重测应交给集群（`run_cluster`），不要压控制面。
5. **适用范围**：修复针对"影子树 + 仓库内测试文件"这一模式，不解决别的污染源（例如 site-packages 里
   同名包、`.pth` 注入）；这些由 `_ShadowFinder` 位于 `sys.meta_path` 首位而免疫，但**插件未加载**的
   非 pytest 用途（例如直接 `python -c` 复现）仍需自己设置隔离。

## 6. 误差判定

本单不含连续量的数值基准（无网格收敛、无解析解对比），因此"是否 < 3 %"**不适用**；不得含糊，声明如下：

| 判定项 | 基准 | 结果 | 是否 < 3 % |
|---|---|---|---|
| 会话内导入的包路径 | 影子树路径（精确子串匹配） | 修复前：仓库原树（不匹配）；修复后：影子树（匹配） | 不适用（分类判定） |
| 4 条已知变异体的判定 | 物理必然（`m ± added_mass` 必被 RAO 测试抓到） | 修复前 4/4 survived；修复后 3 killed + 1 等价变异体 | 不适用（分类判定） |
| 单测 | 修复后插件行为 | 4/4 passed | 不适用 |
| **全量 mutation score（修复后）** | — | **未测（honest FAIL：预算内未跑完）** | **不适用，不许引用** |

honest-FAIL 记录：**修复后的全量重测未完成**，本单只交付"引擎缺陷 + 修复 + 单测护栏 + 3 条逐条翻转证据"，
不交付新的 mutation score。下一单可直接用 §3.2 的命令接续，无需再改引擎。

## 7. 建议的下一单（总账外新缺口）

* **slug 建议**：`GAP-BEM-MUTATION-RERUN`
* **验收**：在修复后的引擎上重跑 m2582/m3108 覆盖的 8 个模块（`kernels`、`free_surface`、`hydrostatic`、
  `solver`、`finite_depth`、`kernels3d`、`pressure`、`helmholtz`、`rao`），`isolation_error = 0`，
  并对每个真实存活者要么补独立护栏测试、要么分类为等价变异体并给出理由；published score 与逐条判定入账。
* **所属域**：TensorBEM（数值方法 / 验证面）。