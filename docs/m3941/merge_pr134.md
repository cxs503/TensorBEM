# PR #134 治愈报告：`agent/m3236` → `main` 三方合并（m3941）

## ① 案例介绍

| 项 | 值 |
|---|---|
| 仓库 | TensorBEM（三维/二维 BEM 频域水动力求解器库） |
| PR | #134，分支 `agent/m3236` |
| 分支旧 SHA | `1615c100895de9d1e487b5886fd68adea5b0f770`（tip，含 "re-run CI against current main"） |
| main SHA（合并目标） | `ecba798833d6c2429fbc897d2ce515efb70dd682` |
| 三方合并基 | `4f5b73e73c4e731582f2b050328ec5fcf62b4a86` |
| 合并后提交 | 见 `git rev-parse HEAD`（merge commit，父提交 = `ecba798` + `1615c10`） |
| PR 状态 | `mergeable=false`（对 main 变脏） |

物理/工程背景：`agent/m3236` 把 Dean–Ursell–Ogilvie 二维自由面绕射研究（`docs/marine_bem_benchmarks.md` 第 4.8 节表 7 的 11 个数字）纳入 `tensorbem.benchmarks.declared_numbers` 注册表，并加回归门。该分支写作期间 main 前进了 5 个合并（m3134/m3125/m3150/m3196/m3224/m3280/m3336/m3383/m3409 等），也独立地把同一注册表扩充到 84 行（Hulme 半球、淹没球、CHIEF 等）。两侧都在同一位置追加，于是 PR 变脏。

参数/规模表（合并涉及面）：

| 文件 | main 侧改动 | 分支侧改动 | 冲突 |
|---|---|---|---|
| `src/tensorbem/benchmarks/declared_numbers.py` | +31 行（hulme_hemisphere / submerged_sphere / hulme_chief 条目 + `_run_group` 分派） | +170 行（11 条 dean 条目 + `dean` 分派） | **2 处 content conflict** |
| `docs/marine_bem_benchmarks.md` | 13 行改（含表 7 上下文） | +8 行（表 7 复算钉桩说明） | 文本自动合并，需人工复核 |
| `tests/test_declared_numbers.py` | +55 行（3D 行/CHIEF 门） | +32 行（`test_fast_tier_reproduces_the_documented_dean_numbers`） | 文本自动合并，需人工复核 |
| `docs/m3236/outputs/m3236/*` | — | 6 个新文件（dean 32/64/128 JSON + 2 图 + registry_fast.json） | 纯新增 |

## ② 怎么做的

1. **建工作树**：`/root/tp-agent/work/m3941`（linked worktree，位于仓库 `.git/worktrees/m3941`），当前分支 `agent/m3941` = `origin/main`；`git fetch origin` 后取 `origin/agent/m3236`。
2. **三方合并**：`git merge --no-ff --no-commit origin/agent/m3236`。Git 自动合并 2 个文件、报 1 个文件冲突（`declared_numbers.py`，2 处 hunk）。
3. **逐文件解冲（语义并集，禁止机械 overlay）**：
   - `declared_numbers.py` 冲突 1（行 327–1051）：registry 元组尾部。main 侧以 `submerged_sphere_320_order_fz` 条目的 `tier="standard",` 结尾，分支侧是 11 条 `dean` 条目。**两侧都保留**，并把 Git 从"公共尾"里省掉的该条目闭合 `),` 手工补回（这是唯一的语义暗礁：直接删标记会语法错，机械 overlay 会丢掉 main 的 84 行）。
   - `declared_numbers.py` 冲突 2（行 1160–1220）：`_run_group` 分派链。main 侧 `hulme_hemisphere` / `submerged_sphere` / `hulme_chief` 三个分支，分支侧 `dean` 分支。**两侧都保留**，`raise ValueError` 仍是唯一兜底。
   - `docs/marine_bem_benchmarks.md`、`tests/test_declared_numbers.py`：Git 自动合并。**没有信任自动结果**——用 `difflib` 把两侧相对 merge-base 的新增行逐行取出，逐行在合并结果里检索：main 新增 9/55 行、分支新增 8/32 行，**全部命中，0 行丢失**。
   - 合并后校验：`ast.parse` 通过；`key=` 正则计数 95 行、唯一；`def` 无重复定义（防"正则并集坏缩进"类事故）。
4. **测试**：`PYTHONPATH=<worktree>/src` 下跑 `tests/test_declared_numbers.py`（合并树）与同一文件在 `origin/main` 独立工作树（`/tmp/m3941-base`，基线）。
5. **证据程序化**：`docs/m3941/make_artifacts.py`（注册表普查 + Dean 复算 JSON）→ `docs/m3941/make_figures.py`（matplotlib PNG）。数字零手抄。

## ③ 计算结果

### 3.1 注册表普查（`docs/m3941/census.json`，机器断言）

| 组 | main | agent/m3236 | 合并 | 父并集 | 丢失 |
|---|---|---|---|---|---|
| cylinder | 2 | 2 | 2 | 2 | 0 |
| dean | 0 | 11 | **11** | 11 | 0 |
| ellipsoid | 3 | 3 | 3 | 3 | 0 |
| hulme_chief | 6 | 0 | 6 | 6 | 0 |
| hulme_hemisphere | 31 | 0 | 31 | 31 | 0 |
| interior_harmonics | 6 | 6 | 6 | 6 | 0 |
| submerged_sphere | 25 | 0 | 25 | 25 | 0 |
| ursell | 9 | 9 | 9 | 9 | 0 |
| ursell_reference | 2 | 2 | 2 | 2 | 0 |
| **合计** | **84** | **33** | **95** | **95** | **0** |

`no_row_lost=True`，重复 key = 0。合并树条目数 = 84 + 11 = 95，与父并集逐一相等（见图 1）。

### 3.2 合并树重算 Dean 表 7（`docs/m3941/registry_merged.json`，`group=dean`, fast 层, ka=0.6, N=64）

| key | 类型 | 文档值 d | 合并树实测 m | 偏差 \|m−d\|/\|d\| | 判定 |
|---|---|---|---|---|---|
| cylinder_reflection_64 | 上限 | 1e-12（守卫） | 1.3215e-15 | 1.3e-03×守卫 | PASS |
| cylinder_transmission_error_64 | 值 | 8.39e-03 | 8.39366e-03 | +4.36e-04 | PASS |
| cylinder_energy_balance_64 | 上限 | 1e-12（守卫） | 0（精确） | 0 | PASS |
| circular_polarisation_64 | 上限 | 1e-10（守卫） | 6.33908e-15 | 6.3e-05×守卫 | PASS |
| haskind_x_64 | 值 | 2.16e-05 | 2.16404e-05 | +1.87e-03 | PASS |
| haskind_z_64 | 值 | 2.16e-05 | 2.16404e-05 | +1.87e-03 | PASS |
| mfs_force_x_64 | 值 | 1.51e-03 | 1.50584e-03 | −2.76e-03 | PASS |
| ellipse_reflection_64 | 值 | 9.751e-02 | 9.75139e-02 | +3.97e-05 | PASS |
| ellipse_transmission_64 | 值 | 0.995234 | 0.995234 | +1.67e-07 | PASS |
| ellipse_haskind_x_64 | 值 | 7.97e-04 | 7.96854e-04 | −1.83e-04 | PASS |
| ellipse_haskind_z_64 | 值 | 1.99e-03 | 1.99491e-03 | +2.47e-03 | PASS |

`summary = {checked: 11, passed: 11, drifted: 0, verdict: PASS}`。三行机器精度守恒量（零反射、能量平衡、圆偏振）按上限语义守护，其余 8 个离散误差量最差 2.76e-3（MFS 水平力），与分支文档声明（≤2.8e-3）一致 → **合并没有让分支数字漂移**。

### 3.3 测试证据（`PYTHONPATH=<worktree>/src`，pytest 9.2.0，`--timeout=300`）

| 树 | 命令 | 结果 |
|---|---|---|
| 合并树 | `pytest tests/test_declared_numbers.py -q`（20 min wall-clock 上限内完成前 9 项） | `s....F..s`：6 passed，2 skipped（heavy 层），1 failed = `test_standard_tier_reproduces_the_documented_ursell_numbers` |
| 基线 main `ecba798` | 同一文件，`/tmp/m3941-base` 独立工作树 | `s...F..s...`：同一位置同一测试 failed，其余一致 |
| 合并树 | `make_artifacts.py --with-check`（Dean 11 项复算） | 11/11 PASS |
| 合并树 | 注册表普查断言 | `no_row_lost=True`，0 重复 |

**基线判定**：唯一红灯 `test_standard_tier_reproduces_the_documented_ursell_numbers` 在 main 上同样红 → 属 main 既有欠账（Ursell 128 面板标准层），**不是本合并引入的回归**；合并树相对基线"不低于"（红灯集合相同，且多通过一条分支新增的 dean 门）。PR 的 dean 门 `test_fast_tier_reproduces_the_documented_dean_numbers` 在合并树上 PASS（`checked==11`，`verdict==PASS`）。

### 3.4 图

- 图 1 `figures/fig_registry_census.png`：三棵树逐组条目数柱状图，合并柱严格落在 `max(main, branch)` 上，dean 组 0→11 而 main 独有组不缩水。
- 图 2 `figures/fig_dean_registry_merged.png`：合并树 11 个 Dean 行的偏差（对数轴），值行用相对偏差、三行上限用 `m/守卫`，并画 1e-2 容差带与 1.0 守卫线。

## ④ 注意点（坑与局限）

1. **editable 安装指向别人的工作树**：本机 `site-packages/__editable__.tensorbem-0.5.0.pth` 指向 `/root/tp-agent/work/m3862`。直接 `python3 -m pytest` 会**静默测试另一棵代码树**（表现为 `dean` 组"消失"、`checked==0` 的假失败）。必须显式 `PYTHONPATH=<自己的 worktree>/src`，并用 `declared_numbers.__file__` 核实。
2. **冲突形状是"两侧同点追加"**：Git 把公共尾（`    ),`）只留在标记之后，手工解冲必须补回，否则 `ast` 直接语法错；而任何"取一侧"或 overlay 都会静默丢 84 或 11 行。本单用 `key=` 计数 + 父并集比对把它变成可断言量（`census.json`）。
3. **自动合并也要复核**：md 与测试文件虽无冲突标记，仍用 difflib 逐新增行核对（0 丢失）；这是"机械 overlay 有静默丢内容前科"的对症检查。
4. **宿主负载**：审计时 `load average ≈ 700–840`（128 核，多智能体并发跑测试），单测实际耗时被放大约 5–10 倍，20 min 上限内只跑完 18 项中的 9 项；因此本报告**不声称全仓 94 个测试文件全绿**，只声称"冲突相关核心测试不低于基线"。
5. **推回分支**：按纪律不自推，合并提交留在 `agent/m3941`（平台自动推送）；`agent/m3236` 旧 SHA `1615c10`，新合并提交 SHA 见 `git rev-parse HEAD`（`6613185f8f41`，父 = `ecba798` + `1615c10`）。PR #134 的最终合并由 merge_governor / pr-gate 复验完成。

## ⑤ 误差判定（逐项）

| 关键量 | 基准 | 误差 | 是否 <3% |
|---|---|---|---|
| Dean 表 7 值类 8 行（最大者 MFS 水平力 1.51e-3） | 文档表 7 声明值 | max 2.76e-3 = **0.276%** | **是** |
| Dean 行反射模量 9.751e-2 | 文档 9.751e-2 | 3.97e-5 = 0.0040% | 是 |
| 椭圆透射模量 0.995234 | 文档 0.995234 | 1.67e-7 = 1.7e-5% | 是 |
| 零反射（物理应 R≡0） | 解析 R=0 | 1.32e-15（机器精度量级） | 是（且远低于 1e-12 守卫） |
| 能量平衡 \|R\|²+\|T\|²=1 | 解析 1 | 0（精确） | 是 |
| 圆偏振 F_z=−iF_x | 解析关系 | 6.34e-15 | 是 |
| 注册表条目保留率 | 父并集 95 行 | 丢失 0 / 重复 0 | 是（100% 保留） |
| 测试回归 | main 基线红灯集合 | 相同（1 项 Ursell 既有红灯），无新增红灯 | 是 |

无基准项：无。未完成项（honest-FAIL）：全仓 94 文件测试套件在本宿主负载下未跑完，未取得"全绿"结论。

## ⑥ 复现

```bash
cd <worktree>            # 分支 agent/m3941
PYTHONPATH=<worktree>/src python3 docs/m3941/make_artifacts.py --with-check
PYTHONPATH=<worktree>/src python3 docs/m3941/make_figures.py
PYTHONPATH=<worktree>/src python3 -m pytest tests/test_declared_numbers.py -q
```