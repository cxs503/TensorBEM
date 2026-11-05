# m3940 — PR #128（`agent/m3181`）三方向合并与逐文件解冲记录

**仓库** TensorBEM ｜ **目标** 让 PR #128 重新可合并（`mergeable=false → true`），
两侧新增内容一格不丢 ｜ **工作分支** `agent/m3940` ｜ **推回分支** `agent/m3181`

| 角色 | 提交 | 说明 |
|---|---|---|
| 合并基 | `4f5b73e` | PR 最后一次同步 main 的分叉点 |
| main 侧 | `ecba798` | 领先合并基 40 个提交（含 m3150/m3365 等对表 10–12 的登记） |
| PR 侧 | `55db397` | 领先合并基 3 个提交（m3181：三维自由液面注册表 + 复算证据） |

## 1. 冲突清单（机械合并结果）

`git merge --no-ff origin/main`（HEAD=`agent/m3181`）报出两处文本冲突、一处语义冲突：

| 文件 | 冲突类型 | 冲突规模 |
|---|---|---|
| `src/tensorbem/benchmarks/declared_numbers.py` | 文本冲突 | 3 个冲突块（import 别名、注册表主体、注册表尾部） |
| `tests/test_declared_numbers.py` | 文本冲突 | 2 个冲突块（断言注释、group 覆盖集合） |
| `docs/marine_bem_benchmarks.md` | **语义冲突（无冲突标记）** | 两侧各写了一段"表 10–12 注册表"说明与逐格复算表，自动合并把两套口径**并存**，形成互相矛盾的文档 |

第三项是本单最危险的部分：Git 认为文本不重叠（main 写入 §4.10 订正段与 `hulme_hemisphere/hulme_chief/submerged_sphere` 口径，PR 在 §4.11 之后追加 §4.12），
但两个 registry 分组名（`hulme3d/hulme3d_chief/wang_linton3d` 对 `hulme_hemisphere/hulme_chief/submerged_sphere`）与两组 `--tiers heavy` 命令互相矛盾，属"机械 overlay 静默丢内容"的同类风险。

## 2. 冲突的实质：两侧登记了同一批格子

把两份 registry 按 `(表族, 量)` 归一化后比对（`verify_union.py`，脚本读 git 对象、不手抄）：

| 侧 | 唯一格 | 独有格 |
|---|---:|---|
| PR `agent/m3181` | 76 | 4（表 11 参考列 `exact_a33`/`exact_b33`、`chief_point_count`；表 12 机器零上界 `submerged_sphere_20_rotational_coupling`） |
| main `ecba798` | 75 | 3（表 10 近/远场能量恒等式 `near_far_damping_closure`、Haskind 互易 `haskind_surge`/`haskind_heave`） |
| 合并后 | **79** | 0（两亲本并集 = 79，逐格有且仅有一行） |

**四格数值分歧**（同一物理量、两侧声明值不同，必须判定而非"任选其一"）：

| 格 | PR 值 | main 值 | 判定 |
|---|---:|---:|---|
| 表 10 \(B_{11}'\) 观测阶 64 面元 | 1.62 | **1.8021** | 取 main：PR 用理想 \(\sqrt{N}\) 折算（\(h\) 恰减半），main 用实测平均棱长比 \(h_0/h_1=1.87\)，与同表其余列口径一致；main 文档 §4.10（m3150 段）已记录并订正 |
| 表 10 \(B_{11}'\) 观测阶 256 面元 | 1.88 | **1.9377** | 同上 |
| 表 12 球心转动耦合比 80 面元 | 2.2e-4 | **2.1510e-4** | 取 main：两位有效数字舍入误差 |
| 表 12 球心转动耦合比 320 面元 | 2.6e-5 | **2.6481e-5** | 同上 |

判定依据来自 main 文档中与求解器无关的复算记录（`docs/marine_bem_benchmarks.md` §4.10 m3150 段），
而不是"后合并者胜"：PR 的 1.62/1.88 是**旧表口径**（按理想网格减半折算），main 的 1.8021/1.9377 是同一批实测误差按实测棱长比折算的结果，两者之差恰为度量口径差，不是求解器漂移。

## 3. 逐文件解法

### 3.1 `src/tensorbem/benchmarks/declared_numbers.py`

| 决定 | 内容 |
|---|---|
| 分组命名 | 采 **main 口径** `hulme_hemisphere` / `hulme_chief` / `submerged_sphere`（表 10/11/12）。理由：main 侧另有 `test_3d_free_surface_rows_are_guarded_by_the_default_tiers`、文档 §4.10 订正段与 `--tiers fast,standard` 命令全部绑定这组名字；PR 的 `hulme3d*`/`wang_linton3d` 只在分支内自洽 |
| 注册表形态 | 采 **main 的 inline 84 行**为主体（值与 m3150 订正一致）。PR 的 `_TABLE_HULME`/`_TABLE_WANG_LINTON`/`_refinement_rows()`/`_free_surface_3d_rows()` 生成器机制**不保留**：它编码的是同一批表格格子的另一次抄写，且其声明的 4 格已被上式判定为旧口径；保留会形成第二处真值源（违反 AGENTS.md 的 single source of truth）。此为**显式**取舍，非静默丢弃——PR 的 4 个独有格子全部移植，证据见 §2 与 `verify_union.log` |
| PR 独有格移植 | 追加 4 行：`hulme_chief_2_58_exact_a33`(0.40074)/`hulme_chief_2_58_exact_b33`(0.063122)/`hulme_chief_2_58_chief_point_count`(5)/`submerged_sphere_20_rotational_coupling`(`kind="upper"`, 1e-20) |
| 生成器代码并集 | `_chief_rows()` 扩展：除 `unregularized`/`chief_regularized` 两套方案外，把研究行顶层的 `exact_a33`/`exact_b33` 也展平为 `ka2.58_exact_*_dimless`，否则表 11 的解析参考列无法注册（这是 PR 该行的功能需求，main 的展平器缺此分支） |
| import 别名 | 取 main 的 `floating_hemisphere as hemisphere_benchmark` / `submerged_sphere_wave as sphere_benchmark`（PR 的 `hulme_benchmark`/`wang_linton_benchmark` 与之等价，无内容损失） |

结果：registry **84 → 88 行**，无重复键；表 10 31 行 / 表 11 9 行 / 表 12 26 行（合并后口径），其余 22 行不变。

### 3.2 `tests/test_declared_numbers.py`

| 决定 | 内容 |
|---|---|
| 冲突块 1（注释） | 语义并集：保留"负值合法"的双侧理由（PR：三维 BIE 的非物理负垂荡阻尼正是表 11 的断言；main：带符号量存在，只有 0 非法） |
| 冲突块 2（group 覆盖） | 采 main 的分组名集合 |
| PR 新增测试移植 | `test_three_dimensional_free_surface_tables_are_fully_registered` 移植并改为合并计数 31/9/26 + 单表共用一次求解 + 符号/上界断言 |
| PR 新增 HEAVY 门 | `test_documented_hulme_tables_are_reproduced`（40 格）、`test_documented_wang_linton_table_is_reproduced`（26 格）移植并改键名/计数；保留其"二阶收敛是表的物理内容"与"CHIEF 同时恢复条件数与无源性"断言 |

### 3.3 `docs/marine_bem_benchmarks.md`（语义冲突）

新增 §4.12 的合并订正头：三表 → 分组名 → 行数 → 默认层巡行表；§4.12.1 给出 **PR 标签 → 合并后 registry 键**的映射与上表 4 格订正值；
PR 原复算表与"63/63 PASS"作为历史证据保留（明确标注为 m3181 当时判决），
复算命令由 `--tiers heavy`（合并后该层已空，会跑 0 条）改为 `--tiers fast,standard`。
**没有任何一侧的数字被删除**：被订正的 4 格在两处同时出现（原值 + 订正值 + 理由），读者可自行追溯。

## 4. 验证证据

1. `PYTHONPATH=src python3 docs/m3940/verify_union.py` → `PASS: every cell of both parents has exactly one merged row`（日志 `docs/m3940/verify_union.log`）。
2. `PYTHONPATH=src python3 -m pytest tests/test_declared_numbers.py -q`：见 §5 结果表（对照 main 基线同一命令）。
3. 全量核心测试 `python3 -m pytest tests -q`（本树 vs `main` 基线树）对照。
4. `python3 -m tensorbem.benchmarks.declared_numbers --tiers fast,standard --strict` 端到端复算 88 条注册项。

> 关键环境陷阱（本单实测）：本机 `site-packages/__editable__.tensorbem-0.5.0.pth` 指向**另一个 agent 的工作树**，
> 直接 `pytest` 会静默测到别人的源码（本次实测解析到 `/root/tp-agent/work/m3862/src`）。
> 因此所有测试命令都必须显式 `PYTHONPATH=<本树>/src`。

## 5. 测试结果

见本目录 `test_evidence.md`（由实际运行日志摘录，含命令、耗时、通过/失败计数）。

## 6. 局限与未做

- 4.12.1 的映射表覆盖三张 3D 表；其它表族（cylinder/ellipsoid/ursell/interior_harmonics）两侧键名本来一致，未逐格列映射。
- `_refinement_rows()` 等生成器机制按 §3.1 判据移除；若后续要恢复"表格常量 → 注册行"的生成式登记，应在 main 口径下重写并保留其"同表共用一次求解"断言。
- `TENSORBEM_HEAVY_TESTS` 门的两条二阶收敛测试需要在专用窗口跑（本机负载高，默认层复算已用分钟级时间），见 `test_evidence.md` 的说明。