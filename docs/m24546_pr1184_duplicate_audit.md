# m24546 — 重开 TensorBEM#1184 重复单审计（管到底第 1 轮）

**Mission**: m24546（重开·TensorBEM#1184·管到底第 1 轮）
**Anchor**: `CASE-BEM-CUMMINS-WIGLEY` ｜ **仓**: TensorBEM
**审计基线**: `origin/main` @ `cf8a191f`（2026-11-17）
**裁定**: **重复单** —— 原 PR #1184 的功能要点（m21533 matured mission
deliverables，6 个文件）已经由 **PR #1003（m21533）于 2026-11-15 合入 main**
（merge commit `2a62c824`）。本单不重复交付功能代码，只交付本审计记录。

---

## 1. 任务背景

原 PR #1184（标题 "m21533 matured mission deliverables"，head 分支
`agent/m21533` @ `95d395bd`）于 2026-11-05 被 owner 裁定关闭（未合并）。
本单（m24546）被重开以"管到底"，功能要点即 m21533 的成熟交付物：

| 文件 | 任务书声明 |
| --- | --- |
| `scripts/case_cummins_wigley_band_extension_audit.py` | +249/-0 |
| `scripts/case_cummins_wigley_extended_anchor.py` | +99/-0 |
| `scripts/case_cummins_wigley_extended_band.py` | +162/-0 |
| `scripts/case_cummins_wigley_tail_probe.py` | +65/-0 |
| `src/tensorbem/benchmarks/case_bem_cummins_wigley.py` | +242/-0 |
| `tests/test_case_bem_cummins_wigley.py` | +136/-1 |

## 2. 审计方法与证据链

### 2.1 GitHub 侧（GitHub REST API，2026-11-17 查询）

* **PR #1003**：标题 "matured mission deliverables"，head `agent/m21533`，
  2026-11-15T09:17:43Z 创建、09:18:12Z 合并（30 秒内）。`/files` 端点返回
  6 文件清单与增删行数（249/99/162/65/242/136+1）。
* **PR #1184**：标题 "m21533 matured mission deliverables"，head 分支
  `agent/m21533` @ `95d395bd`，2026-11-16T16:46:48Z 创建、16:54:09Z 被
  owner 关闭（`merged_at=null`，未合并）。其 `/files` 端点返回的 6 个文件
  与增删行数**逐文件**与 PR #1003 及任务书声明完全一致
  （249/99/162/65/242/136+1）。
* 注：任务书所记关闭日期 "2026-11-05" 与 GitHub 实际 `closed_at`
  （2026-11-16T16:54:09Z）不符，以 GitHub 为准。
* 当前仓库 **open PR 数 = 0** → 本单不存在热文件重叠，无需热文件声明。

### 2.2 Git 侧（本工作树，基线 `cf8a191f` = origin/main）

* merge commit `2a62c824`（"Merge #1003 (m21533) via merge-master"，
  2026-11-15 17:18 +0800）的 `--stat` 与任务书声明**逐文件一致**：
  249/99/162/65/242/137(+1-1)。
* 内容级比对（`git show 95d395bd:<file>` vs `git show cf8a191f:<file>`，
  SHA-256）：
  * 4 个 scripts：**逐字节相同**（IDENTICAL）。
  * `case_bem_cummins_wigley.py`：m21533 新增的 229 个非空行中 228 行在
    main 中逐字存在；唯一差异是变量改名
    `float(exponent)` → `float(exponent_single)`（键名
    `exponent_at_20` 不变，语义不变），由后续 mission m22587（PR #1092）
    的装配链路修复引入。
  * `tests/test_case_bem_cummins_wigley.py`：m21533 新增的 119 个非空行
    （diff stat 136 行含空行）全部逐字在位，且该文件此后被后续 mission 继续
    扩展（现 1508 行 / 78 测试）。

### 2.3 功能侧（当前 main 实测）

* `tests/test_case_bem_cummins_wigley.py`：**78/78 passed**
  （`python -m pytest tests/test_case_bem_cummins_wigley.py -q`，
  105.8 s；`tests/conftest.py` 保证测的是本工作树 `src/` 而非别处
  editable 安装）。
* 4 个 scripts 均可正常 import（`__main__` 守卫，无副作用）。
* 已发布判据 `case_result.json`：**16/16 gates PASS**（verdict.passed=true，
  criterion: tensor_case_pool id=145，tol_policy <10%）。
* 全套测试（集群，stage_dir=本工作树，8 核）：
  * `cj5117`：`pytest tests/ -q` 在 loong64 节点（n330）于**收集阶段**报 3 个
    预存错误——`tests/test_case_havelock_round3.py`、`tests/test_neumann_kelvin.py`、
    `tests/test_neumann_kelvin_verification.py`（`numpy.linalg.LinAlgError:
    Eigenvalues did not converge`，import 期特征值分解失败）。本机 x86 上同 3 文件
    收集正常（69 测试），故为节点 LAPACK 环境问题，与本单改动（仅 1 个 docs/
    文件）无关，属 main 预存的环境相关收集失败。
  * 收集面普查（`cj5131`，`--collect-only`）：loong64 节点上共 **5 个**测试文件
    在 import 期因 `numpy.linalg.LinAlgError`（特征值/特征向量不收敛，节点
    LAPACK 环境问题）收集失败：`test_case_havelock_round3.py`、
    `test_core_api_analytic_pins.py`、`test_neumann_kelvin.py`、
    `test_neumann_kelvin_verification.py`、`test_nk_steady_kernel_candidate.py`；
    其余 **6016 个测试**正常收集。本机 x86 上同批文件收集正常（如 3 文件
    69 测试），故为节点环境预存问题，与本单改动（仅 1 个 docs/ 文件）无关。
  * 全套执行（`cj5157`，`--continue-on-collection-errors`，7200 s 上限，
    n330）：6016 个可收集测试的全量执行；前一次尝试 `cj5132` 因 3600 s
    作业上限被截断（非测试失败）。`cj5157` 终态结果以
    `/root/tp-agent/cluster/jobs/cj5157.log` 为准；CI（x86 runner）在 PR
    上重跑全套为权威闸门。

## 3. 时间线

| 日期（UTC） | 事件 |
| --- | --- |
| 2026-11-15 09:17–09:18 | PR #1003（m21533 交付物）创建并合入 main，commit `2a62c824` |
| 2026-11-16 16:46–16:54 | PR #1184（同一交付物再提交）创建，8 分钟后被 owner 关闭（未合并） |
| 2026-11-15/16 | 后续 mission 继续扩展该算例：#1092 (m22587) 装配链路修复、#1117 (m22881)、#1127 (m23091)、#1130 (m23122) |
| 2026-11-17 | 本单（m24546）重开；审计确认功能要点已在 main |

## 4. 裁定

**重复单。** 原 PR #1184 的功能要点（m21533 的 6 个文件）已经由 PR #1003
于 2026-11-15 合入 main，且内容经 SHA-256/逐行比对确认在位；算例
`CASE-BEM-CUMMINS-WIGLEY` 在 main 上 16/16 判据通过、78/78 测试通过。
本单不重复交付功能代码（那将产生空 PR 或与在位内容冲突），交付物为本
审计记录（`docs/m24546_pr1184_duplicate_audit.md`）。

**后续工作流（不在本单范围，仅记录）**：扩展带（0.2..20 rad/s，3840 面板）
扫频工件 `sweep_extended.npz` 从未落盘，闸门 C8/C9 维持 `not_run`——该
欠账由 m22587/m22711 工作流承接（`docs/case_bem_cummins_wigley_extended_band_*.md`），
不属于本单功能要点。

## 5. 验证命令（可复现）

```bash
# 基线
git rev-parse origin/main            # cf8a191f...
# 内容比对
git fetch origin 95d395bdd241a975d7e0a2d0f545be89feabe7a5
git show 95d395bd:scripts/case_cummins_wigley_tail_probe.py | sha256sum
git show cf8a191f:scripts/case_cummins_wigley_tail_probe.py | sha256sum
# 测试
python -m pytest tests/test_case_bem_cummins_wigley.py -q   # 78 passed
```
