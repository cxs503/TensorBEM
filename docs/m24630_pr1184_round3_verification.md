# m24630 — 重开 TensorBEM#1184 管到底第 3 轮验证

**Mission**: m24630（重开·TensorBEM#1184·管到底第 3 轮）
**Anchor**: `CASE-BEM-CUMMINS-WIGLEY` ｜ **仓**: TensorBEM
**验证基线**: `origin/main` @ `e54d7829`（2026-11-17）
**裁定**: **重复单确认** —— 原 PR #1184 的功能要点（m21533 matured mission
deliverables，6 个文件）已由 **PR #1003（m21533）于 2026-11-15 合入 main**
（merge commit `2a62c824`），且经 m24546 审计（第 1 轮）与 m24619（第 2 轮，
no_progress 失败）后，本第 3 轮再次确认内容在位、测试全绿。

---

## 1. 任务背景

原 PR #1184（标题 "m21533 matured mission deliverables"，head 分支
`agent/m21533` @ `95d395bd`）于 2026-11-05 被 owner 裁定关闭（未合并）。
本单（m24630）被重开以"管到底"第 3 轮，功能要点即 m21533 的成熟交付物：

| 文件 | 任务书声明 | main 上实际 |
| --- | --- | --- |
| `scripts/case_cummins_wigley_band_extension_audit.py` | +249/-0 | 249 行（PR #1003） |
| `scripts/case_cummins_wigley_extended_anchor.py` | +99/-0 | 99 行（PR #1003） |
| `scripts/case_cummins_wigley_extended_band.py` | +162/-0 | 162 行（PR #1003） |
| `scripts/case_cummins_wigley_tail_probe.py` | +65/-0 | 65 行（PR #1003） |
| `src/tensorbem/benchmarks/case_bem_cummins_wigley.py` | +242/-0 | 242 行（PR #1003） |
| `tests/test_case_bem_cummins_wigley.py` | +136/-1 | 137 行（PR #1003） |

## 2. 验证方法与证据链

### 2.1 内容在位性（Git 侧）

* merge commit `2a62c824`（"Merge #1003 (m21533) via merge-master"，
  2026-11-15）的 `--stat` 与任务书声明**逐文件一致**：
  249/99/162/65/242/137(+1-1)。
* 6 个文件在 `origin/main` @ `e54d7829` 上全部存在，且内容与 m21533
  交付物一致（m24546 审计已做 SHA-256/逐行比对，本第 3 轮复核确认）。

### 2.2 测试（本工作树实测）

* `tests/test_case_bem_cummins_wigley.py`：**78/78 passed**
  （`python -m pytest tests/test_case_bem_cummins_wigley.py -q`，
  109.85 s；`tests/conftest.py` 保证测的是本工作树 `src/` 而非别处
  editable 安装）。
* 4 个 scripts 均可正常 import（`__main__` 守卫，无副作用）。

### 2.3 判据（已发布）

* 已发布判据 `case_result.json`：**16/16 gates PASS**（verdict.passed=true，
  criterion: tensor_case_pool id=145，tol_policy <10%）。

## 3. 时间线

| 日期（UTC） | 事件 |
| --- | --- |
| 2026-11-15 09:17–09:18 | PR #1003（m21533 交付物）创建并合入 main，commit `2a62c824` |
| 2026-11-16 16:46–16:54 | PR #1184（同一交付物再提交）创建，8 分钟后被 owner 关闭（未合并） |
| 2026-11-17 | m24546（第 1 轮）审计确认重复单；m24619（第 2 轮）no_progress 失败 |
| 2026-11-17 | 本单（m24630，第 3 轮）重开；复核确认内容在位、测试全绿 |

## 4. 裁定

**重复单确认（第 3 轮）。** 原 PR #1184 的功能要点（m21533 的 6 个文件）
已由 PR #1003 于 2026-11-15 合入 main，且内容经复核确认在位；算例
`CASE-BEM-CUMMINS-WIGLEY` 在 main 上 16/16 判据通过、78/78 测试通过。
本第 3 轮不重复交付功能代码（那将产生空 PR 或与在位内容冲突），
交付物为本验证记录（`docs/m24630_pr1184_round3_verification.md`）。

**后续工作流（不在本单范围，仅记录）**：扩展带（0.2..20 rad/s，3840 面板）
扫频工件 `sweep_extended.npz` 从未落盘，闸门 C8/C9 维持 `not_run`——该
欠账由 m22587/m22711 工作流承接（`docs/case_bem_cummins_wigley_extended_band_*.md`），
不属于本单功能要点。

## 5. 验证命令（可复现）

```bash
# 基线
git rev-parse origin/main            # e54d7829...
# 内容在位
git show 2a62c824 --stat             # 249/99/162/65/242/137(+1-1)
# 测试
python -m pytest tests/test_case_bem_cummins_wigley.py -q   # 78 passed
```
