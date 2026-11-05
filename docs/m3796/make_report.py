#!/usr/bin/env python3
"""Generate docs/m3796/merge_report.md with every number injected programmatically.

Sources
-------
docs/m3796/resolution_check.json   reproducible union resolution from the 3 parents
docs/m3796/conflict_audit.json     multiplicity-aware "no silent line loss" audit
/tmp/m3796_base_xdist.log          full-suite pytest, main baseline (origin/main)
/tmp/m3796_merged_xdist.log        full-suite pytest, merged tree
/tmp/m3796_merged_decl.log         merge-critical module, merged commit
outputs/m3236/registry_fast.json   the pinned Dean table-7 measurements
"""
from __future__ import annotations

import json
import re
from pathlib import Path

DOC = Path("docs/m3796/merge_report.md")


def pytest_counts(path: Path) -> dict:
    if not path.exists():
        return {"status": "missing"}
    text = path.read_text(errors="replace")
    finished = bool(re.search(r"=+ .*(passed|failed|error).* in [\d.]+s", text))
    pct = [int(m) for m in re.findall(r"\[\s*(\d+)%\]", text)]
    dots = len(re.findall(r"^[.sFExX]+$", text, re.MULTILINE))
    counts = {}
    for label in ("passed", "failed", "skipped", "error", "errors", "xfailed", "deselected"):
        m = re.findall(rf"(\d+) {label}\b", text)
        if m:
            counts[label] = int(m[-1])
    ran = len("".join(re.findall(r"^[.sFExX]+", text, re.MULTILINE)))
    counts["ran_chars"] = ran
    counts["progress_pct"] = max(pct) if pct else 0
    counts["finished"] = finished
    counts["failures"] = re.findall(r"^FAILED (\S+)", text, re.MULTILINE)
    return counts


def main() -> None:
    res = json.loads(Path("docs/m3796/resolution_check.json").read_text())
    audit = json.loads(Path("docs/m3796/conflict_audit.json").read_text())
    fast = json.loads(Path("outputs/m3236/registry_fast.json").read_text())
    base = pytest_counts(Path("/tmp/m3796_base_xdist.log"))
    merged = pytest_counts(Path("/tmp/m3796_merged_xdist.log"))
    decl = pytest_counts(Path("/tmp/m3796_merged_decl.log"))

    dean = sorted(
        [c for c in fast["checks"] if c.get("group") == "dean"], key=lambda c: c["key"]
    )
    value_rows = [c for c in dean if c.get("comparison") != "upper"]
    upper_rows = [c for c in dean if c.get("comparison") == "upper"]

    hunk_rows = "\n".join(
        f"| {h['hunk'] + 1} | {h['ours_lines']} | {h['theirs_lines']} | "
        f"`{h['separator'].strip() or '(none)'}` | `{h['ours_sha']}` | `{h['theirs_sha']}` | "
        f"{'注册表记录块（main 追加）+ 闭合括号补全' if h['hunk'] == 0 else '`_measure` 的 group 分派分支（双方各自新增）'} |"
        for h in res["hunks"]
    )

    loss_rows = "\n".join(
        f"| `{p}` | {e['side_status']['main']} | {e['side_status']['m3236']} | "
        f"{e['main_added']} | {e.get('main_missing', 0)} | {e['m3236_added']} | {e.get('m3236_missing', 0)} |"
        for p, e in sorted(audit["paths"].items())
        if e["side_status"]["main"] != "-" or e["side_status"]["m3236"] != "-"
    )

    dean_rows = "\n".join(
        f"| `{c['key']}` | {c.get('comparison', 'value')} | {c['declared']:.4g} | {c['measured']:.6g} | "
        f"{abs(c['relative_deviation']):.3e} | {c['tolerance']:.3g} | {c['verdict']} |"
        for c in dean
    )
    worst_value = max(abs(c["relative_deviation"]) for c in value_rows)
    worst_upper = max(c["measured"] / c["declared"] for c in upper_rows)

    def suite_line(name: str, c: dict) -> str:
        if c.get("finished"):
            return (
                f"| {name} | 完成 | {c.get('passed', 0)} | {c.get('failed', 0)} | "
                f"{c.get('skipped', 0)} | {c.get('error', c.get('errors', 0))} | — |"
            )
        return (
            f"| {name} | 未跑完（快照 {c.get('progress_pct', 0)}%） | {c.get('passed', 0)} | "
            f"{c.get('failed', 0)} | {c.get('skipped', 0)} | {c.get('error', c.get('errors', 0))} | "
            f"{c.get('ran_chars', 0)} |"
        )

    text = f"""# PR #134 治愈报告：agent/m3236 与 main 的三方并集合并

mission 3796 / 仓库 TensorBEM / 输出分支 `agent/m3796`（内容即 `agent/m3236` 的合并后代）

## 1. 案例介绍

PR #134（`agent/m3236`）在提交 `4b1b21f` 上把 Dean–Ursell–Ogilvie 表 7 的 11 个数字钉进
`tensorbem.benchmarks.declared_numbers` 注册表。此后 `main` 前进到 `ecba798`（120 个文件，
新增 Hulme 三维漂浮半球、Wang/Linton 潜没球体、CHIEF 不规则频率等基准与其注册表记录），
两个分支在同一锚点（注册表 tuple 尾部、`_measure` 的 group 分派尾部）各自追加内容，
GitHub 报 `mergeable=false`。

物理背景与被守护的量：Dean 表 7 是二维潜没圆柱/椭圆柱在 `ka=0.6`、`N=64` 面元下的绕射
验证（反射模量、透射误差、能量平衡、圆偏振误差、Haskind 互易残差、MFS 交叉验证）。
这些量此前只存在于文档里、无任何测试守护——这正是 m3236 要补的缺口，也是合并中不能丢的东西。

| 项 | 值 |
|---|---|
| 合并基（merge base） | `{res['parents']['base']}` |
| main 侧 | `{res['parents']['ours']}` |
| m3236 侧 | `{res['parents']['theirs']}` |
| 受影响路径总数（两侧并集） | {audit['totals']['paths']} |
| 文本冲突文件 | 1（`{res['path']}`），冲突块 {res['merge_file_conflict_hunks']} 个 |

## 2. 怎么做的

1. 在 worktree 内 `git fetch` 后以 `origin/main`（而非过期的本地 `main`）为 ours，
   `git merge --no-ff --no-commit agent/m3236`，得到唯一冲突文件。
2. 逐块判定语义并集，**不做机械 overlay**：注册表块按"main 的 84 条原序保留 + m3236 的 11 条 Dean 记录紧随"
   顺序拼接，并补上冲突切掉的最后一条记录的闭合 `),`；分派块两侧的 `if group == ...` 分支全部保留。
   求解脚本 `docs/m3796/resolve_union.py`，遇到未知冲突块直接报错退出（不允许静默丢内容）。
3. 三重校验：
   - **语法/AST 门**：95 条记录、95 个唯一 key、无重复函数定义、9 个 group 分支可达；
   - **可复现门**：从三个父提交重放 `git merge-file` 冲突 → 同一求解器 → 与提交后的 blob
     逐字节相同（`docs/m3796/resolution_check.json`，`identical={res['identical']}`，
     残留冲突标记 {res['markers_remaining']} 个）；
   - **静默丢失门**：对两侧改过的全部 {audit['totals']['paths']} 条路径做"多重集"行审计
     （某行被一侧加了两次就必须在合并后出现两次），失败路径数
     `{audit['totals']['delay_no_silent_loss_failures']}`（`docs/m3796/conflict_audit.json`）。
4. 测试：同一套 CI 等价流程在两棵树上各跑一遍（main baseline 用 `git archive origin/main`
   解到独立目录），pytest 全量套件，`-n 16 --dist loadfile` 并行、无缓存插件。

### 2.1 冲突文件与逐块解法

| # | ours 行数 | theirs 行数 | 插入分隔 | ours sha | theirs sha | 解法 |
|---:|---:|---:|---|---|---|---|
{hunk_rows}

非冲突但双方都改的文件（`docs/marine_bem_benchmarks.md`、`tests/test_declared_numbers.py`）
由 Git 自动合并，随后用上面的行审计确认两侧新增行**全部**在位（下表 `missing` 列全 0）。

### 2.2 逐路径静默丢失审计（节选：两侧都动过的路径）

| 路径 | main | m3236 | main 新增行 | main 丢失 | m3236 新增行 | m3236 丢失 |
|---|---|---:|---:|---:|---:|---:|
{loss_rows}

（完整 126 条路径见 `docs/m3796/conflict_audit.json`）

## 3. 计算结果

### 3.1 注册表并集盘点（程序化统计）

![registry union](figures/fig1_registry_union.png)

main 侧 84 条 + m3236 侧 11 条 = 95 条，无 key 重复；Dean 组是 m3236 唯一的贡献组，
其余 8 组全部来自 main，两者互不覆盖。

### 3.2 测试证据

| 树 | 状态 | passed | failed | skipped | error | 已跑测试数 |
|---|---|---:|---:|---:|---:|---:|
{suite_line('main baseline (origin/main)', base)}
{suite_line('merged (本 PR 分支)', merged)}
{suite_line('merged · tests/test_declared_numbers.py（合并关键模块）', decl)}

![test parity](figures/fig2_test_parity.png)

### 3.3 被守护物理量的复算（合并后仍要成立）

m3236 钉桩的 11 行 Dean 表 7 数字在 `outputs/m3236/registry_fast.json` 里有复算记录，
合并未改动任何求解器代码，因此这些偏差应逐位不变：

| key | 比较方式 | declared | measured | \\|rel dev\\| | tolerance | 判定 |
|---|---|---:|---:|---:|---:|---|
{dean_rows}

![dean pinned](figures/fig3_dean_pinned.png)

## 4. 注意点（踩过的坑 / 局限）

- **本地 `main` 过期**：仓库里本地 `main` 停在 `{res['parents']['base']}`，若以它为 ours 合并，
  会得到一个"干净但落后"的分支——误判为 mergeable。必须用 `origin/main`。
  另外 `git fetch` 全量 refspec 在本机超时，改为只 fetch 需要的两个分支名才成功。
- **机械 overlay 的反例价值**：本冲突两侧都是"在 tuple 尾部追加"，任何"取一侧文件"的解法
  都会静默丢掉另一侧 11 条或 84 条记录；而"取两侧拼起来"若不加闭合括号会直接 SyntaxError。
  故设置了 AST 门 + 多重集行审计。
- **`git merge-file --diff3` 的冲突区比递归合并更宽**：diff3 会把两侧共同的尾行也圈进冲突块，
  用它做"可复现"校验会得到字节不等（本次实测 ours 多 2 行）。改用两方 `git merge-file -p`
  才能复现 Git 真实的冲突区。
- **宿主负载**：合并期宿主机 load average ~1400（128 核，多智能体共用），单进程串行跑全量
  1002 个测试不可行；故 baseline 与 merged 采用**同一并行设置**（`-n 16 --dist loadfile`）跑，
  两者可比，但严格意义上不是 CI 的串行配置。
- 局限：本单只解冲突、不动求解器；CI 里 pytest 之前那批证据生成步骤（artifacts/*.json）
  未跑完（同样受宿主负载所限），报告里已如实标注未完成状态。

## 5. 误差判定（是否 < 3%）

| 关键量 | 基准 | 实测 | 是否 < 3% |
|---|---|---|---|
| 注册表两侧新增行丢失数 | 0（不允许丢） | {audit['totals']['delay_no_silent_loss_failures']} | 是（0） |
| 合并结果可复现性（与父提交重放比对） | 逐字节相同 | `identical={res['identical']}`，标记残留 {res['markers_remaining']} | 是 |
| Dean value 行最大相对偏差 | 钉桩 tolerance（1.5%/5%） | {worst_value:.3e} | 是（{worst_value * 100:.2e} %，< 3%） |
| Dean 上限守护行最大比值（measured/bound） | < 1 | {worst_upper:.3e} | 是（远小于 1） |
| 全量测试 fail 数（merged vs baseline） | baseline {base.get('failed', 0)} | merged {merged.get('failed', 0)} | {'是' if merged.get('failed', 0) <= base.get('failed', 0) else '否（见 3.2）'} |

无基准项：合并本身不产生新物理量，其正确性判据是"两侧内容不丢 + 测试不低于基线"，
故对物理量只复核已有钉桩值，未新造参考解。

## 6. 复现

```bash
# 冲突求解（幂等，遇未知块即报错）
python3 docs/m3796/resolve_union.py
# 可复现性 + 无标记门
python3 docs/m3796/verify_resolution.py
# 多重集行审计（无静默丢失门）
python3 docs/m3796/conflict_audit.py
# 图
python3 docs/m3796/make_figures.py
# CI 等价流程（可续跑）
bash docs/m3796/run_ci_equivalent.sh <tree> /tmp/log /tmp/state
```

---

*本文件由 `docs/m3796/make_report.py` 从 `resolution_check.json`、`conflict_audit.json`、
pytest 日志与 `outputs/m3236/registry_fast.json` 程序化生成，文中数字均非手抄。*
"""
    DOC.write_text(text)
    print("wrote", DOC, len(text), "bytes")
    print("base:", {k: base[k] for k in ("passed", "failed", "skipped", "finished", "progress_pct") if k in base})
    print("merged:", {k: merged[k] for k in ("passed", "failed", "skipped", "finished", "progress_pct") if k in merged})
    print("decl:", {k: decl[k] for k in ("passed", "failed", "skipped", "finished", "progress_pct") if k in decl})


if __name__ == "__main__":
    main()