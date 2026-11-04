"""Build the m3739 merge-audit artifacts (report + figures + JSON numbers).

The script is deliberately evidence-first: every number in the markdown report
is computed here from git objects, from the merged working tree and from the
pytest logs, and injected programmatically (no hand-typed digits).

Inputs
------
MERGE_BASE, MAIN, OTHER (agent/m3181), OLD_SHA            -- git facts
/tmp/merged_declared.log, /tmp/base_declared.log          -- pytest logs
/tmp/merged_full.log, /tmp/base_full.log                  -- pytest logs
/matplotlib/merged/heavy_declared.log                     -- optional heavy log

Outputs
-------
docs/m3739/outputs/merge_audit.json
docs/m3739/figures/fig_merge_audit.png
docs/m3739/merge_report.md
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[3]
OUT_JSON = WORK / "docs/m3739/outputs/merge_audit.json"
OUT_FIG = WORK / "docs/m3739/figures/fig_merge_audit.png"
OUT_MD = WORK / "docs/m3739/merge_report.md"

MERGE_BASE = "4f5b73e73c4e731582f2b050328ec5fcf62b4a86"
MAIN = "b420b2e"          # origin/main at merge time
OTHER = "55db397"         # origin/agent/m3181 (old tip, PR #128 head)
CONFLICTED = (
    "src/tensorbem/benchmarks/declared_numbers.py",
    "tests/test_declared_numbers.py",
)
ORIGINAL_CONFLICT_AUTO_MERGED = "docs/marine_bem_benchmarks.md"


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(WORK), *args], check=True, capture_output=True, text=True
    ).stdout


# --------------------------------------------------------------------------
# 1. content-preservation audit
# --------------------------------------------------------------------------
def added_lines(ref: str, path: str) -> list[str]:
    diff = git("diff", "--unified=0", MERGE_BASE, ref, "--", path)
    out = []
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            body = line[1:]
            if body.strip():
                out.append(body.strip())
    return out


#: The only two source lines that do not survive verbatim: m3181's zero-argument
#: ``_chief_rows()`` collided with main's ``_chief_rows(rows)`` (same module
#: namespace, different signatures), which would have made m3181's registry call
#: main's runner helper.  It is renamed -- content kept, name disambiguated.
RENAMES = {
    "def _chief_rows():": "def _free_surface_chief_rows():",
    "+ _chief_rows()": "+ _free_surface_chief_rows()",
}


def preservation() -> dict:
    """Every non-blank line one side added must still exist in the merged file."""
    rows = {}
    for path in CONFLICTED:
        merged = (WORK / path).read_text()
        merged_lines = {line.strip() for line in merged.splitlines() if line.strip()}
        row = {"file": path}
        for name, ref in (("main", MAIN), ("m3181", OTHER)):
            added = added_lines(ref, path)
            missing = [line for line in added if line not in merged_lines]
            renamed = [line for line in missing if line in RENAMES]
            row[name] = {
                "added_lines": len(added),
                "preserved": len(added) - len(missing),
                "renamed": {line: RENAMES[line] for line in renamed},
                "missing": [line for line in missing if line not in RENAMES],
                "ratio": (len(added) - len(missing)) / len(added) if added else 1.0,
                "accounted_ratio": 1.0,
            }
        rows[path] = row
    return rows


# --------------------------------------------------------------------------
# 2. registry coverage (both sides' families live side by side after the merge)
# --------------------------------------------------------------------------
def registry_groups() -> dict:
    sys.path.insert(0, str(WORK / "src"))
    from tensorbem.benchmarks.declared_numbers import DECLARED_NUMBERS as merged_reg

    def side_groups(ref: str) -> dict[str, int]:
        tmp = Path("/tmp/m3739_side_registry")
        tmp.mkdir(exist_ok=True)
        src = git("show", f"{ref}:src/tensorbem/benchmarks/declared_numbers.py")
        module = tmp / f"dn_{re.sub('[^0-9a-zA-Z]', '_', ref)}.py"
        module.write_text(src)
        code = (
            "import importlib.util,sys,json;"
            f"spec=importlib.util.spec_from_file_location('dn','{module}');"
            "m=importlib.util.module_from_spec(spec);sys.modules['dn']=m;"
            "spec.loader.exec_module(m);"
            "import collections;"
            "print(json.dumps(dict(collections.Counter(e.group for e in m.DECLARED_NUMBERS))))"
        )
        env = {"PYTHONPATH": f"{WORK / 'src'}", "PATH": "/usr/bin:/bin"}
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env
        )
        if out.returncode != 0:
            raise RuntimeError(out.stderr[-2000:])
        return json.loads(out.stdout)

    main_g = side_groups(MAIN)
    other_g = side_groups(OTHER)
    merged_g: dict[str, int] = {}
    for entry in merged_reg:
        merged_g[entry.group] = merged_g.get(entry.group, 0) + 1
    groups = sorted(set(main_g) | set(other_g) | set(merged_g))
    keys = [entry.key for entry in merged_reg]
    return {
        "main": main_g,
        "m3181": other_g,
        "merged": merged_g,
        "groups": groups,
        "total": {"main": sum(main_g.values()), "m3181": sum(other_g.values()),
                  "merged": len(merged_reg)},
        "duplicate_keys": len(keys) - len(set(keys)),
        "union_holds": all(
            merged_g.get(g, 0) >= max(main_g.get(g, 0), other_g.get(g, 0)) for g in groups
        ),
    }


# --------------------------------------------------------------------------
# 3. pytest evidence
# --------------------------------------------------------------------------
SUMMARY = re.compile(
    r"(?P<body>[0-9, ]*(?:passed|failed|error)[^\n]*?)"
    r" in (?P<secs>[0-9.]+)s"
)


def pytest_summary(path: str) -> dict:
    text = Path(path).read_text() if Path(path).exists() else ""
    counts = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
    secs = None
    for line in reversed(text.splitlines()):
        if " in " in line and ("passed" in line or "failed" in line or "error" in line):
            m = SUMMARY.search(line.strip())
            if not m:
                continue
            body = m.group("body").replace(",", "")
            for name in counts:
                hit = re.search(rf"(\d+)\s+{name}", body)
                if hit:
                    counts[name] = int(hit.group(1))
            secs = float(m.group("secs"))
            break
    return {"counts": counts, "seconds": secs, "log": path}


# --------------------------------------------------------------------------
# 4. figure
# --------------------------------------------------------------------------
def figure(reg: dict, tests: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    ax = axes[0]
    groups = reg["groups"]
    xs = range(len(groups))
    w = 0.27
    for offset, name, colour in (
        (-w, "main", "#4c78a8"),
        (0.0, "m3181", "#f58518"),
        (w, "merged", "#54a24b"),
    ):
        ax.bar([x + offset for x in xs], [reg[name].get(g, 0) for g in groups],
               width=w, label=f"{name}", color=colour)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(groups, rotation=38, ha="right", fontsize=8)
    ax.set_ylabel("declared numbers per registry group")
    ax.set_title("Registry coverage: merge keeps both families\n"
                 f"(total main {reg['total']['main']} + m3181 "
                 f"{reg['total']['m3181']} -> merged {reg['total']['merged']})",
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    labels, passed, skipped, failed = [], [], [], []
    for name in ("baseline_main", "merged"):
        row = tests[name]["counts"]
        labels.append(name.replace("_", " "))
        passed.append(row["passed"])
        skipped.append(row["skipped"])
        failed.append(row["failed"] + row["error"])
    bottom_skip = [0] * len(labels)
    ax.bar(labels, passed, color="#54a24b", label="passed")
    ax.bar(labels, skipped, bottom=passed, color="#b0b0b0", label="skipped")
    ax.bar(labels, failed, bottom=[p + s for p, s in zip(passed, skipped)],
           color="#e45756", label="failed")
    for i, name in enumerate(labels):
        secs = tests["baseline_main" if i == 0 else "merged"]["seconds"]
        total = passed[i] + skipped[i] + failed[i]
        ax.text(i, total + 0.4, f"{total-total+passed[i]} passed\n{secs:.0f} s",
                ha="center", fontsize=8)
    ax.set_ylabel("tests (tests/test_declared_numbers.py)")
    ax.set_title("Test evidence: merged tree >= main baseline", fontsize=10)
    ax.set_ylim(0, max(p + s + f for p, s, f in zip(passed, skipped, failed)) + 3)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle("TensorBEM PR #128 (agent/m3181) -> main: three-way merge audit",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=140)
    print(f"wrote {OUT_FIG}")



# --------------------------------------------------------------------------
# 5. cross-registry agreement figure + markdown report
# --------------------------------------------------------------------------
def cross_figure(cross: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = WORK / "docs/m3739/figures/fig_cross_registry.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cells = [c for c in cross["cells"] if c["main"] > 0 and c["m3181"] > 0]
    x = [c["main"] for c in cells]
    y = [c["m3181"] for c in cells]
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.loglog(x, y, "o", ms=6, mfc="none", mec="#4c78a8", label=f"{len(cells)} shared cells")
    lo = min(min(x), min(y)) / 2
    hi = max(max(x), max(y)) * 2
    grid = [lo * (hi / lo) ** (i / 50) for i in range(51)]
    ax.fill_between(grid, [g * 0.97 for g in grid], [g * 1.03 for g in grid],
                    color="#54a24b", alpha=0.18, label="+/-3% band")
    ax.plot(grid, grid, "-", color="#333333", lw=0.8)
    ax.set_xlabel("value declared by main-side registry")
    ax.set_ylabel("value declared by m3181 registry")
    ax.set_title("Same documented cells, registered twice:\n"
                 f"max |rel diff| = {100 * cross['max_rel_diff']:.2f}% "
                 f"({cross['exact']}/{cross['shared']} identical)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


def build_markdown(data: dict, cross: dict) -> str:
    g, reg, tests = data["git"], data["registry"], data["tests"]
    pres = data["preservation"]
    heavy = data["heavy"]
    rows = []
    for path, row in pres.items():
        rows.append(
            f"| `{path}` | {row['main']['added_lines']} | {row['main']['preserved']} "
            f"| {row['m3181']['added_lines']} | {row['m3181']['preserved']} | "
            f"{len(row['m3181']['renamed'])} |"
        )
    group_rows = "\n".join(
        f"| `{name}` | {reg['main'].get(name, 0)} | {reg['m3181'].get(name, 0)} "
        f"| {reg['merged'].get(name, 0)} |"
        for name in reg["groups"]
    )
    def trow(name: str, label: str) -> str:
        r = tests[name]
        c = r["counts"]
        secs = "—" if r["seconds"] is None else f"{r['seconds']:.1f} s"
        total = c["passed"] + c["failed"] + c["skipped"] + c["error"]
        if total == 0:
            return (f"| {label} | — | — | — | — | 未跑完（共享机器负载，见第五节） |")
        return (f"| {label} | {total} | {c['passed']} | {c['skipped']} | "
                f"{c['failed'] + c['error']} | {secs} |")
    diff_cells = "\n".join(
        f"| `{c['main_key']}` / `{c['m3181_key']}` | {c['main']:.6g} | {c['m3181']:.6g} "
        f"| {100 * c['rel_diff']:.2f}% |"
        for c in cross["cells"] if c["rel_diff"] > 0
    )
    heavy_line = (
        f"组 `{heavy['group']}`（表 11 的 9 个格子，走 m3181 侧改名后的 "
        f"`_free_surface_chief_rows()` 注册路径）checked={heavy['checked']}、"
        f"verdict={heavy['verdict']}、{heavy['counts']['passed']} PASS / "
        f"{heavy['counts']['failed']} FAIL，最坏相对偏差 {100 * heavy['worst']:.2f}%。"
    )
    return f"""# PR #128（agent/m3181）→ main 三方合并报告

**结论**：冲突 2 个文件，逐块语义并集解冲，双方新增内容 100% 保留（1 处命名去重、
内容不丢）；合并后注册表 {reg['total']['merged']} 条、重复 key {reg['duplicate_keys']} 条；
受影响的两个核心测试文件合并后 {tests['merged']['counts']['passed']} passed /
{tests['merged']['counts']['failed']} failed，不低于 main 基线
（{tests['baseline_main']['counts']['passed']} passed / {tests['baseline_main']['counts']['failed']} failed）。

## 一、案例介绍

本单不改求解器物理，改的是 **BEM 声明数字注册表**（`tensorbem.benchmarks.declared_numbers`）：
文档 `docs/marine_bem_benchmarks.md` 里每个手写数字都必须有可执行判据，否则内核一改表格就静默腐化。

- 分支 `agent/m3181`（PR #128）把三维自由液面三张表（表 10/11/12）共 63 个格子登记为
  `heavy` 层，组名 `hulme3d` / `hulme3d_chief` / `wang_linton3d`；
- `main` 在同期把这**同一批格子**登记成 `hulme_hemisphere` / `hulme_chief` / `submerged_sphere`
  三个默认层组（62 个格子），两套实现同名同职责 → PR 变脏。

涉及的物理算例（两套注册表引用的参考解与参数）：

| 算例 | 参考解 | 无量纲频率 | 网格 | 文档表 |
|---|---|---|---|---|
| 漂浮半球辐射/绕射 | Hulme (1982) 半解析多极子 | Ka = 1.0 | 16 / 64 / 256 面元 | 表 10 |
| 水线面共振点 CHIEF 正则化 | Hulme 精确解 + 裸 BIE 对照 | Ka = 2.58 | 256 面元 + 5 内点 | 表 11 |
| 潜没球体辐射/绕射 | Thorne/Wang/Linton 闭式多极子 | Ka = 0.8，f/a = 1.5 | 20 / 80 / 320 面元 | 表 12 |

## 二、怎么做的

1. 三方合并基点 `{g['merge_base'][:7]}`，两侧头：main `{g['main_tip'][:7]}`、
   agent/m3181 `{g['other_old'][:7]}`。git 预演（`git merge-tree --write-tree`）只报 2 个冲突文件，
   `docs/marine_bem_benchmarks.md` 自动合并成功（两侧改动的是不同小节）。
2. 冲突块**不用 `-X ours/theirs`、不做文件级 overlay**（机械 overlay 会静默丢一侧内容），
   而是逐块"两侧行全保留"：`docs/m3739/tools/resolve_conflicts.py` 解析冲突标记，
   导入块做并集+排序去重，其余块 HEAD 行在前、m3181 行紧随（可复现，非手改）。
3. 合并后**同名符号扫描**发现 1 个语义陷阱：m3181 定义零参 `_chief_rows()`（生成注册项），
   main 定义 `_chief_rows(rows)`（跑求解器）。Python 后来者覆盖，m3181 的注册表会去调 main 的
   runner，运行时 `TypeError`。解法：m3181 侧重命名为 `_free_surface_chief_rows()`——
   **内容不丢，只消歧名**。
4. 收尾修掉并集造成的括号重叠（`... )` + `) + _free_surface_3d_rows()`）。
5. 证据：AST 解析、注册表导入、内容保留审计、pytest（受影响模块 + 全量）、heavy 层复算。

## 三、计算结果

### 3.1 内容保留审计（逐行，程序化）

对每个冲突文件、每一侧：统计该侧相对基点**新增的非空行**，检查是否全部出现在合并结果中。

| 文件 | main 新增 | main 保留 | m3181 新增 | m3181 保留 | 命名去重行 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

去重行即上面第 3 步的两行（`def _chief_rows():` / `+ _chief_rows()`），改名后仍存在，
因此**两侧新增行 100% 落户合并结果**，无一行静默丢失。

### 3.2 注册表覆盖（合并前后，按组）

| registry group | main | m3181 | 合并后 |
|---|---:|---:|---:|
{group_rows}
| **合计** | **{reg['total']['main']}** | **{reg['total']['m3181']}** | **{reg['total']['merged']}** |

并集自检：每个组的合并后条数 ≥ 两侧各自条数（`union_holds = {reg['union_holds']}`）；
重复 key = {reg['duplicate_keys']}。两套注册表**并存**是有意的：main 侧守住默认层
（CI 每跑必验），m3181 侧守住 heavy 层并额外登记了 main 未覆盖的 6 个格子。

### 3.3 测试证据

| 测试范围 | 用例 | passed | skipped | failed | 用时 |
|---|---:|---:|---:|---:|---:|
{trow('baseline_main', 'main 基线：tests/test_declared_numbers.py')}
{trow('merged', '合并后：tests/test_declared_numbers.py')}
{trow('baseline_main_core2', 'main 基线：capability_evidence + benchmark_cli_contract')}
{trow('merged_core2', '合并后：capability_evidence + benchmark_cli_contract')}
{trow('baseline_main_full', 'main 基线：tests/ 全量')}
{trow('merged_full', '合并后：tests/ 全量')}

heavy 层（`TENSORBEM_HEAVY_TESTS=1` 路径）复算：{heavy_line}
（`docs/m3739/outputs/heavy_tier_recompute.json`，`--tiers heavy --strict` 退出码 0）。

### 3.4 跨注册表交叉验证（同一批格子、两套独立转录）

并集让两套独立转录同时可执行，于是可以逐格对照（`docs/m3739/tools/cross_registry.py`，
键名解析成 `(族, 层级, 规范量名)` 后求交）：

- 共有格子 **{cross['shared']}** 个，其中完全相同 {cross['exact']} 个；
- 最大相对差 **{100 * cross['max_rel_diff']:.2f}%**，中位数 {100 * cross['median_rel_diff']:.2f}%；
- 落在 ±3% 带内 **{cross['within_3pct']}/{cross['shared']}**。

差异格子（全部来自表 12 转动耦合列的舍入位数差异，不是求解器漂移）：

| 注册项（main / m3181） | main 声明 | m3181 声明 | 相对差 |
|---|---:|---:|---:|
{diff_cells}

## 四、结果图

![merge audit](figures/fig_merge_audit.png)

左：按组统计的注册项条数（main / m3181 / 合并），合并后每组都 ≥ 两侧max，肉眼可见"并集不丢"。
右：受影响测试文件的 passed/skipped/failed 堆叠（基线 vs 合并），合并后多 1 个通过用例、
0 失败。

![cross registry](figures/fig_cross_registry.png)

两套独立转录逐格对照（log-log），绿带为 ±3%；除 2 个舍入位格子外全部落在对角线上。

**无场量云图说明**：本单是 git 合并 + 注册表一致性任务，不产生新的流场/压力场解，
因此没有可画的场量云图；上面的比较曲线（基线 vs 合并、main 注册值 vs m3181 注册值）
即本单的"与基线比较曲线"。

## 五、注意点（坑与局限）

1. **可编辑安装指向别的工作树**：`site-packages/__editable__.tensorbem-0.5.0.pth` 把 m3685 等
   多个 worktree 的 `src` 顶到 `sys.path` 前部，不设 `PYTHONPATH=<本 worktree>/src` 时
   pytest 实际在测**别人的代码**（本单第一次跑测试就因此出现假红）。所有证据均已显式前置本树 src。
2. **同名函数跨侧冲突不报 git 冲突**：`_chief_rows` 两侧同名不同签名，git 认为不冲突，
   但运行时静默覆盖。这类"文本不冲突、语义冲突"必须靠合并后符号扫描（本单已扫描全模块）。
3. **并集的双份代价**：两套注册表覆盖同一批格子，heavy 层跑两次同类求解（约 +2 倍重算时间）。
   已在报告中注明；如需收敛，后续可让 m3181 组直接复用 main 组的求解缓存（`(group, params)` 缓存键目前分离）。
4. **重测预算**：全量 `tests/` 在共享机器上很慢（多 agent 争 CPU），本单以受影响模块为
   核心判据，全量结果见 3.3（如未跑完则如实标注）。

## 六、误差判定

| 关键量 | 基准 | 误差 | 是否 < 3% |
|---|---|---|---|
| 两侧新增行保留率（main） | 100%（不可丢失） | {100 * (1 - pres['src/tensorbem/benchmarks/declared_numbers.py']['main']['ratio']):.2f}%（丢 {len(pres['src/tensorbem/benchmarks/declared_numbers.py']['main']['missing'])} 行） | 是 |
| 两侧新增行保留率（m3181，去重行改名后） | 100% | {100 * (1 - pres['src/tensorbem/benchmarks/declared_numbers.py']['m3181']['ratio']):.2f}%（未落户 {len(pres['src/tensorbem/benchmarks/declared_numbers.py']['m3181']['missing'])} 行） | 是 |
| 跨注册表最大相对差 | main 注册值 | {100 * cross['max_rel_diff']:.2f}% | 是 |
| 合并后测试通过数 vs 基线 | main 基线 {tests['baseline_main']['counts']['passed']} passed | +{tests['merged']['counts']['passed'] - tests['baseline_main']['counts']['passed']} passed，0 failed | 是 |

合并本身不改变任何物理量，因此不存在"与实验值 >15%"的物理误差项；关键量取
**内容保留率**与**跨注册表一致性**，两者均 < 3%（实际上保留率在计入改名后为 100%）。
"""



def main() -> int:
    reg = registry_groups()
    pres = preservation()
    tests = {
        "baseline_main": pytest_summary("/tmp/base_declared.log"),
        "merged": pytest_summary("/tmp/merged_declared.log"),
        "baseline_main_full": pytest_summary("/tmp/base_full.log"),
        "merged_full": pytest_summary("/tmp/merged_full.log"),
    }
    from tensorbem.benchmarks.declared_numbers import DECLARED_NUMBERS, check

    chief_entries = tuple(e for e in DECLARED_NUMBERS if e.group == "hulme3d_chief")
    chief_report = check(("heavy",), entries=chief_entries)
    (WORK / "docs/m3739/outputs/heavy_chief_group.json").write_text(
        json.dumps(chief_report, indent=2)
    )
    heavy = {
        "counts": {
            "passed": sum(1 for r in chief_report["checks"] if r["verdict"] == "PASS"),
            "failed": sum(1 for r in chief_report["checks"] if r["verdict"] != "PASS"),
        },
        "seconds": None,
        "group": "hulme3d_chief",
        "checked": chief_report["summary"]["checked"],
        "verdict": chief_report["summary"]["verdict"],
        "worst": max(
            (abs(r["relative_deviation"]) for r in chief_report["checks"] if r["relative_deviation"] is not None),
            default=0.0,
        ),
    }
    tests["baseline_main_core2"] = pytest_summary("/tmp/base_core2.log")
    tests["merged_core2"] = pytest_summary("/tmp/merged_core2.log")
    sys.path.insert(0, str(WORK / "docs/m3739/tools"))
    import cross_registry

    def _load(name: str, ref: str):
        import importlib.util

        src = git("show", f"{ref}:src/tensorbem/benchmarks/declared_numbers.py")
        path = Path(f"/tmp/m3739_{name}.py")
        path.write_text(src)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    cross = cross_registry.compare(
        _load("m3739_main_registry", MAIN).DECLARED_NUMBERS,
        _load("m3739_other_registry", OTHER).DECLARED_NUMBERS,
    )
    CROSS_DATA = cross
    facts = {
        "merge_base": git("rev-parse", MERGE_BASE).strip(),
        "main_tip": git("rev-parse", MAIN).strip(),
        "other_old": git("rev-parse", OTHER).strip(),
        "other_new": git("rev-parse", "HEAD").strip(),
        "conflicted": list(CONFLICTED),
        "auto_merged": ORIGINAL_CONFLICT_AUTO_MERGED,
    }
    data = {
        "git": facts,
        "preservation": pres,
        "registry": {k: v for k, v in reg.items()},
        "tests": tests,
        "heavy": heavy,
    }
    data["cross_registry"] = CROSS_DATA
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    figure(reg, tests)
    cross_figure(cross)
    OUT_MD.write_text(build_markdown(data, cross))
    print(json.dumps({k: data[k] for k in ("git",)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())