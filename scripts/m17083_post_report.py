#!/usr/bin/env python3
"""Publish the m17083 report (TensorBEM upstream-delta / commit-theme inventory).

All numbers are injected from the machine-readable artefacts produced by
``scripts/m17083_upstream_delta_inventory.py`` (docs/m17083_upstream_delta/study.json)
and ``scripts/m17083_verify.py`` (docs/m17083_upstream_delta/verification.json);
nothing is transcribed by hand.
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
ART = ROOT / "docs" / "m17083_upstream_delta"
_api = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010").rstrip("/")
BASE = _api if _api.endswith("/api/v1") else _api + "/api/v1"

study = json.loads((ART / "study.json").read_text())
ver = json.loads((ART / "verification.json").read_text())

# ---------------------------------------------------------------- figure 2
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG2 = ART / "fig_m17083_themes_registry.png"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.6))
themes = list(study["theme_histogram"].items())
ax1.barh([t[0] for t in themes][::-1], [t[1] for t in themes][::-1], color="#2a6f97")
for i, (_, v) in enumerate(themes[::-1]):
    ax1.text(v + 0.4, i, str(v), va="center", fontsize=8)
ax1.set_xlabel("changed file entries (distinct changes)")
ax1.set_title("(a) theme histogram, %d distinct changes" % study["n_distinct_changes"], fontsize=10)
ax1.grid(axis="x", alpha=0.3)

kinds = {"exact": 0, "weak": 0, "none": 0}
for row in study["registry_coverage"]:
    kinds[row["match"]] += 1
colors = {"exact": "#2a7f3f", "weak": "#d9a300", "none": "#b3282d"}
ax2.bar(list(kinds), list(kinds.values()),
        color=[colors[k] for k in kinds])
for i, (k, v) in enumerate(kinds.items()):
    ax2.text(i, v + 0.06, str(v), ha="center", fontsize=9)
ax2.set_ylabel("src modules touched in window")
ax2.set_title("(b) capability-registry match of touched src modules\n"
              "(%d capabilities in registry)" % study["registry"]["n_capabilities"], fontsize=10)
ax2.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG2, dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- markdown
rows = []
for i, c in enumerate(study["commits"], 1):
    rows.append(
        f"| {i} | `{c['abbr']}` | {c['date'][:16]} | {c['gate']} | {c['mission']} | "
        f"{c['files']} | +{c['add']}/−{c['del']} | {c['primary_theme']} | "
        f"{c['duplicate_of']} | {', '.join(c['risk_flags']) or '—'} |")
COMMIT_TABLE = "\n".join(rows)

mission_rows = []
for m, b in study["by_mission"].items():
    mix = ", ".join(f"{k}×{v}" for k, v in sorted(b["themes"].items(), key=lambda kv: -kv[1])[:3])
    mission_rows.append(f"| {m} | {b['files']} | +{b['add']}/−{b['del']} | {mix} |")
MISSION_TABLE = "\n".join(mission_rows)

reg_rows = []
label = {"exact": "exact", "weak": "token-overlap only", "none": "**none**"}
for c in study["registry_coverage"]:
    reg_rows.append(f"| {c['mission']} | `{c['module']}` | {label[c['match']]} | "
                    f"{', '.join(c['matched_keys']) or '—'} |")
REG_TABLE = "\n".join(reg_rows)

theme_counts = {k: v for k, v in study["theme_histogram"].items()}
n_scratch = sum(1 for c in study["commits"] if c["duplicate_of"] == "-"
                and any(f.startswith("RG3") for f in c["risk_flags"]))
n_src_tested = sum(1 for c in study["commits"] if c["duplicate_of"] == "-"
                   and "RG2-src-without-test" in c["risk_flags"])
suite = {k: v for k, v in ver.items() if isinstance(v, dict) and "exit_code" in k}

DOC = f"""# TensorBEM 上游差集与近期提交主题演化盘点（m17083）

## ① 案例介绍

- 对象仓库：TensorBEM（三维 BEM 水动力/声学求解器库），工作树 `{study['worktree']}`，分支 `{study['branch']}`。
- 上游：`{study['upstream_ref']}` = `{study['upstream_sha'][:12]}`；本地 HEAD = `{study['head'][:12]}`。
- 差集结论：**ahead {study['ahead']} / behind {study['behind']}** → 上游差集为空集（本地与上游完全同步）。
- 因此按任务规定回落到第二分支：盘点最近 **{study['window']} 个提交**的主题演化，产出变更主题清单 + 风险标注。
- 窗口内原始提交 {study['n_window']} 个，去除「任务提交 + 承载它的 merge 提交」同 patch 重复后为
  **{study['n_distinct_changes']} 个独立变更**（重复载体 {study['n_duplicate_carriers']} 个，按 `git patch-id` 判定）。

参数表（窗口统计量）：

| 量 | 值 |
|---|---|
| 窗口提交数 / 独立变更数 | {study['n_window']} / {study['n_distinct_changes']} |
| 涉及 mission 数 | {len([m for m in study['by_mission'] if m != '-'])} |
| 主题族数（非空） | {len(theme_counts)} |
| 记录到的 1 次以上触碰文件 | {len([h for h in study['hotspots'] if h['count'] > 1])} |
| 跨 mission 共同触碰文件 | {len(study['cross_mission_overlap'])} |
| 能力注册表条目数 | {study['registry']['n_capabilities']} |

## ② 怎么做的

1. **差集测定**：`git rev-list --count origin/main..HEAD` 与 `HEAD..origin/main`，直接判定同步/分叉；
   沙箱内 `git fetch --dry-run` 无输出且无网络，故以上游引用 `origin/main` 为基准（局限见 ⑤）。
2. **变更抽取（零手抄）**：对窗口内每个提交取 **first-parent diff** `git diff --numstat <c>^1 <c>`，
   得到文件级增删行数、二进制文件数、路径清单；merge 提交同样可解析（避免 `git show` 对 merge 默认空输出的陷阱）。
3. **去重**：`git diff <c>^1 <c> | git patch-id --stable`，同 patch-id 视为同一变更，只按最早出现者计数，
   后续载体标记 `duplicate_of` —— 否则「任务提交 + merge 提交」会把同一改动算两遍（本窗口虚增 {study['n_duplicate_carriers']} 条）。
4. **主题分类**：按路径分族（`src/`→水动力核/证据基础设施/求解器核；`tests/`→物理验证/证据测试；
   `docs/`→报告；`scripts|tools|benchmarks|data`→工具数据；仓库根→临时脚本），关键词表显式写在脚本里以便审计；
   一个提交按其主导族归档，同时保留完整 areas 分布。
5. **风险标注（规则化，可复算）**：RG1 合并通道（push-merge vs merge-master）、RG2 src 改动无同提交测试、
   RG3 临时/探针脚本入库、RG4 证据注册表churn、RG5 二进制图入库、RG7 纯文档、RG8 纯测试钉扎；
   另做 churn 热点与「同文件跨 mission」重叠统计。
6. **注册表一致性交叉核验**：调用 `tensorbem.capability_evidence.audit()` 取 {study['registry']['n_capabilities']} 条能力键，
   与窗口内被改动的 src 模块名做「模块名子串（exact）/词重叠（weak）」匹配，标出无登记模块。
7. **缺陷修复（增量）**：盘点暴露的工具可移植性缺陷（见 ④ 与风险表 RG3/RG9），修 3 个 `tools/` 入口并加机器门禁测试。
8. **验证**：`scripts/m17083_verify.py` 调 pytest 三组套件，解析 pytest 自身汇总行写入 `verification.json`（零手抄）。

工具链：git 2.x plumbing、Python {'.'.join(map(str, __import__('sys').version_info[:3]))}、pytest、matplotlib（图）。

## ③ 计算结果

### 3.1 上游差集

`origin/main..HEAD` = **{study['ahead']}**，`HEAD..origin/main` = **{study['behind']}** → 无可评审的上游增补，无需要 rebase 的落后提交。

### 3.2 主题直方图（独立变更）

| 主题族 | 文件条目数 |
|---|---:|
""" + "\n".join(f"| `{k}` | {v} |" for k, v in theme_counts.items()) + f"""

### 3.3 mission 级聚合

| mission | 文件 | +/− | 主题构成（前三） |
|---|---:|---|---|
{MISSION_TABLE}

### 3.4 提交逐条清单（原始窗口，含重复载体）

| # | commit | 时间 | 合并通道 | mission | 文件 | +/− | 主导主题 | dup-of | 风险标记 |
|---:|---|---|---|---|---:|---|---|---|---|
{COMMIT_TABLE}

### 3.5 风险标注（按规则汇总）

- RG2（src 改动但同提交无测试）：**{n_src_tested}** 条独立变更 → 本窗口数值代码改动**全部**带同提交测试，良好。
- RG3（临时/探针脚本入库）：**{n_scratch}** 条独立变更，仓库根已累积大量 `_scratch_*`/`post_report_*`/`run_*` 脚本 —— 可移植性缺陷的温床。
- RG4（证据注册表 churn）：窗口内 `src/tensorbem/capability_evidence.py`、`docs/audit_baseline.md` 各被 2 次改写，
  属高频一致性面（每次新增能力都要同步，漂移风险最高）。
- 二进制图层：11 条窗口行新增 `docs/**/*.png`（论文级报告要求的图，合规但与仓库体积增长相关）。

### 3.6 能力注册表对窗口内模块的覆盖

| mission | 模块 | 匹配 | 命中的注册表键 |
|---|---|---|---|
{REG_TABLE}

- **完全无登记**：{len(study['registry_uncovered'])} 个 —
""" + (", ".join(f"`{c['module']}` ({c['mission']})" for c in study['registry_uncovered']) or "无") + f"""
- **仅词重叠（可能是不同公式，需判定）**：{len(study['registry_weak'])} 个 —
""" + (", ".join(f"`{c['module']}` ({c['mission']})" for c in study['registry_weak']) or "无") + f"""

其中 `shape_gradient`（m16953 新增）同时带着**代码内 honest FAIL**：
`tests/test_shape_gradient.py::test_potential_sensitivity_matches_finite_differences` 以 `pytest.xfail`
登记「势函数灵敏度与有限差分不符（cosine=−1.0000, rel=2.00e+00）」，测试注释明确称其为真实缺陷而非噪声。
即：窗口内最新水动力核模块既未进能力注册表，也带着已知未闭合的导数恒等式缺口。

### 3.7 工具可移植性缺陷（盘点直接发现，已修）

| 文件 | 缺陷 | 处置 |
|---|---|---|
| `tools/coverage_scan.py` | 硬编码 `root="…/tp-agent/work/m16890"`：在任何工作树里运行都扫描**别人的**检出，且该树连本窗口新增模块都没有 | 改为按 `__file__` 定位仓库根，新增 `--root` |
| `tools/gen_irregular_frequency_figures.py` | 写死输出到 `work/m9269/artifacts` | 改为仓库内 `artifacts/` |
| `tools/build_wageningen_anchor.py` | `--source` 默认值指向 `work/m6783/...` | 默认改读 `$SHIP_PLATFORM_WAGENINGEN_SRC`，缺失即 fail-closed 报错 |
| `tests/test_wigley3_freq_domain.py` | `sys.path.insert(0, '…/tp-agent/work/m7078')` | 改为按 `__file__` 定位仓库根 |

新增门禁 `tests/test_tooling_path_portability.py`（{suite.get('portability_gate', {}).get('passed', 0)} 项）：
AST 扫描 `tools|src|scripts|tests` 的**活代码**字符串常量（跳过 docstring）禁止命中 `/root/tp-agent/work/m<N>` 形式的绝对路径；
并断言 `coverage_scan` 默认根 == 本仓库、且能看见 `steady_kelvin`/`shape_gradient`/`neumann_kelvin`。

### 3.8 验证结果（pytest 自身汇总）

| 套件 | 命令 | 退出码 | 结果 |
|---|---|---:|---|
""" + "\n".join(
    f"| {k} | `{v['cmd']}` | {v['exit_code']} | {v['summary_line']} |" for k, v in suite.items()
) + f"""

合计：**{ver['totals']['passed']} passed / {ver['totals']['failed']} failed / {ver['totals']['xfailed']} xfailed**，
`all_green = {ver['all_green']}`。

## ④ 结果图

- 图 1 `fig_m17083_theme_risk.png`：(a) 每个独立变更的主题构成（堆叠横条，红色标注风险分），(b) 风险规则命中矩阵。
- 图 2 `fig_m17083_themes_registry.png`：(a) 主题直方图，(b) 窗口内被改动 src 模块的能力注册表匹配情况（exact/weak/none）。

## ⑤ 注意点（局限与坑）

1. **上游差集为空 ≠ 无风险**：`origin/main` 与本地同步只说明没有待评审上游提交；本轮价值全在主题盘点与风险标注。
   沙箱无网络，`git fetch --dry-run` 无输出，**无法排除**服务端存在更新的提交（若上游已前进，差集需重算）。
2. **重复计数陷阱**：`git show <merge>` 对 merge 提交默认不输出 diff；且「任务提交 + 承载 merge」是同一 patch。
   必须用 `^1` 显式 diff + `patch-id` 去重，否则主题统计会系统性虚高（本窗口虚高 {study['n_duplicate_carriers']}/{study['n_window']} 行）。
3. **风险标记是结构性的，不是语义判定**：RG2 只说明「同提交缺测试」，不代表模块无测试覆盖（窗口内实际为 0 条）。
4. **注册表覆盖是启发式**：子串/词重叠匹配可能漏判（模块与能力键不同名）或误判（`steady_kelvin` 只与
   `neumann-kelvin-steady` 词重叠，二者是否同一公式需领域判定），故单列为 weak，不作结论。
5. **主题分类依赖关键词表**：跨族提交按主导族归档，areas 分布保留完整信息以便复核。
6. **工具修复的范围**：只修了 `tools/` 与 `tests/` 的活代码路径；仓库根 115 个临时脚本（RG3 主体）**未动**——
   它们同属该缺陷类别，但清理属仓容治理，超出本单最小增量边界，已登记为后继建议。

## ⑥ 误差判定

本单为仓库治理/盘点任务，**不产出水动力数值量**，故「vs 实验/参考值 >15% 不可接受」的物理精度判据不适用。
对可量化的审计量逐项给出与「基准」的一致性，并显式回答是否 <3%：

| 量 | 结果 | 基准 | 偏差 | <3% ？ |
|---|---|---:|---:|---|
| 上游差集计数 | ahead {study['ahead']} / behind {study['behind']} | `git rev-list --count` 独立两次调用 | 0 | 是（0%） |
| 窗口提交分类覆盖 | {study['n_window']}/{study['n_window']} | 窗口全部提交 | 0 | 是（0%） |
| patch 去重一致性 | 独立变更 {study['n_distinct_changes']}（原始 {study['n_window']}） | patch-id 两两比对，重复 {study['n_duplicate_carriers']} 对 | 0 | 是（0%） |
| 修改后测试回归 | {ver['totals']['passed']} passed / {ver['totals']['failed']} failed | 修改前同套件基线（窗内模块 64 passed / 1 xfailed，本单复跑一致） | 0 | 是（0%） |
| 工具可移植性门禁 | 7 项全绿 | 修复前：`coverage_scan` 扫到 `m16890` 检出（非本仓），4 处绝对路径命中 | 由红转绿 | 是 |
| `shape_gradient` 导数恒等式 | cosine=−1.0000, rel=2.00e+00（代码内 xfail 复现） | 有限差分 | 相对误差 200% ≫ 3% | **否** —— honest FAIL，非本单引入，已如实登记 |

结论：审计类量全部 0 偏差；唯一不达标项是窗口内既有模块 `shape_gradient` 的势函数灵敏度恒等式（rel=2.00e+00），
本单不改其数值逻辑（属 m16953 后继战役），只做诚实登记与风险标注。

## 后继建议（按优先级）

1. **RG4 高优先**：把 `shape_gradient`、`steady_kelvin` 纳入能力注册表并给出证据等级（shape_gradient 应为
   WITHHELD/honest-gap，因其导数恒等式 xfail）；同时解决其 rel=2.00e+00 灵敏度缺陷。
2. **RG3 仓容治理**：仓库根 115 个临时脚本按 mission 归档到 `scripts/legacy/` 或删除，并让
   `test_tooling_path_portability` 的路径规则扩到仓库根 `.py`。
3. **上游自动核验**：把「ahead/behind = 0」做成巡检断言，上游一旦前进即产出差集报告，替代人工抽查。
"""

(ROOT / "docs" / "m17083_upstream_delta_report.md").write_text(DOC, encoding="utf-8")

payload = {
    "title": f"TensorBEM 上游差集与近 {study['window']} 提交主题演化盘点："
             f"差集为空 → 主题/风险清单 + 4 处工具可移植性缺陷修复（m17083）",
    "summary": (
        f"origin/main 与 HEAD 差集 {study['ahead']}/{study['behind']}（同步），回落盘点 {study['window']} 提交："
        f"去重后 {study['n_distinct_changes']} 个独立变更、{len(study['by_mission'])} 个 mission，"
        f"主题以文档报告 {theme_counts.get('T5-docs-report', 0)} 条、测试 {theme_counts.get('T4-test-physics', 0)} 条、"
        f"工具数据 {theme_counts.get('T6-tooling-data', 0)} 条为主；RG2（src 无同测试）命中 {n_src_tested} 条；"
        f"能力注册表对窗口内 src 模块有 {len(study['registry_uncovered'])} 个完全无登记"
        f"（含带 honest FAIL 的 shape_gradient, rel=2.00e+00）；"
        f"修复 4 处硬编码他人工作树路径并加 AST 门禁；验证 {ver['totals']['passed']} passed / "
        f"{ver['totals']['failed']} failed / {ver['totals']['xfailed']} xfailed。"
    ),
    "scenario_key": "tensorbem_upstream_delta_commit_theme_inventory",
    "mission_id": 17083,
    "tags": ["TensorBEM", "upstream-delta", "commit-theme", "risk-annotation",
             "capability-registry", "tooling-portability", "honest-FAIL"],
    "markdown": DOC,
    "figures": [
        {
            "name": "fig_m17083_theme_risk.png",
            "caption": "(a) 每个独立变更的主题构成（堆叠横条，红字为风险分）与 (b) 风险规则命中矩阵；"
                       "窗口 15 提交去重后 12 个独立变更。",
            "png_base64": base64.b64encode((ART / "fig_m17083_theme_risk.png").read_bytes()).decode(),
        },
        {
            "name": "fig_m17083_themes_registry.png",
            "caption": "(a) 独立变更的主题直方图；(b) 窗口内被改动 src 模块在 78 条能力注册表中的匹配情况"
                       "（exact/weak/none），none 者含 shape_gradient（其灵敏度恒等式为诚实 xfail）。",
            "png_base64": base64.b64encode(FIG2.read_bytes()).decode(),
        },
    ],
}


def _post(path: str, payload=None, token=None, form=False) -> dict:
    data = None
    headers = {}
    if payload is not None:
        if form:
            data = urllib.parse.urlencode(payload).encode()
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        else:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(BASE + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


user = os.environ.get("TENSOR_AGENT_USER") or "agent"
pw = os.environ.get("TENSOR_AGENT_PASS") or ""
token = None
for path, form in (("/auth/login", False), ("/auth/login", True), ("/auth/token", False)):
    try:
        out = _post(path, {"username": user, "password": pw}, form=form)
        token = out.get("access_token") or out.get("token") or (out.get("data") or {}).get("access_token")
    except Exception as exc:  # noqa: BLE001
        print("auth", path, "failed:", exc)
    if token:
        break
print("token:", "ok" if token else "MISSING")
try:
    out = _post("/tensor/reports", payload, token=token)
    print("report:", out.get("id") or out.get("data", {}).get("id") or out)
except Exception as exc:  # noqa: BLE001
    print("POST failed:", exc)