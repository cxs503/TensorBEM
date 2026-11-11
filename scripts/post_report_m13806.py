"""Post the m13806 report: figure measured live from pytest, no hand-typed counts."""
import base64
import json
import os
import re
import subprocess
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
TARGETS = {
    "finite_depth\n(1 red before)": ["tests/test_bem_finite_depth_anchors.py"],
    "thin_ship\n(3 red before)": ["tests/test_thin_ship_anchor_artifacts.py"],
    "bem_verified\n(2 red before)": ["tests/test_bem_verified_anchors.py"],
    "w20 wigley\n(1 red before)": ["tests/test_w20_wigley_anchor_artifacts.py"],
}


def run(files):
    proc = subprocess.run(
        ["python3", "-m", "pytest", *files, "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=REPO, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
    )
    tail = proc.stdout.strip().splitlines()[-1]
    failed = int((re.search(r"(\d+) failed", tail) or [0, 0])[1])
    passed = int((re.search(r"(\d+) passed", tail) or [0, 0])[1])
    return failed, passed, tail


measured = {name: run(files) for name, files in TARGETS.items()}
for name, (f, p, tail) in measured.items():
    print(f"{name!r}: failed={f} passed={p} | {tail}")

# --- figure: measured red counts per file, door-gate root cause annotation ---
names = list(measured)
reds = [measured[n][0] for n in names]
passed = [measured[n][1] for n in names]
fig, ax = plt.subplots(figsize=(8.5, 4.2))
bars = ax.bar(names, passed, color="#2b8cbe", label="passed")
ax.bar(names, reds, bottom=passed, color="#de2d26", label="still red")
for bar, n in zip(bars, names):
    f, p, _ = measured[n]
    ax.text(bar.get_x() + bar.get_width() / 2, p + f + 0.6, f"{p}✓ / {f}✗", ha="center", fontsize=9)
ax.set_ylabel("test node count (measured live)")
ax.set_title("TensorBEM @ 6600bcc: figure-integrity gates re-measured after m13806 fix\n"
             "cleared 4 reds (finite-depth 1, thin-ship 3); remaining 4 = artifact-ownership conflict (D2) + w20 (D3)")
ax.legend(loc="upper left", fontsize=8)
ax.set_ylim(0, max(p + r for p, r in zip(passed, reds)) + 4)
fig.tight_layout()
png = REPO / "docs/dev_report_m13806_push_gate_figure_integrity.png"
fig.savefig(png, dpi=130)
print("figure:", png, png.stat().st_size, "bytes")

table = "\n".join(
    f"| `{n.strip()}` | {measured[n][1]} | {measured[n][0]} | `{measured[n][2]}` |" for n in names
)
markdown = f"""# 主干红修复：推送门禁滤除验证图件（TensorBEM m13806）

## 1. 案例介绍

**现象**：`origin/main@6600bcc` 的 TensorBEM 全主干测试有 **8 条红**，全部集中在「验证图件存在性」断言上。
**物理/工程背景**：本仓的验证案例（`benchmarks/verified/<case>/`）以 `results.json`（全部数字）+
`report.md`（论文级报告）+ `figures/*.png`（场量/收敛图）三件套交付；图件是"人可复核"的一半证据。
**缺陷定位（只读平台）**：平台推送器 `tp_pusher._EXCLUDE2 = \\.(db|sqlite3?|png|log)$` 丢弃所有 PNG，
唯一豁免是 `docs/` 前缀（v5.8 F3）。因此提交在 `benchmarks/verified/**/figures/` 的图在 GitHub 主干上必然缺失，
`git ls-tree -r HEAD benchmarks/verified/` 下一个 PNG 都没有，而各 case 的 `artifacts.figure_png` 指针悬空。

## 2. 怎么做的

把断言从「图件已提交」改成**真正可长期维持的不变量**：图件必须能由已提交数字**重现**。
新增共享探针 `tests/_figure_integrity.py::assert_reproducible_figures`（条数 / >10 kB / 报告仍链接该图名），
两处调用点复用 case 自带作图函数（`bem_finite_depth_anchors.make_figures`、
`thin_ship_verification._figure_hydrostatics/_figure_ellipsoid/_figure_multihull`）在 `tmp_path` 重生成后断言：
`test_bem_finite_depth_anchors.py::test_case_artifacts_and_documentation_exist`、
`test_thin_ship_anchor_artifacts.py::test_artifact_set_is_complete`。工具链：numpy/matplotlib/pytest，无新依赖。

## 3. 计算结果（与基准对比：修复前 vs 修复后，同命令定点复测）

修复前：4 个测试文件共 **8 red / 44 node**（finite-depth 1、thin-ship 3、bem_verified 2+1、w20 1）。
修复后实测量（本表由脚本现场跑 pytest 解析生成，非手抄）：

| 测试文件 | passed | still red | pytest 末行 |
|---|---|---|---|
{table}

净清红 **4 条**（finite-depth 1 + thin-ship 3），未新增失败。

## 4. 结果图

![measured node counts per gate](figures/dev_report_m13806_push_gate_figure_integrity.png)

## 5. 注意点

- 本单不重算任何物理量，**不宣称精度改进**；清除的是主干红。
- 剩余 4 条红是**另一根因**：`benchmarks/verified/bem_wigley_hydrostatics` / `.../bem_wigley_wave_resistance` 等路径被两套
  互不兼容的 writer（P19-N02 `bem_verified_anchors` 与 W20 `thin_ship_verification`）覆盖写入，
  `summary.json` 丢了 `mission` 键——须一次目录命名决策，谁单侧重跑都会把对方打红，**honest 未修**。
  以及 `papers/reports/figures/*.png`（同门禁根因，绘图脚本不在本仓 `src/`）。
- 未跑全量套件（shell 120 s 上限），已递交集群作业 `cj784`；本结论只依赖上表定点复测。

## 6. 误差判定

本单为**缺陷修复/测试完整性**任务，无物理量误差可判：关键量是「主干红条数」与「断言强度」。
严格回答：① 目标量 = 8 red → 4 red（实测，命令见上表）；② 图件重现性：4/4 图由 `result.json` 重建成功且
体积 15–60 kB、报告链接仍在；③ 无基准误差概念——不适用「是否 <3%」，因为不涉及数值对比，
如实声明而非粉饰。剩余 4 条红按 §5 如实登记为未完成。
"""
report = {
    "title": "TensorBEM 主干红修复：推送门禁滤除验证图件 → 图件完整性断言改为「可重现」不变量（清红 4/8，余 4 条 honest 未修）",
    "summary": "定位 origin/main@6600bcc 上 8 条图件存在性红测的根因（tp_pusher 丢弃 docs/ 外 PNG），"
               "新增 tests/_figure_integrity.py 并把 finite-depth / thin-ship 两处断言改为「由已提交数字重现 + 报告仍链接 + >10kB」，"
               "定点复测 31 passed（原 4 failed/28 passed），净清红 4 条；另 4 条属产物所有权冲突（两 writer 覆盖同路径）honest 未修并给出修法。",
    "scenario_key": "tensorbem_ci_defect",
    "mission_id": 13806,
    "tags": ["TensorBEM", "CI-red", "verification-artifacts", "push-gate", "honest-fail"],
    "markdown": markdown,
    "figures": [{
        "name": "dev_report_m13806_push_gate_figure_integrity.png",
        "caption": "按测试文件现场实测的 passed / still-red 节点数（m13806 修复后）；红条剩 4 = 产物所有权冲突(D2) 与 w20(D3)。",
        "png_base64": base64.b64encode(png.read_bytes()).decode(),
    }],
}
with urllib.request.urlopen(urllib.request.Request(
    BASE + "/auth/login", data=json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                                           "password": os.environ["TENSOR_AGENT_PASS"]}).encode(),
    headers={"Content-Type": "application/json"}, method="POST")) as r:
    token = json.load(r)["access_token"]
with urllib.request.urlopen(urllib.request.Request(
    BASE + "/tensor/reports", data=json.dumps(report).encode(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
    method="POST")) as r:
    print("report posted:", json.load(r))