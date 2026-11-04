"""Finalise m3638: docs, cached tests, figure, dev report, platform report.

Runs in one process to stay inside the sandbox wall clock:
1. patch docs/marine_bem_benchmarks.md (section 5, item 6 + new item 7);
2. patch tests/test_declared_numbers.py so the standard tier is recomputed
   once per session instead of once per test;
3. render the figure from outputs/m3638/group_checks.json;
4. write dev_report_m3638_free_surface3d_registry.md with injected numbers;
5. publish the platform report (/tmp/report.json -> POST /tensor/reports).
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from functools import lru_cache  # noqa: F401  (documented in the patch text)
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(".").resolve()
CHECKS = json.loads((ROOT / "outputs/m3638/group_checks.json").read_text())
ROWS = CHECKS["all_checks"]
SUMMARY = CHECKS["summary"]

# --------------------------------------------------------------------------
# 1. docs
# --------------------------------------------------------------------------
DOC = ROOT / "docs/marine_bem_benchmarks.md"
doc = DOC.read_text()

OLD = ("文中引用的固定数值（二维圆柱 16/64/128 面元的附加质量与相对误差、"
       "Ursell 64/128/256 面元行）")
NEW = ("文中引用的固定数值（二维圆柱 16/64/128 面元的附加质量与相对误差、"
       "Ursell 64/128/256 面元行，以及**三维深水自由液面三张表：Hulme 漂浮半球表 10、"
       "三维内部 CHIEF 表 11、Wang/Linton 潜没球体表 12**）")
assert doc.count(OLD) == 1, doc.count(OLD)
doc = doc.replace(OLD, NEW)

ITEM7 = """
7. **三维自由液面三张表并入同一注册表（m3638）。** 表 10/11/12 的 {n} 个数此前只有正文、没有机器判据：注册表在三维侧只到无界流椭球，Kelvin/Havelock 核、6-DOF 组装或 CHIEF 增广一旦改动，表 10–12 会静默过期。现登记为 `declared_numbers` 的 `hulme`、`hulme_chief`、`submerged_sphere` 三组（成本层 `standard`，三组合计复算 {elapsed:.0f} s），并做了两件事：

   * **把物理不变量也当声明数字**：近/远场 Kochin 阻尼相对差（{far:.3e}）、三维直接绕射解与边界 Haskind 互易积分的相对差（纵荡 {hx:.3e}、垂荡 {hz:.3e}）、观测阶（h 的 2.06/1.98/2.01/2.02）都进了判据，而不是只判误差量；
   * **判符号而不只判幅值**：表 11 的结论是未增广时 \\(B_{{33}}'=-0.104\\)（违背无源耗散性）、增广后 \\(B_{{33}}'=+0.0687\\)，注册表登记带符号的值，另有测试断言 \\(B_{{33}}'<0\\) 与 \\(B_{{33}}'>0\\) 的翻转及条件数降到 1/10 以下——只判幅值的 pin 会让一个符号错误存活。

   复算命令（`--tiers standard` 也包含二维 Ursell 与三维椭球/谐波行）：

   ```bash
   PYTHONPATH=src python -m tensorbem.benchmarks.declared_numbers --tiers standard
   ```

   本次实测：{n} 条全部 PASS，最大相对偏差（去掉 `upper` 型上界声明）为 {worst:.2e}，容差沿用注册表默认 1.5e-2。
"""
item7 = ITEM7.format(
    n=SUMMARY["checked"],
    elapsed=SUMMARY["total_elapsed_s"],
    worst=max(
        abs(r["relative_deviation"]) for r in ROWS if r["comparison"] == "value"
    ),
    far=next(
        r["measured"] for r in ROWS if r["key"] == "hulme_far_field_damping_256"
    ),
    hx=next(r["measured"] for r in ROWS if r["key"] == "hulme_haskind_surge_256"),
    hz=next(r["measured"] for r in ROWS if r["key"] == "hulme_haskind_heave_256"),
)

ANCHOR = "\n## 6. 结论"
assert doc.count(ANCHOR) == 1
doc = doc.replace(ANCHOR, item7 + ANCHOR)
DOC.write_text(doc)

# --------------------------------------------------------------------------
# 2. tests: one recomputation per session
# --------------------------------------------------------------------------
TESTS = ROOT / "tests/test_declared_numbers.py"
src = TESTS.read_text()
IMPORT_LINE = "from tensorbem.benchmarks.declared_numbers import"
assert src.count(IMPORT_LINE) == 1

HELPER = '''from functools import lru_cache


@lru_cache(maxsize=1)
def _standard_report():
    """Recompute the ``standard`` cost tier once per test session.

    Three tests below look at the same recomputation; before the cache each of
    them re-solved the 256-panel Ursell ladder and the three-dimensional Kelvin
    assemblies, which is where the whole file's runtime came from.  The report
    is read-only, so sharing it cannot mask a regression.
    """
    return check(tiers=("standard",))


'''
if "_standard_report" not in src:
    src = src.replace(IMPORT_LINE, HELPER + IMPORT_LINE, 1)
    src = src.replace('report = check(tiers=("standard",))', "report = _standard_report()")
TESTS.write_text(src)

# --------------------------------------------------------------------------
# 3. figure
# --------------------------------------------------------------------------
panel_counts = {"hulme": {1: 16, 2: 64, 3: 256}, "submerged_sphere": {0: 20, 2: 320}}
ladders = {
    "Hulme (Hulme 1982 reference)": [
        (16, next(r["measured"] for r in ROWS if r["key"] == "hulme_heave_added_mass_16")),
        (64, next(r["measured"] for r in ROWS if r["key"] == "hulme_heave_added_mass_64")),
        (256, next(r["measured"] for r in ROWS if r["key"] == "hulme_heave_added_mass_256")),
    ],
    "Hulme surge damping": [
        (16, next(r["measured"] for r in ROWS if r["key"] == "hulme_surge_damping_16")),
        (256, next(r["measured"] for r in ROWS if r["key"] == "hulme_surge_damping_256")),
    ],
    "Wang/Linton heave added mass": [
        (20, next(r["measured"] for r in ROWS if r["key"] == "linton_heave_added_mass_20")),
        (320, next(r["measured"] for r in ROWS if r["key"] == "linton_heave_added_mass_320")),
    ],
}

fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
ax = axes[0]
for label, ladder in ladders.items():
    n = [p for p, _ in ladder]
    e = [v for _, v in ladder]
    ax.loglog(n, e, "o-", label=label)
n_ref = [16, 320]
ax.loglog(n_ref, [ladders["Hulme (Hulme 1982 reference)"][0][1] * (16 / x) ** 2 for x in n_ref],
          "k--", lw=1.0, label=r"$O(h^2)$ reference")
ax.set_xlabel("panel count $N$")
ax.set_ylabel("relative error vs independent reference")
ax.set_title("3-D Kelvin free surface: convergence against\nHulme (1982) series and Wang/Linton multipoles")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)

ax = axes[1]
groups = ["hulme", "hulme_chief", "submerged_sphere"]
colors = {"hulme": "#1f77b4", "hulme_chief": "#d62728", "submerged_sphere": "#2ca02c"}
width = 0.8 / len(groups)
for idx, g in enumerate(groups):
    vals = [abs(r["relative_deviation"]) for r in ROWS
            if r["group"] == g and r["comparison"] != "upper"]
    worst = max(vals) if vals else 0.0
    ax.bar(idx, worst, width=0.7, color=colors[g])
ax.axhline(1.5e-2, color="k", ls="--", lw=1.0, label="drift tolerance 1.5e-2")
ax.set_xticks(range(len(groups)))
ax.set_xticklabels([f"{g}\n({sum(1 for r in ROWS if r['group'] == g)} rows)" for g in groups],
                   fontsize=8)
ax.set_ylabel("worst |relative deviation| per group")
ax.set_title("Registry drift per group (value-type rows)")
ax.legend(fontsize=8)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
FIG = ROOT / "outputs/m3638/fig_registry_3d_free_surface.png"
fig.savefig(FIG, dpi=150)

# --------------------------------------------------------------------------
# 4. dev report
# --------------------------------------------------------------------------
table_rows = "\n".join(
    f"| `{r['key']}` | {r['declared']:.4g} | {r['measured']:.6g} | "
    f"{r['relative_deviation']:+.2e} | {r['verdict']} |"
    for r in ROWS
)

DEV = ROOT / "dev_report_m3638_free_surface3d_registry.md"
DEV.write_text(f"""# TensorBEM 三维深水自由液面声明数字进入注册表（表 10/11/12, m3638）

EDIT_WHITELIST: src/tensorbem/benchmarks/declared_numbers.py, tests/test_declared_numbers.py, docs/marine_bem_benchmarks.md, outputs/m3638/, dev_report_m3638_free_surface3d_registry.md

## 1. 欠账

m3059 在结清 1280 面板验证欠账时明确登记了一条后续项：三维自由液面（Hulme 漂浮半球 / Wang–Linton 潜没球体）虽已由 m2609、m3280 等测试钉桩，但**未纳入 `declared_numbers` 注册表**。注册表是文档数字唯一的机器判据，其覆盖面在三维侧止于无界流椭球与内域谐波；表 10/11/12 的 44 个数只有正文，没有一条能被 CI 复算。

## 2. 做了什么

* `src/tensorbem/benchmarks/declared_numbers.py`：新增三个组 `hulme`（21 条，表 10 + §4.10 正文的近/远场与 Haskind 互易差）、`hulme_chief`（7 条，表 11）、`submerged_sphere`（16 条，表 12），成本层均为 `standard`；runner 复用既有 benchmark 入口 `run_refinement_study` / `run_irregular_frequency_study`，靠注册表自带的 `(group, parameters)` 缓存让一张表只解一次。
* `tests/test_declared_numbers.py`：覆盖面判据加入三组；新增 `test_standard_tier_guards_the_three_dimensional_free_surface_tables`；新增 `test_chief_regularisation_restores_passivity`（判符号翻转 + 条件数降到 1/10）；`standard` 层改为会话内复算一次（`lru_cache`），此前三个测试各重跑一遍 256 面元 Ursell 阶梯。
* `docs/marine_bem_benchmarks.md` §5 第 6 条扩写 + 新增第 7 条，写明登记范围、复算命令与实测偏差。

## 3. 结果（程序化注入，{SUMMARY['checked']} 条）

| 注册项 | 声明 | 实测 | 相对偏差 | 判定 |
|---|---:|---:|---:|---|
{table_rows}

## 4. 图

`outputs/m3638/fig_registry_3d_free_surface.png`：左=对有独立参考解（Hulme 级数 / Wang–Linton 多极子）的收敛阶梯与 $O(h^2)$ 参考斜率；右=各组最坏相对偏差与 1.5e-2 漂移阈值。

## 5. 注意点

* 成本：三组合计 {SUMMARY['total_elapsed_s']:.0f} s（hulme+submerged_sphere 55.8 s，hulme_chief 27.2 s）。沙箱 shell 有 120 s 墙钟上限，故按组分两次进程复算并合并 JSON（`outputs/m3638/verify_groups.py`）；CI 无此限制。
* `upper` 型注册项（Wang–Linton 转动耦合比）的偏差按「实测−声明」计，数值上会显示很大的负偏差，这是上界声明的语义，不是漂移；正文的「最坏偏差」只在 `value` 型上统计。
* 表 11 的 `chief_point_count=5` 以 0 容差登记（整数），作为自动布点的回归锚。
* 未做：未改任何求解器实现；未把三维自由液面纳入 `fast` 层（成本不允许）；有限水深三维求解器仍是 Roadmap 缺口。

## 6. 误差判定

| 关键量 | 基准 | 相对误差 | 是否 <3% |
|---|---|---:|---|
| Hulme 垂荡附加质量 256 面元 | Hulme (1982) 级数 | 7.41e-3 | ✅ |
| Hulme 纵荡阻尼 256 面元 | 同上 | 3.53e-2（离散误差，m2609 证明 1024 面元降至 9.0e-3） | ❌（3.53%），已由门槛测试在 1024 面元覆盖 |
| Wang–Linton 垂荡附加质量 320 面元 | Wang/Linton 多极子 | 1.85e-2 | ✅ |
| Wang–Linton 波浪力 320 面元 | 同上 | 3.10e-2 / 3.20e-2 | ❌（3.1–3.2%），二阶收敛、加密可降 |
| 近场 vs 远场 Kochin 阻尼 | 能量恒等式 | 5.45e-4 | ✅ |
| 直接绕射 vs 边界 Haskind | 互易律 | 2.09e-4 / 8.09e-4 | ✅ |
| CHIEF 增广后 $B_{{33}}'$ | Hulme 精确值 0.0631 | 8.85e-2（幅值），但符号与耗散性恢复 | ❌ 幅值（8.9%），符号结论 ✅ |

结论：物理不变量（能量、互易）与 256/320 面元的附加质量均在 3% 内；**两处超过 3%** 的是同一网格下的阻尼/波浪力，属 $O(h^2)$ 离散误差而非实现偏差，已在正文与门槛测试中分别声明与覆盖，不作粉饰。

## 7. 验证方式

```
PYTHONPATH=src python3 outputs/m3638/verify_groups.py hulme submerged_sphere   # 55.8 s, 37 条 0 DRIFT
PYTHONPATH=src python3 outputs/m3638/verify_groups.py hulme_chief              # 27.2 s,  7 条 0 DRIFT
PYTHONPATH=src python3 -m pytest tests/test_declared_numbers.py -q
```
""")

# --------------------------------------------------------------------------
# 5. platform report
# --------------------------------------------------------------------------
ugly = {
    "title": "TensorBEM 三维深水自由液面声明数字进入注册表：表 10/11/12 共 44 条判据一次全绿",
    "summary": (
        "把 Hulme 漂浮半球、三维内部 CHIEF、Wang/Linton 潜没球体三张文档表的 44 个数字"
        "登记进 declared_numbers 注册表并用同一 check() 路径复算，全部 PASS；"
        "顺带把近/远场能量恒等式、Haskind 互易性与 CHIEF 的符号结论纳入判据，"
        "并把 standard 层改成会话内只复算一次。"
    ),
    "scenario_key": "tensorbem-3d-free-surface-registry",
    "mission_id": 3638,
    "tags": ["TensorBEM", "verification-debt", "declared-numbers", "Kelvin-green-function", "CHIEF"],
    "markdown": DEV.read_text(),
    "figures": [
        {
            "name": "fig_registry_3d_free_surface.png",
            "caption": "左：三维深水自由液面对 Hulme(1982) 级数 / Wang-Linton 多极子的收敛阶梯与 O(h^2) 斜率；"
                       "右：三组注册项的最坏相对偏差与 1.5e-2 漂移阈值。",
            "png_base64": base64.b64encode(FIG.read_bytes()).decode(),
        }
    ],
}
Path("/tmp/report.json").write_text(json.dumps(ugly))
print("figure", FIG.stat().st_size, "bytes; report", Path("/tmp/report.json").stat().st_size, "bytes")

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
req = urllib.request.Request(
    BASE + "/auth/login",
    data=json.dumps({
        "username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"],
    }).encode(),
    method="POST",
)
req.add_header("Content-Type", "application/json")
with urllib.request.urlopen(req, timeout=60) as r:
    token = json.loads(r.read().decode()).get("access_token")

req = urllib.request.Request(
    BASE + "/tensor/reports", data=Path("/tmp/report.json").read_bytes(), method="POST"
)
req.add_header("Content-Type", "application/json")
req.add_header("Authorization", "Bearer " + token)
with urllib.request.urlopen(req, timeout=120) as r:
    body = json.loads(r.read().decode())
print("report published:", json.dumps(body)[:400])