"""Publish the m17021 report: numeric evidence layer (numbers injected from the grader)."""
from __future__ import annotations

import base64
import json
import pathlib
import sys
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:8010/api/v1"
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from tensorbem import numeric_evidence as ne  # noqa: E402

ROWS = ne.audit()
STRUCT = {row.capability.key: row.level for row in __import__(
    "tensorbem.capability_evidence", fromlist=["audit"]).audit()}
COUNTS = ne.summary(ROWS)
CROSS = ne.contradictions(ROWS, STRUCT)

# --------------------------------------------------------------------------
# figure: recomputed error against the declared tolerance, log scale
# --------------------------------------------------------------------------
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

rows = [r for r in ROWS if r.error_pct is not None]
fig, ax = plt.subplots(figsize=(9.2, 4.4))
y = list(range(len(rows)))
errors = [max(r.error_pct, 1e-9) for r in rows]
ax.barh(y, errors, color=["#2a7f3f" if r.level == ne.REAL else "#b3282d" for r in rows])
for i, r in enumerate(rows):
    ax.plot([max(r.tolerance_pct, 1e-9)], [i], marker="|", markersize=16, color="black",
            linestyle="none")
    ax.text(max(r.error_pct, 1e-9) * 1.25, i,
            f"{r.error_pct:.3g}%  (tol {r.tolerance_pct:.3g}%)", va="center", fontsize=8)
ax.set_yticks(y)
ax.set_yticklabels([r.item_id.replace("hydrostatics-", "hydro-") for r in rows], fontsize=8)
ax.set_xscale("log")
ax.set_xlabel("recomputed relative error [%] (log scale); black tick = declared tolerance")
ax.set_title("TensorBEM numeric evidence: error re-derived from raw pairs (m17021)\n"
             "green = REAL, red = registered honest gap (WITHHELD)")
ax.set_xlim(1e-7, max(errors) * 60)
ax.invert_yaxis()
fig.tight_layout()
PNG = ROOT / "docs/fig_m17021_numeric_evidence.png"
fig.savefig(PNG, dpi=160)

table = "\n".join(
    f"| `{r.item_id}` | `{r.capability}` | {r.tolerance_pct:.3g}% | "
    f"{'n/a' if r.error_pct is None else f'{r.error_pct:.3g}%'} | **{r.level}** |"
    for r in ROWS
)
crossing = "\n".join(
    f"- `{c['capability']}`: 结构层 {c['structural_level']} vs 数值层 {c['numeric_level']} — "
    f"{'; '.join(c['reasons'])}"
    for c in CROSS
) or "- 无"

DOC = f"""## ① 案例介绍

**锚点** `CAP-LEDGER-EVIDENCE`（能力台账 21 项逐项证据链）。平台 21 项总账在
`TensorPlatform/config/capabilities.json`，不在本单 worktree 与 `EDIT_WHITELIST` 内；
本仓可交付、且确实缺失的是**证据纪律本身**：TensorBEM 的能力表
（`docs/audit_baseline.md` §4，`tensorbem.capability_evidence`）只按**结构**评级
（模块存在 + 有测试节点 + 无 stub/RNG 标记），claim 是一行自然语言。结构可以给出
`REAL` 而数字不达标——这正是台账要防的事。

物理背景：本层覆盖静水力/RAO 静水恢复力（解析恒等式）与 KP505 螺旋桨敞水性能
（实验 Table 2 基准，J=0.1…1.0 十点）。

## ② 怎么做的

| 部件 | 路径 |
|---|---|
| 评级器 + CLI（`--check`/`--write`）+ 跨层对账 | `src/tensorbem/numeric_evidence.py` |
| 判据登记表（容差+文献出处+工件+选择器） | `docs/numeric_evidence_registry.json` |
| 生成式证据矩阵 | `docs/numeric_evidence_matrix.md` |
| 门禁 + 11 个对抗性反例 | `tests/test_numeric_evidence.py` |

逐项：判据（容差 + 文献/解析出处）→ 工件（committed JSON，sha256 钉扎）→
独立复核（误差由**原始 computed/reference 对重算**，绝不抄工件自报的 error 字段；
两者不一致即 fail-closed）。工件缺失=`MISSING`、摘要不符=`STALE`、
超差=`WITHHELD`，且 `WITHHELD` 必须在登记表里声明 `expected_level`+`gap_note`，
否则门禁变红——honest FAIL 既不能被悄悄添加，也不能被悄悄删除。

## ③ 计算结果

统计：`{json.dumps(COUNTS, sort_keys=True)}`

| item | capability | tolerance | recomputed | level |
|---|---|---|---|---|
{table}

## ④ 结果图

![numeric evidence](fig_m17021_numeric_evidence.png)

横轴为对数尺度的重算相对误差，黑色竖标=声明容差，绿=REAL，红=已登记 honest gap。
两条红柱即 KP505 的 K_T/η₀ 缺口（29.1%/24.4%，容差 3%/5%），与静水力的
1e-6% 恒等量级形成 7 个数量级的对照。

## ⑤ 注意点

- **结构/数值分歧（本层第一个真实捕获）**：`propeller-open-water` 结构层 `REAL`，
  但其两条已提交判据均不达标。现作为 registered gap 公开，结构 `REAL` 不再能读成
  "数字达标"。
{crossing}
- **工件单位缺陷**：`docs/hydrostatic_verification.json` 的 `volume_relative_error`
  是量纲一比值（0.00147），同族字段却是百分数；交叉通道报出 100× 分歧才暴露。
  登记表现按项声明 `reported_error_ratio` / `reported_error_pct`，此类错位不可能再被
  读成"一致"。
- 覆盖为**部分**：4 个能力 8 条判据（结构表共 78 行）。其余能力需先把数字抽成工件。
- `tests/test_capability_coverage.py` 的 3 个失败在 `HEAD` 未改工作树上同样复现
  （`tensorbem.shape_gradient`/`tensorbem.steady_kelvin` unowned），非本单引入，未处理。

## ⑥ 误差判定（是否 <3%）

| 判据 | 重算误差 | 阈值 | <3%? |
|---|---|---|---|
| barge 排水体积 vs L·B·T | 0% | 1e-6% | 是 |
| barge 纵摇恢复 c55 vs ρgI_yy | 0% | 1e-6% | 是 |
| 半球体积（4096 面元）vs 2πR³/3 | 0.147% | 0.2% | 是 |
| 半球 GM_T vs 闭式 | 0.119% | 0.2% | 是 |
| 箱体横摇恢复 c44 vs ρgI_xx | 0% | 1e-6% | 是 |
| Wigley Cb vs 4/9 | 0.010% | 0.02% | 是 |
| KP505 K_T @ J=0.8 | **29.13%** | 3% | **否（honest FAIL）** |
| KP505 η₀ 最差点 @ J=0.9 | **24.38%** | 5% | **否（honest FAIL）** |

六条静水力/几何判据远优于 3%；两条 KP505 判据 29.1%/24.4% 远超 3%，如实登记为缺口，
未调容差、未挑选数据点（J=1.0 的 787% 越出物理区间，在 verdict.json 中另报，不计分于此）。
"""


def _env() -> dict:
    out = {}
    p = pathlib.Path("/root/tp-agent/agent.env")
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


env = _env()


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


user = env.get("TENSOR_AGENT_USER") or "agent"
pw = env.get("TENSOR_AGENT_PASS") or ""
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

(ROOT / "docs/m17021_numeric_evidence_report.md").write_text(DOC, encoding="utf-8")
payload = {
    "title": "TensorBEM 数值证据层：判据+工件+独立复核（含 KP505 honest FAIL 登记）m17021",
    "summary": (
        f"新增 tensorbem.numeric_evidence：{COUNTS['n_items']} 条判据 × {COUNTS['n_capabilities']} 个能力，"
        f"误差由工件原始 computed/reference 对重算并钉扎 sha256，REAL {COUNTS['REAL']}/"
        f"WITHHELD {COUNTS['WITHHELD']}；捕获 propeller-open-water 结构 REAL 而数值不达标"
        "（K_T 29.13% vs 3%，η₀ 24.38% vs 5%），登记为 honest gap；11 个对抗性反例门禁。"
    ),
    "scenario_key": "tensorbem_numeric_evidence_layer",
    "mission_id": 17021,
    "tags": ["TensorBEM", "capability-ledger", "numeric-evidence", "verification-debt",
             "honest-FAIL", "KP505"],
    "markdown": DOC,
    "figures": [{
        "name": "fig_m17021_numeric_evidence.png",
        "caption": "每条判据的重算相对误差（对数轴）与声明容差（黑色竖标）；绿=REAL，红=已登记 honest gap（KP505 K_T 29.1%/容差 3%、η₀ 24.4%/容差 5%）。",
        "png_base64": base64.b64encode(PNG.read_bytes()).decode(),
    }],
}
try:
    out = _post("/tensor/reports", payload, token=token)
    print("report id:", out.get("id") or out.get("data", {}).get("id") or out)
except Exception as exc:  # noqa: BLE001
    print("POST failed:", exc)