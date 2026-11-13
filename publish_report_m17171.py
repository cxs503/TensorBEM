"""Publish the m17171 report (context craft: brief packages + layered memory) to the platform."""
from __future__ import annotations

import base64
import json
import pathlib
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:8010/api/v1"
ROOT = pathlib.Path(__file__).resolve().parent

AB = json.loads((ROOT / "docs/context_craft_ab/ab_results.json").read_text(encoding="utf-8"))
SIM = json.loads((ROOT / "docs/context_craft_ab/simulation.json").read_text(encoding="utf-8"))
DEMO = json.loads((ROOT / "docs/context_craft_demo/demo_brief.json").read_text(encoding="utf-8"))
DOC = (ROOT / "docs/dev_report_m17171_context_craft.md").read_text(encoding="utf-8")


def num(budget: str, arm: str, metric: str) -> float:
    return AB["per_budget"][budget]["summary"][arm][metric]["mean"]


# ---------------------------------------------------------------- tables
def table(metric: str) -> str:
    arms = AB["arms"]
    head = "| 预算 | " + " | ".join(arms) + " |"
    sep = "|---|" + "---|" * len(arms)
    rows = [head, sep]
    for b in AB["per_budget"]:
        rows.append(f"| {b} | " + " | ".join(f"{num(b, a, metric):.3f}" for a in arms) + " |")
    return "\n".join(rows)


RECALL = table("recall")
TOL = table("tol_recall")
GATE = table("gate")
DENS = table("decision_share")

MEMO = (
    f"60 单合成负载：压缩触发 {SIM['compressions']} 次、遗忘 {SIM['total_forgotten']} 条"
    f"（全部归档 + 日志可复算）、终态存活 {SIM['final_entries']} 条；"
    f"终态 L2/L3 token = {SIM['trace'][-1]['kept_tokens'].get('L2')}/"
    f"{SIM['trace'][-1]['kept_tokens'].get('L3')}，超预算部分报 overflow "
    f"{SIM['trace'][-1]['overflow']}（已知局限，未掩饰）"
)

FIG1 = ROOT / "docs/context_craft_ab/fig1_ab_bars.png"
FIG2 = ROOT / "docs/context_craft_ab/fig2_budget_curves.png"
FIG3 = ROOT / "docs/context_craft_ab/fig3_memory_trace.png"

BODY = DOC + f"""

---

## 7. 附：程序化注入的完整数字表（零手抄，全部来自 `docs/context_craft_ab/ab_results.json`）

留出事实召回（n={AB['tasks']} 任务，语料 {AB['corpus_items']} 条事实单元）：

{RECALL}

验收判据（TOLERANCE）召回：

{TOL}

闸门命中率（窗口内至少一条验收阈值）：

{GATE}

决策行占比（非 CONTEXT）：

{DENS}

真实干货包样例：budget {DEMO['budget_tokens']} / used {DEMO['tokens_used']} token、
{len(DEMO['provenance'])} 条、kinds `{json.dumps(DEMO['kinds'], ensure_ascii=False)}`、
package sha256 `{DEMO['package_sha256']}`、verify 全部命中。

三层记忆仿真：{MEMO}。
"""

payload = {
    "title": "干货包协议 + 三层记忆压缩：信息策展的可执行化与离线 A/B（m17171 / FEAT-CONTEXT-CRAFT）",
    "summary": (
        "新增 tools/context_craft（抽取/打分/两流组装/三层记忆/A-B/CLI，25 测试全绿）："
        f"40 任务留出式 A/B，800 token 预算下留出事实召回 {num('800','brief_local','recall'):.3f} "
        f"vs 读自己文档 {num('800','head_dump','recall'):.3f}、验收判据召回 "
        f"{num('800','brief_local','tol_recall'):.3f} vs {num('800','head_dump','tol_recall'):.3f}；"
        "诚实登记两条负面结果：①纯全局策展（无局部性配比）全面劣于基线，故引入 0.7/0.3 两流配比；"
        f"②1600 token 大预算下召回 {num('1600','brief_local','recall'):.3f} < 基线 "
        f"{num('1600','head_dump','recall'):.3f}（优势随预算收缩）。"
        "零手抄闸门 verify 32/32 命中、确定性 sha256 复算一致、遗忘链 ok=189/bad=0。"
    ),
    "scenario_key": "context_craft_brief_packages",
    "mission_id": 17171,
    "tags": ["context-engineering", "brief-package", "memory-compression", "information-curation",
             "ab-test", "offline-proxy", "honest-negative-result", "TensorBEM"],
    "markdown": BODY,
    "figures": [
        {"name": "fig1_ab_bars.png",
         "caption": "四臂 × 四预算的三项指标（留出事实召回 / 验收判据召回 / 闸门命中），误差棒为 40 任务均值的 95% CI。",
         "png_base64": base64.b64encode(FIG1.read_bytes()).decode()},
        {"name": "fig2_budget_curves.png",
         "caption": "召回随 token 预算的曲线（左）与窗口内决策行占比（右）：紧预算区干货包领先，大预算区基线追平。",
         "png_base64": base64.b64encode(FIG2.read_bytes()).decode()},
        {"name": "fig3_memory_trace.png",
         "caption": "三层记忆 60 单仿真：各层 token 占用 vs 预算（虚线）与每次压缩的遗忘条数（全部先归档再记日志）。",
         "png_base64": base64.b64encode(FIG3.read_bytes()).decode()},
    ],
}


def _env() -> dict:
    import os

    return dict(os.environ)


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


def main() -> None:
    env = _env()
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
    out = _post("/tensor/reports", payload, token=token)
    print("report id:", out.get("id") or (out.get("data") or {}).get("id") or out)


if __name__ == "__main__":
    main()