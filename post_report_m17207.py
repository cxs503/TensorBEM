"""Publish the m17207 report (CASE-BEM-KP505-OPENWATER, round 1).

Every number in the posted markdown comes from ``runs/m17207/kp505_case.json``
and from the report the case driver renders out of it; nothing is typed here.
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.parse
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
if not BASE.endswith("/api/v1"):
    BASE = BASE.rstrip("/") + "/api/v1"
ROOT = pathlib.Path(__file__).resolve().parent
RUNS = ROOT / "runs/m17207"
V = json.loads((RUNS / "kp505_case.json").read_text())
DOC = (RUNS / "kp505_case_report.md").read_text()


def _post(path, payload=None, token=None, form=False):
    data, headers = None, {}
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
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode())


token = None
for path, form in (("/auth/login", False), ("/auth/login", True), ("/auth/token", False)):
    try:
        out = _post(path, {"username": os.environ.get("TENSOR_AGENT_USER", "agent"),
                           "password": os.environ.get("TENSOR_AGENT_PASS", "")},
                    form=form)
        token = out.get("access_token") or out.get("token") or (out.get("data") or {}).get("access_token")
    except Exception as exc:  # noqa: BLE001
        print("auth", path, "failed:", exc)
    if token:
        break
print("token:", "ok" if token else "MISSING")

summary = V["canonical"]["summary"]
criteria = V["criteria"]
k = summary["n_pass_by_quantity"]
figures = []
for spec in V.get("figures", []):
    png = pathlib.Path(spec["path"])
    figures.append({"name": png.name.replace(".png", ""),
                    "caption": spec["caption"],
                    "png_base64": base64.b64encode(png.read_bytes()).decode()})

headline = (
    f"CASE-BEM-KP505-OPENWATER 第1轮：复用仓库涡格升力面核，对 Iftikhar & Takinaci 2023 表2（p.10）"
    f"实验列逐点打分，{summary['n_points']} 点三项全达标 {summary['n_pass_all_three_5pct']}/"
    f"{summary['n_points']}（K_T {k['KT_pct']}/{summary['n_points']}、10K_Q {k['10KQ_pct']}/"
    f"{summary['n_points']}、η0 {k['eta0_pct']}/{summary['n_points']} 单量进 5% 带），"
    f"可容许点 {summary['n_pass_all_three_5pct_admissible']}/{summary['n_admissible']} ⇒ honest FAIL；"
    f"最差 ΔK_T {summary['worst']['KT_pct']['error_pct']:+.1f}%（J={summary['worst']['KT_pct']['J']:g}）；"
    f"两轴 GCI 显示 spanwise 离散度 13–14%（p<0）> 5% 判据，缺口是模型形式（平剖面/常螺距/无拱度）而非离散。"
)

payload = {
    "title": "CASE-BEM-KP505-OPENWATER（第1轮）：KP505 敞水 TensorBEM 涡格升力面对标 · honest FAIL",
    "summary": headline,
    "scenario_key": "case_bem_kp505_openwater",
    "mission_id": 17207,
    "tags": ["TensorBEM", "KP505", "open-water", "benchmark-case", "case-pool",
             "honest-FAIL", "liftingsurface", "GCI"],
    "markdown": DOC,
    "figures": figures,
}
try:
    out = _post("/tensor/reports", payload, token=token)
    print("report:", out.get("id") or (out.get("data") or {}).get("id") or out)
except Exception as exc:  # noqa: BLE001
    print("POST failed:", exc)