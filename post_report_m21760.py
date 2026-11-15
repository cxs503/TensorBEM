#!/usr/bin/env python3
"""Publish the CASE-BEM-CUMMINS-WIGLEY band-extension verification-debt report (mission 21760)."""
from __future__ import annotations
import base64, json, os, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = os.environ["TENSOR_API_BASE"]

def _post(path, body, token=None):
    req = urllib.request.Request(BASE+path, data=json.dumps(body).encode(),
                                 headers={"Content-Type":"application/json"})
    if token: req.add_header("Authorization","Bearer "+token)
    return json.load(urllib.request.urlopen(req))

def main():
    token = _post("/auth/login", {"username":os.environ["TENSOR_AGENT_USER"],
                                  "password":os.environ["TENSOR_AGENT_PASS"]})["access_token"]
    md = (ROOT/"docs"/"m21760_cummins_band_extension"/"report_baseline.md").read_text()
    report = {
        "title": "CASE-BEM-CUMMINS-WIGLEY：扩展频带（12→20 rad/s）验证欠账收口——surge/sway/yaw 可判性基线审计与集群递交",
        "summary": "已提交 0.2–12 rad/s medium 扫频上，零自由参数 KK 闸门 C3a 对 surge/sway/yaw 三条 DOF 的 n_decidable=0/12（实测尾指数 1.095/1.529/1.616 < 2.05，截断预算 ∞）——判据对它们从未被带限数据认证，残差只是经验一致。本单把这条验证欠账钉成数字，并递交扩展带（60×16=3840 面板、0.2–20 rad/s、55 节点，cj3708）与扩展锚点（cj3711）以把这三条升级为认证；C8/C9 审计脚本就绪待工件。",
        "scenario_key": "CASE-BEM-CUMMINS-WIGLEY",
        "mission_id": "21760",
        "tags": ["BEM","Cummins","Kramers-Kronig","Wigley","verification-debt","band-extension","6DOF"],
        "markdown": md,
        "figures": [],
    }
    r = _post("/tensor/reports", report, token)
    print("report id:", r.get("id", r))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
