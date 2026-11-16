#!/usr/bin/env python
"""Publish the m23144 progress report to the platform API."""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]


def post(path: str, payload: dict, token: str | None = None) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    login = post("/auth/login", {"username": USER, "password": PASS})
    token = login.get("access_token") or login.get("token")
    md = (ROOT / "docs" / "xval_bem_finite_depth_decidability.md").read_text()
    png = ROOT / "docs" / "fig_m23144_reference_truth_decidability.png"
    audit = json.loads(
        (ROOT / "benchmarks/verified/capytaine_finite_depth/"
         "reference_truth_audit.json").read_text()
    )["summary"]
    payload = {
        "title": "有限水深 BEM 交叉验证的可判性契约 + 参考侧真值审计"
                 "（XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN, m23144）",
        "summary": (
            "把「偏差是否达标」升级为三值可判性判决（PASS/FAIL/UNDECIDABLE），"
            "并用精确多极子锚实测 capytaine 参考自身误差 1.05–3.82 %："
            "「末档变化」代理在 32/32 个量上偏乐观（最大 3.71×），"
            "故 3 %/5 % 预登记带对该参考不可达，owner 15 % 线尚未被检验。"
            f"准入 {audit['admitted_cells']} 格，排除 {audit['excluded_cells']} 格。"
        ),
        "scenario_key": "bem_finite_depth_crossval_decidability",
        "mission_id": 23144,
        "tags": ["TensorBEM", "finite-depth", "cross-validation", "decidability",
                 "verification-debt", "XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN"],
        "markdown": md,
        "figures": [
            {
                "name": "fig_m23144_reference_truth_decidability.png",
                "caption": "左：capytaine 参考对精确多极子锚的实测误差 vs 离底指数，"
                           "红/橙虚线为 3 %/5 % 容差的 1/3 预算；右：实测误差与"
                           "「末档变化」代理之比（>1 表示代理乐观）。",
                "png_base64": base64.b64encode(png.read_bytes()).decode(),
            }
        ],
    }
    out = post("/tensor/reports", payload, token)
    print(json.dumps(out)[:500])
    (ROOT / "report_m23144.json").write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())