#!/usr/bin/env python
"""Publish the m25365 progress report to the platform API (PR#1195 reopen, round 3)."""
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
FIG = ROOT / "benchmarks/verified/casepool_bem_hulme_hemisphere/band_current_code/fig_m25365_band_cells.png"


def post(path, payload, token=None):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode())


def main():
    login = post("/auth/login", {"username": USER, "password": PASS})
    token = login.get("access_token") or login.get("token")
    md = (ROOT / "dev_report_m25365_pr1195_reopen_round3.md").read_text()
    png_b64 = base64.b64encode(FIG.read_bytes()).decode()
    payload = {
        "title": "Hulme 半球带扩展单元交付物钉死（PR#1195 重开·管到底第3轮·m25365）",
        "summary": (
            "PR#1195 的 m21845 交付物（README +19、cells/ 9 个单元 JSON）自 #1026 起已在 main；"
            "本轮补证据面：新增 tests/test_hulme_band_current_cells.py 把 9 个单元交付物钉死"
            "（清单 1:1、schema、rel_error/幅相自文件重推、与 results.json 跨工件逐位一致、"
            "E1/E3/E5 判决重放），10 项断言全过（0.25 s，零求解）。扩展带 ka<=3.5 全部 24 量 <3%；"
            "ka=4 A11 8.94% 为网格受限的诚实 FAIL，与已入库带边裁定一致。"
        ),
        "scenario_key": "bem_hulme_hemisphere_band_cells_pin",
        "mission_id": 25365,
        "tags": ["TensorBEM", "hulme-hemisphere", "benchmark", "verification",
                 "pr1195-reopen", "m21845", "see-through-round3"],
        "markdown": md,
        "figures": [{
            "name": "fig_m25365_band_cells.png",
            "caption": ("(a) 场量云图：committed 场数据（ka=1.0，256 面板）上 |phi_surge| 半球面分布；"
                        "(b) 1024 面板扩展带六量相对误差 vs ka，5% 门与 15% 硬限，ka=4 A11 失败点标注；"
                        "(c) 不规则频率处 CHIEF 开/关对照（带最大误差 + 条件数）。"),
            "png_base64": png_b64,
        }],
    }
    out = post("/tensor/reports", payload, token)
    print("report posted:", json.dumps(out, ensure_ascii=False)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
