#!/usr/bin/env python
"""Publish the m23512 progress report to the platform API."""
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
    md = (ROOT / "docs" / "capytaine_finite_depth_infinite_spine_m23512.md").read_text()
    png = Path("/tmp/spine_fig.png")
    png_b64 = base64.b64encode(png.read_bytes()).decode()
    payload = {
        "title": "无限水深网格脊：有限水深交叉验证偏差的离散化归因"
                 "（XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN, m23512）",
        "summary": (
            "用 TensorBEM 无限水深 Rankine 求解器在相同几何/网格阶梯/频率上"
            "对比 capytaine，把有限水深交叉验证的 A33~12%/B33~34% 偏差归因为"
            "共享常数强度配点格式的离散化伪影（非 John 本征级数核缺陷）："
            "偏差随网格细化单调收缩，sub-3(1280 面板) 已进入 owner 15% 线内且"
            "A33/B33 均 <5%（预注册门限），收敛与频率无关。"
            "新增 6 项测试钉住该结论。"
        ),
        "scenario_key": "bem_finite_depth_infinite_spine",
        "mission_id": 23512,
        "tags": ["TensorBEM", "finite-depth", "cross-validation", "infinite-depth",
                 "discretisation", "verification-debt", "XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN"],
        "markdown": md,
        "figures": [{
            "name": "infinite_depth_spine_convergence.png",
            "caption": "无限水深网格脊：A33/B33 相对 capytaine 误差随网格细化"
                       "（sub 1→2→3 = 80/320/1280 面板）单调收缩，每个频率均如此；"
                       "sub-3 进入 owner 15% 线内且 <5% 门限——证明偏差是共享"
                       "常数强度配点格式的离散化，而非有限水深核缺陷。",
            "png_base64": png_b64,
        }],
    }
    resp = post("/tensor/reports", payload, token)
    print("report id:", resp.get("id") or resp.get("report_id") or resp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
