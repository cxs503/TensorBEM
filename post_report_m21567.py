#!/usr/bin/env python3
"""Publish the XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN verification-debt report (mission 21567)."""
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
    md = (ROOT/"docs"/"xval_bem_finite_depth_crossval_plan.md").read_text()
    report = {
        "title": "XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN: 有限水深 capytaine 交叉验证——全量12格验证欠账收口",
        "summary": "对 tensorbem.finite_depth_solver3d 做有限水深 code-to-code 交叉验证（对标 capytaine 有限水深 Green 函数）。计划已定义（h/a×k0a 12格、Richardson 外推、负控、双码细化），全量 12 格生产已递交集群（8 作业 in-flight）。本单收口验证欠账：文档化计划/当前 1 格状态/收口流程，并显式声明当前未外推不能判 <3%。",
        "scenario_key": "XVAL-BEM-FINITE-DEPTH-CROSSVAL-PLAN",
        "mission_id": 21567,
        "tags": ["BEM","finite-depth","capytaine","crossval","verification-debt"],
        "markdown": md,
        "figures": [],
    }
    r = _post("/tensor/reports", report, token)
    print(json.dumps(r, indent=1)[:500])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
