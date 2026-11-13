#!/usr/bin/env python3
"""Publish the m17396 audit report (paper-quality six-section) to the platform."""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
HERE = Path(__file__).resolve().parent


def req(method, path, payload=None, token=None):
    r = urllib.request.Request(BASE + path, method=method)
    r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", "Bearer " + token)
    data = json.dumps(payload).encode() if payload is not None else None
    with urllib.request.urlopen(r, data=data, timeout=120) as resp:
        return resp.status, json.loads(resp.read() or b"{}")


_, d = req("POST", "/auth/login", {"username": USER, "password": PASS})
tok = d["access_token"]

body = {
    "title": "交差审计 CASE-BEM-IRREG-FREQ-CYL（原单 m17374）：benchmark 是同义反复 → 裁定 L3",
    "summary": (
        "逐字重跑 m17374 的四条 repro_cmd（entry/benchmark/integration/pytest，36 passed，工件逐位一致）后，"
        "独立复算证明其 headline 指标是恒等式：nu_pred=j/R_wp(mesh)、nu_ref=j/a 共享同一批 Bessel 常数，"
        "残差恒等于 a/R_wp−1（恒等式残差 1.0e-13），且 8 个面板算点上与内接多边形面积亏损 pi^2/3N^2 吻合，"
        "即 0.0201%–0.3222% 只度量水线面离散、对零点知识零信息量；平台裁判行审计时缺行（F1 承接 m17304）。"
        "求解器自身证据（route C，审计独立重跑）为 +1.42%(264 panels)/+3.39%(112 panels)，但那属 m16777 而非本单。"
        "裁定 L3（系统集成），补登 tensor_benchmark_cases 裁判行 id=58（非循环口径），verdict id=541。"
    ),
    "scenario_key": "board_audit_case_bem_irreg_freq_cyl",
    "mission_id": 17396,
    "tags": ["TensorBEM", "CASE-BEM-IRREG-FREQ-CYL", "board-audit", "maturity-L3",
             "irregular-frequency", "tautological-benchmark", "m17396"],
    "markdown": (HERE / "AUDIT_REPORT.md").read_text(),
    "figures": [
        {"name": "fig_geom_scaling.png",
         "caption": "审计图1：原单 headline 残差 vs 水线面面板数（双对数），叠加内接正多边形面积亏损模型 "
                    "pi^2/(3N^2)。两个几何（R=1,T/R=0.5 与 R=2,T/R=1）完全重合在几何模型上 → 残差全部来自"
                    "水线面离散，与 Bessel 零点无关。",
         "png_base64": base64.b64encode((HERE / "fig_geom_scaling.png").read_bytes()).decode()},
        {"name": "fig_route_abc.png",
         "caption": "审计图2：左=Route A（交付预测，刚盖 Bessel 零点）vs Route B（自由面一致精确圆柱特征值，"
                    "tan(kzT)=kz/nu），差 18.63%–70.57%；右=Route C（求解器自身 bare BEM 条件数扫描，两张网格），"
                    "伪共振带紧贴 Route A 预测线（+1.42%/+3.39%），说明离散系统跟刚盖模态走、不跟自由面一致解走。",
         "png_base64": base64.b64encode((HERE / "fig_route_abc.png").read_bytes()).decode()},
    ],
}
st, out = req("POST", "/tensor/reports", body, tok)
print(st, json.dumps(out)[:300])