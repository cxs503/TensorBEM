#!/usr/bin/env python3
"""Publish the CASE-BEM-CUMMINS-WIGLEY verdict-coverage report (mission 21960)."""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = os.environ["TENSOR_API_BASE"]


def _post(path, body, token=None):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    if token:
        req.add_header("Authorization", "Bearer " + token)
    return json.load(urllib.request.urlopen(req))


def main() -> int:
    token = _post(
        "/auth/login",
        {"username": os.environ["TENSOR_AGENT_USER"], "password": os.environ["TENSOR_AGENT_PASS"]},
    )["access_token"]
    md = (ROOT / "docs" / "dev_report_m21960_cummins_verdict_closure.md").read_text()
    verdict = json.loads(
        (
            ROOT
            / "benchmarks"
            / "verified"
            / "casepool_bem_cummins_wigley"
            / "case_result.json"
        ).read_text()
    )
    modes = verdict["gates"]["C5b"]["values"]["modes"]
    floor = verdict["external_anchor"]["noise_floor"]["quantities"]
    md += (
        "\n\n## 程序注入数字（与提交件同源）\n\n```json\n"
        + json.dumps(
            {
                "verdict": verdict["verdict"],
                "c5b": {
                    k: {"rel": v["relative_error"], "grade": v["grade"], "scored": v["scored"]}
                    for k, v in modes.items()
                },
                "c10_worst": verdict["gates"]["C10"]["values"]["worst"],
                "noise_floor": {k: v["worst"] for k, v in floor.items()},
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n```\n"
    )
    figure = ROOT / "docs" / "fig_m21960_cummins_verdict_closure.png"
    report = {
        "title": "CASE-BEM-CUMMINS-WIGLEY R8：外部程序锚点（C10）与全可判衰减模态（C5b）收进提交件",
        "summary": (
            "把已存在但未进裁决的两条闸门收进 case_result.json：C10（capytaine 同网格外部锚点）与 "
            "C5b（heave 3.96% / pitch 4.88% 计分、roll 23.89% 因估计器不可判只报告不计分），裁决 13→15 条、"
            "15/15 通过。并定量证明 C10 的 6.09% 残差小于外来程序自身面板扩散（比值 0.74）——PASS 只能说"
            "「落在参考件离散化噪声带内」，钉为 distinguishable=false 并以测试锁住。"
        ),
        "scenario_key": "case_bem_cummins_wigley",
        "mission_id": 21960,
        "tags": [
            "TensorBEM",
            "CASE-BEM-CUMMINS-WIGLEY",
            "verification-debt",
            "external-anchor",
            "capytaine",
            "free-decay",
            "case-pool",
        ],
        "markdown": md,
        "figures": [
            {
                "name": figure.name,
                "caption": (
                    "A: 逐频率 C10 残差 vs capytaine 自身面板阶梯扩散（同一把尺子）；"
                    "B: C5b 逐模态自由衰减（roll 斜杠=不计分）；C: 提交件 15/15 裁决。"
                ),
                "png_base64": base64.b64encode(figure.read_bytes()).decode(),
            }
        ],
    }
    result = _post("/tensor/reports", report, token)
    print("report id:", result.get("id", result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())