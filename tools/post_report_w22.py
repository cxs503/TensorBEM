"""Publish the W22 KP505 report (markdown + figures) to the platform report API."""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARTE = REPO / "benchmarks/verified/kp505_anchor_w22"
REPORT = REPO / "papers/reports/w22_propeller_openwater6.md"

FIGURE_CAPTIONS = {
    "kp505_curves.png": (
        "KP505 敞水曲线：五档 J 上 BEM（L3, 32x8 面板/叶、96 尾涡段；L1/L2 淡线示梯度）"
        "与仓内试验锚（Table 2 Experimental 列）的 K_T、10K_Q、η₀ 对比。"
    ),
    "kp505_deviations.png": (
        "相对试验锚的逐档偏差（%）；红色虚线为本单预声明的设计点 ±5 % 门槛。"
        "K_T 在设计点 J = 0.70 明显出带，J = 0.9 偏差最大。"
    ),
    "kp505_panel_ladder.png": (
        "面元密度梯度（三轴同时加密，r = 2）：K_T、K_Q 随每叶面板数变化；"
        "曲线在 L2→L3 仍未单调收敛，是观测阶为负的直接来源。"
    ),
    "kp505_gci.png": (
        "各 (J, 量) 的观测阶 p 与细网格 GCI 指数；p < 0 的格子不在渐近区，"
        "GCI 只能当量级指示，不能当误差上界。"
    ),
    "kp505_attribution.png": (
        "设计点 J = 0.70 的 K_T 偏差归因：总偏差、Richardson 离散项与残余模型项。"
        "模型项占主体（零弯度平板剖面、径向常数螺距、无毂/厚度堵塞）。"
    ),
}


def main() -> int:
    base = os.environ["TENSOR_API_BASE"].rstrip("/")
    payload = json.dumps({
        "username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"],
    }).encode()
    request = urllib.request.Request(
        f"{base}/auth/login", data=payload, headers={"Content-Type": "application/json"}
    )
    token = json.loads(urllib.request.urlopen(request).read())["access_token"]

    markdown = REPORT.read_text()
    compare = json.loads((ARTE / "openwater_compare.json").read_text())

    figures = []
    for name, caption in FIGURE_CAPTIONS.items():
        path = ARTE / "figures" / name
        if not path.is_file():
            continue
        figures.append({
            "name": name,
            "caption": caption,
            "png_base64": base64.b64encode(path.read_bytes()).decode(),
        })

    sys.path.insert(0, str(REPO / "tools"))
    from emit_w22_report import summary_text

    transcription = json.loads((ARTE / "kp505_transcription.json").read_text())
    summary = summary_text(transcription, compare)

    report = {
        "title": "W22 KP505 敞水：锚可达性回核与五档 BEM 对比（honest FAIL @ 设计点）",
        "summary": summary,
        "scenario_key": "propeller-openwater-bem",
        "mission_id": 14506,
        "tags": ["TensorBEM", "propeller", "KP505", "open-water", "BEM", "GCI", "honest-FAIL", "W22"],
        "markdown": markdown,
        "figures": figures,
    }
    Path("/tmp/report.json").write_text(json.dumps(report))
    request = urllib.request.Request(
        f"{base}/tensor/reports",
        data=json.dumps(report).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    response = json.loads(urllib.request.urlopen(request).read())
    print(json.dumps(response, ensure_ascii=False)[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())