"""Post the m17210 report to the tensor platform (report discipline)."""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts/case_bem_acoustic_pulse"
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
if not BASE.endswith("/api/v1"):
    BASE = BASE.rstrip("/") + "/api/v1"


def post(path, payload, token=None):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def fig(name, caption):
    return {"name": name, "caption": caption,
            "png_base64": base64.b64encode((ART / name).read_bytes()).decode()}


def main():
    token = post("/auth/login", {
        "username": os.environ.get("TENSOR_AGENT_USER", "agent"),
        "password": os.environ.get("TENSOR_AGENT_PASS", ""),
    }).get("access_token")
    md = (ART / "report_m17210.md").read_text()
    payload = {
        "title": "CASE-BEM-ACOUSTIC-PULSE 脉动球声脉冲（瞬态）BEM 验证 — ref 锚定经典推迟势解（m17210）",
        "summary": (
            "复用 TensorBEM acoustic3d 频域 Helmholtz BEM + DFT 时域桥接，验证脉动球高斯速度脉冲在 "
            "r=20a 处的瞬态声压波形；ref=经典推迟势解（Kinsler §5.2 / Morse&Ingard / Brennen §2.3），"
            "新增 ref 锚定模块（6 项独立交叉校验全过，含冲激响应独立路线）；sub2(320 面元) L2 波形误差 "
            "1.735%、峰值误差 1.727%、到达时间偏移 0.0 ms，全部 < 8% 验收线，验收通过。"
        ),
        "scenario_key": "CASE-BEM-ACOUSTIC-PULSE",
        "mission_id": 17210,
        "tags": ["TensorBEM", "acoustic", "transient", "pulsating-sphere", "benchmark", "case-pool"],
        "markdown": md,
        "figures": [
            fig("fig_waveform.png",
                "波形对比（黑=精确推迟势，绿虚=冲激响应独立路线，蓝点=sub1 80 面元，红点线=sub2 320 面元）"
                "与逐点相对误差（对数轴，灰虚线=8% 验收线）"),
            fig("fig_anchor_checks.png",
                "ref 锚定 6 项独立交叉校验（绿=实测值，黑框=预声明容差；全部在容差内）"),
            fig("fig_band.png",
                "频带截断敏感性（sub2 网格）：4 kHz 与 8 kHz 一致 → 频带充分收敛"),
        ],
    }
    out = post("/tensor/reports", payload, token)
    print("posted:", json.dumps(out, ensure_ascii=False)[:300])


if __name__ == "__main__":
    main()
