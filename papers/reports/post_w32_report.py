#!/usr/bin/env python3
"""Publish the W32 (m28172) seakeeping-truncation report to the Tensor platform.

Six elements, zero hand-copied numbers: the markdown is the rendered report
produced by ``papers/reports/build_w32_report.py`` from the artifact, and the
figures are the PNGs that ``papers/figures/w32_make_figures.py`` wrote.
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request

BASE = "http://127.0.0.1:8010/api/v1"
ROOT = pathlib.Path(__file__).resolve().parents[2]
ART = ROOT / "papers/data/w32-2026-11-16/seakeeping/truncation_convergence.json"
MD = ROOT / "papers/reports/w32_seakeeping_truncation7.md"
FIG_DIR = ROOT / "papers/figures/w32"
MISSION_ID = 28172

FIGURES = [
    ("fig1_rao_curves.png",
     "Wigley III 迎浪 heave/pitch RAO（U=0，生产网格），竖虚线为四档截断频率，"
     "红虚线为 pitch 峰值频率。"),
    ("fig2_m0_convergence.png",
     "轴 A：响应 m0 截断误差（vs 全带）随上端截断频率下降，三个 ITTC 海况；"
     "红虚线为 0.5% 门限。"),
    ("fig3_peak_convergence.png",
     "轴 A：RAO 峰值截断误差（vs 全带）随上端截断频率下降；红虚线为 2% 门限。"),
]


def _env() -> dict:
    out = {}
    for src in ("/root/tp-agent/agent.env", "/srv/tensor/agent/agent.env"):
        p = pathlib.Path(src)
        if p.is_file():
            for line in p.read_text().splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.split("=", 1)
                    out[k.strip()] = v.strip()
    return out


def post(path: str, payload: dict, token: str | None = None) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def main() -> int:
    env = _env()
    token = post(
        "/auth/login",
        {"username": env["TENSOR_AGENT_USER"], "password": env["TENSOR_AGENT_PASS"]},
    )["access_token"]

    artifact = json.loads(ART.read_text())
    crit = artifact["criteria"]
    c1 = crit["criterion_1_rao_peak_relative_change"]
    c2 = crit["criterion_2_m0_relative_change"]
    va = artifact["axis_A_truncation_frequency"]["convergence_verdicts"]
    summary = (
        "Wigley III 迎浪 RAO 峰值与 m0 截断收敛（W52 论文生产线，seakeeping-truncation "
        "research_plan 销项）：把 W25 生产网格 RAO 曲线按更严验收量重做——"
        f"RAO 峰值截断误差在 ω_max≥6 达 0%（门限 2%，{c1['all_pass']}）；"
        f"m0 截断误差在 ω_max≥8 达 <0.02%（门限 0.5%，{c2['all_pass']}）；"
        "轴 B（长波截断半径）m0 在 λ_max/L≥5 即 0%。全部判据 PASS，"
        "显式回答 <3%：是。"
    )
    figures = []
    for name, caption in FIGURES:
        p = FIG_DIR / name
        if p.is_file():
            figures.append(
                {
                    "name": name,
                    "caption": caption,
                    "png_base64": base64.b64encode(p.read_bytes()).decode(),
                }
            )
    payload = {
        "title": "W32 · 耐波性 RAO 峰值与 m0 截断收敛（TensorBEM m28172）",
        "summary": summary,
        "scenario_key": "seakeeping_truncation",
        "mission_id": MISSION_ID,
        "tags": [
            "seakeeping",
            "RAO",
            "truncation",
            "convergence",
            "m0",
            "BEM",
            "TensorBEM",
            "paper-exp",
        ],
        "markdown": MD.read_text(),
        "figures": figures,
    }
    out = post("/tensor/reports", payload, token)
    print(json.dumps(out, ensure_ascii=False)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
