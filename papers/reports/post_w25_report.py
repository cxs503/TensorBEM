"""Publish the W25 (m16712) seakeeping-truncation report to the Tensor platform.

Six elements, zero hand-copied numbers: the markdown is the rendered report
produced by ``papers/reports/build_w25_report.py`` from the artifact, and the
figures are the PNGs that builder wrote.
"""

from __future__ import annotations

import base64
import json
import pathlib
import urllib.request

BASE = "http://127.0.0.1:8010/api/v1"
ROOT = pathlib.Path(__file__).resolve().parents[2]
ART_DIR = ROOT / "papers/artifacts/w25/seakeeping-truncation"
ART = ART_DIR / "journee_rao_truncation.json"
MD = ROOT / "papers/reports/w25_seakeeping_truncation5.md"
MISSION_ID = 16712

FIGURES = [
    ("fig1_rao_curves.png", "Wigley III 迎浪 heave/pitch RAO（U=0，TensorBEM 三维 Green 函数 BEM），"
                            "竖虚线为四档截断频率。"),
    ("fig2_truncation_frequency.png", "轴 A：带内响应 rms 随上端截断频率（加倍阶梯）变化，三个 ITTC 海况。"),
    ("fig3_truncation_radius_longwave.png", "轴 B：响应 rms 随长波截断半径 λmax/L（加倍阶梯）变化。"),
    ("fig4_truncation_radius_geometric.png", "轴 C：定常截断域 Rankine 解的 Cw 随横向截断半径变化（标注未知量数），"
                                            "两个任务 Froude 数。"),
]


def _env() -> dict:
    out = {}
    path = pathlib.Path("/root/tp-agent/agent.env")
    for line in path.read_text().splitlines():
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
    c3 = crit["criterion_3_doubling_convergence"]
    summary = (
        "Wigley III 迎浪 RAO 截断收敛研究：轴 A（截断频率）/轴 B（长波截断半径）各 4 档加倍阶梯，"
        f"判据③截断加倍变化最大 {c3['worst_rel_change_pct']:.3f}% → {c3['verdict']}；"
        "判据①②因 Journée(1992) 试验锚不可得（且指定锚年代不自洽）记 WITHHELD；"
        "同时修掉 wigley3_hull KB 公式 +30% 缺陷。"
    )
    figures = []
    for name, caption in FIGURES:
        p = ART_DIR / name
        if p.is_file():
            figures.append(
                {
                    "name": name,
                    "caption": caption,
                    "png_base64": base64.b64encode(p.read_bytes()).decode(),
                }
            )
    payload = {
        "title": "W25 · 耐波性 RAO 截断收敛 vs Journée(1992) 锚（TensorBEM）",
        "summary": summary,
        "scenario_key": "seakeeping_truncation",
        "mission_id": MISSION_ID,
        "tags": [
            "seakeeping",
            "RAO",
            "truncation",
            "BEM",
            "TensorBEM",
            "ITTC-procedure",
            "WITHHELD-anchor",
        ],
        "markdown": MD.read_text(),
        "figures": figures,
    }
    out = post("/tensor/reports", payload, token)
    print(json.dumps(out, ensure_ascii=False)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
