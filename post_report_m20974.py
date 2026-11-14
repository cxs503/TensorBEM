"""Publish the band-edge report for CASE-BEM-CUMMINS-WIGLEY (mission 20974)."""

from __future__ import annotations

import base64
import json
import os
import pathlib

import requests

BASE = os.environ.get("TENSOR_API_BASE", "").rstrip("/") or "http://127.0.0.1:8010/api/v1"
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
ROOT = pathlib.Path(__file__).resolve().parents[0]
DOC = ROOT / "docs" / "case_bem_cummins_wigley_band_edges.md"
FIG = ROOT / "artifacts" / "case_bem_cummins_wigley_ext" / "fig_band_edges.png"
EDGES = ROOT / "artifacts" / "case_bem_cummins_wigley_ext" / "band_edges.json"


def token() -> str:
    r = requests.post(f"{BASE}/auth/login", json={"username": USER, "password": PASS}, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data.get("access_token") or data.get("token") or data["data"]["access_token"]


def main() -> int:
    edges = json.loads(EDGES.read_text())
    body = DOC.read_text()
    fig = base64.b64encode(FIG.read_bytes()).decode()
    q = {r["dof"]: r["low_exponent_q"]["exponent"] for r in edges["rows"]}
    payload = {
        "title": "CASE-BEM-CUMMINS-WIGLEY 频带两端实测：q(heave)=3 与出货偶极闭合一致，"
                 "子带对 K(0) 贡献 <0.0013 %",
        "summary": (
            "扩展带 medium 面板扫频（集群 cj3099，14 节点入仓）首次给出 0.30 rad/s 以下的"
            "Wigley 辐射阻尼：低频幂指数实测 q = 7.00/7.00/2.995/7.00/6.997/10.999（六 DOF，"
            "R²>0.999），出货偶极闭合 B~cω³ 只对 heave 成立；但 [0,0.10] 子带对 "
            "K(0)=(2/π)∫B dω 的贡献 <0.0013 %（六 DOF），故低频闭合作为误差源在真实面板数据上"
            "被定量排除。高频端 4.7→12.5 弦斜率在四个 DOF 仍为正，"
            "“ω_max 以上指数衰减”不可支持——记为局限而非通过。G1–G5 未重算（14/28 节点，"
            "gates 阶段按设计拒判），锚点仍 open。"
        ),
        "scenario_key": "case_bem_cummins_wigley_band_edges",
        "mission_id": 20974,
        "tags": ["TensorBEM", "BEM", "Cummins", "Kramers-Kronig", "Wigley",
                 "case-pool", "verification-debt"],
        "markdown": body,
        "figures": [{
            "name": "fig_band_edges.png",
            "caption": "左：六自由度对角辐射阻尼 B_ii(ω)（实测节点）与实测低频幂律（点线）；"
                       "右：两端指数——低频 q 与 4.7→12.5 弦斜率 p，绿虚线为出货偶极闭合 q=3。",
            "png_base64": fig,
        }],
        "metadata": {"job": "cj3099", "commit": "9c645a7a",
                     "low_exponents": q, "level": edges["level"],
                     "nodes": edges["nodes"]},
    }
    tok = token()
    r = requests.post(f"{BASE}/tensor/reports", json=payload,
                      headers={"Authorization": f"Bearer {tok}"}, timeout=180)
    print(r.status_code, r.text[:400])
    return 0 if r.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())