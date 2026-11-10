"""Publish the W20 item-5 report to the platform report API."""

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent
base = os.environ["TENSOR_API_BASE"].rstrip("/")
payload = json.dumps(
    {
        "username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"],
    }
).encode()
req = urllib.request.Request(
    f"{base}/auth/login", data=payload, headers={"Content-Type": "application/json"}
)
token = json.loads(urllib.request.urlopen(req).read())["access_token"]

figures = []
for name, caption in [
    (
        "w20_kt_kq_eta_vs_anchor.png",
        "KP505 open water: fine-grid (L3, 32x8 panels/blade) BEM lifting-surface "
        "K_T(J), 10K_Q(J) and eta_0(J) against the transcribed EFD anchor at "
        "J = 0.5 / 0.7 / 0.9 (+ history tie J = 0.8). The lattice is inside 4 % on "
        "K_T at J = 0.5 and 18-20 % low at the design point J = 0.70.",
    ),
    (
        "w20_order_and_gci.png",
        "Observed order p versus the predeclared p_th = 2 (+-30 % band, left) and "
        "fine-grid GCI versus the 3 % acceptance line (right). p < 0 in 4 of 6 "
        "cells: no asymptotic range, so GCI is a magnitude indicator, not a bound.",
    ),
    (
        "w20_deviation_split.png",
        "Deviation split: grey = discretisation, green = model, dot = total "
        "deviation. Closes to 1e-9 pct points by construction; with p invalid the "
        "reading is the last-increment interval: J = 0.7 / 0.9 fall outside it, "
        "J = 0.5 inside.",
    ),
]:
    path = REPO / "benchmarks/verified/bem_kp505_openwater_w20/figures" / name
    figures.append(
        {
            "name": name,
            "caption": caption,
            "png_base64": base64.b64encode(path.read_bytes()).decode(),
        }
    )

markdown = (REPO / "papers/reports/w20_propeller_openwater_bem5.md").read_text(encoding="utf-8")

report = {
    "title": "W20-5 KP505 敞水锚回核 + 三工况三密度 GCI 与 29.1% 偏差分解（TensorBEM）",
    "summary": (
        "KP505 敞水锚逐数溯源回核（70 个数带 file+sha256+line+字段+逐字源行；"
        "跨仓回核证明 TensorLBM 流传常数非源表值，ΔK_T 最大 +95%、含源表外 J=1.1）；"
        "TensorBEM 在 J=0.5/0.7/0.9 三工况 × 三套面板密度（r=2）做 GCI："
        "观测阶 6 格中 0 格落在 p_th=2 ±30%（多数为负，未进渐近区）；"
        "设计点 |ΔK_T|=18.0%、|ΔK_Q|=19.7%，均不满足 3%（honest FAIL，阈值未下调）；"
        "历史 −29.13% 在原配置复现（残差 0.004 个百分点），细分后其主体是模型/几何保真度项，"
        "并附带发现 core_ratio 在 0.025–0.10 被条带宽下限钳住（死参数）。"
    ),
    "scenario_key": "propeller-openwater-bem",
    "mission_id": 9641,
    "tags": [
        "W20",
        "propeller",
        "KP505",
        "open-water",
        "BEM",
        "GCI",
        "anchor-provenance",
        "honest-fail",
    ],
    "markdown": markdown,
    "figures": figures,
}

post = urllib.request.Request(
    f"{base}/tensor/reports",
    data=json.dumps(report).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
)
try:
    response = json.loads(urllib.request.urlopen(post).read())
except urllib.error.HTTPError as exc:
    print("STATUS", exc.code)
    print(exc.read().decode()[:3000])
    raise SystemExit(1)
print(json.dumps(response, ensure_ascii=False, indent=2)[:2000])