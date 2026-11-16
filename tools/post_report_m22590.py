"""Publish the m22590 report to the platform (login + POST /tensor/reports)."""

import base64
import json
import os
import urllib.request

base = os.environ["TENSOR_API_BASE"].rstrip("/")
login = urllib.request.Request(
    f"{base}/auth/login",
    data=json.dumps(
        {
            "username": os.environ["TENSOR_AGENT_USER"],
            "password": os.environ["TENSOR_AGENT_PASS"],
        }
    ).encode(),
    headers={"Content-Type": "application/json"},
)
token = json.load(urllib.request.urlopen(login, timeout=60))["access_token"]

png = base64.b64encode(
    open("docs/fig_ice_gate_power_section.png", "rb").read()
).decode()
markdown = open("docs/dev_report_m22590_ice_gate_power_section.md").read()
markdown += (
    "\n\n## Figures\n\n![fig](figures/fig_ice_gate_power_section.png)\n\n"
    "(a) 网格阶梯 vs Richardson 极限与两处争议值；(b) 判决柱状图；"
    "(c) 判据信息量：几何缺陷→偏差曲线与 3 %/15 % 门。\n"
)
body = {
    "title": (
        "GAP-ICE-GATE-POWER / S07′：二维垂荡附加质量锚点给出信息量"
        "（A_inf=225.35 t/m，确认 #19234 的 228.6 t/m 差 +1.44 %）"
    ),
    "summary": (
        "TensorBEM 新增二维垂荡附加质量锚点：镜像加倍体 + 物理面单侧载荷积分，"
        "同时求 ω→∞（φ=0）与 ω→0（刚性壁）两支并给出镜像线残差证明；闭合解基线"
        "（圆 5e-6、椭圆 3.8e-4、半浸圆 5e-6）钉住算子；冻结网格观测阶 p=1.25；"
        "判据可检出 1.3 %/7 % 几何缺陷；五条负控四条被拒（错谱极限 −163.9 %），"
        "粗网格锁 +2.6 % 未被 3 % 门拒（如实记录）。结论：该剖面 A_inf=225.35 t/m，"
        "与审计 #19234 的 228.6 t/m 差 +1.44 %，链声明 86.1 t/m 的 165 % 缺口由第三条"
        "独立路线确认；#19477 的 2D/3D 口径折算（5.97 %）在「按长度系数 vs 收敛二维 "
        "A_inf」口径下不复现。"
    ),
    "scenario_key": "gap-ice-gate-power-section",
    "mission_id": 22590,
    "tags": [
        "TensorBEM",
        "GAP-ICE-GATE-POWER",
        "S07",
        "added-mass",
        "verification",
        "negative-control",
        "criterion-informativeness",
    ],
    "markdown": markdown,
    "figures": [
        {
            "name": "fig_ice_gate_power_section.png",
            "caption": (
                "(a) 网格阶梯 vs Richardson 极限与两处争议值；(b) 判决柱状图；"
                "(c) 判据信息量：几何缺陷→偏差曲线与 3 %/15 % 门"
            ),
            "png_base64": png,
        }
    ],
}
request = urllib.request.Request(
    f"{base}/tensor/reports",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
)
with urllib.request.urlopen(request, timeout=120) as response:
    print(response.status, response.read()[:400].decode())