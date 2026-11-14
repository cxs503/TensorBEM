"""Publish the m2198 backfill report for Ursell convergence study."""

import base64
import json
import os
import pathlib
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
if not BASE.endswith("/api/v1"):
    BASE = BASE.rstrip("/") + "/api/v1"
ROOT = pathlib.Path(__file__).resolve().parent

def post(path, data=None, token=None):
    url = BASE + path
    body = json.dumps(data).encode() if data else None
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        },
    )
    return json.loads(urllib.request.urlopen(request, timeout=180).read())

# Login
login = post(
    "/auth/login",
    {
        "username": os.environ.get("TENSOR_AGENT_USER", "agent"),
        "password": os.environ.get("TENSOR_AGENT_PASS", ""),
    },
)
token = login.get("access_token") or login.get("token")
print(f"Login OK, token length: {len(token) if token else 0}")

# Read the report markdown
report_md = (ROOT / "docs" / "report_m2198.md").read_text()

# Read the figure if exists
figures = []
fig_path = ROOT / "docs" / "m2198" / "ursell_convergence.png"
if fig_path.exists():
    fig_b64 = base64.b64encode(fig_path.read_bytes()).decode()
    figures.append({
        "name": "ursell_convergence",
        "caption": "Ursell 潜没圆柱 BEM 收敛性研究：左图为附加质量/阻尼相对误差随面元数变化（对数坐标），右图为近/远场阻尼一致性检验",
        "png_base64": fig_b64,
    })

payload = {
    "title": "Mission 2198 补报：Ursell 潜没圆柱 BEM 收敛性复核与文档修正（ka=0.6, a/f=0.5, 深水）",
    "summary": (
        "Ursell (1949) 潜没圆柱辐射问题收敛性复核：128 面元附加质量相对误差 8.513e-3、阻尼 1.336e-2、"
        "近/远场一致性 5.34e-3，均 <3% PASS。发现 README 声明值 7.7e-3 与实测 8.513e-3 偏差 10.6% "
        "（honest-FAIL：文档数字来自历史 commit 878834a，后续代码重构未更新文档）。"
        "收敛阶趋近 O(h)，与常数面元+对数核奇异性理论预期一致。"
    ),
    "scenario_key": "report-backfill",
    "mission_id": 2198,
    "tags": ["补报", "TensorBEM", "Ursell", "BEM", "convergence", "benchmark", "honest-fail"],
    "markdown": report_md,
    "figures": figures,
}

result = post("/tensor/reports", payload, token)
print(f"Report posted: {json.dumps(result, indent=2)}")
