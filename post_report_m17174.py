"""Publish the m17174 report: 2-D circular-cylinder added-mass benchmark (CASE-BEM-CYL-ADDED-MASS-2D)."""
from __future__ import annotations

import base64
import json
import pathlib
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:8010/api/v1"
ROOT = pathlib.Path(__file__).resolve().parent

DOC = (ROOT / "docs" / "dev_report_m17174_cylinder_added_mass.md").read_text()
PNG = ROOT / "artifacts" / "fig_cylinder_added_mass.png"


def _env() -> dict:
    out = {}
    p = pathlib.Path("/root/tp-agent/agent.env")
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


env = _env()


def _post(path: str, payload=None, token=None, form=False) -> dict:
    data = None
    headers = {}
    if payload is not None:
        if form:
            data = urllib.parse.urlencode(payload).encode()
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        else:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(BASE + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


user = env.get("TENSOR_AGENT_USER") or "eco_admin"
pw = env.get("TENSOR_AGENT_PASS") or ""
token = None
for path, form in (("/auth/login", False), ("/auth/login", True), ("/auth/token", False)):
    try:
        out = _post(path, {"username": user, "password": pw}, form=form)
        token = out.get("access_token") or out.get("token") or (out.get("data") or {}).get("access_token")
    except Exception as exc:  # noqa: BLE001
        print("auth", path, "failed:", exc)
    if token:
        break
print("token:", "ok" if token else "MISSING")

payload = {
    "title": "二维圆柱附加质量基准（CASE-BEM-CYL-ADDED-MASS-2D）：ρπa² 闭式验证 m17174",
    "summary": (
        "TensorBEM 二维平移圆截面圆柱附加质量基准：复用 cylinder_added_mass 能力，"
        "ref=ρπa²=3220.1325（Lamb/Newman 闭式，机器求值零手抄）。5 面元数收敛 "
        "2.62e-2→1.00e-4（16→256），全部 <3%，64 面元起 <0.2%；同族椭圆参数敏感性 "
        "7.5e-4/9.0e-4；门禁 test_cylinder_benchmark 2 通过。"
    ),
    "scenario_key": "CASE-BEM-CYL-ADDED-MASS-2D",
    "mission_id": 17174,
    "tags": ["TensorBEM", "benchmark-pool", "added-mass", "2D-cylinder", "potential-flow",
             "verification"],
    "markdown": DOC,
    "figures": [{
        "name": "fig_cylinder_added_mass.png",
        "caption": "二维圆柱附加质量收敛：(a) 相对误差 vs 面元数（对数轴，3% 容差虚线）；(b) 计算值 vs 解析参考 ρπa²=3220.1325。数字由 artifacts/refinement_cylinder.json 注入。",
        "png_base64": base64.b64encode(PNG.read_bytes()).decode(),
    }],
}
try:
    out = _post("/tensor/reports", payload, token=token)
    print("report id:", out.get("id") or out.get("data", {}).get("id") or out)
except Exception as exc:  # noqa: BLE001
    print("POST failed:", exc)
