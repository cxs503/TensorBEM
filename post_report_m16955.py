"""Publish the m16955 report (numbers injected from evidence.json)."""
from __future__ import annotations

import base64
import json
import pathlib
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:8010/api/v1"
ROOT = pathlib.Path(__file__).resolve().parent
E = json.loads((ROOT / "docs/m16955_steady_kernel/evidence.json").read_text())
DOC = (ROOT / "docs/m16955_steady_kernel/report.md").read_text()
PNG = (ROOT / "docs/fig_m16955_steady_kernel.png").read_bytes()


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


user = env.get("TENSOR_AGENT_USER") or env.get("TENSOR_USER") or "agent"
pw = env.get("TENSOR_AGENT_PASS") or env.get("TENSOR_PASS") or ""
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

fig = base64.b64encode(PNG).decode()
payload = {
    "title": "稳态 Neumann-Kelvin 核函数：正确表述、精确 k 积分与实测状态（m16955）",
    "summary": (
        "TensorBEM: 重建稳态(NK)核函数表述——旧核是频域 Havelock 径向核因而结构上无法满足 "
        "K G_z+G_xx=0；给出 Kelvin 曲线族谱积分 + 奇异 k 积分的精确闭式（外部求积校验 1e-15/最差 1.4e-7）；"
        "候选核自由面残差仍 0.7%-38% 且上下游对称，能力状态维持 WITHHELD（honest FAIL）。"
    ),
    "scenario_key": "tensorbem_steady_kelvin_kernel",
    "mission_id": 16955,
    "tags": ["TensorBEM", "Neumann-Kelvin", "steady-kernel", "verification-debt", "honest-FAIL"],
    "markdown": DOC,
    "figures": [{
        "name": "fig_m16955_steady_kernel.png",
        "caption": "左：旧核与候选核的稳态自由面相对残差（K=1，源 z=-1，判据与仓内 referee 一致）；右：候选核波部 |W| 上下游对称，辐射条件未实现。",
        "png_base64": fig,
    }],
}
try:
    out = _post("/tensor/reports", payload, token=token)
    print("report id:", out.get("id") or out.get("data", {}).get("id") or out)
except Exception as exc:  # noqa: BLE001
    print("POST failed:", exc)
