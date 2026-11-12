"""Publish the m16109 platform report (numbers/figures from worktree artifacts)."""
from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request

ROOT = pathlib.Path("/root/tp-agent/work/m16109")
API = os.environ["TENSOR_API_BASE"]

# login
req = urllib.request.Request(
    f"{API}/auth/login",
    data=json.dumps({
        "username": os.environ["TENSOR_AGENT_USER"],
        "password": os.environ["TENSOR_AGENT_PASS"],
    }).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
token = json.load(urllib.request.urlopen(req))["access_token"]

md = (ROOT / "docs" / "fmm_setup_contract" / "README.md").read_text()

figures = []
for name, caption in [
    ("fig_far_fraction_vs_distance.png",
     "修复后 far_fraction 随靶距单调上升（0→1）；红点=m16000 bench 靶点 d=4（0.261）。诊断恢复真实语义。"),
    ("fig_failopen_vs_failclosed.png",
     "fail-open→fail-closed：未 setup 时 matvec 静默 100% 错答 / far_fraction 静默 0.0 均改为 RuntimeError（斜纹柱=不再出现的静默值）。"),
]:
    png = (ROOT / "docs" / "fmm_setup_contract" / name).read_bytes()
    figures.append({
        "name": name,
        "caption": caption,
        "png_base64": base64.b64encode(png).decode(),
    })

body = {
    "title": "TensorBEM m16109: FMMOperator 未 setup 静默错答（fail-open）→ fail-closed 契约（收编 m16000 局限#2）",
    "summary": (
        "m16000 登记的后续项：far_fraction() 在未 setup 算子上静默返回 0.0，与远场缓存被使用"
        "（2.46 误差）矛盾。根因=FMMOperator 公开入口 fail-open：matvec 未 setup 返回全零"
        "（相对直和误差 1.0）、far_fraction 返回 0.0。修复=新增 _require_setup 守卫，两入口"
        "未 setup 抛 RuntimeError；空靶点合法早退保留；bench 误用（无 setup 调 far_fraction）"
        "修正。红→绿：新测试 4 failed→57 passed（fmm 四文件）。已 setup 路径数值逐位不变"
        "（p=4 相对直和 1.33e-06；m16000 A/B 行一致）。"
    ),
    "scenario_key": "tensorbem_fmm_setup_contract",
    "mission_id": 16109,
    "tags": ["TensorBEM", "FMM", "defect_fix", "api_contract", "fail_closed", "m16000_followup"],
    "markdown": md,
    "figures": figures,
}

req = urllib.request.Request(
    f"{API}/tensor/reports",
    data=json.dumps(body).encode(),
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    },
    method="POST",
)
resp = json.load(urllib.request.urlopen(req))
print(json.dumps(resp, ensure_ascii=False)[:400])
