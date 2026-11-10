"""m8799: publish paper-level progress report (PR#574 reopen round 2, non-conformal geometry)."""
from __future__ import annotations

import base64
import json
import os
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ.get("TENSOR_AGENT_USER")
PASS = os.environ.get("TENSOR_AGENT_PASS")
ROOT = "/root/tp-agent/work/m8799"


def post(path, payload, token=None):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path, data=data, method="POST",
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})},
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.loads(response.read().decode())


login = post("/auth/login", {"username": USER, "password": PASS})
token = login.get("access_token") or login.get("token")
assert token, f"no token: {sorted(login)}"

md = open(f"{ROOT}/dev_report_m8799_quadratic_panel_pr574_reopen_round2.md").read()
study = json.load(open(f"{ROOT}/docs/quadratic_panel_verification/study.json"))
sh = [g for g in study["geometry"]["geometries"]
      if g["name"].startswith("sheared")][0]
finest = float(sh["flux_rel_error_finest"])

figures = []
for name, path, caption in [
    ("fig_geometry_generic", f"{ROOT}/docs/quadratic_panel_verification/fig_geometry_generic.png",
     "图1 几何泛化门：(a) 球/椭球/剪切椭球（非相似仿射）通量收敛 vs 解析解，剪切 finest "
     f"{finest:.3e} < 3% 验收线，观测阶 {float(sh['observed_order']):.2f}；(b) 薄体实测适用域上限（非门）"),
    ("fig_quadratic_operator", f"{ROOT}/docs/quadratic_panel_verification/fig_quadratic_operator.png",
     "图2 场量云图 + 算子恒等式收敛（既有基线，本 PR 未改动，字节级不变）：最细级通量误差 6.065e-03 < 3%"),
]:
    figures.append({
        "name": name,
        "caption": caption,
        "png_base64": base64.b64encode(open(path, "rb").read()).decode(),
    })

payload = {
    "title": "TensorBEM 二次曲面算子：非相似（仿射剪切）几何泛化门（PR#574 重开·管到底第2轮·m8799）",
    "summary": ("几何适用域从「相似群像 + 对角轴缩放」推进到非相似仿射剪切（det=1）：剪切椭球 finest "
                f"通量误差 {finest:.3e}（0.77% < 3%），观测阶 {float(sh['observed_order']):.2f}；"
                "新增研究用例 + 3 条回归钉（含非真空性守卫），既有标量 0 漂移。"),
    "scenario_key": "tensorbem.quadratic_panel.nonconformal_geometry",
    "mission_id": 8799,
    "tags": ["TensorBEM", "quadratic-panel", "BEM", "geometry-generality", "affine-shear", "pr574-reopen"],
    "markdown": md,
    "figures": figures,
}
resp = post("/tensor/reports", payload, token)
print("report posted:", json.dumps(resp, ensure_ascii=False)[:300])
