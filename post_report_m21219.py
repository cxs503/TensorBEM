"""Publish the m21219 same-surface capytaine crossval report."""
from __future__ import annotations
import base64, json, os, pathlib, urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
if not BASE.endswith("/api/v1"):
    BASE = BASE.rstrip("/") + "/api/v1"
ROOT = pathlib.Path(__file__).resolve().parent
DOC = ROOT / "docs/capytaine_same_surface_crossval.md"
FIG = ROOT / "docs/fig_capytaine_same_surface.png"


def post(path, payload, token=None):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})})
    return json.loads(urllib.request.urlopen(req, timeout=180).read())


token = None
try:
    token = post("/auth/login", {
        "username": os.environ.get("TENSOR_AGENT_USER", "agent"),
        "password": os.environ.get("TENSOR_AGENT_PASS", ""),
    }).get("access_token")
except Exception as exc:
    print("login failed", exc)

figs = [{
    "name": FIG.name,
    "caption": "同面交叉验证：TensorBEM 解在 capytaine 自带面元表面上（实线）vs round 2 各自网格（虚线），"
               "最坏对角 / 耦合相对偏差随面元数的变化；点线 5% 对角带、实线 15% ledger 带。",
    "png_base64": base64.b64encode(FIG.read_bytes()).decode(),
}]
payload = {
    "title": "capytaine 同面交叉验证：TensorBEM 6×6 辐射矩阵（XVAL-BEM-R2 增量）",
    "summary": "把 TensorBEM 解在 capytaine 自带面元表面上，去掉两套网格的混淆项：半球两中心的同面对角偏差 "
               "14.38% → 9.48% 随细分单调下降（离散误差签名，非公式分歧）；已提交 capytaine 参考列可重解复现到 "
               "1.9e-16；Richardson 外推极限在这些阶梯上不可辨识（89% 阶落在搜索边界），故不放宽 5% 容差。",
    "scenario_key": "bem_crossvalidation",
    "mission_id": 21219,
    "tags": ["TensorBEM", "capytaine", "cross-validation", "added-mass", "XVAL-BEM-R2"],
    "markdown": DOC.read_text(),
    "figures": figs,
}
try:
    out = post("/tensor/reports", payload, token)
    print("report id", out.get("id") or out.get("report_id") or out)
except Exception as exc:
    print("post failed", exc)
