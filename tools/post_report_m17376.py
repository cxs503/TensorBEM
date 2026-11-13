"""Publish the m17376 (CASE-BEM-RIGID-BODY-MASS) report."""
from __future__ import annotations

import base64
import json
import os
import pathlib
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
if not BASE.endswith("/api/v1"):
    BASE = BASE.rstrip("/") + "/api/v1"
ROOT = pathlib.Path(__file__).resolve().parents[1]
CASE = json.loads((ROOT / "runs/m17376_rigid_body_mass/results.json").read_text())
PRIMARY = CASE["primary"]
RAO = CASE.get("rao_integration", {})
ELL = CASE["ellipsoid"]["ladder"]


def post(path, payload, token=None):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        },
    )
    return json.loads(urllib.request.urlopen(req, timeout=180).read())


token = None
try:
    r = post("/auth/login", {
        "username": os.environ.get("TENSOR_AGENT_USER", "agent"),
        "password": os.environ.get("TENSOR_AGENT_PASS", ""),
    })
    token = r.get("access_token") or r.get("token")
except Exception as exc:
    print("login failed", exc)

md = (ROOT / "docs/case_rigid_body_mass.md").read_text()
figs = [{
    "name": "rigid_body_mass_error.png",
    "caption": (
        "left: 6x6 entry-wise relative error of the shipped rigid-body mass matrix "
        f"(max {PRIMARY['max_rel_error']:.1e}); right: CG-offset ladder - shipped error at "
        f"round-off vs pre-fix transposed coupling block at {PRIMARY['control_max_rel_error']*100:.0f}%"
    ),
    "png_base64": base64.b64encode(
        (ROOT / "runs/m17376_rigid_body_mass/rigid_body_mass_error.png").read_bytes()
    ).decode(),
}]

summary = (
    "TensorBEM rigid-body 6x6 mass matrix: found and fixed a transposed coupling block "
    "(-m*S^T instead of -m*S, Fossen 2011 eq. 3.51) that symmetry/definiteness tests could not see; "
    f"kinetic-energy identity 2.36e-1 -> {PRIMARY['kinetic_energy_error']:.1e}, RAO roll amplitude on a "
    f"108-panel barge moves {RAO.get('max_rao_change', 0)*100:.1f}%; benchmark vs the closed-form block form "
    f"passes all 36 entries at {PRIMARY['max_rel_error']:.1e} (criterion 3%), pre-fix control fails at "
    f"{PRIMARY['control_max_rel_error']*100:.0f}%, referee ellipsoid ladder reproduces 0.2161%/0.3552%/5.50%"
)

payload = {
    "title": "CASE-BEM-RIGID-BODY-MASS: 刚体质量矩阵闭式块形式校验（TensorBEM）——定位并修复转置耦合块缺陷",
    "summary": summary,
    "scenario_key": "case_pool/bem/rigid_body_mass",
    "mission_id": 17376,
    "tags": ["TensorBEM", "算例池", "rigid body mass matrix", "Fossen 3.3.1", "6-DOF", "defect fix",
             "kinetic energy identity", "RAO"],
    "markdown": md,
    "figures": figs,
}
resp = post("/tensor/reports", payload, token)
print("POSTED", json.dumps(resp)[:300])
print("ellipsoid ladder:", [(r["panels"], r["max_rel_error"]) for r in ELL])