"""Publish the m17242 (CASE-BEM-SUBMERGED-SPHERE-HAVELOCK) report."""
from __future__ import annotations
import base64, json, os, pathlib, urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
if not BASE.endswith("/api/v1"):
    BASE = BASE.rstrip("/") + "/api/v1"
ROOT = pathlib.Path(__file__).resolve().parent
CASE = json.loads((ROOT / "runs/m17242/case_havelock_submerged_sphere.json").read_text())


def post(path, payload, token=None):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


token = None
try:
    body = {"username": os.environ.get("TENSOR_AGENT_USER", "agent"),
            "password": os.environ.get("TENSOR_AGENT_PASS", "")}
    r = post("/auth/login", body)
    token = r.get("access_token") or r.get("token")
except Exception as exc:  # pragma: no cover
    print("login failed", exc)

md = (ROOT / "docs/case_submerged_sphere_havelock.md").read_text()
figs = []
for name, caption in (
    ("case_havelock_field.png", "Heave radiation potential on the sphere surface (ka=1, 320 panels)"),
    ("case_havelock_sweep.png", "A33'/B33'/A11'/B11' vs ka: BEM 320 panels vs exact multipole"),
    ("case_havelock_refinement.png", "Grid refinement: relative error vs mean panel edge h"),
    ("case_havelock_reference_integrity.png", "Reference audit: f/a -> 0.5 (Lamb) and (a/f)^3 decay"),
):
    png = (ROOT / "docs/figures" / name).read_bytes()
    figs.append({"name": name, "caption": caption,
                 "png_base64": base64.b64encode(png).decode()})

prod = CASE["judgement"]["production_mesh"]
summary = (
    "Submerged-sphere frequency-domain radiation (TensorBEM) vs closed-form multipole reference: "
    f"production 320-panel mesh FAILS the pre-declared 5pct band on damping ({prod['n_failures']}/{prod['n_items']} items), "
    "added mass passes (<=2.3 pct); 1280-panel mesh passes all items (<=2.0 pct)"
)
payload = {
    "title": "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK: 全潜球频域辐射算例（TensorBEM）——320 面元阻尼 5% 判据诚实 FAIL",
    "summary": summary,
    "scenario_key": "case_pool/bem/submerged_sphere_havelock",
    "mission_id": 17242,
    "tags": ["TensorBEM", "算例池", "Havelock", "submerged sphere", "radiation", "added mass", "damping"],
    "markdown": md,
    "figures": figs,
}
try:
    resp = post("/tensor/reports", payload, token)
    print("POSTED", json.dumps(resp)[:300])
except Exception as exc:
    print("post failed", exc)
    raise
