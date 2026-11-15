"""Publish the m21100 (CASE-BEM-CUMMINS-WIGLEY) verification-debt report."""
from __future__ import annotations
import base64, json, os, pathlib, urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
if not BASE.endswith("/api/v1"):
    BASE = BASE.rstrip("/") + "/api/v1"
ROOT = pathlib.Path(__file__).resolve().parent
ART = ROOT / "benchmarks/verified/casepool_bem_cummins_wigley"


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

md = (ART / "CASE.md").read_text()
figs = []
for name, caption in (
    ("fig_kk_closure.png",
     "Kramers-Kronig closure residual per DOF (Wigley hull, medium 20x6 mesh): "
     "BEM added mass vs KK reconstruction from the damping sweep; all 6 DOF < 10% "
     "case tolerance (surge 2.62%, sway 2.81%, heave 0.30%, roll 0.72%, pitch 0.48%, yaw 3.60%)"),
):
    png = (ART / name).read_bytes()
    figs.append({"name": name, "caption": caption,
                 "png_base64": base64.b64encode(png).decode()})

summary = (
    "CASE-BEM-CUMMINS-WIGLEY verification-debt fix: the committed case_result.json "
    "was published from an older sweep whose KK residual could not be certified "
    "(worst=inf) and whose mesh_convergence used an older schema, while the committed "
    "sweep_medium.npz/sweep_coarse.npz were refreshed to a converged CHIEF-regularised "
    "solve (C3=0.036, inside the 10% case tolerance).  The two were out of sync and "
    "test_production_artifact_replays_to_the_published_verdict failed.  Since every "
    "gate is a function of the tabulated coefficients + hull constants, the verdict "
    "was regenerated from the committed .npz without re-solving.  Result: 9/10 gates "
    "pass (C1,C0,C0_coarse,C2,C3,C3_coarse,C4,C6a,C6b); C5 (free-decay vs "
    "frequency-domain zeta) FAIL — natural frequency outside the computed band.  "
    "KK closure per DOF all < 10%."
)
payload = {
    "title": "CASE-BEM-CUMMINS-WIGLEY: 发布判据与已提交扫频工件失同步的验证欠账修复（Wigley 6DOF Cummins 记忆核）",
    "summary": summary,
    "scenario_key": "case_pool/bem/cummins_wigley",
    "mission_id": 21100,
    "tags": ["TensorBEM", "算例池", "Cummins", "Wigley", "Kramers-Kronig", "6DOF", "verification-debt"],
    "markdown": md,
    "figures": figs,
}
try:
    resp = post("/tensor/reports", payload, token)
    print("POSTED", json.dumps(resp)[:300])
except Exception as exc:
    print("post failed", exc)
    raise
