"""m8799: push branch, open PR, report pr-gate open (PR#574 reopen round 2)."""
import json
import os
import subprocess
import urllib.error
import urllib.request

line = [l for l in open("/root/.git-credentials") if "github.com" in l][0].strip()
assert line.startswith("https://")
auth = line[len("https://"):].split("@", 1)[0]
user, token = auth.split(":", 1)
print("cred user:", user)

# --- 1) push branch ---
r = subprocess.run(
    ["git", "push", "-u", "origin", "agent/m8799"],
    capture_output=True, text=True, cwd="/root/tp-agent/work/m8799",
)
print("push rc:", r.returncode)
if r.returncode != 0:
    print(r.stdout[-800:]); print(r.stderr[-800:])
    raise SystemExit("push failed")


def gh(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        "https://api.github.com" + path, data=data, method=method,
        headers={"Authorization": "Bearer " + token,
                 "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


s, existing = gh("GET", "/repos/cxs503/TensorBEM/pulls?state=open&per_page=100")
pr = None
for p in (existing if isinstance(existing, list) else []):
    if p["head"]["ref"] == "agent/m8799":
        pr = p
        break

BODY = """Reopen of TensorBEM#574 (m8282 quadratic operator deliverables), see-through round 2 (m8799).

PR#574 functional points are all on main (PR#550/m8282 + PR#558/m8616 + PR#569/m8709 + the
invariance pins of m8723/m8745/m8769/m8788). This PR adds one focused increment: the geometry
generality gate leaves the similarity group.

1. New gated study case `sheared_ellipsoid_1p00_1p60_0p70`: the ellipsoid (1.0, 1.6, 0.7) is
   additionally sheared by a unit-determinant non-orthogonal matrix. The diagonal axis scaling
   of the existing ellipsoid case keeps the parametric axes orthogonal (elements stay near
   equilateral, metric anisotropy 1.85 -> 1.26 over the levels); a shear makes the isoparametric
   mapping non-orthogonal, i.e. the elements genuinely skewed (anisotropy 2.25 -> 1.37), while
   det A = 1 keeps the scale and x0 = 0 (shear fixed point) keeps the source depth - so the case
   measures the Jacobian/normal evaluation on skewed elements, not the geometry representation
   (quadratic Bezier triangles are affine-covariant) and not field sharpness.
   Result: finest-level flux error 7.668e-03 (0.77% < 3%), observed order 2.72 (>= 1.5 floor).
   That is a ~19% constant-factor degradation against the ellipsoid (6.426e-03) with the same
   order - element skew costs accuracy in the pre-asymptotic constant, not the rate.

2. `element_metric_anisotropy()` diagnostic recorded per level as `jacobian_anisotropy_max`, plus
   an `affine_transform` field per geometry record, so the non-conformal case cannot silently
   degenerate into a rescaled conformal copy and accuracy is readable against element quality.

3. Three new pins in tests/test_panel_quadratic_operator.py: the sheared-body flux stays under
   the 15% engineering limit at sub 1 with a valid solid-angle identity; the shear case is not a
   similarity copy (S and D relative Frobenius difference > 5%, anisotropy > 1.15x ellipsoid);
   the committed study.json carries the case with det A = 1 and both acceptance gates passing.

No change to src/tensorbem/panel_quadratic.py (core module untouched, 0 lines).

Regression safety: re-running the committed benchmark reproduces every pre-existing scalar of
study.json exactly (single changed scalar = provenance sha256 of the regenerated geometry figure),
and fig_quadratic_operator.png is byte-identical.

Tests: full suite run; only pre-existing main failure
test_bem_finite_depth_anchors::test_case_artifacts_and_documentation_exist (benchmarks/verified/
bem_finite_depth/figures/ never committed to main, reproduced on clean main, unrelated).

Open PRs from other agents touching these paths: none at the time of opening (checked via API).

See dev_report_m8799_quadratic_panel_pr574_reopen_round2.md.
"""

if pr is None:
    s, pr = gh("POST", "/repos/cxs503/TensorBEM/pulls", {
        "title": "m8799: quadratic panel PR#574 reopen round 2 - non-conformal (affine shear) geometry gate",
        "head": "agent/m8799", "base": "main", "body": BODY,
    })
    print("PR create status:", s)
print("PR:", pr.get("number"), pr.get("html_url"))
prnum = pr.get("number")
if not prnum:
    raise SystemExit("no PR number")

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
lg = urllib.request.Request(
    BASE + "/auth/login",
    data=json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                     "password": os.environ["TENSOR_AGENT_PASS"]}).encode(),
    method="POST", headers={"Content-Type": "application/json"})
with urllib.request.urlopen(lg, timeout=60) as resp:
    TTOKEN = json.loads(resp.read().decode()).get("access_token")

req = urllib.request.Request(
    BASE + "/tensor/missions/8799/pr-gate",
    data=json.dumps({"state": "open", "repo": "TensorBEM", "pr": prnum,
                     "note": "PR#%d opened from agent/m8799 (PR#574 reopen round 2)" % prnum}).encode(),
    method="POST",
    headers={"Authorization": "Bearer " + TTOKEN, "Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        out = json.loads(resp.read().decode())
    print("pr-gate open ->", resp.status, json.dumps(out, ensure_ascii=False)[:400])
except urllib.error.HTTPError as e:
    print("pr-gate HTTP", e.code, e.read().decode()[:600])

print("DONE pr=", prnum)
