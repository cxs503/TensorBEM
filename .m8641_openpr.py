"""m8641: push branch, open PR, report pr-gate open. Single-shot."""
import json
import subprocess
import urllib.request
import urllib.error

# --- parse github credential (https://user:token@github.com) ---
line = [l for l in open("/root/.git-credentials") if "github.com" in l][0].strip()
assert line.startswith("https://")
auth = line[len("https://"):].split("@", 1)[0]
user, token = auth.split(":", 1)
print("cred user:", user)

# --- 1) push branch ---
r = subprocess.run(
    ["git", "push", "-u", "origin", "agent/m8641"],
    capture_output=True, text=True, cwd="/root/tp-agent/work/m8641",
)
print("push rc:", r.returncode)
if r.returncode != 0:
    print(r.stdout[-800:]); print(r.stderr[-800:])
    raise SystemExit("push failed")

# --- 2) open PR (idempotent: reuse if a PR from this head already exists) ---
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
    if p["head"]["ref"] == "agent/m8641":
        pr = p
        break
if pr is None:
    s, pr = gh("POST", "/repos/cxs503/TensorBEM/pulls", {
        "title": "m8641: quadratic panel PR#533 reopen - verify deliverables on main, add non-spherical solid-angle regression pin",
        "head": "agent/m8641", "base": "main",
        "body": ("Reopen of TensorBEM#533 (m7526 quadratic panel verification), see-through round 1.\n\n"
                 "PR#533 functional points are all on main via PR#550 (m8282) + PR#558 (m8616); "
                 "51/51 panel tests pass, finest-mesh flux error 0.61% < 3%. This PR adds "
                 "test_ellipsoid_solid_angle_identity (D 1 = -1/2 on the affine-image ellipsoid, "
                 "off-sphere) and the verification report."),
    })
    if s != 201:
        print("PR open failed:", s, json.dumps(pr)[:800])
        raise SystemExit(1)
print("PR number:", pr["number"], "state:", pr["state"], pr["html_url"])
prnum = pr["number"]

# --- 3) pr-gate open -> awaiting_pr_merge ---
import os
BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
TTOKEN = open("/tmp/m8641_token").read().strip()
req = urllib.request.Request(
    BASE + "/tensor/missions/8641/pr-gate",
    data=json.dumps({"state": "open", "repo": "TensorBEM", "pr": prnum,
                     "note": "PR#%d opened from agent/m8641 (PR#533 reopen round 1)" % prnum}).encode(),
    method="POST",
    headers={"Authorization": "Bearer " + TTOKEN, "Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        out = json.loads(resp.read().decode())
    print("pr-gate open ->", resp.status)
    print(json.dumps(out, ensure_ascii=False)[:600])
except urllib.error.HTTPError as e:
    print("pr-gate HTTP", e.code, e.read().decode()[:600])
    raise SystemExit(1)
print("DONE pr=", prnum)
