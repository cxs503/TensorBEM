"""m8788: push branch, open PR, report pr-gate open. Single-shot."""
import json
import subprocess
import urllib.request
import urllib.error

line = [l for l in open("/root/.git-credentials") if "github.com" in l][0].strip()
assert line.startswith("https://")
auth = line[len("https://"):].split("@", 1)[0]
user, token = auth.split(":", 1)
print("cred user:", user)

# --- 1) push branch ---
r = subprocess.run(
    ["git", "push", "-u", "origin", "agent/m8788"],
    capture_output=True, text=True, cwd="/root/tp-agent/work/m8788",
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
    if p["head"]["ref"] == "agent/m8788":
        pr = p
        break
if pr is None:
    s, pr = gh("POST", "/repos/cxs503/TensorBEM/pulls", {
        "title": "m8788: quadratic panel PR#574 reopen - O(3) reflection invariance pin + outward-normal enforcement",
        "head": "agent/m8788", "base": "main",
        "body": (
            "Reopen of TensorBEM#574 (m8282 quadratic operator deliverables), see-through round 1.\n\n"
            "PR#574 functional points are all on main via PR#550 (m8282) + PR#558 (m8616) + "
            "PR#569 (m8709) + m8769 scale pin; this PR adds the single remaining group element:\n\n"
            "1. test_operators_are_reflection_invariant - mirror reflection (det = -1) must leave "
            "S and D entry-wise invariant, completing the full orthogonal group O(3) after the "
            "translation (m8723) / rotation (m8745) / uniform-scale (m8769) pins. All previous pins "
            "use proper transforms (det = +1) and cannot catch a handedness slip in the normal "
            "construction; a reflection exposes it as D -> -D while S stays put.\n\n"
            "2. test_inward_wound_mesh_is_rewound_outward - white-box companion: a mesh wound "
            "entirely inward is re-wound to the outward convention exactly on construction.\n\n"
            "3. Defect fix in src/tensorbem/panel_quadratic.py (core module, declared): the module "
            "docstring promises 'outward unit normals' but the code never enforced it - the "
            "isoparametric normal follows the face winding, so a mirrored copy of a valid mesh "
            "silently flipped D (D 1 = +1/2) and the recovered flux while every identity on the "
            "original mesh still passed. QuadraticSurfaceMesh.__post_init__ now re-winds faces "
            "whose flat normal points toward the vertex centroid (closed star-shaped surfaces; "
            "limitation documented). No-op proof: re-running the committed benchmark reproduces "
            "study.json with 0 scalar diffs and byte-identical figures (sha256 vs provenance).\n\n"
            "Hot-file declaration: open PR #592 (agent/m8745) also touches "
            "tests/test_panel_quadratic_operator.py, but its increment (rotation pin) is already "
            "on main - it is a stale duplicate; this PR appends after the scale pin, no textual "
            "overlap.\n\n"
            "Tests: full suite green except one pre-existing main failure unrelated to this change "
            "(test_bem_finite_depth_anchors::test_case_artifacts_and_documentation_exist - figures/ "
            "dir never committed; reproduced on clean main).\n\n"
            "See dev_report_m8788_quadratic_panel_pr574_reopen.md."
        ),
    })
    print("PR create status:", s)
print("PR:", pr.get("number"), pr.get("html_url"))
