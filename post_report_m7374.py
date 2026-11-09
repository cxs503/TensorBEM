import base64, json, os, urllib.request

base = os.environ["TENSOR_API_BASE"].rstrip("/")
data = json.dumps({
    "username": os.environ["TENSOR_AGENT_USER"],
    "password": os.environ["TENSOR_AGENT_PASS"],
}).encode()
req = urllib.request.Request(f"{base}/auth/login", data=data,
                             headers={"Content-Type": "application/json"})
token = json.loads(urllib.request.urlopen(req).read())["access_token"]

figdir = "artifacts/cylinder_drift_asymptotic"
figs = []
for name, cap in [
    ("fig_long_wave_law.png",
     "Exact drift force normalised as F/(rho g (ka)^3) versus the closed form "
     "5*pi^2/16 = 3.08425; slope fit 2.977 over ka in [0.02,0.30]."),
    ("fig_near_far_energy.png",
     "Left: near-field (wall pressure + waterline) and far-field (reduced Maruo "
     "kernel) drift force, indistinguishable to 2.8e-15. Right: Kochin energy-balance "
     "residual for nine (ka, R) combinations, all ~1e-15 against the 1 % gate."),
]:
    with open(os.path.join(figdir, name), "rb") as fh:
        figs.append({"name": name, "caption": cap,
                     "png_base64": base64.b64encode(fh.read()).decode()})

markdown = open("docs/cylinder_drift_asymptotic.md").read()
extra = """

---

## 7. Audit chain and machine-readable verdict

* driver: `src/tensorbem/benchmarks/cylinder_drift_asymptotic.py`
* evidence: `artifacts/cylinder_drift_asymptotic/study.json`
  (`verdict.all_gates_pass = true`)
* tests: `tests/test_cylinder_drift_asymptotic.py` — 29 passed
* module: `src/tensorbem/cylinder_drift.py`
* the m5242 honest FAIL (`cylinder (ka)^3 asymptotic gate not met`) is left
  untouched in `benchmarks/drift_force_study.py`; this node supplies a different,
  radius-free route and its own gates.

## 8. Not delivered (honest)

Wigley III first-order wave-exciting force and heave/pitch RAO against towing-tank
experiments at six encounter frequencies is **not** claimed: no digitised
experimental data set exists in the repository and reproducing numbers from memory
would be fabrication. The second-order drift half of the brief, with analytic
double verification, is delivered.
"""
report = {
    "title": "TensorBEM: exact cylinder mean drift force — near/far double verification, closed-form (ka)^3 law, Kochin energy balance",
    "summary": (
        "Exact MacCamy-Fuchs double verification of the cylinder mean drift force: "
        "near-field (wall pressure + waterline) and far-field (reduced Maruo kernel) "
        "agree to 2.8e-15 over ka in [0.2,2.0]; closed-form (ka)^3 law 5*pi^2/16 "
        "matched to 1.2e-4 (fitted slope 2.977); Kochin energy residual 9.6e-15 vs a "
        "1% gate; 29 tests green. Wigley first-order RAO vs experiment not delivered "
        "(no digitised data in-tree)."
    ),
    "scenario_key": "tensorbem_cylinder_drift_asymptotic",
    "mission_id": "7374",
    "tags": ["TensorBEM", "second-order", "drift-force", "Maruo", "MacCamy-Fuchs",
             "Kochin", "verification", "p41w3-n01"],
    "markdown": markdown + extra,
    "figures": figs,
}
req = urllib.request.Request(f"{base}/tensor/reports",
                             data=json.dumps(report).encode(),
                             headers={"Content-Type": "application/json",
                                      "Authorization": f"Bearer {token}"})
try:
    with urllib.request.urlopen(req) as resp:
        print(resp.status, resp.read().decode()[:400])
except urllib.error.HTTPError as exc:
    print("HTTP", exc.code, exc.read().decode()[:800])