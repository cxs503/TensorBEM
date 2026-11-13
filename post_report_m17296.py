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
     "5*pi^2/16 = 3.08425; slope fit 2.977 over ka in [0.02,0.30]. "
     "Return-round L4 ref-vs-got benchmark."),
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

## Return-round 1 (m17296) — maturity-gate closure

The original delivery (m7374) was re-adjudicated at L1 only: `impl_files`, `entry`
and `run_artifacts` were flagged missing because the **run artifacts were never
actually produced** (the `artifacts/` tree was absent). This round closes that gap
with real, machine-checkable evidence.

### L2 — independent function (delivered)
* **impl_files (non-placeholder)**: `src/tensorbem/cylinder_drift.py` (exact
  MacCamy-Fuchs reference field, near/far drift formulas, pure-NumPy Bessel
  series) + `src/tensorbem/benchmarks/cylinder_drift_asymptotic.py` (driver).
* **entry**: `main()` in the benchmark module (CLI `--json/--figures`) plus
  `repro_cylinder_drift_asymptotic.py` (one-command repro).
* **run_artifacts (real)**: `artifacts/cylinder_drift_asymptotic/study.json`
  (8.5 kB, `verdict.all_gates_pass = true`) + two PNG figures + `benchmark.json`.

### L3 — integration callers (delivered)
`src/tensorbem/diffraction_drift_maturity.py` calls `cylinder_drift` from the
main-solver verification surface: `cylinder_asymptotic_gate` (near/far + (ka)^3
law), `cylinder_energy_balance`, `cylinder_bem_case` (discretised BEM vs exact).
Verified live: `cylinder_asymptotic_gate()` returns
`max_near_vs_far_rel_diff = 2.835e-15`, `law_slope = 2.977`,
`coefficient_error_at_ka_min = 1.168e-4`.

### L4 — benchmark + repro_cmd (delivered)
`artifacts/cylinder_drift_asymptotic/benchmark.json` holds the ref-vs-got table:
ref = MacCamy-Fuchs/Maruo closed form `5*pi^2/16 = 3.08425`; got = exact
evaluation. Repro: `PYTHONPATH=src python3 repro_cylinder_drift_asymptotic.py`
(one command, all gates PASS).

### Machine-checkable verdict (from `study.json`)
| gate | target | measured | pass |
|---|---|---|---|
| near vs far, 8 ka in [0.2,2.0] | < 5 % | 2.835e-15 | yes |
| (ka)^3 slope | 3 ± 0.2 | 2.977 | yes |
| coefficient vs 5*pi^2/16 | < 3 % | 1.168e-4 | yes |
| Kochin energy residual | < 1 % | 9.649e-15 | yes |
| wall anchor rho g A^2/2 | exact | 0.0 | yes |
| pytest | >= 15 | 29 passed | yes |

### Not delivered (honest)
Wigley III first-order RAO vs towing-tank experiment remains **not claimed** (no
digitised experimental data set in-tree; typing numbers from memory would be
fabrication). The heavy `bem_diffraction_drift_maturity` full-surface driver
(Wigley RAO + BEM mesh convergence) exceeds the 120 s control-plane budget and is
a cluster_queue candidate; the drift-asymptotic half it aggregates is fully
delivered here.
"""
report = {
    "title": "TensorBEM return-round 1: cylinder mean drift force (ka)^3 law — L2/L3/L4 maturity-gate closure with real run artifacts",
    "summary": (
        "Return-round closure of CASE-BEM-CYL-DRIFT-ASYMP: produced the previously "
        "missing run artifacts (study.json + figures + benchmark.json) by running "
        "the exact MacCamy-Fuchs/Maruo double verification. Near/far drift agree to "
        "2.835e-15; closed-form (ka)^3 law 5*pi^2/16 matched to 1.168e-4 (slope "
        "2.977); Kochin energy residual 9.649e-15; 29 tests green. L3 integration "
        "callers verified live; L4 benchmark + one-command repro delivered."
    ),
    "scenario_key": "tensorbem_cylinder_drift_asymptotic",
    "mission_id": "17296",
    "tags": ["TensorBEM", "second-order", "drift-force", "Maruo", "MacCamy-Fuchs",
             "Kochin", "verification", "maturity-gate", "return-round"],
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
