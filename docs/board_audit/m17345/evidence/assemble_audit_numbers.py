"""m17345 audit: consolidate m17278 delivered numbers + independent 1024-panel level,
compute observed convergence orders / Richardson extrapolation, and emit figures.

Inputs (all inside this audit tree):
  m17278_raw/results_reduced.json  - delivered 64/256-panel ladder (unchanged copy)
  p1024_*.json, hulme_1024_probe.json - 1024-panel level re-run by this audit
Outputs: audit_numbers.json, figures + figures_digest.json
"""
import json, math, glob, hashlib
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE
delivered = json.load(open(HERE / "m17278_raw/results_reduced.json"))
ladder = {}
REFS = {c["ka"]: (c["surge"]["ref"], c["heave"]["ref"]) for c in delivered["comparison"]}
for r in delivered["full_results"]:
    rs, rh = REFS[r["ka"]]
    ladder.setdefault(r["ka"], {})[r["panels"]] = (r["rel_error_surge"], r["rel_error_heave"],
                                                   r["F_surge_mag"], r["F_heave_mag"], rs, rh)
p1024 = {}
for f in glob.glob(str(HERE / "p1024_*.json")) + [str(HERE / "hulme_1024_probe.json")]:
    for r in json.load(open(f)):
        p1024[r["ka"]] = r
for ka, r in p1024.items():
    if "Fx" in r:                      # repro_1024.py shape: {"Fx": {"ref","got","err"}}
        fx, fz = r["Fx"], r["Fz"]
    else:                              # p1024.py shape: flat keys
        fx = {"ref": r["Fx_ref"], "got": r["Fx_got"], "err": r["Fx_err"]}
        fz = {"ref": r["Fz_ref"], "got": r["Fz_got"], "err": r["Fz_err"]}
    ladder.setdefault(ka, {})[1024] = (fx["err"], fz["err"], fx["got"], fz["got"], fx["ref"], fz["ref"])

rows, orders = [], []
for ka in sorted(ladder):
    panels = sorted(ladder[ka])
    row = {"ka": ka, "levels": {}}
    for p in panels:
        es, eh, gs, gh, rs, rh = ladder[ka][p]
        row["levels"][p] = {"err_surge": es, "err_heave": eh, "got_surge": gs, "got_heave": gh,
                            "ref_surge": rs, "ref_heave": rh}
    if 64 in panels and 256 in panels and 1024 in panels:
        e = [abs(ladder[ka][p][0]) for p in (64, 256, 1024)]
        p_obs = [math.log(e[i] / e[i + 1], 2) for i in range(2)]   # h halves on each 4x panel step
        rich = e[2] + (e[2] - e[1]) / 3.0
        row["surge_observed_order"] = p_obs
        row["surge_richardson_limit_abs_err"] = rich
        orders.append((ka, p_obs, rich))
    rows.append(row)

worst_256 = max(abs(r["levels"][256]["err_surge"]) for r in rows if 256 in r["levels"])
worst_1024 = max(max(abs(r["levels"][1024]["err_surge"]), abs(r["levels"][1024]["err_heave"]))
                 for r in rows if 1024 in r["levels"])
summary = {
    "case": "CASE-BEM-HULME-DIFF-VERIF / CASE-BEM-HULME-HEMISPHERE (TensorBEM 3-D diffraction)",
    "delivered_by_m17278": {"ka_grid": delivered.get("comparison") and [c["ka"] for c in delivered["comparison"]],
                            "panels": [64, 256], "tolerance": delivered["tolerance"],
                            "all_pass": delivered["all_pass"], "worst_abs_err_256": worst_256},
    "added_by_audit_m17345": {"panels": 1024, "ka_covered": sorted(p1024),
                              "worst_abs_err_1024": worst_1024,
                              "note": "m17278 doc claimed this level was 'submitted to cluster cj1476'; "
                                      "cj1476 was still queued 79 min after submission and never delivered"},
    "rows": rows,
    "reproduction": {"delivered_results_reduced.json": "re-run byte-identical (assembled via diff of "
                     "sorted-key JSON dumps) in work/m17345 on 2026-11-13"},
}
json.dump(summary, open(OUT / "audit_numbers.json", "w"), indent=1)

# ---- figure 1: |rel err| vs panels, log-log, per ka ----
RHO, G, A_R = 1025.0, 9.80665, 1.0
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (idx, label) in zip(axes, [(0, "surge |Fx|"), (1, "heave |Fz|")]):
    for r in rows:
        ps = sorted(r["levels"])
        key = "err_surge" if idx == 0 else "err_heave"
        ax.plot(ps, [abs(r["levels"][p][key]) * 100 for p in ps], "o-", lw=1.2, ms=4,\
                label=f"ka={r['ka']:g}")
    ax.axhline(5, color="k", ls="--", lw=1, label="5% task tolerance")
    ax.axhline(3, color="grey", ls=":", lw=1, label="3% ecosystem line")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("panels"); ax.set_ylabel("|relative error| (%)")
    ax.set_title(f"{label}: BEM vs Hulme multipole series")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7, ncol=2)
fig.tight_layout(); f1 = OUT / "fig_audit_error_vs_panels.png"; fig.savefig(f1, dpi=150); plt.close(fig)

# ---- figure 2: dimensionless excitation force vs ka, 256 & 1024 panels ----
norm = RHO * G * A_R * A_R
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
for ax, (idx, refkey, gotkey, label) in zip(
        axes2, [(0, "ref_surge", "got_surge", "surge |Fx| / (rho g A a^2)"),
                (1, "ref_heave", "got_heave", "heave |Fz| / (rho g A a^2)")]):
    kas = sorted(r["ka"] for r in rows if 1024 in r["levels"])
    refv = [next(r["levels"][1024][refkey] for r in rows if r["ka"] == ka) / norm for ka in kas]
    ax.plot(kas, refv, "k-", lw=2, label="Hulme multipole series (ref)")
    for p, sty in ((256, "bs"), (1024, "ro")):
        ks = sorted(r["ka"] for r in rows if p in r["levels"])
        gv = [next(r["levels"][p][gotkey] for r in rows if r["ka"] == ka) / norm for ka in ks]
        ax.plot(ks, gv, sty, ms=6, label=f"TensorBEM BEM, {p} panels")
    ax.set_xlabel("ka"); ax.set_ylabel(label); ax.set_title(label.split(":")[0])
    ax.grid(True, alpha=0.3); ax.legend()
fig2.tight_layout(); f2 = OUT / "fig_audit_excitation_dimensionless.png"; fig2.savefig(f2, dpi=150); plt.close(fig2)

digest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (f1, f2)}
digest["audit_numbers.json"] = hashlib.sha256((OUT / "audit_numbers.json").read_bytes()).hexdigest()
json.dump(digest, open(OUT / "figures_digest.json", "w"), indent=1)
print(json.dumps({"worst_256": round(worst_256, 5), "worst_1024": round(worst_1024, 5),
                  "orders": [(o[0], [round(x, 3) for x in o[1]], "%.2e" % o[2]) for o in orders]}, indent=1))
print(json.dumps(digest, indent=1))
