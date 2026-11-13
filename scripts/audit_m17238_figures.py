"""m17238 audit figures: independently recomputed Rw vs audited reference."""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = json.load(open("artifacts/m17238_audit/audited_results.json"))
I = json.load(open("artifacts/m17238_audit/independent_michell.json"))
rows = A["comparison"]; ind = {round(r["froude_number"], 4): r for r in I["rows"]}
fr = [r["froude_number"] for r in rows]
iw = [ind[round(f, 4)]["Rw_independent"] for f in fr]
ref = [r["Rw_reference"] for r in rows]
koch = [r["Rw_kochin"] for r in rows]
LT = [r["Cw_reference_LT"] for r in rows]

fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))

ax[0].plot(fr, ref, "k-o", label="audited ref (Michell $\\theta$-form, in-repo)")
ax[0].plot(fr, koch, "r--x", label="audited got (canonical Kochin)")
ax[0].plot(fr, iw, "b:", marker="s", mfc="none", label="audit recompute (from scratch, no repo code)")
ax[0].set_xlabel("$Fr$"); ax[0].set_ylabel("$R_w$ [N]")
ax[0].set_title("(a) wave resistance, three implementations")
ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)

ax[1].semilogy(fr, [abs(ind[round(f, 4)]["Rw_independent"] - r) / r for f, r in zip(fr, ref)],
               "b-s", label="|audit recompute - ref| / ref")
ax[1].semilogy(fr, [abs(k - r) / r for k, r in zip(koch, ref)], "r--x",
               label="|kochin - ref| / ref")
ax[1].axhline(0.05, color="k", ls=":", label="mission tolerance 5%")
ax[1].axhline(0.15, color="m", ls="-.", label="platform reject line 15%")
ax[1].set_xlabel("$Fr$"); ax[1].set_ylabel("relative difference")
ax[1].set_title("(b) agreement is $10^{-6}$-$10^{-4}$: intra-theory")
ax[1].grid(alpha=.3, which="both"); ax[1].legend(fontsize=8)

for r in rows:
    cnt = [l["panel_count"] for l in r["panels_by_level"]]
    err = [l["rel_error"] for l in r["panels_by_level"]]
    ax[2].loglog(cnt, err, "o-", ms=3, label=f"Fr={r['froude_number']:.2f}")
ax[2].axhline(0.05, color="k", ls=":", label="5% band")
ax[2].set_xlabel("centreplane panels"); ax[2].set_ylabel("$|R_w^{pan}-R_w^{ref}|/R_w^{ref}$")
ax[2].set_title("(c) panel route convergence (audited)")
ax[2].grid(alpha=.3, which="both"); ax[2].legend(fontsize=7)

out = pathlib.Path("docs/board_audit/m17238/figures"); out.mkdir(parents=True, exist_ok=True)
p = out / "audit_m17238_wigley3_crosscheck.png"
fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
print(p, p.stat().st_size)
print("max_rel audit-vs-ref", max(abs(ind[round(f,4)]["Rw_independent"]-r)/r for f,r in zip(fr,ref)))
