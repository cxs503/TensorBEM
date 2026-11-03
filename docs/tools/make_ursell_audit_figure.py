"""Figure for docs/ursell_provenance_audit.md (reads artifacts/ursell_provenance.json)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("artifacts/ursell_provenance.json") as fh:
    data = json.load(fh)
rows = data["rows"]
n = np.array([r["panels"] for r in rows], dtype=float)
am = np.array([r["am_err"] for r in rows])
b = np.array([r["b_err"] for r in rows])
nf = np.array([r["near_far_reldiff"] for r in rows])
mfs_am = np.array([r["mfs_am_err"] for r in rows])
mfs_b = np.array([r["mfs_b_err"] for r in rows])

doc_am = np.array([1.51e-2, 7.72e-3, 3.90e-3])
doc_b = np.array([2.47e-2, 1.30e-2, 6.67e-3])

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

ax = axes[0]
ax.loglog(n, am, "o-", color="#1f77b4", label="panel BEM, added mass (reproduced)")
ax.loglog(n, b, "s-", color="#d62728", label="panel BEM, damping (reproduced)")
ax.loglog(n, doc_am, "o--", color="#1f77b4", alpha=0.45, label="documented added mass (stale)")
ax.loglog(n, doc_b, "s--", color="#d62728", alpha=0.45, label="documented damping (stale)")
ax.loglog(n, mfs_am, "^-", color="#2ca02c", label="MFS, added mass")
ax.loglog(n, mfs_b, "v-", color="#9467bd", label="MFS, damping")
ax.loglog(n, 1.6736e-2 * 64.0 / n, "k:", lw=1.0, label=r"slope $-1$ (first order)")
ax.axhline(3.0e-2, color="grey", ls="-.", lw=1.0)
ax.text(66, 3.1e-2, "3% acceptance", color="grey", fontsize=8)
ax.set_xlabel("panels on the cylinder")
ax.set_ylabel("relative error against Ursell multipole reference")
ax.set_title("(a) Ursell radiation: discretisation error $ka=0.6$, $a/f=0.5$")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=7.5, loc="lower left")

ax = axes[1]
ax.semilogx(n, nf, "o-", color="#ff7f0e", label="near/far-field damping mismatch")
ax.semilogx(n, nf[0] * 64.0 / n, "k:", lw=1.0, label=r"slope $-1$")
ax.set_xlabel("panels on the cylinder")
ax.set_ylabel(r"$|B_{\rm near}-B_{\rm far}|/B_{\rm far}$")
ax.set_title("(b) Radiation condition: Haskind energy identity")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig("docs/figures/ursell_provenance_convergence.png", dpi=160)
print("wrote docs/figures/ursell_provenance_convergence.png")