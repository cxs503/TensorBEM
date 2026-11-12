"""m16696 evidence: the pre-16696 "Ursell/Yeung analytic reference" of the multihull
frequency-dependence benchmark vs (a) the exact multipole series, (b) the 2-D BEM
solver.  Writes docs/m16696/evidence.json and the report figure.
"""
from __future__ import annotations
import json, math, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorbem.analytic import ursell_cylinder_coefficients as exact
from tensorbem.benchmarks.multihull_frequency_dependence import (
    bem_cross_validation, single_cylinder_heave_coefficients, mutual_interaction_coefficients,
    MUTUAL_DAMPING_SUPPORTED, verify_asymptotics,
)

OUT = pathlib.Path("docs/m16696"); OUT.mkdir(parents=True, exist_ok=True)
AOF = 0.3 / 0.8
KA = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]

# (1) exact reference vs the withdraw formula (values re-read from the before file)
before = json.loads((OUT / "before_analytic_vs_exact.json").read_text())
after_rows = []
for ka in KA:
    mu, nu = exact(ka, AOF)
    after_rows.append(dict(ka=ka, A_exact=mu, B_exact=nu,
                           A_ref=single_cylinder_heave_coefficients(ka, AOF)[0],
                           B_ref=single_cylinder_heave_coefficients(ka, AOF)[1]))

# (2) BEM vs exact, panel ladder
bem = {}
for n in (48, 96):
    rows = bem_cross_validation(radius=0.3, depth=0.8, n_panels=n, ka_values=KA[:4])
    bem[n] = [{k: v for k, v in r.items() if k in ("ka", "A_rel_error", "B_rel_error")} for r in rows]

# (3) truncation convergence of the exact series
conv = []
for ka in (0.01, 0.5, 5.0):
    a = single_cylinder_heave_coefficients(ka, AOF, 40)
    b = single_cylinder_heave_coefficients(ka, AOF, 120)
    conv.append(dict(ka=ka, dA=abs(a[0]-b[0]), dB=abs(a[1]-b[1])))

asym = verify_asymptotics(0.3, 0.8)
mutual = {f"d_over_a={d}": mutual_interaction_coefficients(0.5, AOF, d)
          for d in (2.5, 3.0, 5.0, 10.0)}

evidence = dict(
    mission=16696, repo="TensorBEM", a_over_f=AOF,
    retracted_formula="benchmarks/multihull_frequency_dependence.single_cylinder_heave_coefficients (pre-16696)",
    before_vs_exact=before, after_reference_table=after_rows,
    bem_vs_exact=bem, truncation=conv, asymptotics=asym,
    mutual_reference=mutual, mutual_damping_supported=MUTUAL_DAMPING_SUPPORTED,
    worst_before=dict(A=max(r["A_err"] for r in before), B=max(r["B_err"] for r in before)),
    worst_bem=dict(A=max(r["A_rel_error"] for rows in bem.values() for r in rows),
                   B=max(r["B_rel_error"] for rows in bem.values() for r in rows)),
)
(OUT / "evidence.json").write_text(json.dumps(evidence, indent=1, default=str))

# ---- figure: exact vs retracted vs BEM, and the retracted reference's error
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
kb = [r["ka"] for r in before]
ax1.loglog(kb, [r["B_exact"] for r in before], "k-", lw=2, label="exact Ursell multipole")
ax1.loglog(kb, [abs(r["B_old"]) for r in before], "r--", lw=1.6,
           label="pre-16696 'analytic' formula")
bemk = [r["ka"] for r in bem[96]]
ax1.loglog(bemk, [r["B_exact"] for r in before if r["ka"] in bemk], "o", mfc="none",
           mec="C0", ms=9, label="2-D BEM (N=96) reference point")
ax1.set_xlabel(r"$ka$"); ax1.set_ylabel(r"$B_{33}/(\rho\pi a^2\omega)$")
ax1.set_title("heave damping: exact vs the retracted formula")
ax1.grid(True, which="both", alpha=0.3); ax1.legend(fontsize=8)

ax2.semilogx(kb, [100 * r["A_err"] for r in before], "rs-", label=r"$A_{33}$ error")
ax2.semilogx(kb, [100 * r["B_err"] for r in before], "b^-", label=r"$B_{33}$ error")
ax2.axhline(15, color="k", ls=":", lw=1, label="15 % acceptance line")
ax2.set_xlabel(r"$ka$"); ax2.set_ylabel("relative error of the retracted formula [%]")
ax2.set_title(r"pre-16696 'Ursell/Yeung' reference $a/f=0.375$")
ax2.grid(True, which="both", alpha=0.3); ax2.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig_multihull_reference_defect.png", dpi=140)

print(json.dumps(dict(worst_before=evidence["worst_before"], worst_bem=evidence["worst_bem"],
                      asym=asym["infinite_frequency"], mutual=mutual), indent=1, default=str))
for n, rows in bem.items():
    for r in rows:
        print("BEM N=%d ka=%.2f A_err=%.4f B_err=%.4f" % (n, r["ka"], r["A_rel_error"], r["B_rel_error"]))
