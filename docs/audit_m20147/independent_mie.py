#!/usr/bin/env python3
"""Auditor's own partial-wave (Mie) series -- independent of the case module.

No import of tensorbem: scipy spherical Riccati-Bessel only, so the reference
column of the audited case must reproduce from a third code path.
"""
import csv, json, math, sys
import numpy as np
from scipy.special import spherical_jn, spherical_yn

def sigma_hat(ka, nmax=None):
    ka = float(ka)
    n_max = nmax or int(math.ceil(ka + 4.0 * ka ** (1.0 / 3.0) + 12))
    tot = 0.0
    for n in range(n_max + 1):
        jn = spherical_jn(n, ka)
        djn = spherical_jn(n, ka, True)
        yn = spherical_yn(n, ka)
        dyn = spherical_yn(n, ka, True)
        # h_n^(1) = j_n + i y_n  ->  derivative likewise
        num = djn
        den = djn + 1j * dyn
        A = -num / den
        tot += (2 * n + 1) * abs(A) ** 2
    return (4.0 * math.pi / ka ** 2) * tot / (math.pi * 1.0 ** 2)

if __name__ == "__main__":
    sweep = list(csv.DictReader(open(sys.argv[1])))
    out = {"rows": [], }
    worst = 0.0
    for r in sweep:
        ka = float(r["ka"])
        mine = sigma_hat(ka)
        theirs_ref = float(r["sigma_hat_ref"])
        theirs_bem = float(r["sigma_hat_bem"])
        rel_ref = abs(mine - theirs_ref) / theirs_ref
        rel_bem = abs(theirs_bem - mine) / mine
        worst = max(worst, rel_bem)
        out["rows"].append({"ka": ka, "auditor_sigma_hat": mine,
                            "case_sigma_hat_ref": theirs_ref,
                            "case_sigma_hat_bem": theirs_bem,
                            "rel_ref_vs_auditor": rel_ref,
                            "rel_bem_vs_auditor_pct": 100 * rel_bem})
    out["worst_bem_error_pct_vs_auditor"] = 100 * worst
    out["limits"] = {"rayleigh_ka0.02": sigma_hat(0.02) / 0.02 ** 4,
                     "extinction_ka100": sigma_hat(100.0)}
    print(json.dumps(out, indent=2))
    json.dump(out, open(sys.argv[2], "w"), indent=2)
