#!/usr/bin/env python3
"""Audit figure for m17174 / CASE-BEM-CYL-ADDED-MASS-2D.

Reads the auditor's own re-run artifact (artifacts/m17197/refinement_cylinder_repro.json)
and plots, left: added mass vs panel count against the closed form rho*pi*a^2;
right: log-log relative error vs N with the observed second-order slope.
All numbers are injected from the JSON; nothing is typed by hand.

Run: python3 artifacts/m17197/make_audit_figure.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "refinement_cylinder_repro.json"
OUT = HERE / "fig_audit_cylinder_added_mass.png"


def main() -> None:
    d = json.loads(DATA.read_text())
    rows = sorted(d["translating_cylinder_added_mass"], key=lambda r: r["panels"])
    n = np.array([r["panels"] for r in rows], dtype=float)
    got = np.array([r["added_mass_per_unit_span"] for r in rows], dtype=float)
    ref = np.array([r["exact_added_mass_per_unit_span"] for r in rows], dtype=float)
    err = np.array([r["added_mass_relative_error"] for r in rows], dtype=float)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax0.plot(n, got, "o-", lw=1.8, ms=6, label="BEM result (re-run m17197)")
    ax0.axhline(ref[0], color="k", ls="--", lw=1.4, label=r"closed form $\rho\pi a^2$")
    ax0.set_xscale("log", base=2)
    ax0.set_xlabel("panels N")
    ax0.set_ylabel(r"added mass / unit span  [kg/m]")
    ax0.set_title(r"$m_A$ vs N  ($\rho=%.0f$, $a=%.1f$ m)" % (rows[0]["fluid_density"], rows[0]["radius"]))
    ax0.legend(loc="lower right", fontsize=8)
    ax0.grid(alpha=0.3, which="both")

    ax1.loglog(n, err, "o-", lw=1.8, ms=6, color="tab:red",
               label="relative error")
    order = float(np.polyfit(np.log(n), np.log(err), 1)[0])
    ax1.loglog(n, err[0] * (n / n[0]) ** (-2.0), ":", color="tab:blue", lw=1.5,
               label=r"$N^{-2}$ reference slope")
    ax1.axhline(3e-2, color="tab:orange", ls="--", lw=1.4, label="3% acceptance")
    ax1.set_xlabel("panels N")
    ax1.set_ylabel(r"$|m_A - \rho\pi a^2| / (\rho\pi a^2)$")
    ax1.set_title("convergence (fitted order %.2f, %d levels)" % (order, len(rows)))
    ax1.legend(loc="lower left", fontsize=8)
    ax1.grid(alpha=0.3, which="both")

    fig.suptitle("m17174 CASE-BEM-CYL-ADDED-MASS-2D — auditor independent re-run", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160)
    print(json.dumps({"out": str(OUT), "fitted_order": order,
                      "errors": err.tolist(), "panels": n.tolist()}, indent=2))
    assert math.isfinite(order)


if __name__ == "__main__":
    main()