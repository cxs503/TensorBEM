"""Figures for the m14488 TensorBEM quality audit.

Two panels, both fed from the probe artifacts (no hand-copied numbers):

left  -- P1: single-layer near-field quadrature, relative error of the three
         shipped routes against the audit's own brute-force reference, versus
         the target height above the panel (log-log).  The reference's own
         discretisation error is drawn as a band so the plot cannot be read as
         "the module is wrong where the reference is unconverged".
right -- P2: flexural-gravity dispersion, relative divergence between the two
         shipped implementations and the group-velocity self-check.

Run::

    PYTHONPATH=src python3 docs/audit_m14488/make_figures.py
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def load(name: str):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def main() -> int:
    p1 = load("probe_results_p1.json")["p1"]["cases"]
    p2 = load("probe_results.json")["p2"]["rows"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # ---- left: quadrature routes vs the audit's reference -------------------
    ax = axes[0]
    hs = np.array([1e-1, 1e-2, 1e-3])
    label_of = {"h=1e-1_centroid": 0, "h=1e-2_centroid": 1, "h=1e-3_centroid": 2}
    series = {"adaptive": [], "polar": [], "semi-analytic": [], "reference self": []}
    for case in p1:
        if case["case"] not in label_of:
            continue
        series["adaptive"].append(case["adaptive_rel_err_single"])
        series["polar"].append(case["polar_rel_err_single"])
        series["semi-analytic"].append(case["arbiter_rel_err_single"])
        series["reference self"].append(case["reference_self_error_single"])
    for name, marker in (("adaptive", "o"), ("polar", "s"), ("semi-analytic", "^")):
        ax.loglog(hs, np.maximum(series[name], 1e-16), marker + "-", label=name)
    ax.loglog(hs, np.maximum(series["reference self"], 1e-16), "k--",
              label="audit reference self-error")
    ax.invert_xaxis()
    ax.set_xlabel("target height above panel  h  (m), panel edge = 1 m")
    ax.set_ylabel("relative error of single-layer integral")
    ax.set_title("P1  near-field quadrature vs independent reference\n"
                 "(all shipped routes sit at the reference's own error floor)", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    # ---- right: flexural dispersion divergence ------------------------------
    ax = axes[1]
    ks = np.array([r["k"] for r in p2])
    diff = 100.0 * np.array([r["hydroelastic_rel_diff"] for r in p2])
    cgerr = np.maximum([r["cg_rel_err"] for r in p2], 1e-18)
    ax.semilogx(ks, diff, "o-", label="flexural_gravity vs hydroelastic (%)")
    ax.set_xlabel("wavenumber k (1/m)")
    ax.set_ylabel("relative difference (%)", color="C0")
    ax.axhline(3.0, color="C3", ls=":", label="3 % accuracy line")
    ax2 = ax.twinx()
    ax2.semilogy(ks, cgerr, "s--", color="C2", label="group velocity: analytic vs finite diff")
    ax2.set_ylabel("group-velocity self-check rel. err", color="C2")
    ax.set_title("P2  two shipped ice-plate dispersions diverge with k\n"
                 "(no in-repo arbitration of which one is authoritative)", fontsize=9)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("TensorBEM quality audit (m14488): independent probes, 2026-11-11", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = HERE / "fig_audit.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())