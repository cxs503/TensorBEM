#!/usr/bin/env python3
"""UQ self-check: validate :mod:`tensorbem.uq` against known errors, emit data+figure.

The point of this tool is that the uncertainty machinery is itself *verified*
before any report quotes an error bar.  Three independent checks are run:

1. **Manufactured order.**  ``f(h) = f_exact + C h^p`` with ``p = 2``: the GCI
   implementation must return ``p`` and ``f_exact``.  A sign slip in the
   Richardson formula is the single easiest bug to hide in a UQ report, so it
   gets its own check with a known answer.
2. **GCI as an error estimator on a real BEM solve.**  The hydrostatic module
   already ships a five-level refinement study against closed-form references
   (hemisphere volume, waterplane area, transverse second moment, ``KB``).  For
   the levels ``panels = 256, 1024, 4096`` the refinement ratio in ``h`` is
   exactly 2, the analytic error of every level is known, and therefore the GCI
   band can be scored: a valid estimator must be *conservative*, i.e.
   ``true_error <= GCI_fine`` (Roache's factor of safety exists precisely for
   this).  The ratio ``true_error / GCI`` is the honest quality measure of the
   error bar.
3. **Bootstrap interval.**  A 10 % multiplicative offset with 5 % multiplicative
   scatter must give MAPE = 10 % with a 95 % interval that brackets it.

Outputs
-------
``data/m3538_uq/uq_selfcheck.json``
``figures/m3538_uq/uq_selfcheck.png``

Usage::

    PYTHONPATH=src python3 tools/uq_selfcheck.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np

from tensorbem.benchmarks.hydrostatic_convergence import hemisphere_study
from tensorbem.uq import (
    bootstrap_mape,
    grid_convergence_index,
    mape,
    perturbation_band,
    richardson_extrapolate,
)

#: quantities of the hydrostatic study that carry a closed-form reference
QUANTITIES = ("displaced_volume", "waterplane_area", "i_transverse", "kb")


def manufactured_check() -> dict:
    """``f(h) = 5.0 + 0.4 h^2`` at ``h = 0.4, 0.2, 0.1`` (r = 2)."""
    exact, c, p = 5.0, 0.4, 2.0
    vals = [exact + c * h ** p for h in (0.4, 0.2, 0.1)]
    res = grid_convergence_index(vals[2], vals[1], vals[0], refinement_ratio=2.0,
                                 quantity="manufactured")
    out = res.as_dict()
    out["exact"] = exact
    out["exact_order"] = p
    out["order_error"] = abs(res.observed_order - p)
    out["limit_error"] = abs(res.extrapolated - exact)
    out["within_gci"] = abs(vals[2] - exact) <= res.gci_fine * abs(exact)
    return out


def hydrostatic_gci() -> list[dict]:
    """GCI on the existing five-level hydrostatic study, scored against exact errors."""
    study = hemisphere_study()
    levels = study["levels"]
    by_panels = {lv["panels"]: lv for lv in levels}
    rows = []
    for triple in ((64, 256, 1024), (256, 1024, 4096)):
        coarse, medium, fine = (by_panels[p] for p in triple)
        for q in QUANTITIES:
            res = grid_convergence_index(
                fine[q], medium[q], coarse[q], refinement_ratio=2.0,
                quantity=q, label=f"hemisphere panels={triple[2]}")
            true_err = abs(fine[q] - study["reference"].get(q, float("nan")))
            true_rel = true_err / abs(study["reference"][q]) if q in study["reference"] else float("nan")
            ratio = (true_rel / res.gci_fine) if (res.gci_fine and math.isfinite(true_rel)) else float("nan")
            row = {
                "triple_panels": list(triple),
                "quantity": q,
                "fine": fine[q],
                "reference": study["reference"].get(q),
                "true_rel_error": true_rel,
                "gci_fine": res.gci_fine,
                "gci_band95": list(res.band95),
                "observed_order": res.observed_order,
                "asymptotic_ratio": res.asymptotic_ratio,
                "converged": res.converged,
                "conservative": bool(ratio <= 1.0) if math.isfinite(ratio) else None,
                "true_error_over_gci": ratio,
                "extrapolated": res.extrapolated,
            }
            rows.append(row)
    return rows


def bootstrap_check() -> dict:
    """Known 10 % bias plus 5 % scatter: the interval must bracket 10 %."""
    rng = np.random.default_rng(20261103)
    reference = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
    values = reference * 1.10 * (1.0 + rng.normal(0.0, 0.05, reference.size))
    res = bootstrap_mape(values, reference)
    res["reference_case"] = "10% bias, 5% multiplicative scatter, n=8"
    res["covers_10pct"] = bool(res["lo"] <= 10.0 <= res["hi"])
    band = perturbation_band(values, reference, rel_sigma=0.05)
    return {"bootstrap_mape": res, "perturbation_band_5pct": band}


def make_figure(hydro: list[dict], path: pathlib.Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # (a) true error vs h for the fine triple, with the GCI band as an interval
    ax = axes[0]
    rows = [r for r in hydro if r["triple_panels"][0] == 256]
    names = [r["quantity"] for r in rows]
    true_err = np.array([100.0 * r["true_rel_error"] for r in rows])
    gci = np.array([100.0 * r["gci_fine"] for r in rows])
    y = np.arange(len(rows))
    ax.barh(y - 0.18, true_err, height=0.34, color="#2b6cb0", label="true error vs closed form")
    ax.barh(y + 0.18, gci, height=0.34, color="#dd6b20", label="GCI band (Fs = 1.25)")
    ax.set_yticks(y, names)
    ax.set_xscale("log")
    ax.set_xlabel("relative error / GCI  [%]  (log)")
    ax.set_title("(a) GCI is conservative: bar orange >= bar blue")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="x", which="both", alpha=0.3)

    # (b) error vs order: true error of all levels against the observed order
    ax = axes[1]
    study = hemisphere_study()
    panels = [lv["panels"] for lv in study["levels"]]
    for q in QUANTITIES:
        err = [abs(lv[q] - study["reference"][q]) / abs(study["reference"][q])
               for lv in study["levels"]]
        ax.loglog(panels, err, "o-", ms=3.5, lw=1.2, label=q)
    ref = np.array(panels, dtype=float)
    ax.loglog(ref, 0.06 * (ref / ref[-1]) ** -1.0, "k--", lw=1.0, label="slope -1 (h^2)")
    ax.set_xlabel("panels (h^2 ~ 1/panels)")
    ax.set_ylabel("relative error vs closed form")
    ax.set_title("(b) hemisphere hydrostatics, five levels")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=".", help="repository root to write data/ and figures/")
    args = ap.parse_args()
    root = pathlib.Path(args.out)

    manufactured = manufactured_check()
    hydro = hydrostatic_gci()
    boot = bootstrap_check()

    payload = {
        "provenance": {
            "tool": "tools/uq_selfcheck.py",
            "module": "tensorbem.uq",
            "hydrostatic_study": "tensorbem.benchmarks.hydrostatic_convergence.hemisphere_study",
            "refinement_ratio": 2.0,
            "safety_factor": 1.25,
        },
        "manufactured": manufactured,
        "hydrostatic_gci": hydro,
        "bootstrap": boot,
    }
    data_path = root / "data" / "m3538_uq" / "uq_selfcheck.json"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    fig_path = root / "figures" / "m3538_uq" / "uq_selfcheck.png"
    make_figure(hydro, fig_path)

    print("== 1. manufactured order check ==")
    print(f"  p_obs = {manufactured['observed_order']:.6f} (expected 2), "
          f"|dp| = {manufactured['order_error']:.2e}, "
          f"|f_ext - exact| = {manufactured['limit_error']:.2e}, "
          f"GCI covers exact: {manufactured['within_gci']}")
    print("== 2. GCI vs closed-form error (hemisphere hydrostatics) ==")
    print(f"  {'panels(finest)':>15} {'quantity':>17} {'true %':>10} {'GCI %':>10} "
          f"{'true/GCI':>9} {'p':>6} {'conv':>5}")
    for r in hydro:
        print(f"  {r['triple_panels'][2]:>15} {r['quantity']:>17} "
              f"{100*r['true_rel_error']:>10.4f} {100*r['gci_fine']:>10.4f} "
              f"{r['true_error_over_gci']:>9.3f} {r['observed_order']:>6.3f} "
              f"{str(r['converged']):>5}")
    print("== 3. bootstrap / perturbation band ==")
    bm = boot["bootstrap_mape"]
    print(f"  MAPE = {bm['point']:.3f}%  95% CI = [{bm['lo']:.3f}, {bm['hi']:.3f}]  "
          f"n = {bm['n']}  covers 10%: {bm['covers_10pct']}")
    pb = boot["perturbation_band_5pct"]
    print(f"  reference uncertainty 5% -> MAPE band [{pb['lo']:.3f}, {pb['hi']:.3f}]")
    print(f"wrote {data_path}")
    print(f"wrote {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())