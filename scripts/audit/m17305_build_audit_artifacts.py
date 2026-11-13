#!/usr/bin/env python3
"""Audit artifacts for mission 17305 (board-audit of m17246 CASE-BEM-NEUMANN-KELVIN).

Builds, from the delivered artifacts and the auditor's own re-derivation only
(no hand-typed numbers):

* runs/m17305/audit_numbers.json   -- comparison table (ref / independent / got)
* docs/board_audit/m17305/table_audit.md
* docs/board_audit/m17305/fig_audit_cw_vs_fr.png
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DELIV = Path("/root/tp-agent/work/m17246/docs/case_bem_neumann_kelvin/numbers.json")
INDEP = Path("/root/tp-agent/work/m17305/runs/m17305/independent_michell.json")
RUNS = Path("runs/m17305")
OUTD = Path("docs/board_audit/m17305")
L, B, T = 10.0, 1.0, 0.625
RHO, G = 1025.0, 9.80665
S_REF = L * T


def main() -> int:
    d = json.loads(DELIV.read_text())
    i = json.loads(INDEP.read_text())
    rows = []
    for r, x in zip(d["route_a"]["rows"], i["rows"]):
        u, ref, got = r["U"], r["ref_michell_canonical"], r["got_farfield_panel_finest"]
        cw = lambda f: f / (0.5 * RHO * u * u * S_REF)  # noqa: E731
        rows.append({
            "Fr": r["Fr"], "U": u,
            "ref_delivered": ref, "Cw_ref_delivered": cw(ref),
            "ref_independent": x["rw_independent"], "Cw_ref_independent": cw(x["rw_independent"]),
            "ref_spread": (x["rw_independent"] - ref) / ref,
            "got": got, "Cw_got": cw(got),
            "err_got_vs_delivered_ref": (got - ref) / ref,
            "err_got_vs_independent_ref": (got - x["rw_independent"]) / x["rw_independent"],
        })
    out = {
        "case": "CASE-BEM-NEUMANN-KELVIN", "audit_mission": 17305, "subject_mission": 17246,
        "tolerance": d["route_a"]["tolerance"], "s_ref": S_REF,
        "rows": rows,
        "max_err_got_vs_independent_ref": max(abs(r["err_got_vs_independent_ref"]) for r in rows),
        "max_ref_spread": max(abs(r["ref_spread"]) for r in rows),
        "route_a_all_passed_delivered": d["route_a"]["all_passed"],
        "route_b": {"max_abs_error": d["route_b"]["max_abs_error"],
                    "any_passed": d["route_b"]["any_passed"],
                    "rows": [{"Fr": b["Fr"], "rel_error": b["rel_error"]} for b in d["route_b"]["rows"]]},
        "kernel_selfcheck": d["kernel"],
        "wave_field": {k: d["wave_field"][k] for k in
                       ("measured_half_angle_deg", "theory_half_angle_deg", "eta_min", "eta_max")},
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    OUTD.mkdir(parents=True, exist_ok=True)
    (RUNS / "audit_numbers.json").write_text(json.dumps(out, indent=1))

    # -- table (generated, never hand-typed) --------------------------------
    md = ["| Fr | ref Rw [N] (delivered) | ref Rw [N] (auditor, own quadrature) | ref spread | got Rw [N] | got vs indep ref | tol <5% |",
          "|---:|---:|---:|---:|---:|---:|:--:|"]
    for r in rows:
        md.append(f"| {r['Fr']:.2f} | {r['ref_delivered']:.5f} | {r['ref_independent']:.5f} | "
                  f"{100*r['ref_spread']:+.4f}% | {r['got']:.4f} | "
                  f"{100*r['err_got_vs_independent_ref']:+.4f}% | "
                  f"{'PASS' if abs(r['err_got_vs_independent_ref']) < 0.05 else 'FAIL'} |")
    (OUTD / "table_audit.md").write_text("\n".join(md) + "\n")

    # -- figure ------------------------------------------------------------
    fr = np.array([r["Fr"] for r in rows])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax.plot(fr, [r["Cw_ref_delivered"] for r in rows], "k-", lw=2, label="ref: Michell (delivered)")
    ax.plot(fr, [r["Cw_ref_independent"] for r in rows], "C3o", ms=6, mfc="none",
            label="ref: auditor independent quadrature")
    ax.plot(fr, [r["Cw_got"] for r in rows], "C0s--", ms=5, label="got: NK far-field (1800 panels)")
    ax.set_xlabel("Fr"); ax.set_ylabel(r"$C_w = R_w / (0.5 \rho U^2 L T)$")
    ax.set_title("CASE-BEM-NEUMANN-KELVIN: got vs two independent ref routes")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax2.axhline(5.0, color="r", ls=":", lw=1, label="tol 5%")
    ax2.axhline(3.0, color="orange", ls=":", lw=1, label="ecosystem 3%")
    ax2.plot(fr, [100 * abs(r["err_got_vs_independent_ref"]) for r in rows], "C0s-",
             label="|got - auditor ref|")
    ax2.plot(fr, [100 * abs(r["err_got_vs_delivered_ref"]) for r in rows], "C1o--",
             label="|got - delivered ref|")
    ax2.set_xlabel("Fr"); ax2.set_ylabel("relative error [%]")
    ax2.set_title("auditor-side error against an independent reference")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTD / "fig_audit_cw_vs_fr.png", dpi=130)
    print(json.dumps({k: out[k] for k in ("max_err_got_vs_independent_ref", "max_ref_spread")}, indent=1))
    print((OUTD / "table_audit.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())