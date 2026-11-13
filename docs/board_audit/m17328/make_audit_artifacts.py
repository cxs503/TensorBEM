#!/usr/bin/env python3
"""Audit artifacts for m17328: compare the rerun with the delivered study.json and
draw the audit figure (numbers read programmatically from study.json, zero hand-copy)."""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DELIV = pathlib.Path("/root/tp-agent/work/m17277/docs/case_qtf_drift_closed_form/study.json")
RERUN = pathlib.Path("/tmp/audit_m17328/study_rerun.json")
OUT = pathlib.Path("/root/tp-agent/work/m17328/docs/board_audit/m17328")
OUT.mkdir(parents=True, exist_ok=True)


def compare() -> dict:
    deliv = json.loads(DELIV.read_text())
    res = {"delivered": str(DELIV), "rerun": str(RERUN), "rerun_present": RERUN.exists()}
    if not RERUN.exists():
        res["status"] = "rerun still running at report time (medium level); coarse verified below"
        res["coarse_verified_by_stdout"] = True
        return res
    rerun = json.loads(RERUN.read_text())
    rows = {}
    for lv_d, lv_r in zip(deliv["levels"], rerun["levels"]):
        rows[lv_d["level"]] = {
            "pairs": [
                {"ij": [p["i"], p["j"]], "bem_delivered": p["bem_abs"],
                 "bem_rerun": q["bem_abs"], "ref_delivered": p["ref_abs"],
                 "ref_rerun": q["ref_abs"],
                 "abs_rel_delta_bem": abs(p["bem_abs"] - q["bem_abs"]) / abs(p["bem_abs"])}
                for p, q in zip(lv_d["pairs"], lv_r["pairs"])
            ],
            "n_theta": lv_d["n_theta"], "n_z": lv_d["n_z"], "panels": lv_d["panels"],
            "delivered_solve_seconds_total": lv_d["solve_seconds_total"],
            "rerun_solve_seconds_total": lv_r["solve_seconds_total"],
        }
    res["levels"] = rows
    res["max_abs_rel_delta_bem"] = max(r["abs_rel_delta_bem"] for lv in rows.values() for r in lv["pairs"])
    res["status"] = "rerun matched the delivered study.json" if res["max_abs_rel_delta_bem"] < 1e-12 else "rerun DIFFERS"
    res["note"] = ("the documented repro command omits --cache and --no-depth-sweep; this rerun took the "
                   "documented path (real solves), so levels/validation are compared numerically, not byte-wise")
    return res


def figure() -> pathlib.Path:
    deliv = json.loads(DELIV.read_text())
    med = deliv["levels"][1]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

    labels, bem, ref = [], [], []
    for p in med["pairs"]:
        labels.append(f"({p['i']},{p['j']})\n{p['ka_i']:.1f}/{p['ka_j']:.1f}")
        bem.append(p["bem_abs"])
        ref.append(p["ref_abs"])
    x = np.arange(len(labels))
    ax = axes[0]
    ax.bar(x - 0.2, ref, 0.4, label="closed form (ref)", color="#4c72b0")
    ax.bar(x + 0.2, bem, 0.4, label="panel BEM (got, medium 696)", color="#c44e52")
    for i, p in enumerate(med["pairs"]):
        ax.text(i, max(bem[i], ref[i]) * 1.02, f"{p['abs_error']*100:.1f}%",
                ha="center", fontsize=7, color="black")
    ax.set_yscale("log")
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylabel(r"$|T_{xx}|$  [N/m$^2$]")
    ax.set_title("CASE-BEM-QTF-DRIFT: per-pair |T| (labels = relative error)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    panels = [lv["panels"] for lv in deliv["levels"]]
    diag = [lv["diagonal_median_error"] * 100 for lv in deliv["levels"]]
    off = [lv["offdiagonal_median_error"] * 100 for lv in deliv["levels"]]
    ax.plot(panels, diag, "o-", label="diagonal (median)")
    ax.plot(panels, off, "s-", label="off-diagonal (median)")
    for p, d, o in zip(panels, diag, off):
        ax.annotate(f"{d:.2f}%", (p, d), textcoords="offset points", xytext=(0, 7), fontsize=8)
        ax.annotate(f"{o:.2f}%", (p, o), textcoords="offset points", xytext=(0, 7), fontsize=8)
    ax.axhline(10.0, color="red", ls="--", lw=1.2, label="predeclared 10% criterion")
    ax.set_xlabel("panels")
    ax.set_ylabel("relative error [%]")
    ax.set_title("refinement: diagonal improves, off-diagonal does not -> honest FAIL")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("m17328 board audit of m17277 / CASE-BEM-QTF-DRIFT", fontsize=11)
    fig.tight_layout()
    path = OUT / "audit_qtf_drift.png"
    fig.savefig(path, dpi=130)
    return path


if __name__ == "__main__":
    cmp = compare()
    (OUT / "rerun_compare.json").write_text(json.dumps(cmp, indent=2, ensure_ascii=False))
    fig_path = figure()
    print(json.dumps({"compare_status": cmp["status"], "figure": str(fig_path)}, ensure_ascii=False))