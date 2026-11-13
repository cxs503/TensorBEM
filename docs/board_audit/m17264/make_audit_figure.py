"""Board-audit m17264 report figure + platform report POST support (data only).

Reads the audited subject's artifacts (snapshot) and the audit probe outputs and
renders the two-panel figure used in the progress report:

  left  : per-DOF R1 relative error, delivered case (rotation centre = default
          z=0) vs the same mesh re-solved with the reference convention
          (0,0,-T/2), with the 10 % case tolerance drawn as a line;
  right : the five gates as delivered against their own tolerances (log scale).

Numbers are read from
  docs/board_audit/m17264/subject_artifacts/results.json
  artifacts/audit_m17264/center_check_20x10.json
and nothing is typed in by hand.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
SNAP = ROOT / "docs/board_audit/m17264"
SUBJ = json.loads((SNAP / "subject_artifacts/results.json").read_text())
AUDIT = json.loads((ROOT / "artifacts/audit_m17264/center_check_20x10.json").read_text())
DOFS = ["surge", "sway", "heave", "roll", "pitch", "yaw"]


def build(path: pathlib.Path) -> dict:
    delivered = [abs(r["added_mass_rel_err"]) for r in SUBJ["cross_solver"]["rows"]]
    fixed = [abs(AUDIT["runs"]["20x10/center_zmT2"]["added_mass_rel_err"][d]) for d in DOFS]
    gates = SUBJ["gates"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    x = np.arange(len(DOFS))
    ax = axes[0]
    ax.bar(x - 0.2, delivered, 0.4, label="delivered case (centre z=0, default)")
    ax.bar(x + 0.2, fixed, 0.4, label="same mesh, centre z=-T/2 (reference convention)")
    ax.axhline(0.10, color="k", ls="--", lw=1, label="case tolerance 10%")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(DOFS, rotation=20)
    ax.set_ylabel("|A_ii rel. error| vs Capytaine 3.1")
    ax.set_title("R1 gate, omega = 1 rad/s, 400-quad mesh\n"
                 "worst: %.2f%% -> %.2f%%" % (100 * max(delivered), 100 * max(fixed)), fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    labels, vals, tols, passed = [], [], [], []
    for gid, gate in gates.items():
        labels.append(gate["id"].split("_", 1)[0])
        vals.append(np.nan if gate["value"] is None else max(float(gate["value"]), 1e-6))
        tols.append(gate["tol"])
        passed.append(gate["passed"])
    x2 = np.arange(len(labels))
    ax.bar(x2, vals, 0.5, color=["tab:green" if p else "tab:red" for p in passed])
    ax.scatter(x2, tols, marker="_", s=400, color="k", zorder=3, label="gate tolerance")
    ax.set_yscale("log")
    ax.set_xticks(x2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("gate value (rel. error, log)")
    ax.set_title("gates as delivered: %s\n(R4 = nan: no resonance in [0.5,6] rad/s)" % SUBJ["verdict"],
                 fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {"path": str(path), "worst_delivered": max(delivered), "worst_fixed": max(fixed)}


if __name__ == "__main__":
    out = ROOT / "docs/board_audit/m17264/figures/fig_audit_m17264_r1_centre_and_gates.png"
    info = build(out)
    print(json.dumps(info))