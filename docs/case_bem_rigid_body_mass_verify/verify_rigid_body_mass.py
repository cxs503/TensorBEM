"""CASE-BEM-RIGID-BODY-MASS : fresh verification of the 6x6 rigid-body mass
matrix against the closed-form block form (Fossen 2011 eq. 3.44 / Newman 1977
sec. 4.5), with <3% case-pool tolerance.

This is a *verification-debt* run: the capability was already merged and green;
this run re-executes the case-pool driver on the current checkout, extracts the
key numbers (worst accepted entry-relative error, ellipsoid convergence ladder,
orientation congruence, momentum-route round-off) and renders a convergence
figure.  Nothing is typed by hand -- every number is read from the driver's
machine-readable JSON.

Run:
    PYTHONPATH=src python3 runs/m21501_rigid_body_mass/verify_rigid_body_mass.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

TOLERANCE = 0.03


def _run_driver() -> dict:
    """Re-run the case-pool driver on the current checkout (fresh verdict)."""
    tmp = OUT / "_driver_out"
    tmp.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "tensorbem.benchmarks.casepool_bem_rigid_body_mass",
        "--outdir",
        str(tmp),
    ]
    env = {"PYTHONPATH": str(ROOT / "src"), **dict(os.environ)}
    subprocess.run(cmd, cwd=ROOT, env=env, check=True, capture_output=True)
    return json.loads((tmp / "results.json").read_text())


def _convergence_figure(driver: dict) -> str:
    """Ellipsoid ladder: max entry-relative error vs panels, 3% tolerance line."""
    ladder = driver["base_case"]["ellipsoid"]["ladder"]
    panels = [r["panels"] for r in ladder]
    err = [r["max_entry_rel_error"] for r in ladder]
    vol = [r["volume_rel_error"] for r in ladder]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogx(panels, [100 * e for e in err], "o-", label="max entry rel err (M_RB)")
    ax.semilogx(panels, [100 * e for e in vol], "s--", label="volume rel err (geometry)")
    ax.axhline(100 * TOLERANCE, color="r", ls="--", lw=1.2, label="case-pool <3%")
    ax.set_xlabel("panels (ellipsoid, semi-axes 60/12/8 m)")
    ax.set_ylabel("relative error [%]")
    ax.set_title("CASE-BEM-RIGID-BODY-MASS: 6x6 M_RB vs closed-form block form")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    p = FIG / "fig_rigid_body_mass_convergence.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p.name


def main() -> int:
    driver = _run_driver()
    ver = driver["verdict"]
    bc = driver["base_case"]
    oc = driver["orientation_case"]

    # key numbers
    worst = ver["worst_accepted_entry_rel_error"]
    box_worst = bc["box_body"]["worst_accepted_entry_rel_error"]
    # ellipsoid worst accepted = last ladder entry (finest accepted level)
    ell_ladder = bc["ellipsoid"]["ladder"]
    ell_worst = ell_ladder[-1]["max_entry_rel_error"]
    oc_worst = oc["worst_accepted_entry_rel_error"]
    momentum_roundoff = driver["momentum_route"]["worst_relative_error"]
    rao_replay = driver["rao_caller"]["replay_rel_residual"]

    fig_name = _convergence_figure(driver)

    artifact = {
        "benchmark": "CASE-BEM-RIGID-BODY-MASS",
        "mission": "m21501",
        "tolerance": TOLERANCE,
        "reference": (
            "Fossen (2011) Handbook of Marine Craft Hydrodynamics and Motion "
            "Control, Sec. 3.3 eq. (3.44) block form M_RB = [[m I3, -m S(r_g)], "
            "[m S(r_g), I_g + m S(r_g) S(r_g)^T]]; Newman (1977) Marine "
            "Hydrodynamics sec. 4.5; solid-ellipsoid second moments Lamb (1932) "
            "Ch. VI"
        ),
        "verdict": ver["overall"],
        "all_pass": ver["all_pass"],
        "worst_accepted_entry_rel_error": worst,
        "worst_accepted_entry_rel_error_pct": 100.0 * worst,
        "box_body_worst_entry_rel_error": box_worst,
        "ellipsoid_worst_accepted_entry_rel_error": ell_worst,
        "orientation_worst_accepted_entry_rel_error": oc_worst,
        "momentum_route_roundoff_max_rel_error": momentum_roundoff,
        "rao_caller_replay_rel_residual": rao_replay,
        "ellipsoid_ladder": [
            {
                "panels": r["panels"],
                "max_entry_rel_error": r["max_entry_rel_error"],
                "volume_rel_error": r["volume_rel_error"],
            }
            for r in ell_ladder
        ],
        "orientation_levels": [
            {
                "subdivisions": r["subdivisions"],
                "panels": r["panels"],
                "max_entry_rel_error": r["max_entry_rel_error"],
                "within_tolerance": r["within_tolerance"],
            }
            for r in oc["levels"]
        ],
        "figure": fig_name,
        "criterion_met": worst < TOLERANCE,
    }

    (OUT / "verification.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("wrote", OUT / "verification.json")
    print(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
