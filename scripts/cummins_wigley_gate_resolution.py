"""Resolution requirement of the Cummins / KK gates, measured on an exact causal pair.

Why this exists
---------------
``CASE-BEM-CUMMINS-WIGLEY`` decides causality on a *tabulated* radiation sweep.
Two of its gates are therefore limited by the sampling before they are limited by
the solve:

* **G3** evaluates ``A(w) = A_inf + (2/pi) PV int B(s)/(s^2-w^2) ds`` at the grid
  midpoints.  The estimator is exact for the piecewise-linear interpolant, so a
  peak that is one or two samples wide is integrated as a triangle: the residual
  then reports the *sampling*, not the physics.
* **G4** compares a free-decay ``zeta`` with the half-power ``zeta``, i.e. with a
  feature ``2 zeta omega_n`` wide -- 0.074 rad/s at ``zeta = 5e-3``, ``omega_n =
  7.38`` rad/s.  A 0.5 rad/s global grid cannot represent it, which is why G4 was
  *undecidable* on the round-2 band (round 3, ``docs/dev_report_m17773``).

Both statements were previously made in prose.  This script replaces them with a
measurement that needs no panel solve at all: an **exact** causal pair is
available in closed form for the kernel ``K(t) = c exp(-sigma t) cos(Omega t)``,

    B(w)      = (c/pi) [ sigma/(sigma^2 + (w-Omega)^2) + sigma/(sigma^2 + (w+Om))
    A(w)-A_inf = -(c/(2w)) [ (w-Omega)/(sigma^2+(w-Omega)^2)
                             + (w+Omega)/(sigma^2+(w+Omega)^2) ]

so any residual measured on a sampling of it is a defect of the sampling or of
the estimator -- never of the physics.  The pair is sampled on the same grids the
case uses (a 0.5 rad/s global grid, the predeclared closure grid with its 0.0125
rad/s resonance cluster) and on a dense reference, and the three gates are
recomputed.  The output is the resolution the anchor's 10 % tolerance *requires*,
which is the design rule behind :func:`case.closure_grid`.

Usage::

    PYTHONPATH=src python3 scripts/cummins_wigley_gate_resolution.py \\
        --outdir artifacts/case_bem_cummins_wigley_resolution
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

from tensorbem import time_domain as td
from tensorbem.benchmarks import cummins_wigley_kk_causality as case
from tensorbem.benchmarks import kk_band_floor

SIGMA = 0.037          # = zeta * Omega, i.e. a lightly damped resonance
OMEGA = 7.378504598323515
COEFF = 1.0


def exact_pair(omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(A - A_inf, B)`` of ``K(t) = c exp(-sigma t) cos(Omega t)``."""
    w = np.asarray(omega, dtype=float)
    s2m = SIGMA * SIGMA + (w - OMEGA) ** 2
    s2p = SIGMA * SIGMA + (w + OMEGA) ** 2
    damping = (COEFF / math.pi) * (SIGMA / s2m + SIGMA / s2p)
    added = -(COEFF / (2.0 * w)) * ((w - OMEGA) / s2m + (w + OMEGA) / s2p)
    return added, damping


def heave_model(omega: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Exact pair expressed as a heave sweep: ``A_inf`` chosen for ``omega_n = OMEGA``."""
    ref = case.wigley3_hydrostatic_reference() if hasattr(case, "wigley3_hydrostatic_reference") \
        else None
    if ref is None:  # pragma: no cover - import path differs only in tests
        from tensorbem.wigley3_freq import wigley3_hydrostatic_reference
        ref = wigley3_hydrostatic_reference()
    mass = float(ref["volume"]) * case.DENSITY
    restoring = float(ref["C33"])
    deficit, damping = exact_pair(omega)
    # A(w) - A_inf -> 0 as w -> inf; the pair above is already relative to A_inf.
    added_inf = restoring / OMEGA**2 - mass
    # make the *undamped* resonance land on OMEGA with the mean added mass in band
    added_shift = restoring / OMEGA**2 - mass - added_inf - float(np.mean(deficit))
    return mass, restoring, added_inf + added_shift + deficit, damping


def grids() -> dict[str, np.ndarray]:
    closure = np.asarray(case.closure_grid(OMEGA), dtype=float)
    coarse = np.array([0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.5, 7.0,
                       7.4, 7.5, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0])
    global_only = np.asarray(case.closure_grid(), dtype=float)
    widths = {}
    for step in (0.5, 0.25, 0.1, 0.05, 0.025, 0.0125):
        nodes = {round(float(w), 9) for w in case.EXTENDED_GRID}
        steps = int(round(case.RESONANCE_HALF_WIDTH / step))
        for k in range(-steps, steps + 1):
            nodes.add(round(OMEGA + k * step, 9))
        widths[f"{step:g}"] = np.asarray(sorted(nodes), dtype=float)
    return {"closure": closure, "coarse": coarse, "global_only": global_only,
            **{f"cluster_{k}": v for k, v in widths.items()}}


def g3_on(grid: np.ndarray) -> dict:
    added, damping = exact_pair(grid)
    residual = kk_band_floor.kk_residual(grid, added, damping)
    return {k: residual[k] for k in ("max", "rms", "argmax_omega", "points")}


def g4_on(grid: np.ndarray) -> dict:
    _, _, added, damping = heave_model(grid)
    dense = case.decay_response_grid(grid)
    try:
        hp = td.half_power_damping(
            dense, np.interp(dense, grid, added), np.interp(dense, grid, damping),
            mass=heave_model(np.array([OMEGA]))[0],
            restoring=heave_model(np.array([OMEGA]))[1])
        return {"zeta": float(hp["zeta"]), "widenings": hp.get("scan_widenings"),
                "crossings": [float(hp.get("omega_lower", float("nan"))),
                              float(hp.get("omega_upper", float("nan")))]}
    except (ValueError, KeyError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="artifacts/case_bem_cummins_wigley_resolution")
    parser.add_argument("--out", default="resolution.json")
    args = parser.parse_args(argv)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # reference: the same estimator on a sampling so fine that its own error is
    # below the printout (20001 uniform points across the peak, plus the tails)
    reference_grid = np.unique(np.concatenate([
        np.linspace(0.1, OMEGA - 0.6, 2001),
        np.linspace(OMEGA - 0.6, OMEGA + 0.6, 20001),
        np.linspace(OMEGA + 0.6, 20.0, 2001),
    ]))
    reference = {"g3": g3_on(reference_grid), "g4": g4_on(reference_grid),
                 "points": int(reference_grid.size)}
    zeta_ref = reference["g4"].get("zeta")
    print(f"[resolution] reference: {reference_grid.size} nodes, "
          f"G3 max={reference['g3']['max']:.3e}, zeta={zeta_ref}", flush=True)

    rows = {}
    for name, grid in grids().items():
        entry = {"nodes": int(grid.size),
                 "band": [float(grid[0]), float(grid[-1])],
                 "g3": g3_on(grid),
                 "g4": g4_on(grid)}
        z = entry["g4"].get("zeta")
        entry["g4_rel_err"] = (abs(z - zeta_ref) / zeta_ref
                               if z is not None and zeta_ref else None)
        entry["g3_vs_reference"] = entry["g3"]["max"] - reference["g3"]["max"]
        rows[name] = entry
        print(f"[resolution] {name:14s} nodes={entry['nodes']:3d} "
              f"G3max={entry['g3']['max']:.4f} "
              f"G4err={entry['g4_rel_err']}", flush=True)

    payload = {
        "model": {
            "kernel": "K(t) = c exp(-sigma t) cos(Omega t)",
            "sigma": SIGMA, "Omega": OMEGA, "c": COEFF,
            "zeta_design": SIGMA / OMEGA,
            "half_power_width": 2.0 * SIGMA,
            "exact": True,
        },
        "reference": reference,
        "rows": rows,
        "gates": case.GATES,
        "closure_grid": {
            "extended": list(case.EXTENDED_GRID),
            "resonance_half_width": case.RESONANCE_HALF_WIDTH,
            "resonance_step": case.RESONANCE_STEP,
            "dense_step": case.DENSE_STEP,
            "nodes_with_cluster": len(case.closure_grid(OMEGA)),
        },
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # ---- figure -----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
        w = np.linspace(OMEGA - 0.6, OMEGA + 0.6, 4001)
        _, b_exact = exact_pair(w)
        axes[0].plot(w, b_exact, "k-", lw=1.0, label="exact causal pair")
        for name, style in (("cluster_0.0125", "C0o"), ("coarse", "C3s")):
            grid = grids()[name]
            mask = (grid > OMEGA - 0.6) & (grid < OMEGA + 0.6)
            axes[0].plot(grid[mask], exact_pair(grid)[1][mask], style, ms=3.0,
                         lw=0.8, label=f"{name} sampling")
        axes[0].axhline(float(np.max(b_exact)) / math.sqrt(2.0), color="0.5", ls="--",
                        lw=0.8, label="half-power level")
        axes[0].set_xlabel(r"$\omega$ [rad/s]")
        axes[0].set_ylabel(r"$B_{33}$ [kg/s]")
        axes[0].set_title("the feature G4 must resolve")
        axes[0].legend(fontsize=7)

        steps, g3max, g4err = [], [], []
        for name, entry in rows.items():
            if not name.startswith("cluster_"):
                continue
            steps.append(float(name.split("_")[1]))
            g3max.append(entry["g3"]["max"])
            g4err.append(entry["g4_rel_err"] if entry["g4_rel_err"] is not None
                         else np.nan)
        order = np.argsort(steps)
        steps = np.asarray(steps)[order]
        axes[1].loglog(steps, np.asarray(g3max)[order], "C0o-", label="G3 residual (max)")
        axes[1].axhline(case.GATES["G3_kramers_kronig"]["tol"], color="r", ls="--",
                        lw=0.8, label="G3 tolerance 10 %")
        axes[1].set_xlabel("resonance-cluster node spacing [rad/s]")
        axes[1].set_ylabel("relative KK residual")
        axes[1].set_title("G3: sampling, not physics")
        axes[1].legend(fontsize=7)

        axes[2].semilogx(steps, np.asarray(g4err)[order], "C3s-", label="G4 zeta error")
        axes[2].axhline(case.GATES["G4_free_decay"]["tol"], color="r", ls="--", lw=0.8,
                        label="G4 tolerance 10 %")
        axes[2].set_xlabel("resonance-cluster node spacing [rad/s]")
        axes[2].set_ylabel(r"$|\zeta-\zeta_{ref}|/\zeta_{ref}$")
        axes[2].set_title("G4: undecidable until the peak is sampled")
        axes[2].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(outdir / "fig_gate_resolution.png", dpi=140)
        payload["figure"] = str(outdir / "fig_gate_resolution.png")
        print(f"[resolution] figure -> {outdir / 'fig_gate_resolution.png'}", flush=True)
    except Exception as exc:  # pragma: no cover - figure is a convenience
        payload["figure_error"] = f"{type(exc).__name__}: {exc}"

    out = outdir / args.out
    out.write_text(json.dumps(payload, indent=1))
    print(f"[resolution] -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())