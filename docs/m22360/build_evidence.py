"""Provenance audit for CASE-BEM-CUMMINS-WIGLEY: which committed sweep is evidence?

Two sweeps of the *same* case (same hull, same mesh identity, same 41-node band,
same code) live in this tree:

* ``benchmarks/verified/casepool_bem_cummins_wigley/sweep_{coarse,medium}.npz``
  -- the sweeps the case verdict is published from (mission m21960),
  ``max solver relative residual`` O(1e-3), ``B >= 0`` everywhere;
* ``docs/m20021/sweep_{coarse,medium}.npz`` -- the m20021 medium-mesh attempt,
  ``max residual = 1.25``, ``min B = -129.9`` (heave, 5.291 rad/s).

This script quantifies the difference frequency by frequency, replays both
through :func:`tensorbem.benchmarks.case_bem_cummins_wigley.gates_from_artifact`,
and writes ``provenance_audit.json`` + ``fig_medium_evidence.png``.  Every number
in the accompanying report is read back from these two files -- nothing is
transcribed by hand.

Run: ``PYTHONPATH=src python3 docs/m22360/build_evidence.py``
"""

from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorbem.benchmarks import case_bem_cummins_wigley as case  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HEALTHY_DIR = ROOT / "benchmarks" / "verified" / "casepool_bem_cummins_wigley"
CONTAMINATED_DIR = ROOT / "docs" / "m20021"


def _sweep_stats(sweep: case.Sweep) -> dict:
    diag = np.diagonal(sweep.damping, axis1=-2, axis2=-1)
    residual = sweep.solver_relative_residual
    scale = np.max(np.abs(diag))
    return {
        "label": sweep.label,
        "panel_count": int(sweep.panel_count),
        "max_panel_scale": float(sweep.max_panel_scale),
        "n_omega": int(sweep.omega.size),
        "residual_median": float(np.median(residual)),
        "residual_max": float(np.max(residual)),
        "n_nodes_above_c0_tolerance": int(
            np.sum(residual > case.TOLERANCES["solver_residual"])
        ),
        "min_diagonal_damping": float(np.min(diag)),
        "n_negative_diagonal": int(np.sum(diag < 0.0)),
        "damping_scale": float(scale),
        "min_diagonal_over_scale": float(np.min(diag) / scale),
        "passes_c0": bool(case.solver_validity_gate(sweep)["passed"]),
    }


def _per_frequency(healthy: case.Sweep, contaminated: case.Sweep) -> list[dict]:
    rows = []
    for k, omega in enumerate(healthy.omega):
        bh = healthy.damping[k, 2, 2]
        bc = contaminated.damping[k, 2, 2]
        rows.append(
            {
                "omega": float(omega),
                "B33_healthy": float(bh),
                "B33_contaminated": float(bc),
                "B33_rel_deviation": float(abs(bc - bh) / max(abs(bh), 1e-12)),
                "residual_healthy": float(healthy.solver_relative_residual[k]),
                "residual_contaminated": float(contaminated.solver_relative_residual[k]),
            }
        )
    return rows


def _verdict_summary(replay: dict) -> dict:
    gates = replay["gates"]
    keep = {}
    for key, gate in gates.items():
        if gate is None:
            keep[key] = None
            continue
        values = gate.get("values") or {}
        keep[key] = {
            "passed": bool(gate.get("passed")),
            "voided": bool(gate.get("voided", False)),
            "kv": {
                name: values[name]
                for name in (
                    "worst_relative_residual",
                    "solver_relative_residual_max",
                    "min_radiation_damping",
                    "k_times_panel_scale",
                    "zeta_decay",
                    "zeta_from_damping",
                    "relative_error",
                )
                if name in values
            },
        }
    return {
        "verdict": replay["verdict"],
        "gates": keep,
        "sweep": replay["sweep"],
    }


def main() -> int:
    payload: dict = {
        "case": case.CASE_SLUG,
        "healthy_dir": str(HEALTHY_DIR.relative_to(ROOT)),
        "contaminated_dir": str(CONTAMINATED_DIR.relative_to(ROOT)),
        "c0_tolerance": case.TOLERANCES["solver_residual"],
        "meshes": {},
    }
    sweeps = {}
    for label in ("coarse", "medium"):
        healthy = case.load_sweep(HEALTHY_DIR / f"sweep_{label}.npz")
        contaminated = case.load_sweep(CONTAMINATED_DIR / f"sweep_{label}.npz")
        same_identity = (
            healthy.panel_count == contaminated.panel_count
            and healthy.max_panel_scale == contaminated.max_panel_scale
            and healthy.omega.shape == contaminated.omega.shape
            and bool(np.allclose(healthy.omega, contaminated.omega))
        )
        payload["meshes"][label] = {
            "same_mesh_identity_and_grid": bool(same_identity),
            "healthy": _sweep_stats(healthy),
            "contaminated": _sweep_stats(contaminated),
            "per_frequency": _per_frequency(healthy, contaminated),
        }
        sweeps[label] = (healthy, contaminated)

    # replay both artifacts: the healthy one publishes the verdict, the
    # contaminated one is retained as gate C0's negative control
    payload["replay_healthy_medium"] = _verdict_summary(
        case.gates_from_artifact(HEALTHY_DIR / "sweep_medium.npz")
    )
    payload["replay_contaminated_medium"] = _verdict_summary(
        case.gates_from_artifact(CONTAMINATED_DIR / "sweep_medium.npz")
    )

    (HERE / "provenance_audit.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    # ---------------------------------------------------------------- figure
    healthy, contaminated = sweeps["medium"]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    ax = axes[0]
    ax.plot(healthy.omega, healthy.damping[:, 2, 2], "o-", ms=3.2, lw=1.2,
            color="#1f77b4", label="healthy sweep (verified, C0 pass)")
    ax.plot(contaminated.omega, contaminated.damping[:, 2, 2], "s--", ms=3.0, lw=1.0,
            color="#d62728", label="m20021 sweep (C0 fail)")
    ax.axhline(0.0, color="k", lw=0.7)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel(r"$\omega$ [rad/s]")
    ax.set_ylabel(r"$B_{33}$ [kg/s] (heave damping)")
    ax.set_title("same mesh, same band: heave damping")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    ax.semilogy(healthy.omega, healthy.solver_relative_residual, "o-", ms=3.2, lw=1.2,
                color="#1f77b4", label="healthy")
    ax.semilogy(contaminated.omega, contaminated.solver_relative_residual, "s--",
                ms=3.0, lw=1.0, color="#d62728", label="m20021")
    ax.axhline(case.TOLERANCES["solver_residual"], color="k", ls=":", lw=1.2,
               label=f"C0 bar = {case.TOLERANCES['solver_residual']:g}")
    ax.set_xlabel(r"$\omega$ [rad/s]")
    ax.set_ylabel("linear-solve relative residual")
    ax.set_title("what gate C0 reads")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax = axes[2]
    rows = payload["meshes"]["medium"]["per_frequency"]
    dev = np.array([max(row["B33_rel_deviation"], 1e-16) for row in rows])
    ax.semilogy(healthy.omega, dev, "^-", ms=3.0, lw=1.1, color="#2ca02c")
    ax.set_xlabel(r"$\omega$ [rad/s]")
    ax.set_ylabel(r"$|B_{33}^{m20021}-B_{33}^{healthy}|/\max|B_{33}^{healthy}|$")
    ax.set_title("deviation of the contaminated sweep")
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        "CASE-BEM-CUMMINS-WIGLEY, medium mesh (480 panels): admissible vs contaminated sweep",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(HERE / "fig_medium_evidence.png", dpi=160)
    plt.close(fig)

    print(json.dumps({
        "healthy_medium": payload["meshes"]["medium"]["healthy"],
        "contaminated_medium": payload["meshes"]["medium"]["contaminated"],
        "healthy_verdict": payload["replay_healthy_medium"]["verdict"],
        "contaminated_verdict": payload["replay_contaminated_medium"]["verdict"],
    }, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())