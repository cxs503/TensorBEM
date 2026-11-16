"""Solver-admissibility audit for CASE-BEM-CUMMINS-WIGLEY (gate ``G0``).

Why this script exists
----------------------
The five predeclared gates of :mod:`tensorbem.benchmarks.cummins_wigley_kk_causality`
(G1-G5) are all *consistency* statements about one sweep: cross-solver, kernel
at zero, Kramers-Kronig, free decay, round trip.  Every one of them reads the
added-mass/damping coefficients and nothing else, so none of them can see the
one thing that invalidates all of them: a node whose BEM solve never converged.

That is not a hypothetical.  ``docs/m20021/sweep_medium.npz`` -- committed as
evidence for the extended-band work -- has 41/41 nodes above the case pool's
``C0`` solver limit of 1 % (worst 1.245) and a *negative* heave damping
``B33 = -129.9`` at ``w = 5.2912`` rad/s.  Before this audit the closure
pipeline would cache such a node and hand it to the gates; the only thing that
stopped the numbers was an opaque ``ValueError`` from deep inside
``time_domain.radiation_memory_kernel``.

What it computes (no BEM solve, no interpolation -- cached artifacts only)
-------------------------------------------------------------------------
* ``G0_solver_validity`` on every in-tree Cummins-Wigley sweep: node count,
  offenders above ``--tol``, worst residual and its frequency.
* the G0-G5 gate set on each admissible sweep (``G1`` is skipped, not invented,
  when the sweep grid does not carry ``w = 1 rad/s``);
* the *silent* failure mode, quantified: one node of the committed medium sweep
  is perturbed (heave damping scaled, residual set to a diverged value) in a way
  the five consistency gates accept, and the gate movement is recorded.  If G0
  were a formality this row would show no movement.

Writes ``--out`` (default ``docs/m23106/closure_admissibility.json``); every
number in ``docs/dev_report_m23106_cummins_wigley_band_closure_round2.md`` is
read back from that file, none is typed by hand.

Usage::

    PYTHONPATH=src python3 scripts/audit_cummins_closure_admissibility.py
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from tensorbem.benchmarks import case_bem_cummins_wigley as legacy
from tensorbem.benchmarks import cummins_wigley_kk_causality as case

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: In-tree sweeps audited: (key, path, mesh level key).
SWEEPS = (
    ("committed_medium",
     ROOT / "benchmarks/verified/casepool_bem_cummins_wigley/sweep_medium.npz",
     "medium"),
    ("committed_coarse",
     ROOT / "benchmarks/verified/casepool_bem_cummins_wigley/sweep_coarse.npz",
     "coarse"),
    ("m20021_medium", ROOT / "docs/m20021/sweep_medium.npz", "medium"),
    ("m20021_coarse", ROOT / "docs/m20021/sweep_coarse.npz", "coarse"),
    ("n315_coarse",
     ROOT / "docs/casepool_bem_cummins_wigley/evidence_cluster_n315_sweep_coarse.npz",
     "coarse"),
)


def sweep_from_npz(path: pathlib.Path, level_key: str) -> dict:
    """Replay a committed ``sweep_*.npz`` as the dict the gates consume."""
    payload = np.load(path)
    n_sections, n_levels = case.MESH_LEVELS[level_key]
    nodes = payload["omega"].size
    return {
        "level": f"n_sections={n_sections},n_levels={n_levels}",
        "level_key": level_key,
        "rotation_center": list(case.ROTATION_CENTER),
        "omega": np.asarray(payload["omega"], dtype=float).tolist(),
        "added_mass": np.asarray(payload["added_mass"], dtype=float).tolist(),
        "damping": np.asarray(payload["damping"], dtype=float).tolist(),
        "relative_residual": np.asarray(
            payload["solver_relative_residual"], dtype=float).tolist(),
        "wall_seconds": [0.0] * nodes,
        "density": float(case.DENSITY),
        "gravity": float(case.GRAVITY),
    }


def _gate_summary(analysis: dict) -> dict:
    return {
        "verdict": analysis["verdict"],
        "failed": analysis["failed"],
        "admissible": analysis["admissible"],
        "interpretable": analysis["interpretable"],
        "gates": {
            key: {"value": value["value"], "tol": value["tol"],
                  "passed": value["passed"]}
            for key, value in analysis["gates"].items()
        },
        "worst_g3": {
            "dof": max(analysis["studies"],
                       key=lambda k: analysis["studies"][k]["G3"]["rel_err"]),
            "rel_err": max(s["G3"]["rel_err"] for s in analysis["studies"].values()),
        },
        "worst_g5": {
            "dof": max(analysis["studies"],
                       key=lambda k: analysis["studies"][k]["G5"]["rel_err"]),
            "rel_err": max(s["G5"]["rel_err"] for s in analysis["studies"].values()),
        },
        "per_dof": {
            name: {"G3": study["G3"]["rel_err"],
                   "G3_argmax_omega": study["G3"]["argmax_omega"],
                   "G5": study["G5"]["rel_err"],
                   "G3b": study["G3b"]}
            for name, study in analysis["studies"].items()
        },
        "free_decay_rel_err": analysis["free_decay"]["rel_err"],
    }


def _evaluate(sweep: dict, *, residual_tol: float) -> dict:
    """G0 block plus the gate set, keeping a solver refusal as a *result*."""
    admissibility = case.residual_admissibility(sweep, tol=residual_tol)
    out: dict = {"solver_admissibility": admissibility}
    try:
        analysis = case.gates_from_sweep(
            sweep,
            include_cross_solver=any(abs(w - 1.0) < 1e-6 for w in sweep["omega"]),
            residual_tol=residual_tol,
        )
    except ValueError as exc:  # a diverged node can make the kernel builder refuse
        out["gates_error"] = {"type": type(exc).__name__, "message": str(exc)}
        return out
    out["gates"] = _gate_summary(analysis)
    return out


def _perturb_one_node(sweep: dict, omega_node: float, *,
                      damping_scale: float, residual: float) -> dict:
    """Copy of ``sweep`` with one node's heave damping scaled; residual forged.

    The perturbation is chosen so the copy stays *positive* damping and therefore
    passes the only physical check the five consistency gates have
    (``radiation_memory_kernel`` refuses ``B < 0``).  This isolates the blind
    spot: a wrong-but-positive node.
    """
    copy = json.loads(json.dumps(sweep))
    omega = np.asarray(copy["omega"], dtype=float)
    index = int(np.argmin(np.abs(omega - omega_node)))
    damping = np.asarray(copy["damping"], dtype=float)
    damping[index, 2, 2] *= damping_scale
    copy["damping"] = damping.tolist()
    copy["relative_residual"][index] = float(residual)
    copy["_perturbation"] = {
        "omega": float(omega[index]),
        "dof": "heave",
        "damping_scale": damping_scale,
        "forged_residual": float(residual),
    }
    return copy


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tol", type=float, default=case.SOLVER_RESIDUAL_TOL)
    parser.add_argument("--out", default="docs/m23106/closure_admissibility.json")
    parser.add_argument("--perturb-omega", type=float, default=5.2912)
    parser.add_argument("--perturb-residual", type=float, default=1.24506058251773,
                        help="forged residual of the perturbed node (default: "
                             "the worst residual of docs/m20021/sweep_medium.npz)")
    args = parser.parse_args(argv)

    sweeps: dict[str, dict] = {}
    for key, path, level_key in SWEEPS:
        if not path.exists():
            sweeps[key] = {"missing": str(path)}
            continue
        sweeps[key] = _evaluate(sweep_from_npz(path, level_key), residual_tol=args.tol)
        weak = sweeps[key]["solver_admissibility"]
        print(f"[G0] {key:18s} nodes={weak['nodes']:3d} "
              f"offenders={weak['n_offenders']:3d} "
              f"worst={weak['worst_residual']:.3e}@{weak['worst_omega']} "
              f"passed={weak['passed']}")

    # ---- the blind spot of G1-G5: a wrong-but-positive node ----------------
    committed = sweep_from_npz(SWEEPS[0][1], SWEEPS[0][2])
    baseline = _evaluate(committed, residual_tol=args.tol)
    rows = []
    for scale in (1.0, 0.8, 0.6, 0.4, 0.2):
        poison = _perturb_one_node(committed, args.perturb_omega,
                                   damping_scale=scale,
                                   residual=args.perturb_residual)
        without = _evaluate_without_g0(poison)
        with_g0 = _evaluate(poison, residual_tol=args.tol)["gates"]
        rows.append({
            "damping_scale": scale,
            "perturbation": poison["_perturbation"],
            "without_g0": without,
            "with_g0": with_g0,
            "gates_identical_to_baseline_verdict":
                without.get("verdict") == baseline["gates"]["verdict"],
        })
    silence = {
        "baseline": baseline,
        "rows": rows,
        "note": ("one node of the committed medium band is perturbed "
                 "(heave damping scaled inside positive values, residual set to "
                 "the diverged value measured on docs/m20021/sweep_medium.npz). "
                 "The five consistency gates accept it; only G0 names the node."),
    }

    payload = {
        "case": "CASE-BEM-CUMMINS-WIGLEY",
        "gate": "G0_solver_validity",
        "residual_tol": float(args.tol),
        "gates_definition": case.GATES,
        "sweeps": sweeps,
        "silent_failure_mode": silence,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=float))
    print(f"[out] {out}")
    return 0


def _evaluate_without_g0(sweep: dict) -> dict:
    """The pre-G0 behaviour: the same sweep, gates evaluated, no admissibility."""
    try:
        analysis = case.gates_from_sweep(
            sweep,
            include_cross_solver=any(abs(w - 1.0) < 1e-6 for w in sweep["omega"]),
            include_residual_gate=False,
        )
    except ValueError as exc:
        return {"gates_error": {"type": type(exc).__name__, "message": str(exc)}}
    return _gate_summary(analysis)


if __name__ == "__main__":
    raise SystemExit(main())