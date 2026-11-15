"""Publish the route-spread audit of CASE-BEM-CUMMINS-WIGLEY.

Replays the free-decay closure on every restoring mode from the committed
frequency sweep plus the committed ``decay_dof_audit.json`` (no BEM system is
solved here) and writes
``benchmarks/verified/casepool_bem_cummins_wigley/route_spread_audit.json``.

Usage::

    PYTHONPATH=src python3 scripts/case_cummins_wigley_route_spread.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorbem.benchmarks import case_bem_cummins_wigley as case  # noqa: E402
from tensorbem.benchmarks import cummins_wigley_route_spread as spread  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SWEEP = "benchmarks/verified/casepool_bem_cummins_wigley/sweep_medium.npz"
DECAY = "benchmarks/verified/casepool_bem_cummins_wigley/decay_dof_audit.json"
OUTPUT = "benchmarks/verified/casepool_bem_cummins_wigley/route_spread_audit.json"


def main() -> int:
    started = time.time()
    sweep = case.load_sweep(ROOT / SWEEP)
    decay = json.loads((ROOT / DECAY).read_text())
    # Route spread is a between-route statement, so it is held to the case
    # tolerance (10%); the stricter 5% pair bar of gate C5 is written into the
    # artifact as the value the conclusion would have under it (nothing is
    # decidable then, which is itself the finding).
    bar = float(case.TOLERANCES["kk"])
    modes = {}
    for dof in spread.DECAY_MODES:
        name = case.DOF_NAMES[dof]
        modes[name] = spread.route_spread(
            sweep, dof, decay_row=decay["modes"][name]["values"], bar=bar
        )
    leakage = spread.block_leakage(sweep)
    block_modes = {
        name: spread.coupled_block_modes(
            sweep,
            name,
            guesses=[modes[mode]["omega_measured"] for mode in modes
                      if case.DOF_NAMES.index(mode) in spread.SYMMETRY_BLOCKS[name]],
        )
        for name in spread.SYMMETRY_BLOCKS
    }
    audit = {
        "case": case.CASE_SLUG,
        "purpose": (
            "Measure how much of the free-decay closure residual is agreement between "
            "independent routes and how much is the single-frequency reference's own "
            "sensitivity, by adding a complex-resonance route that is not built from the "
            "measured period, and by falsifying the sampled-kernel root route."
        ),
        "sweep": SWEEP,
        "decay_artifact": DECAY,
        "symmetry": leakage,
        "coupled_block_modes": block_modes,
        "modes": modes,
        "gate": spread.route_spread_gate(modes, bar=bar),
        "artifact": {
            "case": case.CASE_SLUG,
            "sweep_summary": sweep.summary() if hasattr(sweep, "summary") else {},
            "elapsed_s": time.time() - started,
            "replay": "PYTHONPATH=src python3 scripts/case_cummins_wigley_route_spread.py",
        },
    }
    target = ROOT / OUTPUT
    target.write_text(json.dumps(audit, indent=2, default=float) + "\n")
    print(f"wrote {target} in {audit['artifact']['elapsed_s']:.1f} s")
    print(f"block leakage: forbidden={leakage['worst_forbidden_relative']:.2e} "
          f"allowed={leakage['worst_allowed_in_block_relative']:.3f}")
    for name, row in modes.items():
        print(
            f"  {name:5s} grade={row['grade']:22s} "
            f"decay={row['zeta_decay']:.5f} quad={row['routes']['quadratic_at_measured']:.5f} "
            f"resonance={row['routes']['complex_resonance']:.5f} "
            f"spread={row['route_spread']:.2%} "
            f"resid_vs_res={row['residual_decay_vs_resonance']:.2%} "
            f"| kernel-root drift over span {row['kernel_root']['spread_over_span']:.1%} "
            f"usable={row['kernel_root']['usable']}"
        )
    for name, block in block_modes.items():
        print(f"  block {name:10s} " + ", ".join(
            f"w={row['omega']:.3f} zeta={row['zeta']:.5f}" for row in block["roots"]))
    gate = audit["gate"]
    print(f"C11 passed={gate['passed']} scored={gate['values']['scored_modes']} "
          f"reported={gate['values']['reported_modes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())