"""Image-method ``A_inf`` anchor on the meshes used by the extended band.

Gate ``C7`` compares the least-squares ``A_inf`` of the band closure with an
independent boundary-value solve on the *same* mesh, so widening the band to 20
rad/s on a 3840-panel hull needs an anchor on that mesh too.  The mirror solve
doubles the panel count (antisymmetric image about the calm waterplane), which is
why these rungs run as separate cluster jobs.

    PYTHONPATH=src python3 scripts/case_cummins_wigley_extended_anchor.py --shard 0 --shards 2
    PYTHONPATH=src python3 scripts/case_cummins_wigley_extended_anchor.py --collect
"""
from __future__ import annotations

import argparse
import json
import pathlib

from tensorbem.benchmarks import case_bem_cummins_wigley as case

RUNGS = ((44, 14), (60, 16))

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "verified" / "casepool_bem_cummins_wigley"
POINTS = OUT / "anchor_extended_points"
LADDER = OUT / "anchor_ladder_extended.json"


def point_path(index: int) -> pathlib.Path:
    return POINTS / f"anchor_{int(index):02d}.npz"


def solve_rung(index: int) -> pathlib.Path:
    n_sections, n_levels = RUNGS[int(index)]
    anchor = case.added_mass_inf_anchor(int(n_sections), int(n_levels))
    POINTS.mkdir(parents=True, exist_ok=True)
    target = point_path(index)
    import numpy as np

    np.savez(
        target,
        n_sections=np.array(int(n_sections)),
        n_levels=np.array(int(n_levels)),
        anchor=np.array(json.dumps(anchor)),
    )
    return target


def collect() -> pathlib.Path:
    import numpy as np

    rungs = []
    for index in range(len(RUNGS)):
        if not point_path(index).exists():
            raise SystemExit(f"missing anchor rung {RUNGS[index]}")
        payload = np.load(point_path(index), allow_pickle=False)
        rungs.append(json.loads(str(payload["anchor"].item())))
    names = case.DOF_NAMES
    convergence = {}
    for dof in names:
        values = [rung["added_mass_inf"][dof] for rung in rungs]
        step = (
            abs(values[-1] - values[-2]) / abs(values[-1]) if len(values) > 1 else None
        )
        convergence[dof] = {"values": values, "top_rung_relative_change": step}
    LADDER.write_text(
        json.dumps(
            {
                "mission": "m21533",
                "purpose": "A_inf anchor ladder on the extended-band meshes",
                "rungs": rungs,
                "convergence": convergence,
            },
            indent=1,
        )
        + "\n"
    )
    return LADDER


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args(argv)
    if args.collect:
        print(collect())
        return 0
    if args.shard is None:
        parser.error("pass --shard i (with --shards n) or --collect")
    total = max(1, int(args.shards))
    for index in range(int(args.shard), len(RUNGS), total):
        print(f"solving anchor rung {index} {RUNGS[index]}", flush=True)
        print(f"  -> {solve_rung(index)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())