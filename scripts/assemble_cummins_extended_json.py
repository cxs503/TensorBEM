"""Rebuild ``sweep_extended.npz`` from the JSON node lines of a sweep job log.

The extended-band sweep (0.2 .. 20 rad/s, 60x16 = 3840 panels, 55 nodes) is solved
by ``scripts/cummins_extended_sweep_job.py``, which prints one ``NODE <index>
<omega> <json>`` line per grid node.  This assembler is the other half: it parses
those lines, refuses anything ambiguous, and writes the artifact the gates read
(``benchmarks/verified/casepool_bem_cummins_wigley/sweep_extended.npz``).

Refusals are the point of the script -- a sweep that is silently incomplete or
silently mixed-mesh would be certified by gate C0/C2 as if it were the declared
run:

* a missing node, a duplicate node, or a node outside the declared grid is fatal;
* every node must agree on ``n_sections``/``n_levels``/``panel_count``
  (``max_panel_scale`` is allowed to vary, the gates take its maximum);
* the ``omega`` field must reproduce the declared grid node to ``1e-9``
  relative, so a log from a different grid cannot masquerade as this one;
* ``solver_relative_residual`` must be present: gate C0 judges the solve on it, and
  defaulting it to zero used to certify non-converged sweeps.

Usage::

    python3 scripts/assemble_cummins_extended_json.py job.log
    python3 scripts/assemble_cummins_extended_json.py --stdin < job.log
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

WORKTREE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKTREE / "src"))

from tensorbem.benchmarks import case_bem_cummins_wigley as case  # noqa: E402

EXTENDED_MESH = (60, 16)
EXTENDED_BAND = (0.2, 20.0)
EXTENDED_N_OMEGA = 55
CASE_DIR = WORKTREE / "benchmarks" / "verified" / "casepool_bem_cummins_wigley"
SWEEP_FILE = CASE_DIR / "sweep_extended.npz"

REQUIRED = (
    "index",
    "omega",
    "added_mass",
    "damping",
    "solver_relative_residual",
    "panel_count",
    "max_panel_scale",
    "n_sections",
    "n_levels",
)


def grid() -> np.ndarray:
    return case.sweep_grid(EXTENDED_BAND, EXTENDED_N_OMEGA)


def parse_lines(lines) -> dict[int, dict]:
    """Parse ``NODE <i> <omega> <json>`` lines into ``{index: record}``."""
    nodes = grid()
    records: dict[int, dict] = {}
    for raw in lines:
        line = raw.strip()
        if not line.startswith("NODE "):
            continue
        parts = line.split(" ", 3)
        if len(parts) != 4:
            raise SystemExit(f"malformed NODE line: {line[:80]!r}")
        head_index = int(parts[1])
        record = json.loads(parts[3])
        missing = [key for key in REQUIRED if key not in record]
        if missing:
            raise SystemExit(
                f"node {head_index} is missing {missing}; refusing to assemble a sweep "
                "whose identity the gates cannot check"
            )
        if int(record["index"]) != head_index:
            raise SystemExit(
                f"line header index {head_index} disagrees with payload "
                f"{record['index']}"
            )
        if head_index in records:
            raise SystemExit(f"node {head_index} appears twice; keep one run's log")
        if not 0 <= head_index < nodes.size:
            raise SystemExit(
                f"node {head_index} is outside the declared {nodes.size}-node grid"
            )
        declared = float(nodes[head_index])
        if not np.isclose(
            float(record["omega"]), declared, rtol=1e-9, atol=0.0
        ):
            raise SystemExit(
                f"node {head_index} solved omega={record['omega']!r} but the declared "
                f"grid node is {declared!r}; the log is from a different grid"
            )
        records[head_index] = record
    return records


def assemble(records: dict[int, dict]) -> case.Sweep:
    nodes = grid()
    missing = [k for k in range(nodes.size) if k not in records]
    if missing:
        raise SystemExit(
            f"cannot assemble: {len(missing)}/{nodes.size} nodes missing "
            f"(first: {missing[:5]})"
        )
    identity = {
        (
            int(records[k]["n_sections"]),
            int(records[k]["n_levels"]),
            int(records[k]["panel_count"]),
        )
        for k in range(nodes.size)
    }
    if len(identity) != 1:
        raise SystemExit(f"nodes disagree on the mesh identity: {sorted(identity)}")
    n_sections, n_levels, panel_count = identity.pop()
    if (n_sections, n_levels) != EXTENDED_MESH:
        raise SystemExit(
            f"sweep was solved on {n_sections}x{n_levels}, the declared extended "
            f"mesh is {EXTENDED_MESH}"
        )
    omega = np.array([float(records[k]["omega"]) for k in range(nodes.size)])
    mass = np.array([np.asarray(records[k]["added_mass"], dtype=float) for k in range(nodes.size)])
    damper = np.array([np.asarray(records[k]["damping"], dtype=float) for k in range(nodes.size)])
    for name, array in (("added_mass", mass), ("damping", damper)):
        if array.shape != (nodes.size, 6, 6):
            raise SystemExit(f"{name} has shape {array.shape}, expected ({nodes.size}, 6, 6)")
        if not np.all(np.isfinite(array)):
            raise SystemExit(f"{name} contains non-finite entries")
    residual = np.array(
        [float(records[k]["solver_relative_residual"]) for k in range(nodes.size)]
    )
    scales = np.array([float(records[k]["max_panel_scale"]) for k in range(nodes.size)])
    elapsed = float(sum(float(records[k].get("wall_s", 0.0)) for k in range(nodes.size)))
    return case.Sweep(
        label="extended",
        n_sections=int(n_sections),
        n_levels=int(n_levels),
        panel_count=int(panel_count),
        max_panel_scale=float(np.max(scales)),
        omega=omega,
        added_mass=mass,
        damping=damper,
        solver_relative_residual=residual,
        elapsed=elapsed,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="?", help="job log with NODE lines")
    parser.add_argument("--stdin", action="store_true")
    parser.add_argument("--out", default=str(SWEEP_FILE))
    args = parser.parse_args(argv)

    if args.stdin or not args.log:
        lines = sys.stdin.readlines()
        source = "<stdin>"
    else:
        lines = pathlib.Path(args.log).read_text().splitlines()
        source = args.log
    records = parse_lines(lines)
    sweep = assemble(records)
    target = case.save_sweep(sweep, args.out)
    print(f"assembled {records.__len__()} nodes from {source} -> {target}")
    print(f"  mesh {sweep.n_sections}x{sweep.n_levels} ({sweep.panel_count} panels)")
    print(f"  band {sweep.omega[0]:.4f} .. {sweep.omega[-1]:.4f} rad/s")
    print(
        "  solver residual max "
        f"{float(np.max(sweep.solver_relative_residual)):.3e}, "
        f"band-top residual {float(sweep.solver_relative_residual[-1]):.3e}"
    )
    print(f"  wall clock summed over nodes {sweep.elapsed:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())