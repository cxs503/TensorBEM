#!/usr/bin/env python3
"""Assemble the extended-band artifacts and re-judge the case with C8/C9 attached.

Two things happen here, in this order:

1. **Assembly** -- the 55 per-node files produced by
   ``scripts/case_cummins_wigley_extended_band.py --shard i --shards 55`` are
   collected into ``sweep_extended.npz``.  Each node carries its own
   ``solver_relative_residual``, panel count and panel scale; the assembler
   refuses to write a sweep whose nodes disagree on the mesh, and it reads the
   residual under the name the solver actually writes (a missing key used to be
   defaulted to ``0.0``, which silently certified a non-converged sweep through
   gate C0).
2. **Re-judging** -- the **committed medium sweep** is replayed with the extended
   sweep attached, so gates ``C8`` (shared-band consistency), ``C3a_extended``
   (zero-free-parameter Kramers-Kronig closure on the wider band) and ``C9``
   (does the damping reach algebraic decay, i.e. does the truncation budget that
   made surge/sway/yaw undecidable at 12 rad/s become finite) join the published
   verdict.  ``case_result.json`` and ``CASE.md`` are rewritten from that replay,
   never hand-edited: ``CASE.md`` stays the render of the JSON.

Usage::

    PYTHONPATH=src python3 scripts/assemble_cummins_extended.py
    PYTHONPATH=src python3 scripts/assemble_cummins_extended.py --mission m22587
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

WORKTREE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKTREE / "src"))

from tensorbem.benchmarks import case_bem_cummins_wigley as case  # noqa: E402

CASE_DIR = WORKTREE / "benchmarks" / "verified" / "casepool_bem_cummins_wigley"
EXTENDED_POINTS_DIR = CASE_DIR / "extended_points"
EXTENDED_SWEEP_PATH = CASE_DIR / "sweep_extended.npz"
ANCHOR_LADDER_PATH = CASE_DIR / "anchor_ladder_extended.json"
COMMITTED_SWEEP_PATH = CASE_DIR / "sweep_medium.npz"
CASE_RESULT_PATH = CASE_DIR / "case_result.json"
CASE_MD_PATH = CASE_DIR / "CASE.md"


def assemble_extended_sweep(
    points_dir: pathlib.Path = EXTENDED_POINTS_DIR,
    output_path: pathlib.Path = EXTENDED_SWEEP_PATH,
) -> "case.Sweep":
    """Concatenate ``point_*.npz`` into one ``Sweep``; all nodes are required.

    Ordering is by *frequency*, not by file name: a shard writes the node index it
    was given, and a re-solved node keeps its index, so sorting the files would
    also work -- sorting the ``omega`` column is simply the check that they agree.
    """
    point_files = sorted(points_dir.glob("point_*.npz"))
    if not point_files:
        raise SystemExit(
            f"no point_*.npz in {points_dir}; run "
            "scripts/case_cummins_wigley_extended_band.py --shard i --shards n first"
        )
    omega, mass, damper, residual = [], [], [], []
    meshes, panels, scales = set(), set(), []
    for path in point_files:
        payload = np.load(path, allow_pickle=False)
        omega.append(float(payload["omega"]))
        mass.append(np.asarray(payload["added_mass"], dtype=float))
        damper.append(np.asarray(payload["damping"], dtype=float))
        # The solver writes ``solver_relative_residual``; defaulting a missing key
        # to 0.0 would hand gate C0 a converged-looking sweep that never was.
        residual.append(float(payload["solver_relative_residual"]))
        meshes.add((int(payload["n_sections"]), int(payload["n_levels"])))
        panels.add(int(payload["panel_count"]))
        scales.append(float(payload["max_panel_scale"]))
    if len(meshes) != 1 or len(panels) != 1:
        raise SystemExit(f"shards disagree on the mesh: {sorted(meshes)}, panels {sorted(panels)}")
    order = np.argsort(np.asarray(omega, dtype=float))
    n_sections, n_levels = meshes.pop()
    sweep = case.Sweep(
        label="extended",
        n_sections=int(n_sections),
        n_levels=int(n_levels),
        panel_count=int(panels.pop()),
        max_panel_scale=float(max(scales)),
        omega=np.asarray(omega, dtype=float)[order],
        added_mass=np.asarray(mass, dtype=float)[order],
        damping=np.asarray(damper, dtype=float)[order],
        solver_relative_residual=np.asarray(residual, dtype=float)[order],
        elapsed=0.0,
    )
    if np.any(np.diff(sweep.omega) <= 0.0):
        raise SystemExit("the assembled frequency grid is not strictly increasing")
    case.save_sweep(sweep, output_path)
    return sweep


def rejudge(*, mission: str) -> dict:
    """Replay the committed sweep with the extended artifact attached, then publish."""
    replay = case.gates_from_artifact(
        COMMITTED_SWEEP_PATH,
        extended_path=EXTENDED_SWEEP_PATH,
    )
    published = (
        json.loads(CASE_RESULT_PATH.read_text(encoding="utf-8"))
        if CASE_RESULT_PATH.exists()
        else {}
    )
    # Only the judged fields are replaced: the hull constants, the settings and the
    # committed sweep summaries describe the *solve*, which this script does not redo.
    published["gates"] = replay["gates"]
    published["verdict"] = replay["verdict"]
    published["extended"] = replay["extended"]
    published.setdefault("case", replay["case"])
    published["gate_replay"] = {
        "mission": mission,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "replayed_artifact": COMMITTED_SWEEP_PATH.name,
        "extended_artifact": EXTENDED_SWEEP_PATH.name,
        "extended_anchor_artifact": (
            ANCHOR_LADDER_PATH.name if ANCHOR_LADDER_PATH.exists() else None
        ),
        "note": (
            "gates re-judged from the committed coefficients; no BEM solve is "
            "redone except the A_inf anchor of gates C3a/C7"
        ),
    }
    CASE_RESULT_PATH.write_text(
        json.dumps(published, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    CASE_MD_PATH.write_text(case.render_markdown(published), encoding="utf-8")
    return published


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mission", default="m22587")
    parser.add_argument(
        "--skip-assembly",
        action="store_true",
        help="re-judge the already assembled sweep_extended.npz",
    )
    args = parser.parse_args(argv)

    if not args.skip_assembly or not EXTENDED_SWEEP_PATH.exists():
        sweep = assemble_extended_sweep()
        print(
            f"assembled {sweep.omega.size} nodes on {sweep.n_sections}x{sweep.n_levels} "
            f"({sweep.panel_count} panels), band "
            f"{sweep.omega[0]:.3f}..{sweep.omega[-1]:.3f} rad/s, "
            f"max solver residual {float(np.max(sweep.solver_relative_residual)):.3e}"
        )
    else:
        sweep = case.load_sweep(EXTENDED_SWEEP_PATH)
        print(f"reusing {EXTENDED_SWEEP_PATH} ({sweep.omega.size} nodes)")

    if not ANCHOR_LADDER_PATH.exists():
        print(
            "warning: no anchor_ladder_extended.json -- C3a_extended falls back to the "
            "band-fitted offset (weaker evidence; the gate records the fallback)"
        )

    published = rejudge(mission=args.mission)
    verdict = published["verdict"]
    print(f"verdict: {verdict['gates_passed']}/{verdict['gates_total']} gates passed")
    for key in ("C8", "C3a_extended", "C9"):
        gate = published["gates"].get(key)
        print(
            f"  {key}: {'PASS' if gate and gate['passed'] else 'FAIL'} "
            f"({gate['status'] if gate else 'not evaluated'})"
        )
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())