#!/usr/bin/env python3
"""Board-audit replay driver for CASE-BEM-MARUO-DRIFT (mission m17276, audited by m17327).

Purpose
-------
Independently re-execute the *route under test* of the delivered case:

  * the four production-mesh cells (ka = 0.2 / 0.5 / 1.0 / 2.0, mesh "fine"
    32x12 grading 3.0, 800 constant panels) -> bit-level replay of the delivery,
  * the declared "finer" rung at ka = 2.0 (48x12, 1200 panels) -> the last rung
    the delivery reported,
  * two rungs *beyond* anything the delivery ran (48x16 = 1584 panels and
    64x16 = 2112 panels) at ka = 2.0, the one cell that fails the <5 % gate.

The point of the extra rungs is to answer one audit question the delivery leaves
open: is the 5.33 % deficit at ka = 2.0 (a) mesh-limited, i.e. it falls below 5 %
with a finer mesh, or (b) an asymptote, i.e. a systematic discrepancy between the
panel route and the closed form?  Nothing here modifies the delivered case; the
driver only calls the delivered public API ``bem_case`` with different meshes.

Usage (inside a TensorBEM checkout, PYTHONPATH=src)
    python3 scripts/audit/m17327_bem_rerun.py --out audit_out/m17327_cells.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

from tensorbem.benchmarks.maruo_drift_far_field_case import bem_case

#: (tag, ka, panels, levels, grading) -- first four = replay, last three = audit
CELLS: tuple[tuple[str, float, int, int, float], ...] = (
    ("replay_fine_ka0.2", 0.2, 32, 12, 3.0),
    ("replay_fine_ka0.5", 0.5, 32, 12, 3.0),
    ("replay_fine_ka1.0", 1.0, 32, 12, 3.0),
    ("replay_fine_ka2.0", 2.0, 32, 12, 3.0),
    ("replay_finer_ka2.0", 2.0, 48, 12, 3.0),
    ("audit_48x16_ka2.0", 2.0, 48, 16, 3.0),
    ("audit_64x16_ka2.0", 2.0, 64, 16, 3.0),
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="audit_out/m17327_cells.json")
    ap.add_argument("--only", nargs="*", default=None, help="subset of cell tags")
    args = ap.parse_args(argv)

    wanted = set(args.only) if args.only else None
    rows = []
    for tag, ka, panels, levels, grading in CELLS:
        if wanted and tag not in wanted:
            continue
        t0 = time.time()
        cell = bem_case(ka, panels=panels, levels=levels, grading=grading)
        cell["tag"] = tag
        cell["wall_seconds"] = time.time() - t0
        rows.append(cell)
        print(
            f"[{tag}] panels={cell['panel_count']:5d} ka={ka:.2f} "
            f"bem={cell['bem_force']:.10g} ref={cell['ref_force']:.10g} "
            f"rel={100.0 * cell['rel_error']:.5f}% "
            f"res={cell['solver_relative_residual']:.2e} "
            f"({cell['wall_seconds']:.0f}s)",
            flush=True,
        )

    payload = {
        "driver": os.path.abspath(__file__),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cells": rows,
    }
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print("[driver] wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())