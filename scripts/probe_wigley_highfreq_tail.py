"""Probe: does the Wigley radiation damping reach an algebraic tail in band?

CASE-BEM-CUMMINS-WIGLEY gate C3 closes the Kramers-Kronig integral with a
power tail ``B ~ C s^-p`` above the top of the sweep.  The closure helper
(:func:`tensorbem.causality.fit_damping_power_tail`) declares the tail invalid
unless ``p > 3`` *and* the log-log fit is clean, and the closure then falls back
to a zero tail -- which is what biases the reconstructed added mass at the top of
the band.

This probe measures ``p`` directly: it solves the same CHIEF-regularised
radiation problem on progressively finer meshes over a high-frequency ladder and
prints, per frequency, the panel resolution number ``k*sqrt(A_panel,max)`` and
the diagonal damping.  The local log-log slope between consecutive ladder points
is the decay exponent the tail fit would see.

Run (heavy: submit through the compute gate):

    PYTHONPATH=src python3 scripts/probe_wigley_highfreq_tail.py \
        --sections 12 20 32 --levels 4 6 10 --omegas 8 10 12 14 17 20 24 30
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

from tensorbem.free_surface_solver3d import solve_radiation_3d
from tensorbem.wigley_hydrostatics import wigley_wetted_hull_mesh

LENGTH, BEAM, DRAFT = 3.0, 0.3, 0.1875
DENSITY, GRAVITY = 1025.0, 9.80665
DOF_NAMES = ["surge", "sway", "heave", "roll", "pitch", "yaw"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sections", nargs="+", type=int, default=[20, 32])
    parser.add_argument("--levels", nargs="+", type=int, default=[6, 10])
    parser.add_argument("--omegas", nargs="+", type=float, default=[8, 10, 12, 14, 17, 20])
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    rows = []
    for n_sections, n_levels in zip(args.sections, args.levels):
        mesh = wigley_wetted_hull_mesh(
            LENGTH, BEAM, DRAFT, n_sections=n_sections, n_levels=n_levels
        )
        areas = np.asarray(mesh.areas, dtype=float)
        h_max = math.sqrt(float(np.max(areas)))
        for omega in args.omegas:
            started = time.time()
            rad = solve_radiation_3d(
                mesh,
                omega**2 / GRAVITY,
                modes=tuple(range(6)),
                density=DENSITY,
                gravity=GRAVITY,
                chief_points="auto",
            )
            damp = np.diag(np.asarray(rad.damping, dtype=float))
            k = omega**2 / GRAVITY
            rows.append(
                {
                    "sections": int(n_sections),
                    "levels": int(n_levels),
                    "panels": int(mesh.panel_count),
                    "h_max_m": h_max,
                    "omega": float(omega),
                    "k_h_max": float(k * h_max),
                    "solver_residual": float(getattr(rad, "relative_residual", 0.0)),
                    "damping": damp.tolist(),
                    "wall_seconds": time.time() - started,
                }
            )
            print(
                f"{n_sections:>3d}x{n_levels:<3d} panels={mesh.panel_count:5d} "
                f"k*h={k * h_max:5.2f} w={omega:5.1f} B_heave={damp[2]:10.4g} "
                f"B_sway={damp[1]:10.4g} res={rows[-1]['solver_residual']:.2e} "
                f"{rows[-1]['wall_seconds']:5.1f}s",
                flush=True,
            )

    # local log-log slopes, per mesh, per DOF
    slopes = []
    for n_sections, n_levels in zip(args.sections, args.levels):
        block = [r for r in rows if r["sections"] == n_sections]
        for a, b in zip(block, block[1:]):
            entry = {"sections": n_sections, "omega": [a["omega"], b["omega"]], "p": {}}
            for index, name in enumerate(DOF_NAMES):
                ba, bb = a["damping"][index], b["damping"][index]
                if ba > 0.0 and bb > 0.0:
                    entry["p"][name] = -math.log(bb / ba) / math.log(
                        b["omega"] / a["omega"]
                    )
            slopes.append(entry)
            print(f"  {n_sections}x{n_levels} {a['omega']:5.1f}->{b['omega']:5.1f} p=" +
                  " ".join(f"{k}={v:6.2f}" for k, v in entry["p"].items()), flush=True)

    payload = {"rows": rows, "local_slopes": slopes}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    print(json.dumps({"n_rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())