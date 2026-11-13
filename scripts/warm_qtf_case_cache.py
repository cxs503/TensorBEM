"""Warm the diffraction cache for the BEM-QTF-DRIFT study, one problem at a time.

The control-plane shell is capped at ~120 s and the box has been running at
load ~300/128, so a single 360-panel cylinder solve can exceed the cap on its
own.  This runner is detachable (``setsid``) and restartable: every finished
solve is dropped into the cache directory, so an interrupted run costs nothing.

    PYTHONPATH=src setsid nohup python3 scripts/warm_qtf_case_cache.py /tmp/qcache \
        > /tmp/warm_qtf.log 2>&1 &
"""

from __future__ import annotations

import pathlib
import sys
import time

from tensorbem.benchmarks.cylinder_qtf_closed_form import (
    CYLINDER_RADIUS,
    DEPTH_FACTOR,
    KA_GRID,
    LEVELS,
    _diffraction,
)
from tensorbem.drift import bottom_mounted_cylinder_mesh


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cache = pathlib.Path(argv[0] if argv else "/tmp/qcache")
    a = CYLINDER_RADIUS
    depth = DEPTH_FACTOR / min(KA_GRID)
    jobs = []
    for level in ("medium", "coarse"):
        n_theta, n_z = LEVELS[level]
        mesh = bottom_mounted_cylinder_mesh(a, depth, n_theta=n_theta, n_z=n_z)
        for ka in KA_GRID:
            jobs.append((f"{level}-ka{ka}", mesh, ka / a, level))
    # depth sweep (diagonal at ka = 0.6, coarse level)
    n_theta, n_z = LEVELS["coarse"]
    for factor in (6.0, 12.0, 24.0):
        mesh = bottom_mounted_cylinder_mesh(
            a, factor / 0.6, n_theta=n_theta, n_z=n_z
        )
        jobs.append((f"depth{factor:.0f}", mesh, 0.6 / a, f"depth{factor:.0f}_coarse"))

    for label, mesh, k, tag in jobs:
        started = time.time()
        try:
            _diffraction(mesh, k, cache=cache, tag=tag)
            print(f"{label}: ok in {time.time() - started:.1f}s", flush=True)
        except Exception as exc:  # pragma: no cover
            print(f"{label}: FAILED {exc!r}", flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())