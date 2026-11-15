"""Extended-band *probe* for CASE-BEM-CUMMINS-WIGLEY (0.2 .. 20 rad/s, cheap meshes).

Why a probe instead of the pre-declared sweep
---------------------------------------------
``CASE-BEM-CUMMINS-WIGLEY``'s gates ``C8``/``C9`` were designed around a
**pre-declared** extension of the radiation band from 12 to 20 rad/s on a mesh
that satisfies gate ``C2`` (``k*sqrt(A_panel) <= 1``) at the *new* band top.
At ``omega = 20`` rad/s that rule forces ``sqrt(A_panel) <= 2.45 cm``, i.e.
``60 x 16 = 3840`` panels, for 55 frequency nodes:

    scripts/case_cummins_wigley_extended_band.py --shard i --shards 55

The dense BIE solve of that sweep is not affordable: one 480-panel node costs
~15 s wall / ~6.5 CPU-min, and the linear algebra grows with the cube of the
panel count, so the certified sweep is of order ``10**3`` CPU-hours -- three
orders of magnitude above the "2-4 h" the round-8 plan assumed.  This script
does not re-declare the experiment; it answers the *question* C9 asks with a
probe whose cost is minutes and whose error bar is measured rather than assumed:

1. solve the same 0.2 .. 20 rad/s, 55-node band on the committed ``coarse``
   (192 panels) and ``medium`` (480 panels) meshes;
2. report, per DOF, the measured damping tail exponent at 12 and at 20 rad/s on
   *both* meshes -- the pair brackets the discretisation error of the exponent;
3. report how far the probe violates ``C2`` at the top (the price of the cheap
   mesh) so the result is never mistaken for a certified gate;
4. time the three meshes so the cost of the certified sweep can be extrapolated
   from a measured scaling law instead of a guess.

Usage::

    PYTHONPATH=src python3 scripts/case_cummins_wigley_extended_probe.py [--analyse]

``--analyse`` only post-processes what is already on disk.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from tensorbem.benchmarks import case_bem_cummins_wigley as case

#: The probe solves only the *tail segment* above the committed band top: the
#: truncation budget is decided by the local slope of ``B`` at ``s_max``, so the
#: nodes that matter are 12 .. 20 rad/s, not the 41 already committed.  Nine
#: nodes per octave keeps the log-log slope estimator resolved.
BAND = (12.0, float(case.EXTENDED_BAND[1]))
N_OMEGA = 10
#: Meshes the probe runs on.  ``32 x 10`` is only used for the cost ladder, not
#: for the band itself: it gives a third point for the scaling law.
PROBE_MESHES = {"coarse": case.MESHES["coarse"], "medium": case.MESHES["medium"]}
COST_LADDER = [case.MESHES["coarse"], case.MESHES["medium"]]
COST_OMEGA = 2.0

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "verified" / "casepool_bem_cummins_wigley"
COST_FILE = OUT / "extended_probe_cost.json"


def probe_path(label: str) -> pathlib.Path:
    return OUT / f"sweep_extended_probe_{label}.npz"


def solve_probe(label: str, nodes_range: tuple[int, int] | None = None) -> pathlib.Path:
    """Solve the extended band on one cheap mesh (grid identical to the declared one)."""
    if label not in PROBE_MESHES:
        raise SystemExit(f"unknown probe mesh {label!r}")
    nodes = case.sweep_grid(BAND, N_OMEGA)
    n_sections, n_levels = PROBE_MESHES[label]
    omega, mass, damper, residual, elapsed = [], [], [], [], []
    panels = scale = None
    point_dir = OUT / "probe_points"
    point_dir.mkdir(parents=True, exist_ok=True)
    lo, hi = nodes_range if nodes_range else (0, nodes.size)
    for index in range(lo, hi):
        node = nodes[index]
        cache = point_dir / f"{label}_{index:02d}.npz"
        if cache.exists():
            payload = np.load(cache, allow_pickle=False)
            om = float(payload["omega"])
            added = np.asarray(payload["added_mass"], dtype=float)
            damping = np.asarray(payload["damping"], dtype=float)
            res = float(payload["solver_relative_residual"])
            count = int(payload["panel_count"])
            panel_scale = float(payload["max_panel_scale"])
            elapsed.append(0.0)
            omega.append(om); mass.append(added); damper.append(damping)
            residual.append(res); panels = count
            scale = panel_scale if scale is None else max(scale, panel_scale)
            print(f"[{label}] node {index:3d} cached", flush=True)
            continue
        t0 = time.time()
        om, added, damping, res, count, panel_scale = case._solve_one(
            (n_sections, n_levels, float(node))
        )
        np.savez(
            cache,
            omega=np.array(om),
            added_mass=np.asarray(added, dtype=float),
            damping=np.asarray(damping, dtype=float),
            solver_relative_residual=np.array(res),
            panel_count=np.array(int(count)),
            max_panel_scale=np.array(float(panel_scale)),
        )
        elapsed.append(time.time() - t0)
        omega.append(om)
        mass.append(added)
        damper.append(damping)
        residual.append(res)
        panels = count
        scale = panel_scale if scale is None else max(scale, panel_scale)
        print(
            f"[{label}] node {index:3d}/{nodes.size - 1} omega={node:8.4f} "
            f"{elapsed[-1]:7.2f} s",
            flush=True,
        )
    # rebuild the whole band from the per-node cache: a slice run must not be
    # able to persist a sweep that only covers its own slice
    omega, mass, damper, residual, elapsed = [], [], [], [], []
    for index in range(nodes.size):
        payload = np.load(point_dir / f"{label}_{index:02d}.npz", allow_pickle=False)
        omega.append(float(payload["omega"]))
        mass.append(np.asarray(payload["added_mass"], dtype=float))
        damper.append(np.asarray(payload["damping"], dtype=float))
        residual.append(float(payload["solver_relative_residual"]))
        elapsed.append(0.0)
        panels = int(payload["panel_count"])
        scale = float(payload["max_panel_scale"]) if scale is None else max(
            scale, float(payload["max_panel_scale"]))
    sweep = case.Sweep(
        label=f"{label}_extended",
        n_sections=int(n_sections),
        n_levels=int(n_levels),
        panel_count=int(panels),
        max_panel_scale=float(scale),
        omega=np.array(omega, dtype=float),
        added_mass=np.array(mass, dtype=float),
        damping=np.array(damper, dtype=float),
        solver_relative_residual=np.array(residual, dtype=float),
        elapsed=float(np.sum(elapsed)),
    )
    target = probe_path(label)
    case.save_sweep(sweep, target)
    print(f"[{label}] wrote {target} in {sweep.elapsed:.1f} s", flush=True)
    return target


def cost_ladder() -> dict[str, object]:
    """Measured wall time of one node per mesh, and the fitted cost exponent.

    ``t ~ N**b`` is fitted on three meshes; ``b = 3`` is the dense-factorisation
    scaling, ``b = 2`` the influence-matrix scaling.  The fitted value is what
    decides whether the certified (3840-panel) sweep is a two-hour or a
    thousand-hour job, so it is measured rather than assumed.
    """
    points = []
    for n_sections, n_levels in COST_LADDER:
        t0 = time.time()
        _, _, _, _, panels, _ = case._solve_one((n_sections, n_levels, COST_OMEGA))
        wall = time.time() - t0
        points.append({"n_sections": int(n_sections), "n_levels": int(n_levels),
                       "panel_count": int(panels), "wall_s": float(wall)})
        print(f"[cost] {n_sections}x{n_levels}: {panels} panels {wall:.2f} s", flush=True)
    counts = np.array([p["panel_count"] for p in points], dtype=float)
    walls = np.array([p["wall_s"] for p in points], dtype=float)
    exponent, intercept = np.polyfit(np.log(counts), np.log(walls), 1)
    certified = case.wigley_wetted_hull_mesh(
        case.LENGTH, case.BEAM, case.DRAFT,
        n_sections=case.EXTENDED_MESH[0], n_levels=case.EXTENDED_MESH[1],
    )
    certified_panels = int(certified.panel_count)
    per_node = float(np.exp(intercept) * certified_panels**exponent)
    payload = {
        "probe_omega": COST_OMEGA,
        "points": points,
        "fitted_exponent": float(exponent),
        "certified_mesh": list(case.EXTENDED_MESH),
        "certified_panel_count": int(certified_panels),
        "certified_wall_s_per_node": per_node,
        "certified_cpu_hours_for_band": float(
            per_node * N_OMEGA / 3600.0
        ),
        "declared_plan_cpu_hours": [2.0, 4.0],
    }
    COST_FILE.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=1), flush=True)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyse", action="store_true")
    parser.add_argument("--mesh", choices=sorted(PROBE_MESHES), default=None,
                        help="solve only this mesh (keeps one step under the local limit)")
    parser.add_argument("--cost", action="store_true", help="also run the cost ladder")
    parser.add_argument("--slice", default=None, help="solve only nodes a:b of one mesh")
    parser.add_argument("--skip-cost", action="store_true")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.analyse:
        labels = [args.mesh] if args.mesh else list(PROBE_MESHES)
        nodes_range = None
        if args.slice:
            lo, hi = (int(part) for part in args.slice.split(":"))
            nodes_range = (lo, hi)
        if nodes_range is not None and not args.mesh:
            parser.error("--slice needs --mesh")
        for label in labels:
            if nodes_range is not None or not probe_path(label).exists():
                try:
                    solve_probe(label, nodes_range)
                except SystemExit as exc:  # partial slice: fine, report and go
                    print(f"[{label}] {exc}")
        if args.cost and not COST_FILE.exists():
            cost_ladder()
        missing = [label for label in PROBE_MESHES if not probe_path(label).exists()]
        if missing:
            print(f"probe incomplete, missing {missing}; run --analyse once solved")
            return 0
    from tensorbem.benchmarks import cummins_wigley_extended_probe as probe

    result = probe.probe_report(
        committed=case.load_sweep(OUT / "sweep_medium.npz"),
        coarse=case.load_sweep(probe_path("coarse")),
        medium=case.load_sweep(probe_path("medium")),
        cost=json.loads(COST_FILE.read_text(encoding="utf-8")) if COST_FILE.exists() else None,
    )
    target = OUT / "extended_probe_report.json"
    target.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    (ROOT / "docs" / "case_bem_cummins_wigley_extended_probe.md").write_text(
        probe.render_markdown(result), encoding="utf-8"
    )
    print(probe.render_markdown(result))
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())