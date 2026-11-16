"""Dense tail probe: is the band-top damping slope ``p`` identifiable at all?

Gate ``C9`` decides whether a DOF's truncation budget is finite by measuring the
log-log slope of the diagonal damping at the band top: ``p > 2.05`` means the
omitted integral ``int^inf B(s) ds/(s^2 - omega^2)`` converges.  On the declared
55-node grid (0.2 .. 20 rad/s) the top window is only three samples, spanning
``log10(20/16.86) = 0.074`` decades.  A slope fitted over 7% of a decade is
dominated by local curvature and by the panel-discretisation wiggle of ``B`` -- so
the *number* ``p`` may be reported to three digits while being physically
unidentifiable.

This probe separates the two questions:

1. **What is p?** -- solve the same hull on the same certified mesh, but on a
   dense grid inside the tail band, so a fit can span enough of a decade for the
   local curvature to show up in the residual.
2. **Is p identifiable?** -- report, per DOF, the slope as a function of the fit
   span (top 3, 5, 9, 13, 25 nodes) together with the fit quality (``R^2`` and the
   relative residual of the log-log line).  A slope that moves by more than the
   distance to the ``2.05`` bar when the span changes is not a decision, it is a
   fit artefact -- which is exactly what the round-9 plan asserted away by quoting
   ``p(20) ~ 3.0``.

The probe is *not* a substitute for the declared sweep and never feeds the published
gates: it answers whether those gates can be read at all.  Nothing here is on the
``C2``-violating meshes of the earlier probe -- the mesh is the certified ``60 x 16``.

Usage::

    PYTHONPATH=src python3 scripts/cummins_tail_dense_probe.py --workers 8
    PYTHONPATH=src python3 scripts/cummins_tail_dense_probe.py --analyze job.log
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np

WORKTREE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKTREE / "src"))

from tensorbem.benchmarks import case_bem_cummins_wigley as case  # noqa: E402
from tensorbem.causality import local_tail_exponent  # noqa: E402

MESH = (60, 16)
DENSE_BAND = (10.0, 20.0)
DENSE_N_OMEGA = 25
CASE_DIR = WORKTREE / "benchmarks" / "verified" / "casepool_bem_cummins_wigley"
OUTPUT = CASE_DIR / "tail_identifiability_probe.json"
SLOPE_BAR = 2.05
#: Fit spans, counted in samples from the band top.
SPANS = (3, 5, 9, 13, 25)


def grid() -> np.ndarray:
    return case.sweep_grid(DENSE_BAND, DENSE_N_OMEGA)


def _solve(task):
    index, omega = task
    n_sections, n_levels = MESH
    om, mass, damper, residual, panels, scale = case._solve_one(
        (n_sections, n_levels, omega)
    )
    return {
        "index": int(index),
        "omega": float(om),
        "damping_diagonal": [
            float(damper[i][i]) for i in range(damper.shape[0])
        ],
        "added_mass_diagonal": [
            float(mass[i][i]) for i in range(mass.shape[0])
        ],
        "solver_relative_residual": float(residual),
        "panel_count": int(panels),
        "max_panel_scale": float(scale),
        "n_sections": int(n_sections),
        "n_levels": int(n_levels),
    }


def solve(workers: int) -> int:
    nodes = grid()
    todo = [(k, float(omega)) for k, omega in enumerate(nodes)]
    print(
        f"dense tail probe {MESH[0]}x{MESH[1]}, {len(todo)} nodes over "
        f"{DENSE_BAND} rad/s, {workers} workers",
        flush=True,
    )
    with mp.Pool(processes=max(1, workers)) as pool:
        for record in pool.imap_unordered(_solve, todo):
            print(
                f"TAIL {record['index']:02d} {record['omega']:.6f} "
                + json.dumps(record, sort_keys=True),
                flush=True,
            )
    return 0


def _loglog_fit(omega: np.ndarray, damping: np.ndarray) -> dict:
    """Slope of ``log B`` vs ``log omega`` with its fit quality."""
    x = np.log(np.asarray(omega, dtype=float))
    y = np.log(np.asarray(damping, dtype=float))
    if x.size < 2 or not np.all(np.isfinite(y)):
        return {"slope": float("nan"), "r_squared": float("nan"),
                "max_relative_residual": float("nan"),
                "decades": float("nan"), "n": int(x.size)}
    coefficients = np.polyfit(x, y, 1)
    log_log_slope = float(coefficients[0])
    fitted = np.polyval(coefficients, x)
    residual = y - fitted
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / ss_tot if ss_tot > 0 else 1.0
    return {
        # ``tensorbem.causality.local_tail_exponent`` reports the decay exponent
        # ``p`` (positive for ``B ~ omega^-p``); keep that convention so a row of
        # this probe can be read next to a C9 row without a sign flip.
        "slope": -log_log_slope,
        "log_log_slope": log_log_slope,
        "r_squared": r_squared,
        "max_relative_residual": float(np.max(np.abs(np.expm1(residual)))),
        "decades": float(x[-1] - x[0]) / np.log(10.0),
        "n": int(x.size),
    }


def analyze(records: dict[int, dict]) -> dict:
    nodes = grid()
    missing = [k for k in range(nodes.size) if k not in records]
    if missing:
        raise SystemExit(f"probe incomplete: nodes {missing[:5]} missing")
    omega = np.array([float(records[k]["omega"]) for k in range(nodes.size)])
    damping = np.array(
        [np.asarray(records[k]["damping_diagonal"], dtype=float) for k in range(nodes.size)]
    )
    residual = np.array(
        [float(records[k]["solver_relative_residual"]) for k in range(nodes.size)]
    )
    panels = {int(records[k]["panel_count"]) for k in range(nodes.size)}
    rows = []
    for index, name in enumerate(case.DOF_NAMES):
        column = damping[:, index]
        spans = {}
        for span in SPANS:
            span = min(int(span), omega.size)
            spans[str(span)] = _loglog_fit(omega[-span:], column[-span:])
        windowed = {}
        for window in (3, 4, 5):
            slope, count = local_tail_exponent(omega, column, window=window)
            windowed[str(window)] = {
                "slope": float(slope),
                "n_used": int(count),
                "decades": float(
                    (np.log(omega[-1]) - np.log(omega[-min(window, omega.size)]))
                    / np.log(10.0)
                ),
            }
        slopes = [span["slope"] for span in spans.values() if np.isfinite(span["slope"])]
        rows.append(
            {
                "dof": int(index),
                "dof_name": name,
                "damping_top": float(column[-1]),
                "damping_at_10": float(column[0]),
                "slope_by_span": spans,
                "slope_window_3_4_5": windowed,
                "slope_spread_over_spans": (
                    float(max(slopes) - min(slopes)) if slopes else float("nan")
                ),
                "decidable_long_span": bool(
                    np.isfinite(spans[str(min(SPANS[-1], omega.size))]["slope"])
                    and spans[str(min(SPANS[-1], omega.size))]["slope"] > SLOPE_BAR
                ),
                "decidable_short_span": bool(
                    np.isfinite(spans["3"]["slope"]) and spans["3"]["slope"] > SLOPE_BAR
                ),
            }
        )
    return {
        "probe": (
            "dense tail probe: is the band-top damping slope identifiable, and does "
            "it exceed the algebraic-decay bar?"
        ),
        "mesh": {"n_sections": MESH[0], "n_levels": MESH[1], "panel_count": sorted(panels)},
        "band": [float(omega[0]), float(omega[-1])],
        "n_omega": int(omega.size),
        "slope_bar": SLOPE_BAR,
        "solver_relative_residual_max": float(np.max(residual)),
        "c2_panel_scale_max": float(
            max(float(records[k]["max_panel_scale"]) for k in range(nodes.size))
        ),
        "spans_samples": list(SPANS),
        "per_dof": rows,
        "verdict": _verdict(rows),
    }


def _verdict(rows) -> dict:
    flips = [
        row["dof_name"]
        for row in rows
        if row["decidable_long_span"] != row["decidable_short_span"]
    ]
    unstable = [
        row["dof_name"]
        for row in rows
        if np.isfinite(row["slope_spread_over_spans"])
        and row["slope_spread_over_spans"] > abs(SLOPE_BAR)
    ]
    return {
        "span_sensitive_dof": flips,
        "slope_spread_exceeds_bar": unstable,
        "reading": (
            "the band-top slope is not identifiable on this band: "
            + ", ".join(sorted(set(flips) | set(unstable)))
            if flips or unstable
            else "the band-top slope is stable under the fit span"
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--analyze", default=None, help="job log with TAIL lines")
    parser.add_argument("--out", default=str(OUTPUT))
    args = parser.parse_args(argv)
    if args.analyze is None:
        return solve(args.workers)
    records = {}
    for line in pathlib.Path(args.analyze).read_text().splitlines():
        if line.startswith("TAIL "):
            record = json.loads(line.split(" ", 3)[3])
            records[int(record["index"])] = record
    payload = analyze(records)
    target = pathlib.Path(args.out)
    target.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {target}")
    print(f"reading: {payload['verdict']['reading']}")
    for row in payload["per_dof"]:
        print(
            f"  {row['dof_name']:6s} p(3)={row['slope_by_span']['3']['slope']:7.3f} "
            f"p(9)={row['slope_by_span']['9']['slope']:7.3f} "
            f"p(all)={row['slope_by_span'][str(min(SPANS[-1], len(records)))]['slope']:7.3f} "
            f"spread={row['slope_spread_over_spans']:6.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())