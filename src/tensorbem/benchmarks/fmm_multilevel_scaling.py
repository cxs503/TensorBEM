"""Measured cost scaling of the multi-level FMM (m17506).

The single-level FMM audited in ``docs/fmm_cost_audit`` keeps one octree level, so
its complexity is still ``O(N**2 / max_leaf)``; the ledger criterion
"measured slope in ``[0.9, 1.2]`` for ``N in [1e3, 1e5]``" could not be met there.
This study measures the *multi-level* operator in
:mod:`tensorbem.fmm_multilevel` (P2M / M2M / M2L / L2L / L2P + leaf near sums):

    PYTHONPATH=src python -m tensorbem.benchmarks.fmm_multilevel_scaling --figures

Predeclared criteria (written before the numbers were seen):

* **accuracy** -- max relative error against the dense single-layer sum must be
  ``<= 1e-3`` (the ledger's criterion (a), unchanged);
* **complexity** -- the log-log slope of the steady-state matvec time over
  ``N in {1280, 5120, 20480, 81920}`` must lie in ``[0.9, 1.2]`` (criterion (c));
* **speed** -- at the largest ``N`` where the dense reference is still affordable
  the ratio ``t_fmm / t_dense`` must be ``< 1``;
* the estimator is the **median of three wall-clock matvec calls in steady state**
  (geometry and translation tables cached).  The one-off setup cost is reported
  separately and is *not* amortised into the ratio.

The geometry is the BEM use case: panel centroids of a subdivided icosphere
(unit sphere) carrying the panel areas as charges, targets on the sphere vertices
(so no self-singularity), i.e. the same object as ``direct_laplace_matvec``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from ..fmm import direct_laplace_matvec
from ..fmm_multilevel import MultiLevelFMM

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "docs" / "fmm_multilevel_scaling"

SIZES = (1280, 5120, 20480, 81920)
DEFAULT_SIZES = (1280, 5120, 20480)
SUBDIVISIONS = {1280: 3, 5120: 4, 20480: 5, 81920: 6}
ORDERS = (6, 8)
MAX_LEAF = 32
ACCURACY_LIMIT = 1e-3
SLOPE_BAND = (0.9, 1.2)


def icosphere(subdivisions: int):
    """Panel centroids, panel areas and vertices of a subdivided icosphere."""
    t = (1.0 + 5.0 ** 0.5) / 2.0
    verts = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    points = [np.asarray(v, dtype=float) / np.linalg.norm(v) for v in verts]
    cache: dict[tuple[int, int], int] = {}

    def midpoint(a: int, b: int) -> int:
        key = (min(a, b), max(a, b))
        if key not in cache:
            m = points[a] + points[b]
            cache[key] = len(points)
            points.append(m / np.linalg.norm(m))
        return cache[key]

    for _ in range(subdivisions):
        refined = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            refined += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = refined
    vertices = np.asarray(points)
    centroids = np.asarray([(vertices[a] + vertices[b] + vertices[c]) / 3.0 for a, b, c in faces])

    def arc(p, q):
        return math.acos(float(np.clip(np.dot(p, q) / np.linalg.norm(p) / np.linalg.norm(q), -1.0, 1.0)))

    areas = []
    for a, b, c in faces:
        va, vb, vc = vertices[a], vertices[b], vertices[c]
        sa, sb, sc = arc(vb, vc), arc(va, vc), arc(va, vb)
        s = 0.5 * (sa + sb + sc)
        areas.append(
            4.0 * math.atan(math.sqrt(max(
                math.tan(s / 2) * math.tan((s - sa) / 2) * math.tan((s - sb) / 2) * math.tan((s - sc) / 2),
                0.0,
            )))
        )
    return centroids, np.asarray(areas), vertices


def _median_time(fn, repeats: int = 3) -> tuple[float, float]:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return float(np.median(times)), float(min(times))


def run(sizes=DEFAULT_SIZES, orders=ORDERS, max_leaf: int = MAX_LEAF) -> dict:
    rows = []
    for n_panels in sizes:
        centroids, areas, vertices = icosphere(SUBDIVISIONS[n_panels])
        if len(centroids) != n_panels:
            raise RuntimeError("expected %d panels, built %d" % (n_panels, len(centroids)))
        # accuracy is checked on a seeded 1024-vertex subsample (dense reference is
        # affordable at every size); the *cost* study uses the whole vertex set, i.e.
        # a square operator with about as many targets as sources -- the BEM case
        sample = vertices
        if len(vertices) > 1024:
            rng = np.random.default_rng(20260101)
            sample = vertices[np.sort(rng.choice(len(vertices), 1024, replace=False))]
        reference = direct_laplace_matvec(sample, centroids, areas)
        dense_time, dense_min = (None, None)
        if n_panels <= 20480:  # full-target dense still affordable here
            dense_time, dense_min = _median_time(
                lambda: direct_laplace_matvec(vertices, centroids, areas), repeats=2
            )
        for order in orders:
            operator = MultiLevelFMM(p=order, max_leaf=max_leaf)
            start = time.perf_counter()
            operator.setup(centroids)
            setup_time = time.perf_counter() - start
            start = time.perf_counter()
            first = operator.matvec(vertices, centroids, areas)
            first_time = time.perf_counter() - start
            steady, steady_min = _median_time(lambda: operator.matvec(vertices, centroids, areas))
            check = operator.matvec(sample, centroids, areas)
            error = float(np.max(np.abs(check - reference)) / np.max(np.abs(reference)))
            rows.append(
                {
                    "n_panels": n_panels,
                    "n_targets": int(len(vertices)),
                    "n_error_targets": int(len(sample)),
                    "order": order,
                    "max_leaf": max_leaf,
                    "depth": operator.depth,
                    "levels": operator.depth + 1,
                    "setup_s": setup_time,
                    "first_matvec_s": first_time,
                    "matvec_s": steady,
                    "matvec_min_s": steady_min,
                    "dense_s": dense_time,
                    "dense_min_s": dense_min,
                    "ratio": (steady / dense_time) if dense_time else None,
                    "max_rel_error": error,
                }
            )
            print(
                "N=%-6d p=%d depth=%d setup=%.2fs first=%.2fs matvec=%.3fs dense=%s err=%.1e"
                % (
                    n_panels, order, operator.depth, setup_time, first_time, steady,
                    ("%.3fs" % dense_time) if dense_time else "-", error,
                )
            )
    summary = {}
    for order in orders:
        sel = [r for r in rows if r["order"] == order]
        sel.sort(key=lambda r: r["n_panels"])
        x = np.log([r["n_panels"] for r in sel])
        y = np.log([r["matvec_s"] for r in sel])
        slope, intercept = np.polyfit(x, y, 1)
        summary[str(order)] = {
            "slope": float(slope),
            "intercept": float(intercept),
            "slope_first_quartile": float(
                np.polyfit(np.log([r["n_panels"] for r in sel[:2]]),
                           np.log([r["matvec_s"] for r in sel[:2]]), 1)[0]
            ),
            "slope_upper_half": float(
                np.polyfit(np.log([r["n_panels"] for r in sel[-2:]]),
                           np.log([r["matvec_s"] for r in sel[-2:]]), 1)[0]
            ),
            "worst_error": float(max(r["max_rel_error"] for r in sel)),
            "best_ratio": float(min(r["ratio"] for r in sel if r["ratio"])),
        }
    return {
        "mission": "m17506",
        "subject": "tensorbem.fmm_multilevel.MultiLevelFMM",
        "estimator": "median of 3 wall-clock matvec calls in steady state",
        "criteria": {
            "accuracy_limit": ACCURACY_LIMIT,
            "slope_band": list(SLOPE_BAND),
            "speed_ratio_below_one": True,
        },
        "geometry": "icosphere panel centroids (unit sphere), charges = panel areas, targets = sphere vertices",
        "rows": rows,
        "slope": summary,
    }


def _render(document: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "scaling.json").write_text(json.dumps(document, indent=2) + "\n")
    rows = document["rows"]
    lines = [
        "# Multi-level FMM: measured cost scaling (m17506)",
        "",
        "Regenerate with `PYTHONPATH=src python -m tensorbem.benchmarks.fmm_multilevel_scaling --figures`.",
        "",
        "Geometry: panel centroids of a subdivided icosphere (unit sphere), charges = panel",
        "areas, targets = sphere vertices (no self-singularity), dense reference =",
        "`tensorbem.fmm.direct_laplace_matvec`.  Estimator: median of three steady-state",
        "matvec calls (wall clock); the one-off setup is listed separately.",
        "",
        "| N | p | depth | setup (s) | first matvec (s) | matvec (s) | dense (s) | FMM/dense | max rel. err |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| %d | %d | %d | %.2f | %.2f | %.3f | %.3f | %.2f | %.1e |"
            % (
                row["n_panels"], row["order"], row["depth"], row["setup_s"],
                row["first_matvec_s"], row["matvec_s"], row["dense_s"],
                row["ratio"], row["max_rel_error"],
            )
        )
    lines += ["", "## Log-log slope of the steady-state matvec", ""]
    for order, block in sorted(document["slope"].items()):
        lines.append(
            "- p=%s: slope **%.2f** (lower half %.2f, upper half %.2f), worst error %.1e, best FMM/dense %.2f"
            % (order, block["slope"], block["slope_first_quartile"], block["slope_upper_half"],
               block["worst_error"], block["best_ratio"])
        )
    lines += [
        "",
        "Criterion (c) of the original audit (`slope in [0.9, 1.2]`) is met when the slope",
        "above falls inside that band; criterion (a) is met when every max relative error is",
        "below 1e-3.  Both are checked mechanically in `tests/test_fmm_multilevel.py`.",
        "",
        "![scaling](fig_scaling.png)",
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines))


def _figure(document: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = document["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for order in sorted({r["order"] for r in rows}):
        sel = sorted([r for r in rows if r["order"] == order], key=lambda r: r["n_panels"])
        n = [r["n_panels"] for r in sel]
        axes[0].loglog(n, [r["matvec_s"] for r in sel], "o-", label="multi-level FMM, p=%d" % order)
        axes[0].loglog(n, [r["dense_s"] for r in sel], "s--", label="dense matvec")
        axes[1].loglog(n, [r["ratio"] for r in sel], "o-", label="FMM/dense, p=%d" % order)
        axes[1].loglog(n, [r["max_rel_error"] for r in sel], "^:", label="max rel. err, p=%d" % order)
    axes[0].set_xlabel("panels N")
    axes[0].set_ylabel("wall clock per matvec [s]")
    axes[0].set_title("(a) cost vs N")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].axhline(1.0, color="k", lw=0.8, ls="--")
    axes[1].axhline(1e-3, color="r", lw=0.8, ls="--")
    axes[1].set_xlabel("panels N")
    axes[1].set_title("(b) FMM/dense cost ratio and accuracy")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_scaling.png", dpi=140)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", action="store_true", help="write README.md and fig_scaling.png")
    parser.add_argument("--json", action="store_true", help="write docs/fmm_multilevel_scaling/scaling.json")
    parser.add_argument("--large", action="store_true", help="extend the ladder to N = 81920")
    parser.add_argument("--orders", default="6,8", help="comma separated expansion orders")
    args = parser.parse_args(argv)
    sizes = SIZES if args.large else DEFAULT_SIZES
    orders = tuple(int(x) for x in args.orders.split(",") if x.strip())
    document = run(sizes=sizes, orders=orders)
    if args.json or args.figures:
        _render(document)
    if args.figures:
        _figure(document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
