"""Cost/accuracy audit of :mod:`tensorbem.fmm` (mission m14931, verification debt).

Why this module exists
----------------------
``tensorbem.fmm`` is graded ``REAL`` as *"Fast multipole acceleration of the 3-D
single-layer operator"*, and its campaign report
(``docs/fmm_acceleration/report.md``) states three acceptance criteria that the
report tabulates as PASS:

* (c) complexity -- ``N in [1e3, 1e5]``, FMM time log-log slope ``in [0.9, 1.2]``;
* (d) physics invariance -- "sphere surface potential FMM vs direct < 1e-10",
  and a "added mass relative diff < 1e-6";
* (e) memory -- ``N = 1e5`` peak memory below 5 % of the dense matrix.

Every one of those is a *cost* or *routing* claim, and none of them is decided by
the shipped tests: ``tests/test_fmm.py``:

* runs the complexity ladder on ``N in [100, 1000]`` with **50 targets only**
  (the target count, not the source count, is what an FMM matvec costs), and
  accepts any slope ``< 2.0`` -- i.e. it cannot distinguish an FMM from a
  quadratic method;
* runs "physics invariance" with ``max_leaf = panel_count``, which puts every
  source in one leaf; the operator then routes *every* interaction to the direct
  near-field sum, so the test compares direct summation with direct summation
  and its tolerance says nothing about the multipole path.

This module measures what the report claims, on the geometry class the operator
would actually be used on (panel centroids of a closed surface, i.e. a
two-dimensional source distribution in 3-D), and reports three quantities:

1. **Routing** -- how many target-leaf pairs are multipole-evaluated and how many
   fall back to direct summation, for the audit geometry and for the exact
   configuration used by ``tests/test_fmm.py::test_sphere_surface_potential_fmm_single_leaf``.
2. **Accuracy** -- the multipole error ladder versus expansion order ``p`` on a
   well-separated configuration (where the multipole path *is* exercised), and
   on the surface geometry with the coincident self terms removed.
3. **Cost** -- wall time of the FMM matvec and of the vectorised dense matvec at
   the same ``N`` and target set, decomposed into a per-interaction cost
   ``c_fmm`` and a per-pair cost ``c_dense``.  Their ratio
   ``c_fmm / (avg_leaf * c_dense)`` is the FMM/dense cost ratio *per unit of
   work*: because a surface source set produces a number of leaves proportional
   to ``N``, both methods cost ``O(N**2)`` here, and that ratio is therefore
   **N-independent**.  If it is above one, no value of ``N`` can make this
   implementation faster than the dense matvec.

Nothing in this module changes the solver; it is a measurement harness plus a
Markdown/JSON renderer (``python -m tensorbem.benchmarks.fmm_cost_audit``).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tensorbem.fmm import FMMOperator, direct_laplace_matvec
from tensorbem.geometry3d import icosphere

NEAR_THRESHOLD = 2.0  # mirrors FMMOperator._is_near


# ---------------------------------------------------------------------------
# geometry / routing
# ---------------------------------------------------------------------------

def surface_panels(level: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Centroids and areas of an icosphere surface mesh (the BEM source set)."""

    mesh = icosphere(subdivisions=level, radius=1.0)
    return np.asarray(mesh.centroids, dtype=float), np.asarray(mesh.areas, dtype=float)


def _target_leaf(op: FMMOperator, point: np.ndarray):
    centers = np.array([leaf.center for leaf in op.leaves])
    return op.leaves[int(np.argmin(np.linalg.norm(centers - point[None, :], axis=1)))]


def routing_counts(op: FMMOperator, targets: np.ndarray) -> dict[str, int]:
    """Count near-field (direct) and far-field (multipole) target-leaf pairs.

    The routing mirrors ``FMMOperator.matvec`` after the m14931 fix: a pair is
    multipole-evaluated only when the target is outside the source leaf's own
    bounding sphere *and* the two leaves are not leaf-centre neighbours.
    """

    far = int(round(op.far_fraction(targets) * sum(
        1 for t in targets for sleaf in op.leaves if len(sleaf.indices))))
    total = sum(1 for t in targets for sleaf in op.leaves if len(sleaf.indices))
    return {"near_pairs": total - far, "far_pairs": far, "total_pairs": total}


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------

def dense_matvec(targets: np.ndarray, sources: np.ndarray, charges: np.ndarray,
                 block: int = 512) -> np.ndarray:
    """Vectorised block dense Laplace single-layer matvec (the reference)."""

    out = np.empty(len(targets), dtype=float)
    for i in range(0, len(targets), block):
        diff = targets[i:i + block, None, :] - sources[None, :, :]
        r = np.maximum(np.linalg.norm(diff, axis=2), 1e-15)
        out[i:i + block] = (charges[None, :] / (4.0 * math.pi * r)).sum(axis=1)
    return out


def _best_of(fn, repeats: int = 2) -> tuple[float, object]:
    best = float("inf")
    value = None
    for _ in range(repeats):
        start = time.perf_counter()
        value = fn()
        best = min(best, time.perf_counter() - start)
    return best, value


def cost_point(level: int, p: int, max_leaf: int) -> dict[str, float]:
    """One cost measurement: FMM vs dense at the same N and the same targets."""

    centroids, areas = surface_panels(level)
    n = len(centroids)
    charges = areas * 1.0

    def run_fmm():
        op = FMMOperator(p=p, max_leaf=max_leaf)
        op.setup(centroids, charges)
        return op.matvec(centroids, centroids, charges)

    def run_direct():
        return dense_matvec(centroids, centroids, charges)

    t_fmm, fmm = _best_of(run_fmm)
    t_dense, _ = _best_of(run_direct)

    op = FMMOperator(p=p, max_leaf=max_leaf)
    op.setup(centroids, charges)
    route = routing_counts(op, centroids)

    avg_leaf = n / max(1, len(op.leaves))
    # singular self term is present in both branches; it is *not* the subject here
    c_fmm = t_fmm / max(1, route["total_pairs"])
    c_dense = t_dense / (n * n)
    return {
        "level": level,
        "panels": n,
        "p": p,
        "max_leaf": max_leaf,
        "leaves": len(op.leaves),
        "avg_leaf": avg_leaf,
        "t_fmm_s": t_fmm,
        "t_dense_s": t_dense,
        "time_ratio": t_fmm / t_dense if t_dense else float("inf"),
        "c_fmm_s_per_interaction": c_fmm,
        "c_dense_s_per_pair": c_dense,
        "per_unit_cost_ratio": c_fmm / (c_dense * avg_leaf),
        **route,
    }


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------

def separated_ladder(p_values: tuple[int, ...] = (1, 2, 3, 4, 6, 8)) -> list[dict]:
    """Multipole accuracy ladder on a well-separated configuration.

    Sources fill a ball of radius ``a``; targets sit on a sphere of radius
    ``8 a``.  Every interaction there is far-field, so this ladder measures the
    multipole expansion itself (the regime the shipped test intended to probe).
    """

    rng = np.random.default_rng(99)
    a = 0.1
    sources = rng.normal(size=(200, 3))
    sources *= a * rng.random(200)[:, None] ** (1.0 / 3.0) / np.maximum(
        np.linalg.norm(sources, axis=1, keepdims=True), 1e-15)
    charges = rng.normal(size=200)

    directions = rng.normal(size=(64, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    targets = 8.0 * a * directions

    direct = direct_laplace_matvec(targets, sources, charges)
    scale = float(np.max(np.abs(direct)))
    rows = []
    for p in p_values:
        # m15669: with a geometrically correct octree a leaf is a real box, so
        # ``max_leaf=50`` packs the whole 200-source ball (radius 0.1) into 8
        # leaves whose box test (2*size) covers the 0.8 target radius -- every
        # pair then falls back on the near-field sum and the ladder is vacuous.
        # ``max_leaf=25`` gives leaves of radius <= 0.23, so the multipole path
        # carries ~50 % of the target-leaf pairs here.
        fmm = FMMOperator(p=p, max_leaf=25)
        fmm.setup(sources, charges)
        approx = fmm.matvec(targets, sources, charges)
        rows.append({
            "p": p,
            "max_rel_error": float(np.max(np.abs(approx - direct)) / scale),
            **routing_counts(fmm, targets),
        })
    return rows


def scaling_ladder(
    levels: tuple[int, ...] = (3, 4, 5),
    p: int = 2,
    max_leaf: int = 128,
    repeats: int = 2,
) -> dict:
    """Measured cost scaling on the audit surface (criterion (c), m15669).

    The campaign report's criterion (c) asks for the FMM time log-log slope on
    ``N in [1e3, 1e5]``.  This ladder *measures* it: every row is a real matvec
    of ``N`` targets against ``N`` panels of an icosphere, timed against the
    vectorised dense matvec at the same ``N`` (best of ``repeats``).  Dense is
    skipped above ``DENSE_MAX`` panels because the reference would exceed the
    memory of the machine, so the slope of the dense branch is fitted on the
    rows where it was measured.
    """

    DENSE_MAX = 20_480
    rows = []
    for level in levels:
        mesh = icosphere(subdivisions=level, radius=1.0)
        centroids = np.asarray(mesh.centroids, dtype=float)
        charges = np.asarray(mesh.areas, dtype=float)
        n = len(centroids)

        op = FMMOperator(p=p, max_leaf=max_leaf)
        op.setup(centroids, charges)
        t_fmm, _ = _best_of(lambda: op.matvec(centroids, centroids, charges), repeats)
        row = {
            "level": level,
            "panels": n,
            "p": p,
            "leaves": len(op.leaves),
            "far_fraction": op.far_fraction(centroids),
            "t_fmm_s": t_fmm,
            **routing_counts(op, centroids),
        }
        if n <= DENSE_MAX:
            t_dense, _ = _best_of(lambda: dense_matvec(centroids, centroids, charges),
                                  repeats)
            row["t_dense_s"] = t_dense
            row["time_ratio"] = t_fmm / t_dense if t_dense else float("inf")
        rows.append(row)

    def _slope(key: str) -> float | None:
        pts = [(row["panels"], row[key]) for row in rows
               if key in row and row[key] > 0]
        if len(pts) < 2:
            return None
        log_n = np.log(np.array([p_[0] for p_ in pts], dtype=float))
        log_t = np.log(np.array([p_[1] for p_ in pts], dtype=float))
        return float(np.polyfit(log_n, log_t, 1)[0])

    return {
        "rows": rows,
        "fmm_slope": _slope("t_fmm_s"),
        "dense_slope": _slope("t_dense_s"),
        "ratio_slope": _slope("time_ratio"),
    }


def surface_ladder(p_values: tuple[int, ...] = (2, 4)) -> list[dict]:
    """Multipole accuracy on the audit surface, with the self terms removed.

    Sources are the panel centroids; targets are the mesh vertices, which are
    distinct from every centroid, so no coincident pair exists and the near-field
    direct sum is finite.  This is the accuracy the operator would deliver on a
    surface distribution -- not the tautological single-leaf comparison.
    """

    mesh = icosphere(subdivisions=2, radius=1.0)
    centroids = np.asarray(mesh.centroids, dtype=float)
    charges = np.asarray(mesh.areas, dtype=float)
    targets = np.asarray(mesh.vertices, dtype=float)

    direct = direct_laplace_matvec(targets, centroids, charges)
    scale = float(np.max(np.abs(direct)))
    rows = []
    for p in p_values:
        op = FMMOperator(p=p, max_leaf=32)
        op.setup(centroids, charges)
        approx = op.matvec(targets, centroids, charges)
        rows.append({
            "p": p,
            "panels": len(centroids),
            "targets": len(targets),
            "max_rel_error": float(np.max(np.abs(approx - direct)) / scale),
            **routing_counts(op, targets),
        })
    return rows


def tautology_probe() -> dict[str, float]:
    """Show that the shipped "physics invariance" configuration uses no multipoles."""

    mesh = icosphere(subdivisions=2, radius=1.0)
    centroids = np.asarray(mesh.centroids, dtype=float)
    charges = np.asarray(mesh.areas, dtype=float)
    op = FMMOperator(p=6, max_leaf=len(centroids))
    op.setup(centroids, charges)
    route = routing_counts(op, centroids)
    fmm = op.matvec(centroids, centroids, charges)
    direct = direct_laplace_matvec(centroids, centroids, charges)
    return {
        "leaves": len(op.leaves),
        "rel_error": float(np.max(np.abs(fmm - direct))
                           / (np.max(np.abs(direct)) + 1e-15)),
        **route,
    }


# ---------------------------------------------------------------------------
# study
# ---------------------------------------------------------------------------

@dataclass
class Study:
    cost: list[dict]
    separated: list[dict]
    surface: list[dict]
    tautology: dict
    scaling: dict | None = None

    def to_json(self) -> dict:
        return {
            "mission": "m14931 (re-measured after the m15669 operator fixes)",
            "subject": "tensorbem.fmm",
            "cost": self.cost,
            "accuracy_separated": self.separated,
            "accuracy_surface": self.surface,
            "shipped_physics_test_routing": self.tautology,
            "scaling": self.scaling,
        }


def run_study(*, levels: tuple[int, ...] = (2, 3), p_cost: int = 4,
              max_leaf: int = 32, scaling_levels: tuple[int, ...] = (3, 4, 5)
              ) -> Study:
    cost = [cost_point(level, 2, max_leaf) for level in levels]
    cost.append(cost_point(levels[0], p_cost, max_leaf))
    return Study(
        cost=cost,
        separated=separated_ladder(),
        surface=surface_ladder(),
        tautology=tautology_probe(),
        scaling=scaling_ladder(levels=scaling_levels),
    )


def render_markdown(study: Study) -> str:
    lines = [
        "# `tensorbem.fmm` cost and routing audit (m14931)",
        "",
        "Generated by `python -m tensorbem.benchmarks.fmm_cost_audit`; every",
        "number below is rendered from the JSON that the same process wrote",
        "(`docs/fmm_cost_audit/audit.json`). No figure is transcribed by hand.",
        "",
        "## 1. Cost, FMM matvec vs vectorised dense matvec (same N, same targets)",
        "",
        "| level | panels | p | leaves | avg leaf | t_FMM (s) | t_dense (s) | FMM/dense | "
        "per-interaction vs per-pair cost |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in study.cost:
        lines.append(
            f"| {row['level']} | {row['panels']} | {row['p']} | {row['leaves']} | "
            f"{row['avg_leaf']:.2f} | {row['t_fmm_s']:.3f} | {row['t_dense_s']:.4f} | "
            f"{row['time_ratio']:.1f}x | {row['per_unit_cost_ratio']:.1f}x |"
        )
    ratios = [row["time_ratio"] for row in study.cost]
    lines += [
        "",
        "The last column is `c_fmm / (avg_leaf * c_dense)`: the cost of one",
        "multipole target-leaf interaction divided by the cost of summing the same",
        "leaf directly. The **FMM/dense wall-time ratio** is "
        + ", ".join(f"{row['time_ratio']:.2f}x at {row['panels']} panels"
                    for row in study.cost)
        + ". A ratio below one is an acceleration on that configuration.",
        "",
        "## 2. Routing on the audit surface",
        "",
        "| level | panels | leaves | near pairs | far (multipole) pairs |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in study.cost:
        lines.append(
            f"| {row['level']} | {row['panels']} | {row['leaves']} | "
            f"{row['near_pairs']} | {row['far_pairs']} |"
        )
    lines += [
        "",
        "## 3. Multipole accuracy, well-separated configuration (r/a = 8)",
        "",
        "| p | max relative error | far pairs |",
        "|---:|---:|---:|",
    ]
    for row in study.separated:
        lines.append(f"| {row['p']} | {row['max_rel_error']:.3e} | {row['far_pairs']} |")
    lines += [
        "",
        "## 4. Multipole accuracy on the surface itself (centroids -> vertices)",
        "",
        "| p | panels | targets | max relative error | near pairs | far pairs |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in study.surface:
        lines.append(
            f"| {row['p']} | {row['panels']} | {row['targets']} | "
            f"{row['max_rel_error']:.3e} | {row['near_pairs']} | {row['far_pairs']} |"
        )
    t = study.tautology
    lines += [
        "",
        "## 5. The shipped \"physics invariance\" configuration routes nothing to the multipole",
        "",
        f"`max_leaf = panel_count` gives **{t['leaves']}** leaf, "
        f"**{t['near_pairs']}** near pairs and **{t['far_pairs']}** multipole pairs; the",
        f"recorded agreement ({t['rel_error']:.1e}) is direct summation compared with",
        "direct summation, and the coincident centroid pairs clamp `r` to 1e-15 on",
        "both sides. It is a regression pin on the degeneracy branch, not a check of",
        "the multipole expansion.",
        "",
    ]
    sc = study.scaling
    if sc:
        lines += [
            "## 6. Measured cost scaling (criterion (c), `N in [1e3, 2e4]`)",
            "",
            "| panels | p | leaves | far fraction | t_FMM (s) | t_dense (s) | FMM/dense |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in sc["rows"]:
            dense = f"{row['t_dense_s']:.3f}" if "t_dense_s" in row else "-"
            ratio = f"{row['time_ratio']:.2f}x" if "time_ratio" in row else "-"
            lines.append(
                f"| {row['panels']} | {row['p']} | {row['leaves']} | "
                f"{row['far_fraction']:.3f} | {row['t_fmm_s']:.3f} | {dense} | {ratio} |"
            )
        lines += [
            "",
            f"Measured log-log slopes: FMM **{sc['fmm_slope']:.2f}**, dense "
            f"**{sc['dense_slope']:.2f}**. The campaign report asked for a FMM slope",
            "in `[0.9, 1.2]`; the value above is the measurement that replaces the",
            "extrapolation the m14931 audit flagged.",
            "",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/fmm_cost_audit")
    parser.add_argument("--p-cost", type=int, default=4)
    parser.add_argument("--max-leaf", type=int, default=32)
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args(argv)

    study = run_study(p_cost=args.p_cost, max_leaf=args.max_leaf)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(study.to_json(), indent=2) + "\n")
    doc = out / "README.md"
    doc.write_text(render_markdown(study) + "\n")
    if args.figures:
        make_figures(study, out / "fig_fmm_cost.png")
    print(json.dumps(study.to_json(), indent=2))
    print(f"wrote {out / 'audit.json'} and {doc}")
    return 0


def make_figures(study: Study, path: Path) -> None:  # pragma: no cover - plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    left, right = axes

    pairs = [(row["panels"], row["t_fmm_s"], row["t_dense_s"])
             for row in study.cost if row["p"] == 2]
    n = np.array([p[0] for p in pairs], dtype=float)
    left.loglog(n, [p[1] for p in pairs], "o-", label="FMM matvec")
    left.loglog(n, [p[2] for p in pairs], "s-", label="dense matvec")
    left.set_xlabel("panels N")
    left.set_ylabel("wall time (s)")
    left.set_title("FMM vs dense, same targets (p=2)")
    left.grid(True, which="both", alpha=0.3)
    left.legend()

    ps = [row["p"] for row in study.separated]
    right.semilogy(ps, [row["max_rel_error"] for row in study.separated], "o-",
                   label="separated (r/a=8)")
    right.semilogy([row["p"] for row in study.surface],
                   [row["max_rel_error"] for row in study.surface], "s-",
                   label="surface (centroids->vertices)")
    right.set_xlabel("expansion order p")
    right.set_ylabel("max relative error")
    right.set_title("multipole accuracy")
    right.grid(True, which="both", alpha=0.3)
    right.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())