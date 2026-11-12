"""m16000 — evidence generator for the ``FMMOperator`` charge-cache defect.

Reproduces the *pre-fix* semantics by forcing the multipole cache to be reused
for every charge vector (``_PrefixCacheOperator``: the only difference from the
shipped class is the cache-validity predicate), and compares it against the
fixed operator on the same geometry, so the A/B has a single varying input.

Outputs
-------
``docs/fmm_operator_linearity/evidence.json``
    every number quoted in the report (no hand-copied values)
``docs/fmm_operator_linearity/fig_error_vs_p.png``
    second-call relative error vs expansion order, prefix vs fixed
``docs/fmm_operator_linearity/fig_gmres_residual.png``
    GMRES preconditioned-residual history, prefix vs fixed, and the
    distance of each iterate's solution from the dense solve

Run::

    PYTHONPATH=src python scripts/fmm_linearity_bench.py
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from tensorbem import geometry3d
from tensorbem.fmm import FMMOperator, direct_laplace_matvec

OUT = pathlib.Path("docs/fmm_operator_linearity")
PS = [1, 2, 3, 4, 5, 6, 7, 8]


def _rel_err(approx: np.ndarray, exact: np.ndarray) -> float:
    return float(np.max(np.abs(approx - exact)) / np.max(np.abs(exact)))


class _PrefixCacheOperator(FMMOperator):
    """Pre-m16000 cache policy: multipoles built once, never invalidated."""

    def _multipole_cache_valid(self, sources, charges) -> bool:  # noqa: D102
        return self._multipole_matrix is not None


def _separated_geometry(seed: int = 42):
    rng = np.random.default_rng(seed)
    sources = rng.normal(size=(200, 3)) * 0.5
    targets = np.array([[4.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 6.0, 0.0]])
    charges_a = rng.normal(size=200)
    charges_b = rng.normal(size=200)
    return sources, targets, charges_a, charges_b


def error_versus_order() -> dict:
    sources, targets, charges_a, charges_b = _separated_geometry()
    rows = []
    for p in PS:
        direct_a = direct_laplace_matvec(targets, sources, charges_a)
        direct_b = direct_laplace_matvec(targets, sources, charges_b)
        row = {"p": p}
        for label, cls in (("prefix", _PrefixCacheOperator), ("fixed", FMMOperator)):
            op = cls(p=p, max_leaf=25)
            op.setup(sources)
            first = op.matvec(targets, sources, charges_a)
            second = op.matvec(targets, sources, charges_b)
            row[f"{label}_first_err"] = _rel_err(first, direct_a)
            row[f"{label}_second_err"] = _rel_err(second, direct_b)
        rows.append(row)
    # m16156: far_fraction must be measured on a *set-up* operator.  The m16000
    # run of this line forgot ``setup`` and therefore published 0.0, which read
    # as "no far pairs" although the multipole path is what carried the defect.
    probe = FMMOperator(p=4, max_leaf=25)
    probe.setup(sources)
    return {"far_fraction": float(probe.far_fraction(targets)),
            "far_work_fraction": float(probe.far_work_fraction(targets)),
            "rows": rows}


def gmres_experiment() -> dict:
    mesh = geometry3d.icosphere(1)
    centroids = mesh.centroids
    areas = mesh.areas
    n = len(centroids)
    diff = centroids[:, None, :] - centroids[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    dense = (1.0 / (4.0 * np.pi * np.maximum(r, 1e-15))) * areas[None, :]
    rhs = np.ones(n)
    q_dense = np.linalg.solve(dense, rhs)

    out = {"panels": n, "p": 10, "max_leaf": 16}
    histories = {}
    for label, cls in (("prefix", _PrefixCacheOperator), ("fixed", FMMOperator)):
        op = cls(p=10, max_leaf=16)
        op.setup(centroids, np.ones(n))  # prime the cache with q = 1
        op.matvec(centroids, centroids, np.linspace(1.0, 2.0, n))  # unrelated call
        hist: list[float] = []
        lin = LinearOperator(
            (n, n),
            matvec=lambda q: op.matvec(centroids, centroids, q * areas),
            dtype=float,
        )
        q_iter, info = gmres(lin, rhs, rtol=1e-12, atol=0.0, restart=n, maxiter=60,
                             callback=hist.append, callback_type="pr_norm")
        true_residual = float(np.linalg.norm(dense @ q_iter - rhs) / np.linalg.norm(rhs))
        histories[label] = hist
        out[label] = {
            "info": int(info),
            "iterations": len(hist),
            "reported_residual": float(hist[-1]) if hist else None,
            "true_residual": true_residual,
            "solution_error_vs_dense": float(
                np.linalg.norm(q_iter - q_dense) / np.linalg.norm(q_dense)
            ),
        }
    out["histories"] = histories
    return out


def cache_overhead() -> dict:
    """Cost of the charge-key check when nothing changed (the hot path)."""
    sources, targets, charges_a, _ = _separated_geometry()
    op = FMMOperator(p=6, max_leaf=25)
    op.setup(sources)
    op.matvec(targets, sources, charges_a)  # warm

    def timed(fn, repeats: int = 20) -> float:
        best = np.inf
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best

    hit = timed(lambda: op.matvec(targets, sources, charges_a))
    rebuild = timed(lambda: op.matvec(targets, sources, charges_a * 1.0001))
    return {
        "cache_hit_ms": hit * 1e3,
        "rebuild_ms": rebuild * 1e3,
        "rebuild_overhead_ms": (rebuild - hit) * 1e3,
        "overhead_ratio": rebuild / hit,
    }


def figures(data: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    written = []

    rows = data["error_versus_order"]["rows"]
    ps = [r["p"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogy(ps, [r["fixed_first_err"] for r in rows], "o-", label="fixed, 1st call")
    ax.semilogy(ps, [r["fixed_second_err"] for r in rows], "s-", label="fixed, 2nd call (new q)")
    ax.semilogy(ps, [r["prefix_second_err"] for r in rows], "x--",
                label="pre-m16000, 2nd call (new q)")
    ax.axhline(1e-6, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("multipole order p")
    ax.set_ylabel("max relative error vs direct sum")
    ax.set_title("FMMOperator accuracy: charge-vector cache invalidation")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    path = OUT / "fig_error_vs_p.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    g = data["gmres"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for label, style in (("prefix", "x--"), ("fixed", "o-")):
        hist = g["histories"].get(label, [])
        if hist:
            ax.semilogy(range(len(hist)), hist, style,
                        label=f"{label} (true rel. residual {g[label]['true_residual']:.1e})")
    ax.set_xlabel("GMRES iteration")
    ax.set_ylabel("reported preconditioned residual")
    ax.set_title(f"GMRES+FMM single-layer solve, N={g['panels']}, p={g['p']}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    path = OUT / "fig_gmres_residual.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))
    return written


def main() -> None:
    data = {
        "mission": "m16000",
        "error_versus_order": error_versus_order(),
        "gmres": gmres_experiment(),
        "cache_overhead": cache_overhead(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    data["figures"] = figures(data)
    (OUT / "evidence.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({k: v for k, v in data.items() if k != "gmres"}, indent=2))
    print(json.dumps({k: v for k, v in data["gmres"].items() if k != "histories"}, indent=2))


if __name__ == "__main__":
    main()