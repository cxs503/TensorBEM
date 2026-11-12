"""m16265 -- evidence generator for the ``FMMOperator`` geometry contract.

The operator builds its octree, leaf source indices and leaf radii from the
point set given to :meth:`FMMOperator.setup`.  :meth:`FMMOperator.matvec` takes
``sources`` as an argument as well; before m16265 that argument was trusted.

This script measures the *pre-fix* semantics through ``_PrefixGeometryOperator``
(the shipped class with the geometry check overridden to a no-op, so the only
varying input is the contract), compares it against the dense direct sum, and
records what the shipped class does now.

Outputs
-------
``docs/fmm_geometry_contract/evidence.json``
    every number quoted in ``docs/fmm_geometry_contract/README.md``
``docs/fmm_geometry_contract/fig_geometry_contract.png``
    pre-fix relative error per case (log scale) versus the FMM truncation floor

Run::

    PYTHONPATH=src python scripts/fmm_geometry_contract.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from tensorbem.fmm import FMMOperator, direct_laplace_matvec

OUT = pathlib.Path("docs/fmm_geometry_contract")


class _PrefixGeometryOperator(FMMOperator):
    """Pre-m16265 semantics: ``matvec`` does not validate ``sources``."""

    def _require_setup_geometry(self, sources):  # noqa: D102
        return None


def geometry(seed: int = 7, n: int = 200):
    rng = np.random.default_rng(seed)
    sources = rng.normal(size=(n, 3)) * 0.5
    charges = rng.normal(size=n)
    return sources, np.array([[8.0, 0.0, 0.0], [0.0, 6.0, 0.0], [4.0, 4.0, 0.0]]), charges


def rel(approx: np.ndarray, exact: np.ndarray) -> float:
    return float(np.max(np.abs(approx - exact)) / np.max(np.abs(exact)))


def case_table():
    setup_points, targets, charges = geometry()
    rng = np.random.default_rng(11)
    mutations = {
        "shift_+10x": lambda p: p + np.array([10.0, 0.0, 0.0]),
        "scale_x3": lambda p: p * 3.0,
        "scale_x20": lambda p: p * 20.0,
        "permuted": lambda p: p[rng.permutation(len(p))],
        "epsilon_shift": lambda p: p + 1.0e-9,
    }
    rows = []
    for name, mutate in mutations.items():
        moved = mutate(setup_points.copy())

        prefix = _PrefixGeometryOperator(p=4, max_leaf=25)
        prefix.setup(setup_points)
        try:
            prefix_err = rel(prefix.matvec(targets, moved, charges),
                             direct_laplace_matvec(targets, moved, charges))
            prefix_outcome = "silent wrong answer"
        except Exception as exc:  # pragma: no cover - defensive
            prefix_err = None
            prefix_outcome = f"{type(exc).__name__}"

        fixed = FMMOperator(p=4, max_leaf=25)
        fixed.setup(setup_points)
        try:
            fixed.matvec(targets, moved, charges)
            fixed_outcome = "accepted (CONTRACT BROKEN)"
        except ValueError as exc:
            fixed_outcome = f"ValueError: {str(exc).split(';')[0]}"

        rows.append(
            {
                "case": name,
                "prefix_outcome": prefix_outcome,
                "prefix_rel_err": prefix_err,
                "fixed_outcome": fixed_outcome,
                "sources": moved.tolist(),
                "targets": targets.tolist(),
            }
        )

    # baseline: the contract respected (same geometry, p=4 truncation floor)
    base = FMMOperator(p=4, max_leaf=25)
    base.setup(setup_points)
    baseline_err = rel(base.matvec(targets, setup_points, charges),
                       direct_laplace_matvec(targets, setup_points, charges))

    # length mismatch: bare IndexError pre-fix, ValueError now
    length_rows = []
    for n in (50, 199, 201):
        other, _, other_charges = geometry(seed=13, n=n)
        prefix = _PrefixGeometryOperator(p=4, max_leaf=25)
        prefix.setup(setup_points)
        try:
            prefix.matvec(targets, other, other_charges)
            outcome = "accepted (CONTRACT BROKEN)"
        except Exception as exc:
            outcome = type(exc).__name__
        fixed = FMMOperator(p=4, max_leaf=25)
        fixed.setup(setup_points)
        try:
            fixed.matvec(targets, other, other_charges)
            fixed_outcome = "accepted (CONTRACT BROKEN)"
        except Exception as exc:
            fixed_outcome = type(exc).__name__
        length_rows.append({"n_sources": n, "prefix": outcome, "fixed": fixed_outcome})

    return {
        "schema": "tensorbem.fmm_geometry_contract/1",
        "setup_points": setup_points.tolist(),
        "operator": {"p": 4, "max_leaf": 25, "n_sources": len(setup_points)},
        "baseline_same_geometry_rel_err": baseline_err,
        "cases": rows,
        "length_mismatch": length_rows,
    }


def figure(payload, out_png: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [row["case"] for row in payload["cases"]]
    errs = [row["prefix_rel_err"] for row in payload["cases"]]
    floor = payload["baseline_same_geometry_rel_err"]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(names, errs, color="#c1554d", label="pre-m16265 (no geometry check)")
    ax.axhline(floor, color="#2c6e9b", ls="--", lw=1.4,
               label=f"same geometry, p=4 FMM vs dense ({floor:.1e})")
    ax.set_yscale("log")
    ax.set_ylabel("max relative error vs dense direct sum")
    ax.set_title("FMMOperator.matvec with sources != setup geometry (N=200, 3 targets)")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = case_table()
    (OUT / "evidence.json").write_text(json.dumps(payload, indent=2), encoding="utf8")
    figure(payload, OUT / "fig_geometry_contract.png")
    print(json.dumps({k: v for k, v in payload.items()
                      if k not in ("setup_points", "cases")}, indent=2))
    for row in payload["cases"]:
        print(f"{row['case']:14s} prefix_rel_err={row['prefix_rel_err']!r:>10} "
              f"| fixed={row['fixed_outcome'][:60]}")


if __name__ == "__main__":
    main()