"""Verification study for :mod:`tensorbem.sk_fit` (Sanathanan-Koerner damping fit).

Two curves, both with a *known* answer, are fitted over an order scan by

* the SK iteration of ``sk_fit.sk_fit_damping_report`` (relocated poles), and
* ``sk_fit.fixed_grid_damping_fit`` -- the pre-rewrite fixed-log-grid
  least-squares behaviour, kept as the measured baseline.

Case A is *representable*: a synthetic damping curve that is exactly a sum of
three real decay rates, with the decay rates deliberately off the logarithmic
grid the baseline uses (0.5, 1.0, 2.5 rad/s inside a 0.02...50 rad/s band), so
the recovery error is a statement about the method and not about the grid.

Case B is the *measured* Hulme floating-hemisphere heave damping curve shipped
in ``tests/data/time_domain_hemisphere_sweep.json`` (Hulme 1982 reference
solution of the radiation problem, used by ``tests/test_state_space.py``).  It
is not a finite sum of real exponentials, so the real-decay-rate basis has a
model-class floor: the study reports the unconstrained rational error as the
ceiling that no real-pole fit can beat there.

Nothing below is transcribed by hand: ``run_study`` writes ``results.json`` and
``render_markdown`` renders the document from it.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..sk_fit import SKFitReport, fixed_grid_damping_fit, sk_fit_damping_report

__all__ = [
    "ORDERS",
    "representable_case",
    "hulme_case",
    "run_study",
    "render_markdown",
    "write_figure",
]

ORDERS: tuple[int, ...] = (2, 3, 4, 6, 8, 10, 12)
SK_MAX_ITER = 200
SK_TOL = 1e-12


def representable_case() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """A damping curve that is exactly ``d + sum c_k a_k/(a_k^2+omega^2)``."""
    decay_rates = (0.5, 1.0, 2.5)
    residues = (1.0, -1.2, 0.6)
    asymptote = 0.05
    omega = np.logspace(math.log10(0.02), math.log10(50.0), 300)
    damping = np.full_like(omega, asymptote)
    for alpha, c in zip(decay_rates, residues):
        damping = damping + c * alpha / (alpha**2 + omega**2)
    truth = {
        "decay_rates": list(decay_rates),
        "residues": list(residues),
        "asymptote": asymptote,
        "omega_band": [float(omega[0]), float(omega[-1])],
        "points": int(omega.size),
    }
    return omega, damping, truth


def hulme_case() -> tuple[np.ndarray, np.ndarray]:
    """Measured Hulme (1982) hemisphere heave damping curve (repo fixture)."""
    path = Path(__file__).resolve().parents[3] / "tests" / "data" / "time_domain_hemisphere_sweep.json"
    record = json.loads(path.read_text())
    return np.asarray(record["omega"], dtype=float), np.asarray(record["damping"], dtype=float)


def _row(order: int, result, report: SKFitReport, baseline, truth: dict | None) -> dict:
    row: dict[str, Any] = {
        "order": order,
        "pole_source": report.pole_source,
        "sk_iterations": report.sk_iterations,
        "sk_converged": report.sk_converged,
        "rational_rel_error": report.rational_rel_error,
        "sk_pole_rel_error": report.sk_pole_rel_error,
        "grid_pole_rel_error": report.grid_pole_rel_error,
        "returned_rel_error": float(result.relative_error_max),
        "returned_pointwise_error": float(result.relative_error_max_pointwise),
        "baseline_rel_error": float(baseline.relative_error_max),
        "real_root_count": report.real_root_count,
        "complex_pair_count": report.complex_pair_count,
        "asymptote": float(result.constant),
    }
    if truth is not None:
        truth_rates = np.asarray(truth["decay_rates"], dtype=float)
        rates = np.sort(np.asarray(result.poles, dtype=float))
        taken = rates[: truth_rates.size] if rates.size >= truth_rates.size else rates
        row["recovered_decay_rates"] = [float(x) for x in taken]
        row["decay_rate_rel_error"] = (
            float(np.max(np.abs(taken - truth_rates) / truth_rates))
            if taken.size == truth_rates.size
            else float("inf")
        )
        if taken.size == truth_rates.size and np.asarray(result.residues).size == truth_rates.size:
            row["residue_abs_error"] = float(
                np.max(np.abs(np.asarray(result.residues, dtype=float)
                              - np.asarray(truth["residues"], dtype=float)))
            )
        else:
            row["residue_abs_error"] = float("inf")
        row["asymptote_abs_error"] = float(abs(result.constant - truth["asymptote"]))
    return row


def run_study(orders: Sequence[int] = ORDERS) -> dict[str, Any]:
    omega_a, damping_a, truth = representable_case()
    omega_b, damping_b = hulme_case()
    cases: dict[str, Any] = {}
    for name, (omega, damping, truth_case) in {
        "representable": (omega_a, damping_a, truth),
        "hulme_hemisphere": (omega_b, damping_b, None),
    }.items():
        rows = []
        for order in orders:
            result, report = sk_fit_damping_report(
                omega, damping, n_poles=order, max_iter=SK_MAX_ITER, tol=SK_TOL
            )
            baseline = fixed_grid_damping_fit(omega, damping, n_poles=order)
            rows.append(_row(order, result, report, baseline, truth_case))
        cases[name] = {
            "points": int(omega.size),
            "omega_band": [float(omega[0]), float(omega[-1])],
            "damping_max": float(np.max(damping)),
            "damping_min": float(np.min(damping)),
            "truth": truth_case,
            "rows": rows,
            "regression_free": all(
                r["returned_rel_error"] <= r["baseline_rel_error"] * (1.0 + 1e-12) for r in rows
            ),
            "sk_wins_order_count": sum(1 for r in rows if r["pole_source"] == "sk"),
        }
    return {
        "mission": "m16021",
        "sk_max_iter": SK_MAX_ITER,
        "sk_tol": SK_TOL,
        "orders": list(orders),
        "cases": cases,
    }


def _fmt(value: float) -> str:
    return f"{value:.3e}"


def render_markdown(results: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("| case | order | returned max err | baseline max err | " "SK wins? | pole source | SK iters | converged |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, case in results["cases"].items():
        for row in case["rows"]:
            lines.append(
                "| {name} | {order} | {err} | {base} | {win} | {src} | {it} | {conv} |".format(
                    name=name,
                    order=row["order"],
                    err=_fmt(row["returned_rel_error"]),
                    base=_fmt(row["baseline_rel_error"]),
                    win="yes" if row["returned_rel_error"] < row["baseline_rel_error"] else "no",
                    src=row["pole_source"],
                    it=row["sk_iterations"],
                    conv="yes" if row["sk_converged"] else "no",
                )
            )
    return "\n".join(lines)


def write_figure(results: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for axis, (name, case) in zip(axes, results["cases"].items()):
        orders = [row["order"] for row in case["rows"]]
        axis.loglog(orders, [max(row["returned_rel_error"], 1e-16) for row in case["rows"]],
                    "o-", label="SK fit (returned)")
        axis.loglog(orders, [max(row["baseline_rel_error"], 1e-16) for row in case["rows"]],
                    "s--", label="fixed-grid baseline")
        axis.loglog(orders, [max(row["rational_rel_error"], 1e-16) for row in case["rows"]],
                    "^:", label="unconstrained rational ceiling")
        axis.set_xlabel("decay rates (order)")
        axis.set_ylabel("max |dB| / max|B|")
        axis.set_title(name)
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--write-doc", default=None)
    parser.add_argument("--figure", default=None)
    args = parser.parse_args(argv)

    results = run_study()
    if args.out_root:
        out = Path(args.out_root)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        (out / "results.md").write_text(render_markdown(results) + "\n")
        write_figure(results, out / "sk_fit_verification.png")
        print(f"wrote {out / 'results.json'}")
    if args.write_doc:
        doc = Path(args.write_doc)
        text = doc.read_text()
        begin = "<!-- sk-fit-table:begin -->"
        end = "<!-- sk-fit-table:end -->"
        block = begin + "\n" + render_markdown(results) + "\n" + end
        if begin in text and end in text:
            head = text.split(begin)[0]
            tail = text.split(end)[1]
            doc.write_text(head + block + tail)
        else:
            doc.write_text(text.rstrip() + "\n\n" + block + "\n")
        print(f"updated {doc}")
    if args.figure:
        write_figure(results, Path(args.figure))
        print(f"wrote {args.figure}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())