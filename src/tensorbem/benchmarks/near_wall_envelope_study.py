"""Near-wall accuracy envelope and the fixed-clearance convergence test (m6445).

The question
------------
``docs/near_surface_error_budget.md`` closed the near-wall *honest FAIL* of
``docs/near_surface_field.md`` with a corrected diagnosis: the residual is the
piecewise-constant density, not the quadrature, and at fixed ``d / h_max`` the
error is scale invariant (amplification ``~1/d`` cancels the trace variation
``~h``).  Its §5 then draws a stronger conclusion -- "*mesh refinement is **not**
the lever*" and "the observed order is -0.03 over 320 -> 1280 panels" -- and
lists only higher-order densities or tangential recovery as cures.

That stronger conclusion is a statement about the *representation at fixed
relative clearance*, not about the *operator*.  This driver measures the missing
experiment: refine the mesh while holding the probe at a **fixed physical
clearance** ``d``.  Then ``h_max`` shrinks, ``d / h_max`` grows, and the probe
walks up the envelope instead of along it.  Both statements are reported side by
side so neither is over-claimed:

* ``fixed_clearance`` — error vs ``h`` at fixed ``d``, with the observed order of
  convergence (predicted ``~2`` asymptotically: the centroid rule is second order
  for the trace, and the near-singular amplification is bounded because ``d`` is
  fixed);
* ``ratio_sweep`` — error vs ``d / h_max`` for several meshes, which is where the
  scale invariance lives.

The combined output is :class:`tensorbem.near_wall_envelope.NearWallEnvelope`, a
measured worst-probe lookup ``d / h_max -> relative velocity error`` with its
inverse (the clearance needed for a target error).  It is the operational rule
that replaces "refinement does not help": refinement *does* help, provided the
mesh is chosen so that ``d >= (required ratio) * h_max``.

Physics and reference
---------------------
Unit sphere (radius ``a = 1 m``), disturbance potential of a sphere in a uniform
stream along ``+z``: ``u = z / (2 r^3)``, ``grad u = e_z / (2 r^3) - 3 z x / (2 r^5)``;
exact traces ``u = cos(theta)/2``, ``du/dn = cos(theta)`` at the panel centroids.
Reference values are closed form, so every number here is a discretisation
measure.  Probe directions: equator ``+x`` (worst), pole ``+z``, oblique
``(1,0,1)/sqrt(2)``.

Run as::

    PYTHONPATH=src python3 -m tensorbem.benchmarks.near_wall_envelope_study
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tensorbem.geometry3d import icosphere
from tensorbem.near_wall_envelope import NearWallEnvelope, build_envelope
from tensorbem.benchmarks.near_surface_error_budget import (
    analytic_gradient,
    analytic_traces,
    near_wall_velocity,
)

OUTPUT_DIR = Path("docs/near_wall_envelope")

#: Icosphere subdivisions on the mesh ladder (320 / 1280 / 5120 / 20480 panels).
SUBDIVISIONS = (2, 3, 4, 5)

#: Mesh ladder used for the *envelope* sweep (the 20480-panel mesh is only used
#: by the fixed-clearance ladder, where it buys the asymptotic order).
ENVELOPE_SUBDIVISIONS = (2, 3, 4)

#: Physical clearances of the fixed-``d`` ladder, in metres.
FIXED_CLEARANCES = (0.2, 0.1, 0.05, 0.025, 0.0125)

#: Relative clearances ``d / h_max`` of the envelope sweep.
CLEARANCE_RATIOS = (3.0, 2.0, 1.2, 0.8, 0.6, 0.45, 0.3, 0.2, 0.12, 0.08, 0.05)

#: Probe directions on the unit sphere (equator is the worst of the three).
DIRECTIONS = ("equator", "pole", "oblique")

#: Error targets for which a clearance ratio is reported.
TARGETS = (0.10, 0.05, 0.03, 0.01)

#: Direction vectors (unit), for documentation and reproducibility.
DIRECTION_VECTORS = {
    "equator": (1.0, 0.0, 0.0),
    "pole": (0.0, 0.0, 1.0),
    "oblique": (1.0 / math.sqrt(2.0), 0.0, 1.0 / math.sqrt(2.0)),
}

#: Gate thresholds.
MIN_FIXED_CLEARANCE_ORDER = 1.0  # asymptotic order at d = 0.05 m must exceed this
SCALE_INVARIANCE_FACTOR = 3.0  # max/min error across meshes at one d/h_max
TARGET_ERROR = 0.03  # the 3 % bar of docs/near_surface_field.md


def _direction(name: str) -> np.ndarray:
    try:
        return np.asarray(DIRECTION_VECTORS[name], dtype=float)
    except KeyError:  # pragma: no cover - guard
        raise ValueError(f"unknown direction {name!r}") from None


def _target(direction: str, clearance: float) -> np.ndarray:
    return (1.0 + float(clearance)) * _direction(direction)


def measure(mesh, direction: str, clearance: float) -> dict:
    """Relative error of the near-wall velocity at one probe."""
    values, flux = analytic_traces(mesh)
    target = _target(direction, clearance)
    exact = analytic_gradient(target)
    velocity = near_wall_velocity(mesh, values, flux, target)
    error = float(np.linalg.norm(velocity - exact))
    scale = float(np.linalg.norm(exact))
    return {
        "clearance": float(clearance),
        "d_over_h": float(clearance) / float(mesh.h_max),
        "direction": direction,
        "panels": int(mesh.panel_count),
        "h_max": float(mesh.h_max),
        "relative_error": error / scale,
        "velocity_error": error,
        "reference_scale": scale,
    }


def observed_order(h_values, errors) -> dict:
    """Least-squares slope of ``log(error)`` against ``log(h)``.

    Reported for the full ladder and for the two finest rungs (the asymptotic
    order, which is the one that matters for the convergence claim).  A ladder
    that has not entered its asymptotic range yields a low full-ladder slope;
    the two numbers are kept apart on purpose.
    """
    h = np.asarray(h_values, dtype=float)
    e = np.asarray(errors, dtype=float)
    positive = e > 0.0
    full = float(np.polyfit(np.log(h[positive]), np.log(e[positive]), 1)[0]) if positive.sum() >= 2 else float("nan")
    tail = positive[-2:]
    if int(tail.sum()) >= 2:
        h_tail = np.log(h)[positive][-2:]
        e_tail = np.log(e)[positive][-2:]
        asymptotic = float(np.polyfit(h_tail, e_tail, 1)[0])
    else:  # pragma: no cover - guard
        asymptotic = float("nan")
    return {
        "order_full_ladder": full,
        "order_two_finest": asymptotic,
        "rungs": int(positive.sum()),
    }


def fixed_clearance_ladder(
    subdivisions=SUBDIVISIONS, clearances=FIXED_CLEARANCES, directions=DIRECTIONS
) -> dict:
    meshes = {s: icosphere(subdivisions=s, radius=1.0) for s in subdivisions}
    rows: list[dict] = []
    orders: list[dict] = []
    for direction in directions:
        for clearance in clearances:
            ladder = []
            for subdiv, mesh in meshes.items():
                row = measure(mesh, direction, clearance)
                row["subdivisions"] = subdiv
                rows.append(row)
                ladder.append(row)
            ladder.sort(key=lambda r: r["h_max"], reverse=True)
            order = observed_order([r["h_max"] for r in ladder], [r["relative_error"] for r in ladder])
            order.update(
                {
                    "direction": direction,
                    "clearance": float(clearance),
                    "coarsest_error": ladder[0]["relative_error"],
                    "finest_error": ladder[-1]["relative_error"],
                    "improvement_factor": ladder[0]["relative_error"] / ladder[-1]["relative_error"],
                }
            )
            orders.append(order)
    return {"rows": rows, "orders": orders}


def ratio_sweep(
    ratios=CLEARANCE_RATIOS, subdivisions=ENVELOPE_SUBDIVISIONS, directions=DIRECTIONS
) -> dict:
    meshes = {s: icosphere(subdivisions=s, radius=1.0) for s in subdivisions}
    rows: list[dict] = []
    for subdiv, mesh in meshes.items():
        for direction in directions:
            for ratio in ratios:
                rows.append({**measure(mesh, direction, ratio * mesh.h_max), "subdivisions": subdiv})
    return {
        "rows": rows,
        "span": {
            "min": min(r["d_over_h"] for r in rows),
            "max": max(r["d_over_h"] for r in rows),
        },
    }


def scale_invariance(sweep: dict, ratio: float = 0.3) -> dict:
    """Error spread across meshes at one ``d / h_max`` (the m3xxx scale invariance)."""
    rows = [
        r
        for r in sweep["rows"]
        if math.isclose(r["d_over_h"], ratio, rel_tol=0.25) and r["direction"] == "equator"
    ]
    if len(rows) < 2:
        return {"ratio": ratio, "evaluable": False}
    errors = [r["relative_error"] for r in rows]
    return {
        "ratio": ratio,
        "evaluable": True,
        "errors": errors,
        "panels": [r["panels"] for r in rows],
        "spread": max(errors) / min(errors),
    }


def envelope_from_sweep(sweep: dict) -> NearWallEnvelope:
    """Worst-probe envelope: per ratio, the largest error over meshes and directions."""
    worst: dict[float, float] = {}
    for row in sweep["rows"]:
        key = round(float(row["d_over_h"]), 12)
        worst[key] = max(worst.get(key, 0.0), float(row["relative_error"]))
    ratios = sorted(worst)
    return build_envelope(ratios, [worst[r] for r in ratios])


def clearance_requirements(envelope: NearWallEnvelope, targets=TARGETS) -> list[dict]:
    rows = []
    for target in targets:
        try:
            ratio = envelope.required_clearance_ratio(target)
            measured = float(envelope.bound(ratio))
            rows.append(
                {
                    "target_relative_error": float(target),
                    "required_d_over_h": ratio,
                    "measured_worst_error_at_ratio": measured,
                    "certified": bool(measured <= target),
                }
            )
        except ValueError as exc:  # pragma: no cover - only if the sweep is truncated
            rows.append({"target_relative_error": float(target), "error": str(exc), "certified": False})
    return rows


def audit(
    subdivisions=SUBDIVISIONS,
    envelope_subdivisions=ENVELOPE_SUBDIVISIONS,
    clearances=FIXED_CLEARANCES,
    ratios=CLEARANCE_RATIOS,
    directions=DIRECTIONS,
    targets=TARGETS,
) -> dict:
    ladder = fixed_clearance_ladder(subdivisions, clearances, directions)
    sweep = ratio_sweep(ratios, envelope_subdivisions, directions)
    envelope = envelope_from_sweep(sweep)
    requirements = clearance_requirements(envelope, targets)

    equator_orders = {
        row["clearance"]: row
        for row in ladder["orders"]
        if row["direction"] == "equator"
    }
    gate_clearance = 0.05
    gate_row = equator_orders.get(gate_clearance)
    order_gate = bool(gate_row is not None and gate_row["order_two_finest"] >= MIN_FIXED_CLEARANCE_ORDER)
    all_orders = [row["order_two_finest"] for row in ladder["orders"] if math.isfinite(row["order_two_finest"])]
    invariance = scale_invariance(sweep)
    requirement_3pct = next(
        (row for row in requirements if math.isclose(row["target_relative_error"], TARGET_ERROR)),
        None,
    )
    envelope_gate = bool(requirement_3pct and requirement_3pct.get("certified"))

    verdict = {
        "target_error_bar": TARGET_ERROR,
        "min_fixed_clearance_order": MIN_FIXED_CLEARANCE_ORDER,
        "scale_invariance_factor": SCALE_INVARIANCE_FACTOR,
        "fixed_clearance_gate": {
            "clearance": gate_clearance,
            "direction": "equator",
            "order_two_finest": None if gate_row is None else gate_row["order_two_finest"],
            "improvement_factor": None if gate_row is None else gate_row["improvement_factor"],
            "pass": order_gate,
        },
        "order_min_over_ladder": float(min(all_orders)) if all_orders else float("nan"),
        "order_max_over_ladder": float(max(all_orders)) if all_orders else float("nan"),
        "scale_invariance": invariance,
        "scale_invariance_pass": bool(
            invariance.get("evaluable") and invariance["spread"] <= SCALE_INVARIANCE_FACTOR
        ),
        "envelope_3pct": requirement_3pct,
        "envelope_certifies_3pct": envelope_gate,
        "refinement_is_a_lever": bool(order_gate),
        "statement": (
            "At fixed physical clearance the constant-panel near-wall velocity converges "
            "(order %.2f at d = %.3g m, equator); at fixed d/h_max it is scale invariant "
            "(spread x%.2f). The m3xxx claim that refinement is not a lever holds only for "
            "the fixed-ratio experiment."
            % (
                gate_row["order_two_finest"] if gate_row else float("nan"),
                gate_clearance,
                invariance.get("spread", float("nan")),
            )
        ),
    }

    return {
        "case": {
            "geometry": "unit sphere a = 1 m, disturbance potential u = z/(2 r^3), unit stream +z",
            "reference": "closed form grad u = e_z/(2 r^3) - 3 z x/(2 r^5) at the probe",
            "traces": "exact analytic values at panel centroids (no solve)",
            "directions": list(directions),
            "direction_vectors": {k: list(v) for k, v in DIRECTION_VECTORS.items()},
            "subdivisions": list(subdivisions),
            "panels": {
                str(s): int(icosphere(subdivisions=s, radius=1.0).panel_count) for s in subdivisions
            },
            "h_max": {
                str(s): float(icosphere(subdivisions=s, radius=1.0).h_max) for s in subdivisions
            },
            "fixed_clearances_m": list(clearances),
            "clearance_ratios": list(ratios),
            "envelope_subdivisions": list(envelope_subdivisions),
            "python": __import__("platform").python_version(),
            "numpy": np.__version__,
        },
        "fixed_clearance": ladder,
        "ratio_sweep": sweep,
        "envelope": envelope.as_dict(),
        "clearance_requirements": requirements,
        "verdict": verdict,
    }


def _fmt(value: float, digits: int = 3) -> str:
    return "%.*e" % (digits, value)


def write_summary(payload: dict, path: Path) -> None:
    """Program-generated Markdown: every number below comes from ``payload``."""
    verdict = payload["verdict"]
    lines: list[str] = []
    lines.append("# Near-wall accuracy envelope (program-generated, m6445)\n")
    lines.append(
        "Generated by `python -m tensorbem.benchmarks.near_wall_envelope_study`. "
        "Do not edit: regenerate.\n"
    )
    lines.append("## 1. Mesh ladder\n")
    lines.append("| subdivisions | panels | h_max [m] |")
    lines.append("|---:|---:|---:|")
    for subdiv, panels in payload["case"]["panels"].items():
        lines.append(f"| {subdiv} | {panels} | {payload['case']['h_max'][subdiv]:.5f} |")

    lines.append("\n## 2. Fixed physical clearance: error vs h (convergence)\n")
    lines.append("| direction | d [m] | coarsest error | finest error | improvement | order (full) | order (2 finest) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in payload["fixed_clearance"]["orders"]:
        lines.append(
            "| %s | %.4f | %s | %s | x%.2f | %.2f | %.2f |"
            % (
                row["direction"],
                row["clearance"],
                _fmt(row["coarsest_error"]),
                _fmt(row["finest_error"]),
                row["improvement_factor"],
                row["order_full_ladder"],
                row["order_two_finest"],
            )
        )

    lines.append("\n## 3. Fixed relative clearance: the same errors vs d / h_max\n")
    lines.append("| direction | d/h_max | mesh panels | measured error |")
    lines.append("|---|---:|---:|---:|")
    for row in payload["ratio_sweep"]["rows"]:
        lines.append(
            "| %s | %.4f | %d | %s |"
            % (row["direction"], row["d_over_h"], row["panels"], _fmt(row["relative_error"]))
        )

    lines.append("\n## 4. Worst-probe envelope (program output)\n")
    lines.append("| d/h_max | worst relative error |")
    lines.append("|---:|---:|")
    for ratio, error in zip(
        payload["envelope"]["clearance_ratio"], payload["envelope"]["relative_error"]
    ):
        lines.append(f"| {ratio:.4f} | {_fmt(error)} |")

    lines.append("\n## 5. Clearance required for a target error\n")
    lines.append("| target | required d/h_max | measured worst error at that ratio | certified |")
    lines.append("|---:|---:|---:|:--:|")
    for row in payload["clearance_requirements"]:
        if "required_d_over_h" in row:
            lines.append(
                "| %.0f %% | %.4f | %s | %s |"
                % (
                    100.0 * row["target_relative_error"],
                    row["required_d_over_h"],
                    _fmt(row["measured_worst_error_at_ratio"]),
                    "yes" if row["certified"] else "no",
                )
            )
        else:
            lines.append("| %.0f %% | not measured | %s | no |" % (100.0 * row["target_relative_error"], row["error"]))

    lines.append("\n## 6. Verdict (verbatim from the JSON)\n")
    lines.append("```json")
    lines.append(json.dumps(verdict, indent=2, default=float))
    lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figures(payload: dict, outdir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # Figure 1: fixed physical clearance -> convergence
    fig, axes = plt.subplots(figsize=(6.6, 4.4), dpi=150)
    orders = {(row["clearance"]): row for row in payload["fixed_clearance"]["orders"] if row["direction"] == "equator"}
    for clearance in payload["case"]["fixed_clearances_m"]:
        rows = [
            r
            for r in payload["fixed_clearance"]["rows"]
            if r["direction"] == "equator" and math.isclose(r["clearance"], clearance)
        ]
        rows.sort(key=lambda r: r["h_max"], reverse=True)
        order = orders[clearance]["order_two_finest"]
        axes.loglog(
            [r["h_max"] for r in rows],
            [r["relative_error"] for r in rows],
            "o-",
            label="d = %.4g m (order %.2f)" % (clearance, order),
        )
    axes.axhline(TARGET_ERROR, color="tab:red", ls=":", lw=0.9, label="3 % bar")
    axes.set_xlabel(r"$h_{\max}$ [m]")
    axes.set_ylabel("relative velocity error (equator)")
    axes.set_title("Fixed physical clearance: the near-wall field does converge")
    axes.grid(alpha=0.3, which="both")
    axes.legend(fontsize=7)
    fig.tight_layout()
    path = outdir / "fig1_fixed_clearance_convergence.png"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    # Figure 2: error collapse against d / h_max
    fig, axes = plt.subplots(figsize=(6.6, 4.4), dpi=150)
    for subdiv in payload["case"]["envelope_subdivisions"]:
        for direction, style in (("equator", "-"), ("pole", "--"), ("oblique", ":")):
            rows = [
                r
                for r in payload["ratio_sweep"]["rows"]
                if r["subdivisions"] == subdiv and r["direction"] == direction
            ]
            rows.sort(key=lambda r: r["d_over_h"])
            axes.loglog(
                [r["d_over_h"] for r in rows],
                [r["relative_error"] for r in rows],
                style,
                marker="o",
                ms=3,
                lw=1.0,
                label="%d panels, %s" % (payload["case"]["panels"][str(subdiv)], direction),
            )
    axes.axhline(TARGET_ERROR, color="tab:red", ls=":", lw=0.9, label="3 % bar")
    axes.set_xlabel(r"$d / h_{\max}$")
    axes.set_ylabel("relative velocity error")
    axes.set_title("Fixed relative clearance: scale invariance (curves collapse)")
    axes.grid(alpha=0.3, which="both")
    axes.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    path = outdir / "fig2_ratio_collapse.png"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)

    # Figure 3: clearance requirement from the envelope
    ratios = np.asarray(payload["envelope"]["clearance_ratio"], dtype=float)
    errors = np.asarray(payload["envelope"]["relative_error"], dtype=float)
    fig, axes = plt.subplots(figsize=(6.6, 4.0), dpi=150)
    axes.loglog(ratios, errors, "o-", label="worst-probe envelope (measured)")
    for row in payload["clearance_requirements"]:
        if "required_d_over_h" in row:
            axes.axvline(
                row["required_d_over_h"],
                color="tab:green",
                ls="--",
                lw=0.8,
                alpha=0.7,
            )
            axes.annotate(
                "%.0f %%" % (100.0 * row["target_relative_error"]),
                (row["required_d_over_h"], errors[0]),
                fontsize=7,
                rotation=90,
                va="top",
            )
    axes.set_xlabel(r"$d / h_{\max}$")
    axes.set_ylabel("relative velocity error")
    axes.set_title("Clearance required for a target near-wall error")
    axes.grid(alpha=0.3, which="both")
    axes.legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "fig3_clearance_requirement.png"
    fig.savefig(path)
    plt.close(fig)
    paths.append(path)
    return paths


def main(outdir: Path = OUTPUT_DIR, figures: bool = True) -> dict:
    payload = audit()
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "study.json").write_text(
        json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8"
    )
    write_summary(payload, outdir / "summary.md")
    if figures:
        payload["figures"] = [str(p) for p in make_figures(payload, outdir)]
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    result = main(args.outdir, figures=not args.no_figures)
    print(json.dumps(result["verdict"], indent=2, default=float))