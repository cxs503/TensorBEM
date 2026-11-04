"""Two-sided convergence study of the finite-depth radiation coefficients.

The finite-depth panel solver (:mod:`tensorbem.finite_depth_solver`) is verified
against two things: the near-field/far-field energy identity and the method of
fundamental solutions (MFS) reference of the same kernel.  ``docs/finite_depth_radiation.md``
reports that the two discretisations disagree by 2.8-4.0 % in the damping at 64
panels and explicitly declines to call that gap a panel error, because the MFS
"reference" had never been shown to be converged either.  A gap between two
approximations is only evidence once the size of each approximation is known.

This module measures both sides:

* **panel side** - a panel sweep ``N = 64 ... 256`` at fixed frequency, giving the
  observed order of the piecewise-constant formulation and a Richardson
  extrapolated value for every coefficient;
* **reference side** - an MFS stability grid over source count (16-96) and source
  radius (0.35-0.65 ``a``), giving the reference its own spread;
* **closure** - the Richardson limit of the panel sequence and the converged MFS
  value are two independent discretisations of the same analytic kernel, so their
  residual is the error bar that the comparison inherits;
* **energy identity** - the near-field/far-field damping ratio is fitted in the
  same sweep, which turns the "closed to about 1 %" statement of the earlier
  report into a measured order and an extrapolated limit.

Everything printed in ``docs/m2952/finite_depth_refinement.md`` comes from the
JSON this module writes; nothing is transcribed by hand.

Usage::

    PYTHONPATH=src python3 -m tensorbem.benchmarks.finite_depth_refinement \
        --json docs/m2952/outputs/finite_depth_refinement/benchmark.json \
        --figures docs/m2952/figures
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from tensorbem.benchmarks.finite_depth_radiation import (
    GRAVITY,
    MOTIONS,
    submerged_circle_mesh,
)
from tensorbem.finite_depth_solver import radiation_reference, solve_radiation

#: Panel counts per (depth, motion) case.  The heave case at the tightest
#: clearance is carried two levels further because it is the one the earlier
#: report flagged (4.0 % damping disagreement at 64 panels).
PANEL_SWEEP: dict[tuple[float, str], tuple[int, ...]] = {
    (3.2, "heave"): (64, 96, 128, 192, 256),
    (3.2, "sway"): (64, 96, 128, 192),
    (5.0, "heave"): (64, 96, 128),
    (5.0, "sway"): (64, 96, 128),
}

#: MFS stability grid: the default of the earlier benchmark is (24, 0.55).
MFS_SOURCE_COUNTS = (16, 24, 48, 96)
MFS_RADIUS_RATIOS = (0.35, 0.45, 0.55, 0.65)


def _value(result, index: int) -> tuple[float, float]:
    """Return the scalar ``(added mass, damping)`` of one motion component."""
    return float(result.added_mass[index]), float(result.damping[index])


def panel_row(
    depth: float,
    motion: str,
    panel_count: int,
    *,
    ka: float = 0.6,
    radius: float = 1.0,
    submergence: float = 2.0,
    density: float = 1025.0,
    source_count: int = 48,
    source_radius_ratio: float = 0.55,
) -> dict:
    """Solve one panel resolution and compare it with the MFS reference."""
    nu = ka / radius
    index = MOTIONS[motion]
    mesh = submerged_circle_mesh(panel_count, radius, submergence)
    started = time.time()
    result, far = solve_radiation(
        depth=depth, mesh=mesh, nu=nu, direction=motion, density=density, gravity=GRAVITY
    )
    panel_seconds = time.time() - started
    reference = radiation_reference(
        mesh,
        nu,
        depth,
        motion,
        source_count=source_count,
        source_radius_ratio=source_radius_ratio,
        density=density,
        gravity=GRAVITY,
    )
    added_mass, damping = _value(result, index)
    mfs_mass, mfs_damping = _value(reference, index)
    return {
        "depth": float(depth),
        "motion": motion,
        "panels": int(panel_count),
        "panel_width": float(2.0 * math.pi * radius / panel_count),
        "added_mass": added_mass,
        "damping": damping,
        "far_field_damping": float(far),
        "far_over_near": float(far / damping),
        "mfs_added_mass": mfs_mass,
        "mfs_damping": mfs_damping,
        "mass_rel_diff": float(mfs_mass / added_mass - 1.0),
        "damping_rel_diff": float(mfs_damping / damping - 1.0),
        "condition_number": float(result.condition_number),
        "relative_residual": float(result.relative_residual),
        "panel_seconds": round(panel_seconds, 3),
    }


def mfs_stability(
    depth: float,
    motion: str,
    panel_count: int,
    *,
    ka: float = 0.6,
    radius: float = 1.0,
    submergence: float = 2.0,
    density: float = 1025.0,
) -> dict:
    """Spread of the MFS reference over its own two discretisation parameters.

    A reference that moves with its parameters cannot arbitrate a 1 % question.
    The collocation mesh is held fixed, so this measures the MFS approximation
    only.
    """
    nu = ka / radius
    index = MOTIONS[motion]
    mesh = submerged_circle_mesh(panel_count, radius, submergence)
    rows = []
    for ratio in MFS_RADIUS_RATIOS:
        for sources in MFS_SOURCE_COUNTS:
            reference = radiation_reference(
                mesh,
                nu,
                depth,
                motion,
                source_count=sources,
                source_radius_ratio=ratio,
                density=density,
                gravity=GRAVITY,
            )
            mass, damping = _value(reference, index)
            rows.append(
                {
                    "source_count": int(sources),
                    "source_radius_ratio": float(ratio),
                    "added_mass": mass,
                    "damping": damping,
                }
            )
    masses = np.array([row["added_mass"] for row in rows])
    dampings = np.array([row["damping"] for row in rows])
    default = next(
        row
        for row in rows
        if row["source_count"] == 48 and row["source_radius_ratio"] == 0.55
    )
    return {
        "depth": float(depth),
        "motion": motion,
        "panels": int(panel_count),
        "rows": rows,
        "mass_spread": float(masses.max() / masses.min() - 1.0),
        "damping_spread": float(dampings.max() / dampings.min() - 1.0),
        "mass_default": float(default["added_mass"]),
        "damping_default": float(default["damping"]),
        "mass_max_dev_from_default": float(
            np.max(np.abs(masses / default["added_mass"] - 1.0))
        ),
        "damping_max_dev_from_default": float(
            np.max(np.abs(dampings / default["damping"] - 1.0))
        ),
    }


def observed_order(coarse: float, fine: float, factor: float = 2.0) -> float:
    """Order ``p`` from two error magnitudes separated by ``factor`` in panels.

    ``p = log(e_coarse/e_fine)/log(factor)``.  The refinement factor matters:
    the panel sequences used here are 64-96-128-192-256, i.e. 1.5x and 1.33x
    steps, and assuming dyadic spacing would overstate the order.  Arguments are
    error magnitudes; a non-positive argument or a stalled sequence (``p <= 0``)
    returns ``nan`` instead of a meaningless number.
    """
    if factor <= 1.0:
        raise ValueError("factor must exceed 1")
    if coarse <= 0.0 or fine <= 0.0:
        return float("nan")
    order = math.log(coarse / fine) / math.log(factor)
    return order if order > 0.0 else float("nan")


def richardson_limit(
    coarse: float, fine: float, factor: float = 2.0, order: float | None = None
) -> dict:
    """Extrapolate two values of a converging sequence to zero mesh size.

    ``L = v_fine + (v_fine - v_coarse)/(factor**p - 1)``.  Returns the order
    used, the limit and how far the limit moves from the finest computed value;
    a stalled or one-sided pair returns ``nan`` rather than an extrapolation.
    """
    p = observed_order(abs(fine - coarse) or 1.0, 1.0, factor) if order is None else order
    if not math.isfinite(p) or p <= 0.0:
        return {"order": float("nan"), "limit": float("nan"), "finest": fine, "limit_shift": float("nan")}
    limit = fine + (fine - coarse) / (factor**p - 1.0)
    return {
        "order": float(p),
        "limit": float(limit),
        "finest": float(fine),
        "limit_shift": float(abs(limit / fine - 1.0)) if fine else float("nan"),
    }


def refine(rows: list[dict], coefficient: str, error_key: str) -> dict:
    """Order and Richardson limit of one coefficient along a panel sequence.

    ``coefficient`` is the solved value (added mass or damping) and ``error_key``
    the column holding its difference from the MFS reference, which is used as
    the error yardstick: the MFS value is converged in its own parameters to
    ~0.02 % (see :func:`mfs_stability`), so ``|panel - MFS|`` is a usable proxy
    for the panel error at every level.  The order is measured on the *finest*
    pair so non-dyadic spacing is handled exactly, and every pair is kept in
    ``steps`` for the report.
    """
    ordered = sorted(rows, key=lambda row: row["panels"])
    values = [row[coefficient] for row in ordered]
    errors = [abs(row[error_key]) for row in ordered]
    if len(values) < 2:
        return {"order": float("nan"), "limit": float("nan"), "steps": [], "finest": float("nan"), "limit_shift": float("nan")}
    steps = []
    for i in range(len(values) - 1):
        factor = ordered[i + 1]["panels"] / ordered[i]["panels"]
        steps.append(
            {
                "from_panels": ordered[i]["panels"],
                "to_panels": ordered[i + 1]["panels"],
                "factor": factor,
                "order": observed_order(errors[i], errors[i + 1], factor),
            }
        )
    factor = steps[-1]["factor"]
    order = steps[-1]["order"]
    limit = richardson_limit(values[-2], values[-1], factor, order if math.isfinite(order) else None)
    limit["steps"] = steps
    return limit


def far_field_steps(rows: list[dict], key: str = "far_over_near") -> dict:
    """Observed order of the energy-identity offset ``|ratio - 1|``."""
    ordered = sorted(rows, key=lambda row: row["panels"])
    offsets = [abs(row[key] - 1.0) for row in ordered]
    ratios = []
    for i in range(len(offsets) - 1):
        if offsets[i] <= 0.0 or offsets[i + 1] <= 0.0:
            ratios.append(float("nan"))
            continue
        factor = ordered[i + 1]["panels"] / ordered[i]["panels"]
        ratios.append(math.log(offsets[i] / offsets[i + 1]) / math.log(factor))
    return {
        "panels": [row["panels"] for row in ordered],
        "offsets": offsets,
        "ratios": ratios,
        "mean_order": float(np.nanmean(ratios)) if ratios else float("nan"),
    }


def run(
    *,
    ka: float = 0.6,
    radius: float = 1.0,
    submergence: float = 2.0,
    density: float = 1025.0,
    stability_panels: int = 128,
) -> dict:
    """Run the panel sweep, the MFS stability grid and the closure checks."""
    started = time.time()
    cases = []
    for (depth, motion), panels in PANEL_SWEEP.items():
        rows = [
            panel_row(
                depth,
                motion,
                count,
                ka=ka,
                radius=radius,
                submergence=submergence,
                density=density,
            )
            for count in panels
        ]
        mass = refine(rows, "added_mass", "mass_rel_diff")
        damping = refine(rows, "damping", "damping_rel_diff")
        damp = [row["damping"] for row in sorted(rows, key=lambda r: r["panels"])]
        mass_values = [row["added_mass"] for row in sorted(rows, key=lambda r: r["panels"])]
        ordered = sorted(rows, key=lambda row: row["panels"])
        last_factor = ordered[-1]["panels"] / ordered[-2]["panels"]
        damping_rich = richardson_limit(damp[-2], damp[-1], last_factor, damping["order"])
        mass_rich = richardson_limit(mass_values[-2], mass_values[-1], last_factor, mass["order"])
        finest = max(rows, key=lambda row: row["panels"])
        cases.append(
            {
                "depth": float(depth),
                "motion": motion,
                "rows": rows,
                "mass_order": mass["order"],
                "damping_order": damping["order"],
                "mass_richardson": mass_rich,
                "damping_richardson": damping_rich,
                "far_field": far_field_steps(rows),
                "mfs_mass_residual_at_limit": float(
                    finest["mfs_added_mass"] / mass_rich["limit"] - 1.0
                )
                if math.isfinite(mass_rich["limit"])
                else float("nan"),
                "mfs_damping_residual_at_limit": float(
                    finest["mfs_damping"] / damping_rich["limit"] - 1.0
                )
                if math.isfinite(damping_rich["limit"])
                else float("nan"),
                "finest_panels": int(finest["panels"]),
                "finest_mass_rel_diff": finest["mass_rel_diff"],
                "finest_damping_rel_diff": finest["damping_rel_diff"],
                "panels_for_3_percent_damping": _panels_for(
                    rows, "damping_rel_diff", 0.03
                ),
                "panels_for_3_percent_mass": _panels_for(rows, "mass_rel_diff", 0.03),
                "panels_for_1_percent_damping": _panels_for(
                    rows, "damping_rel_diff", 0.01
                ),
            }
        )
    stability = [
        mfs_stability(
            depth,
            motion,
            stability_panels,
            ka=ka,
            radius=radius,
            submergence=submergence,
            density=density,
        )
        for depth, motion in ((3.2, "heave"), (3.2, "sway"))
    ]
    return {
        "benchmark": "finite_depth_radiation_two_sided_convergence",
        "ka": ka,
        "radius": radius,
        "submergence": submergence,
        "density": density,
        "gravity": GRAVITY,
        "nu": ka / radius,
        "elapsed_seconds": round(time.time() - started, 1),
        "cases": cases,
        "mfs_stability": stability,
        "mfs_stability_panels": stability_panels,
    }


def _panels_for(rows: list[dict], key: str, threshold: float) -> int | None:
    """Smallest panel count in the sweep whose ``|key|`` is below ``threshold``."""
    for row in sorted(rows, key=lambda r: r["panels"]):
        if abs(row[key]) < threshold:
            return int(row["panels"])
    return None


def plot(payload: dict, directory: Path) -> list[Path]:
    """Three-panel figure: panel error, energy identity, reference spread."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    cases = {f"{c['depth']:.1f}/{c['motion']}": c for c in payload["cases"]}

    figure, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))

    colours = {"3.2/heave": "tab:blue", "3.2/sway": "tab:orange"}
    markers = {"added mass": "o", "damping": "s"}
    for label, colour in colours.items():
        if label not in cases:
            continue
        rows = sorted(cases[label]["rows"], key=lambda row: row["panels"])
        width = np.array([row["panel_width"] for row in rows])
        for key, name in (("mass_rel_diff", "added mass"), ("damping_rel_diff", "damping")):
            axes[0].loglog(
                width,
                np.abs([row[key] for row in rows]),
                marker=markers[name],
                color=colour,
                label=f"{name}, h/a={label}",
            )
    reference_width = np.array([0.4, 0.11])
    axes[0].loglog(
        reference_width,
        0.03 * reference_width / 0.11,
        "k--",
        linewidth=1.0,
        label="first order (slope 1)",
    )
    axes[0].axhline(0.03, color="grey", linestyle=":", linewidth=1.0)
    axes[0].set_xlabel("panel width $h/a$")
    axes[0].set_ylabel("|panel $-$ MFS| / MFS")
    axes[0].set_title("Panel discretisation error against the MFS reference")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7)

    for label, colour in colours.items():
        if label not in cases:
            continue
        rows = sorted(cases[label]["rows"], key=lambda row: row["panels"])
        panels = np.array([row["panels"] for row in rows])
        axes[1].semilogx(
            panels,
            [row["far_over_near"] - 1.0 for row in rows],
            marker="o",
            color=colour,
            label=f"h/a={label}",
        )
    axes[1].axhline(0.0, color="k", linewidth=1.0)
    axes[1].set_xlabel("panels")
    axes[1].set_ylabel("$B_{far}/B_{near} - 1$")
    axes[1].set_title("Energy identity: offset against panel count")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)

    for index, stability in enumerate(payload["mfs_stability"]):
        masses = np.array([row["added_mass"] for row in stability["rows"]])
        dampings = np.array([row["damping"] for row in stability["rows"]])
        offsets = np.concatenate(
            (
                masses / stability["mass_default"] - 1.0,
                dampings / stability["damping_default"] - 1.0,
            )
        )
        axes[2].plot(
            np.full(offsets.shape, index, dtype=float) + np.linspace(-0.12, 0.12, offsets.size),
            offsets,
            "o",
            markersize=4,
            label=f"h/a={stability['depth']:.1f} {stability['motion']}",
        )
    axes[2].axhline(0.0, color="k", linewidth=1.0)
    axes[2].axhline(0.01, color="grey", linestyle=":", linewidth=1.0)
    axes[2].axhline(-0.01, color="grey", linestyle=":", linewidth=1.0)
    axes[2].set_xticks(range(len(payload["mfs_stability"])))
    axes[2].set_xticklabels(
        [f"{s['depth']:.1f}\n{s['motion']}" for s in payload["mfs_stability"]]
    )
    axes[2].set_ylabel("MFS value / default $-$ 1")
    axes[2].set_title("MFS reference spread over source count and radius")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=8)

    figure.suptitle(
        "TensorBEM finite-depth radiation: panel error, energy identity, reference spread "
        f"($ka$={payload['ka']})"
    )
    figure.tight_layout()
    path = directory / "finite_depth_refinement.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--figures", type=Path, default=None)
    parser.add_argument("--ka", type=float, default=0.6)
    args = parser.parse_args(argv)

    payload = run(ka=args.ka)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.figures:
        for path in plot(payload, args.figures):
            print(f"wrote {path}")
    for case in payload["cases"]:
        print(
            f"h/a={case['depth']:.1f} {case['motion']:5s} "
            f"mass order {case['mass_order']:.2f} damping order {case['damping_order']:.2f} "
            f"finest {case['finest_panels']} panels: "
            f"dA {100 * case['finest_mass_rel_diff']:+.2f}% "
            f"dB {100 * case['finest_damping_rel_diff']:+.2f}%"
        )
    print(f"elapsed {payload['elapsed_seconds']} s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())