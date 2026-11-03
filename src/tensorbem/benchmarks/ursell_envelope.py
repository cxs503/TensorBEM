"""Frequency and depth envelope of the 2-D radiation solver (Ursell cylinder).

``docs/ursell_error_budget.md`` §5.2 records an open item::

    Only one ``(ka, a/f)`` point was audited -- at ``ka >= 1`` the free-surface
    term is stronger and the convergence behaviour may differ.  Unverified.

That is the gap this module closes.  The shipped seakeeping workflow does not
run at a single non-dimensional frequency: a ship RAO is evaluated over
roughly ``ka = 0.2 ... 3``, and the waterline clearance ``a/f`` decides how
strong the free-surface image is.  An accuracy claim measured at
``(0.6, 0.5)`` therefore says nothing about the corners of the range that is
actually used.  This module measures the error of the constant-panel indirect
formulation against Ursell's multipole solution on a *grid*

* ``ka`` in ``(0.3, 0.6, 1.0, 1.5, 2.0)`` -- from long waves to the regime where
  the exponential kernel ``exp(-k A) cos(k X)`` oscillates across the panel,
* ``a/f`` in ``(0.5, 0.7, 0.85)`` -- from ``f = 2a`` down to ``f = 1.18a``,
  where the free-surface image sits closest to the body,

and at each grid point also checks the two things that could make the
comparison itself dishonest:

1. **Reference adequacy.**  The multipole system is truncated; the panel error
   is only a *discretisation* error if the truncated reference value has
   settled.  The change from 40 to 120 harmonics is recorded per grid point.
2. **Method independence.**  The method of fundamental solutions (no singular
   quadrature, exponential convergence) is solved at the same grid point, so a
   grid point whose panel error is large can be attributed to the panel code
   rather than to the reference.

A refinement ladder ``N = 64, 128, 256`` at ``ka = 0.6, 1.0, 2.0`` supplies the
observed convergence order, so the envelope can be stated as *how many panels
each corner of the range needs* rather than as a single error number.

Run::

    python -m tensorbem.benchmarks.ursell_envelope --json artifacts/ursell_envelope.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tensorbem.analytic import ursell_cylinder_coefficients
from tensorbem.benchmarks.ursell_cylinder import submerged_circle_mesh
from tensorbem.free_surface_solver import (
    radiation_reference,
    solve_radiation_indirect,
)

RADIUS = 1.0
DENSITY = 1025.0
GRAVITY = 9.80665

#: Non-dimensional frequency grid (``ka`` with ``a = 1 m``, so ``ka`` is also
#: the wave number in 1/m).
KA_VALUES: tuple[float, ...] = (0.6, 1.0, 1.5, 2.0)
#: Submergence ratios ``a/f``: 0.5 is the historically audited point, 0.85 is
#: the shallowest clearance for which the multipole solution stays regular.
DEPTH_RATIOS: tuple[float, ...] = (0.5, 0.7, 0.85)
#: Panel counts used for the envelope itself.
PANEL_COUNT = 128
PANEL_NODES = 24
#: Refinement ladder and the frequencies at which it is run.
REFINEMENT_COUNTS: tuple[int, ...] = (64, 128)
REFINEMENT_KA: tuple[float, ...] = (0.6, 1.0, 2.0)
#: Sources in the independent (MFS) cross-check.
MFS_SOURCES = 64
#: Harmonics for the reference value and for the truncation check.
CONVERGED_TERMS = 120
CHECK_TERMS = 40

#: Acceptance bands (fractional error on the larger of the two coefficients).
PASS_BAND = 0.03
FAIL_BAND = 0.15


def _submergence(a_over_f: float) -> float:
    if not (0.0 < a_over_f < 1.0):
        raise ValueError("a_over_f must satisfy 0 < a/f < 1")
    return RADIUS / a_over_f


def _dimensionless(added_mass: float, damping: float, wavenumber: float) -> tuple[float, float]:
    """Normalise the way Ursell does: by ``rho pi a^2`` and ``rho pi a^2 omega``."""
    omega = math.sqrt(wavenumber * GRAVITY)
    displaced = DENSITY * math.pi * RADIUS**2
    return float(added_mass) / displaced, float(damping) / (displaced * omega)


def _panel(
    ka: float,
    a_over_f: float,
    panels: int = PANEL_COUNT,
    *,
    panel_nodes: int = PANEL_NODES,
    direction: str = "heave",
) -> tuple[float, float]:
    """Dimensionless ``(mu, nu)`` from the constant-panel indirect solve."""
    mesh = submerged_circle_mesh(panels, RADIUS, _submergence(a_over_f))
    wavenumber = ka / RADIUS
    result = solve_radiation_indirect(
        mesh, wavenumber, direction, density=DENSITY, panel_nodes=panel_nodes
    )
    index = 1 if direction == "heave" else 0
    return _dimensionless(result.added_mass[index], result.damping[index], wavenumber)


def _mfs(
    ka: float,
    a_over_f: float,
    sources: int = MFS_SOURCES,
    *,
    panels: int = 128,
    direction: str = "heave",
) -> tuple[float, float]:
    """Dimensionless ``(mu, nu)`` from the method of fundamental solutions."""
    mesh = submerged_circle_mesh(panels, RADIUS, _submergence(a_over_f))
    wavenumber = ka / RADIUS
    result = radiation_reference(
        mesh, wavenumber, direction, source_count=sources, density=DENSITY
    )
    index = 1 if direction == "heave" else 0
    return _dimensionless(result.added_mass[index], result.damping[index], wavenumber)


def _relative_error(value: tuple[float, float], reference: tuple[float, float]) -> tuple[float, float]:
    return (
        abs(value[0] - reference[0]) / abs(reference[0]),
        abs(value[1] - reference[1]) / abs(reference[1]),
    )


def classify(added_mass_error: float, damping_error: float) -> str:
    """Three-band verdict on the worse of the two coefficients."""
    worst = max(added_mass_error, damping_error)
    if worst <= PASS_BAND:
        return "pass"
    if worst > FAIL_BAND:
        return "fail"
    return "marginal"


def reference_truncation(ka: float, a_over_f: float) -> dict[str, float]:
    """Change in the reference value between 40 and 120 harmonics."""
    short = ursell_cylinder_coefficients(ka, a_over_f, CHECK_TERMS)
    long = ursell_cylinder_coefficients(ka, a_over_f, CONVERGED_TERMS)
    delta = _relative_error(short, long)
    return {
        "ka": ka,
        "a_over_f": a_over_f,
        "mu_check": short[0],
        "nu_check": short[1],
        "mu_converged": long[0],
        "nu_converged": long[1],
        "added_mass_shift": delta[0],
        "damping_shift": delta[1],
    }


def envelope_point(ka: float, a_over_f: float) -> dict[str, object]:
    """One grid point: reference, panel solve, independent MFS solve."""
    truncation = reference_truncation(ka, a_over_f)
    reference = (truncation["mu_converged"], truncation["nu_converged"])
    mu, nu = _panel(ka, a_over_f)
    error = _relative_error((mu, nu), reference)
    mfs_mu, mfs_nu = _mfs(ka, a_over_f)
    mfs_error = _relative_error((mfs_mu, mfs_nu), reference)
    return {
        "ka": ka,
        "a_over_f": a_over_f,
        "panels": PANEL_COUNT,
        "panel_nodes": PANEL_NODES,
        "mu_reference": reference[0],
        "nu_reference": reference[1],
        "mu_panel": mu,
        "nu_panel": nu,
        "added_mass_relative_error": error[0],
        "damping_relative_error": error[1],
        "worst_relative_error": max(error),
        "verdict": classify(*error),
        "mu_mfs": mfs_mu,
        "nu_mfs": mfs_nu,
        "mfs_added_mass_relative_error": mfs_error[0],
        "mfs_damping_relative_error": mfs_error[1],
        "reference_added_mass_shift_40_to_120": truncation["added_mass_shift"],
        "reference_damping_shift_40_to_120": truncation["damping_shift"],
    }


def envelope_grid(
    ka_values: tuple[float, ...] = KA_VALUES,
    depth_ratios: tuple[float, ...] = DEPTH_RATIOS,
) -> list[dict[str, object]]:
    return [
        envelope_point(ka, ratio) for ka in ka_values for ratio in depth_ratios
    ]


def refinement_study(
    ka_values: tuple[float, ...] = REFINEMENT_KA,
    depth_ratios: tuple[float, ...] = DEPTH_RATIOS,
    counts: tuple[int, ...] = REFINEMENT_COUNTS,
) -> list[dict[str, object]]:
    """Panel ladder per (frequency, clearance) with an observed order."""
    rows: list[dict[str, object]] = []
    for ka in ka_values:
        for ratio in depth_ratios:
            reference_truncation(ka, ratio)  # keeps the reference identical to above
            reference = ursell_cylinder_coefficients(ka, ratio, CONVERGED_TERMS)
            previous: dict[str, object] | None = None
            for count in counts:
                mu, nu = _panel(ka, ratio, count)
                error = _relative_error((mu, nu), reference)
                row: dict[str, object] = {
                    "ka": ka,
                    "a_over_f": ratio,
                    "panels": count,
                    "mu": mu,
                    "nu": nu,
                    "added_mass_relative_error": error[0],
                    "damping_relative_error": error[1],
                }
                if previous is not None:
                    order_mu = _observed_order(
                        previous["added_mass_relative_error"], error[0]
                    )
                    order_nu = _observed_order(
                        previous["damping_relative_error"], error[1]
                    )
                    row["added_mass_observed_order"] = order_mu
                    row["damping_observed_order"] = order_nu
                rows.append(row)
                previous = row
    return rows


def _observed_order(coarse: float, fine: float) -> float | None:
    """Order from an exact factor-2 refinement; ``None`` if not measurable."""
    if coarse <= 0.0 or fine <= 0.0:
        return None
    return math.log(coarse / fine) / math.log(2.0)


def panels_needed(
    rows: list[dict[str, object]], band: float = PASS_BAND
) -> list[dict[str, object]]:
    """Smallest panel count in the ladder whose worst error meets ``band``."""
    grouped: dict[tuple[float, float], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((row["ka"], row["a_over_f"]), []).append(row)
    answer = []
    for (ka, ratio), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda row: row["panels"])
        qualified = [
            row
            for row in ordered
            if max(row["added_mass_relative_error"], row["damping_relative_error"]) <= band
        ]
        answer.append(
            {
                "ka": ka,
                "a_over_f": ratio,
                "smallest_passing_panels": (
                    qualified[0]["panels"] if qualified else None
                ),
                "smallest_measured_error": max(
                    ordered[0]["added_mass_relative_error"],
                    ordered[0]["damping_relative_error"],
                ),
                "order_estimate_added_mass": ordered[-1].get("added_mass_observed_order"),
                "order_estimate_damping": ordered[-1].get("damping_observed_order"),
            }
        )
    return answer


def summarise(grid: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate the grid: worst point, per-frequency and per-clearance maxima."""
    worst = max(grid, key=lambda row: row["worst_relative_error"])
    by_ka: dict[str, dict[str, float]] = {}
    for ka in sorted({row["ka"] for row in grid}):
        rows = [row for row in grid if row["ka"] == ka]
        by_ka[str(ka)] = {
            "max_added_mass_error": max(r["added_mass_relative_error"] for r in rows),
            "max_damping_error": max(r["damping_relative_error"] for r in rows),
        }
    by_ratio: dict[str, dict[str, float]] = {}
    for ratio in sorted({row["a_over_f"] for row in grid}):
        rows = [row for row in grid if row["a_over_f"] == ratio]
        by_ratio[str(ratio)] = {
            "max_added_mass_error": max(r["added_mass_relative_error"] for r in rows),
            "max_damping_error": max(r["damping_relative_error"] for r in rows),
        }
    return {
        "grid_points": len(grid),
        "verdict_counts": {
            verdict: sum(1 for row in grid if row["verdict"] == verdict)
            for verdict in ("pass", "marginal", "fail")
        },
        "worst_point": {
            "ka": worst["ka"],
            "a_over_f": worst["a_over_f"],
            "added_mass_relative_error": worst["added_mass_relative_error"],
            "damping_relative_error": worst["damping_relative_error"],
            "verdict": worst["verdict"],
        },
        "max_reference_shift_40_to_120": max(
            max(row["reference_added_mass_shift_40_to_120"],
                row["reference_damping_shift_40_to_120"])
            for row in grid
        ),
        "max_mfs_error": max(
            max(row["mfs_added_mass_relative_error"], row["mfs_damping_relative_error"])
            for row in grid
        ),
        "by_ka": by_ka,
        "by_depth_ratio": by_ratio,
    }


def run(
    ka_values: tuple[float, ...] = KA_VALUES,
    depth_ratios: tuple[float, ...] = DEPTH_RATIOS,
    *,
    with_refinement: bool = True,
) -> dict[str, object]:
    grid = envelope_grid(ka_values, depth_ratios)
    payload: dict[str, object] = {
        "radius": RADIUS,
        "density": DENSITY,
        "gravity": GRAVITY,
        "panel_count": PANEL_COUNT,
        "panel_nodes": PANEL_NODES,
        "converged_terms": CONVERGED_TERMS,
        "mfs_sources": MFS_SOURCES,
        "pass_band": PASS_BAND,
        "fail_band": FAIL_BAND,
        "grid": grid,
        "summary": summarise(grid),
    }
    if with_refinement:
        rows = refinement_study(ka_values=(0.6, 1.0, 2.0), depth_ratios=depth_ratios)
        payload["refinement"] = rows
        payload["panels_needed"] = panels_needed(rows)
    return payload


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def plot(payload: dict[str, object], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = payload["grid"]
    ratios = sorted({row["a_over_f"] for row in grid})
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    colours = plt.cm.viridis(np.linspace(0.1, 0.8, len(ratios)))

    for axis, key, title in (
        (axes[0], "added_mass_relative_error", "added mass $|\\Delta\\mu|/\\mu$"),
        (axes[1], "damping_relative_error", "damping $|\\Delta\\nu|/\\nu$"),
    ):
        for colour, ratio in zip(colours, ratios):
            rows = sorted(
                (row for row in grid if row["a_over_f"] == ratio),
                key=lambda row: row["ka"],
            )
            axis.semilogy(
                [row["ka"] for row in rows],
                [row[key] for row in rows],
                "o-",
                color=colour,
                label=f"$a/f={ratio}$",
            )
        axis.axhline(PASS_BAND, color="0.4", ls="--", lw=1.0, label="3 % bar")
        axis.axhline(FAIL_BAND, color="crimson", ls=":", lw=1.2, label="15 % bar")
        axis.set_xlabel("$ka$")
        axis.set_title(title + f"  ({PANEL_COUNT} panels)")
        axis.grid(alpha=0.3)

    axes[0].legend(fontsize=8)
    rows = sorted(
        (row for row in grid if row["a_over_f"] == ratios[0]), key=lambda row: row["ka"]
    )
    axes[2].plot(
        [row["ka"] for row in rows],
        [row["mu_reference"] for row in rows],
        "k-",
        label="$\\mu$ reference",
    )
    axes[2].plot(
        [row["ka"] for row in rows],
        [row["mu_panel"] for row in rows],
        "ko",
        mfc="none",
        label="$\\mu$ panels",
    )
    axes[2].plot(
        [row["ka"] for row in rows],
        [row["nu_reference"] for row in rows],
        "b-",
        label="$\\nu$ reference",
    )
    axes[2].plot(
        [row["ka"] for row in rows],
        [row["nu_panel"] for row in rows],
        "bs",
        mfc="none",
        label="$\\nu$ panels",
    )
    axes[2].set_xlabel("$ka$")
    axes[2].set_title(f"coefficients, $a/f={ratios[0]}$")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    figure.suptitle(
        "Ursell heave radiation: accuracy envelope over frequency and clearance",
        fontsize=12,
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def render_markdown(payload: dict[str, object]) -> str:
    summary = payload["summary"]
    grid = payload["grid"]
    lines = [
        "## Grid (heave, "
        f"{payload['panel_count']} panels, {payload['panel_nodes']} nodes/panel)",
        "",
        "| $ka$ | $a/f$ | $\\mu$ ref | $\\mu$ panels | err | $\\nu$ ref | $\\nu$ panels | err | worst | verdict |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in grid:
        lines.append(
            f"| {row['ka']:.2f} | {row['a_over_f']:.2f} | {row['mu_reference']:.6f} "
            f"| {row['mu_panel']:.6f} | {row['added_mass_relative_error']:.3e} "
            f"| {row['nu_reference']:.6f} | {row['nu_panel']:.6f} "
            f"| {row['damping_relative_error']:.3e} "
            f"| {row['worst_relative_error']:.3e} | {row['verdict']} |"
        )
    lines += [
        "",
        "## Checks that protect the comparison",
        "",
        "| $ka$ | $a/f$ | reference shift 40→120 harmonics | MFS $\\mu$ err | MFS $\\nu$ err |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in grid:
        shift = max(
            row["reference_added_mass_shift_40_to_120"],
            row["reference_damping_shift_40_to_120"],
        )
        lines.append(
            f"| {row['ka']:.2f} | {row['a_over_f']:.2f} | {shift:.2e} "
            f"| {row['mfs_added_mass_relative_error']:.2e} "
            f"| {row['mfs_damping_relative_error']:.2e} |"
        )
    if payload.get("panels_needed"):
        lines += [
            "",
            "## Panels needed for the 3 % band (heave)",
            "",
            "| $ka$ | $a/f$ | smallest passing $N$ | worst error at $N=64$ | order ($\\mu$) | order ($\\nu$) |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for row in payload["panels_needed"]:
            order_mu = row["order_estimate_added_mass"]
            order_nu = row["order_estimate_damping"]
            lines.append(
                f"| {row['ka']:.2f} | {row['a_over_f']:.2f} "
                f"| {row['smallest_passing_panels']} "
                f"| {row['smallest_measured_error']:.2e} "
                f"| {'n/a' if order_mu is None else f'{order_mu:.2f}'} "
                f"| {'n/a' if order_nu is None else f'{order_nu:.2f}'} |"
            )
    lines += [
        "",
        "## Summary",
        "",
        f"- grid points: {summary['grid_points']}, verdicts: {summary['verdict_counts']}",
        f"- worst point: $ka={summary['worst_point']['ka']}$, "
        f"$a/f={summary['worst_point']['a_over_f']}$, "
        f"worst error {summary['worst_point']['added_mass_relative_error']:.3e} (mu) / "
        f"{summary['worst_point']['damping_relative_error']:.3e} (nu)",
        f"- largest reference shift 40→120 harmonics: "
        f"{summary['max_reference_shift_40_to_120']:.2e}",
        f"- largest MFS (panel-independent) error: {summary['max_mfs_error']:.2e}",
        "",
        "| $ka$ | max $\\mu$ error | max $\\nu$ error |",
        "|---:|---:|---:|",
    ]
    for ka, block in summary["by_ka"].items():
        lines.append(
            f"| {ka} | {block['max_added_mass_error']:.3e} "
            f"| {block['max_damping_error']:.3e} |"
        )
    lines += ["", "| $a/f$ | max $\\mu$ error | max $\\nu$ error |", "|---:|---:|---:|"]
    for ratio, block in summary["by_depth_ratio"].items():
        lines.append(
            f"| {ratio} | {block['max_added_mass_error']:.3e} "
            f"| {block['max_damping_error']:.3e} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="write the envelope as JSON here")
    parser.add_argument("--figure", type=Path, help="write the figure here")
    parser.add_argument("--markdown", type=Path, help="write the generated tables here")
    parser.add_argument(
        "--quick", action="store_true", help="two low-cost grid points only"
    )
    args = parser.parse_args(argv)

    if args.quick:
        payload = run((0.6, 1.0), (0.5,), with_refinement=False)
    else:
        payload = run()

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.figure:
        plot(payload, args.figure)
    markdown = render_markdown(payload)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown)
    print(markdown)
    return payload


if __name__ == "__main__":
    main()