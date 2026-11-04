"""Where the Kramers-Kronig audit's floor lives, and how far it can be pushed down.

The causality audit of ``tensorbem.causality`` is only a verification tool if its
own floor sits well below the discretisation error it is supposed to expose. The
m2913 audit reported exactly that it could not resolve the panel error: the
*analytical* Ursell reference already showed a 1.9e-1 residual in the damping
direction (equation (1), ``B`` predicted from the added-mass sweep) and a
9.7e-3 residual in the added-mass direction (equation (2), ``A`` predicted from
the damping sweep), while the panel solutions sat at 1.9e-2 - 3.5e-2. It located
the limit in "the infinite-frequency added mass and the two truncated bands"
without separating the three candidate mechanisms.

This module separates them, one control at a time, on sweeps that need no new
solver code:

``nodes``
    Sweep resolution of the piecewise-linear interpolant (32 / 64 / 128 nodes on
    the same band). Isolates interpolation error.
``bands``
    The lower end of the band (``ka_min`` from 5e-2 down to 1e-3) and the upper
    end (``ka_max`` 3 / 8 / 30). Isolates the two truncations.
``offset``
    The residual of equation (1) split into a *constant* part and a
    frequency-dependent part. A constant part is the fingerprint of an omitted
    high-frequency band, because for ``omega`` well below the last node the
    equation (1) kernel ``s^2/(s^2-omega^2)`` is 1 and any band left out acts as
    a pure additive offset in ``B``.
``tail``
    The high-frequency behaviour of the reference coefficients themselves: the
    damping decays exponentially (as the radiation condition at a free surface
    that becomes rigid demands) while the added mass approaches its limit only
    algebraically. The two cannot both belong to one causal kernel; the exponent
    is fitted here.
``cross``
    An independent panel solution of the *same* sweeps at ``ka = 8, 12``, where
    the free-surface kernel is smooth and damping is ~1e-8 or below, so the two
    answers can be compared without any causality argument.
``resolution``
    The payoff: with the lower band pushed to ``ka = 5e-3`` the reference floor
    in the added-mass direction drops by more than an order of magnitude, and the
    panel discretisation error becomes resolvable. Whether it *is* resolved is
    reported, not assumed.

Run ``PYTHONPATH=src python -m tensorbem.benchmarks.kkr_floor --write`` to
regenerate ``docs/m3051/kkr_floor.json``; ``--table`` prints the markdown tables.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tensorbem import causality as kk
from tensorbem.analytic import ursell_cylinder_added_mass
from tensorbem.benchmarks.ursell_cylinder import submerged_circle_mesh
from tensorbem.free_surface_solver import solve_radiation_indirect

RADIUS = 1.0
SUBMERGENCE = 2.0
DENSITY = 1025.0
GRAVITY = 9.80665

#: ``A_inf`` recorded by the m2913 audit (1/ka extrapolation over ka = 8..30).
ADDED_MASS_INF = 2845.7367316322266

REPO = Path(__file__).resolve().parents[3]
BASELINE_SWEEP = REPO / "docs" / "m2913" / "outputs" / "kkr_causality" / "sweep.json"
DEFAULT_OUTPUT = REPO / "docs" / "m3051" / "kkr_floor.json"


# --------------------------------------------------------------------------- #
# sweeps
# --------------------------------------------------------------------------- #
def analytic_sweep(ka_min: float, ka_max: float, points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ursell multipole reference on a log-spaced band; returns ``(ka, A, B)``."""
    ka = np.geomspace(ka_min, ka_max, points)
    omega = np.sqrt(ka * GRAVITY / RADIUS)
    mass = np.empty(points)
    damper = np.empty(points)
    for index, value in enumerate(omega):
        mass[index], damper[index] = ursell_cylinder_added_mass(
            RADIUS, SUBMERGENCE, float(value), DENSITY, gravity=GRAVITY
        )
    return ka, mass, damper


def panel_sweep(panels: int, ka_min: float, ka_max: float, points: int) -> tuple[np.ndarray, np.ndarray]:
    """Indirect panel solution of the same band; returns ``(A, B)``."""
    mesh = submerged_circle_mesh(panels, RADIUS, SUBMERGENCE)
    ka = np.geomspace(ka_min, ka_max, points)
    mass = np.empty(points)
    damper = np.empty(points)
    for index, wavenumber in enumerate(ka):
        result = solve_radiation_indirect(mesh, float(wavenumber), "heave", density=DENSITY)
        mass[index] = float(result.added_mass[1])
        damper[index] = float(result.damping[1])
    return mass, damper


def floors(ka: np.ndarray, mass: np.ndarray, damper: np.ndarray) -> dict[str, float]:
    """The two KK floor numbers of a sweep, plus the equation (1) offset."""
    omega = np.sqrt(np.asarray(ka) * GRAVITY / RADIUS)
    check = kk.kkr_check(omega, mass, damper, added_mass_inf=ADDED_MASS_INF)
    samples = kk._default_samples(np.asarray(omega, dtype=float))
    predicted = np.array(
        [
            kk.kkr_damping(omega, mass, float(sample), added_mass_inf=ADDED_MASS_INF)
            for sample in samples
        ]
    )
    measured = np.interp(samples, omega, damper)
    offset = predicted - measured
    return {
        "damping_floor": check.max_damping_relative_error,
        "added_mass_floor": check.max_added_mass_relative_error,
        "damping_offset_mean": float(np.mean(offset)),
        "damping_offset_spread": float(np.std(offset)),
        "damping_scale": float(check.damping_scale),
    }


# --------------------------------------------------------------------------- #
# controls
# --------------------------------------------------------------------------- #
def nodes_leg(
    ka_min: float = 0.05,
    ka_max: float = 3.0,
    points: tuple[int, ...] = (32, 64, 128, 256),
) -> list[dict[str, object]]:
    """Interpolation control: more nodes, same band."""
    rows = []
    for point_count in points:
        ka, mass, damper = analytic_sweep(ka_min, ka_max, point_count)
        row: dict[str, object] = {"points": point_count, "ka_min": ka_min, "ka_max": ka_max}
        row.update(floors(ka, mass, damper))
        rows.append(row)
    return rows


def band_leg(
    lower: tuple[float, ...] = (0.05, 0.01, 0.005, 0.001),
    upper: tuple[float, ...] = (3.0, 8.0, 30.0),
) -> list[dict[str, object]]:
    """Truncation control: move the two band ends, 64 nodes."""
    rows = []
    for ka_min in lower:
        ka, mass, damper = analytic_sweep(ka_min, 3.0, 64)
        row: dict[str, object] = {"ka_min": ka_min, "ka_max": 3.0, "points": 64}
        row.update(floors(ka, mass, damper))
        rows.append(row)
    for ka_max in upper:
        ka, mass, damper = analytic_sweep(0.05, ka_max, 64)
        row = {"ka_min": 0.05, "ka_max": ka_max, "points": 64}
        row.update(floors(ka, mass, damper))
        rows.append(row)
    return rows


def tail_leg() -> dict[str, object]:
    """High-frequency character of the reference pair.

    ``A - A_inf`` and ``B`` are sampled on a decade grid up to ``ka = 120`` (the
    multipole evaluator refuses ``2*k*f`` beyond double precision, which is why
    the grid stops there) and a power law is fitted on two windows.
    """
    ka = np.geomspace(3.0, 120.0, 25)
    omega = np.sqrt(ka * GRAVITY / RADIUS)
    deficit = np.empty(ka.size)
    damper = np.empty(ka.size)
    for index, value in enumerate(omega):
        mass, damping = ursell_cylinder_added_mass(
            RADIUS, SUBMERGENCE, float(value), DENSITY, gravity=GRAVITY
        )
        deficit[index] = mass - ADDED_MASS_INF
        damper[index] = damping

    def exponent(lo: float, hi: float) -> float:
        mask = (ka >= lo) & (ka <= hi)
        slope, _ = np.polyfit(np.log(ka[mask]), np.log(-deficit[mask]), 1)
        return float(-slope)

    def damping_rate(lo: float, hi: float) -> float:
        mask = (ka >= lo) & (ka <= hi) & (damper > 0.0)
        slope, _ = np.polyfit(ka[mask], np.log(damper[mask]), 1)
        return float(-slope)

    return {
        "ka": ka.tolist(),
        "added_mass_deficit": deficit.tolist(),
        "damping": damper.tolist(),
        "added_mass_tail_exponent_8_30": exponent(8.0, 30.0),
        "added_mass_tail_exponent_30_120": exponent(30.0, 120.0),
        "damping_decay_rate_per_ka_3_8": damping_rate(3.0, 8.0),
        "damping_decay_rate_per_ka_8_12": damping_rate(8.0, 12.0),
        "added_mass_at_ka_120": float(deficit[-1] + ADDED_MASS_INF),
    }


def cross_leg(panel_counts: tuple[int, ...] = (64, 96)) -> list[dict[str, object]]:
    """Panel solution against the multipole reference where damping is ~0."""
    rows = []
    for ka_value in (8.0, 12.0):
        omega = math.sqrt(ka_value * GRAVITY / RADIUS)
        ref_mass, ref_damper = ursell_cylinder_added_mass(
            RADIUS, SUBMERGENCE, omega, DENSITY, gravity=GRAVITY
        )
        for panels in panel_counts:
            mesh = submerged_circle_mesh(panels, RADIUS, SUBMERGENCE)
            result = solve_radiation_indirect(mesh, ka_value, "heave", density=DENSITY)
            rows.append(
                {
                    "ka": ka_value,
                    "panels": panels,
                    "added_mass_panel": float(result.added_mass[1]),
                    "added_mass_reference": float(ref_mass),
                    "added_mass_offset": float(result.added_mass[1] - ref_mass),
                    "added_mass_relative_offset": float(
                        (result.added_mass[1] - ref_mass) / ref_mass
                    ),
                    "damping_panel": float(result.damping[1]),
                    "damping_reference": float(ref_damper),
                }
            )
    return rows


def resolution_leg(ka_min: float = 0.005, points: int = 24) -> dict[str, object]:
    """Reference floor and panel floors on the widened band, same grid."""
    ka, ref_mass, ref_damper = analytic_sweep(ka_min, 3.0, points)
    reference = {"case": "analytic_multipole"}
    reference.update(floors(ka, ref_mass, ref_damper))
    rows = [reference]
    for panels in (32, 64):
        mass, damper = panel_sweep(panels, ka_min, 3.0, points)
        row: dict[str, object] = {
            "case": f"panels_{panels}",
            "panels": panels,
            "added_mass_error_max": float(np.max(np.abs(mass - ref_mass)) / np.max(np.abs(ref_mass))),
            "damping_error_max": float(
                np.max(np.abs(damper - ref_damper)) / np.max(np.abs(ref_damper))
            ),
        }
        row.update(floors(ka, mass, damper))
        rows.append(row)
    return {"ka_min": ka_min, "ka_max": 3.0, "points": points, "cases": rows}


def baseline_leg() -> dict[str, object] | None:
    """The m2913 audit numbers, read from its recorded sweep (no re-solve)."""
    if not BASELINE_SWEEP.exists():  # pragma: no cover - artifact is in the repo
        return None
    record = json.loads(BASELINE_SWEEP.read_text(encoding="utf-8"))
    rows = []
    for name, case in record["cases"].items():
        kkr = case["kkr_with_limit_added_mass"]
        rows.append(
            {
                "case": name,
                "panels": case.get("panels"),
                "damping_floor": kkr["max_damping_relative_error"],
                "added_mass_floor": kkr["max_added_mass_relative_error"],
            }
        )
    return {
        "source": str(BASELINE_SWEEP.relative_to(REPO)),
        "ka_min": record["ka_min"],
        "ka_max": record["ka_max"],
        "points": record["points"],
        "cases": rows,
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def build(output: Path | None = None, *, include_panels: bool = True) -> dict[str, object]:
    record: dict[str, object] = {
        "benchmark": "kramers_kronig_floor_decomposition",
        "case": "ursell_submerged_circular_cylinder_deep_water_heave",
        "radius": RADIUS,
        "submergence": SUBMERGENCE,
        "density": DENSITY,
        "gravity": GRAVITY,
        "added_mass_inf": ADDED_MASS_INF,
        "baseline": baseline_leg(),
        "nodes": nodes_leg(),
        "bands": band_leg(),
        "tail": tail_leg(),
    }
    if include_panels:
        record["cross"] = cross_leg()
        record["resolution"] = resolution_leg()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}e}"


def tables(record: dict[str, object]) -> str:
    """Markdown tables rendered straight from the record (no hand-typed number)."""
    lines: list[str] = []
    baseline = record.get("baseline")
    if baseline:
        lines.append("### Baseline (m2913 sweep record)\n")
        lines.append("| case | damping floor | added-mass floor |")
        lines.append("|---|---:|---:|")
        for row in baseline["cases"]:
            lines.append(
                f"| {row['case']} | {_fmt(row['damping_floor'])} | {_fmt(row['added_mass_floor'])} |"
            )
        lines.append("")

    lines.append("### Control 1 - interpolation nodes (band 0.05..3, reference)\n")
    lines.append("| nodes | damping floor | added-mass floor | equation (1) offset |")
    lines.append("|---:|---:|---:|---:|")
    for row in record["nodes"]:
        lines.append(
            f"| {row['points']} | {_fmt(row['damping_floor'])} | {_fmt(row['added_mass_floor'])} "
            f"| {row['damping_offset_mean']:+.1f} +- {row['damping_offset_spread']:.1f} |"
        )
    lines.append("")

    lines.append("### Control 2 - band ends (64 nodes, reference)\n")
    lines.append("| ka_min | ka_max | damping floor | added-mass floor | equation (1) offset |")
    lines.append("|---:|---:|---:|---:|---:|")
    for row in record["bands"]:
        lines.append(
            f"| {row['ka_min']:g} | {row['ka_max']:g} | {_fmt(row['damping_floor'])} "
            f"| {_fmt(row['added_mass_floor'])} | {row['damping_offset_mean']:+.1f} +- "
            f"{row['damping_offset_spread']:.1f} |"
        )
    lines.append("")

    tail = record["tail"]
    lines.append("### Control 3 - high-frequency character of the reference pair\n")
    lines.append("| quantity | value |")
    lines.append("|---|---:|")
    lines.append(f"| added-mass deficit exponent on ka 8..30 | {tail['added_mass_tail_exponent_8_30']:.3f} |")
    lines.append(f"| added-mass deficit exponent on ka 30..120 | {tail['added_mass_tail_exponent_30_120']:.3f} |")
    lines.append(f"| damping decay rate per ka on 3..8 | {tail['damping_decay_rate_per_ka_3_8']:.3f} |")
    lines.append(f"| damping decay rate per ka on 8..12 | {tail['damping_decay_rate_per_ka_8_12']:.3f} |")
    lines.append(f"| A at ka = 120 | {tail['added_mass_at_ka_120']:.3f} |")
    lines.append("")

    if "cross" in record:
        lines.append("### Control 4 - panel solution vs reference at high ka\n")
        lines.append("| ka | panels | A (panel) | A (reference) | offset | relative | B (panel) | B (reference) |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in record["cross"]:
            lines.append(
                f"| {row['ka']:g} | {row['panels']} | {row['added_mass_panel']:.2f} "
                f"| {row['added_mass_reference']:.2f} | {row['added_mass_offset']:+.2f} "
                f"| {_fmt(row['added_mass_relative_offset'])} | {_fmt(row['damping_panel'])} "
                f"| {_fmt(row['damping_reference'])} |"
            )
        lines.append("")

    resolution = record.get("resolution")
    if resolution:
        reference_floor = resolution["cases"][0]["added_mass_floor"]
        lines.append(
            f"### Control 5 - widened band, ka in [{resolution['ka_min']:g}, "
            f"{resolution['ka_max']:g}], {resolution['points']} nodes\n"
        )
        lines.append(
            "| case | A error vs reference | B error vs reference | added-mass floor "
            "| damping floor | floor ratio to reference |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in resolution["cases"]:
            a_error = row.get("added_mass_error_max")
            b_error = row.get("damping_error_max")
            lines.append(
                f"| {row['case']} | {'-' if a_error is None else _fmt(a_error)} "
                f"| {'-' if b_error is None else _fmt(b_error)} | {_fmt(row['added_mass_floor'])} "
                f"| {_fmt(row['damping_floor'])} "
                f"| {row['added_mass_floor'] / reference_floor:.1f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-panels", action="store_true")
    parser.add_argument("--table", action="store_true")
    args = parser.parse_args()
    record = build(args.output, include_panels=not args.no_panels)
    if args.table:
        print(tables(record))
    else:
        print(json.dumps({k: v for k, v in record.items() if k in ("nodes", "bands", "tail")}, indent=2))
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())