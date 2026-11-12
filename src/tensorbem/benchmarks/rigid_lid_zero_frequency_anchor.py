"""Zero-frequency (rigid-lid) anchor for the 3-D free-surface radiation solver.

Why this module exists
----------------------
The 3-D free-surface stack already carries two *exact* heave limits for the
floating hemisphere (``radiation_diffraction_anchors_3d``):

* ``ka -> inf`` pressure release -- ``A33 = rho pi a^3 / 3`` (anchor A2, exact
  by the odd-symmetry of the heave potential about the free surface);
* an unbounded-fluid ladder -- ``A33 = 2/3 rho pi a^3`` (anchor A1).

The opposite end of the band, ``ka -> 0``, is only covered *qualitatively*:
anchor C6 in that module asserts that ``A33`` stops changing and that
``B33/(omega A33)`` vanishes for ``omega <= 0.2`` on a barge.  Flatness is not a
magnitude statement.  The rigid-lid limit itself is exact and available:

    omega -> 0  =>  the free-surface condition ``omega^2 phi = g dphi/dz``
    degenerates to ``dphi/dz = 0`` on ``z = 0``, a symmetry plane.  A floating
    hemisphere heaving against that plane is, by the method of images, exactly
    a *full sphere* translating in unbounded fluid, whose added mass is the
    classical ``2/3 rho pi a^3``.  Since the hemisphere displaces
    ``grad = 2/3 pi a^3``, the statement reads

        A33(ka -> 0) / (rho * grad) = 1        (exact),

    to be read together with ``A33(ka -> inf) / (rho * grad) = 1/2`` (exact).
    The two limits bracket a factor of two; a solver that reproduces both, and
    an added mass that moves monotonically between them, is pinned at *both*
    ends of the frequency band with no literature constant involved.

Nothing here is transcribed: every number in the generated summary is read back
from this module's own JSON artifact.

Run::

    python -m tensorbem.benchmarks.rigid_lid_zero_frequency_anchor \
        --output docs/rigid_lid_zero_frequency_anchor/results.json \
        --summary docs/rigid_lid_zero_frequency_anchor/summary.md \
        --figure docs/rigid_lid_zero_frequency_anchor/fig_rigid_lid_limit.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorbem.free_surface_solver3d import solve_radiation_3d
from tensorbem.geometry3d import hemisphere_hull_mesh

DENSITY = 1025.0
GRAVITY = 9.80665
RADIUS = 1.0

#: Wave-number times radius at which the sweep is evaluated.  ``ka <= 0.4`` is
#: the rigid-lid end, ``ka >= 3`` the pressure-release end.
DEFAULT_KAS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.6, 1.0, 2.0, 3.0, 6.0)


def exact_rigid_lid_limit(density: float = DENSITY, radius: float = RADIUS) -> float:
    """``A33(ka -> 0) = 2/3 rho pi a^3`` (hemisphere == imaged sphere)."""
    return 2.0 / 3.0 * density * math.pi * radius**3


def exact_pressure_release_limit(
    density: float = DENSITY, radius: float = RADIUS
) -> float:
    """``A33(ka -> inf) = rho pi a^3 / 3`` (half the imaged sphere)."""
    return density * math.pi * radius**3 / 3.0


def swept_added_mass(
    kas: Sequence[float] = DEFAULT_KAS,
    subdivisions: int = 3,
    radius: float = RADIUS,
    density: float = DENSITY,
    gravity: float = GRAVITY,
) -> dict[str, Any]:
    """Heave added mass / damping of a floating hemisphere over ``ka``."""
    hull = hemisphere_hull_mesh(int(subdivisions), radius=radius)
    rigid = exact_rigid_lid_limit(density, radius)
    release = exact_pressure_release_limit(density, radius)
    displaced = density * 2.0 / 3.0 * math.pi * radius**3
    rows: list[dict[str, Any]] = []
    for ka in kas:
        res = solve_radiation_3d(
            hull, float(ka), modes=(2,), density=density, gravity=gravity
        )
        a33 = float(res.added_mass[0, 0])
        b33 = float(res.damping[0, 0])
        omega = math.sqrt(gravity * float(ka) / radius)
        rows.append(
            {
                "ka": float(ka),
                "omega": omega,
                "a33": a33,
                "b33": b33,
                "a33_over_rho_displaced": a33 / displaced,
                "a33_over_rigid_lid": a33 / rigid,
                "b33_over_omega_mass": b33 / (omega * displaced),
                "condition_number": float(res.condition_number),
                "relative_residual": float(res.relative_residual),
            }
        )
    return {
        "subdivisions": int(subdivisions),
        "panels": int(hull.panel_count),
        "radius": radius,
        "density": density,
        "gravity": gravity,
        "rho_displaced": displaced,
        "analytic_rigid_lid": rigid,
        "analytic_pressure_release": release,
        "rows": rows,
    }


def _drift(rows: Sequence[dict[str, Any]], kas: Sequence[float]) -> dict[str, Any]:
    """Relative change of ``A33`` between consecutive ``ka`` at the low end."""
    pairs = []
    for ka in kas:
        match = [row for row in rows if abs(row["ka"] - ka) < 1e-12]
        if match:
            pairs.append(match[0])
    pairs.sort(key=lambda row: row["ka"])
    drifts = []
    for lo, hi in zip(pairs, pairs[1:]):
        drifts.append(
            {
                "ka_from": lo["ka"],
                "ka_to": hi["ka"],
                "relative_change": (hi["a33"] - lo["a33"]) / lo["a33"],
            }
        )
    return {"consecutive_relative_change": drifts}


def run_study(
    subdivisions: Sequence[int] = (2, 3, 4),
    kas: Sequence[float] = DEFAULT_KAS,
    radius: float = RADIUS,
) -> dict[str, Any]:
    """Mesh ladder: the low-``ka`` value must be mesh independent."""
    sweep = [
        swept_added_mass(kas=kas, subdivisions=int(n), radius=radius)
        for n in subdivisions
    ]
    finest = max(sweep, key=lambda item: item["panels"])
    first = finest["rows"][0]
    last = finest["rows"][-1]
    rigid = finest["analytic_rigid_lid"]
    release = finest["analytic_pressure_release"]

    #: lowest-ka mesh sensitivity: finest vs coarsest at the smallest ka
    low_ka_mesh_change = (
        abs(finest["rows"][0]["a33"] - sweep[0]["rows"][0]["a33"])
        / abs(sweep[0]["rows"][0]["a33"])
    )


    criteria = [
        {
            "name": "R1-rigid-lid-limit",
            "statement": "A33(ka -> 0) -> 2/3 rho pi a^3 (exact imaged sphere)",
            "measured": first["a33"],
            "reference": rigid,
            "relative_error": abs(first["a33"] - rigid) / rigid,
            "pass": abs(first["a33"] - rigid) / rigid < 0.05,
        },
        {
            "name": "R2-pressure-release-limit",
            "statement": "A33(ka -> inf) -> rho pi a^3 / 3 (exact, anchor A2)",
            "measured": last["a33"],
            "reference": release,
            "relative_error": abs(last["a33"] - release) / release,
            "pass": abs(last["a33"] - release) / release < 0.05,
        },
        {
            "name": "R3-infinite-frequency-floor",
            "statement": "A33(ka) never falls below the exact ka -> inf limit (0.5 rho grad)",
            "measured": min(row["a33_over_rho_displaced"] for row in finest["rows"]),
            "reference": release / rigid,
            "relative_error": (
                (release / rigid - min(row["a33_over_rho_displaced"] for row in finest["rows"]))
                / (release / rigid)
            ),
            "pass": min(row["a33_over_rho_displaced"] for row in finest["rows"])
            >= release / rigid,
        },
        {
            "name": "R4-low-band-monotone",
            "statement": "A33 decreases monotonically over ka <= 1 (rigid-lid approach)",
            "measured": [
                row["a33_over_rho_displaced"] for row in finest["rows"] if row["ka"] <= 1.0
            ],
            "reference": "strictly decreasing",
            "relative_error": None,
            "pass": all(
                hi["a33"] <= lo["a33"]
                for lo, hi in zip(
                    [row for row in finest["rows"] if row["ka"] <= 1.0],
                    [row for row in finest["rows"] if row["ka"] <= 1.0][1:],
                )
            ),
        },
        {
            "name": "R5-low-ka-mesh-independence",
            "statement": "lowest-ka A33 is mesh independent (< 2 %)",
            "measured": low_ka_mesh_change,
            "reference": 0.02,
            "relative_error": low_ka_mesh_change,
            "pass": low_ka_mesh_change < 0.02,
        },
    ]

    return {
        "case": {
            "geometry": "half-submerged hemisphere, radius a = %.3f m" % radius,
            "density": finest["density"],
            "gravity": finest["gravity"],
            "displaced_mass": finest["rho_displaced"],
            "ka_range": [min(kas), max(kas)],
            "exact_limits": {
                "rigid_lid_ka0": rigid,
                "pressure_release_kainf": release,
            },
        },
        "sweeps": sweep,
        "drift": _drift(finest["rows"], kas),
        "criteria": criteria,
        "all_pass": all(item["pass"] for item in criteria),
    }


def write_summary(payload: dict[str, Any], path: Path) -> Path:
    """Emit the markdown tables straight from the artifact (no hand copying)."""
    finest = max(payload["sweeps"], key=lambda item: item["panels"])
    rigid = payload["case"]["exact_limits"]["rigid_lid_ka0"]
    release = payload["case"]["exact_limits"]["pressure_release_kainf"]
    lines = [
        "# Zero-frequency (rigid-lid) anchor for the 3-D free-surface radiation solver",
        "",
        "Generated by `python -m tensorbem.benchmarks.rigid_lid_zero_frequency_anchor`",
        "from the same payload that produced `results.json`; no number is transcribed.",
        "",
        "Floating hemisphere, radius a = %.3f m, rho = %.1f kg/m^3, rho*grad = %.4f kg."
        % (
            payload["sweeps"][0]["radius"],
            payload["sweeps"][0]["density"],
            payload["case"]["displaced_mass"],
        ),
        "",
        "Exact limits: `A33(ka->0) = 2/3 rho pi a^3 = %.4f kg` (rigid lid = imaged"
        " sphere), `A33(ka->inf) = rho pi a^3/3 = %.4f kg`."
        % (rigid, release),
        "",
        "## Sweep on the finest mesh (subdivisions=%d, %d panels)"
        % (finest["subdivisions"], finest["panels"]),
        "",
        "| ka | omega [rad/s] | A33 [kg] | A33/(rho grad) | A33/A33(0) | B33/(omega rho grad) |",
        "|---|---|---|---|---|---|",
    ]
    for row in finest["rows"]:
        lines.append(
            "| %.2f | %.4f | %.4f | %.4f | %.4f | %.4f |"
            % (
                row["ka"],
                row["omega"],
                row["a33"],
                row["a33_over_rho_displaced"],
                row["a33_over_rigid_lid"],
                row["b33_over_omega_mass"],
            )
        )
    lines += [
        "",
        "## Mesh ladder at the two band ends",
        "",
        "| subdivisions | panels | A33 at lowest ka | A33 at highest ka |",
        "|---|---|---|---|",
    ]
    for sweep in payload["sweeps"]:
        lines.append(
            "| %d | %d | %.4f | %.4f |"
            % (
                sweep["subdivisions"],
                sweep["panels"],
                sweep["rows"][0]["a33"],
                sweep["rows"][-1]["a33"],
            )
        )
    lines += ["", "## Criteria", "", "| id | statement | measured | reference | pass |", "|---|---|---|---|---|"]
    for item in payload["criteria"]:
        measured = item["measured"]
        if isinstance(measured, float):
            measured = "%.6g" % measured
        elif isinstance(measured, list):
            measured = ", ".join(
                "%.4f" % value if isinstance(value, float) else str(value)
                for value in measured
            )
        reference = item["reference"]
        if isinstance(reference, float):
            reference = "%.6g" % reference
        elif isinstance(reference, list):
            reference = ", ".join("%.4f" % value for value in reference)
        lines.append(
            "| %s | %s | %s | %s | %s |"
            % (item["name"], item["statement"], measured, reference, "PASS" if item["pass"] else "FAIL")
        )
    lines += [
        "",
        "**all_pass = %s**" % payload["all_pass"],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_figure(payload: dict[str, Any], path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finest = max(payload["sweeps"], key=lambda item: item["panels"])
    rigid = payload["case"]["exact_limits"]["rigid_lid_ka0"]
    release = payload["case"]["exact_limits"]["pressure_release_kainf"]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))

    for sweep in payload["sweeps"]:
        kas = [row["ka"] for row in sweep["rows"]]
        a33 = [row["a33"] / rigid for row in sweep["rows"]]
        axes[0].plot(kas, a33, marker="o", ms=3, lw=1.2, label="subdiv %d (%d panels)" % (sweep["subdivisions"], sweep["panels"]))
    axes[0].axhline(1.0, color="k", ls="--", lw=1.0, label=r"exact $ka\to 0$: $2/3\rho\pi a^3$")
    axes[0].axhline(release / rigid, color="k", ls=":", lw=1.0, label=r"exact $ka\to\infty$: $\rho\pi a^3/3$")
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$ka$")
    axes[0].set_ylabel(r"$A_{33}/(\rho\nabla)$")
    axes[0].set_title("heave added mass vs the two exact limits")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    kas = [row["ka"] for row in finest["rows"]]
    a33 = [row["a33"] for row in finest["rows"]]
    b33 = [row["b33"] for row in finest["rows"]]
    axes[1].loglog(kas, a33, marker="o", ms=3, lw=1.2, label=r"$A_{33}$ [kg]")
    axes[1].loglog(kas, [max(value, 1e-12) for value in b33], marker="s", ms=3, lw=1.2, label=r"$B_{33}$ [kg/s]")
    axes[1].set_xlabel(r"$ka$")
    axes[1].set_ylabel("coefficient [kg, kg/s]")
    axes[1].set_title("finest mesh (%d panels): A33 plateau and B33 roll-off" % finest["panels"])
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/rigid_lid_zero_frequency_anchor/results.json"))
    parser.add_argument("--summary", type=Path, default=Path("docs/rigid_lid_zero_frequency_anchor/summary.md"))
    parser.add_argument("--figure", type=Path, default=Path("docs/rigid_lid_zero_frequency_anchor/fig_rigid_lid_limit.png"))
    parser.add_argument("--subdivisions", type=int, nargs="+", default=[2, 3, 4])
    args = parser.parse_args(argv)

    payload = run_study(subdivisions=tuple(args.subdivisions))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_summary(payload, args.summary)
    write_figure(payload, args.figure)
    for item in payload["criteria"]:
        print("%-32s %s" % (item["name"], "PASS" if item["pass"] else "FAIL"))
    print("all_pass =", payload["all_pass"])
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())