"""Irregular-frequency *band* study for the 2-D exterior Helmholtz solver.

Why this exists
---------------
:mod:`tensorbem.helmholtz` documents that on a closed cross-section the direct
BIE ``(1/2 I - H_k) varphi = varphi_I`` loses rank whenever ``k`` is an interior
Dirichlet eigenvalue of the cross-section.  Until now the project only
*checked* that statement at two hand-picked frequencies
(:func:`tensorbem.benchmarks.maccamy_fuchs.run_irregular_frequency_check`), at
one panel count, and only for the axisymmetric ``J_0`` and dipole ``J_1``
resonances.  Three questions a user actually asks were therefore unanswered:

1. **Where is the resonance of the *discrete* operator?**  The analytic zero
   ``j_{m,1}`` is the resonance of the continuous problem; a panel mesh has its
   own (slightly shifted) singularity.  Evaluating at the analytic zero can
   therefore *miss* the peak of the pathology.
2. **How wide is the danger band?**  The published accuracy (``1.3e-4`` at 128
   panels, ``ka = 1.0``) is worthless if the error is still several percent a
   fraction of a percent away from a resonance.
3. **Is the CHIEF failure mode documented in the solver docstring real?**  The
   docstring says suppression works "unless every CHIEF point happens to lie on
   a nodal curve of the resonant interior mode".  That claim had no test.

What is measured
----------------
For each interior Dirichlet resonance ``j_{m,n}`` (zeros of ``J_m``) on a
regular-polygon approximation of a circle, against the closed-form MacCamy &
Fuchs (1954) Bessel/Hankel solution:

* the **discrete resonance** ``ka`` -- the argument that maximises the condition
  number of the unaugmented system -- and its detuning from the analytic zero,
  as a function of panel count;
* the **boundary-potential error** (relative ``L2`` norm of the total potential
  on the wetted surface) and the horizontal-force error, at the analytic zero
  and at the discrete resonance, for three CHIEF variants (none, automatic,
  nodal);
* the **amplification band**: the ``ka`` window in which the unaugmented
  potential error exceeds 1 % and 3 % (3 % is the platform's acceptance
  threshold, so this is the window where the solver is *not acceptable*);
* a **negative control**: CHIEF points deliberately placed on the nodal line of
  the resonant interior mode, which must leave the cure inert.

Run::

    python -m tensorbem.benchmarks.irregular_frequency_study --write

which writes ``artifacts/irregular_frequency_study.json``; the test
``tests/test_irregular_frequency_study.py`` re-derives its assertions from the
module (not from the artifact) so the study cannot drift.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from tensorbem.analytic import (
    BESSEL_J0_FIRST_ZERO,
    BESSEL_J1_FIRST_ZERO,
    bessel_j01_y01,
    maccamy_fuchs_diffraction,
    maccamy_fuchs_surface_potential,
)
from tensorbem.geometry import PanelMesh, regular_polygon
from tensorbem.helmholtz import solve_helmholtz_diffraction, solve_vertical_cylinder_diffraction

FloatArray = NDArray[np.float64]

#: First zero of ``J_2``; the quadrupole interior resonance.  Reproduced by
#: :func:`dirichlet_zeros` in the test against this literal
#: (Abramowitz & Stegun table 9.5: ``j_{2,1} = 5.135622302``).
BESSEL_J2_FIRST_ZERO = 5.135622301840683

#: Published resonances used as the study centres: name -> (angular order, ka).
RESONANCES: dict[str, tuple[int, float]] = {
    "j01_axisymmetric": (0, BESSEL_J0_FIRST_ZERO),
    "j11_dipole": (1, BESSEL_J1_FIRST_ZERO),
    "j21_quadrupole": (2, BESSEL_J2_FIRST_ZERO),
}

_DEFAULT_RADIUS = 1.0
_DEFAULT_DEPTH = 3.0
_DEFAULT_DENSITY = 1025.0
_DEFAULT_GRAVITY = 9.80665


# ---------------------------------------------------------------------------
# Bessel zeros (no scipy: the package declares numpy only)
# ---------------------------------------------------------------------------


def bessel_j(order: int, x: FloatArray | float) -> FloatArray:
    """Return ``J_order(x)`` for integer ``order >= 0`` by upward recurrence.

    The recurrence starts from the accurate ``J_0``/``J_1`` pair of
    :func:`tensorbem.analytic.bessel_j01_y01`.  Upward recurrence is stable for
    ``order < x``, which holds for every root used here (``x >= 1`` for the
    scan grid and ``order <= 3`` for x above ~2.4; the first bracket where
    ``J_3`` is needed already has ``x > 4``).
    """
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)):
        raise ValueError("order must be a non-negative integer")
    if int(order) < 0:
        raise ValueError("order must be a non-negative integer")
    values = np.asarray(x, dtype=float)
    if np.any(values <= 0.0):
        raise ValueError("x must be positive")
    j0, j1, _, _ = bessel_j01_y01(values)
    if int(order) == 0:
        return j0
    if int(order) == 1:
        return j1
    jm2, jm1 = j0, j1
    for n in range(1, int(order)):
        jm2, jm1 = jm1, (2.0 * n / values) * jm1 - jm2
    return jm1


def dirichlet_zeros(order: int, count: int = 2, *, upper: float = 40.0) -> list[float]:
    """Return the first ``count`` positive zeros of ``J_order``.

    Bracketed on a fine grid and closed with 200 bisection steps, so the zeros
    are good to machine precision without any third-party special function
    library.

    The scan starts at ``order + 1`` for ``order >= 1`` because
    :func:`bessel_j` builds ``J_order`` by upward recurrence from ``J_0, J_1``,
    which is only stable for ``x > order``.  Scanning from ``1e-3`` instead made
    the recurrence overflow at small ``x`` and produced **spurious** sign changes
    near ``x = 0.0027``: ``dirichlet_zeros(4, 2)`` used to return
    ``[0.00274, 0.00490]`` (m9703/m17244 case audit).  The shift is lossless
    because ``j_{m,1} > m + 1`` for every ``m >= 1``.
    """
    if int(count) < 1:
        raise ValueError("count must be >= 1")
    lower = 1.0e-3 if int(order) == 0 else float(order) + 1.0
    grid = np.linspace(lower, float(upper), 40001)
    values = bessel_j(int(order), grid)
    zeros: list[float] = []
    for i in range(len(grid) - 1):
        a, b = float(values[i]), float(values[i + 1])
        if a == 0.0:
            zeros.append(float(grid[i]))
            continue
        if a * b > 0.0:
            continue
        lo, hi = float(grid[i]), float(grid[i + 1])
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if bessel_j(int(order), lo) * bessel_j(int(order), mid) <= 0.0:
                hi = mid
            else:
                lo = mid
        zeros.append(0.5 * (lo + hi))
        if len(zeros) >= int(count):
            break
    if len(zeros) < int(count):
        raise RuntimeError(f"only {len(zeros)} zeros of J_{order} found below {upper}")
    return zeros[: int(count)]


# ---------------------------------------------------------------------------
# Solver probes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Probe:
    """One ``(ka, CHIEF variant)`` evaluation against the exact solution."""

    ka: float
    variant: str
    condition_number: float
    potential_relative_error: float
    force_relative_error: float
    moment_relative_error: float
    chief_point_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "ka": self.ka,
            "variant": self.variant,
            "condition_number": self.condition_number,
            "potential_relative_error": self.potential_relative_error,
            "force_relative_error": self.force_relative_error,
            "moment_relative_error": self.moment_relative_error,
            "chief_point_count": self.chief_point_count,
        }


def nodal_chief_points(resonance: str, radius: float = _DEFAULT_RADIUS) -> FloatArray:
    """CHIEF points lying on the nodal line(s) of the resonant interior mode.

    The interior Dirichlet eigenfunction at ``j_{m,1}`` is
    ``J_m(k r) cos(m theta)`` (a standing wave), so its nodal lines inside the
    circle are the rays ``theta = (2p + 1) pi / (2 m)``.  Every point below sits
    on one of those rays, which is exactly the configuration the solver
    docstring says CHIEF cannot rescue.
    """
    if resonance not in RESONANCES:
        raise ValueError(f"resonance must be one of {sorted(RESONANCES)}")
    order, _ = RESONANCES[resonance]
    if order == 0:
        raise ValueError(
            "the axisymmetric j01 mode has no interior nodal line: the nodal-risk "
            "negative control is only defined for order >= 1"
        )
    radii = np.array([0.25, 0.40, 0.55, 0.70], dtype=float) * float(radius)
    angles = [
        (2 * p + 1) * math.pi / (2 * order) for p in range(2 * int(order))
    ]
    points = [
        [r * math.cos(angle), r * math.sin(angle)] for r in radii for angle in angles
    ]
    return np.array(points, dtype=float)


def probe(
    mesh: PanelMesh,
    ka: float,
    variant: str,
    *,
    resonance: str | None = None,
    radius: float = _DEFAULT_RADIUS,
    depth: float = _DEFAULT_DEPTH,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
) -> Probe:
    """Solve one case and compare the wave loads and surface potential.

    ``variant`` is ``"unaugmented"`` (``chief_points=None``), ``"auto"``
    (:func:`tensorbem.helmholtz.default_chief_points`, 5 interior points), or
    ``"nodal"`` (CHIEF points on the resonant mode's nodal line).
    """
    if variant not in ("unaugmented", "auto", "nodal"):
        raise ValueError("variant must be 'unaugmented', 'auto' or 'nodal'")
    if variant == "unaugmented":
        chief: Any = None
    elif variant == "auto":
        chief = "auto"
    else:
        if resonance is None:
            raise ValueError("variant='nodal' requires a resonance name")
        chief = nodal_chief_points(resonance, radius=radius)

    exact = maccamy_fuchs_diffraction(
        ka, radius=radius, depth=depth, density=density, gravity=gravity
    )
    exact_fx = complex(exact["force_x"])
    exact_my = complex(exact["moment_about_seabed"])
    k = float(ka) / radius
    solved = solve_vertical_cylinder_diffraction(
        mesh,
        k,
        depth=depth,
        density=density,
        gravity=gravity,
        chief_points=chief,
    )
    fx = complex(solved.force[0])
    my = complex(solved.moment_about_seabed[0])

    angles = np.arctan2(mesh.centers[:, 1], mesh.centers[:, 0])
    exact_phi = maccamy_fuchs_surface_potential(float(ka), angles)
    phi = solved.helmholtz.total_potential
    potential_error = float(
        np.linalg.norm(phi - exact_phi) / np.linalg.norm(exact_phi)
    )
    return Probe(
        ka=float(ka),
        variant=variant,
        condition_number=float(solved.helmholtz.condition_number),
        potential_relative_error=potential_error,
        force_relative_error=float(abs(fx - exact_fx) / abs(exact_fx)),
        moment_relative_error=float(abs(my - exact_my) / abs(exact_my)),
        chief_point_count=int(solved.helmholtz.chief_points.shape[0]),
    )


def _bare_condition_number(
    mesh: PanelMesh, ka: float, *, radius: float
) -> float:
    """Condition number of the unaugmented system only (cheap inner-loop probe)."""
    return float(
        solve_helmholtz_diffraction(mesh, float(ka) / radius, chief_points=None)
        .condition_number
    )


# ---------------------------------------------------------------------------
# Resonance location and band width
# ---------------------------------------------------------------------------


def locate_resonance(
    mesh: PanelMesh,
    ka_lo: float,
    ka_hi: float,
    *,
    radius: float = _DEFAULT_RADIUS,
    coarse_samples: int = 9,
    refine_iterations: int = 13,
) -> dict[str, float]:
    """Locate the ``ka`` that maximises the unaugmented condition number.

    A coarse scan picks the bracket around the largest ``kappa``, then a golden
    section search on ``kappa(ka)`` resolves the peak.  The condition number of
    a nearly singular matrix is sharply spiked (observed ratio between the peak
    and its value one coarse step away: 30-80x), so a coarse grid alone would
    under-report the peak by more than an order of magnitude.
    """
    if not (0.0 < ka_lo < ka_hi):
        raise ValueError("require 0 < ka_lo < ka_hi")
    grid = np.linspace(ka_lo, ka_hi, int(coarse_samples))
    kappas = [_bare_condition_number(mesh, float(ka), radius=radius) for ka in grid]
    index = int(np.argmax(kappas))
    lo = float(grid[max(index - 1, 0)])
    hi = float(grid[min(index + 1, len(grid) - 1)])
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - golden * (hi - lo)
    d = lo + golden * (hi - lo)
    fc = _bare_condition_number(mesh, c, radius=radius)
    fd = _bare_condition_number(mesh, d, radius=radius)
    for _ in range(int(refine_iterations)):
        if fc > fd:
            hi, d, fd = d, c, fc
            c = hi - golden * (hi - lo)
            fc = _bare_condition_number(mesh, c, radius=radius)
        else:
            lo, c, fc = c, d, fd
            d = lo + golden * (hi - lo)
            fd = _bare_condition_number(mesh, d, radius=radius)
    candidates = (
        (0.5 * (lo + hi), None),
        (float(grid[index]), None),
    )
    best_ka = candidates[0][0]
    best_kappa = _bare_condition_number(mesh, best_ka, radius=radius)
    coarse_ka = candidates[1][0]
    coarse_kappa = float(kappas[index])
    if coarse_kappa > best_kappa:
        best_ka, best_kappa = coarse_ka, coarse_kappa
    return {
        "ka": float(best_ka),
        "condition_number": float(best_kappa),
        "bracket_lo": float(lo),
        "bracket_hi": float(hi),
        "coarse_max_ka": float(coarse_ka),
        "coarse_max_condition_number": coarse_kappa,
    }


def amplification_band(
    mesh: PanelMesh,
    centre: float,
    *,
    half_width: float = 0.30,
    samples: int = 25,
    thresholds: Sequence[float] = (0.01, 0.03),
    radius: float = _DEFAULT_RADIUS,
    depth: float = _DEFAULT_DEPTH,
) -> list[dict[str, Any]]:
    """Width (in ``ka``) of the window where the unaugmented error exceeds a threshold."""
    if not (0.0 < centre - half_width and centre + half_width < 8.0):
        raise ValueError("band must stay inside 0 < ka < 8")
    grid = np.linspace(centre - half_width, centre + half_width, int(samples))
    probes = [
        probe(mesh, float(ka), "unaugmented", radius=radius, depth=depth) for ka in grid
    ]
    errors = np.array([p.potential_relative_error for p in probes])
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        above = np.flatnonzero(errors > float(threshold))
        if above.size == 0:
            rows.append(
                {
                    "threshold": float(threshold),
                    "exceeded": False,
                    "ka_lo": None,
                    "ka_hi": None,
                    "full_width": 0.0,
                    "relative_half_width": 0.0,
                    "peak_potential_error": float(errors.max()),
                }
            )
            continue
        width = float(grid[above[-1]] - grid[above[0]])
        rows.append(
            {
                "threshold": float(threshold),
                "exceeded": True,
                "ka_lo": float(grid[above[0]]),
                "ka_hi": float(grid[above[-1]]),
                "full_width": width,
                "relative_half_width": float(0.5 * width / centre),
                "peak_potential_error": float(errors.max()),
            }
        )
    return rows


def band_rows(
    mesh: PanelMesh,
    centre: float,
    *,
    offsets: Sequence[float] = (-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10),
    resonance: str | None = None,
    variants: Sequence[str] = ("unaugmented", "auto"),
    radius: float = _DEFAULT_RADIUS,
    depth: float = _DEFAULT_DEPTH,
) -> list[dict[str, Any]]:
    """Rows of ``kappa`` and load/potential error across a detuning grid."""
    rows: list[dict[str, Any]] = []
    for offset in offsets:
        ka = float(centre) + float(offset)
        if ka <= 0.0:
            continue
        for variant in variants:
            if variant == "nodal" and (
                resonance is None or RESONANCES[resonance][0] == 0
            ):
                continue
            rows.append(
                probe(
                    mesh,
                    ka,
                    variant,
                    resonance=resonance,
                    radius=radius,
                    depth=depth,
                ).as_dict()
            )
    return rows


def run(
    panel_counts: Sequence[int] = (32, 64, 128),
    radius: float = _DEFAULT_RADIUS,
    depth: float = _DEFAULT_DEPTH,
    density: float = _DEFAULT_DENSITY,
    gravity: float = _DEFAULT_GRAVITY,
    *,
    half_width: float = 0.30,
    band_samples: int = 25,
) -> dict[str, Any]:
    """Run the full study: resonance locations, band widths, CHIEF, nodal control."""
    zeros = {
        f"j{order}{index + 1}": float(value)
        for order in (0, 1, 2)
        for index, value in enumerate(dirichlet_zeros(order, 2))
    }
    offband = maccamy_fuchs_diffraction(
        1.0, radius=radius, depth=depth, density=density, gravity=gravity
    )
    study: dict[str, Any] = {
        "benchmark": "2d_helmholtz_irregular_frequency_band_study",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "parameters": {
            "radius": radius,
            "depth": depth,
            "density": density,
            "gravity": gravity,
            "panel_counts": [int(n) for n in panel_counts],
            "half_width": float(half_width),
            "band_samples": int(band_samples),
        },
        "bessel_zeros": zeros,
        "offband_reference": {
            "ka": 1.0,
            "exact_force_modulus": float(abs(complex(offband["force_x"]))),
        },
        "resonances": {},
    }

    for name, (order, analytic_ka) in RESONANCES.items():
        entry: dict[str, Any] = {
            "order": int(order),
            "analytic_ka": float(analytic_ka),
            "panel_counts": {},
        }
        variants = ["unaugmented", "auto"]
        if order >= 1:
            variants.append("nodal")
        for n in panel_counts:
            mesh = regular_polygon(int(n), radius=radius)
            located = locate_resonance(
                mesh, analytic_ka - half_width, analytic_ka + half_width, radius=radius
            )
            entry["panel_counts"][str(int(n))] = {
                "discrete_resonance": located,
                "detuning_relative": float(
                    (located["ka"] - analytic_ka) / analytic_ka
                ),
                "peak_to_zero_kappa_ratio": float(
                    located["condition_number"]
                    / probe(mesh, analytic_ka, "unaugmented", radius=radius).condition_number
                ),
                "at_analytic_zero": [
                    probe(
                        mesh,
                        analytic_ka,
                        variant,
                        resonance=name,
                        radius=radius,
                        depth=depth,
                        density=density,
                        gravity=gravity,
                    ).as_dict()
                    for variant in variants
                ],
                "at_discrete_resonance": [
                    probe(
                        mesh,
                        located["ka"],
                        variant,
                        resonance=name,
                        radius=radius,
                        depth=depth,
                        density=density,
                        gravity=gravity,
                    ).as_dict()
                    for variant in variants
                ],
                "band": band_rows(
                    mesh,
                    analytic_ka,
                    resonance=name,
                    variants=variants,
                    radius=radius,
                    depth=depth,
                ),
                "amplification": amplification_band(
                    mesh,
                    analytic_ka,
                    half_width=half_width,
                    samples=band_samples,
                    radius=radius,
                    depth=depth,
                ),
            }
        study["resonances"][name] = entry
    return study


def _summarise(study: dict[str, Any]) -> str:
    header = (
        f"{'resonance':<18}{'panels':>7}{'ka_peak':>11}{'detune%':>9}"
        f"{'k_peak/k_0':>11}{'potErr_none':>12}{'potErr_chief':>13}"
        f"{'F_err_none':>11}{'F_err_chief':>12}"
    )
    lines = [header]
    for name, entry in study["resonances"].items():
        for panels, row in entry["panel_counts"].items():
            peak = {p["variant"]: p for p in row["at_discrete_resonance"]}
            lines.append(
                f"{name:<18}{panels:>7}{row['discrete_resonance']['ka']:>11.6f}"
                f"{100.0 * row['detuning_relative']:>+9.3f}"
                f"{row['peak_to_zero_kappa_ratio']:>11.1f}"
                f"{peak['unaugmented']['potential_relative_error']:>12.3e}"
                f"{peak['auto']['potential_relative_error']:>13.3e}"
                f"{peak['unaugmented']['force_relative_error']:>11.3e}"
                f"{peak['auto']['force_relative_error']:>12.3e}"
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--radius", type=float, default=_DEFAULT_RADIUS)
    parser.add_argument("--depth", type=float, default=_DEFAULT_DEPTH)
    parser.add_argument("--half-width", type=float, default=0.30)
    parser.add_argument("--band-samples", type=int, default=25)
    parser.add_argument("--write", action="store_true", help="write the JSON artifact")
    parser.add_argument("--output", type=Path, help="explicit JSON output path")
    args = parser.parse_args(argv)

    study = run(
        panel_counts=args.panels,
        radius=args.radius,
        depth=args.depth,
        half_width=args.half_width,
        band_samples=args.band_samples,
    )
    print(_summarise(study))
    if args.write or args.output is not None:
        target = args.output
        if target is None:
            root = Path(__file__).resolve().parents[3]
            target = root / "artifacts" / "irregular_frequency_study.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(study, indent=2, sort_keys=True) + "\n")
        print(f"wrote {target}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())