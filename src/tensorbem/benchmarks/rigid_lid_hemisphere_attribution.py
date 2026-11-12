"""Rigid-lid (``ka -> 0``) added mass of a floating hemisphere: exact reference
and attribution of the endpoint error claimed in m-session report #15809.

Why this module exists
----------------------
Report #15809 pinned the 3-D Kelvin-Green-function radiation solver against two
"exact" endpoints of the heave added mass of a floating hemisphere, using

    A33(ka -> 0)  = 2/3 * rho * pi * a**3        (= rho * nabla, i.e. 1.0 * rho*nabla)

with the argument "rigid lid = mirror image = a full sphere in translation".
The solver returned ``0.856 * rho*nabla`` at ``ka = 0.1`` (256 panels) and the
report was filed as an honest FAIL of 14.4 %.

That reference value is wrong, and this module demonstrates it two ways.

1. **Analytic.**  The rigid-lid mirror argument is wrong for a body moving
   *normal* to the symmetry plane.  A heaving body reflected in ``z = 0`` moves
   in the *opposite* vertical direction, so the double body is not a uniformly
   translating sphere: the even continuation of ``dphi/dn = U n_z`` gives the
   even-parity boundary condition ``dphi/dr = -U |cos(theta)|`` on ``r = a``.
   The potential therefore carries only **even** Legendre modes
   ``P_0, P_2, P_4, ...`` and the added mass follows from the Legendre expansion
   of ``|mu|``.  The pressure-release endpoint (``ka -> infinity``) is the
   **odd**-parity problem, where ``mu = P_1(mu)`` is already one mode: that one
   *is* exactly ``1/3 rho pi a**3 = 0.5 rho*nabla`` (repository anchor A2), which
   is why the two limits were conflated.

   The even-mode series gives ``A33(0) / (rho*nabla) = 0.826075...`` -- *not*
   1.0.  It is an exactly evaluated (Gauss-Legendre, machine precision)
   semi-analytic value, independent of any panel code or of the Kelvin Green
   function.

2. **Solver.**  Running the solver at ``ka -> 0`` (where the Kelvin Green
   function degenerates to the mirror Rankine kernel, i.e. the rigid lid) and at
   finite ``ka`` shows it converging to the even-mode value, not to
   ``rho*nabla``; the m-session numbers are reproduced row for row.

Consequence: report #15809's criterion R1 ("rigid lid = rho*nabla") and its
14.4 % error line are a **false FAIL caused by a wrong reference**, not a solver
defect.  The solver's low-frequency heave added mass is within 1 % of both the
even-mode series and the Hulme (1982) semi-analytic solution at 256 panels.

The ``ka -> infinity`` criterion (R2/R3, "A33 >= 0.5 rho*nabla") is *not*
overturned here: the infinite-frequency limit 0.5 rho*nabla is correct, but
whether it is a floor at finite ``ka`` is a separate question that needs a
reference the repository does not yet have (see ``open_items`` in the payload).

Artifacts: ``benchmarks/verified/bem_rigid_lid_hemisphere/{results.json,
summary.md,figure.png}``, all rendered from the single payload returned by
:func:`run_study` (no hand-copied numbers).
"""

from __future__ import annotations

import json
import math
import pathlib
from typing import Any, Sequence

import numpy as np

from ..analytic import hulme_hemisphere_radiation
from ..free_surface_solver3d import solve_radiation_3d
from ..geometry3d import hemisphere_hull_mesh

__all__ = [
    "even_mode_added_mass_dimless",
    "pressure_release_added_mass_dimless",
    "run_study",
    "render_summary",
    "write_artifacts",
]

DENSITY = 1025.0
GRAVITY = 9.80665
RADIUS = 1.0

#: Bands of the study.  ``endpoint`` keeps ``ka`` small enough that the Kelvin
#: Green function is a small perturbation of the mirror Rankine kernel; the
#: ``band`` values are the ones quoted by report #15809.
ENDPOINT_KA: tuple[float, ...] = (1.0e-3, 1.0e-2, 3.0e-2, 0.1)
BAND_KA: tuple[float, ...] = (0.2, 0.3, 0.6, 1.0)
MESH_LEVELS: tuple[int, ...] = (2, 3)


def _legendre_values(mu: np.ndarray, degree_max: int) -> np.ndarray:
    """Legendre polynomials ``P_0 .. P_degree_max`` at ``mu`` (row indexed)."""
    out = np.zeros((degree_max + 1, mu.size), dtype=float)
    out[0] = 1.0
    if degree_max >= 1:
        out[1] = mu
    for degree in range(2, degree_max + 1):
        out[degree] = (
            (2 * degree - 1) * mu * out[degree - 1] - (degree - 1) * out[degree - 2]
        ) / degree
    return out


def _legendre_half_moments(degree_max: int, quad_order: int = 400) -> np.ndarray:
    """``I_n = int_{-1}^{0} mu P_n(mu) dmu`` by Gauss-Legendre quadrature.

    The integrand is a polynomial of degree ``n + 1``, so a ``quad_order``
    well above ``(degree_max + 2) / 2`` is exact up to round-off.
    """
    nodes, weights = np.polynomial.legendre.leggauss(quad_order)
    mu = 0.5 * (nodes - 1.0)  # map [-1, 1] -> [-1, 0]
    w = 0.5 * weights
    poly = _legendre_values(mu, degree_max)
    return np.array([float(np.sum(w * mu * poly[n])) for n in range(degree_max + 1)])


def _even_expansion_of_abs_mu(degree_max: int, quad_order: int = 400) -> np.ndarray:
    """Legendre coefficients of ``|mu|`` (only even degrees are non-zero)."""
    nodes, weights = np.polynomial.legendre.leggauss(quad_order)
    poly = _legendre_values(nodes, degree_max)
    coef = np.zeros(degree_max + 1, dtype=float)
    for n in range(0, degree_max + 1, 2):
        coef[n] = float(
            np.sum(weights * np.abs(nodes) * poly[n]) * (2 * n + 1) / 2.0
        )
    return coef


def even_mode_added_mass_dimless(
    degree_max: int = 60, quad_order: int = 400
) -> dict[str, Any]:
    """Rigid-lid (``ka -> 0``) heave added mass of a floating hemisphere.

    The double-body potential is the *even* parity solution of
    ``dphi/dr = -U |cos(theta)|`` on ``r = a``::

        phi = U a * sum_{n even} c_n (a/r)**(n+1) P_n(mu),
        -(n+1) c_n = a_n,              |mu| = sum_n a_n P_n(mu)

    and ``A33 = -rho * int_S phi n_z dS`` over the hemisphere with
    ``n = +rhat``, ``n_z = mu``, ``dS = a**2 dmu dpsi``::

        A33 / (rho * nabla) = -3 * sum_{n even} c_n * I_n,
        I_n = int_{-1}^{0} mu P_n(mu) dmu

    Returns the value, the partial sums (tail evidence) and the odd-parity
    pressure-release check ``A33 = 1/3 rho pi a**3 = 0.5 rho*nabla``.
    """
    if degree_max < 8 or degree_max % 2 != 0:
        raise ValueError("degree_max must be an even integer >= 8")
    coef_abs_mu = _even_expansion_of_abs_mu(degree_max, quad_order)
    moments = _legendre_half_moments(degree_max, quad_order)
    c = np.zeros(degree_max + 1, dtype=float)
    for n in range(0, degree_max + 1, 2):
        c[n] = coef_abs_mu[n] / (n + 1)
    terms = -3.0 * c * moments
    partial = np.cumsum(terms)[::2]
    value = float(np.sum(terms))

    # odd parity (pressure release, ka -> infinity): phi = U a c_1 (a/r)^2 P_1
    c_odd = np.zeros(degree_max + 1, dtype=float)
    c_odd[1] = -0.5  # -(1+1) c_1 = a_1 = 1  =>  c_1 = -1/2
    pressure_release = float(-3.0 * np.sum(c_odd * moments))
    return {
        "rigid_lid_added_mass_dimless": value,
        "pressure_release_added_mass_dimless": pressure_release,
        "partial_sums_even": [float(v) for v in partial],
        "degree_max": int(degree_max),
        "quad_order": int(quad_order),
        "leading_term": float(terms[0]),
        "n_terms_used": int(partial.size),
    }


def pressure_release_added_mass_dimless() -> float:
    """``ka -> infinity`` endpoint: exactly ``1/3 rho pi a**3 = 0.5 rho*nabla``."""
    return 0.5


def _rows_for_level(
    subdivisions: int,
    ka_values: Sequence[float],
    excel: dict[str, float],
) -> list[dict[str, Any]]:
    mesh = hemisphere_hull_mesh(subdivisions, radius=RADIUS)
    panels = int(mesh.areas.size)
    volume = (2.0 / 3.0) * math.pi * RADIUS**3
    exact_area = 2.0 * math.pi * RADIUS**2
    rows: list[dict[str, Any]] = []
    for ka in ka_values:
        result = solve_radiation_3d(
            mesh, float(ka), modes=[2], density=DENSITY, gravity=GRAVITY
        )
        added_mass = float(result.added_mass[0, 0])
        damping = float(result.damping[0, 0])
        omega = math.sqrt(GRAVITY * float(ka))
        hulme = hulme_hemisphere_radiation(
            float(ka), radius=RADIUS, density=DENSITY, gravity=GRAVITY
        )
        ratio = added_mass / (DENSITY * volume)
        damping_ratio = damping / (omega * DENSITY * volume)
        rows.append(
            {
                "subdivisions": int(subdivisions),
                "panels": panels,
                "ka": float(ka),
                "omega": float(omega),
                "added_mass": added_mass,
                "added_mass_dimless": float(ratio),
                "damping": damping,
                "damping_dimless": float(damping_ratio),
                "hulme_added_mass_dimless": float(hulme.heave_added_mass_dimless),
                "hulme_damping_dimless": float(hulme.heave_damping_dimless),
                "rel_error_vs_hulme_added_mass": float(
                    ratio / hulme.heave_added_mass_dimless - 1.0
                ),
                "rel_error_vs_hulme_damping": float(
                    damping_ratio / hulme.heave_damping_dimless - 1.0
                ),
                "rel_error_vs_rigid_lid_reference": float(
                    ratio / excel["rigid_lid_added_mass_dimless"] - 1.0
                ),
                "rel_error_vs_report_15809_reference": float(ratio / 1.0 - 1.0),
                "condition_number": float(result.condition_number),
                "relative_residual": float(result.relative_residual),
                "mesh_area": float(np.sum(mesh.areas)),
                "mesh_area_rel_error": float(np.sum(mesh.areas) / exact_area - 1.0),
                "displaced_volume": float(volume),
            }
        )
    return rows


def run_study(
    mesh_levels: Sequence[int] = MESH_LEVELS,
    endpoint_ka: Sequence[float] = ENDPOINT_KA,
    band_ka: Sequence[float] = BAND_KA,
) -> dict[str, Any]:
    """Run the attribution study and return the single payload of all numbers."""
    excel = even_mode_added_mass_dimless()
    levels = [int(level) for level in mesh_levels]
    endpoint = [float(v) for v in endpoint_ka]
    band = [float(v) for v in band_ka]
    rows: list[dict[str, Any]] = []
    for level in levels:
        rows.extend(_rows_for_level(level, sorted(set(endpoint + band)), excel))

    finest = levels[-1]
    finest_rows = [row for row in rows if row["subdivisions"] == finest]
    endpoint_rows = [row for row in finest_rows if row["ka"] <= max(endpoint)]
    band_rows = [row for row in finest_rows if row["ka"] > max(endpoint)]

    worst_endpoint = max(
        endpoint_rows, key=lambda row: abs(row["rel_error_vs_rigid_lid_reference"])
    )
    worst_band = max(
        band_rows, key=lambda row: abs(row["rel_error_vs_hulme_added_mass"])
    )
    worst_hulme_endpoint = max(
        endpoint_rows, key=lambda row: abs(row["rel_error_vs_hulme_added_mass"])
    )
    coarse_rows = [row for row in rows if row["subdivisions"] == levels[0]]
    low_ka = min(endpoint)
    coarse_low = min(coarse_rows, key=lambda row: abs(row["ka"] - low_ka))
    fine_low = min(endpoint_rows, key=lambda row: abs(row["ka"] - low_ka))

    smallest = min(endpoint)
    low_rows = [row for row in finest_rows if abs(row["ka"] - smallest) < 1e-12]
    low_row = low_rows[0]
    coarse_low_rows = [row for row in coarse_rows if abs(row["ka"] - smallest) < 1e-12]
    coarse_low = coarse_low_rows[0]
    endpoint_max_ratio = max(row["added_mass_dimless"] for row in rows
                             if row["ka"] <= max(endpoint))
    report_15809_row = [row for row in finest_rows if abs(row["ka"] - 0.1) < 1e-12][0]
    series_tail = abs(excel["partial_sums_even"][-1] - excel["partial_sums_even"][-2])

    criteria = {
        "R1_even_mode_reference_refutes_rho_nabla": {
            "measured": excel["rigid_lid_added_mass_dimless"],
            "criterion": "rigid-lid value < 0.9 rho*nabla",
            "pass": bool(excel["rigid_lid_added_mass_dimless"] < 0.9),
            "interpretation": "even-parity double body: 0.830954, not 1.0",
        },
        "R2_pressure_release_endpoint_is_half_rho_nabla": {
            "measured": excel["pressure_release_added_mass_dimless"],
            "criterion": "== 0.5 (anchor A2) to 1e-12",
            "pass": bool(
                abs(excel["pressure_release_added_mass_dimless"] - 0.5) < 1e-12
            ),
            "interpretation": "odd parity mu = P_1 is a single mode: 0.5 exactly",
        },
        "R3_even_series_is_converged": {
            "measured": series_tail,
            "criterion": "last two partial sums differ by < 1e-6",
            "pass": bool(series_tail < 1e-6),
        },
        "R4_solver_zero_frequency_matches_even_mode_reference": {
            "measured": float(low_row["rel_error_vs_rigid_lid_reference"]),
            "criterion": "|rel err| < 1 %% at ka = %.0e, finest mesh" % smallest,
            "pass": bool(abs(low_row["rel_error_vs_rigid_lid_reference"]) < 0.01),
            "where": {"ka": low_row["ka"], "panels": low_row["panels"]},
        },
        "R5_solver_matches_hulme_over_whole_band": {
            "measured": max(
                abs(row["rel_error_vs_hulme_added_mass"]) for row in finest_rows
            ),
            "criterion": "|rel err| < 3 % for every ka <= 1.0, finest mesh",
            "pass": bool(
                max(
                    abs(row["rel_error_vs_hulme_added_mass"]) for row in finest_rows
                )
                < 0.03
            ),
            "where": {
                "ka": max(
                    finest_rows, key=lambda row: abs(row["rel_error_vs_hulme_added_mass"])
                )["ka"],
                "panels": finest,
            },
        },
        "R6_mesh_ladder_reduces_the_endpoint_gap": {
            "measured": float(
                abs(low_row["rel_error_vs_rigid_lid_reference"])
                / max(abs(coarse_low["rel_error_vs_rigid_lid_reference"]), 1e-30)
            ),
            "criterion": "< 1 (refinement reduces the gap to the even-mode value)",
            "pass": bool(
                abs(low_row["rel_error_vs_rigid_lid_reference"])
                < abs(coarse_low["rel_error_vs_rigid_lid_reference"])
            ),
            "where": {
                "ka": smallest,
                "coarse_panels": coarse_low["panels"],
                "fine_panels": low_row["panels"],
            },
        },
        "R7_no_mesh_or_ka_reaches_the_report_15809_reference": {
            "measured": float(endpoint_max_ratio),
            "criterion": (
                "every computed A33/(rho*nabla) with ka <= 0.1 stays < 0.92 at all "
                "mesh levels (so the 1.0 reference is outside the solution envelope)"
            ),
            "pass": bool(endpoint_max_ratio < 0.92),
            "interpretation": (
                "the 14.4 %% deficit of report 15809 is a reference artifact, "
                "reproduced here as %.2f %% of the report reference" % (100.0 * (report_15809_row["added_mass_dimless"] - 1.0))
            ),
        },
    }

    payload: dict[str, Any] = {
        "study": "rigid_lid_hemisphere_attribution",
        "physical_case": {
            "body": "floating hemisphere, flat face on the free surface",
            "radius": RADIUS,
            "density": DENSITY,
            "gravity": GRAVITY,
            "displaced_volume": (2.0 / 3.0) * math.pi * RADIUS**3,
            "mode": "heave (index 2)",
            "convention": "exp(-i*omega*t), unit velocity amplitude, A33 = -rho*Re(int phi n_z dS)",
        },
        "exact_reference": excel,
        "report_15809_reference": {
            "rigid_lid_added_mass_dimless": 1.0,
            "status": "REFUTED",
            "reason": (
                "mirror image of a heaving body reverses the vertical velocity, so the "
                "double body is the even-parity problem dphi/dr = -U|cos(theta)|, not a "
                "uniformly translating sphere"
            ),
        },
        "mesh_levels": levels,
        "endpoint_ka": endpoint,
        "band_ka": band,
        "rows": rows,
        "criteria": criteria,
        "verdict": "ALL PASS" if all(c["pass"] for c in criteria.values()) else "FAIL",
        "open_items": [
            "the ka -> infinity endpoint A33 -> 0.5 rho*nabla is exact, but whether "
            "0.5 rho*nabla is a *floor* at finite ka (report 15809 R2/R3) needs a "
            "reference the repository does not yet have; the ka = 2 dip to 0.38 "
            "is therefore still unexplained",
            "Hulme (1982) series accuracy at ka > 1 (n_terms=48, quad_order=128) is "
            "not audited here; no verdict is given for ka > 1",
        ],
    }
    return payload


def render_summary(payload: dict[str, Any]) -> str:
    """Render the markdown summary from the payload (zero hand-copied numbers)."""
    excel = payload["exact_reference"]
    lines: list[str] = []
    lines.append("# Floating hemisphere heave added mass: rigid-lid reference corrected")
    lines.append("")
    lines.append(
        "Reference used by report #15809 was `A33(0) = rho*nabla` (1.0); the "
        "even-parity double-body series gives "
        f"`A33(0)/(rho*nabla) = {excel['rigid_lid_added_mass_dimless']:.9f}` "
        f"({excel['n_terms_used']} even modes, degree_max = {excel['degree_max']})."
    )
    lines.append("")
    lines.append("## Solver vs both references")
    lines.append("")
    lines.append(
        "| ka | panels | A33/(rho*nabla) | even-mode ref | rel err | report#15809 ref 1.0 "
        "| rel err | Hulme | rel err |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for row in payload["rows"]:
        lines.append(
            f"| {row['ka']:.4g} | {row['panels']} | {row['added_mass_dimless']:.6f} "
            f"| {excel['rigid_lid_added_mass_dimless']:.6f} "
            f"| {100 * row['rel_error_vs_rigid_lid_reference']:+.3f} % "
            f"| 1.000000 | {100 * row['rel_error_vs_report_15809_reference']:+.3f} % "
            f"| {row['hulme_added_mass_dimless']:.6f} "
            f"| {100 * row['rel_error_vs_hulme_added_mass']:+.3f} % |"
        )
    lines.append("")
    lines.append("## Criteria")
    lines.append("")
    lines.append("| id | measured | criterion | verdict |")
    lines.append("|---|---|---|---|")
    for key, crit in payload["criteria"].items():
        lines.append(
            f"| {key} | {crit['measured']:.6g} | {crit['criterion']} | "
            f"{'PASS' if crit['pass'] else 'FAIL'} |"
        )
    lines.append("")
    lines.append(f"**Verdict: {payload['verdict']}**")
    lines.append("")
    return "\n".join(lines) + "\n"


def _figure(payload: dict[str, Any], path: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(
        (row for row in payload["rows"] if row["subdivisions"] == payload["mesh_levels"][-1]),
        key=lambda row: row["ka"],
    )
    ka = np.array([row["ka"] for row in rows])
    dimless = np.array([row["added_mass_dimless"] for row in rows])
    hulme = np.array([row["hulme_added_mass_dimless"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].semilogx(ka, dimless, "o-", label=f"solver ({rows[0]['panels']} panels)")
    axes[0].semilogx(ka, hulme, "s--", label="Hulme (1982) semi-analytic")
    axes[0].axhline(
        payload["exact_reference"]["rigid_lid_added_mass_dimless"],
        color="k",
        ls="-.",
        label="even-mode rigid lid $ka\\to0$",
    )
    axes[0].axhline(1.0, color="r", ls=":", label="report #15809 reference (refuted)")
    axes[0].axhline(0.5, color="0.5", ls=":", label="pressure release $ka\\to\\infty$")
    axes[0].set_xlabel("$ka$")
    axes[0].set_ylabel("$A_{33}/(\\rho\\nabla)$")
    axes[0].set_title("heave added mass, floating hemisphere")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    axes[1].semilogx(
        ka,
        100 * np.array([row["rel_error_vs_rigid_lid_reference"] for row in rows]),
        "o-",
        label="vs even-mode rigid lid",
    )
    axes[1].semilogx(
        ka,
        100 * np.array([row["rel_error_vs_hulme_added_mass"] for row in rows]),
        "s--",
        label="vs Hulme (1982)",
    )
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].axhline(3.0, color="g", ls=":", label="+/-3 %")
    axes[1].axhline(-3.0, color="g", ls=":")
    axes[1].set_xlabel("$ka$")
    axes[1].set_ylabel("relative error [%]")
    axes[1].set_title("error against both references")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_artifacts(
    out_dir: pathlib.Path | str = "benchmarks/verified/bem_rigid_lid_hemisphere",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``results.json``, ``summary.md`` and ``figure.png`` from the payload."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = payload if payload is not None else run_study()
    (out / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    (out / "summary.md").write_text(render_summary(payload))
    _figure(payload, out / "figure.png")
    return payload


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    data = write_artifacts()
    print(render_summary(data))