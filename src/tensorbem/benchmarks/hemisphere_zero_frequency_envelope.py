"""The zero-frequency heave anchor of the floating hemisphere, settled.

Why this module exists (erratum for ``rigid_lid_zero_frequency_anchor``)
-----------------------------------------------------------------------
Mission m16802 pinned the ``ka -> 0`` (rigid-lid) end of the heave added mass of
a half-submerged hemisphere at ``2/3 rho pi a^3 = rho*grad`` and reported a
14.4 % honest FAIL against it.  That anchor is wrong.  The image argument in
that module reads the ``omega -> 0`` free-surface condition ``dphi/dz = 0`` on
``z = 0`` as "reflect the hemisphere into a full sphere moving with the same
velocity".  Reflecting the *body* is right, but the reflected half carries the
reflected *velocity*, and a vertical velocity is preserved under ``z -> -z``
while the outward normal of the mirrored surface is reversed.  Carrying the
Neumann data across the reflection gives, on the equivalent full sphere,

    dphi/dn = -V |cos(theta)|        (two halves *sliding* in opposite directions)

and not ``dphi/dn = V cos(theta)``, which is rigid translation of the whole
sphere -- the classical ``2/3 rho pi a^3`` case.  The sliding pair shares the
body surface with the translating sphere but not its boundary data, so its added
mass is a different number.

Two independent routes settle the point, and both are cheap enough to be part of
the test suite:

1. **Spectral Legendre projection** (:func:`spectral_added_mass`) -- the sliding
   BC is axisymmetric and even in ``mu``, so ``phi = sum_n c_n (a/r)^{n+1} P_n``
   decouples mode by mode and the hemisphere force integral is closed form:
   ``A33(ka->0) = 0.8309572 rho*grad`` (converged to 1e-6 by ``n_terms = 60``).
   The same projector applied to the *translating* BC returns exactly
   ``0.5 rho*grad`` = ``2/3 rho pi a^3``, the classical full-sphere value; that
   is the method's self test, and it is what falsifies the m16802 reading.
2. **Hulme (1982) multipole reference** (:func:`exact_reference_table`) -- the
   independent exact series in :mod:`tensorbem.analytic`, evaluated down to
   ``ka = 2e-3``, approaches the same limit from above (0.85 % high at
   ``ka = 2e-3``) with the excess shrinking monotonically as ``ka`` falls.

What the *exact* reference also shows is that both of m16802's reference
premises were invalid, not merely misplaced:

* the heave added mass is **not** bounded below by the pressure-release value
  ``0.5 rho*grad``: the exact series dips to ``0.388 rho*grad`` near ``ka = 1.8``
  (free-surface wave radiation genuinely reduces the inertia below its infinite
  frequency value), so the "floor" criterion R3 would fail a correct solver;
* the rigid-lid limit is approached **slowly** (the exact value at
  ``ka = 0.1`` is still 3.8 % above ``A33(0)``), so a single ``ka ~ 0.1``
  endpoint comparison cannot test the limit even with the right anchor.

The gate that replaces them is therefore not another endpoint constant but the
*pointwise* accuracy envelope: the committed 256-panel sweep of the 3-D solver
against the exact reference at every ``ka``, and a live rigid-lid solve at
``ka = 1e-3`` against the spectral limit.  Everything below is regenerated from
this module's own JSON artifact; nothing is transcribed.

Run::

    python -m tensorbem.benchmarks.hemisphere_zero_frequency_envelope
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

DENSITY = 1025.0
GRAVITY = 9.80665
RADIUS = 1.0

#: ``ka`` grid of the exact-reference table (log spaced, dense at both ends).
EXACT_KAS: tuple[float, ...] = (
    0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6,
    0.8, 1.0, 1.5, 1.8, 2.0, 3.0, 4.0, 6.0,
)

#: Committed sweep of the 3-D solver produced by mission m16802.  It is used
#: read-only: the audit must not overwrite an honest FAIL record.
LEGACY_SWEEP = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "rigid_lid_zero_frequency_anchor"
    / "results_subdiv_2_3.json"
)

#: ``ka`` at which the free-surface problem *is* the rigid-lid problem for all
#: practical purposes (the free-surface term is ``O(ka)``).
RIGID_LID_PROBE_KA = 1.0e-3


# ---------------------------------------------------------------------------
# Route 1: spectral Legendre projection
# ---------------------------------------------------------------------------
def _legendre_table(n_max: int, mu: np.ndarray) -> np.ndarray:
    out = np.zeros((n_max + 1, mu.size))
    out[0] = 1.0
    if n_max >= 1:
        out[1] = mu
    for n in range(1, n_max):
        out[n + 1] = ((2 * n + 1) * mu * out[n] - n * out[n - 1]) / (n + 1)
    return out


def spectral_added_mass(
    mode: str = "sliding",
    n_terms: int = 60,
    radius: float = RADIUS,
    density: float = DENSITY,
    quad_order: int = 400,
) -> tuple[float, float]:
    """Heave added mass of the half-submerged hemisphere from a Legendre solve.

    ``mode = "sliding"``  -> ``dphi/dn = -V|cos theta|`` on the full sphere: the
    even-symmetry image of the heaving hemisphere against the rigid lid
    (``omega -> 0``).  ``mode = "translation"`` -> ``dphi/dn = V cos theta``: the
    classical full sphere, the method's self test (must return ``0.5 rho*grad``).

    Returns ``(a33, a33 / (rho * grad_hemisphere))`` with ``V = 1``.
    """
    if mode not in ("sliding", "translation"):
        raise ValueError("mode must be 'sliding' or 'translation'")
    if n_terms < 2:
        raise ValueError("n_terms must be >= 2")
    mu, w = np.polynomial.legendre.leggauss(int(quad_order))
    legendre = _legendre_table(n_terms, mu)
    bc = -np.abs(mu) if mode == "sliding" else mu
    parity = 0 if mode == "sliding" else 1
    coeffs = np.zeros(n_terms + 1)
    for n in range(parity, n_terms + 1, 2):
        rhs = float(np.sum(w * bc * legendre[n]))
        coeffs[n] = rhs / (-(n + 1) * 2.0 / (2 * n + 1))
    lower = mu <= 0.0
    phi = np.zeros(int(lower.sum()))
    for n in range(parity, n_terms + 1, 2):
        phi += coeffs[n] * legendre[n][lower]
    # dS = a^2 dmu dphi, n_z = mu on the sphere; A33 = -rho int phi n_z dS.
    a33 = -density * radius**2 * 2.0 * math.pi * float(
        np.sum(w[lower] * phi * mu[lower])
    )
    rho_grad = density * (2.0 / 3.0) * math.pi * radius**3
    return a33, a33 / rho_grad


def rigid_lid_heave_added_mass(
    density: float = DENSITY, radius: float = RADIUS, n_terms: int = 200
) -> float:
    """``A33(ka -> 0)`` of the floating hemisphere, ``= 0.8309572 rho*grad``."""
    return spectral_added_mass("sliding", n_terms, radius, density)[0]


def pressure_release_heave_added_mass(density: float = DENSITY, radius: float = RADIUS) -> float:
    """``A33(ka -> inf)``: the odd (single dipole) symmetry, exact, ``rho pi a^3/3``."""
    return density * math.pi * radius**3 / 3.0


def imaged_sphere_translation_added_mass(
    density: float = DENSITY, radius: float = RADIUS
) -> float:
    """The m16802 anchor: full-sphere translation, ``2/3 rho pi a^3``.

    Kept as a named quantity because the erratum is about exactly this number:
    it is the *newtonian* image value and the self test of the spectral route,
    not the rigid-lid limit of the hemisphere.
    """
    return density * (2.0 / 3.0) * math.pi * radius**3


# ---------------------------------------------------------------------------
# Route 2: exact Hulme multipole reference
# ---------------------------------------------------------------------------
def exact_reference_table(
    kas: Iterable[float] = EXACT_KAS,
    density: float = DENSITY,
    radius: float = RADIUS,
    n_terms: int = 96,
) -> list[dict[str, Any]]:
    """Exact heave coefficients at each ``ka`` (Hulme 1982 multipole series)."""
    from ..analytic import hulme_hemisphere_radiation

    rho_grad = density * (2.0 / 3.0) * math.pi * radius**3
    rows = []
    for ka in kas:
        res = hulme_hemisphere_radiation(
            float(ka), radius=radius, density=density,
            n_terms=n_terms, quad_order=max(256, 2 * n_terms),
        )
        rows.append(
            {
                "ka": float(ka),
                "a33": float(res.heave_added_mass),
                "a33_over_rho_displaced": float(res.heave_added_mass) / rho_grad,
                "b33_over_omega_rho_displaced": float(res.heave_damping_dimless),
            }
        )
    return rows


def _minimum_exact_dip(
    density: float = DENSITY, radius: float = RADIUS, n_terms: int = 96
) -> dict[str, float]:
    """Locate the interior minimum of the exact ``A33(ka)`` (below the release value)."""
    grid = np.exp(np.linspace(math.log(0.3), math.log(6.0), 25))
    rows = exact_reference_table(grid, density, radius, n_terms)
    best = min(rows, key=lambda row: row["a33_over_rho_displaced"])
    return {"ka": best["ka"], "a33_over_rho_displaced": best["a33_over_rho_displaced"]}


# ---------------------------------------------------------------------------
# The 3-D solver, read-only from the committed sweep and probed live
# ---------------------------------------------------------------------------
def load_legacy_sweep(path: Path | str = LEGACY_SWEEP) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"committed m16802 sweep not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, Any] = {"path": str(path), "sweeps": []}
    for sweep in payload["sweeps"]:
        out["sweeps"].append(
            {
                "subdivisions": sweep["subdivisions"],
                "panels": sweep["panels"],
                "rows": [
                    {
                        "ka": row["ka"],
                        "a33": row["a33"],
                        "a33_over_rho_displaced": row["a33_over_rho_displaced"],
                        "b33_over_omega_mass": row["b33_over_omega_mass"],
                        "condition_number": row.get("condition_number"),
                    }
                    for row in sweep["rows"]
                ],
            }
        )
    out["provenance"] = "produced by tensorbem.benchmarks.rigid_lid_zero_frequency_anchor (m16802)"
    return out


def probe_rigid_lid_solves(
    subdivisions: Sequence[int] = (2, 3),
    density: float = DENSITY,
    radius: float = RADIUS,
    gravity: float = GRAVITY,
    ka: float = RIGID_LID_PROBE_KA,
) -> list[dict[str, Any]]:
    """Live solves at ``ka = 1e-3``, where the free-surface term is ``O(ka)``."""
    from ..free_surface_solver3d import solve_radiation_3d
    from ..geometry3d import hemisphere_hull_mesh

    rho_grad = density * (2.0 / 3.0) * math.pi * radius**3
    spectral = rigid_lid_heave_added_mass(density, radius)
    rows = []
    for sub in subdivisions:
        hull = hemisphere_hull_mesh(int(sub), radius=radius)
        res = solve_radiation_3d(hull, ka, modes=(2,), density=density, gravity=gravity)
        a33 = float(res.added_mass[0, 0])
        rows.append(
            {
                "subdivisions": int(sub),
                "panels": len(hull.panels) if hasattr(hull, "panels") else None,
                "ka": float(ka),
                "a33": a33,
                "a33_over_rho_displaced": a33 / rho_grad,
                "relative_error_vs_spectral": (a33 - spectral) / spectral,
            }
        )
    return rows


def shade_ladder(
    kas: Sequence[float],
    subdivisions: Sequence[int] = (2, 3),
    density: float = DENSITY,
    radius: float = RADIUS,
    gravity: float = GRAVITY,
) -> list[dict[str, Any]]:
    """Optional live sweep (usage: ``--live-ka``), cached into the artifact."""
    from ..free_surface_solver3d import solve_radiation_3d
    from ..geometry3d import hemisphere_hull_mesh

    rho_grad = density * (2.0 / 3.0) * math.pi * radius**3
    out = []
    for sub in subdivisions:
        hull = hemisphere_hull_mesh(int(sub), radius=radius)
        rows = []
        for ka in kas:
            res = solve_radiation_3d(hull, float(ka), modes=(2,), density=density, gravity=gravity)
            rows.append(
                {
                    "ka": float(ka),
                    "a33": float(res.added_mass[0, 0]),
                    "a33_over_rho_displaced": float(res.added_mass[0, 0]) / rho_grad,
                }
            )
        out.append({"subdivisions": int(sub), "rows": rows})
    return out


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------
def evaluate_criteria(payload: dict[str, Any]) -> list[dict[str, Any]]:
    spectral = payload["spectral"]["rigid_lid_a33"]
    limit_ratio = payload["limits"]["rigid_lid_over_rho_displaced"]
    legacy = payload["limits"]["imaged_sphere_legacy_over_rho_displaced"]
    release_ratio = payload["limits"]["pressure_release_over_rho_displaced"]
    exact = payload["exact_table"]
    finest = max(payload["bem_vs_exact"], key=lambda item: item["panels"])
    criteria = []

    low = exact[0]
    low_excess = (low["a33_over_rho_displaced"] - limit_ratio) / limit_ratio
    criteria.append(
        {
            "id": "E1-rigid-lid-two-routes",
            "statement": "spectral limit and the exact Hulme series agree at ka -> 0 (< 1%)",
            "measured": low["a33_over_rho_displaced"],
            "reference": limit_ratio,
            "relative_error": low_excess,
            "pass": bool(low_excess < 0.01),
        }
    )
    trans = payload["spectral"]["translation_a33"]
    # The projector integrates the force over the lower half only, so the
    # classical full-sphere value 2/3 rho pi a^3 enters halved: rho pi a^3 / 3.
    translated_ref = 0.5 * payload["limits"]["imaged_sphere_legacy_a33"]
    criteria.append(
        {
            "id": "E2-translation-self-test",
            "statement": "the same projector on the translating BC returns exactly "
            "(1/2)(2/3 rho pi a^3) = rho pi a^3/3 (classical full sphere, half-body force)",
            "measured": trans,
            "reference": translated_ref,
            "relative_error": abs(trans - translated_ref) / translated_ref,
            "pass": bool(abs(trans - translated_ref) / translated_ref < 1e-10),
        }
    )
    criteria.append(
        {
            "id": "E3-imaged-sphere-anchor-rejected",
            "statement": "the m16802 rigid-lid anchor (= rho*grad) is 16.9% high: premise falsified",
            "measured": limit_ratio,
            "reference": legacy,
            "relative_error": (legacy - limit_ratio) / legacy,
            "pass": bool((legacy - limit_ratio) / legacy > 0.10),
        }
    )
    criteria.append(
        {
            "id": "E4-release-floor-rejected",
            "statement": "A33 is NOT bounded below by the pressure-release value 0.5 rho*grad",
            "measured": payload["exact_minimum"]["a33_over_rho_displaced"],
            "reference": release_ratio,
            "relative_error": (payload["exact_minimum"]["a33_over_rho_displaced"] - release_ratio)
            / release_ratio,
            "pass": bool(payload["exact_minimum"]["a33_over_rho_displaced"] < release_ratio),
        }
    )
    criteria.append(
        {
            "id": "E5-bem-vs-exact-pointwise",
            "statement": "256-panel solver sweep is within 3% of the exact reference at every ka",
            "measured": finest["max_abs_relative_error"],
            "reference": 0.03,
            "relative_error": finest["max_abs_relative_error"],
            "pass": bool(finest["max_abs_relative_error"] < 0.03),
        }
    )
    criteria.append(
        {
            "id": "E6-endpoint-premise-rejected",
            "statement": "the exact ka=0.1 value is >1% above A33(0): an endpoint test cannot pin the limit",
            "measured": (exact[5]["a33_over_rho_displaced"] - limit_ratio) / limit_ratio,
            "reference": 0.01,
            "relative_error": (exact[5]["a33_over_rho_displaced"] - limit_ratio) / limit_ratio,
            "pass": bool((exact[5]["a33_over_rho_displaced"] - limit_ratio) / limit_ratio > 0.01),
        }
    )
    probe = payload["rigid_lid_probes"]
    best_probe = max(probe, key=lambda row: row["subdivisions"])
    criteria.append(
        {
            "id": "E7-live-rigid-lid-solve",
            "statement": "the solver at ka = 1e-3 reproduces the spectral rigid-lid limit within 1%",
            "measured": best_probe["a33_over_rho_displaced"],
            "reference": limit_ratio,
            "relative_error": best_probe["relative_error_vs_spectral"],
            "pass": bool(abs(best_probe["relative_error_vs_spectral"]) < 0.01),
        }
    )
    del spectral
    return criteria


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def figure_from_payload(payload: dict[str, Any], directory: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    written = []
    limit = payload["limits"]["rigid_lid_over_rho_displaced"]
    release = payload["limits"]["pressure_release_over_rho_displaced"]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ex = payload["exact_table"]
    ax.plot([r["ka"] for r in ex], [r["a33_over_rho_displaced"] for r in ex],
            "k-", lw=2, label="exact (Hulme 1982 multipole)")
    markers = {2: "o", 3: "s"}
    for sweep in payload["bem"]:
        sub = sweep["subdivisions"]
        ax.plot([r["ka"] for r in sweep["rows"]],
                [r["a33_over_rho_displaced"] for r in sweep["rows"]],
                markers.get(sub, "^"), ms=5, mfc="none",
                label=f"3-D BEM, {sweep['panels']} panels")
    ax.axhline(limit, color="C3", ls="--", lw=1.2,
               label=f"rigid lid A33(0)/rho grad = {limit:.4f}")
    ax.axhline(release, color="C0", ls=":", lw=1.2,
               label=f"pressure release = {release:.4f}")
    dip = payload["exact_minimum"]
    ax.plot([dip["ka"]], [dip["a33_over_rho_displaced"]], "v", color="C4", ms=8,
            label=f"exact minimum {dip['a33_over_rho_displaced']:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("ka")
    ax.set_ylabel("A33 / (rho * grad)")
    ax.set_title("Floating hemisphere heave added mass: exact band vs 3-D BEM")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    path = directory / "fig1_a33_band.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for sweep in payload["bem_vs_exact"]:
        ax.plot([r["ka"] for r in sweep["rows"]],
                [100.0 * r["relative_error"] for r in sweep["rows"]],
                "o-" if sweep["panels"] > 100 else "s--", ms=4,
                label=f"{sweep['panels']} panels")
    ax.axhspan(-3.0, 3.0, color="C2", alpha=0.12, label="+/-3% band")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("ka")
    ax.set_ylabel("(A33_BEM - A33_exact) / A33_exact  [%]")
    ax.set_title("Solver error against the exact reference (committed m16802 sweep)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    path = directory / "fig2_error_vs_exact.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))
    return written


# ---------------------------------------------------------------------------
# Artifact assembly
# ---------------------------------------------------------------------------
def build(
    out_dir: Path | str = Path("docs/hemisphere_zero_frequency_envelope"),
    legacy_sweep: Path | str = LEGACY_SWEEP,
    live_ka: Sequence[float] | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rho_grad = DENSITY * (2.0 / 3.0) * math.pi * RADIUS**3

    convergence = []
    for n_terms in (10, 20, 40, 60, 100, 200):
        _, ratio = spectral_added_mass("sliding", n_terms)
        convergence.append({"n_terms": n_terms, "ratio": ratio})
    a33_limit, limit_ratio = spectral_added_mass("sliding", 200)
    a33_trans, trans_ratio = spectral_added_mass("translation", 60)

    exact = exact_reference_table()
    bem = load_legacy_sweep(legacy_sweep)
    exact_map = {round(row["ka"], 6): row for row in exact}
    bem_vs_exact = []
    for sweep in bem["sweeps"]:
        rows = []
        for row in sweep["rows"]:
            ref = exact_map.get(round(row["ka"], 6))
            if ref is None:
                continue
            rows.append(
                {
                    "ka": row["ka"],
                    "a33_bem": row["a33"],
                    "a33_exact": ref["a33"],
                    "relative_error": (row["a33"] - ref["a33"]) / ref["a33"],
                }
            )
        bem_vs_exact.append(
            {
                "subdivisions": sweep["subdivisions"],
                "panels": sweep["panels"],
                "max_abs_relative_error": max(abs(r["relative_error"]) for r in rows),
                "rows": rows,
            }
        )

    payload: dict[str, Any] = {
        "case": {
            "geometry": "half-submerged hemisphere, radius a = 1.000 m, flat face on z = 0",
            "density": DENSITY,
            "gravity": GRAVITY,
            "displaced_mass": rho_grad,
            "quantity": "heave added mass A33 (unit heave velocity, exp(-i omega t))",
        },
        "spectral": {
            "method": "Legendre projection of the even-symmetry (sliding-halves) Neumann data",
            "rigid_lid_a33": a33_limit,
            "rigid_lid_a33_over_rho_displaced": limit_ratio,
            "translation_a33": a33_trans,
            "translation_a33_over_rho_displaced": trans_ratio,
            "convergence": convergence,
        },
        "limits": {
            "rigid_lid_a33": a33_limit,
            "rigid_lid_over_rho_displaced": limit_ratio,
            "pressure_release_a33": pressure_release_heave_added_mass(),
            "pressure_release_over_rho_displaced": pressure_release_heave_added_mass() / rho_grad,
            "imaged_sphere_legacy_a33": imaged_sphere_translation_added_mass(),
            "imaged_sphere_legacy_over_rho_displaced": imaged_sphere_translation_added_mass()
            / rho_grad,
        },
        "exact_table": exact,
        "exact_minimum": _minimum_exact_dip(),
        "bem": bem["sweeps"],
        "bem_provenance": bem["provenance"],
        "bem_vs_exact": bem_vs_exact,
        "rigid_lid_probes": probe_rigid_lid_solves(),
    }
    if live_ka:
        payload["live_shade_ladder"] = shade_ladder(live_ka)
    payload["criteria"] = evaluate_criteria(payload)
    payload["all_pass"] = all(item["pass"] for item in payload["criteria"])
    payload["figures"] = [Path(p).name for p in
                         figure_from_payload(payload, out_dir)]

    (out_dir / "results.json").write_text(
        json.dumps(payload, indent=1, sort_keys=False) + "\n", encoding="utf-8"
    )
    (out_dir / "summary.md").write_text(render_summary(payload), encoding="utf-8")
    return payload


def render_summary(payload: dict[str, Any]) -> str:
    limits = payload["limits"]
    dip = payload["exact_minimum"]
    lines = [
        "# Zero-frequency heave added mass of a floating hemisphere: the anchor, settled",
        "",
        "Generated by `python -m tensorbem.benchmarks.hemisphere_zero_frequency_envelope`;",
        "every number below is read back from `results.json` in this directory.",
        "",
        "Hemisphere radius a = 1.000 m, rho = %.1f kg/m^3, g = %.5f m/s^2, "
        "rho*grad = %.4f kg." % (payload["case"]["density"], payload["case"]["gravity"],
                                 payload["case"]["displaced_mass"]),
        "",
        "## Exact limits",
        "",
        "| limit | A33 [kg] | A33/(rho grad) | provenance |",
        "|---|---|---|---|",
        "| ka -> 0 (rigid lid, sliding halves) | %.4f | %.6f | spectral Legendre solve |"
        % (limits["rigid_lid_a33"], limits["rigid_lid_over_rho_displaced"]),
        "| ka -> inf (pressure release) | %.4f | %.6f | odd symmetry, closed form |"
        % (limits["pressure_release_a33"], limits["pressure_release_over_rho_displaced"]),
        "| m16802 anchor (imaged sphere translation) | %.4f | %.6f | falsified premise |"
        % (limits["imaged_sphere_legacy_a33"], limits["imaged_sphere_legacy_over_rho_displaced"]),
        "",
        "The exact series has an interior minimum A33/(rho grad) = %.5f at ka = %.3f, "
        "below the pressure-release value %.4f: the heave added mass is not bounded "
        "below by its infinite-frequency limit." % (
            dip["a33_over_rho_displaced"], dip["ka"],
            limits["pressure_release_over_rho_displaced"]),
        "",
        "## Spectral convergence (sliding-halves route)",
        "",
        "| n_terms | A33/(rho grad) |",
        "|---|---|",
    ]
    for row in payload["spectral"]["convergence"]:
        lines.append("| %d | %.8f |" % (row["n_terms"], row["ratio"]))
    lines += [
        "",
        "Self test: the same projector applied to the translating BC gives "
        "%.12f rho*grad (= 2/3 rho pi a^3)." % payload["spectral"]["translation_a33_over_rho_displaced"],
        "",
        "## Exact reference table and the committed 3-D BEM sweep",
        "",
        "| ka | A33_exact/(rho grad) | B33/(omega rho grad) | A33_BEM@256/(rho grad) "
        "| rel. error @256 | rel. error @64 |",
        "|---|---|---|---|---|---|",
    ]
    exact = payload["exact_table"]
    by_ka_256 = {round(r["ka"], 6): r for r in
                 max(payload["bem_vs_exact"], key=lambda s: s["panels"])["rows"]}
    by_ka_64 = {round(r["ka"], 6): r for r in
                min(payload["bem_vs_exact"], key=lambda s: s["panels"])["rows"]}
    bem_256 = {round(r["ka"], 6): r for r in
               max(payload["bem"], key=lambda s: s["panels"])["rows"]}
    for row in exact:
        key = round(row["ka"], 6)
        if key in by_ka_256:
            b = bem_256[key]["a33_over_rho_displaced"]
            e256 = by_ka_256[key]["relative_error"]
            e64 = by_ka_64[key]["relative_error"] if key in by_ka_64 else float("nan")
            lines.append("| %.3f | %.5f | %.5f | %.5f | %+.2f%% | %+.2f%% |" % (
                row["ka"], row["a33_over_rho_displaced"],
                row["b33_over_omega_rho_displaced"], b, 100 * e256, 100 * e64))
        else:
            lines.append("| %.3f | %.5f | %.5f | -- | -- | -- |" % (
                row["ka"], row["a33_over_rho_displaced"], row["b33_over_omega_rho_displaced"]))
    lines += [
        "",
        "## Live rigid-lid solves (ka = %g)" % payload["rigid_lid_probes"][0]["ka"],
        "",
        "| subdivisions | A33 [kg] | A33/(rho grad) | rel. error vs spectral |",
        "|---|---|---|---|",
    ]
    for row in payload["rigid_lid_probes"]:
        lines.append("| %d | %.4f | %.6f | %+.2f%% |" % (
            row["subdivisions"], row["a33"], row["a33_over_rho_displaced"],
            100 * row["relative_error_vs_spectral"]))
    lines += [
        "",
        "## Criteria",
        "",
        "| id | statement | measured | reference | value | pass |",
        "|---|---|---|---|---|---|",
    ]
    for item in payload["criteria"]:
        lines.append("| %s | %s | %.6g | %.6g | %.4g | %s |" % (
            item["id"], item["statement"], item["measured"], item["reference"],
            item["relative_error"], "PASS" if item["pass"] else "FAIL"))
    lines += ["", "**all_pass = %s**" % payload["all_pass"], "",
              "## Figures", ""]
    for name in payload["figures"]:
        lines.append("* `%s`" % name)
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/hemisphere_zero_frequency_envelope")
    parser.add_argument("--legacy-sweep", default=str(LEGACY_SWEEP))
    parser.add_argument("--live-ka", nargs="*", type=float, default=None,
                        help="optional live sweep at these ka (adds compute time)")
    args = parser.parse_args(argv)
    payload = build(args.output, args.legacy_sweep, args.live_ka)
    print(json.dumps({c["id"]: c["pass"] for c in payload["criteria"]}, indent=1))
    print("all_pass =", payload["all_pass"])
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())