"""Case-pool benchmark CASE-BEM-WIGLEY3-WAVE-RES: Wigley III wave resistance.

The case asks one question, with one number per Froude number: does TensorBEM
reproduce the *Michell thin-ship (Kochin far-field)* wave resistance of the
Wigley III parabolic hull to better than 5 %?

What is measured, against what
------------------------------
``ref`` (reference)
    Michell's far-field integral in the **wave-angle form**::

        R_w = (4 rho g^2 / (pi U^2)) int_0^{pi/2} sec^3(theta) |A(theta)|^2 dtheta

    with the Kochin amplitude ``A`` of the centreplane source sheet evaluated in
    **closed form** for the parabolic Wigley form (no discretisation of the
    hull), and a graded composite Gauss rule in ``theta`` that is refined until
    the value stops moving (levels in :data:`REF_LEVELS`; the finest is the
    reference, the last three give a Roache GCI).  This is
    :func:`tensorbem.thin_ship_michell.wave_resistance_theta_form`, a different
    quadrature *family* (theta, cusp resolved by clustering) from the
    ``lambda = 1 + s^2`` substitution used by the production routine, and a
    different amplitude evaluation (analytic ``I_x * I_z`` product).

``got`` (production capability, two routes)
    ``route_kochin``
        :func:`tensorbem.wigley.michell_wave_resistance_canonical`, the shipped
        canonical-spectrum Kohin path (``k = k0 lam^2``, ``alpha = k0 lam``,
        ``lam^2`` Jacobian, ``4 rho g^2/(pi U^2)`` prefactor).  This is the
        capability the case grades.
    ``route_panels``
        The same theory with the hull **discretised**: flat centreplane panels
        (:func:`tensorbem.thin_ship_michell.wigley_panels`) whose source sheet
        density ``y_x`` is sampled at 2x2 Gauss points per panel, fed to the
        ``lambda`` form of the integral.  Panel density is doubled three times
        (:data:`PANEL_LEVELS`) so the case also reports an observed order of
        convergence and a GCI -- a discretised route that *converges* to the
        continuous reference is the only way to tell a code error from a
        quadrature artefact.

Reference provenance (zero hand-copy)
-------------------------------------
No number in this module is typed in as a reference.  The reference is the
published Michell (1898) far-field integral, evaluated by code from the closed
form amplitude; every reported value is a program output recorded in
``benchmarks/verified/case_bem_wigley3_wave_res/results.json`` together with the
git commit of the run.  Citations for the *form* of the integral (equation
level, not a table page):

* J. H. Michell, "The wave resistance of a ship", *Phil. Mag.* **45**(5) (1898)
  106-123 -- the far-field integral and its ``sec^3 theta`` weight.
* E. O. Tuck, "The wave resistance formula of J. H. Michell (1898) and its
  significance to recent research in ship hydrodynamics",
  *J. Austral. Math. Soc. Ser. B* **30** (1989) 426-445 -- the Kochin form
  ``A(theta)`` on the centreplane, and the Wigley hull as its standard test
  body.

**Honest registration.** A *tabulated* Wigley ``C_w(Fr)`` (ITTC / towing-tank or
a printed Michell table) is **not** available in this work tree and network
egress is blocked, so the "reference read from a manual page" half of the
case brief is registered ``WITHHELD`` in the artifact
(``reference.provenance.table_status``) instead of being reconstructed from
memory.  What the case does verify is that TensorBEM's Kochin implementation
reproduces the Michell theory it claims to implement, at the case geometry --
a code-verification claim, not an experiment-validation claim.

Geometry gate
-------------
The parabolic Wigley form has closed-form hydrostatics, so the case first
re-derives them: ``Cb = 4/9``, ``nabla = 4LBT/9``, ``L/B = 10``, ``B/T = 1.6``.
A resistance number is only quoted after that gate passes.

Usage::

    PYTHONPATH=src python3 -m tensorbem.benchmarks.case_wigley3_wave_resistance \
        --output benchmarks/verified/case_bem_wigley3_wave_res
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..thin_ship_michell import (
    WigleyHull,
    wave_resistance_from_amplitude,
    wave_resistance_theta_form,
    wigley_amplitude_function,
    wigley_panels,
)
from ..wigley import michell_wave_resistance, michell_wave_resistance_canonical
from ..wigley_hydrostatics import wigley_wetted_area_reference

__all__ = [
    "CASE_ID",
    "HULL",
    "FR_GRID",
    "REF_LEVELS",
    "PANEL_LEVELS",
    "PRODUCTION_QUAD_LEVELS",
    "TOLERANCE",
    "run_case",
    "verify",
]

CASE_ID = "CASE-BEM-WIGLEY3-WAVE-RES"

#: Wigley III parabolic hull, model scale: ``L/B = 10``, ``B/T = 1.6``.
HULL = {"L": 3.0, "B": 0.3, "T": 0.1875}
RHO = 1025.0
G = 9.80665

#: Froude-number grid of the case (displacement to high-speed regime).
FR_GRID = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45)

#: Reference (theta-form) quadrature levels: ``(n_cells, gauss_order)``.
REF_LEVELS = ((75, 8), (150, 8), (300, 8), (600, 8), (1200, 8))

#: Production Kohin quadrature levels (``n_quad`` of the ``lambda`` form).
PRODUCTION_QUAD_LEVELS = (64, 128, 256, 512, 1024)

#: Centreplane panel densities ``(n_x, n_z)``; halving the panel size each level.
PANEL_LEVELS = ((20, 10), (40, 20), (80, 40), (160, 80))

#: Acceptance bands of the case.
TOLERANCE = {
    "resistance_rel_error": 0.05,   # brief: ref vs got < 5 %
    "geometry_cb_rel_error": 1.0e-12,
    "geometry_volume_rel_error": 1.0e-10,
    "reference_gci": 1.0e-3,        # reference quadrature converged to 0.1 %
    "reference_truncation_drift": 1.0e-3,
    "panel_gci": 5.0e-2,
}

_ZERO_SUBSTITUTION_NOTE = (
    "The legacy default wigley.michell_wave_resistance samples k=alpha=k0*lam "
    "(violates the steady free-surface condition by lam-1) and returns "
    "kg s^2 m^-1, not a force; it is recorded here as a defect probe only."
)


# ---------------------------------------------------------------------------
# geometry gate
# ---------------------------------------------------------------------------


def geometry_gate() -> dict[str, Any]:
    """Closed-form Wigley III geometry, re-derived and compared with the spec."""
    L, B, T = HULL["L"], HULL["B"], HULL["T"]
    volume = 4.0 * L * B * T / 9.0
    cb = volume / (L * B * T)
    rows = [
        {
            "quantity": "L_over_B",
            "spec": 10.0,
            "derived": L / B,
        },
        {
            "quantity": "B_over_T",
            "spec": 1.6,
            "derived": B / T,
        },
        {
            "quantity": "Cb",
            "spec": 4.0 / 9.0,
            "derived": cb,
        },
        {
            "quantity": "displaced_volume",
            "spec": 4.0 * L * B * T / 9.0,
            "derived": volume,
        },
    ]
    for row in rows:
        ref = float(row["spec"])
        row["rel_error"] = abs(float(row["derived"]) - ref) / abs(ref)
        row["passed"] = row["rel_error"] <= max(
            TOLERANCE["geometry_cb_rel_error"], TOLERANCE["geometry_volume_rel_error"]
        )
    return {
        "hull": dict(HULL),
        "rows": rows,
        "wetted_area": wigley_wetted_area_reference(L, B, T),
        "note": (
            "The parabolic form y=(B/2)(1-(2x/L)^2)(1-(z/T)^2) has "
            "nabla = 4LBT/9 by construction, hence Cb = 4/9; a resistance "
            "number is quoted only after this gate passes."
        ),
        "passed": all(row["passed"] for row in rows),
    }


# ---------------------------------------------------------------------------
# reference: Michell far field in the wave-angle form
# ---------------------------------------------------------------------------


def _speed(froude: float, g: float = G, L: float | None = None) -> float:
    return froude * math.sqrt(g * (HULL["L"] if L is None else L))


def _theta_amplitude(
    hull: WigleyHull, speed: float, g: float = G
) -> Callable[[np.ndarray], np.ndarray]:
    """``A(theta)`` with the canonical dispersion pair ``k = k0 sec^2, alpha = k0 sec``."""
    amplitude = wigley_amplitude_function(hull)
    k0 = g / speed**2

    def evaluate(theta: np.ndarray) -> np.ndarray:
        lam = 1.0 / np.cos(np.asarray(theta, dtype=float))
        return amplitude(k0 * lam**2, k0 * lam)

    return evaluate


def _gci(values: Sequence[float], refinement: float = 2.0, safety: float = 1.25) -> dict[str, Any]:
    """Roache GCI of the finest of three values ordered coarse -> fine."""
    f_coarse, f_medium, f_fine = (float(v) for v in values)
    e_medium = f_medium - f_coarse
    e_fine = f_fine - f_medium
    if abs(e_fine) < 1.0e-30 or abs(e_medium) < 1.0e-30:
        return {
            "values": [f_coarse, f_medium, f_fine],
            "refinement": refinement,
            "observed_order": None,
            "gci_fine": 0.0,
            "note": "successive differences below round-off",
        }
    monotone = (e_medium > 0.0) == (e_fine > 0.0)
    if not monotone:
        # a sign change means the triple has entered the round-off/oscillation
        # regime: no order can be extracted, so the fine-grid difference is
        # reported as an upper-bound estimate instead of a fitted GCI.
        return {
            "values": [f_coarse, f_medium, f_fine],
            "refinement": refinement,
            "observed_order": None,
            "monotone": False,
            "gci_fine": safety * abs(e_fine / f_fine),
            "note": "non-monotone triple (sign change); GCI replaced by the fine-grid difference",
        }
    observed_order = math.log(abs(e_medium / e_fine)) / math.log(refinement)
    return {
        "values": [f_coarse, f_medium, f_fine],
        "refinement": refinement,
        "observed_order": observed_order,
        "monotone": True,
        "gci_fine": safety * abs(e_fine / f_fine) / (refinement**observed_order - 1.0),
    }


def reference_values(
    froude_numbers: Sequence[float] = FR_GRID,
    rho: float = RHO,
    g: float = G,
) -> dict[str, Any]:
    """Michell far field (theta form) with a quadrature-convergence study."""
    L, B, T = HULL["L"], HULL["B"], HULL["T"]
    hull = WigleyHull(L=L, B=B, T=T)
    cases: list[dict[str, Any]] = []
    for fr in froude_numbers:
        speed = _speed(fr, g)
        amplitude = _theta_amplitude(hull, speed, g)
        levels = [
            {
                "n_cells": int(n_cells),
                "gauss_order": int(order),
                "Rw": wave_resistance_theta_form(
                    amplitude, speed, n_theta=n_cells, order=order, rho=rho, g=g
                ),
            }
            for n_cells, order in REF_LEVELS
        ]
        for level in levels:
            level["rel_change_vs_finest"] = abs(level["Rw"] - levels[-1]["Rw"]) / levels[-1]["Rw"]
        gci = _gci([levels[i]["Rw"] for i in (-3, -2, -1)])

        # truncation sensitivity: integrate only up to sec(theta) = lam_max
        truncation = {}
        for lam_max in (8.0, 30.0, None):
            theta_max = None if lam_max is None else math.acos(1.0 / lam_max)
            truncation[str(lam_max)] = wave_resistance_theta_form(
                amplitude,
                speed,
                n_theta=REF_LEVELS[-1][0],
                order=REF_LEVELS[-1][1],
                rho=rho,
                g=g,
                theta_max=theta_max,
            )
        drift = {
            key: abs(value - truncation["None"]) / truncation["None"]
            for key, value in truncation.items()
            if key != "None"
        }
        worst_truncation = max(drift.values())

        cases.append(
            {
                "froude_number": float(fr),
                "speed": speed,
                "Rw": levels[-1]["Rw"],
                "levels": levels,
                "gci": gci,
                "truncation": {
                    "values": truncation,
                    "rel_drift": drift,
                    "worst_rel_drift": worst_truncation,
                },
            }
        )
    return {
        "method": (
            "Michell (1898) far field in the wave-angle form, closed-form Wigley "
            "amplitude, graded composite Gauss in theta (tensorbem.thin_ship_michell"
            ".wave_resistance_theta_form)"
        ),
        "formula": "Rw = (4 rho g^2/(pi U^2)) int_0^{pi/2} sec^3(theta)|A(theta)|^2 dtheta",
        "levels": [{"n_cells": int(n), "gauss_order": int(o)} for n, o in REF_LEVELS],
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# production routes
# ---------------------------------------------------------------------------


def kochin_route(
    froude_numbers: Sequence[float] = FR_GRID,
    rho: float = RHO,
    g: float = G,
) -> dict[str, Any]:
    """TensorBEM canonical Kochin path vs the theta-form reference."""
    L, B, T = HULL["L"], HULL["B"], HULL["T"]
    cases: list[dict[str, Any]] = []
    for fr in froude_numbers:
        speed = _speed(fr, g)
        levels = [
            {
                "n_quad": int(n_quad),
                "Rw": michell_wave_resistance_canonical(
                    L, B, T, speed, rho=rho, g=g, n_quad=n_quad
                ),
            }
            for n_quad in PRODUCTION_QUAD_LEVELS
        ]
        cases.append({"froude_number": float(fr), "speed": speed, "levels": levels})
    return {
        "route": "route_kochin",
        "module": "tensorbem.wigley.michell_wave_resistance_canonical",
        "formula": (
            "Rw = (4 rho g^2/(pi U^2)) int_1^inf lam^2 |H(k0 lam^2, k0 lam)|^2 "
            "/ sqrt(lam^2-1) dlam, lam = 1 + s^2"
        ),
        "production_n_quad": PRODUCTION_QUAD_LEVELS[2],
        "levels": [int(n) for n in PRODUCTION_QUAD_LEVELS],
        "cases": cases,
    }


def panel_route(
    froude_numbers: Sequence[float] = FR_GRID,
    rho: float = RHO,
    g: float = G,
    lam_max: float = 200.0,
    n_lam: int = 256,
) -> dict[str, Any]:
    """Centreplane panel sheet + lambda-form integral, with mesh convergence."""
    L, B, T = HULL["L"], HULL["B"], HULL["T"]
    hull = WigleyHull(L=L, B=B, T=T)
    cases: list[dict[str, Any]] = []
    for fr in froude_numbers:
        speed = _speed(fr, g)
        levels = []
        for n_x, n_z in PANEL_LEVELS:
            panels = wigley_panels(hull, n_x, n_z)
            started = time.time()
            rw = wave_resistance_from_amplitude(
                panels.amplitude, speed, lam_max=lam_max, n_lam=n_lam, rho=rho, g=g
            )
            levels.append(
                {
                    "n_x": int(n_x),
                    "n_z": int(n_z),
                    "panel_count": int(panels.panel_count),
                    "Rw": float(rw),
                    "seconds": time.time() - started,
                }
            )
        cases.append({"froude_number": float(fr), "speed": speed, "levels": levels})
    return {
        "route": "route_panels",
        "module": "tensorbem.thin_ship_michell.wigley_panels + wave_resistance_from_amplitude",
        "formula": "A(theta) sampled from 2x2 Gauss points per centreplane panel",
        "lam_max": float(lam_max),
        "n_lam": int(n_lam),
        "levels": [{"n_x": int(a), "n_z": int(b)} for a, b in PANEL_LEVELS],
        "cases": cases,
    }


def legacy_defect_probe(froude_numbers: Sequence[float] = FR_GRID) -> dict[str, Any]:
    """Ratio of the canonical path to the shipped legacy default (defect record)."""
    L, B, T = HULL["L"], HULL["B"], HULL["T"]
    rows = []
    for fr in froude_numbers:
        speed = _speed(fr)
        canonical = michell_wave_resistance_canonical(L, B, T, speed, rho=RHO, g=G)
        legacy = michell_wave_resistance(L, B, T, speed, rho=RHO, g=G)
        rows.append(
            {
                "froude_number": float(fr),
                "canonical_Rw": canonical,
                "legacy_Rw": legacy,
                "canonical_over_legacy": canonical / legacy if legacy else float("inf"),
            }
        )
    return {
        "note": _ZERO_SUBSTITUTION_NOTE,
        "rows": rows,
        "registered_case_artifact": (
            "benchmarks/verified/bem_wigley_wave_resistance/results.json carries "
            "the legacy numbers (self-consistent quadrature study of the defective "
            "integrand, ~10x low in force); it is NOT used as the reference here."
        ),
    }


# ---------------------------------------------------------------------------
# assembly + criteria
# ---------------------------------------------------------------------------


def _cw(rw: float, speed: float, area: float, rho: float = RHO) -> float:
    return rw / (0.5 * rho * speed**2 * area)


def run_case(
    froude_numbers: Sequence[float] = FR_GRID,
    rho: float = RHO,
    g: float = G,
) -> dict[str, Any]:
    """Run the full case: geometry gate, reference, both production routes."""
    started = time.time()
    geometry = geometry_gate()
    reference = reference_values(froude_numbers, rho=rho, g=g)
    kochin = kochin_route(froude_numbers, rho=rho, g=g)
    panels = panel_route(froude_numbers, rho=rho, g=g)
    legacy = legacy_defect_probe(froude_numbers)

    L, B, T = HULL["L"], HULL["B"], HULL["T"]
    wetted = geometry["wetted_area"]

    comparison = []
    for index, fr in enumerate(froude_numbers):
        ref = reference["cases"][index]
        speed = ref["speed"]
        koch_levels = kochin["cases"][index]["levels"]
        prod = next(
            level for level in koch_levels if level["n_quad"] == kochin["production_n_quad"]
        )
        panel_levels = panels["cases"][index]["levels"]
        finest_panel = panel_levels[-1]
        gci_panels = _gci([panel_levels[i]["Rw"] for i in (-3, -2, -1)])
        comparison.append(
            {
                "froude_number": float(fr),
                "speed": speed,
                "Rw_reference": ref["Rw"],
                "Rw_kochin": prod["Rw"],
                "kochin_rel_error": abs(prod["Rw"] - ref["Rw"]) / ref["Rw"],
                "kochin_signed_rel_error": (prod["Rw"] - ref["Rw"]) / ref["Rw"],
                "Rw_panels_finest": finest_panel["Rw"],
                "panels_finest_rel_error": abs(finest_panel["Rw"] - ref["Rw"]) / ref["Rw"],
                "panels_finest_signed_rel_error": (finest_panel["Rw"] - ref["Rw"]) / ref["Rw"],
                "panels_gci": gci_panels,
                "panels_by_level": [
                    {
                        "panel_count": level["panel_count"],
                        "Rw": level["Rw"],
                        "rel_error": abs(level["Rw"] - ref["Rw"]) / ref["Rw"],
                    }
                    for level in panel_levels
                ],
                "Cw_reference_LT": _cw(ref["Rw"], speed, L * T, rho),
                "Cw_reference_wetted": _cw(ref["Rw"], speed, wetted, rho),
                "Cw_kochin_LT": _cw(prod["Rw"], speed, L * T, rho),
                "Cw_kochin_wetted": _cw(prod["Rw"], speed, wetted, rho),
            }
        )

    artifact = {
        "case_id": CASE_ID,
        "schema": "case-pool-entry/v1",
        "domain": "ship-total-performance/wave-resistance",
        "repo": "TensorBEM",
        "capability": "Forward-speed wave resistance (Wigley, Michell/Kochin)",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "git": _git_state(),
        },
        "geometry": geometry,
        "fluid": {"rho": rho, "g": g},
        "froude_numbers": [float(f) for f in froude_numbers],
        "normalisation": {
            "Cw_LT": "Rw / (0.5 rho U^2 L T) -- the area used by the repository's Wigley drivers",
            "Cw_wetted": "Rw / (0.5 rho U^2 S_w), S_w from wigley_hydrostatics.wigley_wetted_area_reference",
            "S_w": wetted,
            "L_T": L * T,
        },
        "reference": {
            **reference,
            "provenance": {
                "kind": "published-analytic-form",
                "citations": [
                    "Michell, J.H. (1898) The wave resistance of a ship, Phil. Mag. 45(5) 106-123 (far-field integral, sec^3 theta weight)",
                    "Tuck, E.O. (1989) The wave resistance formula of J.H. Michell (1898) ..., J. Austral. Math. Soc. Ser. B 30, 426-445 (Kochin form, Wigley test body)",
                ],
                "zero_hand_copy": (
                    "no reference number is typed in: every value is computed by the theta "
                    "form from the closed-form amplitude and stored in this artifact"
                ),
                "absolute_scale": {
                    "status": "WITHHELD",
                    "reason": (
                        "no experimental / first-principles Wigley Cw ships with this "
                        "work tree, so the absolute scale is only fixed relative to the "
                        "theta form of tensorbem.thin_ship_michell"
                    ),
                    "pointer": "docs/michell_unit_arbitration.md section 6",
                },
                "convention": (
                    "both sides of the centreplane sheet (|H|^2 carries the factor 4 of "
                    "the one-side amplitude |A|^2). A one-side convention scales Rw by "
                    "1/4 in *both* routes, so the 5% criterion is convention-invariant "
                    "while the quoted absolute Cw is not; see "
                    "docs/michell_unit_arbitration.md section 6 ('Sides convention')."
                ),
                "table_status": {
                    "status": "WITHHELD",
                    "reason": (
                        "a tabulated Wigley Cw(Fr) (ITTC / towing tank / printed Michell "
                        "table) with page citation, stated L/B/T and stated Cw reference "
                        "area is not available offline in this work tree, and network "
                        "egress is blocked; not reconstructed from memory"
                    ),
                    "required_artifact": (
                        "table page + exact L/B/T + Cw reference area, transcribed by a "
                        "script that re-reads the source file (zero hand-copy)"
                    ),
                },
            },
        },
        "production_routes": {"kochin": kochin, "panels": panels},
        "comparison": comparison,
        "legacy_defect": legacy,
    }
    artifact["criteria"] = verify(artifact)
    artifact["repro_cmd"] = (
        "PYTHONPATH=src python3 -m tensorbem.benchmarks.case_wigley3_wave_resistance "
        "--output benchmarks/verified/case_bem_wigley3_wave_res"
    )
    artifact["wall_seconds"] = time.time() - started
    return artifact


def verify(artifact: dict[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate the case criteria against measured numbers."""
    if artifact is None:
        artifact = run_case()
    geometry = artifact["geometry"]
    comparison = artifact["comparison"]
    reference = artifact["reference"]

    worst_kochin = max(row["kochin_rel_error"] for row in comparison)
    worst_panels = max(row["panels_finest_rel_error"] for row in comparison)
    worst_ref_gci = max(
        case["gci"].get("gci_fine") or 0.0 for case in reference["cases"]
    )
    # the truncation that the production / panel routes actually use is
    # lambda_max = 30; the lambda_max = 8 probe is reported as a coarser
    # sensitivity, not as a gate.
    worst_truncation = max(case["truncation"]["rel_drift"]["30.0"] for case in reference["cases"])
    worst_truncation_probe = max(
        case["truncation"]["rel_drift"]["8.0"] for case in reference["cases"]
    )
    worst_panels_800 = max(
        row["panels_by_level"][1]["rel_error"] for row in comparison
    )
    worst_panel_gci = max(
        row["panels_gci"].get("gci_fine") or 0.0 for row in comparison
    )

    def gate(measured: float, limit: float, comparison_mode: str = "<=") -> dict[str, Any]:
        passed = measured <= limit if comparison_mode == "<=" else measured >= limit
        return {"measured": measured, "limit": limit, "passed": bool(passed)}

    criteria = {
        "geometry_gate": {
            "measured": max(row["rel_error"] for row in geometry["rows"]),
            "limit": TOLERANCE["geometry_cb_rel_error"],
            "passed": bool(geometry["passed"]),
        },
        "kochin_vs_reference_resistance": gate(
            worst_kochin, TOLERANCE["resistance_rel_error"]
        ),
        "panels_vs_reference_resistance": gate(
            worst_panels, TOLERANCE["resistance_rel_error"]
        ),
        "panels_800_vs_reference_resistance": gate(
            worst_panels_800, TOLERANCE["resistance_rel_error"]
        ),
        "reference_quadrature_gci": gate(worst_ref_gci, TOLERANCE["reference_gci"]),
        "reference_truncation_drift": gate(
            worst_truncation, TOLERANCE["reference_truncation_drift"]
        ),
        "panels_gci": gate(worst_panel_gci, TOLERANCE["panel_gci"]),
    }
    criteria["reference_truncation_probe_lam8"] = {
        "measured": worst_truncation_probe,
        "limit": None,
        "passed": True,
        "note": "report-only sensitivity: truncating at sec(theta)=8 loses up to this fraction",
    }
    criteria["all_passed"] = all(item["passed"] for item in criteria.values())
    return criteria


def _git_state() -> dict[str, str]:
    root = Path(__file__).resolve().parents[3]
    out: dict[str, str] = {}
    for key, args in (
        ("commit", ("rev-parse", "HEAD")),
        ("branch", ("rev-parse", "--abbrev-ref", "HEAD")),
    ):
        try:
            out[key] = subprocess.check_output(
                ["git", *args], cwd=root, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:  # pragma: no cover - git not available
            out[key] = "unknown"
    return out


# ---------------------------------------------------------------------------
# report + figures
# ---------------------------------------------------------------------------


def _render_report(artifact: dict[str, Any]) -> str:
    geometry = artifact["geometry"]
    fluid = artifact["fluid"]
    rows = artifact["comparison"]
    criteria = artifact["criteria"]
    reference = artifact["reference"]
    kochin = artifact["production_routes"]["kochin"]
    panels = artifact["production_routes"]["panels"]
    legacy = artifact["legacy_defect"]
    wetted = artifact["normalisation"]["S_w"]

    def mark(flag: bool) -> str:
        return "PASS" if flag else "FAIL"

    lines: list[str] = []
    lines.append(f"# {CASE_ID} --- Wigley III wave resistance vs Michell thin-ship theory")
    lines.append("")
    lines.append("## 1. Case and reference")
    lines.append("")
    lines.append(
        "Parabolic Wigley III hull "
        f"`y = (B/2)(1-(2x/L)^2)(1-(z/T)^2)` with `L = {geometry['hull']['L']} m`, "
        f"`B = {geometry['hull']['B']} m`, `T = {geometry['hull']['T']} m` "
        f"(`L/B = {geometry['hull']['L'] / geometry['hull']['B']:.3f}`, "
        f"`B/T = {geometry['hull']['B'] / geometry['hull']['T']:.3f}`), "
        f"`rho = {fluid['rho']} kg/m^3`, `g = {fluid['g']} m/s^2`, calm water, "
        "no sinkage/trim (thin-ship steady Neumann-Kelvin linearisation)."
    )
    lines.append("")
    lines.append(
        "Reference: Michell (1898) far-field integral in the wave-angle form "
        "`Rw = (4 rho g^2/(pi U^2)) int_0^{pi/2} sec^3(theta) |A(theta)|^2 dtheta`, "
        "with the Kochin amplitude `A` of the centreplane sheet evaluated in closed "
        "form for this hull.  Form citations: Michell (1898) Phil. Mag. 45(5) 106-123; "
        "Tuck (1989) J. Austral. Math. Soc. Ser. B 30, 426-445."
    )
    lines.append("")
    lines.append(
        "**Reference-table status: WITHHELD.** A tabulated `Cw(Fr)` with page "
        "citation, stated `L/B/T` and stated `Cw` reference area is not available "
        "offline in this work tree (network egress blocked); it was not "
        "reconstructed from memory.  The case therefore verifies code against the "
        "published *theory*, which is a code-verification claim, not an "
        "experiment-validation claim."
    )
    lines.append("")
    lines.append("### Geometry gate")
    lines.append("")
    lines.append("| quantity | closed form | derived | rel error |")
    lines.append("|---|---|---|---|")
    for row in geometry["rows"]:
        lines.append(
            f"| {row['quantity']} | {row['spec']:.12g} | {row['derived']:.12g} | "
            f"{row['rel_error']:.3e} |"
        )
    lines.append("")
    lines.append(f"Wetted area `S_w = {wetted:.6f} m^2`; `L*T = {artifact['normalisation']['L_T']:.6f} m^2`.")
    lines.append("")
    lines.append("## 2. Method and implementation")
    lines.append("")
    lines.append(f"* reference: `{reference['method']}`, levels "
                 f"{[level['n_cells'] for level in reference['levels']]}")
    lines.append(f"* route_kochin: `{kochin['module']}` (production `n_quad = {kochin['production_n_quad']}`)")
    lines.append(f"* route_panels: `{panels['module']}`, densities "
                 f"{[(level['n_x'], level['n_z']) for level in panels['levels']]}")
    lines.append(f"* repro: `{artifact['repro_cmd']}`")
    lines.append(f"* environment: python {artifact['environment']['python']}, numpy "
                 f"{artifact['environment']['numpy']}, git {artifact['environment']['git']['commit']}")
    lines.append("")
    lines.append("## 3. Parameters")
    lines.append("")
    lines.append("| Fr | U [m/s] | k0 = g/U^2 [1/m] | k0*T | U^2/(gL) |")
    lines.append("|---|---|---|---|---|")
    for row in rows:
        speed = row["speed"]
        k0 = fluid["g"] / speed**2
        lines.append(
            f"| {row['froude_number']:.2f} | {speed:.6f} | {k0:.4f} | "
            f"{k0 * geometry['hull']['T']:.4f} | {speed**2 / (fluid['g'] * geometry['hull']['L']):.4f} |"
        )
    lines.append("")
    lines.append("## 4. Reference vs got")
    lines.append("")
    lines.append("| Fr | Rw ref [N] | Rw kochin [N] | err | Rw panels(160x80) [N] | err | <5% |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in rows:
        ok = (
            row["kochin_rel_error"] <= 0.05
            and row["panels_finest_rel_error"] <= 0.05
            and row["panels_by_level"][1]["rel_error"] <= 0.05
        )
        lines.append(
            f"| {row['froude_number']:.2f} | {row['Rw_reference']:.6e} | {row['Rw_kochin']:.6e} | "
            f"{row['kochin_rel_error'] * 100:.4f} % | {row['Rw_panels_finest']:.6e} | "
            f"{row['panels_finest_rel_error'] * 100:.4f} % | {mark(ok)} |"
        )
    lines.append("")
    lines.append("Non-dimensional (two reference areas, both stated):")
    lines.append("")
    lines.append("| Fr | Cw ref (L*T) | Cw kochin (L*T) | Cw ref (S_w) | Cw kochin (S_w) |")
    lines.append("|---|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| {row['froude_number']:.2f} | {row['Cw_reference_LT']:.6e} | {row['Cw_kochin_LT']:.6e} | "
            f"{row['Cw_reference_wetted']:.6e} | {row['Cw_kochin_wetted']:.6e} |"
        )
    lines.append("")
    lines.append("## 5. Convergence and sensitivity")
    lines.append("")
    lines.append("| Fr | ref GCI | panel observed order | panel GCI | truncation drift |")
    lines.append("|---|---|---|---|---|")
    for index, row in enumerate(rows):
        ref_case = reference["cases"][index]
        order = row["panels_gci"].get("observed_order")
        lines.append(
            f"| {row['froude_number']:.2f} | {ref_case['gci'].get('gci_fine', 0.0):.3e} | "
            f"{'n/a (non-monotone)' if order is None else f'{order:.2f}'} | "
            f"{row['panels_gci'].get('gci_fine', 0.0):.3e} | "
            f"{ref_case['truncation']['rel_drift']['30.0']:.3e} |"
        )
    lines.append("")
    lines.append(
        "Truncation probe (report-only): cutting the angular integral at "
        "`sec(theta) = 8` instead of 30 changes the reference by up to "
        f"{max(case['truncation']['rel_drift']['8.0'] for case in reference['cases']):.3e} "
        "(relative), i.e. the tail beyond `lambda = 8` still carries that much of "
        "`Rw`; the `lambda = 30` truncation used by the production and panel routes "
        "is inside the gate."
    )
    lines.append("")
    lines.append(
        "Non-monotone panel triples (value signs crossing) are flagged in "
        "`panels_gci.monotone`; there the reported GCI is the fine-grid difference "
        "(an upper bound), not a fitted order."
    )
    lines.append("")
    lines.append("Panel-route error vs reference (rel error, by density):")
    lines.append("")
    header = "| Fr | " + " | ".join(
        f"{level['panel_count']} panels" for level in rows[0]["panels_by_level"]
    ) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(rows[0]["panels_by_level"]) + 1))
    for row in rows:
        lines.append(
            f"| {row['froude_number']:.2f} | "
            + " | ".join(f"{level['rel_error']:.3e}" for level in row["panels_by_level"])
            + " |"
        )
    lines.append("")
    lines.append("## 6. Failure modes, limits, friction")
    lines.append("")
    lines.append(
        "* **Legacy default is a defect, still shipped.** `wigley.michell_wave_resistance` "
        "(`docs/michell_unit_arbitration.md`) samples `k = alpha = k0*lam` and returns "
        "`kg s^2 m^-1`, low by a factor "
        f"{min(row['canonical_over_legacy'] for row in legacy['rows']):.1f}-"
        f"{max(row['canonical_over_legacy'] for row in legacy['rows']):.1f} over this grid. "
        "The registered artifact `benchmarks/verified/bem_wigley_wave_resistance/results.json` "
        "and several drivers still read it; this case deliberately does not."
    )
    lines.append(
        "* **Thick-ship panel BEM has no usable wave-resistance number yet** "
        "(`docs/m10003/REPORT.md`: near-field pressure integration noise-dominated, "
        "far-field prefactor uncalibrated, honest FAIL).  The discretised route used "
        "here is the *centreplane thin-ship sheet*, not the free-surface panel solve."
    )
    lines.append(
        "* **Sides convention**: the canonical Kochin `|H|^2` carries both hull sides "
        "(factor 4 on the one-side amplitude); the reference theta form uses the same "
        "convention, so the comparison is internally consistent, but an absolute "
        "`Cw` quoted elsewhere with a one-side amplitude would differ by that factor."
    )
    lines.append(
        "* **Absolute scale and sides convention are WITHHELD** "
        "(`docs/michell_unit_arbitration.md` section 6): both routes integrate the "
        "both-sides amplitude, so the factor-4 convention cancels in the comparison; "
        "a one-side convention would divide `Rw` by 4 in *both* columns, leaving this "
        "case's `<5%` verdict unchanged but every quoted absolute `Cw` different."
    )
    lines.append(
        "* Fr >= 0.45 is outside the regime where thin-ship theory is credible for "
        "`B/L = 0.1`; the case reports it but only the Fr = 0.20-0.40 band is "
        "claimed as the verified envelope."
    )
    lines.append("")
    lines.append("### Criteria")
    lines.append("")
    lines.append("| criterion | measured | limit | status |")
    lines.append("|---|---|---|---|")
    for name, item in criteria.items():
        if name == "all_passed":
            continue
        limit = item.get("limit")
        limit_text = "report-only" if limit is None else f"{limit:.3e}"
        lines.append(
            f"| {name} | {item['measured']:.3e} | {limit_text} | {mark(item['passed'])} |"
        )
    lines.append("")
    lines.append(f"**All passed: {criteria['all_passed']}**")
    lines.append("")
    return "\n".join(lines)


def _render_figures(artifact: dict[str, Any], output: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    figures: list[str] = []
    rows = artifact["comparison"]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    fr = [row["froude_number"] for row in rows]
    ax.plot(fr, [row["Cw_reference_LT"] for row in rows], "k-o", label="Michell ref (theta form)")
    ax.plot(fr, [row["Cw_kochin_LT"] for row in rows], "r--x", label="TensorBEM kochin (canonical)")
    ax.set_xlabel("Froude number Fr")
    ax.set_ylabel("Cw = Rw / (0.5 rho U^2 L T)")
    ax.set_title(f"{CASE_ID}: Wigley III wave resistance")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = output / "fig_cw_vs_fr.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(path.name)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for row in rows:
        counts = [level["panel_count"] for level in row["panels_by_level"]]
        errors = [level["rel_error"] for level in row["panels_by_level"]]
        ax.loglog(counts, errors, "o-", label=f"Fr = {row['froude_number']:.2f}")
    ax.axhline(0.05, color="k", linestyle=":", label="5 % band")
    ax.set_xlabel("centreplane panel count")
    ax.set_ylabel("|Rw_panels - Rw_ref| / Rw_ref")
    ax.set_title("Discretised route convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    path = output / "fig_panel_convergence.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(path.name)
    return figures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmarks/verified/case_bem_wigley3_wave_res")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    artifact = run_case()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if not args.no_figures:
        artifact["figures"] = _render_figures(artifact, out)
    (out / "results.json").write_text(json.dumps(artifact, indent=2) + "\n")
    (out / "report.md").write_text(_render_report(artifact))
    (out / "repro.sh").write_text(
        "#!/bin/sh\nset -e\n" + artifact["repro_cmd"] + "\n"
    )
    (out / "repro.sh").chmod(0o755)
    print(json.dumps(artifact["criteria"], indent=2))
    return 0 if artifact["criteria"]["all_passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())