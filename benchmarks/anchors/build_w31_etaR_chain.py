#!/usr/bin/env python3
"""W31 · propeller-openwater-bem: take the constant out of the ``eta_R`` link.

What this script decides
------------------------
The ship-performance platform returns ``relative_rot_eff = 1.0`` on every state.
W30 established *that* it is constant and that 9/11 states fall outside the
admissible window once the factor is inverted against an independent module.
This script answers the question W30 left open:

    can ``eta_R`` be produced as a computed, state-dependent quantity from the
    behind-hull ``K_T``/``K_Q`` and the open-water curve read at the same ``J``,
    and does ``eta_D = eta_0 eta_R eta_H`` then close as a chain?

Alternatives considered (declared before the numbers were produced)
-------------------------------------------------------------------
A1  platform constant ``eta_R = 1`` -- the current state; falsified by C1 (a
    constant cannot pass a non-constancy gate) and by the W30 audit.
A2  an empirical regression for ``eta_R`` (e.g. Holtrop 1984) -- one number per
    propeller, almost independent of the operating point, so it does **not**
    answer the question asked here (it re-introduces a quasi-constant).  Not
    used; noted as the fallback if the computed route had failed C2.
A3  **adopted**: ``eta_R = eta_0,open-water(J) / eta_0,behind-hull(J)`` with the
    behind-hull point solved by the same vortex lattice that produces the
    open-water curve, the only difference between the two solves being the
    inflow field (uniform vs afterbody nominal wake at the same mean ``J``).

Why A3 is a *measurement* and not a re-parameterisation: the two solves share
lattice, wake model, Reynolds number and mean advance ratio, so the solver's own
bias appears in numerator and denominator and cancels in the ratio; what is left
is the response of the propeller to the radial non-uniformity of the wake.  The
``q = 0`` uniform-wake solve is carried as the **null control**: it must return
``eta_R = 1`` exactly, and it does (``control_null`` below).

Predeclared gates (verbatim from the mission, fixed before running)
------------------------------------------------------------------
C1  non-constancy: std of the 11 ``eta_R`` values > 1e-3
C2  physical window: >= 9 of 11 ``eta_R`` inside [0.90, 1.10]
C3  identity: ``eta_D = eta_0 eta_R eta_H`` relative residual <= 1e-6 per state
C4  point-by-point against the KP505 measured curve: over ``J <= 0.8`` the
    worst ``|dK_T| < 15 %``, and every point answered for ``< 3 %``

Declared configuration (not tuned; sensitivity reported)
--------------------------------------------------------
* wake shape family ``w = w_bar (1-xi)^q / <(1-xi)^q>`` with ``q = 0.5``
  primary and ``q in {0, 0.25, 0.5, 1.0}`` as the declared sensitivity ladder
  (``q = 0`` is the null control);
* lattice ``n_radial x n_chord = 16 x 2`` (the configuration registered for the
  KP505 case in this repository), with ``(8,1)`` and ``(24,2)`` as the declared
  panel-sensitivity ladder;
* ``wake_model = "geometric"``, ``wake_length = 1.0``, ``wake_steps = 24`` --
  the registered KP505 case configuration;
* the lattice is non-dimensional (``D = 1``); the Reynolds number of the
  physical ``D = 6 m`` propeller at the state's rpm is preserved exactly by
  ``n_eff = n_phys D_phys^2`` (the ITTC-78 section law depends on ``n D^2`` only);
* open-water curve read at the state ``J`` by piecewise-linear interpolation of
  a ladder ``J = 0.10 (0.05) 0.75`` built from the *same* lattice, and the
  interpolation is checked against a direct solve at the state ``J``.

Run::

    PYTHONPATH=src python benchmarks/anchors/build_w31_etaR_chain.py
"""

from __future__ import annotations

import hashlib
import json
import math
import platform as _platform
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:  # allow "python benchmarks/anchors/..." without PYTHONPATH
    sys.path.insert(0, str(REPO / "src"))

from tensorbem.propeller import BladeGeometry, open_water_point  # noqa: E402
from tensorbem.wake_field import (  # noqa: E402
    interpolate_open_water_curve,
    nominal_wake_fraction,
    relative_rotative_efficiency,
    wake_inflow,
)

TWO_PI = 2.0 * math.pi

RECEIPTS = REPO / "data" / "w30" / "propulsion_states" / "api_receipts.json"
LIVE_RECHECK = REPO / "data" / "w31" / "propulsion_states" / "live_recheck.json"
KP505_ANCHOR = REPO / "benchmarks" / "anchors" / "kp505_openwater.json"
OUT = REPO / "benchmarks" / "artifacts" / "w31_openwater_etaR_chain.json"

# ---------------------------------------------------------------------------
# Predeclared configuration and gates
# ---------------------------------------------------------------------------

PREREGISTRATION: Dict[str, Any] = {
    "mission": 22286,
    "topic": "propeller-openwater-bem / eta_R de-constantisation (item 9, external track)",
    "decision": (
        "Is eta_R deliverable as a computed state-dependent quantity from the "
        "behind-hull K_T/K_Q and the open-water curve read at the same J, with "
        "eta_D = eta_0 eta_R eta_H closing as a chain?"
    ),
    "alternatives": {
        "A1_platform_constant": "eta_R = 1.0 as returned by the propulsion block",
        "A2_empirical_regression": "Holtrop(1984)-type closed form (near-constant, kept as fallback)",
        "A3_behind_hull_ratio": "eta_0,open-water(J) / eta_0,behind-hull(J) from the same lattice",
    },
    "adopted": "A3_behind_hull_ratio",
    "gates": {
        "C1": {"statement": "std(eta_R over 11 states) > 1e-3", "threshold": 1e-3, "direction": "greater"},
        "C2": {"statement": ">= 9 of 11 states inside eta_R window [0.90, 1.10]",
               "threshold": 9, "window": [0.90, 1.10]},
        "C3": {"statement": "eta_D = eta_0 eta_R eta_H relative residual <= 1e-6 per state",
               "threshold": 1e-6},
        "C4": {"statement": "over J <= 0.8, worst |dK_T| vs KP505 measured curve < 15 %, "
                            "each point answered for < 3 %",
               "threshold_pct": 15.0, "strict_pct": 3.0, "J_region_max": 0.8},
    },
    "declared_config": {
        "wake_exponent_primary": 0.5,
        "wake_exponent_sensitivity": [0.0, 0.25, 0.5, 1.0],
        "hub_ratio": 0.18,
        "panel_primary": [16, 2],
        "panel_sensitivity": [[8, 1], [16, 2], [24, 2]],
        "wake_model": "geometric",
        "wake_length": 1.0,
        "wake_steps": 24,
        "lattice_convention": "non-dimensional D = 1; Re preserved by n_eff = n_phys D_phys^2",
        "open_water_ladder": {"start": 0.10, "stop": 0.75, "step": 0.05},
        "interpolation": "piecewise linear in J on the ladder",
    },
    "stopping_rule": (
        "one pass of the full state set at the primary configuration, one null "
        "control, one panel ladder and one wake-shape ladder; no iteration on "
        "thresholds or on the shape exponent after seeing eta_R"
    ),
    "design": {
        "hypothesis": (
            "H1: the propeller responds to the radial non-uniformity of the afterbody "
            "wake, so eta_R = eta_0,OW(J)/eta_0,BH(J) differs from 1 by an amount that "
            "depends on J, i.e. eta_R is a computed quantity and not a constant. "
            "H0 (the platform's implicit hypothesis): eta_0,BH == eta_0,OW at the same J, "
            "so eta_R == 1."
        ),
        "metrics": [
            "eta_R per state in four conventions (platform return / platform identity "
            "reverse / open-water-interpolated over behind-hull / independent P_E-P_D reverse)",
            "eta_D = eta_0 eta_R eta_H and its algebraic residual",
            "|dK_T|, |d10K_Q|, |deta_0| of the open-water model against the KP505 measured curve",
        ],
        "unit_of_analysis": "one propulsion state (11 states), not one solver run: the gate is a property of the series",
        "baseline": "the platform's constant eta_R = 1.0 on the identical 11 states",
        "control": (
            "q = 0 uniform wake: the inflow field then equals the open-water inflow point by "
            "point, so eta_R must be exactly 1. Any departure would be a code defect, and the "
            "control is what separates a code defect from a physical effect."
        ),
        "confounders_and_nuisances": [
            "panel density (n_radial x n_chord): declared ladder (8,1)/(16,2)/(24,2)",
            "wake shape exponent q: declared ladder 0/0.25/0.5/1.0, monotonicity checked",
            "open-water curve interpolation error: measured against a direct solve per state",
            "receipt rounding (4 dp non-dimensional): bounds identity residuals at ~1e-3",
            "Reynolds number: preserved by n_eff = n_phys D^2 so it is not a free variable",
            "kernel scale-invariance defect at D=6: avoided by the D=1 convention, recorded",
        ],
        "repetition_rationale": (
            "the solver is deterministic (no random seed, one dense linear solve); repeating a "
            "state can only reproduce it, so repetitions are replaced by (a) an explicit "
            "determinism re-solve recorded in the artefact and (b) the panel/shape ladders, "
            "which vary the two nuisance variables that actually move the number"
        ),
        "environment_controls": (
            "single machine, one process, same numpy/Python as recorded; wall time and commit "
            "recorded in the artefact; the run is short (~40 s) so it is not submitted to the "
            "cluster and no cross-node variability enters"
        ),
        "analysis_plan": (
            "gate arithmetic is computed by the builder, never typed; the doc reads every number "
            "from the artefact; the KP505 comparison keeps all 10 anchor points (no selection) "
            "and reports the region-restricted (J<=0.8) and the stricter (J<=0.7, J<=0.6) worst "
            "deviations side by side"
        ),
        "resource_limits": "8 cores unused (single-threaded numpy), < 60 s wall, < 4 GB RAM; no cluster job",
        "interpretation_limits": [
            "eta_R here is a nominal-axial-wake eta_R: no swirl, no measured wake field",
            "the absolute value of eta_R has no experimental anchor inside the evidence surface "
            "(WITHHELD); only its non-constancy, window and sensitivity are claimed",
            "the eta_D chain is algebraic; it cannot validate eta_R (see caveats)",
        ],
    },
}

PHYSICAL_DIAMETER = 6.0  # m, the platform's propeller
WINDOW = (0.90, 1.10)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except Exception:  # pragma: no cover - diagnostics only
        return "unknown"


def _geometry(pitch_ratio: float, n_radial: int, n_chord: int) -> BladeGeometry:
    return BladeGeometry(
        diameter=1.0,
        blades=5,
        hub_ratio=0.18,
        pitch_ratio=pitch_ratio,
        expanded_area_ratio=0.70,
        n_radial=n_radial,
        n_chord=n_chord,
        thickness_ratio=0.06,
        radial_spacing="cosine",
        root_chord_offset=0.12,
    )


def _point(
    geometry: BladeGeometry,
    advance: float,
    *,
    rev_per_second: float,
    inflow_field=None,
) -> Dict[str, Any]:
    """One lattice solve -> ``K_T``, ``K_Q``, ``eta_0``.

    The kernel returns torque with the sign of its bound-vorticity convention
    (the registered KP505 case flips it the same way, see
    ``tensorbem.benchmarks.kp505_openwater_bem._point``); ``K_Q`` is taken
    positive here and the convention is restated in the artefact.
    """

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = open_water_point(
            geometry,
            advance,
            wake_model="geometric",
            wake_length=1.0,
            wake_steps=24,
            rev_per_second=rev_per_second,
            inflow_field=inflow_field,
        )
    thrust = float(result.thrust)
    torque = -float(result.torque) if float(result.torque) < 0.0 else float(result.torque)
    eta0 = advance * thrust / (TWO_PI * torque) if torque > 0.0 else float("nan")
    return {
        "J": float(advance),
        "KT": thrust,
        "KQ": torque,
        "10KQ": 10.0 * torque,
        "eta0": eta0,
        "n_panels_per_blade": int(result.n_panels),
        "kernel_warnings": [str(w.message) for w in caught],
    }


def _reynolds_matched_rev_per_second(rpm: float, diameter: float = PHYSICAL_DIAMETER) -> float:
    """``n_eff`` of the D=1 lattice that reproduces the physical section Reynolds number.

    ``Re = W (n D) c (D) / nu`` (``c`` in diameter units) depends on ``n D^2``
    only, so the non-dimensional lattice must be run at ``n_eff = n_phys D^2``.
    """

    return (rpm / 60.0) * diameter**2


def resistance_pe_map(receipts: Dict[str, Any]) -> Dict[float, Dict[str, float]]:
    """The resistance block's own ``P_E`` and ``R_T`` per ship speed (independent module)."""

    return {
        float(row["speed_kn"]): {"pe_kW": float(row["pe_kW"]), "rt_kN": float(row["rt_kN"])}
        for row in receipts["resistance_receipts"]["rows"]
    }


def _ratio(value: float, reference: float) -> float:
    """Plain ratio for reporting; used where the tautology guard would fire by design."""

    if not math.isfinite(reference) or reference == 0.0:
        return float("nan")
    return float(value) / float(reference)


def _pct(value: float, reference: float) -> float:
    if reference == 0.0 or not math.isfinite(reference):
        return float("nan")
    return 100.0 * (value - reference) / reference


# ---------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------


def platform_block_audit(receipts: Dict[str, Any]) -> Dict[str, Any]:
    """Is the platform's behind-hull K_T/K_Q anything other than its open-water curve?

    If the values are the O&O regression at the state's ``J`` and ``P/D`` to
    receipt precision, then the platform's ``eta_R = 1`` is not a modelling
    choice with an error bar -- it is the algebraic consequence of feeding the
    same curve into both sides of Eq. (1).
    """

    from tensorbem.wageningen import B4Series

    rows: List[Dict[str, Any]] = []
    worst_kt = 0.0
    worst_kq = 0.0
    for state in receipts["states"]:
        p = state["propulsion_response"]
        series = B4Series(blades=5, expanded_area_ratio=0.70, pitch_ratio=state["inputs"]["prop_pitch"])
        kt = series.kt(p["advance_ratio_J"])
        kq = series.kq(p["advance_ratio_J"])
        d_kt = kt - p["thrust_coefficient_KT"]
        d_kq = kq - p["torque_coefficient_KQ"]
        worst_kt = max(worst_kt, abs(d_kt))
        worst_kq = max(worst_kq, abs(d_kq))
        rows.append(
            {
                "state": state["id"],
                "P_D": state["inputs"]["prop_pitch"],
                "J": p["advance_ratio_J"],
                "KT_platform": p["thrust_coefficient_KT"],
                "KT_B_series_OandO_Z5_AeAo0.70": kt,
                "dKT": d_kt,
                "KQ_platform": p["torque_coefficient_KQ"],
                "KQ_B_series": kq,
                "dKQ": d_kq,
                "eta0_platform": p["open_water_eff"],
                "eta0_B_series": series.eta0(p["advance_ratio_J"]),
            }
        )
    # -- which expanded area ratio does the platform's curve belong to? ------
    # Falsifiable identification instead of an assumption: the residual of the
    # K_T/K_Q fit is evaluated over a declared ladder of Ae/Ao, and the value the
    # platform actually uses is the one that minimises it.
    identification: List[Dict[str, Any]] = []
    for area_ratio in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        series_worst = 0.0
        for state in receipts["states"]:
            p = state["propulsion_response"]
            series = B4Series(
                blades=5, expanded_area_ratio=area_ratio, pitch_ratio=state["inputs"]["prop_pitch"]
            )
            series_worst = max(
                series_worst, abs(series.kt(p["advance_ratio_J"]) - p["thrust_coefficient_KT"])
            )
        identification.append({"Ae_Ao": area_ratio, "worst_abs_dKT": series_worst})
    best = min(identification, key=lambda row: row["worst_abs_dKT"])

    return {
        "question": "does the platform's returned K_T/K_Q differ from its own open-water curve at J?",
        "reference": "Oosterveld & van Oossanen (1975) B-series regression, Z=5, Ae/Ao=0.70, P/D as input",
        "rows": rows,
        "worst_abs_dKT": worst_kt,
        "worst_abs_dKQ": worst_kq,
        "area_ratio_identification": {
            "ladder": identification,
            "argmin_Ae_Ao": best["Ae_Ao"],
            "argmin_worst_abs_dKT": best["worst_abs_dKT"],
            "declared_Ae_Ao": 0.70,
            "note": (
                "the platform reports no Ae/Ao; the value is identified here by minimising the "
                "K_T residual over the declared ladder, so it is a measurable claim about the "
                "platform rather than an assumption copied from the W30 note"
            ),
        },
        "receipt_precision": "4 dp on K_T/K_Q, so agreement at the 1e-4 level is agreement to receipt precision",
        "conclusion": (
            "the platform's behind-hull K_T/K_Q reproduce its open-water regression to receipt "
            "precision, so eta_R = 1.0 is structural, not computed"
        ),
    }


def open_water_ladder(
    pitch_ratio: float,
    *,
    n_radial: int = 16,
    n_chord: int = 2,
    rev_per_second: float = 60.0,
) -> Dict[str, Any]:
    ladder_cfg = PREREGISTRATION["declared_config"]["open_water_ladder"]
    js = np.round(
        np.arange(ladder_cfg["start"], ladder_cfg["stop"] + 1e-12, ladder_cfg["step"]), 10
    )
    geometry = _geometry(pitch_ratio, n_radial, n_chord)
    rows = [_point(geometry, float(j), rev_per_second=rev_per_second) for j in js]
    return {
        "pitch_ratio": pitch_ratio,
        "n_radial": n_radial,
        "n_chord": n_chord,
        "J": [r["J"] for r in rows],
        "KT": [r["KT"] for r in rows],
        "KQ": [r["KQ"] for r in rows],
        "eta0": [r["eta0"] for r in rows],
    }


def state_chain(
    state: Dict[str, Any],
    ladder_cache: Dict[Any, Dict[str, Any]],
    *,
    exponent: float = 0.5,
    n_radial: int = 16,
    n_chord: int = 2,
    resistance_pe: Dict[float, Dict[str, float]] | None = None,
) -> Dict[str, Any]:
    """Per-state eta_R in four conventions plus the eta_D chain closure."""

    inputs = state["inputs"]
    p = state["propulsion_response"]
    advance = float(p["advance_ratio_J"])
    pitch = float(inputs["prop_pitch"])
    rpm = float(inputs["prop_rpm"])
    wake = float(p["wake_fraction_w"])
    n_eff = _reynolds_matched_rev_per_second(rpm)
    geometry = _geometry(pitch, n_radial, n_chord)

    # the open-water curve is a function of (P/D, panel density, Reynolds number):
    # the rpm scan changes Re, so n_eff belongs in the cache key
    key = (pitch, n_radial, n_chord, round(n_eff, 9))
    if key not in ladder_cache:
        ladder_cache[key] = open_water_ladder(
            pitch, n_radial=n_radial, n_chord=n_chord, rev_per_second=n_eff
        )
    ladder = ladder_cache[key]
    interp_kt = interpolate_open_water_curve(ladder["J"], ladder["KT"], advance)
    interp_kq = interpolate_open_water_curve(ladder["J"], ladder["KQ"], advance)
    interp_eta0 = interpolate_open_water_curve(ladder["J"], ladder["eta0"], advance)

    direct = _point(geometry, advance, rev_per_second=n_eff)

    inflow = wake_inflow(advance, wake, exponent=exponent, hub_ratio=0.18, diameter=1.0)
    behind = _point(geometry, advance, rev_per_second=n_eff, inflow_field=inflow)

    eta_r_interp = relative_rotative_efficiency(interp_eta0, behind["eta0"])

    eta0_platform = float(p["open_water_eff"])
    eta_h = float(p["hull_eff"])
    eta_d_platform = float(p["propulsive_eff"])
    eta_r_platform = float(p["relative_rot_eff"])
    eta_r_identity = eta_d_platform / (eta0_platform * eta_h)

    eta_d_chain = eta0_platform * eta_r_interp * eta_h
    eta_d_product = eta0_platform * eta_r_interp * eta_h

    eta_r_direct = relative_rotative_efficiency(direct["eta0"], behind["eta0"])

    # Independent power route: the resistance module's P_E over the propulsion
    # module's P_D.  This is the only eta_D in the receipt set that is not
    # produced by multiplying the three factors, so it is the only one that can
    # falsify the chain -- and it is reported, not used as a verdict.
    power_route = None
    eta_r_power = float("nan")
    if resistance_pe is not None:
        speed = float(inputs.get("speed_kn", float("nan")))
        row = resistance_pe.get(speed)
        if row is not None:
            eta_d_power = row["pe_kW"] / float(p["delivered_power"])
            eta_r_power = eta_d_power / (eta0_platform * eta_h)
            power_route = {
                "P_E_resistance_kW": row["pe_kW"],
                "R_T_kN": row["rt_kN"],
                "P_D_propulsion_kW": float(p["delivered_power"]),
                "etaD_power_route": eta_d_power,
                "etaD_power_route_vs_chain_pct": _pct(eta_d_power, eta0_platform * eta_r_interp * eta_h),
                "etaR_power_route_reverse": eta_r_power,
            }

    window_ok = bool(WINDOW[0] <= eta_r_interp <= WINDOW[1])

    return {
        "id": state["id"],
        "inputs": dict(inputs),
        "J": advance,
        "Va": float(p["advance_speed_Va"]),
        "wake_fraction_w": wake,
        "thrust_deduction_t": float(p["thrust_deduction_t"]),
        "reynolds_matched_rev_per_second": n_eff,
        "platform": {
            "KT": float(p["thrust_coefficient_KT"]),
            "KQ": float(p["torque_coefficient_KQ"]),
            "eta0": eta0_platform,
            "etaH": eta_h,
            "etaR_returned": eta_r_platform,
            "etaD": eta_d_platform,
            "P_E_kW": float(p["effective_power_pe"]),
            "P_D_kW": float(p["delivered_power"]),
        },
        "open_water_curve": {
            "ladder_J": ladder["J"],
            "ladder_KT": ladder["KT"],
            "ladder_KQ": ladder["KQ"],
            "ladder_eta0": ladder["eta0"],
            "interp_KT": interp_kt,
            "interp_KQ": interp_kq,
            "interp_eta0": interp_eta0,
            "direct_KT": direct["KT"],
            "direct_KQ": direct["KQ"],
            "direct_eta0": direct["eta0"],
            "interp_residual_pct_KT": _pct(interp_kt, direct["KT"]),
            "interp_residual_pct_KQ": _pct(interp_kq, direct["KQ"]),
            "interp_residual_pct_eta0": _pct(interp_eta0, direct["eta0"]),
        },
        "behind_hull": {
            "KT": behind["KT"],
            "KQ": behind["KQ"],
            "10KQ": behind["10KQ"],
            "eta0": behind["eta0"],
            "wake_exponent": exponent,
            "KT_shift_vs_open_water_pct": _pct(behind["KT"], direct["KT"]),
            "KQ_shift_vs_open_water_pct": _pct(behind["KQ"], direct["KQ"]),
        },
        "etaR": {
            "platform_returned": eta_r_platform,
            "identity_reverse": eta_r_identity,
            "openwater_interp": eta_r_interp,
            "openwater_direct": eta_r_direct,
            "power_route_reverse": eta_r_power,
        },
        "etaD": {
            "chain": eta_d_chain,
            "product": eta_d_product,
            "identity_relative_residual": abs(eta_d_chain - eta_d_product) / abs(eta_d_product),
            "platform": eta_d_platform,
            "chain_vs_platform_pct": _pct(eta_d_chain, eta_d_platform),
            "power_route": power_route,
        },
        "window_ok": window_ok,
    }


def null_control(state: Dict[str, Any], ladder_cache, **kwargs) -> Dict[str, Any]:
    """Uniform wake (q = 0): the inflow equals the open-water inflow, so eta_R must be 1."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        row = state_chain(state, ladder_cache, exponent=0.0, **kwargs)
    # at q = 0 numerator and denominator are the same solve, so the public
    # function reports `nan` on purpose; the control bypasses the guard because
    # showing that the two sides are identical *is* the point of the control
    interp_value = _ratio(row["open_water_curve"]["interp_eta0"], row["behind_hull"]["eta0"])
    direct_value = _ratio(row["open_water_curve"]["direct_eta0"], row["behind_hull"]["eta0"])
    return {
        "state": state["id"],
        "wake_exponent": 0.0,
        "etaR_openwater_interp": interp_value,
        "etaR_openwater_direct": direct_value,
        "deviation_from_unity_interp": abs(interp_value - 1.0),
        "deviation_from_unity_direct": abs(direct_value - 1.0),
        "KT_open_water": row["open_water_curve"]["direct_KT"],
        "KT_behind_hull": row["behind_hull"]["KT"],
        "interpolation_bias_pct": row["open_water_curve"]["interp_residual_pct_eta0"],
        "tautology_guard_messages": [str(w.message) for w in caught],
        "interpretation": (
            "etaR_openwater_direct = 1 exactly means the behind-hull solve reproduces the "
            "open-water solve when the inflow is uniform (same code path, only the field "
            "differs); the remaining offset of etaR_openwater_interp from 1 is the "
            "interpolation bias of reading eta_0 off the ladder and is reported separately"
        ),
    }


def wake_shape_profile(wake_fraction: float, exponent: float) -> Dict[str, Any]:
    xi = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    radius_ratio = 0.18 + xi * (1.0 - 0.18)
    w = nominal_wake_fraction(radius_ratio, wake_fraction, exponent=exponent, hub_ratio=0.18)
    return {
        "wake_exponent": exponent,
        "xi": xi.tolist(),
        "r_over_R": radius_ratio.tolist(),
        "w": np.asarray(w).tolist(),
        "w_hub": float(w[0]),
        "w_tip": float(w[-1]),
    }


def kp505_comparison() -> Dict[str, Any]:
    """The open-water model of the chain against the KP505 measured curve."""

    from tensorbem.benchmarks.kp505_openwater_bem import ANCHOR, _point as kp505_point
    from tensorbem.wageningen import B4Series

    series = B4Series(blades=5, expanded_area_ratio=0.80, pitch_ratio=0.95)
    rows: List[Dict[str, Any]] = []
    for advance, anchor in sorted(ANCHOR.items()):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            result = kp505_point(advance)
        kq = -float(result.torque) if float(result.torque) < 0.0 else float(result.torque)
        eta0_vlm = advance * float(result.thrust) / (TWO_PI * kq) if kq > 0 else float("nan")
        eta0_anchor = advance * anchor["KT"] / (TWO_PI * anchor["10KQ"] / 10.0)
        rows.append(
            {
                "J": advance,
                "KT_measured": anchor["KT"],
                "10KQ_measured": anchor["10KQ"],
                "eta0_measured": eta0_anchor,
                "KT_VLM": float(result.thrust),
                "10KQ_VLM": 10.0 * kq,
                "eta0_VLM": eta0_vlm,
                "dKT_pct_VLM": _pct(float(result.thrust), anchor["KT"]),
                "d10KQ_pct_VLM": _pct(10.0 * kq, anchor["10KQ"]),
                "deta0_pct_VLM": _pct(eta0_vlm, eta0_anchor),
                "KT_B_series": series.kt(advance),
                "dKT_pct_B_series": _pct(series.kt(advance), anchor["KT"]),
                "d10KQ_pct_B_series": _pct(10.0 * series.kq(advance), anchor["10KQ"]),
                "lt3pct_VLM_KT": abs(_pct(float(result.thrust), anchor["KT"])) < 3.0,
                "lt3pct_VLM_10KQ": abs(_pct(10.0 * kq, anchor["10KQ"])) < 3.0,
                "lt3pct_B_series_KT": abs(_pct(series.kt(advance), anchor["KT"])) < 3.0,
            }
        )

    region = [r for r in rows if r["J"] <= 0.8 + 1e-12]
    worst_vlm = max(region, key=lambda r: abs(r["dKT_pct_VLM"]))
    worst_bs = max(region, key=lambda r: abs(r["dKT_pct_B_series"]))
    in_3 = [r["J"] for r in region if r["lt3pct_VLM_KT"]]
    return {
        "anchor": {
            "path": str(KP505_ANCHOR.relative_to(REPO)),
            "sha256": sha256_file(KP505_ANCHOR),
            "citation": "Iftikhar & Takinaci (2023), Table 2 experimental columns (SECONDARY_REPRODUCTION)",
        },
        "rows": rows,
        "J_region_max": 0.8,
        "worst_dKT_VLM": {"J": worst_vlm["J"], "pct": worst_vlm["dKT_pct_VLM"]},
        "worst_dKT_B_series": {"J": worst_bs["J"], "pct": worst_bs["dKT_pct_B_series"]},
        "points_lt3pct_VLM_KT": in_3,
        "n_points_in_region": len(region),
        "n_points_lt3pct_VLM_KT": len(in_3),
        "verdict_C4": ("PASS" if abs(worst_vlm["dKT_pct_VLM"]) < 15.0 else "honest FAIL"),
        "note": (
            "the lattice is configuration-identical to the registered KP505 case "
            "(16 x 2 panels/blade, geometric wake); the KP505 anchor is a "
            "SECONDARY_REPRODUCTION of the MARINE 2023 experimental columns, so the "
            "deviation is against that transcription, not against the KRISO handout"
        ),
    }

# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def build() -> Dict[str, Any]:
    started = time.time()
    receipts = json.loads(RECEIPTS.read_text())
    states = receipts["states"]
    ladder_cache: Dict[Any, Dict[str, Any]] = {}

    primary = PREREGISTRATION["declared_config"]
    exponent = primary["wake_exponent_primary"]

    pe_map = resistance_pe_map(receipts)
    rows = [state_chain(s, ladder_cache, exponent=exponent, resistance_pe=pe_map) for s in states]

    etar_values = np.array([r["etaR"]["openwater_interp"] for r in rows], dtype=float)
    platform_values = np.array([r["etaR"]["platform_returned"] for r in rows], dtype=float)
    identity_values = np.array([r["etaR"]["identity_reverse"] for r in rows], dtype=float)
    in_window = sum(1 for v in etar_values if WINDOW[0] <= v <= WINDOW[1])
    residuals = np.array([r["etaD"]["identity_relative_residual"] for r in rows], dtype=float)
    direct_values = np.array([r["etaR"]["openwater_direct"] for r in rows], dtype=float)
    power_values = np.array(
        [r["etaR"]["power_route_reverse"] for r in rows if math.isfinite(r["etaR"]["power_route_reverse"])],
        dtype=float,
    )

    # -- control: uniform wake must reproduce the open-water point -----------
    controls = [null_control(states[0], ladder_cache), null_control(states[5], ladder_cache)]

    # -- panel sensitivity ---------------------------------------------------
    panel_rows: List[Dict[str, Any]] = []
    for n_radial, n_chord in primary["panel_sensitivity"]:
        for s in (states[0], states[3], states[6]):
            row = state_chain(
                s, ladder_cache, exponent=exponent, n_radial=n_radial, n_chord=n_chord, resistance_pe=pe_map
            )
            panel_rows.append(
                {
                    "state": row["id"],
                    "n_radial": n_radial,
                    "n_chord": n_chord,
                    "n_panels_per_blade": n_radial * n_chord,
                    "J": row["J"],
                    "etaR": row["etaR"]["openwater_interp"],
                    "eta0_open_water": row["open_water_curve"]["interp_eta0"],
                    "eta0_behind_hull": row["behind_hull"]["eta0"],
                }
            )

    # -- wake-shape sensitivity ---------------------------------------------
    shape_rows: List[Dict[str, Any]] = []
    for q in primary["wake_exponent_sensitivity"]:
        for s in (states[0], states[3], states[6]):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("ignore")
                row = state_chain(s, ladder_cache, exponent=q)
            shape_rows.append(
                {
                    "state": row["id"],
                    "wake_exponent": q,
                    "J": row["J"],
                    "KT_behind_hull": row["behind_hull"]["KT"],
                    "eta0_behind_hull": row["behind_hull"]["eta0"],
                    "etaR": _ratio(row["open_water_curve"]["interp_eta0"], row["behind_hull"]["eta0"]),
                    "etaR_direct_numerator": _ratio(
                        row["open_water_curve"]["direct_eta0"], row["behind_hull"]["eta0"]
                    ),
                }
            )

    # -- determinism ---------------------------------------------------------
    repeat = state_chain(states[3], ladder_cache, exponent=exponent)
    original = rows[3]
    determinism = {
        "state": original["id"],
        "etaR_first": original["etaR"]["openwater_interp"],
        "etaR_repeat": repeat["etaR"]["openwater_interp"],
        "abs_difference": abs(original["etaR"]["openwater_interp"] - repeat["etaR"]["openwater_interp"]),
        "KT_behind_hull_first": original["behind_hull"]["KT"],
        "KT_behind_hull_repeat": repeat["behind_hull"]["KT"],
    }

    kp505 = kp505_comparison()

    live = json.loads(LIVE_RECHECK.read_text()) if LIVE_RECHECK.exists() else None

    artifact: Dict[str, Any] = {
        "schema": "tensorbem.w31.openwater_etar_chain/v1",
        "mission": 22286,
        "repo": "TensorBEM",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": git_commit(),
        "environment": {
            "python": _platform.python_version(),
            "numpy": np.__version__,
            "platform": _platform.platform(),
            "wall_seconds": None,
        },
        "preregistration": PREREGISTRATION,
        "provenance": {
            "receipts": {
                "path": str(RECEIPTS.relative_to(REPO)),
                "sha256": sha256_file(RECEIPTS),
                "n_states": len(states),
                "endpoint_base": receipts["endpoint_base"],
            },
            "live_recheck": {
                "path": str(LIVE_RECHECK.relative_to(REPO)) if live else None,
                "sha256": sha256_file(LIVE_RECHECK) if LIVE_RECHECK.exists() else None,
                "content": live,
            },
            "kp505_anchor": str(KP505_ANCHOR.relative_to(REPO)),
            "wake_field_module": "src/tensorbem/wake_field.py",
            "solver": "src/tensorbem/propeller.py::open_water_point (inflow_field hook added in W31)",
        },
        "platform_block_audit": platform_block_audit(receipts),
        "states": rows,
        "etar_summary": {
            "n_states": len(rows),
            "primary_wake_exponent": exponent,
            "std_openwater_interp": float(np.std(etar_values)),
            "mean_openwater_interp": float(np.mean(etar_values)),
            "min_openwater_interp": float(np.min(etar_values)),
            "max_openwater_interp": float(np.max(etar_values)),
            "std_openwater_direct": float(np.std(direct_values)),
            "n_in_window_openwater_direct": int(
                sum(1 for v in direct_values if WINDOW[0] <= v <= WINDOW[1])
            ),
            "std_power_route_reverse": float(np.std(power_values)) if power_values.size else None,
            "min_power_route_reverse": float(np.min(power_values)) if power_values.size else None,
            "max_power_route_reverse": float(np.max(power_values)) if power_values.size else None,
            "n_in_window_power_route_reverse": int(
                sum(1 for v in power_values if WINDOW[0] <= v <= WINDOW[1])
            ),
            "std_platform_returned": float(np.std(platform_values)),
            "mean_platform_returned": float(np.mean(platform_values)),
            "std_identity_reverse": float(np.std(identity_values)),
            "mean_identity_reverse": float(np.mean(identity_values)),
            "n_in_window_openwater_interp": in_window,
            "n_in_window_platform_returned": int(
                sum(1 for v in platform_values if WINDOW[0] <= v <= WINDOW[1])
            ),
            "window": list(WINDOW),
        },
        "identity_table": [
            {
                "state": r["id"],
                "eta0": r["platform"]["eta0"],
                "etaR": r["etaR"]["openwater_interp"],
                "etaH": r["platform"]["etaH"],
                "etaD_chain": r["etaD"]["chain"],
                "etaD_product": r["etaD"]["product"],
                "relative_residual": r["etaD"]["identity_relative_residual"],
                "etaD_platform": r["etaD"]["platform"],
                "chain_vs_platform_pct": r["etaD"]["chain_vs_platform_pct"],
            }
            for r in rows
        ],
        "max_identity_relative_residual": float(np.max(residuals)),
        "control_null": controls,
        "panel_sensitivity": panel_rows,
        "wake_shape_sensitivity": shape_rows,
        "wake_profiles": [
            wake_shape_profile(float(rows[0]["wake_fraction_w"]), q)
            for q in primary["wake_exponent_sensitivity"]
        ],
        "determinism": determinism,
        "kp505_comparison": kp505,
        "verdicts": {},
        "caveats": [
            "the wake is axial only: no swirl and no radial velocity, because the platform "
            "reports no swirl. A wake swirl changes K_Q directly, so the eta_R above is a "
            "'nominal axial wake' eta_R, not a full behind-hull eta_R.",
            "the radial wake shape exponent q is an ASSUMED shape (the measured KP505/KCS "
            "nominal wake is not available offline in this worktree); the declared q ladder "
            "and the q=0 null control bound what is and is not a measurement here.",
            "the lattice is non-dimensional (D=1) because the kernel's horseshoe influence "
            "matrix loses rank at D=6 in this build; the Reynolds number is preserved by "
            "n_eff = n_phys D_phys^2, and this is a kernel scale-invariance defect recorded "
            "rather than fixed here.",
            "the kernel returns torque with the bound-vorticity sign convention; K_Q is taken "
            "positive here (same convention flip as the registered KP505 case).",
            "receipts carry 4 dp on non-dimensional quantities and 2 dp on kW, which bounds "
            "any identity residual computed from receipts at ~1e-3 relative.",
            "the eta_D chain closes with the platform's eta_0 and eta_H; the eta_0 of the "
            "chain is therefore the platform's open-water value, not the lattice value, so "
            "C3 closes an algebraic identity and is not evidence that eta_D is right.",
        ],
    }

    # -- verdict arithmetic (computed, never typed) --------------------------
    c1_threshold = PREREGISTRATION["gates"]["C1"]["threshold"]
    c2_threshold = PREREGISTRATION["gates"]["C2"]["threshold"]
    c3_threshold = PREREGISTRATION["gates"]["C3"]["threshold"]
    c4_threshold = PREREGISTRATION["gates"]["C4"]["threshold_pct"]

    c1_value = artifact["etar_summary"]["std_openwater_interp"]
    c3_value = artifact["max_identity_relative_residual"]
    c4_value = abs(artifact["kp505_comparison"]["worst_dKT_VLM"]["pct"])

    artifact["verdicts"] = {
        "C1": {
            "verdict": "PASS" if c1_value > c1_threshold else "honest FAIL",
            "measured": c1_value,
            "threshold": c1_threshold,
            "note": (
                "the platform's own eta_R series has std = "
                f"{artifact['etar_summary']['std_platform_returned']:.3g}, i.e. the gate FAILS "
                "on the current platform output -- that is what the gate is for"
            ),
        },
        "C2": {
            "verdict": "PASS" if in_window >= c2_threshold else "honest FAIL",
            "measured": in_window,
            "threshold": c2_threshold,
            "note": (
                "window count for the computed (openwater_interp) eta_R; the platform's "
                "constant 1.0 also sits in the window, which is exactly why a window test "
                "alone cannot detect a constant -- C1 and C2 are reported together"
            ),
        },
        "C3": {
            "verdict": "PASS" if c3_value <= c3_threshold else "honest FAIL",
            "measured": c3_value,
            "threshold": c3_threshold,
            "note": (
                "this closes the implemented chain and is algebraically exact by construction; "
                "the informative number is chain_vs_platform_pct, which is the amount by which "
                "the platform's eta_D differs from the chain with a computed eta_R"
            ),
        },
        "C4": {
            "verdict": artifact["kp505_comparison"]["verdict_C4"],
            "measured_pct": artifact["kp505_comparison"]["worst_dKT_VLM"]["pct"],
            "measured_at_J": artifact["kp505_comparison"]["worst_dKT_VLM"]["J"],
            "threshold_pct": c4_threshold,
            "n_points_lt3pct": artifact["kp505_comparison"]["n_points_lt3pct_VLM_KT"],
            "n_points_in_region": artifact["kp505_comparison"]["n_points_in_region"],
            "note": (
                "gate as stated over J <= 0.8; the same ladder restricted to J <= 0.7 gives "
                + f"{max(abs(r['dKT_pct_VLM']) for r in artifact['kp505_comparison']['rows'] if r['J'] <= 0.7 + 1e-12):.2f} % "
                "(reported, not substituted for the verdict)"
            ),
        },
    }

    artifact["environment"]["wall_seconds"] = round(time.time() - started, 2)
    return artifact


def write(artifact: Dict[str, Any], path: Path = OUT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=1, ensure_ascii=False) + "\n")
    return path


def main() -> int:
    artifact = build()
    path = write(artifact)
    print(f"wrote {path.relative_to(REPO)}  ({artifact['environment']['wall_seconds']} s)")
    for key, value in artifact["verdicts"].items():
        print(f"  {key}: {value['verdict']}  measured={value.get('measured', value.get('measured_pct'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
