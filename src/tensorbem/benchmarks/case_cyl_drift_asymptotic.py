"""CASE-BEM-CYL-DRIFT-ASYMP — case-pool benchmark: mean drift force on a
bottom-mounted vertical circular cylinder and its ``(ka)^3`` long-wave law.

Anchor
------
``CASE-BEM-CYL-DRIFT-ASYMP`` (case pool, owner 2026-11-12).  Reference = the two
*published* closed forms of the second-order mean drift force on a fixed
bottom-mounted vertical cylinder in deep water:

* **MacCamy & Fuchs (1954)** diffraction potential (the ``A_m`` mode
  amplitudes) -- the near-field momentum integral over the wetted wall,
* **Maruo (1960)** far-field mean-drift formula in its reduced,
  radiation-radius-free form.

Target quantity: the dimensionless mean drift coefficient

    ``C(ka) = F_x / (rho g A^2 a)``

and, as ``ka -> 0``, the closed-form long-wave law

    ``F_x = (5 pi^2 / 16) rho g A^2 a (ka)^3``      (``5 pi^2/16 = 3.0842513753...``)

Tolerance (case-pool criterion): **< 5 %** on every gate below.

What the driver measures (all numbers written by code, none transcribed)
-----------------------------------------------------------------------
1. ``long_wave``: ladder ``ka = 0.02 .. 0.30``; reference = the published
   long-wave law, got = the exact MacCamy-Fuchs near-field and Maruo far-field
   evaluations.  Also the width of the ``< 5 %`` band of the law
   (``law_valid_ka``) -- the law is asymptotic, so the band edge is a result,
   not an input.
2. ``slope``: log-log slope of ``F(ka)`` for ``ka <= 0.1`` -> 3 (the exponent of
   the law), independent of the coefficient.
3. ``wide_ka``: ``ka = 0.2 .. 3.0``; near-field vs far-field of the *same*
   closed form (two independent second-order formulas) point by point.
4. ``bem``: the **real solver route** -- ``tensorbem.free_surface_solver3d``
   diffraction solve on a panelled cylinder -> Kochin function
   (``tensorbem.drift_far_field3d.bem_kochin``) -> Maruo far-field, and the
   near-field pressure/waterline integral on the same BEM solution.  Both are
   compared with the closed form; this is the "run the capability, do not
   rebuild it" part of the case.
5. ``convergence``: panel ladder at ``ka = 0.5`` with observed order; the
   pressure-integral (near-field) route is quadrature limited and needs the
   finest rung to reach the 5 % gate -- reported, not hidden.
6. ``azimuth``: azimuth-resolution sensitivity of the BEM Kochin table, and
   ``scaling``: the ``F ~ A^2`` second-order amplitude law.

Usage
-----
    python -m tensorbem.benchmarks.case_cyl_drift_asymptotic \
        --json artifacts/case_cyl_drift_asymptotic/results.json

    # figures only (reads the JSON; cheap, runs on the control node)
    python -m tensorbem.benchmarks.case_cyl_drift_asymptotic \
        --json artifacts/case_cyl_drift_asymptotic/results.json \
        --figures artifacts/case_cyl_drift_asymptotic --figures-only

    # analytic part only (seconds, no BEM)
    python -m tensorbem.benchmarks.case_cyl_drift_asymptotic --skip-bem --json /tmp/x.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .. import cylinder_drift as cd

DENSITY = float(cd.DEFAULT_DENSITY)
GRAVITY = float(cd.DEFAULT_GRAVITY)
RADIUS = 1.0
AMPLITUDE = 1.0
SCALE = DENSITY * GRAVITY * AMPLITUDE**2 * RADIUS  # rho g A^2 a
TOL = 0.05

LONG_WAVE_KA = (0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
LAW_SCAN = tuple(float(v) for v in np.round(np.linspace(0.01, 0.40, 40), 6))
SLOPE_KA = (0.02, 0.05, 0.10)
WIDE_KA = (0.2, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)

BEM_KA = (0.5, 1.0, 2.0)
BEM_MESH = (32, 12)  # 2 x 32 x 12 + 1 = 800 panels
CONV_KA = 0.5
CONV_MESH = ((16, 6), (24, 9), (32, 12), (48, 18))
AZIMUTH_KA = 1.0
AZIMUTH_VALUES = (180, 360)

# ---------------------------------------------------------------------------
# literature (citations + programmatic transcription provenance)
# ---------------------------------------------------------------------------
LITERATURE = {
    "maccamy_fuchs_1954": {
        "citation": (
            "MacCamy, R.C. & Fuchs, R.A. (1954) Wave forces on a pile: a diffraction theory. "
            "Technical Memorandum No. 69, Beach Erosion Board, US Army, 17 pp."
        ),
        "equation": (
            "psi(r,theta) = sum_m [ i^m J_m(kr) + A_m H_m^(1)(kr) ] e^{i m theta}, "
            "A_m = -i^m J_m'(ka) / H_m'^(1)(ka)"
        ),
        "implemented_in": "src/tensorbem/cylinder_drift.py: maccamy_fuchs_modes / near_field_cylinder_drift",
        "transcription": {
            "reproduced_in": (
                "Chakrabarti, S.K. & Gupta, A. (2007) Steady wave drift force on basic objects of "
                "symmetry. WIT Transactions on the Built Environment, Vol. 92 (Fluid Structure "
                "Interaction and Moving Boundary Problems IV), WIT Press, pp. 1-11."
            ),
            "section": "4.1 Vertical cylinder, Eq. (3)",
            "url": "http://www.witpress.com/Secure/elibrary/papers/FSI07/FSI07001FU1.pdf",
            "pdf_sha256": "c8506effb013fcd8d55029d10f1c006a5abb0d0ffd9917ef14c0d517f4c5db86",
            "pdf_page": 3,
            "snippet": (
                "For a bottom-seated cylinder extended above the free surface, MacCamy and Fuchs [4] "
                "developed the forces on a fixed vertical cylinder in a closed form using the Bessel "
                "function series. ... The total linear velocity potential including diffraction at the "
                "surface of a bottom-mounted cylinder, r = a ... is written as Eq. (3) [psi = "
                "sum_n (a_n cos n theta + b_n sin n theta) ...] with the coefficients A_n = "
                "2 J_n'(ka) / ... ; A_n^2 = [J_n'(ka)]^2 + [Y_n'(ka)]^2"
            ),
            "extraction": "pypdf 6.x text extraction, page index 3 (1-based); PDF fetched by this node",
        },
    },
    "maruo_1960": {
        "citation": (
            "Maruo, H. (1960) On the increase of the resistance of a ship in rough seas "
            "(the drift force). J. Soc. Nav. Archit. Japan 107, 59-67."
        ),
        "equation": (
            "F_x = (rho g A^2 / (2 pi k)) Int_0^{2 pi} |S(theta)|^2 (1 - cos theta) dtheta  "
            "(deep water, reduced / radiation-radius-free form)"
        ),
        "implemented_in": "src/tensorbem/cylinder_drift.py: far_field_cylinder_drift",
        "transcription": {
            "reproduced_in": (
                "McIver, P. (1986) Mean drift forces on arrays of bodies due to incident long waves. "
                "Brunel University, Department of Mathematics, Report TR/13/86."
            ),
            "section": "5 The mean drift force, Eqs. (5.1)-(5.2)",
            "url": "http://bura.brunel.ac.uk/bitstream/2438/2010/1/TR_13_86.pdf",
            "pdf_sha256": "d24ff37ec34abb7846983dba873c94be51bc3131d17e2aebe7413c5c3aebac16",
            "pdf_page": 15,
            "snippet": (
                "The mean drift forces and moments on a body have been related to the far field of the "
                "first order scattering potential by Maruo (1960) and Newman (1967). The result for the "
                "mean horizontal drift force in the direction of wave advance is [Eq. (5.1): "
                "f_x^(2) = rho g A^2/2 * k/(...) * Int_0^{2 pi} (1 - cos psi) |A(psi)|^2 dpsi] ... "
                "where A(psi) is related to the far field form of the scattered wave through "
                "[Eq. (5.2): phi ~ ... (2/(pi k r))^(1/2) e^{i(kr-pi/4)} A(psi) cosh k(z+h)/cosh kh]"
            ),
            "extraction": (
                "pypdf 6.x text extraction, page index 15 (1-based); the fraction layout of (5.1) is "
                "mangled by PDF text extraction, so the deep-water reduction used here is the one "
                "verified numerically against the independent near-field momentum integral"
            ),
        },
    },
    "long_wave_law": {
        "expression": "5*pi**2/16",
        "value": 5.0 * math.pi**2 / 16.0,
        "status": (
            "derived, not quoted verbatim: no freely reachable copy of Maruo (1960) carries a "
            "typeset statement of this coefficient (publisher full text blocked in this sandbox). "
            "The value is obtained here three independent ways -- (i) truncated-mode asymptotics of "
            "the MacCamy-Fuchs A_m (m = 0, +-1) + analytic Maruo kernel integral, (ii) exact "
            "near-field momentum integral, (iii) exact far-field integral over ka = 0.01 .. 0.4 -- "
            "so the *number* is verified even though the literature quote is not. Recorded as a "
            "known gap, see report section 6."
        ),
    },
}


def literature_coefficient() -> float:
    """``5 pi^2 / 16`` evaluated from the recorded expression string."""
    assert LITERATURE["long_wave_law"]["expression"] == "5*pi**2/16"
    return 5.0 * math.pi**2 / 16.0


def _rel(value: float, reference: float) -> float:
    return float(abs(value - reference) / abs(reference)) if reference else float("inf")


def _closed_form(ka: float) -> dict:
    near = cd.near_field_cylinder_drift(float(ka), amplitude=AMPLITUDE, radius=RADIUS,
                                        density=DENSITY, gravity=GRAVITY)
    far = cd.far_field_cylinder_drift(float(ka), amplitude=AMPLITUDE, radius=RADIUS,
                                      density=DENSITY, gravity=GRAVITY)
    return {"near": float(near["mean_drift_force_x"]), "far": float(far["mean_drift_force_x"])}


def _law(ka: float) -> float:
    return literature_coefficient() * SCALE * float(ka) ** 3


# ---------------------------------------------------------------------------
# 1. long-wave ladder + width of the <5 % band
# ---------------------------------------------------------------------------
def long_wave() -> dict:
    rows = []
    for ka in LONG_WAVE_KA:
        cf = _closed_form(ka)
        ref = _law(ka)
        rows.append(
            {
                "ka": ka,
                "reference_law": ref,
                "near_field": cf["near"],
                "far_field": cf["far"],
                "rel_error_near": _rel(cf["near"], ref),
                "rel_error_far": _rel(cf["far"], ref),
                "normalised_coefficient": cf["far"] / (SCALE * ka**3),
                "pass": _rel(cf["far"], ref) < TOL,
            }
        )
    band = 0.0
    for ka in LAW_SCAN:
        if _rel(_closed_form(ka)["far"], _law(ka)) < TOL:
            band = ka
        else:
            break
    return {
        "rows": rows,
        "tolerance": TOL,
        "law_valid_ka": band,
        "law_valid_ka_gate": 0.20,
        "closed_form_coefficient": literature_coefficient(),
        "module_coefficient": float(cd.long_wave_drift_coefficient()),
        "coefficient_abs_diff": abs(literature_coefficient() - float(cd.long_wave_drift_coefficient())),
    }


# ---------------------------------------------------------------------------
# 2. exponent of the law from the numerical slope
# ---------------------------------------------------------------------------
def slope() -> dict:
    ks = np.asarray(SLOPE_KA, dtype=float)
    forces = np.asarray([_closed_form(k)["far"] for k in ks])
    p = np.polyfit(np.log(ks), np.log(forces), 1)
    return {
        "ka": list(SLOPE_KA),
        "forces": forces.tolist(),
        "fitted_slope": float(p[0]),
        "fitted_intercept": float(p[1]),
        "target_slope": 3.0,
        "slope_tolerance": 0.05,
        "pass": abs(float(p[0]) - 3.0) <= 0.05,
    }


# ---------------------------------------------------------------------------
# 3. two independent second-order formulas, wide ka
# ---------------------------------------------------------------------------
def wide_ka() -> dict:
    rows = []
    for ka in WIDE_KA:
        cf = _closed_form(ka)
        rows.append(
            {
                "ka": ka,
                "near_field": cf["near"],
                "far_field": cf["far"],
                "near_vs_far": _rel(cf["near"], cf["far"]),
                "drift_coefficient": cf["far"] / SCALE,
                "law_rel_error": _rel(cf["far"], _law(ka)),
            }
        )
    worst = max(r["near_vs_far"] for r in rows)
    return {"rows": rows, "max_near_vs_far": worst, "gate": 1.0e-6, "pass": worst < 1.0e-6}


# ---------------------------------------------------------------------------
# 4./5./6. real BEM solver route
# ---------------------------------------------------------------------------
def bem_case(ka: float, mesh=BEM_MESH, n_azimuth: int = 180) -> dict:
    from .drift_far_field_3d_study import run_case

    r = run_case(float(ka), n_theta=int(mesh[0]), n_z=int(mesh[1]), growth=1.35,
                 density=DENSITY, gravity=GRAVITY, n_azimuth=int(n_azimuth))
    cf = _closed_form(ka)
    row = {
        "ka": float(ka),
        "n_theta": int(mesh[0]),
        "n_z": int(mesh[1]),
        "n_azimuth": int(n_azimuth),
        "panels": int(r["panels"]),
        "depth_over_radius": float(r["depth_over_radius"]),
        "bem_near_field": float(r["near_field_x"]),
        "bem_far_field": float(r["far_field_x"]),
        "closed_form": float(cf["far"]),
        "closed_form_near": float(cf["near"]),
        "law": _law(ka),
        "bem_far_vs_closed_error": _rel(float(r["far_field_x"]), cf["far"]),
        "bem_near_vs_closed_error": _rel(float(r["near_field_x"]), cf["near"]),
        "bem_near_vs_far_gap": _rel(float(r["near_field_x"]), float(r["far_field_x"])),
        "drift_coefficient_bem": float(r["far_field_x"]) / SCALE,
        "energy_identity_residual": float(r["energy_residual"]),
        "solver_relative_residual": float(r["solver_relative_residual"]),
        "azimuth_rel_change": float(r["azimuth_rel_change"]),
        "seconds": float(r["seconds"]),
    }
    return row


def bem_gate(mesh=BEM_MESH) -> dict:
    rows = [bem_case(ka, mesh=mesh) for ka in BEM_KA]
    far_worst = max(r["bem_far_vs_closed_error"] for r in rows)
    near_worst = max(r["bem_near_vs_closed_error"] for r in rows)
    return {
        "rows": rows,
        "mesh": list(mesh),
        "tolerance": TOL,
        "max_bem_far_error": far_worst,
        "max_bem_near_error": near_worst,
        "far_pass": far_worst < TOL,
        "near_pass": near_worst < TOL,
        "seconds": sum(r["seconds"] for r in rows),
    }


def convergence() -> dict:
    rows = []
    for mesh in CONV_MESH:
        r = bem_case(CONV_KA, mesh=mesh)
        rows.append(
            {
                "panels": r["panels"],
                "n_theta": r["n_theta"],
                "n_z": r["n_z"],
                "far_error": r["bem_far_vs_closed_error"],
                "near_error": r["bem_near_vs_closed_error"],
                "gap": r["bem_near_vs_far_gap"],
                "seconds": r["seconds"],
            }
        )
    orders = []
    for a, b in zip(rows, rows[1:]):
        ha = a["panels"] ** -0.5
        hb = b["panels"] ** -0.5
        for key in ("far_error", "near_error"):
            if a[key] > 0.0 and b[key] > 0.0:
                orders.append(
                    {
                        "panels": [a["panels"], b["panels"]],
                        "quantity": key,
                        "observed_order": float(
                            math.log(a[key] / b[key]) / math.log(ha / hb)
                        ),
                    }
                )
    far_monotone = all(b["far_error"] <= a["far_error"] for a, b in zip(rows, rows[1:]))
    near_monotone = all(b["near_error"] <= a["near_error"] for a, b in zip(rows, rows[1:]))
    near_top = rows[-1]["near_error"]
    far_top = rows[-1]["far_error"]
    return {
        "ka": CONV_KA,
        "rows": rows,
        "orders": orders,
        "far_monotone": far_monotone,
        "near_monotone": near_monotone,
        "near_error_at_finest": near_top,
        "far_error_at_finest": far_top,
        "near_pass_at_finest": near_top < TOL,
        "far_pass_at_finest": far_top < TOL,
        "pass": bool(far_monotone and near_monotone and near_top < TOL and far_top < TOL),
    }


def azimuth() -> dict:
    rows = [bem_case(AZIMUTH_KA, mesh=BEM_MESH, n_azimuth=n) for n in AZIMUTH_VALUES]
    spread = _rel(rows[-1]["bem_far_field"], rows[0]["bem_far_field"])
    return {
        "ka": AZIMUTH_KA,
        "rows": rows,
        "spread": spread,
        "gate": 0.01,
        "pass": spread < 0.01,
    }


def scaling() -> dict:
    """Second-order amplitude law: F ~ A^2 (checked on the exact closed form)."""
    k = 1.0
    f1 = cd.far_field_cylinder_drift(k, amplitude=1.0, radius=RADIUS,
                                     density=DENSITY, gravity=GRAVITY)["mean_drift_force_x"]
    f2 = cd.far_field_cylinder_drift(k, amplitude=2.0, radius=RADIUS,
                                     density=DENSITY, gravity=GRAVITY)["mean_drift_force_x"]
    ratio = float(f2) / float(f1)
    return {"ka": k, "force_amplitude_1": float(f1), "force_amplitude_2": float(f2),
            "ratio": ratio, "target": 4.0, "gate": 1.0e-9, "pass": abs(ratio - 4.0) < 1.0e-9}


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def _git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def run_meta(repro_cmd: str, wall: float) -> dict:
    return {
        "case": "CASE-BEM-CYL-DRIFT-ASYMP",
        "repro_cmd": repro_cmd,
        "git_rev": _git_rev(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "host": platform.node(),
        "platform": platform.platform(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - wall)),
        "wall_seconds": wall,
        "units": {
            "rho": "kg/m^3", "g": "m/s^2", "A": "m", "a": "m", "ka": "dimensionless",
            "force": "N", "drift_coefficient": "F_x/(rho g A^2 a)",
        },
        "parameters": {
            "density": DENSITY, "gravity": GRAVITY, "radius": RADIUS, "amplitude": AMPLITUDE,
            "wave_heading": "beam seas (beta = 0), fixed body, deep water (omega^2 = g k)",
            "water_depth_rule": "BEM meshes use depth = 5/k (kd = 5, i.e. e^-5 = 0.7 % depth decay)",
        },
    }


def build(skip_bem: bool = False, repro_cmd: str = "") -> dict:
    t0 = time.time()
    out: dict = {}
    out["literature"] = LITERATURE
    out["long_wave"] = long_wave()
    out["slope"] = slope()
    out["wide_ka"] = wide_ka()
    out["scaling"] = scaling()
    if skip_bem:
        out["bem"] = {"skipped": True}
        out["convergence"] = {"skipped": True}
        out["azimuth"] = {"skipped": True}
    else:
        out["bem"] = bem_gate()
        out["convergence"] = convergence()
        out["azimuth"] = azimuth()
    out["gates"] = gates(out)
    out["run"] = run_meta(repro_cmd, time.time() - t0)
    return out


def gates(res: dict) -> dict:
    lw, sl = res["long_wave"], res["slope"]
    g = {
        "G1_law_at_ka002": {
            "got": lw["rows"][0]["rel_error_far"], "threshold": TOL,
            "pass": lw["rows"][0]["rel_error_far"] < TOL,
            "what": "exact closed form vs published long-wave law at ka = 0.02",
        },
        "G2_law_band": {
            "got": lw["law_valid_ka"], "threshold": lw["law_valid_ka_gate"], "pass": lw["law_valid_ka"] >= lw["law_valid_ka_gate"],
            "what": "widest ka at which the law holds to < 5 %",
        },
        "G3_coefficient_identity": {
            "got": lw["coefficient_abs_diff"], "threshold": 1.0e-12,
            "pass": lw["coefficient_abs_diff"] < 1.0e-12,
            "what": "5 pi^2/16 (literature expression) vs module long_wave_drift_coefficient()",
        },
        "G4_slope": {"got": sl["fitted_slope"], "threshold": 3.0, "tol": sl["slope_tolerance"], "pass": sl["pass"],
                     "what": "log-log exponent of F(ka), ka <= 0.1"},
        "G5_near_vs_far": {"got": res["wide_ka"]["max_near_vs_far"], "threshold": res["wide_ka"]["gate"],
                           "pass": res["wide_ka"]["pass"],
                           "what": "two independent closed-form routes, ka = 0.2 .. 3"},
        "G6_amplitude_square": {"got": res["scaling"]["ratio"], "threshold": 4.0, "pass": res["scaling"]["pass"],
                                "what": "F(2A)/F(A) = 4 (second order)"},
    }
    if not res["bem"].get("skipped"):
        g.update({
            "G7_bem_far_vs_closed_form": {
                "got": res["bem"]["max_bem_far_error"], "threshold": TOL, "pass": res["bem"]["far_pass"],
                "what": "real BEM solve -> Kochin -> Maruo vs closed form, 800 panels, ka = 0.5/1/2",
            },
            "G8_bem_near_converged": {
                "got": res["convergence"]["near_error_at_finest"], "threshold": TOL,
                "pass": res["convergence"]["near_pass_at_finest"],
                "what": "BEM near-field pressure integral at the finest panel rung (ka = 0.5)",
            },
            "G9_convergence_monotone": {
                "got": {"far": res["convergence"]["far_monotone"], "near": res["convergence"]["near_monotone"]},
                "threshold": True, "pass": bool(res["convergence"]["far_monotone"] and res["convergence"]["near_monotone"]),
                "what": "both BEM routes converge monotonically under panel refinement",
            },
            "G10_azimuth": {"got": res["azimuth"]["spread"], "threshold": res["azimuth"]["gate"],
                            "pass": res["azimuth"]["pass"],
                            "what": "BEM Kochin azimuth resolution 180 -> 360"},
        })
    g["all_pass"] = all(v["pass"] for k, v in g.items() if isinstance(v, dict))
    return g


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def figures(results: dict, outdir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []
    c_lit = results["long_wave"]["closed_form_coefficient"]

    # (1) drift coefficient C(ka) = F/(rho g A^2 a)
    ks = np.logspace(-2, np.log10(3.2), 200)
    c_exact = np.array([_closed_form(k)["far"] / SCALE for k in ks])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.loglog(ks, c_exact, "-", lw=2.0, label="closed form (MacCamy-Fuchs + Maruo)")
    ax.loglog(ks, c_lit * ks**3, "--", lw=1.4, label=r"long-wave law $\frac{5\pi^2}{16}(ka)^3$")
    bem = results.get("bem", {})
    if not bem.get("skipped"):
        bka = [r["ka"] for r in bem["rows"]]
        bf = [r["bem_far_field"] / SCALE for r in bem["rows"]]
        bn = [r["bem_near_field"] / SCALE for r in bem["rows"]]
        ax.loglog(bka, bf, "o", ms=6, mfc="none", label="BEM solve -> Maruo (800 panels)")
        ax.loglog(bka, bn, "s", ms=5, mfc="none", label="BEM solve -> near-field integral")
        ax.axvspan(results["long_wave"]["law_valid_ka"], 3.2, color="0.85", alpha=0.5, lw=0)
        ax.text(results["long_wave"]["law_valid_ka"] * 1.15, 2e-4,
                f"law fails >5 % for ka > {results['long_wave']['law_valid_ka']:.2f}", fontsize=7)
    ax.set_xlabel("ka")
    ax.set_ylabel(r"$C(ka)=F_x/(\rho g A^2 a)$")
    ax.set_title("Mean drift force, bottom-mounted vertical cylinder (deep water)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")
    p = outdir / "cyl_drift_coefficient.png"
    fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig); made.append(p)

    # (2) (ka)^3-normalised plateau
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    norm = c_exact / ks**3
    ax.semilogx(ks, norm, "-", lw=2.0, label=r"$F/(\rho g A^2 a (ka)^3)$ closed form")
    ax.axhline(c_lit, color="k", ls="--", lw=1.2, label=r"$5\pi^2/16 = %.6f$" % c_lit)
    ax.fill_between([1e-2, 3.2], c_lit * 0.95, c_lit * 1.05, color="C0", alpha=0.12,
                    label="±5 % band")
    if not bem.get("skipped"):
        ax.plot([r["ka"] for r in bem["rows"]],
                [r["bem_far_field"] / (SCALE * r["ka"] ** 3) for r in bem["rows"]],
                "o", ms=6, mfc="none", label="BEM -> Maruo / $(ka)^3$")
    ax.set_xlabel("ka")
    ax.set_ylabel(r"$F_x/(\rho g A^2 a (ka)^3)$")
    ax.set_ylim(0.0, 4.2)
    ax.set_title(r"$(ka)^3$ law: plateau value %s" % f"{c_lit:.6f}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    p = outdir / "cyl_drift_long_wave_plateau.png"
    fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig); made.append(p)

    # (3) BEM panel convergence
    conv = results.get("convergence", {})
    if not conv.get("skipped"):
        rows = conv["rows"]
        panels = [r["panels"] for r in rows]
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.loglog(panels, [100 * r["far_error"] for r in rows], "o-", label="far field (Maruo from BEM Kochin)")
        ax.loglog(panels, [100 * r["near_error"] for r in rows], "s-", label="near field (BEM pressure integral)")
        ax.axhline(100 * TOL, color="r", ls="--", lw=1.2, label="5 % gate")
        ref = 100 * rows[0]["far_error"] * (np.asarray(panels) / panels[0]) ** -0.5
        ax.loglog(panels, ref, ":", color="0.5", label=r"slope $-1/2$ guide")
        ax.set_xlabel("panels")
        ax.set_ylabel("relative error vs closed form  [%]")
        ax.set_title(f"BEM convergence at ka = {conv['ka']}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
        p = outdir / "cyl_drift_bem_convergence.png"
        fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig); made.append(p)
    return made


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default="artifacts/case_cyl_drift_asymptotic/results.json")
    ap.add_argument("--figures", default=None, help="directory for PNG figures")
    ap.add_argument("--figures-only", action="store_true", help="read --json and only draw figures")
    ap.add_argument("--skip-bem", action="store_true", help="analytic part only (no BEM solves)")
    args = ap.parse_args(argv)

    repro = (
        "PYTHONPATH=src python3 -m tensorbem.benchmarks.case_cyl_drift_asymptotic "
        f"--json {args.json}" + (f" --figures {args.figures}" if args.figures else "")
    )
    if args.figures_only:
        res = json.loads(Path(args.json).read_text())
        made = figures(res, Path(args.figures or Path(args.json).parent))
        print(json.dumps({"figures": [str(p) for p in made]}, indent=1))
        return 0

    res = build(skip_bem=args.skip_bem, repro_cmd=repro)
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, sort_keys=False))
    if args.figures:
        res["figures"] = [str(p) for p in figures(res, Path(args.figures))]
        out.write_text(json.dumps(res, indent=1, sort_keys=False))
    print(json.dumps({"gates": res["gates"], "json": str(out)}, indent=1)[:4000])
    return 0 if res["gates"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())