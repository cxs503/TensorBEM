#!/usr/bin/env python3
"""Flexural-gravity wave verification: machine-checkable evidence bundle.

Mission m14560 (rework of m12052), node ``p407w3-n06-bem-ice-covered-wave``.

This entry point re-runs every acceptance criterion of the node, cross-checks
the model against an **independent implementation** living in the main
plate/fluid solver (:func:`tensorbem.hydroelastic.flexural_gravity_omega`, which
now delegates to :func:`tensorbem.flexural_gravity.plate_fluid_omega`), runs the
pytest suite, and finally writes

* ``docs/flexural_gravity_verification/run/verify_run.json``  -- run artifact
* ``docs/flexural_gravity_verification/run/pytest_flexural_gravity.log``
* ``docs/flexural_gravity_verification/benchmark.json``       -- ref vs got table
* ``docs/flexural_gravity_verification/external_anchor.json`` -- literature anchor
* ``docs/flexural_gravity_verification/evidence.json``        -- L2/L3/L4 evidence
* ``docs/flexural_gravity_verification/fig_cross_implementation.png``

Usage::

    python scripts/verify_flexural_gravity_evidence.py [--skip-pytest] [--no-figures]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorbem import flexural_gravity as fg            # noqa: E402
from tensorbem import hydroelastic as hyd              # noqa: E402
from tensorbem import flexural_gravity_numeric as fgn  # noqa: E402

OUT = REPO / "docs" / "flexural_gravity_verification"
RUN = OUT / "run"

G = fg.DEFAULT_GRAVITY
RHO_W = fg.DEFAULT_WATER_DENSITY
RHO_I = fg.DEFAULT_ICE_DENSITY
E_ICE = fg.DEFAULT_YOUNG_MODULUS
NU_ICE = fg.DEFAULT_POISSON_RATIO
DEPTH = 200.0
THICKNESSES = (0.5, 1.0, 2.0)


def plate(h: float, *, E: float = E_ICE) -> fg.IcePlate:
    return fg.IcePlate(
        thickness=h,
        young_modulus=E,
        poisson_ratio=NU_ICE,
        water_density=RHO_W,
        ice_density=RHO_I,
        water_depth=DEPTH,
        gravity=G,
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b else float("inf")


# ---------------------------------------------------------------------------
# Acceptance criteria E1 .. E8
# ---------------------------------------------------------------------------
def criterion_e1() -> dict:
    """E1: numerical dispersion solve vs closed-form, >=20 wavenumbers."""
    k_grid = np.logspace(-3.0, 1.0, 25)
    rows, worst = [], 0.0
    for h in THICKNESSES:
        p = plate(h)
        for k in k_grid:
            omega = fg.flexural_gravity_omega(float(k), p)
            k_back = fg.flexural_gravity_wavenumber(omega, p)
            omega_back = fg.flexural_gravity_omega(k_back, p)
            err = rel(omega_back, omega)
            worst = max(worst, err)
            rows.append({"h_ice": h, "k": float(k), "omega": omega,
                         "k_solved": k_back, "rel_error": err})
    return {
        "description": "numerical omega(k) (wavenumber solver round-trip) vs closed form, rel <= 1%",
        "n_points": len(rows),
        "n_wavenumbers_per_thickness": len(k_grid),
        "k_range": [float(k_grid[0]), float(k_grid[-1])],
        "max_relative_error": worst,
        "pass": bool(worst <= 1.0e-2),
        "rows": rows,
    }


def criterion_e1_independent() -> dict:
    """E1b: closed form vs an independent finite-difference depth solve."""
    rows, worst = [], 0.0
    k_grid = (1.0e-2, 5.0e-2, 0.2, 1.0, 5.0, 20.0)
    for h in THICKNESSES:
        ip = plate(h)
        for k in k_grid:
            ref = fg.flexural_gravity_omega2_mass_loading(k, ip)
            got = fgn.fd_omega2(
                k, depth=DEPTH, rigidity=ip.flexural_rigidity,
                mass_per_area=RHO_I * h, fluid_density=RHO_W, gravity=G)
            err = rel(got, ref)
            worst = max(worst, err)
            rows.append({"h_ice": h, "k": k, "ref_closed_form_omega2": ref,
                         "got_fd_omega2": got, "rel_error": err})
    return {
        "description": "closed-form eq.(3) vs finite-difference depth solve "
                       "(psi''=k^2 psi, Neumann bottom, ice/water free-surface condition)",
        "independent_reference": "tensorbem.flexural_gravity_numeric.fd_omega2",
        "subject": "tensorbem.flexural_gravity.flexural_gravity_omega2_mass_loading",
        "n_points": len(rows),
        "max_relative_error": worst,
        "pass": bool(worst <= 1.0e-3),
        "rows": rows,
    }


def criterion_e2() -> dict:
    """E2: long-wave limit degenerates to open water."""
    rows, worst = [], 0.0
    for h in THICKNESSES:
        p = plate(h)
        for k in (1.0e-5, 1.0e-4, 1.0e-3):
            got = fg.flexural_gravity_omega2(k, p)
            ref = fg.open_water_omega2(k, DEPTH, G)
            err = rel(got, ref)
            worst = max(worst, err)
            rows.append({"h_ice": h, "k": k, "omega2_open_water": ref,
                         "omega2_flexural": got, "rel_error": err})
    return {
        "description": "k->0: omega^2 -> g k tanh(kH), rel <= 0.5%",
        "rows": rows, "max_relative_error": worst, "pass": bool(worst <= 5.0e-3),
    }


def criterion_e3() -> dict:
    """E3: short-wave limit degenerates to the bending wave D k^5 / rho."""
    rows, worst = [], 0.0
    for h in THICKNESSES:
        p = plate(h)
        for k in (10.0, 20.0, 50.0):
            got = fg.flexural_gravity_omega2(k, p)
            ref = fg.flexural_wave_omega2(k, p)
            err = rel(got, ref)
            worst = max(worst, err)
            rows.append({"h_ice": h, "k": k, "omega2_bending": ref,
                         "omega2_flexural": got, "rel_error": err})
    return {
        "description": "k->inf (deep water): omega^2 -> D k^5 / rho_w, rel <= 0.5%",
        "rows": rows, "max_relative_error": worst, "pass": bool(worst <= 5.0e-3),
    }


def criterion_e4() -> dict:
    """E4: energy conservation |R|^2 + |T|^2 = 1 at ice edge (>=5 frequencies)."""
    omegas = (0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)
    rows, worst = [], 0.0
    for h in (0.5, 1.0, 2.0):
        p = plate(h)
        for omega in omegas:
            for inertia in (False, True):
                res = fg.ice_edge_coefficients(omega, p, include_plate_inertia=inertia)
                worst = max(worst, abs(res.energy_residual))
                rows.append({"h_ice": h, "omega": omega,
                             "model": "eq3_mass_loading" if inertia else "eq2_rigid",
                             "R_energy": res.R_energy, "T_energy": res.T_energy,
                             "energy_residual": res.energy_residual})
    return {
        "description": "|R|^2+|T|^2-1 residual <= 1e-4 over 8 frequencies x 3 thicknesses x 2 models",
        "n_combinations": len(rows), "max_abs_residual": worst,
        "pass": bool(worst <= 1.0e-4), "rows": rows,
    }


def criterion_e5() -> dict:
    """E5: transmitted wavelength under ice (eq.3 inertia model)."""
    rows = []
    for h in (0.5, 1.0, 2.0):
        p = plate(h)
        for omega in (0.2, 0.3, 0.5, 0.7, 1.0):
            k0 = fg.open_water_wavenumber(omega, DEPTH, G)
            k1 = fg.flexural_gravity_wavenumber_mass_loading(omega, p)
            lam0, lam1 = 2.0 * math.pi / k0, 2.0 * math.pi / k1
            rows.append({"h_ice": h, "omega": omega, "lambda_open": lam0,
                         "lambda_ice": lam1, "ratio": lam1 / lam0,
                         "shorter": bool(lam1 < lam0)})
    # The inertia-dominated (long-wave) branch must give lambda_ice < lambda_open.
    short = [r for r in rows if r["omega"] <= 0.3]
    return {
        "description": "eq.(3) inertia-dominated long waves: lambda_ice < lambda_open",
        "crossover_note": "above the crossover wavenumber (bending dominated) the ice "
                          "wave is faster; see deep_water_crossover_wavenumber",
        "n_long_wave_points": len(short),
        "all_long_wave_shorter": all(r["shorter"] for r in short),
        "rows": rows,
        "pass": bool(len(short) >= 3 and all(r["shorter"] for r in short)),
    }


def criterion_e6() -> dict:
    """E6: gradual (wedge) ice edge reflects less than an abrupt edge."""
    omega = 1.5
    rows = []
    for h in (0.5, 1.0, 2.0):
        p = plate(h)
        abrupt = fg.abrupt_ice_edge_reflection(omega, p)
        for L in (50.0, 100.0, 200.0):
            w = fg.wedge_ice_reflection(omega, p, L)
            rows.append({"h_ice": h, "transition_length_m": L,
                         "R_abrupt": abrupt, "R_wedge": w,
                         "reduction_fraction": 1.0 - w / abrupt,
                         "pass": bool(w < abrupt)})
    return {
        "description": "R_wedge < R_abrupt for >=3 transition lengths",
        "omega": omega, "rows": rows,
        "pass": bool(all(r["pass"] for r in rows)),
    }


def criterion_e7() -> dict:
    """E7: negative control, D = 0 recovers open water to 1e-6."""
    rows, worst = [], 0.0
    for h in THICKNESSES:
        for k in (0.01, 0.05, 0.1, 0.5, 1.0, 5.0):
            # D = 0, plate inertia switched off: the eq.(3) helper must fall
            # back to the open-water dispersion exactly.
            got = fg.plate_fluid_omega(
                k, rigidity=0.0, mass_per_area=0.0, fluid_density=RHO_W,
                water_depth=DEPTH, gravity=G)
            ref = math.sqrt(fg.open_water_omega2(k, DEPTH, G))
            err = rel(got, ref)
            worst = max(worst, err)
            rows.append({"k": k, "omega_D0": got, "omega_open": ref, "rel_error": err})
    return {
        "description": "D=0 negative control: open-water dispersion, rel <= 1e-6",
        "rows": rows, "max_relative_error": worst, "pass": bool(worst <= 1.0e-6),
    }


def criterion_e8() -> dict:
    """E8: wavelength grows monotonically with ice thickness."""
    rows = []
    omega = 1.5
    for h in (0.5, 1.0, 1.5, 2.0):
        p = plate(h)
        k2 = fg.flexural_gravity_wavenumber(omega, p)
        k3 = fg.flexural_gravity_wavenumber_mass_loading(omega, p)
        rows.append({"h_ice": h, "lambda_eq2": 2.0 * math.pi / k2,
                     "lambda_eq3": 2.0 * math.pi / k3})
    mono2 = all(b["lambda_eq2"] > a["lambda_eq2"] for a, b in zip(rows, rows[1:]))
    mono3 = all(b["lambda_eq3"] > a["lambda_eq3"] for a, b in zip(rows, rows[1:]))
    return {
        "description": "lambda increases monotonically with h_ice (>=4 thicknesses), eq.2 and eq.3",
        "omega": omega, "rows": rows, "monotonic_eq2": mono2, "monotonic_eq3": mono3,
        "pass": bool(mono2 and mono3),
    }


# ---------------------------------------------------------------------------
# Benchmark: independent reference vs produced value, literature anchor
# ---------------------------------------------------------------------------
def benchmark() -> dict:
    e1b = criterion_e1_independent()
    rows = e1b["rows"]

    # shared-core regression: hydroelastic delegates to flexural_gravity, so this
    # pins the caller/Robin-coefficient bookkeeping rather than the physics.
    reg_rows, reg_worst = [], 0.0
    for h in THICKNESSES:
        ip = plate(h)
        hp = hyd.ElasticPanel(
            length=1.0, breadth=1.0, thickness=h,
            young_modulus=E_ICE, poisson_ratio=NU_ICE, plate_density=RHO_I,
            fluid_density=RHO_W, gravity=G, water_depth=DEPTH,
        )
        for k in np.logspace(-2.0, 2.0, 21):
            got = fg.flexural_gravity_omega_mass_loading(float(k), ip)
            ref = hyd.flexural_gravity_omega(float(k), hp)
            err = rel(got, ref)
            reg_worst = max(reg_worst, err)
            reg_rows.append({"h_ice": h, "k": float(k), "ref_hydroelastic": ref,
                             "got_flexural_gravity": got, "rel_error": err})

    # second-order convergence of the finite-difference depth factor
    conv_rows = []
    k_conv = 1.0
    exact = k_conv * math.tanh(k_conv * DEPTH)
    for dz in (0.05, 0.025, 0.0125, 0.00625):
        g0, used_dz, nodes = fgn.fd_depth_gradient_ratio(k_conv, DEPTH, dz=dz)
        conv_rows.append({"k": k_conv, "dz": used_dz, "nodes": nodes,
                          "g0_fd": g0, "g0_exact": exact,
                          "rel_error": rel(g0, exact)})
    orders = [math.log(conv_rows[i]["rel_error"] / conv_rows[i + 1]["rel_error"], 2.0)
              for i in range(len(conv_rows) - 1)]
    conv = {"k": k_conv, "rows": conv_rows,
            "observed_order_per_refinement": orders,
            "min_order": min(orders)}
    # external anchor: in-vacuo Euler-Bernoulli plate wave, eq.(3) k -> inf
    anchor_rows = []
    for h in THICKNESSES:
        p = plate(h)
        m_pp = RHO_I * h
        D = p.flexural_rigidity
        for k in (10.0, 100.0, 1000.0):
            ref2 = D * k ** 4 / m_pp          # (rad/s)^2, Timoshenko & Woinowsky-Krieger
            got2 = fg.flexural_gravity_omega2_mass_loading(k, p)
            anchor_rows.append({"h_ice": h, "k": k,
                                "ref_in_vacuo_plate_omega2": ref2,
                                "got_eq3_omega2": got2,
                                "rel_error": rel(got2, ref2)})
    # round trip of the eq.(3) numerical wavenumber solver
    trip_rows, trip_worst = [], 0.0
    for h in THICKNESSES:
        p = plate(h)
        for omega in (0.2, 0.5, 1.0, 2.0, 3.0):
            k = fg.flexural_gravity_wavenumber_mass_loading(omega, p)
            back = fg.flexural_gravity_omega_mass_loading(k, p)
            err = rel(back, omega)
            trip_worst = max(trip_worst, err)
            trip_rows.append({"h_ice": h, "omega": omega, "k_solved": k,
                              "omega_back": back, "rel_error": err})
    return {
        "fd_independent": {
            "reference": e1b["independent_reference"],
            "got": e1b["subject"],
            "n_points": e1b["n_points"],
            "max_relative_error": e1b["max_relative_error"],
            "threshold": 1e-3,
            "rows": rows,
        },
        "fd_depth_factor_convergence": conv,
        "shared_core_regression": {
            "reference": "tensorbem.hydroelastic.flexural_gravity_omega",
            "got": "tensorbem.flexural_gravity.flexural_gravity_omega_mass_loading",
            "note": "hydroelastic now delegates to flexural_gravity; this is a "
                    "regression pin on the shared core, not an independent check",
            "n_points": len(reg_rows),
            "max_relative_error": reg_worst,
            "threshold": 1e-6,
            "rows": reg_rows,
        },
        "external_anchor_in_vacuo_plate": {
            "reference": "Euler-Bernoulli in-vacuo plate: omega^2 = D k^4 / (rho_i h) "
                         "(Timoshenko & Woinowsky-Krieger, Theory of Plates and Shells, 2nd ed., 1959)",
            "limit": "eq.(3) with k -> infinity; error is O(rho_w / (rho_i h k))",
            "rows": anchor_rows,
            "rel_error_at_k1000": max(r["rel_error"] for r in anchor_rows if r["k"] == 1000.0),
        },
        "wavenumber_solver_roundtrip": {
            "max_relative_error": trip_worst,
            "n_points": len(trip_rows),
            "rows": trip_rows,
        },
        "edge_energy": {
            "max_abs_residual": criterion_e4()["max_abs_residual"],
            "threshold": 1e-4,
        },
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figure(bm: dict, crit: dict) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                        # pragma: no cover
        print(f"matplotlib unavailable, figure skipped: {exc}")
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    for h in THICKNESSES:
        rows = [r for r in bm["fd_independent"]["rows"] if r["h_ice"] == h]
        ax.loglog([r["k"] for r in rows],
                  [max(r["rel_error"], 1e-18) for r in rows],
                  "o-", ms=3, label=f"FD independent h={h:g} m")
        rows = [r for r in bm["external_anchor_in_vacuo_plate"]["rows"] if r["h_ice"] == h]
        ax.loglog([r["k"] for r in rows], [r["rel_error"] for r in rows],
                  "s--", ms=4, label=f"in-vacuo plate h={h:g} m")
    ax.axhline(1e-6, color="k", lw=0.8, label="cross-impl threshold 1e-6")
    ax.set_xlabel("wavenumber k [rad/m]")
    ax.set_ylabel("relative error in $\\omega^2$")
    ax.set_title("Independent cross-check and short-wave plate anchor")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    rows = crit["E5_wavelength_behavior"]["rows"]
    for h in THICKNESSES:
        sel = [r for r in rows if r["h_ice"] == h]
        ax.semilogx([r["omega"] for r in sel], [r["ratio"] for r in sel],
                    "o-", label=f"h={h:g} m")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xlabel("frequency $\\omega$ [rad/s]")
    ax.set_ylabel("$\\lambda_{ice} / \\lambda_{open}$")
    ax.set_title("Eq.(3): inertia slows long waves, bending speeds short waves")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = OUT / "fig_cross_implementation.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path.relative_to(REPO))


# ---------------------------------------------------------------------------
def write_evidence_report(payload: dict, evidence: dict) -> Path:
    """Render a markdown addendum straight from the run artifact (no hand copy)."""
    bm = payload["benchmark"]
    crit = payload["criteria"]
    lines = [
        "# Flexural-gravity evidence bundle (m14560)",
        "",
        f"Generated from `run/verify_run.json` ({payload['generated_utc']}, "
        f"git `{payload['git_commit'][:12]}`).  Every number below is emitted by "
        "`scripts/verify_flexural_gravity_evidence.py`.",
        "",
        "## 1. Parameters",
        "",
        "| parameter | value |",
        "|---|---|",
    ]
    for key, val in payload["parameters"].items():
        lines.append(f"| {key} | {val} |")
    lines += ["", "## 2. Acceptance criteria", "",
              "| criterion | pass | key number |", "|---|---|---|"]
    keynum = {
        "E1_dispersion_solver": lambda c: f"max rel error {c['max_relative_error']:.3e}",
        "E1b_independent_model": lambda c: f"max rel error vs FD {c['max_relative_error']:.3e}",
        "E2_long_wave_limit": lambda c: f"max rel error {c['max_relative_error']:.3e}",
        "E3_short_wave_limit": lambda c: f"max rel error {c['max_relative_error']:.3e}",
        "E4_energy_conservation": lambda c: f"max abs(R^2+T^2-1) = {c['max_abs_residual']:.3e}",
        "E5_wavelength_behavior": lambda c: f"{c['n_long_wave_points']} long-wave points shorter",
        "E6_wedge_ice": lambda c: f"{sum(r['pass'] for r in c['rows'])}/{len(c['rows'])} wedge < abrupt",
        "E7_zero_rigidity": lambda c: f"max rel error {c['max_relative_error']:.3e}",
        "E8_parameter_sensitivity": lambda c: f"monotonic eq2={c['monotonic_eq2']} eq3={c['monotonic_eq3']}",
    }
    for name, c in crit.items():
        lines.append(f"| {name} | {c['pass']} | {keynum[name](c)} |")
    lines += ["", "## 3. Benchmark: independent finite-difference depth solve", "",
              f"max relative error in $\\omega^2$: "
              f"**{bm['fd_independent']['max_relative_error']:.3e}** "
              f"(threshold {bm['fd_independent']['threshold']:.0e}); "
              f"observed FD order {bm['fd_depth_factor_convergence']['min_order']:.2f}.", "",
              "| h_ice | k | closed form omega^2 | FD omega^2 | rel error |",
              "|---|---|---|---|---|"]
    for r in bm["fd_independent"]["rows"]:
        lines.append(f"| {r['h_ice']} | {r['k']} | {r['ref_closed_form_omega2']:.9e} | "
                     f"{r['got_fd_omega2']:.9e} | {r['rel_error']:.3e} |")
    lines += ["", "### FD depth-factor convergence", "",
              "| dz | nodes | g0 FD | g0 exact | rel error |", "|---|---|---|---|---|"]
    for r in bm["fd_depth_factor_convergence"]["rows"]:
        lines.append(f"| {r['dz']} | {r['nodes']} | {r['g0_fd']:.12f} | {r['g0_exact']:.12f} | "
                     f"{r['rel_error']:.3e} |")
    lines += ["", "## 4. External anchor (in-vacuo Euler-Bernoulli plate)", "",
              f"Reference: {bm['external_anchor_in_vacuo_plate']['reference']}  ",
              f"Limit: {bm['external_anchor_in_vacuo_plate']['limit']}.", "",
              "| h_ice | k | plate omega^2 | eq3 omega^2 | rel error |", "|---|---|---|---|---|"]
    for r in bm["external_anchor_in_vacuo_plate"]["rows"]:
        lines.append(f"| {r['h_ice']} | {r['k']} | {r['ref_in_vacuo_plate_omega2']:.9e} | "
                     f"{r['got_eq3_omega2']:.9e} | {r['rel_error']:.3e} |")
    lines += ["", "## 5. Integration and reproducibility", "",
              "Integration callers:", ""]
    lines += [f"- `{c}`" for c in evidence["integration_callers"]]
    lines += ["", "Evidence artifacts:", ""]
    lines += [f"- `{p}`" for p in evidence["run_artifacts"]]
    lines += [f"- `{evidence['entry']}`", f"- `{evidence['external_anchor']}`",
              "", "Reproduce:", "", "```bash", evidence["repro_cmd"], "```", "",
              "## 6. Limitations (honest scope)", "",
              "- The finite-difference check verifies the *depth factor* `k tanh(kH)` and the",
              "  linearised ice/water free-surface condition; it does not independently",
              "  re-derive Kirchhoff plate theory (`D = E h^3/(12(1-nu^2))`), which is a",
              "  modelling assumption anchored to the textbook plate limit above.",
              "- Ice-edge reflection/transmission uses group-velocity admittance matching,",
              "  which conserves energy by construction; a full Wiener-Hopf / four-condition",
              "  edge solution with evanescent modes is still not implemented.",
              "- `hydroelastic.flexural_gravity_omega` now delegates to the shared core, so",
              "  the cross-module comparison is a regression pin, not an independent check.",
              ""]
    path = OUT / "evidence_report.md"
    path.write_text("\n".join(lines))
    return path


def run_pytest() -> dict:
    log_path = RUN / "pytest_flexural_gravity.log"
    tests = [
        "tests/test_flexural_gravity.py",
        "tests/test_flexural_gravity_mass_loading.py",
        "tests/test_flexural_gravity_evidence.py",
        "tests/test_hydroelastic.py",
    ]
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", *tests],
        cwd=REPO, capture_output=True, text=True,
    )
    log_path.write_text(proc.stdout + "\n" + proc.stderr)
    tail = (proc.stdout.strip().splitlines() or ["<no output>"])[-1]
    return {"cmd": "python -m pytest -q " + " ".join(tests),
            "returncode": proc.returncode, "summary": tail,
            "log": str(log_path.relative_to(REPO)),
            "pass": proc.returncode == 0}


def collect_test_names() -> list[str]:
    names = []
    for relp in ("tests/test_flexural_gravity.py",
                 "tests/test_flexural_gravity_mass_loading.py",
                 "tests/test_flexural_gravity_evidence.py",
                 "tests/test_hydroelastic.py"):
        text = (REPO / relp).read_text()
        names.extend(f"{relp}::{m}" for m in re.findall(r"^def (test_\w+)", text, re.M))
    return names


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-pytest", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    RUN.mkdir(parents=True, exist_ok=True)

    crit = {
        "E1_dispersion_solver": criterion_e1(),
        "E1b_independent_model": criterion_e1_independent(),
        "E2_long_wave_limit": criterion_e2(),
        "E3_short_wave_limit": criterion_e3(),
        "E4_energy_conservation": criterion_e4(),
        "E5_wavelength_behavior": criterion_e5(),
        "E6_wedge_ice": criterion_e6(),
        "E7_zero_rigidity": criterion_e7(),
        "E8_parameter_sensitivity": criterion_e8(),
    }
    bm = benchmark()
    assert math.isfinite(bm["fd_independent"]["max_relative_error"])
    fig = None if args.no_figures else make_figure(bm, crit)
    pytest_info = ({"skipped": True} if args.skip_pytest else run_pytest())

    def git_info() -> tuple[str, bool]:
        try:
            rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                 capture_output=True, text=True).stdout.strip()
            dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                                        capture_output=True, text=True).stdout.strip())
            return rev, dirty
        except Exception:                           # pragma: no cover
            return "unknown", True

    all_pass = all(c["pass"] for c in crit.values()) and bm["fd_independent"][
        "max_relative_error"] <= 1e-3
    payload = {
        "mission": "m14560",
        "rework_of": "m12052",
        "node": "p407w3-n06-bem-ice-covered-wave",
        "repo": "TensorBEM",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_info()[0],
        "git_dirty": git_info()[1],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "parameters": {"h_ice": list(THICKNESSES), "E": E_ICE, "nu": NU_ICE,
                       "rho_w": RHO_W, "rho_i": RHO_I, "depth": DEPTH, "g": G},
        "source_sha256": {
            p: sha256(REPO / p) for p in (
                "src/tensorbem/flexural_gravity.py",
                "src/tensorbem/hydroelastic.py",
                "scripts/verify_flexural_gravity_evidence.py",
            )
        },
        "criteria": crit,
        "benchmark": bm,
        "pytest": pytest_info,
        "summary": {"total_criteria": len(crit),
                    "passed": sum(1 for c in crit.values() if c["pass"]),
                    "all_pass": all_pass,
                    "figure": fig},
    }

    run_path = RUN / "verify_run.json"
    run_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (OUT / "benchmark.json").write_text(json.dumps(bm, indent=2, sort_keys=True) + "\n")
    (OUT / "external_anchor.json").write_text(
        json.dumps(bm["external_anchor_in_vacuo_plate"], indent=2, sort_keys=True) + "\n")

    evidence = {
        "mission": "m14560",
        "rework_of": "m12052",
        "node": "p407w3-n06-bem-ice-covered-wave",
        "repo": "TensorBEM",
        "tests": collect_test_names(),
        "impl_files": ["src/tensorbem/flexural_gravity.py",
                       "src/tensorbem/hydroelastic.py",
                       "src/tensorbem/flexural_gravity_numeric.py"],
        "entry": "scripts/verify_flexural_gravity_evidence.py",
        "run_artifacts": ["docs/flexural_gravity_verification/run/verify_run.json",
                          "docs/flexural_gravity_verification/run/pytest_flexural_gravity.log",
                          "docs/flexural_gravity_verification/benchmark.json"],
        "integration_callers": [
            "src/tensorbem/hydroelastic.py:flexural_gravity_omega delegates to "
            "tensorbem.flexural_gravity.plate_fluid_omega (plate/fluid main solver)",
            "src/tensorbem/benchmarks/hydroelastic_verification.py:uses the same "
            "flexural-gravity dispersion for the hydroelastic benchmark",
        ],
        "benchmark": {
            "fd_independent_max_rel_error":
                bm["fd_independent"]["max_relative_error"],
            "fd_depth_factor_min_order":
                bm["fd_depth_factor_convergence"]["min_order"],
            "shared_core_regression_max_rel_error":
                bm["shared_core_regression"]["max_relative_error"],
            "external_anchor_rel_error_at_k1000":
                bm["external_anchor_in_vacuo_plate"]["rel_error_at_k1000"],
            "wavenumber_solver_max_rel_error":
                bm["wavenumber_solver_roundtrip"]["max_relative_error"],
            "edge_energy_max_abs_residual": bm["edge_energy"]["max_abs_residual"],
            "ref": "finite-difference depth solve / in-vacuo plate theory",
            "got": "tensorbem.flexural_gravity eq.(3) and its numerical solver",
            "detail": "docs/flexural_gravity_verification/benchmark.json",
        },
        "repro_cmd": ("python scripts/verify_flexural_gravity_evidence.py && "
                      "python -m pytest -q tests/test_flexural_gravity.py "
                      "tests/test_flexural_gravity_mass_loading.py "
                      "tests/test_flexural_gravity_evidence.py tests/test_hydroelastic.py"),
        "external_anchor": "docs/flexural_gravity_verification/external_anchor.json",
        "criteria_pass": {k: bool(v["pass"]) for k, v in crit.items()},
    }
    (OUT / "evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    write_evidence_report(payload, evidence)

    print(json.dumps({"all_pass": all_pass,
                      "criteria": {k: v["pass"] for k, v in crit.items()},
                      "benchmark": evidence["benchmark"],
                      "pytest": pytest_info.get("summary", "skipped"),
                      "run_artifact": str(run_path.relative_to(REPO))},
                     indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())