#!/usr/bin/env python3
"""m17429 board audit of m17373 (CASE-BEM-ICE-DISPERSION).

Independent re-measurement.  Nothing is taken on trust from the mission's own
driver: every number below is recomputed here from the *published* artifacts and
from the delivered code, using code that does not import the mission driver.

Routes
------
R1  registered case row  : num  n/a  -- reproduce the quantities named in
    ``tensor_benchmark_cases.metric`` for the anchor (deep/shallow k error,
    RK4 shooting, Robin residual, thickness study) straight from the closed-form
    module ``tensorbem.ice``.
R2  independent dispersion check : ref recomputed from Weitz-Keller closed form
    evaluated here, got read from the mission's ``numbers.json``.
R3  independent BEM convergence : own ppw ladder calling
    ``strip_bem_depth_factor`` (the mission's own convergence_study is not used).
R4  reproduce the mission driver : numbers.json regenerated in a scratch tree
    and compared field by field (``environment`` excluded).
R5  test suite : pytest on the delivered tests.

Usage:  python3 docs/audit_m17429/audit_m17373.py
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import sys

WS = pathlib.Path(__file__).resolve().parents[2]          # audit worktree (m17429)
SRC_MISSION = pathlib.Path("/tmp/aud17373")               # byte-identical copy of m17373 @26cbde4
MISSION_WT = pathlib.Path("/root/tp-agent/work/m17373")   # m17373 work tree (read-only)
OUT = WS / "docs" / "audit_m17429"
NUMBERS = MISSION_WT / "docs/case_bem_ice_dispersion/numbers.json"

sys.path.insert(0, str(SRC_MISSION / "src"))

G, MU_DEFAULT = 9.80665, None  # taken from the artifact parameter table


def sha256(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def weitz_keller_omega(k: float, H: float, mu: float, g: float) -> float:
    """Closed form, written here from the report's appendix (not imported)."""
    th = math.tanh(k * H)
    return math.sqrt(g * k * th / (1.0 + mu * k * th))


def r2_independent_series(numbers: dict) -> dict:
    P = numbers["parameters"]
    g, mu = P["gravity"], P["mu"]
    rows, worst = [], 0.0
    for series in ("deep", "shallow"):
        for r in numbers["series"][series]["rows"]:
            k, H = r["k"], r["depth"]
            om_ref = weitz_keller_omega(k, H, mu, g)
            rel = abs(r["omega_got"] - om_ref) / om_ref
            df_ref = k * math.tanh(k * H)
            df_rel = abs(r["depth_factor_bem"] - df_ref) / df_ref
            # the mission reports its own ref; cross-check the two refs agree
            ref_agrees = abs(r["omega_ref"] - om_ref) / om_ref
            worst = max(worst, rel)
            rows.append(dict(series=series, k=k, depth=H, kH=k * H,
                             omega_got=r["omega_got"], omega_ref_audit=om_ref,
                             omega_ref_mission=r["omega_ref"],
                             ref_route_rel_diff=ref_agrees,
                             omega_rel_error=rel,
                             depth_factor_bem=r["depth_factor_bem"],
                             depth_factor_ref=df_ref,
                             depth_factor_rel_error=df_rel,
                             n_panels=r["n_panels"],
                             panels_per_wavelength=r["panels_per_wavelength"]))
    return dict(rows=rows, n_rows=len(rows), max_omega_rel_error=worst,
                tolerance=0.05, all_within_tolerance=worst <= 0.05)


def r3_independent_convergence() -> dict:
    from tensorbem import ice_mass_loading_dispersion as disp

    levels = (8, 16, 32, 64, 128)
    out = {}
    for name, (k, H) in (("deep", (0.1, 200.0)), ("intermediate", (0.1, 20.0)),
                         ("shallow", (0.1, 2.0))):
        errs, panels = [], []
        for ppw in levels:
            s = disp.strip_bem_depth_factor(k, H, panels_per_wavelength=ppw)
            errs.append(abs(s.depth_factor_rq / s.exact_depth_factor - 1.0))
            panels.append(s.n_panels)
        orders = [math.log(errs[i] / errs[i + 1]) / math.log(2.0) for i in range(len(errs) - 1)]
        out[name] = dict(k=k, depth=H, panels_per_wavelength=list(levels),
                         n_panels=panels, rel_error=errs,
                         observed_order=orders,
                         monotone=all(errs[i] > errs[i + 1] for i in range(len(errs) - 1)),
                         last_error=errs[-1])
    return out


def r1_registered_metric() -> dict:
    """Quantities named in tensor_benchmark_cases.metric for this anchor.

    The registered row is a prose summary ("deep-water k rel err 4.50e-14;
    shallow k rel err 2.87e-02 (kH=0.426 ...); thickness 0.0%; RK4 1.03e-13;
    Robin 4.47e-15") with a single <5% tolerance -- it carries no per-item
    ref/got pair, so each item is re-derived here from the delivered
    ``tensorbem.ice`` API with an explicitly stated definition.
    """
    from tensorbem import ice as ice_module

    ICE = ice_module.IceCover(thickness=1.0, water_depth=200.0)
    out = {}

    # (a) two independent root solvers on the same Robin residual
    def residual_factory(omega, ice):
        return lambda k: ice_module.robin_residual(k, omega, ice)

    def brentq(f, lo, hi, tol=1e-15):
        flo, fhi = f(lo), f(hi)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            fmid = f(mid)
            if flo * fmid <= 0.0:
                hi, fhi = mid, fmid
            else:
                lo, flo = mid, fmid
            if hi - lo < tol * max(1.0, abs(mid)):
                break
        return 0.5 * (lo + hi)

    worst = 0.0
    for period in (4.0, 6.0, 10.0, 14.0, 30.0):
        omega = 2.0 * math.pi / period
        k_newton = ice_module.mass_loading_wavenumber(omega, ICE).wavenumber
        f = residual_factory(omega, ICE)
        k_bisect = brentq(f, 1e-8, 5.0)
        worst = max(worst, abs(k_bisect - k_newton) / k_newton)
    out["deep_water_k_solver_agreement"] = worst

    # (b) deep-water branch solved independently: omega^2 = g k / (1 + mu k)
    worst_deep = 0.0
    for period in (4.0, 6.0, 10.0, 14.0):
        omega = 2.0 * math.pi / period
        k = ice_module.mass_loading_wavenumber(omega, ICE).wavenumber
        g, mu = ICE.gravity, ICE.mu
        k_deep = brentq(lambda x: g * x / (1.0 + mu * x) - omega * omega, 1e-8, 50.0)
        worst_deep = max(worst_deep, abs(k - k_deep) / k_deep)
    out["deep_water_branch_k_rel_err"] = worst_deep

    # (c) shallow long-wave deficit at kH = 0.426 (registered: 2.87 %)
    shallow = ice_module.IceCover(thickness=0.9, water_depth=6.0)
    omega = 2.0 * math.pi / 12.0
    wave = ice_module.mass_loading_wavenumber(omega, shallow)
    k = wave.wavenumber
    k_long = omega / math.sqrt(shallow.gravity * shallow.water_depth)
    out["shallow_k_rel_err_vs_long_wave"] = abs(k - k_long) / k_long
    out["shallow_kH"] = k * shallow.water_depth
    out["shallow_omega_recovered_rel_err"] = abs(
        ice_module.mass_loading_omega(k, shallow) - omega) / omega
    scan = {}
    for h_i in (0.3, 0.6, 0.9, 1.0):
        cov = ice_module.IceCover(thickness=h_i, water_depth=6.0)
        kk = ice_module.mass_loading_wavenumber(omega, cov).wavenumber
        scan[f"h_i={h_i}"] = dict(kH=kk * 6.0,
                                  long_wave_rel_err=abs(kk - k_long) / k_long)
    out["shallow_thickness_scan"] = scan

    # (d) RK4 shooting of W'' = k^2 W versus the Robin root
    worst_rk4 = 0.0
    for period in (4.0, 6.0, 10.0, 14.0):
        omega = 2.0 * math.pi / period
        k = ice_module.mass_loading_wavenumber(omega, ICE).wavenumber
        om_shoot = ice_module.shooting_omega(k, ICE, steps=600)
        worst_rk4 = max(worst_rk4, abs(om_shoot - omega) / omega)
    out["rk4_shooting_rel_err"] = worst_rk4

    # (e) Robin residual at the solved root
    worst_robin = 0.0
    for period in (3.5, 5.0, 8.0, 12.0):
        omega = 2.0 * math.pi / period
        k = ice_module.mass_loading_wavenumber(omega, ICE).wavenumber
        worst_robin = max(worst_robin, abs(ice_module.robin_residual(k, omega, ICE)))
    out["robin_residual_abs"] = worst_robin

    # (f) thickness study (k(h_i) versus the independent Robin root)
    study = ice_module.thickness_study(2.0 * math.pi / 6.0, [0.3, 0.6, 0.9, 1.5, 2.5], ICE)
    out["thickness_study_max_rel_diff"] = float(max(study["relative_difference"]))
    out["thickness_study_monotone_k"] = bool(
        all(study["k_ice"][i + 1] > study["k_ice"][i] for i in range(len(study["k_ice"]) - 1)))
    return out


def r4_reproduction(regen: pathlib.Path) -> dict:
    a = json.loads(NUMBERS.read_text())
    b = json.loads(regen.read_text())
    a.pop("environment", None)
    b.pop("environment", None)
    sa = json.dumps(a, sort_keys=True)
    sb = json.dumps(b, sort_keys=True)
    return dict(committed_sha256=hashlib.sha256(sa.encode()).hexdigest(),
                regenerated_sha256=hashlib.sha256(sb.encode()).hexdigest(),
                identical=sa == sb,
                environment_committed=json.loads(NUMBERS.read_text()).get("environment"),
                environment_regenerated=json.loads(regen.read_text()).get("environment"))


def main() -> int:
    numbers = json.loads(NUMBERS.read_text())
    metrics = dict(
        audit_mission=17429, audited_mission=17373, repo="TensorBEM",
        anchor="CASE-BEM-ICE-DISPERSION",
        audited_commit="26cbde4516fcf6756105ec6584dcd097b51b399d",
        audited_branch="agent/m17373",
        provenance={
            "impl_src_sha256": sha256(SRC_MISSION / "src/tensorbem/ice_mass_loading_dispersion.py"),
            "driver_sha256": sha256(SRC_MISSION / "src/tensorbem/benchmarks/ice_mass_loading_dispersion.py"),
            "tests_sha256": sha256(SRC_MISSION / "tests/test_ice_mass_loading_dispersion.py"),
            "numbers_json_sha256": sha256(NUMBERS),
            "scratch_tree": str(SRC_MISSION),
        },
        R1_registered_case_metric=r1_registered_metric(),
        R2_independent_dispersion=r2_independent_series(numbers),
        R3_independent_convergence=r3_independent_convergence(),
        R4_reproduction=r4_reproduction(
            SRC_MISSION / "docs/case_bem_ice_dispersion/numbers.json"),
        R5_tests="see logs/repro_tests.log (31 passed, 16.2 s)",
    )
    (OUT / "audit_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    print("R1 registered metric (tensor_benchmark_cases row, tolerance <5%):")
    for k, v in metrics["R1_registered_case_metric"].items():
        print(f"   {k:38s} {v if isinstance(v, dict) else format(v, '.6g')}")
    print("\nR2 independent Weitz-Keller check (ref recomputed here):")
    print(f"   {'series':7s}{'k':>7s}{'H':>7s}{'omega_got':>12s}{'omega_ref':>12s}{'rel%':>9s}"
          f"{'df_bem':>11s}{'df_ref':>11s}{'dfrel%':>8s}{'npan':>6s}")
    for r in metrics["R2_independent_dispersion"]["rows"]:
        print(f"   {r['series']:7s}{r['k']:>7.3g}{r['depth']:>7.0f}{r['omega_got']:>12.7f}"
              f"{r['omega_ref_audit']:>12.7f}{100*r['omega_rel_error']:>9.4f}"
              f"{r['depth_factor_bem']:>11.6f}{r['depth_factor_ref']:>11.6f}"
              f"{100*r['depth_factor_rel_error']:>8.4f}{r['n_panels']:>6.0f}")
    print(f"   max rel err = {metrics['R2_independent_dispersion']['max_omega_rel_error']:.5f}"
          f"  (tol {metrics['R2_independent_dispersion']['tolerance']})")
    print("\nR3 independent BEM convergence (own ppw ladder):")
    for name, d in metrics["R3_independent_convergence"].items():
        print(f"   {name:12s} err%=" + " ".join(f"{100*e:7.4f}" for e in d["rel_error"])
              + f"  orders=" + " ".join(f"{o:5.2f}" for o in d["observed_order"])
              + f"  monotone={d['monotone']}")
    print("\nR4 reproduction:", metrics["R4_reproduction"]["identical"],
          metrics["R4_reproduction"]["committed_sha256"][:16],
          metrics["R4_reproduction"]["regenerated_sha256"][:16])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())