#!/usr/bin/env python3
"""Adversarial probe of m20091's CASE-BEM-CRACK-SIF delivery (read-only import).

Import path points at the audited worktree; nothing there is modified.
Part A cross-examines the auditee code as delivered (default route + constant-DD
route up to n=640, the largest mesh that ran inside the audit budget).
Part B is an independent vectorised re-implementation of the same hypersingular
DD-BEM scheme so the mesh ladder can be pushed past n=640 and the auditee
assembly can be checked matrix-for-matrix.

Writes docs/audit_m20091/probe_result.json.
"""
from __future__ import annotations

import ast
import json
import math
import pathlib
import sys

import numpy as np

AUDITEE = pathlib.Path("/root/tp-agent/work/m20091")
OUT = pathlib.Path(__file__).with_name("probe_result.json")
sys.path.insert(0, str(AUDITEE / "src"))

from tensorbem.crack_sif import (  # noqa: E402
    analytical_cod,
    analytical_sif,
    solve_crack_sif,
)

SIGMA, A, E, NU = 100.0e6, 0.01, 2.0e11, 0.3
E_PRIME = E / (1.0 - NU ** 2)
K_REF = analytical_sif(SIGMA, A)
out: dict = {"auditee": str(AUDITEE), "params": {"sigma_Pa": SIGMA, "a_m": A, "E_Pa": E, "nu": NU}}


# ---------------------------------------------------------------- P1 identity
def p1_tautology() -> dict:
    """Default route: is K_got algebraically identical to the closed-form K_ref?"""
    rng = np.random.default_rng(20261111)
    rel = []
    for _ in range(120):
        sigma = float(rng.uniform(1e5, 1e9))
        a = float(10 ** rng.uniform(-3, 0))
        r = solve_crack_sif(sigma=sigma, a=a, young=E, poisson=NU, method="chebyshev")
        rel.append(abs(r.sif_got - analytical_sif(sigma, a)) / analytical_sif(sigma, a))
    ks = {}
    for n in (4, 40, 200, 1000):
        r = solve_crack_sif(sigma=SIGMA, a=A, young=E, poisson=NU, n_elements=n, method="chebyshev")
        ks[n] = r.sif_got
    r = solve_crack_sif(sigma=SIGMA, a=A, young=E, poisson=NU, method="chebyshev")
    src = (AUDITEE / "src/tensorbem/crack_sif.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_chebyshev_solve")
    attrs = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    mods = {n.value.id for n in ast.walk(fn)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    return {
        "n_random_cases": len(rel),
        "max_rel_diff_vs_closed_form": max(rel),
        "n_within_machine_eps": sum(1 for x in rel if x <= 2.3e-16),
        "k_got_by_n_elements": ks,
        "k_got_spread_over_mesh_Pa": max(ks.values()) - min(ks.values()),
        "chebyshev_path_modules_used": sorted(mods),
        "chebyshev_path_linalg_attrs": sorted(a for a in attrs if a in ("solve", "inv", "lstsq", "norm")) or "none",
        "chebyshev_seed_assignment_present": "c_0 = 4.0 * sigma / e_prime" in src,
        "reported_error_pct": r.error_pct,
        "extracted_c0": float(r.chebyshev_coefficients[0]),
        "analytic_c0_4sigma_over_Ep": 4.0 * SIGMA / E_PRIME,
    }


# ------------------------------------------------------- Part A: auditee route
def p2_constant_dd() -> dict:
    ladder = []
    for n in (10, 20, 40, 80, 160, 320, 640):
        r = solve_crack_sif(sigma=SIGMA, a=A, young=E, poisson=NU, n_elements=n, method="constant")
        ref_cod = analytical_cod(r.cod_profile_x, SIGMA, A, E_PRIME)
        dd_rel = float(np.linalg.norm(r.element_dd - ref_cod) / np.linalg.norm(ref_cod))
        ladder.append({
            "n_elements": n,
            "K_MPa_sqrt_m": r.sif_got / 1e6,
            "error_pct": r.error_pct,
            "cod_rel_err_pct": 100.0 * dd_rel,
            "cond": r.condition_number,
            "residual_rel": r.residual_l2 / SIGMA,
        })
    sens = {}
    for n_fit in (1, 2, 3, 5, 9):
        r = solve_crack_sif(sigma=SIGMA, a=A, young=E, poisson=NU,
                            n_elements=200, method="constant", n_fit=n_fit)
        sens[n_fit] = round(r.error_pct, 4)
    return {"ref_MPa_sqrt_m": K_REF / 1e6, "ladder": ladder,
            "n_fit_sensitivity_error_pct_at_n200": sens}


# ------------------------------------ Part B: independent vectorised replicate
def _assemble(n_half: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, n_half + 1)
    x_half = A * (1.0 - np.cos(t * math.pi / 2.0))
    edges = np.sort(np.concatenate([-x_half[::-1][:-1], x_half]))
    edges[0], edges[-1] = -A, A
    centers = 0.5 * (edges[:-1] + edges[1:])
    h = np.diff(edges)
    n = len(centers)
    Aop = E_PRIME / (4.0 * math.pi) * (1.0 / (centers[:, None] - edges[:-1][None, :])
                                       - 1.0 / (centers[:, None] - edges[1:][None, :]))
    np.fill_diagonal(Aop, 0.0)
    Aop[np.diag_indices(n)] = E_PRIME / (math.pi * h)          # finite-part self term
    return Aop, centers, edges


def _k_from_tip(dd, centers, edges, n_fit):
    def one(ctrs, tip_idx, r):
        c = float(np.mean(dd[tip_idx] / np.sqrt(r[tip_idx])))
        return E_PRIME * c * math.sqrt(math.pi / 32.0)
    r_r = A - centers
    r_l = centers + A
    return 0.5 * (one(centers, slice(-n_fit, None), r_r) + one(centers, slice(0, n_fit), r_l))


def p3_independent() -> dict:
    rows = []
    for n_half in (5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560):
        Aop, centers, edges = _assemble(n_half)
        rhs = SIGMA * np.ones(len(centers))
        dd = np.linalg.solve(Aop, rhs)
        k = _k_from_tip(dd, centers, edges, 5)
        ref_cod = analytical_cod(centers, SIGMA, A, E_PRIME)
        rows.append({
            "n_elements_approx": len(centers),
            "K_MPa_sqrt_m": k / 1e6,
            "error_pct": abs(k - K_REF) / K_REF * 100.0,
            "cod_rel_err_pct": 100.0 * float(np.linalg.norm(dd - ref_cod) / np.linalg.norm(ref_cod)),
        })
    # cross-check the replicate against the delivered code at n_elements=200
    r = solve_crack_sif(sigma=SIGMA, a=A, young=E, poisson=NU, n_elements=200, method="constant")
    aop, centers, _ = _assemble(100)
    dd = np.linalg.solve(aop, SIGMA * np.ones(len(centers)))
    match = float(np.max(np.abs(dd - r.element_dd) / np.abs(r.element_dd).max()))
    return {"ladder": rows, "replicate_vs_auditee_max_rel_dd_diff": match,
            "ref_MPa_sqrt_m": K_REF / 1e6}


out["P1_default_route_is_algebraic_identity"] = p1_tautology()
out["P2_auditee_constant_dd"] = p2_constant_dd()
out["P3_independent_replicate"] = p3_independent()

lad = out["P2_auditee_constant_dd"]["ladder"]
ind = out["P3_independent_replicate"]["ladder"]
out["P4_acceptance"] = {
    "platform_standard_pct": 3.0,
    "ledger_standard_pct": 5.0,
    "delivered_default_error_pct": 0.0,
    "delivered_default_error_is_tautology": True,
    "auditee_numeric_best_error_pct": min(x["error_pct"] for x in lad),
    "auditee_numeric_finest_error_pct": lad[-1]["error_pct"],
    "auditee_numeric_converging": all(b["error_pct"] <= a["error_pct"] for a, b in zip(lad, lad[1:])),
    "independent_finest_error_pct": ind[-1]["error_pct"],
    "independent_min_error_pct": min(x["error_pct"] for x in ind),
    "K_within_3pct_by_numeric_route": min(x["error_pct"] for x in ind) < 3.0,
    "verdict": "numeric route not convergent; algebraic route not a computation",
}
OUT.write_text(json.dumps(out, indent=2))
print(json.dumps({k: out[k] for k in ("P1_default_route_is_algebraic_identity", "P4_acceptance")}, indent=2))
print("wrote", OUT)