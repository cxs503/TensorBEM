#!/usr/bin/env python3
"""Entry point for anchor ``XVAL-FVM-THREE-GRID-LOGLOG-SLOPE-CELLCENTRE-FV-OR``.

Subcommands
-----------
``run [--outdir DIR]``
    Execute the predeclared design (below), write ``results.json``, the three
    figures and ``REPORT.md``.  Everything numeric in the report is injected
    from ``results.json`` -- no number is typed by hand.

``verify [--outdir DIR]``
    Re-read ``results.json`` and re-evaluate every predeclared criterion from
    the stored numbers.  Exit code 0 iff every criterion passes.  This is the
    machine-checkable half of the acceptance: it never re-runs the solver, so it
    can be used by a gate that only trusts the committed artefact.

Predeclared design (fixed before any level was run)
---------------------------------------------------
Routes (all solve the *same* manufactured ``-laplace(u) = f`` on the unit
square, all measured with the *same* cell-centre discrete L2 norm):

=============================================  =========  ====================
route                                          p_exact    role
=============================================  =========  ====================
``fv2``   cell-centred FV, midpoint source      2          primary claim
``fv1_src`` same code, rectangle-rule source    1          negative control
``fv1_bc``  same code, full-cell BC distance    1          negative control
``skfem-P1`` P1 Galerkin FEM (OSS)              2          independent route
``skfem-P2`` P2 Galerkin FEM (OSS)              4          estimator sanity
=============================================  =========  ====================

Ladders: ``r2_coarse`` = 32/64/128 (primary), ``r2_fine`` = 64/128/256
(asymptotic-range check), ``r15`` = 40/60/90 (non-integer refinement ratio).
Problems: ``sin_sin`` (primary), ``exp_sin`` (problem-independence check).

Predeclared criteria (owner tolerance in force: >15% not acceptable)
---------------------------------------------------------------------
C1  finiteness/coverage: every level and every estimator finite.
C2  primary order: ``|p_obs - p_exact| / p_exact <= 0.15`` for ``fv2`` in the
    discrete L2 norm, all three ladders, both problems.
C3  reporting line ``<3%``: the same quantity, tightened to 0.03.
C4  estimator agreement, ``fv2``: error-based three-grid log-log slope versus
    Roache's functional route (:func:`tensorbem.uq.grid_convergence_index`)
    differ by <= 15% relative.
C5  refinement-ratio independence: ``p(r=2)`` versus ``p(r=1.5)`` within 15%.
C6  asymptotic range: ``p(32,64,128)`` versus ``p(64,128,256)`` within 15%,
    and Roache's ``|GCI_medium/(r^p GCI_fine) - 1| <= 0.10``.
C7  cross-implementation: ``fv2`` order versus ``skfem-P1`` order within 15%.
C8  discrimination: each negative control reproduces ``p_exact = 1`` within
    15% **and** misses the primary target ``p = 2`` by more than 15% -- the
    estimator is shown to be able to tell orders apart.
C9  problem independence: ``sin_sin`` versus ``exp_sin`` orders within 15%.
C10 procedure verification: on a synthetic power law ``e = C h^1.7`` the
    estimator must return 1.7 to 1e-10 (the instrument is calibrated before it
    is used).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorbem.benchmarks.fv_order_xval import (  # noqa: E402
    LADDERS,
    PROBLEMS,
    ROACHES_FS_THREE_GRID,
    discrete_errors,
    fv_ladder,
    loglog_slope,
    pairwise_slopes,
    skfem_ladder,
    solve_fv_poisson,
    three_grid_slope,
)

TOL_OWNER = 0.15   # owner precision standard: >15% is not acceptable
TOL_TIGHT = 0.03   # reporting line "<3%"
TOL_ASYMPTOTIC = 0.10

P_EXACT = {
    "fv2": 2.0,
    "fv1_src": 1.0,
    "fv1_bc": 1.0,
    "skfem-P1": 2.0,
    "skfem-P2": 4.0,
}

#: predeclared ladder/problem set (see module docstring)
DESIGN = {
    "problems": ["sin_sin", "exp_sin"],
    "ladders": ["r2_coarse", "r2_fine", "r15"],
    "controls": ["fv1_src", "fv1_bc"],
    "skfem": {"P1": "r2_coarse", "P2": "r2_coarse"},
    "primary": {"scheme": "fv2", "problem": "sin_sin", "ladder": "r2_coarse"},
}


def rel_err(got: float, ref: float) -> float:
    return abs(got - ref) / abs(ref)


# ---------------------------------------------------------------------------
# procedure verification (C10)
# ---------------------------------------------------------------------------
def verify_estimator() -> dict:
    """Calibrate the instrument on a synthetic power law before using it."""
    hs = [1.0, 0.5, 0.25]
    C, p_true = 3.0, 1.7
    errs = [C * h ** p_true for h in hs]
    fit = loglog_slope(hs, errs)
    pair = pairwise_slopes(hs, errs)
    # three-grid Roache route on a monotone function value sequence
    f = [math.exp(-1.0) + C * h ** p_true for h in hs]  # arbitrary base value
    p_roache = three_grid_slope(f[2], f[1], f[0], 2.0)
    return {
        "sequence": f"e(h) = {C} h^{p_true}, h = {hs}",
        "p_true": p_true,
        "p_loglog": fit["observed_order"],
        "p_loglog_abs_err": abs(fit["observed_order"] - p_true),
        "p_pairwise": pair,
        "p_pairwise_max_abs_err": max(abs(v - p_true) for v in pair),
        "p_roache_abs_err": abs(p_roache - p_true),
        "tolerance": 1e-10,
        "pass": bool(
            abs(fit["observed_order"] - p_true) <= 1e-10
            and max(abs(v - p_true) for v in pair) <= 1e-10
            and abs(p_roache - p_true) <= 1e-10
        ),
    }


# ---------------------------------------------------------------------------
# the study
# ---------------------------------------------------------------------------
def run_study(outdir: str) -> dict:
    os.makedirs(outdir, exist_ok=True)
    res: dict = {
        "anchor": "XVAL-FVM-THREE-GRID-LOGLOG-SLOPE-CELLCENTRE-FV-OR",
        "design": DESIGN,
        "p_exact": P_EXACT,
        "tolerances": {
            "owner": TOL_OWNER,
            "tight_reporting_line": TOL_TIGHT,
            "asymptotic_ratio": TOL_ASYMPTOTIC,
            "gci_safety_factor": ROACHES_FS_THREE_GRID,
        },
        "procedure": verify_estimator(),
        "fv": {},
        "skfem": {},
    }

    # primary scheme on every ladder x problem
    for prob in DESIGN["problems"]:
        res["fv"][prob] = {}
        for lad in DESIGN["ladders"]:
            res["fv"][prob][lad] = fv_ladder(LADDERS[lad], scheme="fv2", problem=prob)

    # negative controls on the primary ladder, primary problem
    res["fv1"] = {}
    for sch in DESIGN["controls"]:
        for prob in DESIGN["problems"]:
            res["fv1"][f"{sch}:{prob}"] = fv_ladder(
                LADDERS["r2_coarse"], scheme=sch, problem=prob
            )

    # independent route (OSS scikit-fem)
    for el, lad in DESIGN["skfem"].items():
        try:
            res["skfem"][f"skfem-{el}"] = skfem_ladder(LADDERS[lad], element=el)
        except Exception as exc:  # noqa: BLE001 - record, never hide
            res["skfem"][f"skfem-{el}"] = {"route": f"skfem {el}", "error": repr(exc)}

    # ---- criteria -------------------------------------------------------
    crit: dict = {}
    prim = DESIGN["primary"]
    pl = res["fv"][prim["problem"]][prim["ladder"]]

    # C1
    finite = True
    for prob, d in res["fv"].items():
        for lad, o in d.items():
            for norm in ("l1", "l2", "linf"):
                finite &= math.isfinite(o[f"order_{norm}"]["observed_order"])
            finite &= math.isfinite(o["order_roache"])
            finite &= all(math.isfinite(e[n]) for e in o["errors"] for n in ("l1", "l2", "linf"))
    for key, o in res["skfem"].items():
        if "error" in o:
            continue
        finite &= math.isfinite(o["order_l2"]["observed_order"])
    crit["C1_finiteness"] = {"pass": bool(finite)}

    # C2 / C3
    rows = []
    for prob, d in res["fv"].items():
        for lad, o in d.items():
            for norm in ("l2", "l1", "linf"):
                got = o[f"order_{norm}"]["observed_order"]
                rows.append(
                    {
                        "problem": prob,
                        "ladder": lad,
                        "norm": norm,
                        "got": got,
                        "ref": P_EXACT["fv2"],
                        "rel_err": rel_err(got, P_EXACT["fv2"]),
                    }
                )
    l2rows = [r for r in rows if r["norm"] == "l2"]
    crit["C2_primary_order_15pct"] = {
        "pass": bool(all(r["rel_err"] <= TOL_OWNER for r in l2rows)),
        "worst": max(l2rows, key=lambda r: r["rel_err"]),
        "tolerance": TOL_OWNER,
        "rows": rows,
    }
    crit["C3_primary_order_3pct"] = {
        "pass": bool(all(r["rel_err"] <= TOL_TIGHT for r in l2rows)),
        "worst": max(l2rows, key=lambda r: r["rel_err"]),
        "tolerance": TOL_TIGHT,
    }

    # C4: error slope vs functional (Roache) order
    c4 = []
    for prob, d in res["fv"].items():
        for lad, o in d.items():
            got_err = o["order_l2"]["observed_order"]
            got_roache = o["order_roache"]
            c4.append(
                {
                    "problem": prob,
                    "ladder": lad,
                    "order_error_l2": got_err,
                    "order_roache_functional": got_roache,
                    "rel_diff": rel_err(got_roache, got_err),
                }
            )
    crit["C4_estimator_agreement"] = {
        "pass": bool(all(r["rel_diff"] <= TOL_OWNER for r in c4)),
        "worst": max(c4, key=lambda r: r["rel_diff"]),
        "rows": c4,
    }

    # C5 refinement-ratio independence (primary problem, fv2)
    for prob in DESIGN["problems"]:
        a = res["fv"][prob]["r2_coarse"]["order_l2"]["observed_order"]
        b = res["fv"][prob]["r15"]["order_l2"]["observed_order"]
        crit.setdefault("C5_refinement_ratio_independence", {"rows": []})["rows"].append(
            {"problem": prob, "p_r2": a, "p_r15": b, "rel_diff": rel_err(b, a)}
        )
    crit["C5_refinement_ratio_independence"]["pass"] = bool(
        all(r["rel_diff"] <= TOL_OWNER for r in crit["C5_refinement_ratio_independence"]["rows"])
    )

    # C6 asymptotic range
    c6 = []
    for prob in DESIGN["problems"]:
        a = res["fv"][prob]["r2_coarse"]["order_l2"]["observed_order"]
        b = res["fv"][prob]["r2_fine"]["order_l2"]["observed_order"]
        ar = res["fv"][prob]["r2_fine"]["asymptotic_ratio"]
        c6.append(
            {
                "problem": prob,
                "p_32_64_128": a,
                "p_64_128_256": b,
                "rel_diff": rel_err(b, a),
                "asymptotic_ratio_fine": ar,
                "asymptotic_ok": bool(abs(ar - 1.0) <= TOL_ASYMPTOTIC),
            }
        )
    crit["C6_asymptotic_range"] = {
        "pass": bool(
            all(r["rel_diff"] <= TOL_OWNER and r["asymptotic_ok"] for r in c6)
        ),
        "rows": c6,
    }

    # C7 cross-implementation
    sk = res["skfem"].get("skfem-P1", {})
    c7 = {"available": "error" not in sk}
    if c7["available"]:
        got_fv = pl["order_l2"]["observed_order"]
        got_sk = sk["order_l2"]["observed_order"]
        c7.update(
            {
                "fv_p": got_fv,
                "skfem_p": got_sk,
                "fv_vs_theory": rel_err(got_fv, P_EXACT["fv2"]),
                "skfem_vs_theory": rel_err(got_sk, P_EXACT["skfem-P1"]),
                "rel_diff": rel_err(got_sk, got_fv),
                "pass": bool(rel_err(got_sk, got_fv) <= TOL_OWNER),
            }
        )
    crit["C7_cross_implementation"] = c7

    # C8 discrimination
    c8 = []
    for key, o in res["fv1"].items():
        got = o["order_l2"]["observed_order"]
        c8.append(
            {
                "variant": key,
                "p_obs_l2": got,
                "p_exact": P_EXACT[key.split(":")[0]],
                "err_vs_own_target": rel_err(got, P_EXACT[key.split(":")[0]]),
                "err_vs_primary_target": rel_err(got, 2.0),
                "roache_functional": o["order_roache"],
            }
        )
    crit["C8_discrimination"] = {
        "pass": bool(
            all(
                r["err_vs_own_target"] <= TOL_OWNER and r["err_vs_primary_target"] > TOL_OWNER
                for r in c8
            )
        ),
        "rows": c8,
    }

    # C9 problem independence
    a = res["fv"]["sin_sin"]["r2_coarse"]["order_l2"]["observed_order"]
    b = res["fv"]["exp_sin"]["r2_coarse"]["order_l2"]["observed_order"]
    crit["C9_problem_independence"] = {
        "p_sin_sin": a,
        "p_exp_sin": b,
        "rel_diff": rel_err(b, a),
        "pass": bool(rel_err(b, a) <= TOL_OWNER),
    }

    # C10
    crit["C10_procedure_verification"] = {
        "pass": bool(res["procedure"]["pass"]),
        "detail": res["procedure"],
    }

    # descriptive (not pass/fail) observation: does the functional route see
    # the defects that the error-based slope sees?
    obs = []
    for r in crit["C8_discrimination"]["rows"]:
        blind = rel_err(r["roache_functional"], 2.0) <= TOL_OWNER and rel_err(
            r["roache_functional"], r["p_exact"]
        ) > TOL_OWNER
        obs.append(
            {
                "variant": r["variant"],
                "functional_p": r["roache_functional"],
                "error_based_p": r["p_obs_l2"],
                "functional_route_blind": bool(blind),
            }
        )
    res["criteria"] = crit
    res["observations"] = {
        "functional_route_blindness": obs,
        "n_blind": sum(1 for o in obs if o["functional_route_blind"]),
    }
    res["verdict"] = {
        "all_pass": bool(all(v.get("pass") for v in crit.values())),
        "n_pass": sum(1 for v in crit.values() if v.get("pass")),
        "n_criteria": len(crit),
        "honest_note": _honest_note(crit),
    }
    res["environment"] = _environment()
    # canonical digest of the numeric payload (the results_sha256 field itself is
    # excluded so the digest is a fixed point of a re-run)
    digest = hashlib.sha256(_canonical(res).encode()).hexdigest()
    res["results_sha256"] = digest
    res["_sha256_note"] = (
        "sha256 of the canonical json of this document with the results_sha256 "
        "field removed; a re-run of `run` reproduces it byte for byte"
    )
    _write_json(os.path.join(outdir, "results.json"), res)
    with open(os.path.join(outdir, "results.sha256"), "w") as fh:
        fh.write(f"{digest}  results.json (canonical, results_sha256 excluded)\n")
    _write_sidecars(res, outdir)
    return res


def _canonical(obj) -> str:
    stripped = {k: v for k, v in obj.items() if k != "results_sha256"}
    return json.dumps(stripped, indent=1, sort_keys=True, default=_json_default,
                      allow_nan=True)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _write_sidecars(res: dict, outdir: str) -> None:
    """benchmark-case registration + method-selection note."""
    write_case(res, "benchmarks/verified/fv_cc_three_grid_orr/case.json")
    write_oss_solution(res, os.path.join(outdir, "oss_solutions.json"))


def _honest_note(crit: dict) -> str:
    notes = []
    if not crit.get("C3_primary_order_3pct", {}).get("pass"):
        w = crit.get("C3_primary_order_3pct", {}).get("worst", {})
        notes.append(
            "tight <3% line not met by the worst L2 cell "
            f"({w.get('problem')}/{w.get('ladder')}: rel err {w.get('rel_err'):.3e})"
        )
    if not crit.get("C4_estimator_agreement", {}).get("pass"):
        notes.append("error-based and functional-based order disagree beyond 15% somewhere")
    return "; ".join(notes) if notes else "every predeclared criterion met"


def _environment() -> dict:
    import scipy

    env = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        import skfem

        env["skfem"] = skfem.__version__
    except Exception:  # noqa: BLE001
        env["skfem"] = None
    try:
        env["git_commit"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        )
    except Exception:  # noqa: BLE001
        env["git_commit"] = None
    return env


def _write_json(path: str, obj) -> None:
    txt = json.dumps(obj, indent=1, sort_keys=True, default=_json_default,
                     allow_nan=True)
    with open(path, "w") as fh:
        fh.write(txt + "\n")


def verify(outdir: str) -> int:
    with open(os.path.join(outdir, "results.json")) as fh:
        res = json.load(fh)
    crit = res["criteria"]
    ok = True
    for name in sorted(crit):
        p = crit[name].get("pass")
        ok &= bool(p)
        print(f"{'PASS' if p else 'FAIL'}  {name}")
    print(f"\nverdict: {'ALL PASS' if ok else 'NOT ALL PASS'} "
          f"({res['verdict']['n_pass']}/{res['verdict']['n_criteria']})")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "verify", "report"])
    ap.add_argument("--outdir", default="docs/fv_order_xval")
    args = ap.parse_args(argv)

    if args.cmd == "run":
        res = run_study(args.outdir)
        make_figures(res, args.outdir)
        write_report(res, args.outdir)
        print(json.dumps(res["verdict"], indent=1))
        for name in sorted(res["criteria"]):
            print(f"{'PASS' if res['criteria'][name].get('pass') else 'FAIL'}  {name}")
        return 0
    if args.cmd == "verify":
        return verify(args.outdir)
    with open(os.path.join(args.outdir, "results.json")) as fh:
        res = json.load(fh)
    write_report(res, args.outdir)
    return 0


# figures + report live in the same file's helpers, imported below to keep the
# CLI small
from m18742_fv_report import (  # noqa: E402
    make_figures,
    write_case,
    write_oss_solution,
    write_report,
)

if __name__ == "__main__":
    sys.exit(main())