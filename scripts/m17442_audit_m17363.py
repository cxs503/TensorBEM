#!/usr/bin/env python3
"""m17442 — board audit of m17363 (XVAL-BEM-ROUND2-METHOD-SELECTION).

Read-only evidence collector.  It does **not** re-deliver m17363's work; it
re-derives every number the audit relies on, so nothing in the audit report is
hand-typed.

Inputs
------
MISSION_TREE  worktree/commit snapshot of m17363 (default
              /root/tp-agent/work/m17363, commit 8bf9345)
REPRO_TREE    independent re-run of the mission driver on a clean
              ``git archive`` snapshot of the same commit
              (default /tmp/xv)

Outputs (all under artifacts/audit_m17363_xval_round2/)
------------------------------------------------------
results_mission.json   byte-copy of the mission artefact (+ sha256)
results_repro.json     byte-copy of the independent re-run (+ sha256)
audit_evidence.json    machine-readable findings used by the audit report
fig_audit_m17363.png   convergence orders + independent sign check

Usage
-----
    PYTHONPATH=/root/tp-agent/work/m17363/src \
        python3 scripts/m17442_audit_m17363.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np

MISSION = Path(os.environ.get("MISSION_TREE", "/root/tp-agent/work/m17363"))
REPRO = Path(os.environ.get("REPRO_TREE", "/tmp/xv"))
OUT = Path("artifacts/audit_m17363_xval_round2")
OUT.mkdir(parents=True, exist_ok=True)

MISSION_RES = MISSION / "artifacts/xval_laplace_sphere/results.json"
REPRO_RES = REPRO / "artifacts/xval_laplace_sphere/results.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flatten(v, f"{prefix}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def numeric_diff(a: dict, b: dict) -> dict:
    fa, fb = dict(flatten(a)), dict(flatten(b))
    missing_a = sorted(set(fa) - set(fb))
    missing_b = sorted(set(fb) - set(fa))
    worst = 0.0
    worst_key = ""
    n_bad = 0
    for key in set(fa) & set(fb):
        x, y = fa[key], fb[key]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            scale = max(abs(x), abs(y), 1e-300)
            d = abs(x - y) / scale
            if d > worst:
                worst, worst_key = d, key
            if d > 1e-9:
                n_bad += 1
        elif x != y:
            n_bad += 1
    return {"max_rel_diff": worst, "worst_key": worst_key,
            "n_fields_differing": n_bad,
            "only_in_mission": missing_a, "only_in_repro": missing_b,
            "identical": n_bad == 0 and not missing_a and not missing_b}


def richardson_tautology(res: dict) -> dict:
    """Show that the reported Richardson error 0.000% is an algebraic identity.

    ``route_skfem`` estimates the order ``p`` from the two errors of the *same*
    pair and then extrapolates with it:
        extrapolated = I_fine + (I_fine - I_coarse) / (2**p - 1),
        p = log2(e_coarse / e_fine).
    With e = |I - I_exact| and a single leading error term this returns
    ``I_exact`` exactly, for *any* observed order -- so ``richardson_rel_error``
    carries no information about the reference quality.
    """
    out = []
    for order, route in res["skfem"].items():
        for rec in route.get("richardson_current", []):
            p = rec["observed_order_current"]
            out.append({"fem_order": order, "outer_ratio": rec["outer_ratio"],
                        "observed_order": p,
                        "reported_richardson_rel_error":
                            rec["richardson_rel_error"],
                        "reported_richardson_current_A":
                            rec["richardson_current_A"],
                        "note": "p is derived from the same two errors that are "
                                "extrapolated -> error 0.0 is an identity, not "
                                "independent evidence"})
    return {"records": out,
            "conclusion": "richardson_rel_error == 0.0 for every pair "
                          "(observed orders %.2f and %.2f, i.e. not even 2.0)"
                          % tuple(r["observed_order"] for r in out)}


def separate_test_bugs(res: dict) -> dict:
    """Re-derive the two analytic-reference assertions used by the red tests.

    The committed tests compare the analytic series evaluated at *panel
    centroids* (r < a, since a triangle centroid of an inscribed polyhedron lies
    inside the sphere) with the boundary data ``-(1+mu)``, and subtract ``1``
    instead of the monopole ``a/r`` from the probe reference.  Both are test-side
    convention errors; the kernel is unaffected.  Reproduced here so the audit
    claim is evidence, not opinion.
    """
    from tensorbem.benchmarks import laplace_sphere_xval as xval
    from tensorbem.geometry3d import icosphere

    mesh = icosphere(2, 1.0)
    mu = xval.panel_mu(mesh)
    phi_centroid = xval.exact_potential(mesh.centroids)
    probes = xval.probe_points()
    ref_wrong = xval.exact_potential(probes, phi_a=1.0) - 1.0
    r = np.linalg.norm(probes, axis=1)
    ref_right = xval.exact_potential(probes, phi_a=1.0) - 1.0 / r  # strip monopole a/r
    phi_repr = xval.representation_potential(mesh, 2.0 * mu, mu, probes)
    return {
        "test_exact_series": {
            "max_abs_diff_vs_boundary_data":
                float(np.max(np.abs(phi_centroid + (1.0 + mu)))),
            "centroid_r_min": float(np.min(np.linalg.norm(mesh.centroids, axis=1))),
            "explanation": "centroids sit at r/a<1, exact_potential is evaluated "
                           "there, so it cannot equal the r=a boundary data",
        },
        "test_dipole_sign_reference": {
            "max_rel_err_wrong_reference":
                float(np.max(np.abs(phi_repr - ref_wrong)) / np.max(np.abs(ref_wrong))),
            "max_rel_err_correct_reference":
                float(np.max(np.abs(phi_repr - ref_right)) / np.max(np.abs(ref_right))),
            "explanation": "test strips the monopole with 1.0 instead of a/r; the "
                           "fixed '+' representation is accurate to ~1%",
        },
    }


def sign_defect_check(res: dict) -> dict:
    """Independent check of the double-layer sign fix (not using the repo test).

    Exact Dirichlet data + exact density are injected into the representation
    formula with both signs; only the '+' form reproduces the analytic exterior
    potential.
    """
    from tensorbem.benchmarks import laplace_sphere_xval as xval
    from tensorbem.geometry3d import icosphere
    from tensorbem.kernels3d import duffy_quadrature_batch

    mesh = icosphere(3, 1.0)
    mu = xval.panel_mu(mesh)
    density = xval.exact_density(mu)
    centre = mesh.centroids / np.linalg.norm(mesh.centroids, axis=1)[:, None]
    phi_b = xval.exact_potential(centre)          # boundary data, r = a
    probes = xval.probe_points()
    ref = xval.exact_potential(probes)

    qp, qw = duffy_quadrature_batch(mesh.v0, mesh.v1, mesh.v2, order=10)
    plus, minus = [], []
    for x in probes:
        delta = x[None, None, :] - qp
        dist = np.linalg.norm(delta, axis=2)
        green = np.sum(qw / (4.0 * math.pi * dist), axis=1)
        dgreen = np.sum(qw * np.einsum("fqi,fi->fq", delta, mesh.normals)
                        / (4.0 * math.pi * dist**3), axis=1)
        sl = float(np.dot(green, density))
        dl = float(np.dot(dgreen, phi_b))
        plus.append(sl + dl)
        minus.append(sl - dl)
    rel = lambda v: float(np.max(np.abs(np.asarray(v) - ref)) / np.max(np.abs(ref)))
    return {"panels": int(mesh.n_panels) if hasattr(mesh, "n_panels") else len(mesh.areas),
            "rel_err_fixed_plus": rel(plus),
            "rel_err_prefix_minus": rel(minus),
            "conclusion": "the '+' (fixed) sign reproduces the analytic exterior "
                          "potential; the pre-fix '-' sign is off by >20%",
            "bem_finest_current_rel_error": res["tensorbem"]["levels"][-1]["current_rel_error"]}


def figure(res: dict, sign: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    lv = res["tensorbem"]["levels"]
    h = np.array([l["h_over_a"] for l in lv])
    for key, label in (("current_rel_error", "total current"),
                       ("density_linf_rel_error", "density $L_\\infty$"),
                       ("probe_potential_l2_rel_error", "exterior $\\phi$ $L_2$")):
        ax.loglog(h, [l[key] for l in lv], "o-", label=label)
    for p, style in ((1, "--"), (2, ":")):
        ax.loglog(h, 0.35 * (h / h[0]) ** p, style, color="grey",
                  label=f"slope $h^{p}$")
    ax.invert_xaxis()
    ax.set_xlabel("$h/a$ (panels $N$ = 80, 320, 1280)")
    ax.set_ylabel("relative error")
    oc = res["tensorbem_observed_order"]["current"]
    ax.set_title("TensorBEM ladder: observed order of $I$ = "
                 + " / ".join(f"{v:.2f}" for v in oc) + " (not 2.0)")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    names = ["representation\n$-$(pre-fix)", "representation\n$+$(fixed)",
             "BEM current\n$N$=1280"]
    vals = [100.0 * sign["rel_err_prefix_minus"], 100.0 * sign["rel_err_fixed_plus"],
            100.0 * res["tensorbem"]["levels"][-1]["current_rel_error"]]
    bars = ax.bar(names, vals, color=["#c0392b", "#27ae60", "#2980b9"])
    ax.axhline(3.0, color="k", ls="--", lw=1, label="3% acceptance")
    ax.axhline(15.0, color="k", ls=":", lw=1, label="15% reject")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.2f}%",
                ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("relative error [%]")
    ax.set_title("Independent sign check (exact Dirichlet data injected)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print("wrote", path)


def main() -> None:
    res = json.loads(MISSION_RES.read_text())
    repro = json.loads(REPRO_RES.read_text())
    shutil.copyfile(MISSION_RES, OUT / "results_mission.json")
    shutil.copyfile(REPRO_RES, OUT / "results_repro.json")

    diff = numeric_diff(res, repro)
    sign = sign_defect_check(res)
    bugs = separate_test_bugs(res)
    rich = richardson_tautology(res)

    ev = {
        "mission": 17363,
        "audit": 17442,
        "mission_commit": "8bf9345",
        "repo": "TensorBEM",
        "artifacts": {
            "mission_results_sha256": sha256(MISSION_RES),
            "repro_results_sha256": sha256(REPRO_RES),
        },
        "reproduction": {
            **diff,
            "cmd": "XVAL_MIN=1 PYTHONPATH=src python3 scripts/m17363_xval_run.py run",
            "note": "driver default (no XVAL_MIN) uses subdivisions 1..4, Duffy "
                    "4/6/8/12, FEM R/a 2/4/8 and 96x64 -- a different "
                    "configuration from the published artefact",
        },
        "headline_numbers": {
            "analytic_current_A": res["exact_current_A"],
            "bem_finest_current_A": res["tensorbem"]["levels"][-1]["current_A"],
            "bem_finest_current_rel_error":
                res["tensorbem"]["levels"][-1]["current_rel_error"],
            "capytaine_finest_current_rel_error":
                res["capytaine"]["levels"][-1]["current_rel_error"],
            "skfem_best_current_rel_error": min(
                l["current_rel_error"] for v in res["skfem"].values()
                for l in v["levels"]),
            "capytaine_single_fro_at_1280":
                res["capytaine"]["matrix_comparison"][-1]["single_fro_rel"],
            "capytaine_double_fro_at_1280":
                res["capytaine"]["matrix_comparison"][-1]["double_fro_rel"],
            "capytaine_double_fro_at_320":
                res["capytaine"]["matrix_comparison"][1]["double_fro_rel"],
            "quadrature_orders_published": [
                q["quadrature_order"] for q in res["quadrature_sensitivity"]],
            "observed_orders": res["tensorbem_observed_order"],
        },
        "independent_sign_check": sign,
        "separate_test_bugs": bugs,
        "richardson": rich,
        "benchmark_registry_rows": 0,
    }
    (OUT / "audit_evidence.json").write_text(json.dumps(ev, indent=2))
    figure(res, sign, OUT / "fig_audit_m17363.png")
    print(json.dumps({k: ev[k] for k in ("artifacts", "reproduction",
                                         "headline_numbers",
                                         "independent_sign_check",
                                         "richardson")}, indent=2))


if __name__ == "__main__":
    main()