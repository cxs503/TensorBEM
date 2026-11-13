#!/usr/bin/env python3
"""Anchor driver: CASE-FEM-HERTZ-CONTACT by the boundary-integral (Boussinesq) route.

Prescribed load (10 kN), sphere radius and moduli of the two-50 mm-steel-sphere
equivalent problem.  Outputs (artifacts/elastic_contact_hertz/):
``results.json``, ``pressure_profile.png``, ``convergence.png``.

No number in the report is typed by hand: every quantity below comes out of
``tensorbem.elastic_contact`` and is written to ``results.json`` first.
"""
from __future__ import annotations

import json
import math
import pathlib
import platform
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "artifacts" / "elastic_contact_hertz"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorbem import elastic_contact as ec  # noqa: E402

LOAD = 10.0e3
YOUNG = 210.0e9
POISSON = 0.3
RADIUS = 0.025          # effective radius: two 50 mm spheres
E_STAR = ec.effective_modulus(YOUNG, POISSON, YOUNG, POISSON)
ANCHOR = ec.hertz_sphere_halfspace(LOAD, RADIUS, E_STAR)
LADDER = (100, 200, 400)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    solutions = {}
    for n in LADDER:
        t0 = time.time()
        sol = ec.solve_normal_contact(lambda r: ec.parabolic_gap(r, RADIUS), E_STAR,
                                      load=LOAD, n_elements=n, grading=2.0)
        cmp = ec.hertz_comparison(sol, ANCHOR)
        cmp["n_elements"] = n
        cmp["wall_seconds"] = round(time.time() - t0, 2)
        cmp["domain_radius"] = sol.domain_radius
        records.append(cmp)
        solutions[n] = sol
        print(f"n={n:4d} a={sol.contact_radius*1e3:.5f} mm delta={sol.penetration*1e6:.4f} um "
              f"p0={sol.peak_pressure/1e9:.4f} GPa ({cmp['wall_seconds']} s)", flush=True)

    # independent checks: flat punch (different closed form) and load scaling
    punch_r, punch_d = 1.0e-3, 2.0e-6
    punch = ec.solve_normal_contact(ec.flat_gap, E_STAR, penetration=punch_d,
                                    domain_radius=punch_r, n_elements=200, grading=2.0)
    punch_ref = ec.flat_punch_load(punch_r, punch_d, E_STAR)
    heavy = ec.solve_normal_contact(lambda r: ec.parabolic_gap(r, RADIUS), E_STAR,
                                    load=8.0 * LOAD, n_elements=160, grading=2.0)
    heavy_ref = ec.hertz_sphere_halfspace(8.0 * LOAD, RADIUS, E_STAR)

    # kernel / influence-matrix sanity, quoted in the report as its own evidence
    a_disk, mod, p0 = 1.0e-3, 1.0e11, 2.0e6
    edges = np.linspace(0.0, a_disk, 201)
    centers = 0.5 * (edges[1:] + edges[:-1])
    C = ec.displacement_influence(centers, edges, mod)
    w_uniform = float((C @ np.full(centers.size, p0))[0])

    reference = {
        "load_N": LOAD, "radius_m": RADIUS, "modulus_Pa": E_STAR,
        "contact_radius_m": ANCHOR.contact_radius,
        "penetration_m": ANCHOR.penetration,
        "peak_pressure_Pa": ANCHOR.peak_pressure,
        "source": "Hertz 1882 J. reine angew. Math. 92:156; Johnson 1985 Contact "
                  "Mechanics Sec. 4.2 (DOI 10.1017/cbo9781139171731)",
    }
    checks = {
        "flat_punch_load_N": punch.load,
        "flat_punch_reference_N": punch_ref,
        "flat_punch_rel_error": punch.load / punch_ref - 1.0,
        "load_scaling_contact_radius_rel_error": heavy.contact_radius / heavy_ref.contact_radius - 1.0,
        "load_scaling_penetration_rel_error": heavy.penetration / heavy_ref.penetration - 1.0,
        "uniform_disk_w_center_m": w_uniform,
        "uniform_disk_w_center_reference_m": 2.0 * p0 * a_disk / mod,
        "uniform_disk_w_center_rel_error": w_uniform / (2.0 * p0 * a_disk / mod) - 1.0,
    }
    finest = records[-1]
    payload = {
        "anchor": "CASE-FEM-HERTZ-CONTACT",
        "route": "boundary integral (Boussinesq half-space kernel, axisymmetric LCP)",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "reference": reference,
        "ladder": records,
        "independent_checks": checks,
        "verdict": {
            "contact_radius_rel_error": finest["contact_radius_rel_error"],
            "penetration_rel_error": finest["penetration_rel_error"],
            "within_3_percent": bool(abs(finest["contact_radius_rel_error"]) < 0.03
                                     and abs(finest["penetration_rel_error"]) < 0.03),
        },
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")

    # --- figure 1: pressure profile against Hertz -------------------------
    sol = solutions[LADDER[-1]]
    r, p = sol.profile()
    rr = np.linspace(0.0, 1.3 * ANCHOR.contact_radius, 800)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(rr * 1e3, ANCHOR.pressure(rr) / 1e9, "-", lw=2, label="Hertz closed form")
    ax.plot(r * 1e3, p / 1e9, "o", ms=3.0, mfc="none",
            label=f"BEM (n={sol.n_elements}, rim fit a={sol.contact_radius*1e3:.4f} mm)")
    ax.axvline(ANCHOR.contact_radius * 1e3, ls=":", c="k", lw=1)
    ax.set_xlabel("r [mm]")
    ax.set_ylabel("contact pressure [GPa]")
    ax.set_title("Hertz sphere-on-half-space: BEM pressure field vs closed form")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "pressure_profile.png", dpi=150)
    plt.close(fig)

    # --- figure 2: convergence of the two anchor metrics -------------------
    ns = [rec["n_elements"] for rec in records]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogx(ns, [abs(rec["contact_radius_rel_error"]) * 100 for rec in records],
                "o-", label="contact radius a")
    ax.semilogx(ns, [abs(rec["contact_radius_rim_rel_error"]) * 100 for rec in records],
                "s--", mfc="none", label="contact radius, shape-free rim edge")
    ax.semilogx(ns, [abs(rec["penetration_rel_error"]) * 100 for rec in records],
                "^-", label="penetration delta")
    ax.semilogx(ns, [abs(rec["peak_pressure_rel_error"]) * 100 for rec in records],
                "d-", mfc="none", label="peak pressure p0")
    ax.axhline(3.0, c="r", ls=":", label="3 % acceptance")
    ax.set_xlabel("radial elements")
    ax.set_ylabel("|relative error| vs Hertz [%]")
    ax.set_title("Mesh convergence of the BEM contact metrics")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "convergence.png", dpi=150)
    plt.close(fig)

    print(json.dumps(payload["verdict"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
