"""Analytic verification of the 6-node quadratic panel operator (mission m7526).

Why this driver exists
----------------------
:mod:`tensorbem.panel_quadratic` shipped with an isoparametric 6-node element,
a Duffy self-integral and a Dirichlet solver, and its own test file -- but no
test compared the assembled *operator* with anything analytic.  The suite only
checked self-consistency (partition of unity, positive diagonals, linear-solve
residual ``< 1e-10``), all of which pass for a wrong operator.  Re-derived here
against a closed-form reference, three independent defects were found (see
``docs/quadratic_panel_verification.md``): a wrong shape-gradient table, a Duffy
apex permutation pointing at the wrong vertex, and an assembly that summed only
element-local rows, so the matrix was a patch operator instead of the boundary
operator.

The anchors
-----------
All three are exact and need no BEM code:

1. **Surface area.**  The isoparametric elements must tile the unit sphere:
   ``sum_e int_{T_e} dS = 4 pi``.
2. **Spectral image of a harmonic.**  For a spherical shell of radius ``R`` the
   single-layer operator with kernel ``G = 1/(4 pi |x - y|)`` acts on the trace
   of a *homogeneous harmonic polynomial of degree n* as multiplication by
   ``R / (2 n + 1)`` (addition theorem).  So ``S q = R q / (2n+1)`` node by node,
   with ``q`` sampled from ``1, z, x y, x y z`` (n = 0..3).  This pins the whole
   matrix, including the singular self terms and the near-singular regular ones.
3. **Dirichlet solve.**  Prescribing ``g = z`` on the unit sphere has the exact
   density ``q = 3 z`` (n = 1), and ``g = x y`` gives ``q = 5 x y`` (n = 2), so
   the solved nodal density can be compared with a closed form.

Every number in the report is written to ``docs/quadratic_panel_verification/
study.json`` by this driver; nothing is transcribed by hand.

Usage::

    python -m tensorbem.benchmarks.quadratic_panel_verification                # table
    python -m tensorbem.benchmarks.quadratic_panel_verification --max-sub 3
    python -m tensorbem.benchmarks.quadratic_panel_verification --write        # + JSON/PNG
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import numpy as np

from ..panel_quadratic import (
    assemble_quadratic_matrices,
    gauss_triangle_rule,
    icosphere_quadratic,
    solve_laplace_dirichlet_quadratic,
)

FOUR_PI = 4.0 * math.pi
RADIUS = 1.0

#: Homogeneous harmonic polynomials of degree n, evaluated on the unit sphere.
HARMONICS = {
    0: ("1", lambda v: np.ones(v.shape[0])),
    1: ("z", lambda v: v[:, 2]),
    2: ("x*y", lambda v: v[:, 0] * v[:, 1]),
    3: ("x*y*z", lambda v: v[:, 0] * v[:, 1] * v[:, 2]),
}

#: Dirichlet data g with a closed-form single-layer density q = (2n+1) g / R.
SOLVE_CASES = {
    "n=1 (g=z)": ("z", lambda v: v[:, 2]),
    "n=2 (g=x*y)": ("x*y", lambda v: v[:, 0] * v[:, 1]),
}

DOC_DIR = pathlib.Path("docs/quadratic_panel_verification")


def loglog_slope(h: list[float], err: list[float]) -> float:
    """Least-squares slope of log(err) against log(h)."""
    x = np.log(np.asarray(h, dtype=float))
    y = np.log(np.asarray(err, dtype=float))
    if x.size < 2:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def mesh_area(mesh) -> float:
    """Area of the quadratic (curved) mesh by Gauss quadrature on each element."""
    L, w = gauss_triangle_rule(8)
    total = 0.0
    for e in range(mesh.element_count):
        elem = mesh.element(e)
        for q in range(len(w)):
            _, det = elem.jacobian(L[q, 0], L[q, 1])
            total += w[q] * det
    return total


def spectral_image(mesh, S) -> dict:
    """Eigenvalue of the single-layer operator on each harmonic trace."""
    rows = {}
    for n, (name, fn) in HARMONICS.items():
        q = fn(mesh.vertices)
        scale = math.sqrt(float(q @ q))
        image = S @ q
        lam = float(q @ image) / float(q @ q)
        resid = float(np.linalg.norm(image - (RADIUS / (2 * n + 1)) * q) / scale)
        ref = RADIUS / (2 * n + 1)
        rows[f"n={n} ({name})"] = {
            "analytic_lambda": ref,
            "measured_lambda": lam,
            "lambda_rel_err": abs(lam - ref) / ref,
            "image_residual": resid,
        }
    return rows


def run_study(max_sub: int = 3, quadrature_order: int = 8, self_order: int = 16) -> dict:
    levels = []
    for sub in range(1, max_sub + 1):
        mesh = icosphere_quadratic(sub, radius=RADIUS)
        t0 = time.time()
        S, D = assemble_quadratic_matrices(
            mesh, quadrature_order=quadrature_order, self_order=self_order
        )
        assemble_s = time.time() - t0
        h = math.sqrt(FOUR_PI / mesh.element_count)
        area = mesh_area(mesh)
        spectral = spectral_image(mesh, S)

        solves = {}
        for label, (_, fn) in SOLVE_CASES.items():
            n = int(label.split("=")[1][0])
            g = fn(mesh.vertices)
            q_exact = (2 * n + 1) * g / RADIUS
            q_h = np.linalg.solve(S, g)
            solves[label] = {
                "density_rel_err": float(
                    np.linalg.norm(q_h - q_exact) / np.linalg.norm(q_exact)
                ),
                "volume_ratio": float(
                    q_h @ mesh.vertices[:, 2] / (q_exact @ mesh.vertices[:, 2])
                ),
            }

        res = solve_laplace_dirichlet_quadratic(
            mesh, SOLVE_CASES["n=1 (g=z)"][1](mesh.vertices),
            quadrature_order=quadrature_order, self_order=self_order,
        )
        levels.append({
            "subdivisions": sub,
            "elements": mesh.element_count,
            "nodes": mesh.node_count,
            "h_mean": h,
            "assemble_seconds": assemble_s,
            "area": area,
            "area_rel_err": abs(area - FOUR_PI) / FOUR_PI,
            "double_layer_all_zero": bool(not D.any()),
            "spectral": spectral,
            "solves": solves,
            "condition_number": res.condition_number,
            "relative_residual_l2": res.relative_residual_l2,
        })

    h = [lv["h_mean"] for lv in levels]
    orders = {
        "area": loglog_slope(h, [lv["area_rel_err"] for lv in levels]),
        "density_n1": loglog_slope(
            h, [lv["solves"]["n=1 (g=z)"]["density_rel_err"] for lv in levels]
        ),
        "spectral_n3": loglog_slope(
            h, [lv["spectral"]["n=3 (x*y*z)"]["image_residual"] for lv in levels]
        ),
    }
    return {
        "study": "quadratic_panel_verification",
        "generated_by": "python -m tensorbem.benchmarks.quadratic_panel_verification --write",
        "radius": RADIUS,
        "quadrature_order": quadrature_order,
        "self_order": self_order,
        "levels": levels,
        "observed_orders": orders,
        "acceptance": {
            "density_rel_err_lt_1pct_at_finest": bool(
                levels[-1]["solves"]["n=1 (g=z)"]["density_rel_err"] < 0.01
            ),
            "area_rel_err_lt_0.1pct_at_finest": bool(levels[-1]["area_rel_err"] < 1e-3),
            "spectral_residual_decreasing": bool(
                all(
                    levels[i + 1]["spectral"]["n=1 (z)"]["image_residual"]
                    < levels[i]["spectral"]["n=1 (z)"]["image_residual"]
                    for i in range(len(levels) - 1)
                )
            ),
        },
        "not_run": [
            "double-layer assembly (D is returned as zeros; no free term, no solid-angle self term)",
            "normal-derivative / pressure / added-mass output from the quadratic path",
            "near-singular (target within ~h/2 of a non-incident element) adaptive quadrature",
            "convergence on non-spherical or non-smooth geometry",
        ],
    }


def render_table(study: dict) -> str:
    lines = [
        "| subdiv | elements | h | area err | lambda n=0 err | lambda n=1 err |"
        " lambda n=2 err | lambda n=3 err | q(n=1) err | q(n=2) err | kappa |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for lv in study["levels"]:
        sp = lv["spectral"]
        lines.append(
            "| {sub} | {el} | {h:.3f} | {area:.2e} | {a} | {b} | {c} | {d} | {q1:.2e} |"
            " {q2:.2e} | {k:.1f} |".format(
                sub=lv["subdivisions"], el=lv["elements"], h=lv["h_mean"],
                area=lv["area_rel_err"],
                a=f"{sp['n=0 (1)']['lambda_rel_err']:.2e}",
                b=f"{sp['n=1 (z)']['lambda_rel_err']:.2e}",
                c=f"{sp['n=2 (x*y)']['lambda_rel_err']:.2e}",
                d=f"{sp['n=3 (x*y*z)']['lambda_rel_err']:.2e}",
                q1=lv["solves"]["n=1 (g=z)"]["density_rel_err"],
                q2=lv["solves"]["n=2 (g=x*y)"]["density_rel_err"],
                k=lv["condition_number"],
            )
        )
    orders = study["observed_orders"]
    lines.append("")
    lines.append(
        "observed order: area {area:.2f}, spectral n=3 {spec:.2f}, density n=1 {dens:.2f}".format(
            area=orders["area"], spec=orders["spectral_n3"], dens=orders["density_n1"]
        )
    )
    return "\n".join(lines)


def write_figure(study: dict, path: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = [lv["h_mean"] for lv in study["levels"]]
    dens = [lv["solves"]["n=1 (g=z)"]["density_rel_err"] for lv in study["levels"]]
    spec = [lv["spectral"]["n=3 (x*y*z)"]["image_residual"] for lv in study["levels"]]
    area = [lv["area_rel_err"] for lv in study["levels"]]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax.loglog(h, dens, "o-", label="solved density, n=1 (g=z)")
    ax.loglog(h, spec, "s-", label="operator image, n=3 (xyz)")
    ax.loglog(h, area, "^-", label="mesh area vs 4 pi")
    ref = np.array(h) ** 4
    ax.loglog(h, ref * dens[0] / ref[0], "k--", lw=0.8, label="O(h^4) guide")
    ax.axhline(3e-2, color="r", ls=":", label="3 % acceptance line")
    ax.set_xlabel("h (mean element size)")
    ax.set_ylabel("relative error")
    ax.set_title("Quadratic panel operator vs analytic sphere anchors")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    labels = [f"n={n}" for n in range(4)]
    width = 0.2
    for i, lv in enumerate(study["levels"]):
        errs = [lv["spectral"][k]["lambda_rel_err"] for k in lv["spectral"]]
        ax2.bar(np.arange(4) + i * width, errs, width, label=f"sub={lv['subdivisions']}")
    ax2.set_yscale("log")
    ax2.set_xticks(np.arange(4) + width * (len(study["levels"]) - 1) / 2)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("|lambda_h - 1/(2n+1)| / (1/(2n+1))")
    ax2.set_title("Single-layer eigenvalue error by harmonic degree")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-sub", type=int, default=3)
    ap.add_argument("--quadrature-order", type=int, default=8)
    ap.add_argument("--self-order", type=int, default=16)
    ap.add_argument("--write", action="store_true", help="write JSON + figure into docs/")
    args = ap.parse_args(argv)

    study = run_study(args.max_sub, args.quadrature_order, args.self_order)
    print(render_table(study))
    print(json.dumps(study["acceptance"], indent=2))
    if args.write:
        DOC_DIR.mkdir(parents=True, exist_ok=True)
        (DOC_DIR / "study.json").write_text(json.dumps(study, indent=2) + "\n")
        write_figure(study, DOC_DIR / "fig_quadratic_operator.png")
        print(f"wrote {DOC_DIR/'study.json'} and {DOC_DIR/'fig_quadratic_operator.png'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())