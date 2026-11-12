#!/usr/bin/env python3
"""Verification benchmark for :mod:`tensorbem.sk_fit` (SK iteration vs fixed poles).

Two cases, both with an independent route to the answer:

* ``exact_model_class`` -- a synthetic radiation-damping curve that *is* in the
  fitted model class with known ``alpha_k``/``c_k``/``d``.  The Cummins memory
  kernel of a relaxation spectrum has a closed form, ``K(t) = sum_k c_k
  exp(-alpha_k t)`` (``t > 0``), so the fit can be checked against an analytic
  kernel rather than against another fit.
* ``hulme_hemisphere`` -- the tabulated heave damping of the Hulme hemisphere
  (``tests/data/time_domain_hemisphere_sweep.json``), i.e. a real band-limited
  curve whose pole locations are only weakly observable.  Here the check is the
  *fit-induced kernel error*: closed-form kernel of the fitted model against the
  exact piecewise-linear quadrature kernel of the table
  (:func:`tensorbem.time_domain.radiation_memory_kernel`).

Everything written by this module (JSON, Markdown, figure) is rendered from the
same in-memory numbers -- no transcribed digits.

Usage
-----
    python -m tensorbem.benchmarks.sk_fit_verification [--out docs/sk_verification]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..sk_fit import fixed_pole_fit_damping, sk_fit_damping
from ..time_domain import radiation_memory_kernel


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def relaxation_model(omega, poles, residues, constant):
    """``B = d + sum_k c_k alpha_k/(alpha_k^2 + omega^2)``."""
    omega = np.asarray(omega, dtype=float)
    a = np.asarray(poles, dtype=float)[None, :]
    return float(constant) + (np.asarray(residues, float)[None, :] * a / (a * a + omega[:, None] ** 2)).sum(axis=1)


def closed_form_kernel(tau, poles, residues):
    """``K(t) = sum_k c_k exp(-alpha_k t)`` for ``t > 0`` (zero at ``t <= 0``)."""
    tau = np.asarray(tau, dtype=float)
    a = np.asarray(poles, dtype=float)[None, :]
    values = (np.asarray(residues, float)[None, :] * np.exp(-a * np.maximum(tau, 0.0)[:, None])).sum(axis=1)
    return np.where(tau > 0.0, values, 0.0)


def fit_metrics(fit, damping):
    damping = np.asarray(damping, dtype=float)
    scale = float(np.max(np.abs(damping)))
    err = np.abs(fit.B_reconstructed - damping)
    return {
        "max_rel_error": float(np.max(err) / scale),
        "rms_rel_error": float(np.sqrt(np.mean(err**2)) / scale),
        "n_iter": int(fit.n_iter),
        "converged": bool(fit.converged),
        "poles": [float(x) for x in fit.poles],
        "residues": [float(x) for x in fit.residues],
        "constant": float(fit.constant),
    }


def kernel_error(tau, k_ref, k_test):
    # tau = 0 is excluded: the constant term d of the model carries an
    # impulsive 2*d*delta(t) there, which the closed form (t > 0) omits.
    tau, k_ref, k_test = np.asarray(tau)[1:], np.asarray(k_ref)[1:], np.asarray(k_test)[1:]
    scale = float(np.max(np.abs(k_ref)))
    err = np.abs(k_test - k_ref)
    return {
        "tau_max": float(tau[-1]),
        "n_tau": int(tau.size),
        "max_abs_error": float(np.max(err)),
        "max_rel_error": float(np.max(err) / scale),
        "rms_rel_error": float(np.sqrt(np.mean(err**2)) / scale),
    }


def tau_grid(omega_max, dt_factor=0.4, span=None):
    dt = dt_factor * np.pi / float(omega_max)
    if span is None:
        span = 40.0 * dt
    n = int(np.ceil(span / dt)) + 1
    return np.linspace(0.0, dt * (n - 1), n)


# ---------------------------------------------------------------------------
# case A: synthetic curve inside the model class
# ---------------------------------------------------------------------------
def case_exact_model_class(n_poles=3):
    alpha = np.array([0.4, 1.1, 3.0])
    residues = np.array([2.0, 1.2, 0.6])
    constant = 0.05
    # log-spaced grid: a piecewise-linear table of a peaked damping curve over
    # four decades would otherwise dominate the quadrature-identity check.
    omega = np.geomspace(0.02, 60.0, 600)
    damping = relaxation_model(omega, alpha, residues, constant)

    tau = tau_grid(omega[-1], span=8.0)
    k_analytic = closed_form_kernel(tau, alpha, residues)
    k_quad = radiation_memory_kernel(
        omega, damping, tau, low_frequency_model="zero"
    ).values

    fixed = fixed_pole_fit_damping(omega, damping, n_poles=n_poles)
    sk = sk_fit_damping(omega, damping, n_poles=n_poles)

    out = {
        "omega_range": [float(omega[0]), float(omega[-1])],
        "n_points": int(omega.size),
        "n_poles": int(n_poles),
        "truth": {
            "poles": alpha.tolist(),
            "residues": residues.tolist(),
            "constant": float(constant),
        },
        "fixed_pole": fit_metrics(fixed, damping),
        "sk": fit_metrics(sk, damping),
        # analytic kernel identity: closed form vs exact quadrature of the table
        "kernel_analytic_vs_quadrature": kernel_error(tau, k_analytic, k_quad),
        # what each fit costs in the time-domain kernel
        "kernel_error_fixed_pole": kernel_error(
            tau, k_analytic, closed_form_kernel(tau, fixed.poles, fixed.residues)
        ),
        "kernel_error_sk": kernel_error(
            tau, k_analytic, closed_form_kernel(tau, sk.poles, sk.residues)
        ),
        "_curves": {
            "omega": omega.tolist(),
            "damping": damping.tolist(),
            "fixed": fixed.B_reconstructed.tolist(),
            "sk": sk.B_reconstructed.tolist(),
            "tau": tau.tolist(),
            "k_analytic": k_analytic.tolist(),
            "k_quad": k_quad.tolist(),
            "k_sk": closed_form_kernel(tau, sk.poles, sk.residues).tolist(),
        },
    }
    return out


# ---------------------------------------------------------------------------
# case B: real band-limited curve (Hulme hemisphere heave)
# ---------------------------------------------------------------------------
def case_hulme_hemisphere(n_poles=12, data_path=None):
    if data_path is None:
        data_path = Path(__file__).resolve().parents[3] / "tests" / "data" / "time_domain_hemisphere_sweep.json"
    record = json.loads(Path(data_path).read_text())
    omega = np.asarray(record["omega"], dtype=float)
    damping = np.asarray(record["damping"], dtype=float)

    fixed = fixed_pole_fit_damping(omega, damping, n_poles=n_poles)
    sk = sk_fit_damping(omega, damping, n_poles=n_poles)

    tau = tau_grid(omega[-1], span=20.0)
    k_table = radiation_memory_kernel(
        omega, damping, tau, low_frequency_model="cubic"
    ).values
    k_fixed = closed_form_kernel(tau, fixed.poles, fixed.residues)
    k_sk = closed_form_kernel(tau, sk.poles, sk.residues)

    return {
        "reference": record.get("reference"),
        "omega_range": [float(omega[0]), float(omega[-1])],
        "n_points": int(omega.size),
        "n_poles": int(n_poles),
        "damping_range": [float(damping.min()), float(damping.max())],
        "fixed_pole": fit_metrics(fixed, damping),
        "sk": fit_metrics(sk, damping),
        "kernel_fixed_pole_vs_table": kernel_error(tau, k_table, k_fixed),
        "kernel_sk_vs_table": kernel_error(tau, k_table, k_sk),
        "_curves": {
            "omega": omega.tolist(),
            "damping": damping.tolist(),
            "fixed": fixed.B_reconstructed.tolist(),
            "sk": sk.B_reconstructed.tolist(),
            "tau": tau.tolist(),
            "k_table": k_table.tolist(),
            "k_fixed": k_fixed.tolist(),
            "k_sk": k_sk.tolist(),
        },
    }


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def render_markdown(results):
    a = results["exact_model_class"]
    b = results["hulme_hemisphere"]
    rows = [
        ("case", "n_poles", "method", "max rel err B", "rms rel err B", "n_iter", "converged"),
    ]
    for case_key, case in (("exact_model_class", a), ("hulme_hemisphere", b)):
        for method in ("fixed_pole", "sk"):
            m = case[method]
            rows.append(
                (
                    case_key,
                    str(case["n_poles"]),
                    method,
                    f"{m['max_rel_error']:.4e}",
                    f"{m['rms_rel_error']:.4e}",
                    str(m["n_iter"]),
                    str(m["converged"]),
                )
            )

    def table(rows_in):
        head = rows_in[0]
        lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
        for r in rows_in[1:]:
            lines.append("| " + " | ".join(r) + " |")
        return "\n".join(lines)

    ka = a["kernel_analytic_vs_quadrature"]
    lines = [
        "# SK iteration verification (`tensorbem.sk_fit`)",
        "",
        "Rendered programmatically by `python -m tensorbem.benchmarks.sk_fit_verification`.",
        "",
        "## 1. Fit accuracy: SK iteration vs the pre-fix fixed-pole solve",
        "",
        table(rows),
        "",
        "## 2. Memory-kernel consequences",
        "",
        "Case A (synthetic, exact model class) -- closed-form kernel `sum c_k exp(-a_k t)`"
        " against the exact piecewise-linear quadrature of the damping table"
        f" (tau <= {ka['tau_max']:.3f} s, {ka['n_tau']} samples):",
        "",
        f"* analytic identity, max rel err = {ka['max_rel_error']:.4e}",
        f"* fixed-pole fit kernel, max rel err = {a['kernel_error_fixed_pole']['max_rel_error']:.4e}",
        f"* SK fit kernel, max rel err = {a['kernel_error_sk']['max_rel_error']:.4e}",
        "",
        "Case B (Hulme hemisphere heave) -- closed-form kernel of the fitted model against the"
        " quadrature kernel of the table:",
        "",
        f"* fixed-pole fit, max rel err = {b['kernel_fixed_pole_vs_table']['max_rel_error']:.4e}",
        f"* SK fit, max rel err = {b['kernel_sk_vs_table']['max_rel_error']:.4e}",
        "",
    ]
    return "\n".join(lines)


def render_figure(results, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = results["exact_model_class"]["_curves"]
    b = results["hulme_hemisphere"]["_curves"]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0))

    axes[0, 0].loglog(a["omega"], a["damping"], "k-", lw=1.6, label="target (exact class)")
    axes[0, 0].loglog(a["omega"], a["fixed"], "C0--", lw=1.0, label="fixed-pole")
    axes[0, 0].loglog(a["omega"], a["sk"], "C3:", lw=1.6, label="SK iteration")
    axes[0, 0].set_xlabel(r"$\omega$ [rad/s]")
    axes[0, 0].set_ylabel(r"$B(\omega)$")
    axes[0, 0].set_title("case A: synthetic damping (exact model class)")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].semilogy(a["tau"][1:], np.abs(np.asarray(a["k_quad"]) - np.asarray(a["k_analytic"]))[1:] + 1e-30, "k-", lw=1.2, label="identity: analytic vs quadrature")
    axes[0, 1].semilogy(a["tau"][1:], np.abs(np.asarray(a["k_sk"]) - np.asarray(a["k_analytic"]))[1:] + 1e-30, "C3:", lw=1.4, label="SK fit kernel")
    axes[0, 1].set_xlabel(r"$t$ [s]")
    axes[0, 1].set_ylabel("absolute kernel error")
    axes[0, 1].set_title("case A: Cummins kernel error")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].semilogy(b["omega"], b["damping"], "k-", lw=1.6, label="Hulme table")
    axes[1, 0].semilogy(b["omega"], b["fixed"], "C0--", lw=1.0, label="fixed-pole")
    axes[1, 0].semilogy(b["omega"], b["sk"], "C3:", lw=1.6, label="SK iteration")
    axes[1, 0].set_xlabel(r"$\omega$ [rad/s]")
    axes[1, 0].set_ylabel(r"$B(\omega)$")
    axes[1, 0].set_title("case B: Hulme hemisphere heave damping")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].semilogy(b["tau"], np.asarray(b["k_table"]), "k-", lw=1.6, label="quadrature of table")
    axes[1, 1].semilogy(b["tau"], np.asarray(b["k_fixed"]), "C0--", lw=1.0, label="fixed-pole model")
    axes[1, 1].semilogy(b["tau"], np.asarray(b["k_sk"]), "C3:", lw=1.6, label="SK model")
    axes[1, 1].set_xlabel(r"$t$ [s]")
    axes[1, 1].set_ylabel(r"$K(t)$")
    axes[1, 1].set_title("case B: Cummins kernel, model vs quadrature")
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/sk_verification")
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "exact_model_class": case_exact_model_class(),
        "hulme_hemisphere": case_hulme_hemisphere(),
    }
    public = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in results.items()}
    (out_dir / "results.json").write_text(json.dumps(public, indent=2))
    (out_dir / "results.md").write_text(render_markdown(public))
    render_figure(results, out_dir / "fig_sk_verification.png")

    for case, payload in public.items():
        for method in ("fixed_pole", "sk"):
            m = payload[method]
            print(
                f"{case:22s} {method:11s} max_rel={m['max_rel_error']:.4e} "
                f"rms_rel={m['rms_rel_error']:.4e} n_iter={m['n_iter']} converged={m['converged']}"
            )
        kern = {k: v for k, v in payload.items() if k.startswith("kernel")}
        for name, val in kern.items():
            print(f"  {name:32s} max_rel={val['max_rel_error']:.4e}")
    print(f"\nwrote {out_dir}/results.json, results.md, fig_sk_verification.png")
    return public


if __name__ == "__main__":
    main()