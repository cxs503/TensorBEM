"""Maruo far-field mean drift force verification driver (m6271).

Generates every number quoted in ``docs/maruo_drift.md`` so that no digit in
that document is transcribed by hand.  Three things are measured:

1. **Closed-form azimuthal integrals.**  For a scattering pattern
   ``|K(theta)|^2 = C^2 (1 + a cos(theta - gamma))`` the covariant Maruo
   integral is analytic (see ``tests/test_maruo_drift.py``).  The module is
   compared against it over a heading sweep.
2. **The pre-m6271 weight** ``(cos beta - cos(theta - beta),
   sin beta - sin(theta - beta))``, re-implemented here side by side, so the
   before/after error of the fix is quantified rather than asserted.
3. **Where the accuracy floor comes from**: azimuth-table resolution versus
   azimuthal quadrature count.  The integrand is the linear interpolant of the
   caller's table, so refining ``n_theta`` cannot remove table-resolution
   error.

Usage::

    python -m tensorbem.benchmarks.maruo_drift_verification \\
        --json docs/maruo_drift/verification.json --figures docs/maruo_drift
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tensorbem.maruo_drift import (
    maruo_mean_drift_force,
    maruo_mean_drift_force_general,
)

DENSITY = 1025.0
GRAVITY = 9.81
WAVENUMBER = 0.5
#: rho*omega^2/(4k) with omega^2 = g k, i.e. independent of the wave number.
COEFF = DENSITY * GRAVITY / 4.0
#: azimuth samples of the Kochin table used for the heading sweep
N_TABLE = 3600
PATTERN_C = 1.0
PATTERN_A = 0.5
EARTH_HEADINGS = (0.0, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2, 2 * math.pi / 3, math.pi)

AZIMUTH = np.linspace(0.0, 2.0 * math.pi, N_TABLE, endpoint=False)


# ---------------------------------------------------------------------------
# analytic reference
# ---------------------------------------------------------------------------
def pattern(gamma: float, c: float = PATTERN_C, a: float = PATTERN_A) -> np.ndarray:
    """Kochin amplitude table with ``|K|^2 = c^2 (1 + a cos(theta - gamma))``."""
    return c * np.sqrt(np.maximum(1.0 + a * np.cos(AZIMUTH - gamma), 0.0))


def closed_form(beta: float, gamma: float = 0.0, c: float = PATTERN_C, a: float = PATTERN_A):
    """(Fx, Fy) from the covariant Maruo integral -- analytic in theta."""
    s = 2.0 * math.pi * c * c
    moment = a * math.pi * c * c * np.array([math.cos(gamma), math.sin(gamma)])
    return COEFF * (s * np.array([math.cos(beta), math.sin(beta)]) - moment)


# ---------------------------------------------------------------------------
# the two weights
# ---------------------------------------------------------------------------
def covariant_force(beta: float, gamma: float = 0.0) -> np.ndarray:
    """Force from the module under test (covariant weight ``e_beta - e_theta``)."""
    result = maruo_mean_drift_force_general(
        pattern(gamma), AZIMUTH, WAVENUMBER, wave_angle=beta
    )
    return np.array([result["mean_drift_force_x"], result["mean_drift_force_y"]])


def legacy_force(beta: float, gamma: float = 0.0) -> np.ndarray:
    """Force from the pre-m6271 weight ``e_beta - e_{theta-beta}``.

    Re-implemented here (it no longer exists in the library) so the defect
    stays measurable and the regression test has a named counterfactual.
    """
    koch = pattern(gamma)
    ksq = np.abs(koch) ** 2
    dtheta = 2.0 * math.pi / N_TABLE
    flux_x = float(np.sum(ksq * (math.cos(beta) - np.cos(AZIMUTH - beta)) * dtheta))
    flux_y = float(np.sum(ksq * (math.sin(beta) - np.sin(AZIMUTH - beta)) * dtheta))
    return COEFF * np.array([flux_x, flux_y])


def _rel_err(force: np.ndarray, reference: np.ndarray) -> float:
    scale = np.linalg.norm(reference)
    return float(np.max(np.abs(force - reference)) / scale) if scale else float(np.max(np.abs(force)))


# ---------------------------------------------------------------------------
# studies
# ---------------------------------------------------------------------------
def heading_study(headings=EARTH_HEADINGS) -> list[dict]:
    """Closed-form comparison over a heading sweep, covariant versus legacy."""
    rows = []
    for beta in headings:
        reference = closed_form(beta)
        cov = covariant_force(beta)
        leg = legacy_force(beta)
        rows.append(
            {
                "beta_deg": round(math.degrees(beta), 3),
                "fx_covariant": cov[0],
                "fy_covariant": cov[1],
                "fx_legacy": leg[0],
                "fy_legacy": leg[1],
                "fx_reference": reference[0],
                "fy_reference": reference[1],
                "rel_err_covariant": _rel_err(cov, reference),
                "rel_err_legacy": _rel_err(leg, reference),
                "abs_err_fx_legacy": abs(leg[0] - reference[0]),
                "abs_err_fy_legacy": abs(leg[1] - reference[1]),
            }
        )
    return rows


def covariance_study() -> list[dict]:
    """Rotate body and incident waves together by ``alpha``; F must rotate with them."""
    rows = []
    gamma = 0.3
    for alpha in (0.3, 0.7, 1.3, 2.5):
        beta = math.pi / 3
        f0 = covariant_force(beta, gamma)
        f1 = covariant_force(beta + alpha, gamma + alpha)
        g0 = legacy_force(beta, gamma)
        g1 = legacy_force(beta + alpha, gamma + alpha)
        rot = np.array(
            [[math.cos(alpha), -math.sin(alpha)], [math.sin(alpha), math.cos(alpha)]]
        )
        rows.append(
            {
                "alpha_deg": round(math.degrees(alpha), 3),
                "beta_deg": round(math.degrees(beta), 3),
                "rel_residual_covariant": _rel_err(f1, rot @ f0),
                "rel_residual_legacy": _rel_err(g1, rot @ g0),
            }
        )
    return rows


def resolution_study() -> dict:
    """Accuracy floor: table spacing versus quadrature count (dipole pattern)."""
    magnitude = COEFF * math.pi
    table_rows = []
    for n in (8, 12, 24, 48, 96, 360, 3600):
        theta = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
        force = maruo_mean_drift_force(np.cos(theta), theta, WAVENUMBER)["mean_drift_force_x"]
        table_rows.append(
            {"n_table": n, "force": force, "rel_err": abs(force - magnitude) / magnitude}
        )
    quadrature_rows = []
    fine = np.linspace(0.0, 2.0 * math.pi, N_TABLE, endpoint=False)
    for n_theta in (128, 256, 512, 1024, 4096, 16384):
        force = maruo_mean_drift_force(
            np.cos(fine), fine, WAVENUMBER, n_theta=n_theta
        )["mean_drift_force_x"]
        quadrature_rows.append(
            {"n_theta": n_theta, "force": force, "rel_err": abs(force - magnitude) / magnitude}
        )
    return {
        "dipole_magnitude_analytic": magnitude,
        "table": table_rows,
        "quadrature": quadrature_rows,
    }


def study() -> dict:
    headings = heading_study()
    covariance = covariance_study()
    resolution = resolution_study()
    legacy_errors = [r["rel_err_legacy"] for r in headings]
    covariant_errors = [r["rel_err_covariant"] for r in headings]
    return {
        "schema": "tensorbem.maruo_drift_verification.v1",
        "mission": "m6271",
        "parameters": {
            "density": DENSITY,
            "gravity": GRAVITY,
            "wavenumber": WAVENUMBER,
            "prefactor_rho_g_over_4": COEFF,
            "n_table": N_TABLE,
            "pattern": "|K|^2 = 1 + 0.5 cos(theta - gamma)",
            "azimuth_convention": "theta and beta share one frame; 0 = +x",
        },
        "headings": headings,
        "summary": {
            "rel_err_covariant_max": max(covariant_errors),
            "rel_err_legacy_max": max(legacy_errors),
            "rel_err_legacy_min": min(legacy_errors),
            "covariance_residual_covariant_max": max(
                r["rel_residual_covariant"] for r in covariance
            ),
            "covariance_residual_legacy_max": max(r["rel_residual_legacy"] for r in covariance),
            "closed_form_within_3pct_covariant": max(covariant_errors) < 0.03,
            "closed_form_within_3pct_legacy": max(legacy_errors) < 0.03,
        },
        "covariance": covariance,
        "resolution": resolution,
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def figure_polar(path: Path, report: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    headings = report["headings"]
    fig, ax = plt.subplots(figsize=(6.2, 5.0), subplot_kw={"projection": "polar"})
    for key, label, style in (
        ("covariant", "covariant weight (m6271 fix)", dict(color="tab:blue", marker="o")),
        ("legacy", "pre-m6271 weight", dict(color="tab:red", marker="s", linestyle="--")),
    ):
        beta = np.array([math.radians(r["beta_deg"]) for r in headings])
        mag = np.array(
            [math.hypot(r[f"fx_{key}"], r[f"fy_{key}"]) for r in headings]
        )
        ax.plot(beta, mag, label=label, linewidth=1.6, markersize=5, **style)
    beta = np.array([math.radians(r["beta_deg"]) for r in headings])
    ref = np.array([math.hypot(r["fx_reference"], r["fy_reference"]) for r in headings])
    ax.plot(beta, ref, color="k", linewidth=1.0, linestyle=":", label="closed form (analytic)")
    ax.set_theta_zero_location("E")
    ax.set_title(
        "Mean drift force magnitude vs wave heading\n"
        r"$|K|^2=1+0.5\cos\theta$, $k=0.5$ m$^{-1}$",
        fontsize=10,
    )
    ax.legend(loc="lower left", fontsize=7, bbox_to_anchor=(-0.18, -0.12))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_resolution(path: Path, report: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = report["resolution"]
    t = res["table"]
    q = res["quadrature"]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.loglog(
        [r["n_table"] for r in t],
        [r["rel_err"] for r in t],
        "o-",
        label="azimuth-table refinement ($n_\\theta$=8192)",
    )
    ax.loglog(
        [r["n_theta"] for r in q],
        [r["rel_err"] for r in q],
        "s--",
        label="quadrature refinement (table = 3600 samples)",
    )
    n = np.array([r["n_table"] for r in t], dtype=float)
    ref = t[-1]["rel_err"] * (n[-1] / n) ** 2
    ax.loglog(n, ref, "k:", label="slope 2 reference")
    ax.set_xlabel("samples / points")
    ax.set_ylabel("relative error of $\\bar F_x$ (dipole, analytic $\\pi\\rho g/4$)")
    ax.set_title("Accuracy is set by the Kochin table, not by the $\\theta$ quadrature", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=Path("docs/maruo_drift/verification.json"))
    parser.add_argument("--figures", type=Path, default=None)
    args = parser.parse_args(argv)

    report = study()
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.figures is not None:
        args.figures.mkdir(parents=True, exist_ok=True)
        figure_polar(args.figures / "maruo_drift_heading_polar.png", report)
        figure_resolution(args.figures / "maruo_drift_resolution.png", report)

    summary = report["summary"]
    print(f"covariant max rel. error vs closed form : {summary['rel_err_covariant_max']:.3e}")
    print(f"legacy    max rel. error vs closed form : {summary['rel_err_legacy_max']:.3e}")
    print(f"covariance residual (covariant / legacy) : "
          f"{summary['covariance_residual_covariant_max']:.3e} / "
          f"{summary['covariance_residual_legacy_max']:.3e}")
    print(f"written: {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())