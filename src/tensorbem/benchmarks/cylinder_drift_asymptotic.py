"""Cylinder mean drift force: exact two-formula verification and ``(ka)^3`` law.

Driver for ``docs/cylinder_drift_asymptotic.md``.  It evaluates, for a
bottom-mounted vertical circular cylinder in deep water,

* the **exact** MacCamy-Fuchs near-field momentum integral over the wetted wall
  (surface + waterline terms),
* the **exact** reduced Maruo far-field integral over the scattering amplitude
  ``S(theta)`` (no radiation-radius dependence),
* their agreement point by point, the ``(ka)^3`` long-wave law against the closed
  form ``5 pi^2/16``, the Kochin energy balance (total outward flux must vanish
  for a non-absorbing body) at several probe radii, and
* an audit of the repository's 3-D Maruo prefactor convention.

Run::

    PYTHONPATH=src python3 -m tensorbem.benchmarks.cylinder_drift_asymptotic \\
        --json artifacts/cylinder_drift_asymptotic/study.json \\
        --figures artifacts/cylinder_drift_asymptotic
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

from .. import cylinder_drift as cd

RHO = cd.DEFAULT_DENSITY
G = cd.DEFAULT_GRAVITY
KA_SCAN = [round(v, 6) for v in np.linspace(0.2, 2.0, 8)]
KA_LONG_WAVE = [0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]
PROBE_RADII = [5.0, 10.0, 20.0]
ENERGY_KA = [0.5, 1.0, 1.5]


def _wall_anchor() -> dict:
    """Fully reflecting vertical wall: near-field formula must give rho g A^2/2."""
    k = 1.0
    amp = 1.0
    rho, g = RHO, G
    depth_integral = 1.0 / (2.0 * k)
    # standing wave psi = 2 cos(kx): |psi|^2 = 4, |dpsi/dx|^2 = 0 at the wall,
    # |zeta_hat| = 2 A on the wall, n_x = -1 (body is the half space x > 0).
    grad_sq_depth_integrated = amp**2 * g / k * (k**2 * 4.0) * depth_integral
    n_x = -1.0
    surface = 0.25 * rho * grad_sq_depth_integrated * n_x
    waterline = -0.25 * rho * g * (2.0 * amp) ** 2 * n_x
    total = surface + waterline
    return {
        "case": "fully reflecting vertical wall, deep water",
        "surface_term": surface,
        "waterline_term": waterline,
        "near_field_force": total,
        "anchor_rho_g_A2_over_2": 0.5 * rho * g * amp**2,
        "rel_error": abs(total - 0.5 * rho * g * amp**2) / (0.5 * rho * g * amp**2),
    }


def _rows() -> list[dict]:
    rows = []
    for ka in KA_SCAN:
        near = cd.near_field_cylinder_drift(ka)
        far = cd.far_field_cylinder_drift(ka)
        no_wl = near["mean_drift_force_x"] - (-0.25 * RHO * G * near["waterline_integral"])
        rows.append(
            {
                "ka": ka,
                "near_field_force": near["mean_drift_force_x"],
                "far_field_force": far["mean_drift_force_x"],
                "near_far_rel_diff": abs(near["mean_drift_force_x"] - far["mean_drift_force_x"])
                / abs(far["mean_drift_force_x"]),
                "surface_integral": near["surface_integral"],
                "waterline_integral": near["waterline_integral"],
                "normalised_force": far["mean_drift_force_x"] / (RHO * G * (ka**3)),
                "kochin_flux_weighted": far["flux_integral_weighted"],
                "kochin_flux_total": far["flux_integral_total"],
            }
        )
    return rows


def _long_wave() -> dict:
    ks = np.asarray(KA_LONG_WAVE, dtype=float)
    coeffs = np.asarray(
        [cd.far_field_cylinder_drift(k)["mean_drift_force_x"] / (RHO * G * k**3) for k in ks]
    )
    fit_mask = ks <= 0.3
    slope, intercept = np.polyfit(np.log(ks[fit_mask]), np.log(coeffs[fit_mask] * ks[fit_mask] ** 3), 1)
    closed = cd.long_wave_drift_coefficient()
    return {
        "ka_values": KA_LONG_WAVE,
        "normalised_coefficient": coeffs.tolist(),
        "closed_form_coefficient": closed,
        "coefficient_rel_error_at_smallest_ka": abs(coeffs[0] - closed) / closed,
        "fitted_slope": float(slope),
        "fitted_slope_target": 3.0,
        "fitted_slope_tolerance": 0.2,
        "fit_band": [float(ks[fit_mask][0]), float(ks[fit_mask][-1])],
        "ansatz": "S ~ i kappa (2 cos theta - 1), kappa = pi (ka)^2 / 4",
    }


def _energy() -> list[dict]:
    rows = []
    for ka in ENERGY_KA:
        for radius in PROBE_RADII:
            bal = cd.energy_balance(ka, radius_probe=radius)
            rows.append(
                {
                    "ka": ka,
                    "probe_radius": radius,
                    "flux_scattered": bal["flux_scattered"],
                    "flux_interference": bal["flux_interference"],
                    "flux_total": bal["flux_total"],
                    "residual": bal["residual"],
                }
            )
    return rows


def _repo_convention_audit() -> dict:
    """Compare the repository's 3-D Maruo prefactor with the exact calibration."""
    from ..maruo_drift import maruo_mean_drift_force

    ka = 1.0
    theta = np.linspace(0.0, 2.0 * math.pi, 2048, endpoint=False)
    kochin = cd.kochin_3d(ka, theta)
    repo = maruo_mean_drift_force(kochin, theta, ka)
    exact = cd.far_field_cylinder_drift(ka)["mean_drift_force_x"]
    ratio = repo["mean_drift_force_x"] / exact
    return {
        "ka": ka,
        "repo_maruo_force": repo["mean_drift_force_x"],
        "exact_force": exact,
        "ratio_repo_over_exact": ratio,
        "two_pi": 2.0 * math.pi,
        "convention_note": (
            "tensorbem.maruo_drift applies the classical 1/R Kochin weight "
            "rho*omega^2/(4k); the free-surface wave part of the present "
            "deep-water solution decays as R^(-1/2) with the (i k/2) H_0^(1) "
            "normalisation, so the same |K|^2 integral carries rho g A^2 k/(8 pi)."
        ),
    }


def build_study() -> dict:
    rows = _rows()
    long_wave = _long_wave()
    energy = _energy()
    wall = _wall_anchor()
    audit = _repo_convention_audit()

    max_nf = max(r["near_far_rel_diff"] for r in rows)
    max_energy = max(r["residual"] for r in energy)
    verdict = {
        "near_far_max_rel_diff": max_nf,
        "near_far_target": 0.05,
        "near_far_pass": bool(max_nf < 0.05),
        "long_wave_slope": long_wave["fitted_slope"],
        "long_wave_slope_pass": bool(abs(long_wave["fitted_slope"] - 3.0) <= 0.2),
        "long_wave_coefficient_rel_error": long_wave["coefficient_rel_error_at_smallest_ka"],
        "long_wave_coefficient_pass": bool(
            long_wave["coefficient_rel_error_at_smallest_ka"] < 0.03
        ),
        "energy_residual_max": max_energy,
        "energy_target": 0.01,
        "energy_pass": bool(max_energy < 0.01),
        "wall_anchor_rel_error": wall["rel_error"],
        "wall_anchor_pass": bool(wall["rel_error"] < 1e-12),
        "radius_independence_pass": bool(max_energy < 1e-9),
        "all_gates_pass": bool(
            max_nf < 0.05
            and abs(long_wave["fitted_slope"] - 3.0) <= 0.2
            and long_wave["coefficient_rel_error_at_smallest_ka"] < 0.03
            and max_energy < 0.01
        ),
        "scope_limits": [
            "deep water only (omega^2 = g k); finite depth is not covered here",
            "vertical circular cylinder, bottom-mounted, rigid and fixed: no "
            "forward speed, no mooring, no second-order potential",
            "Wigley-hull RAO/first-order excitation comparison against towing-tank "
            "experiments is NOT part of this study (no digitised experimental "
            "data set is carried in the repository; inventing one would be "
            "fabrication)",
        ],
    }
    return {
        "case": {
            "geometry": "bottom-mounted vertical circular cylinder, radius a, deep water",
            "motion": "fixed (no radiation)",
            "density_kg_m3": RHO,
            "gravity_m_s2": G,
            "amplitude_m": 1.0,
            "radius_m": 1.0,
            "ka_scan": KA_SCAN,
            "ka_long_wave_scan": KA_LONG_WAVE,
        },
        "wall_anchor": wall,
        "rows": rows,
        "long_wave": long_wave,
        "energy_balance": energy,
        "repo_convention_audit": audit,
        "verdict": verdict,
    }


def make_figures(study: dict, outdir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    lw = study["long_wave"]
    rows = study["rows"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    ks = np.asarray(lw["ka_values"])
    coeff = np.asarray(lw["normalised_coefficient"])
    ax.semilogx(ks, coeff, "o-", label=r"exact MF far field,  $F/(\rho g A^2 (ka)^3)$")
    ax.axhline(
        lw["closed_form_coefficient"],
        color="k",
        ls="--",
        label=r"closed form $5\pi^2/16 = %.4f$" % lw["closed_form_coefficient"],
    )
    ax.axvspan(0.24, 1.74, color="tab:orange", alpha=0.12, label=r"$ka$ scan of table 1")
    ax.set_xlabel(r"$ka$")
    ax.set_ylabel("normalised drift force")
    ax.set_title("Long-wave $(ka)^3$ law: exact vs closed form (slope %.3f)"
                 % lw["fitted_slope"])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    path = outdir / "fig_long_wave_law.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    written.append(str(path))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.0), dpi=150)
    ka = [r["ka"] for r in rows]
    near = np.asarray([r["near_field_force"] for r in rows])
    far = np.asarray([r["far_field_force"] for r in rows])
    ax1.loglog(ka, near, "o", label="near field (wall pressure + waterline)")
    ax1.loglog(ka, far, "x", label="far field (reduced Maruo kernel)")
    ax1.set_xlabel(r"$ka$")
    ax1.set_ylabel(r"$\bar F_x$  [N]")
    ax1.set_title("Two independent formulas agree to %.1e"
                  % max(r["near_far_rel_diff"] for r in rows))
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")

    energy = study["energy_balance"]
    labels = sorted({(e["ka"], e["probe_radius"]) for e in energy}, key=lambda t: (t[0], t[1]))
    res = [next(e["residual"] for e in energy if (e["ka"], e["probe_radius"]) == lab) for lab in labels]
    ax2.semilogy(range(len(labels)), res, "s")
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels([r"$ka$=%.1f\n$R$=%g" % lab for lab in labels], fontsize=6)
    ax2.axhline(0.01, color="k", ls="--", label="1 % gate")
    ax2.set_ylabel("energy-balance residual")
    ax2.set_title("Kochin energy balance (non-absorbing body)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    path = outdir / "fig_near_far_energy.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    written.append(str(path))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--figures", type=Path, default=None)
    args = parser.parse_args(argv)

    study = build_study()
    study["figures"] = make_figures(study, args.figures) if args.figures else []
    payload = json.dumps(study, indent=2, sort_keys=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload + "\n")
    print(payload)
    return 0 if study["verdict"]["all_gates_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())