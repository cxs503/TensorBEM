"""Build the m6287 artefact, figure and report for the Maruo heading fix.

Everything in ``docs/m6287/maruo_general_heading_fix.md`` is rendered from
``docs/m6287/maruo_heading_fix.json`` by ``render_report.py``; nothing is
transcribed by hand.

The pre-fix expression ``e_beta - e_{theta-beta}`` is re-implemented here (it is
no longer in the source tree) so that the before/after comparison is
reproducible from the committed code alone.

Usage:  python docs/m6287/build_report.py
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorbem.maruo_drift import (  # noqa: E402
    maruo_mean_drift_force,
    maruo_mean_drift_force_general,
)

DENSITY = 1025.0
GRAVITY = 9.81
COEFF = DENSITY * GRAVITY / 4.0  # rho*omega^2/(4k) in deep water


def pre_fix_general(kochin, azimuths, wavenumber, wave_angle, *, n_theta=8192,
                    density=DENSITY, gravity=GRAVITY):
    """The general-heading expression as it stood before the fix.

    ``F = coeff * sum |K|^2 (e_beta - e_{theta - beta})`` -- the radiation
    direction was rotated with the heading, which is only right at beta = 0.
    """

    omega = math.sqrt(gravity * wavenumber)
    theta = 2.0 * math.pi * np.arange(n_theta) / n_theta
    dtheta = 2.0 * math.pi / n_theta
    koch = np.asarray(kochin, dtype=np.complex128)
    ang = np.mod(np.asarray(azimuths, dtype=float), 2.0 * math.pi)
    order = np.argsort(ang)
    ang, koch = ang[order], koch[order]
    ang = np.concatenate((ang, ang[:1] + 2.0 * math.pi))
    koch = np.concatenate((koch, koch[:1]))
    interp = np.interp(theta, ang, koch.real) + 1j * np.interp(theta, ang, koch.imag)
    k_sq = np.abs(interp) ** 2
    flux_x = float(np.sum(k_sq * (math.cos(wave_angle) - np.cos(theta - wave_angle)) * dtheta))
    flux_y = float(np.sum(k_sq * (math.sin(wave_angle) - np.sin(theta - wave_angle)) * dtheta))
    coeff = density * omega * omega / (4.0 * wavenumber)
    return {"mean_drift_force_x": coeff * flux_x, "mean_drift_force_y": coeff * flux_y}


def grid(n: int = 4096) -> np.ndarray:
    return 2.0 * math.pi * np.arange(n) / n


def intensity(theta, gamma: float, a: float = 0.5):
    return 1.0 + a * np.cos(np.asarray(theta) - gamma)


def exact_vector(gamma: float, a: float = 0.5) -> tuple[float, float]:
    """Closed form for ``|K|^2 = 1 + a cos(theta - gamma)`` at heading ``gamma``."""

    magnitude = COEFF * (2.0 * math.pi - a * math.pi)
    return magnitude * math.cos(gamma), magnitude * math.sin(gamma)


def rel(vec, ref) -> float:
    ref_norm = math.hypot(*ref)
    return math.hypot(vec[0] - ref[0], vec[1] - ref[1]) / ref_norm


def main() -> int:
    theta = grid(4096)
    k, rho, g = 0.7, DENSITY, GRAVITY

    headings = [0.0, math.pi / 12.0, math.pi / 6.0, math.pi / 4.0, math.pi / 3.0,
                math.pi / 2.0, 2.0, math.pi]
    sweep = []
    for gamma in headings:
        koch = np.sqrt(intensity(theta, gamma)).astype(np.complex128)
        new = maruo_mean_drift_force_general(koch, theta, k, wave_angle=gamma, n_theta=4096)
        old = pre_fix_general(koch, theta, k, gamma, n_theta=4096)
        ex = exact_vector(gamma)
        sweep.append({
            "heading_deg": math.degrees(gamma),
            "exact": list(ex),
            "pre_fix": [old["mean_drift_force_x"], old["mean_drift_force_y"]],
            "fixed": [new["mean_drift_force_x"], new["mean_drift_force_y"]],
            "pre_fix_rel_error": rel((old["mean_drift_force_x"], old["mean_drift_force_y"]), ex),
            "fixed_rel_error": rel((new["mean_drift_force_x"], new["mean_drift_force_y"]), ex),
        })

    # isotropic + head-seas closed-form anchors, measured
    k0 = 0.37
    iso = maruo_mean_drift_force_general(
        np.full_like(theta, k0).astype(np.complex128), theta, k, wave_angle=math.pi / 3.0,
        n_theta=4096,
    )
    iso_exact = (2.0 * math.pi * k0 * k0 * COEFF * math.cos(math.pi / 3.0),
                 2.0 * math.pi * k0 * k0 * COEFF * math.sin(math.pi / 3.0))
    head = maruo_mean_drift_force(
        np.sqrt(1.0 + 0.5 * np.cos(theta)).astype(np.complex128), theta, k, n_theta=4096
    )
    head_exact = COEFF * (2.0 * math.pi - 0.5 * math.pi)

    anchors = [
        {"name": "isotropic scatterer, heading beta = 60 deg (vector)",
         "computed": [iso["mean_drift_force_x"], iso["mean_drift_force_y"]],
         "exact": list(iso_exact),
         "rel_error": rel((iso["mean_drift_force_x"], iso["mean_drift_force_y"]), iso_exact)},
        {"name": "head seas, |K|^2 = 1 + 0.5 cos(theta)",
         "computed": [head["mean_drift_force_x"]],
         "exact": [head_exact],
         "rel_error": abs(head["mean_drift_force_x"] - head_exact) / head_exact},
    ]

    worst_pre = max(row["pre_fix_rel_error"] for row in sweep)
    worst_post = max(row["fixed_rel_error"] for row in sweep)
    spurious = [row for row in sweep if row["heading_deg"] > 1.0 and row["heading_deg"] < 179.0]
    worst_spurious_pre = max(spurious, key=lambda r: abs(r["pre_fix"][0]))

    payload = {
        "mission": "m6287",
        "repo": "TensorBEM",
        "subject": "maruo_mean_drift_force_general heading convention",
        "parameters": {
            "wavenumber_k": k,
            "omega": math.sqrt(g * k),
            "density": rho,
            "gravity": g,
            "coefficient_rho_omega2_over_4k": COEFF,
            "scatterer": "|K|^2(theta) = 1 + 0.5 cos(theta - gamma), theta the absolute "
                         "radiation azimuth, gamma the body rotation = wave heading",
            "azimuth_table": len(theta),
            "quadrature_points": 4096,
        },
        "sweep": sweep,
        "anchors": anchors,
        "verdict": {
            "worst_pre_fix_rel_error": worst_pre,
            "worst_fixed_rel_error": worst_post,
            "worst_pre_fix_oblique_heading_deg": worst_spurious_pre["heading_deg"],
            "worst_pre_fix_oblique_transverse_force": worst_spurious_pre["pre_fix"][0],
            "worst_pre_fix_oblique_transverse_exact": worst_spurious_pre["exact"][0],
            "gate": 0.03,
            "fixed_passes_gate": worst_post < 0.03,
        },
    }
    (HERE / "maruo_heading_fix.json").write_text(json.dumps(payload, indent=2) + "\n")

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = HERE / "figures"
    figs.mkdir(exist_ok=True)
    dense = np.linspace(0.0, math.pi, 121)
    old_curve, new_curve, ex_curve = [], [], []
    for gamma in dense:
        koch = np.sqrt(intensity(theta, gamma)).astype(np.complex128)
        newv = maruo_mean_drift_force_general(koch, theta, k, wave_angle=gamma, n_theta=4096)
        oldv = pre_fix_general(koch, theta, k, gamma, n_theta=4096)
        old_curve.append(rel((oldv["mean_drift_force_x"], oldv["mean_drift_force_y"]), exact_vector(gamma)))
        new_curve.append(rel((newv["mean_drift_force_x"], newv["mean_drift_force_y"]), exact_vector(gamma)))
        ex_curve.append(exact_vector(gamma)[0])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    deg = np.degrees(dense)
    ax.plot(deg, np.array(old_curve) * 100.0, label="pre-fix  $e_{θ-β}$", lw=2.0)
    ax.plot(deg, np.maximum(np.array(new_curve) * 100.0, 1e-12), label="fixed  $e_{θ}$", lw=2.0)
    ax.axhline(3.0, color="k", ls="--", lw=1.0, label="3 % acceptance band")
    ax.set_yscale("log")
    ax.set_xlabel("wave heading β = body rotation γ [deg]")
    ax.set_ylabel("drift-force vector error vs closed form [%]")
    ax.set_title(r"$|K|^2 = 1 + 0.5\cos(\theta-\gamma)$: heading error")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(np.degrees(dense), ex_curve, "k-", lw=2.0, label="closed form $F_x$")
    pre_x = [pre_fix_general(np.sqrt(intensity(theta, gm)).astype(np.complex128), theta, k, gm,
                             n_theta=4096)["mean_drift_force_x"] for gm in dense]
    fix_x = [maruo_mean_drift_force_general(np.sqrt(intensity(theta, gm)).astype(np.complex128),
                                           theta, k, wave_angle=gm,
                                           n_theta=4096)["mean_drift_force_x"] for gm in dense]
    ax.plot(np.degrees(dense), pre_x, "--", lw=1.6, label="pre-fix $F_x$")
    ax.plot(np.degrees(dense), fix_x, ":", lw=2.4, label="fixed $F_x$")
    ax.set_xlabel("wave heading β = body rotation γ [deg]")
    ax.set_ylabel("longitudinal drift force $\\bar F_x$ [N/m]")
    ax.set_title("longitudinal component: spurious peak near beam seas")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "fig1_heading_error.png", dpi=150)
    print(f"wrote {figs / 'fig1_heading_error.png'}")

    print(json.dumps(payload["verdict"], indent=2))
    print("anchors:", json.dumps(anchors, indent=2)[:600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())