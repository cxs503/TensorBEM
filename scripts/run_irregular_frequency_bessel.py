#!/usr/bin/env python3
"""Entry point for CASE-BEM-IRREG-FREQ-CYL: irregular-frequency Bessel-zero prediction.

Produces run artifacts demonstrating the TensorBEM irregular-frequency predictor
against analytical Bessel-zero references for a surface-piercing circular cylinder.

Usage:
    python scripts/run_irregular_frequency_bessel.py [--out-dir DIR]

Artifacts produced:
    <out_dir>/prediction.json      — predicted vs reference Bessel zeros
    <out_dir>/ref_vs_got.json      — numerical comparison with error analysis
    <out_dir>/run_log.txt          — execution log
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# Ensure repo root is on sys.path
REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tensorbem.diffraction_drift_maturity import vertical_cylinder_hull_mesh
from tensorbem.irregular_frequency import (
    predict_irregular_frequencies,
    _BESSEL_J01,
    _BESSEL_J11,
    _BESSEL_J21,
)

DEFAULT_GRAVITY = 9.80665

#: Analytical Bessel zeros (reference, from Abramowitz & Stegun Table 9.5)
BESSEL_REF = {
    "j01": 2.404825557695773,
    "j11": 3.831705970207512,
    "j21": 5.135622301840683,
}

MODE_LABELS = ["j01_axisymmetric", "j11_dipole", "j21_quadrupole"]


def run_case(
    radius: float = 1.0,
    draft_ratio: float = 0.5,
    n_theta: int = 64,
    n_levels: int = 5,
    gravity: float = DEFAULT_GRAVITY,
) -> dict:
    """Run the irregular-frequency Bessel-zero prediction case.

    Parameters
    ----------
    radius : float
        Cylinder radius (m).
    draft_ratio : float
        Draft-to-radius ratio T/R.
    n_theta : int
        Azimuthal panel count.
    n_levels : int
        Vertical panel count.
    gravity : float
        Gravity (m/s²).

    Returns
    -------
    dict
        Full results including prediction, reference, and error analysis.
    """
    draft = radius * draft_ratio
    t0 = time.time()

    # Build mesh
    mesh = vertical_cylinder_hull_mesh(radius, draft, n_theta=n_theta, n_levels=n_levels)

    # Predict irregular frequencies
    pred = predict_irregular_frequencies(mesh, gravity=gravity, n_modes=3)

    elapsed = time.time() - t0

    # Reference: analytical Bessel zeros / true radius
    ref_nu = np.array([BESSEL_REF["j01"], BESSEL_REF["j11"], BESSEL_REF["j21"]]) / radius
    ref_omega = np.sqrt(gravity * ref_nu)

    # Error analysis
    nu_errors = np.abs(pred.predicted_nu - ref_nu) / ref_nu * 100
    omega_errors = np.abs(pred.predicted_omega - ref_omega) / ref_omega * 100

    # Waterplane radius error (polygon approximation vs true circle)
    r_wp_error = abs(pred.waterplane_radius - radius) / radius * 100

    result = {
        "case": "CASE-BEM-IRREG-FREQ-CYL",
        "geometry": {
            "type": "surface-piercing circular cylinder",
            "radius_m": radius,
            "draft_m": draft,
            "draft_ratio": draft_ratio,
        },
        "mesh": {
            "n_theta": n_theta,
            "n_levels": n_levels,
            "panels": mesh.panel_count,
            "waterline_edges": int(mesh.waterline_edges.shape[0]),
        },
        "waterplane_radius": {
            "predicted_m": float(pred.waterplane_radius),
            "true_m": radius,
            "error_pct": float(r_wp_error),
        },
        "prediction": {
            "nu_rad_per_m": pred.predicted_nu.tolist(),
            "omega_rad_per_s": pred.predicted_omega.tolist(),
            "mode_labels": pred.mode_labels,
        },
        "reference": {
            "bessel_zeros": BESSEL_REF,
            "nu_rad_per_m": ref_nu.tolist(),
            "omega_rad_per_s": ref_omega.tolist(),
            "source": "Abramowitz & Stegun, Handbook of Mathematical Functions, Table 9.5",
        },
        "error_analysis": {
            "nu_error_pct": nu_errors.tolist(),
            "omega_error_pct": omega_errors.tolist(),
            "max_nu_error_pct": float(np.max(nu_errors)),
            "max_omega_error_pct": float(np.max(omega_errors)),
            "tolerance_pct": 2.0,
            "all_within_tolerance": bool(np.all(nu_errors < 2.0)),
        },
        "runtime_s": round(elapsed, 3),
    }

    return result


def main():
    out_dir = REPO / "runs" / "m17374_irregular_frequency_bessel"
    out_dir.mkdir(parents=True, exist_ok=True)

    log_lines = []

    def log(msg: str):
        log_lines.append(msg)
        print(msg)

    log("=" * 70)
    log("CASE-BEM-IRREG-FREQ-CYL: Irregular-Frequency Bessel-Zero Prediction")
    log("=" * 70)

    # Run multiple configurations
    configs = [
        {"radius": 1.0, "draft_ratio": 0.5, "n_theta": 32, "n_levels": 3, "label": "coarse"},
        {"radius": 1.0, "draft_ratio": 0.5, "n_theta": 64, "n_levels": 5, "label": "medium"},
        {"radius": 1.0, "draft_ratio": 0.5, "n_theta": 128, "n_levels": 8, "label": "fine"},
        {"radius": 2.0, "draft_ratio": 1.0, "n_theta": 64, "n_levels": 5, "label": "R2_T2"},
    ]

    all_results = []
    for cfg in configs:
        label = cfg.pop("label")
        log(f"\n--- Configuration: {label} ---")
        result = run_case(**cfg)
        result["label"] = label
        all_results.append(result)

        log(f"  Mesh: {result['mesh']['panels']} panels")
        log(f"  R_wp: {result['waterplane_radius']['predicted_m']:.6f} m (true: {result['waterplane_radius']['true_m']} m, error: {result['waterplane_radius']['error_pct']:.4f}%)")
        log(f"  Predicted nu: {result['prediction']['nu_rad_per_m']}")
        log(f"  Reference nu: {result['reference']['nu_rad_per_m']}")
        log(f"  Max nu error: {result['error_analysis']['max_nu_error_pct']:.4f}%")
        log(f"  Within 2% tolerance: {result['error_analysis']['all_within_tolerance']}")

    # Save prediction artifacts
    prediction_data = {
        "case": "CASE-BEM-IRREG-FREQ-CYL",
        "description": "Irregular-frequency Bessel-zero prediction for surface-piercing circular cylinder",
        "configurations": all_results,
    }
    pred_path = out_dir / "prediction.json"
    with open(pred_path, "w") as f:
        json.dump(prediction_data, f, indent=2)
    log(f"\nSaved: {pred_path}")

    # Save ref vs got comparison
    ref_vs_got = {
        "case": "CASE-BEM-IRREG-FREQ-CYL",
        "reference_source": "Abramowitz & Stegun, Handbook of Mathematical Functions, Table 9.5",
        "bessel_zeros": BESSEL_REF,
        "comparisons": [],
    }
    for r in all_results:
        comp = {
            "label": r["label"],
            "mesh_panels": r["mesh"]["panels"],
            "waterplane_radius_error_pct": r["waterplane_radius"]["error_pct"],
            "nu": {
                "predicted": r["prediction"]["nu_rad_per_m"],
                "reference": r["reference"]["nu_rad_per_m"],
                "error_pct": r["error_analysis"]["nu_error_pct"],
            },
            "omega": {
                "predicted": r["prediction"]["omega_rad_per_s"],
                "reference": r["reference"]["omega_rad_per_s"],
                "error_pct": r["error_analysis"]["omega_error_pct"],
            },
            "max_error_pct": r["error_analysis"]["max_nu_error_pct"],
            "within_2pct_tolerance": r["error_analysis"]["all_within_tolerance"],
        }
        ref_vs_got["comparisons"].append(comp)

    ref_vs_got_path = out_dir / "ref_vs_got.json"
    with open(ref_vs_got_path, "w") as f:
        json.dump(ref_vs_got, f, indent=2)
    log(f"Saved: {ref_vs_got_path}")

    # Save run log
    log_path = out_dir / "run_log.txt"
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    log(f"Saved: {log_path}")

    # Summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    for r in all_results:
        log(f"  {r['label']:10s}: panels={r['mesh']['panels']:4d}, max_nu_err={r['error_analysis']['max_nu_error_pct']:.4f}%, pass={r['error_analysis']['all_within_tolerance']}")
    log("=" * 70)


if __name__ == "__main__":
    main()
