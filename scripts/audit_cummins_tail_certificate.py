#!/usr/bin/env python3
"""Audit the ``C3a`` truncation budget of CASE-BEM-CUMMINS-WIGLEY on real data.

Produces ``docs/case_bem_cummins_wigley_tail_certificate.json`` and
``docs/fig_m22460_cummins_tail_certificate.png``.

What is measured
----------------
* per DOF: the closed-form power-tail budget ``budget_relative`` and the
  allowance ``kappa = tolerance/budget_relative`` (how much the real damping may
  exceed its own continuation before the omitted band alone could move the
  residual past the 10% case tolerance);
* the omission actually carried by the committed 12..20 rad/s probe
  (``omission_lower_relative``, a strict lower bound), and what fraction of it
  comes from below the probe's own ``C2`` resolution limit;
* the probe data's admissibility: the coarse probe is rejected where its damping
  is non-positive or its mesh violates ``k*sqrt(A_panel) <= 1``;
* a sensitivity family: how large a resonance just above the band top has to be
  before the certification of the widest-margin DOF would break.

No BEM solve is performed: everything is read from committed artifacts.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorbem.causality import high_band_truncation_budget  # noqa: E402
from tensorbem.kk_tail_certificate import (  # noqa: E402
    certify_tail_budget,
    tail_omission,
)

CASE_DIR = ROOT / "benchmarks" / "verified" / "casepool_bem_cummins_wigley"
OUT_JSON = ROOT / "docs" / "case_bem_cummins_wigley_tail_certificate.json"
OUT_FIG = ROOT / "docs" / "fig_m22460_cummins_tail_certificate.png"
DOF_NAMES = ("surge", "sway", "heave", "roll", "pitch", "yaw")
TOLERANCE = 0.10
BUDGET_TOLERANCE = 0.05
GRAVITY = 9.80665


def _probe_validity(probe) -> dict:
    scale = float(probe["max_panel_scale"])
    c2_top = math.sqrt(GRAVITY / scale)
    worst = {}
    for index, name in enumerate(DOF_NAMES):
        damping = probe["damping"][:, index, index]
        bad = probe["omega"][damping <= 0.0]
        worst[name] = {
            "min_damping": float(np.min(damping)),
            "non_positive_at": [float(v) for v in bad],
        }
    return {
        "panel_scale_m": scale,
        "c2_admissible_top": c2_top,
        "omega_max": float(probe["omega"][-1]),
        "per_dof": worst,
        "usable_as_evidence": all(not row["non_positive_at"] for row in worst.values()),
    }


def _resolved_split(band, probe, index: int, scale: float) -> dict:
    """Fraction of the measured omission carried below the probe's C2 limit."""
    c2_top = math.sqrt(GRAVITY / float(probe["max_panel_scale"]))
    full = tail_omission(
        band["omega"], band["damping"][:, index, index],
        probe["omega"], probe["damping"][:, index, index],
        panel_scale=float(probe["max_panel_scale"]),
    )
    if not c2_top < float(probe["omega"][-1]):
        return {"c2_admissible_top": c2_top, "resolved_fraction": 1.0}
    # enough nodes for the local-slope fit the omission helper performs (only
    # the measured part is used here, the continuation beyond c2_top is discarded)
    nodes = np.logspace(
        math.log10(float(probe["omega"][0])), math.log10(c2_top), 12
    )
    values = np.exp(
        np.interp(np.log(nodes), np.log(probe["omega"]),
                  np.log(probe["damping"][:, index, index]))
    )
    part = tail_omission(band["omega"], band["damping"][:, index, index], nodes, values)
    fraction = float(np.max(part.measured / np.maximum(full.measured, 1e-300)))
    return {
        "c2_admissible_top": c2_top,
        "resolved_omission_relative": float(np.max(part.measured) / scale),
        "resolved_fraction": fraction,
    }


def _resonance(omega, peak_value, *, centre=17.0, width=2.5):
    """Lorentzian resonance whose value at ``centre`` is ``peak_value``."""
    return peak_value * width**4 / ((np.asarray(omega, dtype=float) ** 2 - centre**2) ** 2 + width**4)


def _sensitivity_family(band, probe, index: int) -> dict:
    """How large an un-solved resonance must be before the pass stops being safe.

    Real committed damping is kept everywhere; a synthetic resonance is added to
    the *probe* only, at 17 rad/s (above the 12 rad/s band top), with a peak
    ``r`` times the damping at the band top.  ``r`` is therefore a physical
    ratio: the un-solved peak against the damping that was solved.
    """
    omega = band["omega"]
    damping = band["damping"][:, index, index]
    scale = float(np.abs(band["added_mass"][:, index, index]).max())
    at_top = float(np.interp(12.0, omega, damping))
    rows = []
    for ratio in (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
        probe_damping = probe["damping"][:, index, index] + _resonance(
            probe["omega"], ratio * at_top
        )
        certificate = certify_tail_budget(
            omega, damping, scale,
            probe_omega=probe["omega"],
            probe_damping=probe_damping,
            panel_scale=float(probe["max_panel_scale"]),
        )
        rows.append(
            {
                "peak_over_damping_at_band_top": float(ratio),
                "measured_over_budget": float(certificate.worst_ratio),
                "worst_omission_relative": float(
                    np.nanmax(certificate.omission_lower_relative)
                ),
                "verdict": certificate.verdict,
            }
        )
    first_break = next(
        (
            row["peak_over_damping_at_band_top"]
            for row in rows
            if row["verdict"] == "falsified"
        ),
        None,
    )
    return {
        "damping_at_band_top": at_top,
        "first_falsified_peak_ratio": first_break,
        "rows": rows,
    }


def background_at(omega):
    return 5.0 / (1.0 + np.asarray(omega, dtype=float) ** 2) ** 1.65


def main() -> None:
    band = np.load(CASE_DIR / "sweep_medium.npz", allow_pickle=True)
    probe = np.load(CASE_DIR / "sweep_extended_probe_medium.npz", allow_pickle=True)
    coarse = np.load(CASE_DIR / "sweep_coarse.npz", allow_pickle=True)
    coarse_probe = np.load(
        CASE_DIR / "sweep_extended_probe_coarse.npz", allow_pickle=True
    )
    result = json.loads((CASE_DIR / "case_result.json").read_text())
    c3a = {
        row["dof_name"]: row for row in result["gates"]["C3a"]["values"]["per_dof"]
    }

    payload: dict = {
        "case": "CASE-BEM-CUMMINS-WIGLEY",
        "mission": "m22460",
        "question": (
            "gate C3a reads the added-mass residual as a causality statement on the "
            "samples where the power-law truncation budget is small; is that budget "
            "an upper bound on the tail the band never solved, and how much tail "
            "error can the certification absorb?"
        ),
        "tolerance": TOLERANCE,
        "budget_tolerance": BUDGET_TOLERANCE,
        "committed_band": {
            "mesh": str(band["label"]),
            "panel_count": int(band["panel_count"]),
            "band": [float(band["omega"][0]), float(band["omega"][-1])],
        },
        "probe": _probe_validity(probe),
        "coarse_probe": _probe_validity(coarse_probe),
        "per_dof": {},
    }

    for index, name in enumerate(DOF_NAMES):
        scale = float(np.abs(band["added_mass"][:, index, index]).max())
        certificate = certify_tail_budget(
            band["omega"], band["damping"][:, index, index], scale,
            probe_omega=probe["omega"],
            probe_damping=probe["damping"][:, index, index],
            panel_scale=float(probe["max_panel_scale"]),
            tolerance=TOLERANCE,
            budget_tolerance=BUDGET_TOLERANCE,
        )
        coarse_certificate = certify_tail_budget(
            coarse["omega"], coarse["damping"][:, index, index],
            float(np.abs(coarse["added_mass"][:, index, index]).max()),
            tolerance=TOLERANCE,
            budget_tolerance=BUDGET_TOLERANCE,
        )
        payload["per_dof"][name] = {
            "tail_exponent": certificate.exponent,
            "budget_relative_max": float(
                np.nanmax(
                    np.where(
                        np.isfinite(certificate.budget_relative),
                        certificate.budget_relative,
                        np.nan,
                    )
                )
            )
            if np.any(np.isfinite(certificate.budget_relative))
            else None,
            "n_declared_decidable": certificate.n_declared_decidable,
            "allowance_min": certificate.min_allowance,
            "omission_lower_relative_max": float(
                np.nanmax(certificate.omission_lower_relative)
            ),
            "omission_total_relative_max": float(
                np.nanmax(
                    np.where(
                        np.isfinite(certificate.omission_total_relative),
                        certificate.omission_total_relative,
                        np.nan,
                    )
                )
            )
            if np.any(np.isfinite(certificate.omission_total_relative))
            else None,
            "measured_over_budget_max": certificate.worst_ratio,
            "verdict": certificate.verdict,
            "reason": certificate.reason,
            "c3a_residual": c3a[name]["max_relative_residual"],
            "c3a_n_decidable": c3a[name]["n_decidable"],
            "probe_resolution": _resolved_split(band, probe, index, scale),
            "coarse_mesh": {
                "tail_exponent": coarse_certificate.exponent,
                "n_declared_decidable": coarse_certificate.n_declared_decidable,
                "verdict": coarse_certificate.verdict,
            },
        }

    payload["sensitivity"] = {
        name: _sensitivity_family(band, probe, index)
        for index, name in enumerate(DOF_NAMES)
        if payload["per_dof"][name]["verdict"] == "certified"
    }
    payload["headline"] = {
        "certified_dof": [
            name
            for name, row in payload["per_dof"].items()
            if row["verdict"] == "certified"
        ],
        "uncertified_dof": [
            name
            for name, row in payload["per_dof"].items()
            if row["verdict"] == "uncertified"
        ],
        "smallest_allowance_among_certified": min(
            (
                row["allowance_min"]
                for row in payload["per_dof"].values()
                if row["verdict"] == "certified"
            ),
            default=0.0,
        ),
        "omission_vs_residual": {
            name: {
                "omission_lower_relative": row["omission_lower_relative_max"],
                "c3a_residual": row["c3a_residual"],
                "ratio": row["omission_lower_relative_max"] / row["c3a_residual"],
            }
            for name, row in payload["per_dof"].items()
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    for name, row in payload["per_dof"].items():
        print(
            f"{name:6s} p={row['tail_exponent']:6.2f} n_dec={row['n_declared_decidable']:2d} "
            f"budget_rel={row['budget_relative_max']} kappa={row['allowance_min']:.2f} "
            f"omission={row['omission_lower_relative_max']:.4f} "
            f"ratio={row['measured_over_budget_max']:.3f} c3a={row['c3a_residual']:.4f} "
            f"{row['verdict']}"
        )
    print("headline:", json.dumps(payload["headline"]))


if __name__ == "__main__":
    main()