"""What the *cheap-mesh* extended-band probe says about gates ``C8``/``C9``.

The question
------------
Gate ``C3a`` (Kramers-Kronig closure with an independently anchored
``A_inf``) can only score a sample whose truncation budget -- the omitted band
``s > s_max`` -- is smaller than half the case tolerance.  On the committed
0.2 .. 12 rad/s sweep that budget is unbounded for surge, sway and yaw, because
their measured damping tail exponent at the band top is not above 2.05.  The
declared remedy was a wider band (0.2 .. 20 rad/s) on a mesh that still satisfies
gate ``C2``, which costs 3840 panels x 55 nodes of *dense* BIE solves.

Why a probe
-----------
Two things about that plan were asserted and never measured: that the extension
actually moves the horizontal modes into algebraic decay, and that it is
affordable.  Both are measured here on the meshes that already exist.  This
module is explicitly **not** a gate: the probe meshes violate ``C2`` at the top
of the extended band, so their high-frequency coefficients carry an error the
committed band did not have.  The error is not assumed away -- the probe runs on
*two* meshes and the exponent is reported as a bracket.  A conclusion is called
``robust`` only when both meshes agree it, ``inconclusive`` when the bracket
straddles the 2.05 bar, and the certified (``C2``-compliant) sweep stays
``not_run`` regardless: a probe cannot promote itself to a gate.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from tensorbem.causality import high_band_truncation_budget, local_tail_exponent

from .case_bem_cummins_wigley import DOF_NAMES, GRAVITY, Sweep

__all__ = ["probe_report", "render_markdown", "PROBE_NOT_A_GATE", "SLOPE_BAR"]

#: ``C3a`` admits a sample only when the measured slope ``p`` exceeds twice the
#: algebraic-decay threshold (see :func:`tensorbem.causality.high_band_truncation_budget`).
SLOPE_BAR = 2.05

PROBE_NOT_A_GATE = (
    "probe: the probe meshes exceed gate C2 (k*sqrt(A_panel) > 1) above the "
    "frequency where the certified mesh starts to matter, so these numbers "
    "bound the physics but do not certify it"
)


def _slope(omega: np.ndarray, damping: np.ndarray, dof: int) -> tuple[float, float]:
    """Local tail exponent of ``B_ii`` and the truncation-budget admissibility.

    The exponent is measured on the *truncated* grid so the same estimator sees
    the same band on both meshes; the budget is the one gate ``C3a`` uses.
    """
    column = np.asarray(damping, dtype=float)[:, dof, dof]
    exponent, _ = local_tail_exponent(np.asarray(omega, dtype=float), column)
    return float(exponent), column


def _budget(omega: np.ndarray, damping: np.ndarray, dof: int, scale: float) -> dict[str, Any]:
    column = np.asarray(damping, dtype=float)[:, dof, dof]
    budget = high_band_truncation_budget(np.asarray(omega, dtype=float), column)
    relative = budget.relative(max(float(scale), 1e-300))
    finite = relative[np.isfinite(relative)]
    return {
        "admissible": bool(budget.admissible),
        "reason": budget.reason,
        "max_relative": float(np.max(finite)) if finite.size else float("inf"),
    }


def _exponent_at(sweep: Sweep, omega_top: float, dof: int) -> float:
    """Tail exponent of one DOF on the part of the band ending at ``omega_top``."""
    mask = np.asarray(sweep.omega, dtype=float) <= float(omega_top) * (1.0 + 1e-9)
    exponent, _ = local_tail_exponent(
        np.asarray(sweep.omega, dtype=float)[mask],
        np.asarray(sweep.damping, dtype=float)[mask][:, dof, dof],
    )
    return float(exponent)


def _band_limit(panel_scale: float) -> float:
    """Highest frequency at which gate ``C2`` still holds for this panel scale."""
    return float(math.sqrt(GRAVITY / max(float(panel_scale), 1e-300)))


def _shared_band_difference(committed: Sweep, probe: Sweep) -> dict[str, Any]:
    """Does the probe's tail segment continue the committed datum at the join?

    The probe starts exactly at the committed band top, so the two sweeps share
    one frequency.  On the *same* mesh that shared node must be the same number:
    a difference there means the probe's grid or mesh is not the committed one
    and the tail slope is not a continuation of the committed datum.
    """
    grid = np.asarray(committed.omega, dtype=float)
    probe_omega = np.asarray(probe.omega, dtype=float)
    join = float(probe_omega[0])
    worst = 0.0
    worst_dof = None
    same_mesh = (int(committed.n_sections), int(committed.n_levels)) == (
        int(probe.n_sections), int(probe.n_levels))
    for index, name in enumerate(DOF_NAMES):
        probe_b = np.interp(join, probe_omega,
                            np.asarray(probe.damping, dtype=float)[:, index, index])
        ref_b = np.interp(join, grid,
                          np.asarray(committed.damping, dtype=float)[:, index, index])
        scale = max(float(np.abs(ref_b)), 1e-300)
        rel = float(abs(probe_b - ref_b) / scale)
        if rel > worst:
            worst, worst_dof = rel, name
    return {
        "n_shared_frequencies": 1,
        "join_frequency_rad_s": join,
        "same_mesh_as_committed": bool(same_mesh),
        "worst_dof": worst_dof,
        "worst_relative_difference_of_B": worst,
        "note": (
            "the probe shares only the join frequency with the committed sweep; "
            "this checks continuation, not the accuracy of the tail"
        ),
    }


def probe_report(
    *,
    committed: Sweep,
    coarse: Sweep,
    medium: Sweep,
    cost: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Judge the extended-band probe: does the widening reach algebraic decay?"""
    meshes = {"coarse": coarse, "medium": medium}
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(DOF_NAMES):
        # the band top of the *committed* datum is the "before" value; the probe
        # only owns the tail above it, so p(12) is not a probe number
        row: dict[str, Any] = {
            "dof": int(index),
            "dof_name": name,
            "exponent_at_12": _exponent_at(committed, 12.0, index),
        }
        for label, sweep in meshes.items():
            scale = max(float(np.max(np.abs(np.asarray(sweep.added_mass)[:, index, index]))), 1e-300)
            exponent_top, column = _slope(sweep.omega, sweep.damping, index)
            row[f"exponent_at_20_{label}"] = exponent_top
            row[f"budget_{label}"] = _budget(sweep.omega, sweep.damping, index, scale)
            row[f"B_top_{label}"] = float(column[-1])
            row[f"panel_scale_{label}"] = float(sweep.max_panel_scale)
            row[f"band_limit_{label}"] = _band_limit(sweep.max_panel_scale)
        exponents = [row[f"exponent_at_20_{label}"] for label in meshes]
        admissible = [row[f"budget_{label}"]["admissible"] for label in meshes]
        if min(exponents) > SLOPE_BAR and all(admissible):
            verdict = "robust: both meshes put the tail above the algebraic-decay bar"
        elif max(exponents) > SLOPE_BAR:
            verdict = (
                "inconclusive: the two meshes straddle the bar, so the mesh error "
                "is as large as the margin"
            )
        else:
            verdict = "not widened: no mesh reaches algebraic decay by 20 rad/s"
        row["probe_verdict"] = verdict
        rows.append(row)

    robust = [row["dof_name"] for row in rows if row["probe_verdict"].startswith("robust")]
    inconclusive = [row["dof_name"] for row in rows if row["probe_verdict"].startswith("inconclusive")]
    manifest = {
        "band": [float(medium.omega[0]), float(medium.omega[-1])],
        "n_omega": int(medium.omega.size),
        "slope_bar": SLOPE_BAR,
        "meshes": {
            label: {
                "n_sections": int(sweep.n_sections),
                "n_levels": int(sweep.n_levels),
                "panel_count": int(sweep.panel_count),
                "max_panel_scale": float(sweep.max_panel_scale),
                "band_limit_rad_s": _band_limit(sweep.max_panel_scale),
                "c2_holds_over_probe_band": bool(_band_limit(sweep.max_panel_scale) >= sweep.omega[-1]),
            }
            for label, sweep in meshes.items()
        },
        "shared_band_reproduction": _shared_band_difference(committed, medium),
    }
    return {
        "probe": PROBE_NOT_A_GATE,
        "certified_gates": {"C8": "not_run", "C9": "not_run"},
        "manifest": manifest,
        "per_dof": rows,
        "robust_dof": robust,
        "inconclusive_dof": inconclusive,
        "cost": cost,
    }


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "inf"
    return f"{value:.{digits}f}"


def render_markdown(result: dict[str, Any]) -> str:
    """Render the probe document from ``result`` (no hand-copied numbers)."""
    manifest = result["manifest"]
    lines = ["# Extended-band probe: CASE-BEM-CUMMINS-WIGLEY", ""]
    lines.append(
        "Auto-rendered by `tensorbem/benchmarks/cummins_wigley_extended_probe.py` "
        "from `extended_probe_report.json`; every number below is programmatic."
    )
    lines.append("")
    lines.append("## 1. What was asked")
    lines.append("")
    lines.append(
        "Gate `C3a` scores a Kramers-Kronig sample only when the omitted band "
        "`s > s_max` cannot produce the residual on its own, i.e. when the measured "
        "damping slope at the band top exceeds "
        f"`p > {manifest['slope_bar']}`. On the committed 12 rad/s band surge, sway "
        "and yaw fail that test, so three of six DOF are uncertified. The declared "
        "remedy is to widen the band to 20 rad/s on a `C2`-compliant mesh."
    )
    lines.append("")
    lines.append("## 2. Why this document is a probe and not a gate")
    lines.append("")
    lines.append("| mesh | sections x levels | panels | max panel scale [m] | C2 band limit [rad/s] | C2 over 0.2..20 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for label, info in manifest["meshes"].items():
        lines.append(
            f"| {label} | {info['n_sections']} x {info['n_levels']} | "
            f"{info['panel_count']} | {info['max_panel_scale']:.4f} | "
            f"{info['band_limit_rad_s']:.2f} | "
            f"{'yes' if info['c2_holds_over_probe_band'] else 'NO'} |"
        )
    lines.append("")
    lines.append(result["probe"] + ".")
    lines.append("")
    share = manifest["shared_band_reproduction"]
    lines.append(
        f"Shared-band reproduction of the committed sweep on "
        f"{share['n_shared_frequencies']} overlapping frequencies: worst relative "
        f"difference of `B` = {_fmt(share['worst_relative_difference_of_B'], 6)} "
        f"({share['worst_dof']})."
    )
    lines.append("")
    lines.append("## 3. Does the widening reach algebraic decay?")
    lines.append("")
    lines.append("| DOF | p at 12 (committed) | p at 20 (coarse/medium) | budget admissible at 20 (coarse/medium) | verdict |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in result["per_dof"]:
        lines.append(
            f"| {row['dof_name']} | {_fmt(row['exponent_at_12'], 2)} | "
            f"{_fmt(row['exponent_at_20_coarse'], 2)} / {_fmt(row['exponent_at_20_medium'], 2)} | "
            f"{'yes' if row['budget_coarse']['admissible'] else 'no'} / "
            f"{'yes' if row['budget_medium']['admissible'] else 'no'} | "
            f"{row['probe_verdict']} |"
        )
    lines.append("")
    lines.append(
        "Robust conclusions: "
        + (", ".join(result["robust_dof"]) if result["robust_dof"] else "none")
        + ". Mesh-bracketed (inconclusive): "
        + (", ".join(result["inconclusive_dof"]) if result["inconclusive_dof"] else "none")
        + "."
    )
    lines.append("")
    cost = result.get("cost")
    if cost:
        lines.append("## 4. What the certified sweep would cost (measured, then extrapolated)")
        lines.append("")
        lines.append("| sections x levels | panels | wall time of one node [s] |")
        lines.append("| --- | --- | --- |")
        for point in cost["points"]:
            lines.append(
                f"| {point['n_sections']} x {point['n_levels']} | {point['panel_count']} | "
                f"{point['wall_s']:.2f} |"
            )
        lines.append("")
        lines.append(
            f"Fitted cost law `t ~ N^{cost['fitted_exponent']:.2f}`; at the certified "
            f"mesh ({cost['certified_panel_count']} panels) that is "
            f"{cost['certified_wall_s_per_node'] / 60.0:.1f} min per node, i.e. "
            f"**{cost['certified_cpu_hours_for_band']:.0f} CPU-hours** for the "
            f"{manifest['n_omega']}-node band, against the "
            f"{cost['declared_plan_cpu_hours'][0]:.0f}-"
            f"{cost['declared_plan_cpu_hours'][1]:.0f} h the plan assumed."
        )
        lines.append("")
    lines.append("## 5. Disposition")
    lines.append("")
    lines.append(
        "Gates `C8`/`C9` remain **not_run**: the probe bounds the physics but its "
        "meshes violate `C2`, and a probe cannot promote itself to a gate. The "
        "declared certified sweep stays executable "
        "(`scripts/case_cummins_wigley_extended_band.py`) and stays out of the "
        "verdict until it is actually run."
    )
    lines.append("")
    return "\n".join(lines)