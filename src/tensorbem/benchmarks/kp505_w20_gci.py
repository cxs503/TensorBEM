"""W20 item 5: three-density GCI study of the KP505 open-water lattice.

What the item asks for
----------------------
Research plan item 5 of topic ``propeller-openwater-bem`` ("exp-close-gap")
carries the historical statement *"the lattice is 29.1 % low on ``K_T``"*
(report 8822, ``J = 0.8``).  That statement mixes two things: a
discretisation error of the panel lattice and a model error of the
lifting-surface + ITTC-78-drag formulation.  This driver separates them at the
mission advance ratios ``J = 0.5 / 0.7 / 0.9`` (``J = 0.8`` is carried as an
extra diagnostic so the historical 29.1 % is decomposed with the same
machinery) using one uniform three-density ladder and Roache's GCI.

Predeclared design (fixed before any level was run)
---------------------------------------------------
Panel-density ladder, refinement ratio ``r = 2`` on **all three** panel axes at
once (a genuine "panel density", not a single-axis sweep)::

    level   n_radial   n_chord   wake_steps   panels/blade
    L1          8         2          24            16
    L2         16         4          48            64
    L3         32         8          96           256

Two vortex-core policies, because a lone 3-level study cannot tell a
discretisation trend from a regularisation trend:

``fixed`` (PRIMARY)
    ``core_ratio = 0.05`` at every level, i.e. the base configuration of the
    existing sweep study.  The regularisation is held constant, so the ladder
    measures the discretisation alone -- which is what the GCI procedure
    requires of the "same model at three grids".
``similarity`` (SECONDARY)
    ``core_ratio = 0.05 * 16 / n_radial`` (0.10 / 0.05 / 0.025), i.e. the
    desingularisation radius halves with the panel size (geometrically similar
    lattice).  This is the classical panel-method choice, but it changes the
    model between levels, so it is reported as a diagnostic only.

Theory order
------------
``p_th = 2.0`` is the primary design order: the lattice's forces are midpoint
quadrature of ``Gamma W x dl`` over the panels, and the composite midpoint rule
is second order for a smooth loading.  The blade-tip loading however behaves
like ``sqrt(1 - r)``, which degrades the achievable order towards
``p_th = 1.5``; that value is carried as a declared secondary and both
deviations are tabulated.  Neither number depends on any measured value.

Criteria (each reported PASS/FAIL in the artefact)
--------------------------------------------------
W1  coverage: 3 mission advance ratios x 3 densities x 2 quantities, all finite.
W2  physical admissibility at every level of every cell: ``K_T > 0``,
    ``K_Q > 0`` (resisting torque).
W3  observed order within 30 % of ``p_th = 2.0`` for ``K_T`` and ``10 K_Q`` at
    every mission advance ratio (primary core policy).
W4  fine-grid GCI tabulated for both quantities at every advance ratio, with the
    explicit answer to "GCI <= 3 %" and "GCI <= 1 %".
W5  the decomposition closes: ``|d_disc + d_model - d_total| <= 1e-9`` pct
    points in every cell of both quantities.
W6  the design point J = 0.70 reports ``|dK_T|`` and ``|dK_Q|`` and answers the
    3 % question explicitly (threshold never moved: 3 %).
W7  the two core policies agree on the W3 verdict (robustness of the order
    estimate to the regularisation policy).

Run with ``python -m tensorbem.benchmarks.kp505_w20_gci [outdir]``; the driver
caches every lattice evaluation under ``runs/w20_kp505_gci_cache`` so an
interrupted run resumes instead of restarting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..propeller import BladeGeometry, open_water_point
from ..uq import DEFAULT_SAFETY_FACTOR, grid_convergence_index
from .kp505_openwater_bem import ANCHOR, PRIMARY

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTDIR = REPO_ROOT / "benchmarks" / "verified" / "bem_kp505_openwater_w20"
CACHE_DIR = REPO_ROOT / "runs" / "w20_kp505_gci_cache"
ARTEFACT_NAME = "gci.json"
SCHEMA = "tensorbem.w20_kp505_bem_gci/1"

#: Mission advance ratios (fixed by the contract) plus the historical tie point.
MISSION_J = (0.5, 0.7, 0.9)
HISTORY_J = 0.8
ALL_J = MISSION_J + (HISTORY_J,)
DESIGN_J = 0.7

#: Panel-density ladder: every axis refined by r = 2 simultaneously.
PANEL_LEVELS = (
    {"level": 1, "label": "L1-coarse", "n_radial": 8, "n_chord": 2, "wake_steps": 24},
    {"level": 2, "label": "L2-medium", "n_radial": 16, "n_chord": 4, "wake_steps": 48},
    {"level": 3, "label": "L3-fine", "n_radial": 32, "n_chord": 8, "wake_steps": 96},
)
REFINEMENT_RATIO = 2.0
CORE_BASE = 0.05
CORE_POLICIES = {
    "fixed": "core_ratio = 0.05 at every level (regularisation held constant)",
    "similarity": "core_ratio = 0.05 * 16 / n_radial (radius halves with the panel size)",
}
PRIMARY_CORE_POLICY = "fixed"

#: Everything else is the study's base configuration, held constant.
HELD_FIXED = {
    "wake_model": PRIMARY["wake_model"],
    "wake_length": PRIMARY["wake_length"],
    "rev_per_second": PRIMARY["rev_per_second"],
    "thickness_ratio": PRIMARY["thickness_ratio"],
    "viscous": PRIMARY["viscous"],
    "radial_spacing": PRIMARY["radial_spacing"],
    "root_chord_offset": PRIMARY["root_chord_offset"],
}

#: Orders and thresholds, fixed before running.
THEORY_ORDER = 2.0
THEORY_ORDER_SECONDARY = 1.5
ORDER_TOLERANCE = 0.30
GCI_ACCEPT_PCT = 3.0
GCI_STRONG_PCT = 1.0
DECOMPOSITION_TOL = 1e-9
DESIGN_TOL_PCT = 3.0
REPORT_8822_DKT_PCT = -29.13

#: Filled by :func:`assemble` before the criteria are evaluated.
PROBE: dict = {}
QUANTITIES = ("KT", "10KQ", "eta0")


# ---------------------------------------------------------------------------
# lattice evaluation (cached)
# ---------------------------------------------------------------------------
def core_ratio_for(policy: str, level: dict) -> float:
    if policy == "fixed":
        return CORE_BASE
    if policy == "similarity":
        return CORE_BASE * 16.0 / float(level["n_radial"])
    raise ValueError(f"unknown core policy {policy!r}")


def _cache_path(advance: float, policy: str, level: int) -> Path:
    return CACHE_DIR / f"cell_J{advance:.2f}_{policy}_L{level}.json"


def evaluate(advance: float, policy: str, level: dict) -> dict:
    """One lattice evaluation.  Cached, because the fine level is ~1 min."""

    path = _cache_path(advance, policy, level["level"])
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))

    geometry = BladeGeometry(
        n_radial=int(level["n_radial"]),
        n_chord=int(level["n_chord"]),
        thickness_ratio=HELD_FIXED["thickness_ratio"],
        radial_spacing=HELD_FIXED["radial_spacing"],
        root_chord_offset=HELD_FIXED["root_chord_offset"],
    )
    result = open_water_point(
        geometry,
        advance,
        wake_model=HELD_FIXED["wake_model"],
        wake_length=HELD_FIXED["wake_length"],
        wake_steps=int(level["wake_steps"]),
        rev_per_second=HELD_FIXED["rev_per_second"],
        viscous=HELD_FIXED["viscous"],
        core_ratio=core_ratio_for(policy, level),
    )
    # `open_water_point` returns K_Q with the sign of the driving torque; the
    # resisting (physical) torque and hence 10 K_Q are positive for an
    # absorbing propeller, exactly as in the existing ladder study.
    torque = -result.torque
    ten_kq = 10.0 * torque
    efficiency = (
        advance * result.thrust / (2.0 * math.pi * torque)
        if torque != 0.0
        else float("nan")
    )
    cell = {
        "J": advance,
        "core_policy": policy,
        "level": level["level"],
        "n_radial": level["n_radial"],
        "n_chord": level["n_chord"],
        "wake_steps": level["wake_steps"],
        "core_ratio": core_ratio_for(policy, level),
        "KT": float(result.thrust),
        "10KQ": float(ten_kq),
        "eta0": float(efficiency),
        "n_panels_per_blade": int(level["n_radial"] * level["n_chord"]),
        "n_panels_reported": int(result.n_panels),
        "viscous_drag_nd": float(result.viscous_drag),
        "momentum_bound": float(result.momentum_bound),
        "physical": bool(
            result.thrust > 0.0
            and ten_kq > 0.0
            and math.isfinite(efficiency)
            and 0.0 < efficiency <= result.momentum_bound
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cell, indent=2) + "\n", encoding="utf-8")
    return cell


def core_probe(advance: float, level: dict, core_ratio: float) -> dict:
    """One lattice evaluation at an explicit core ratio (inertness probe)."""

    key = f"probe_J{advance:.2f}_nr{level['n_radial']}_nc{level['n_chord']}_ws{level['wake_steps']}_cr{core_ratio:.3f}"
    path = CACHE_DIR / f"{key}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    geometry = BladeGeometry(
        n_radial=int(level["n_radial"]),
        n_chord=int(level["n_chord"]),
        thickness_ratio=HELD_FIXED["thickness_ratio"],
        radial_spacing=HELD_FIXED["radial_spacing"],
        root_chord_offset=HELD_FIXED["root_chord_offset"],
    )
    result = open_water_point(
        geometry,
        advance,
        wake_model=HELD_FIXED["wake_model"],
        wake_length=HELD_FIXED["wake_length"],
        wake_steps=int(level["wake_steps"]),
        rev_per_second=HELD_FIXED["rev_per_second"],
        viscous=HELD_FIXED["viscous"],
        core_ratio=core_ratio,
    )
    record = {
        "J": advance,
        "n_radial": level["n_radial"],
        "n_chord": level["n_chord"],
        "wake_steps": level["wake_steps"],
        "core_ratio": core_ratio,
        "KT": float(result.thrust),
        "10KQ": float(-10.0 * result.torque),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


#: The configuration that produced report 8822's ``dK_T(J = 0.8) = -29.13 %``.
HISTORY_CONFIG = {
    "n_radial": 16,
    "n_chord": 2,
    "wake_steps": 24,
    "core_ratio": 0.05,
    "level": 0,
}


def history_tie() -> dict:
    """Reproduce the historical -29.13 % before decomposing it.

    Report 8822's number was produced at the base lattice of the existing sweep
    study (16 radial x 2 chord panels, 24 wake steps, core ratio 0.05), not at
    the fine grid of this ladder.  Reproducing it at that exact configuration is
    what licenses the sentence "the deviation this item decomposes is the
    historical deviation".
    """

    row = core_probe(HISTORY_J, HISTORY_CONFIG, HISTORY_CONFIG["core_ratio"])
    anchor = anchor_values(HISTORY_J)
    dkt = 100.0 * (row["KT"] - anchor["KT"]) / anchor["KT"]
    dkq = 100.0 * (row["10KQ"] - anchor["10KQ"]) / anchor["10KQ"]
    return {
        "J": HISTORY_J,
        "configuration": {
            key: value for key, value in HISTORY_CONFIG.items() if key != "level"
        },
        "KT": row["KT"],
        "10KQ": row["10KQ"],
        "anchor_KT": anchor["KT"],
        "anchor_10KQ": anchor["10KQ"],
        "dKT_pct": dkt,
        "dKQ_pct": dkq,
        "stated_report_8822_dKT_pct": REPORT_8822_DKT_PCT,
        "reproduction_residual_pct_points": abs(dkt - REPORT_8822_DKT_PCT),
        "reproduced_within_0p1_pct_point": abs(dkt - REPORT_8822_DKT_PCT) <= 0.1,
    }


def inertness_probe(advance: float = DESIGN_J) -> dict:
    """Is the declared core policy able to move the answer at these densities?

    The two predeclared policies use ``core_ratio`` = 0.10 / 0.05 / 0.025 across
    the ladder.  If those values all return the same ``K_T`` to the last bit,
    the policy cannot discriminate and the robustness criterion attached to it
    is not testable; the probe then walks the ratio up to 0.8 to show at which
    value the parameter does bite, so the finding is bounded rather than vague.
    """

    sweeps = {}
    for label, level in (("L1-coarse", PANEL_LEVELS[0]), ("L3-fine", PANEL_LEVELS[2])):
        ratios = (
            (0.025, 0.05, 0.10, 0.40)
            if label == "L3-fine"
            else (0.025, 0.05, 0.10, 0.40, 0.80)
        )
        rows = [core_probe(advance, level, ratio) for ratio in ratios]
        values = [row["KT"] for row in rows]
        sweeps[label] = {
            "n_radial": level["n_radial"],
            "n_chord": level["n_chord"],
            "wake_steps": level["wake_steps"],
            "rows": rows,
            "KT_values": values,
            "bit_identical_over_policy_range": len({f"{v:.15g}" for v in values[:3]})
            == 1,
            "max_rel_change_pct": 100.0 * (max(values) - min(values)) / abs(values[0]),
            "first_ratio_that_moves_KT": next(
                (
                    row["core_ratio"]
                    for row in rows
                    if f"{row['KT']:.15g}" != f"{values[0]:.15g}"
                ),
                None,
            ),
        }
    within = all(sweep["bit_identical_over_policy_range"] for sweep in sweeps.values())
    return {
        "J": advance,
        "purpose": (
            "test whether the predeclared core policies (0.025/0.05/0.10) can change "
            "K_T at the densities of this study"
        ),
        "sweeps": sweeps,
        "policy_range_is_inert": within,
        "mechanism": (
            "src/tensorbem/propeller.py solve_lifting_surface clamps the segment core to "
            "max(core_ratio*|segment|, 0.5*strip_width); at these panel densities the "
            "0.5*strip_width floor dominates for core_ratio <= 0.10, so the desingularisation "
            "radius does not follow core_ratio there"
        ),
    }


# ---------------------------------------------------------------------------
# anchor helpers
# ---------------------------------------------------------------------------
def anchor_values(advance: float) -> dict:
    values = ANCHOR[advance]
    return {"KT": float(values["KT"]), "10KQ": float(values["10KQ"])}


def anchor_eta0(advance: float) -> float:
    values = anchor_values(advance)
    return advance * values["KT"] / (2.0 * math.pi * values["10KQ"] / 10.0)


# ---------------------------------------------------------------------------
# GCI + decomposition
# ---------------------------------------------------------------------------
def gci_block(
    fine: float, medium: float, coarse: float, quantity: str, label: str
) -> dict:
    result = grid_convergence_index(
        fine=fine,
        medium=medium,
        coarse=coarse,
        refinement_ratio=REFINEMENT_RATIO,
        safety_factor=DEFAULT_SAFETY_FACTOR,
        quantity=quantity,
        label=label,
    )
    block = result.as_dict()
    block["usable"] = bool(
        math.isfinite(block.get("observed_order", float("nan")))
        and math.isfinite(block.get("gci_fine", float("nan")))
    )
    return block


def decompose(fine: float, extrapolated: float, anchor: float) -> dict:
    """Split the anchor deviation into a discretisation and a model term.

    All three terms are relative to the *anchor*, so they add exactly::

        d_total = 100 (f_fine - f_anchor) / f_anchor
        d_disc  = 100 (f_fine - f_ext)    / f_anchor
        d_model = 100 (f_ext  - f_anchor) / f_anchor
    """

    d_total = 100.0 * (fine - anchor) / anchor
    d_disc = 100.0 * (fine - extrapolated) / anchor
    d_model = 100.0 * (extrapolated - anchor) / anchor
    return {
        "d_total_pct": d_total,
        "d_discretisation_pct": d_disc,
        "d_model_pct": d_model,
        "closure_residual_pct_points": abs(d_disc + d_model - d_total),
        "f_fine": fine,
        "f_extrapolated": extrapolated,
        "f_anchor": anchor,
    }


def cell_study(advance: float, policy: str) -> dict:
    levels = [evaluate(advance, policy, level) for level in PANEL_LEVELS]
    anchor = anchor_values(advance)
    quantities: dict[str, dict] = {}

    for quantity in QUANTITIES:
        values = [level[quantity] for level in levels]
        fine, medium, coarse = values[2], values[1], values[0]
        block = gci_block(
            fine, medium, coarse, quantity, f"J={advance:g} {policy} {quantity}"
        )
        ref = anchor_eta0(advance) if quantity == "eta0" else anchor[quantity]
        decomposition = decompose(fine, block.get("extrapolated", fine), ref)
        order = block.get("observed_order", float("nan"))
        order_dev = (
            abs(order - THEORY_ORDER) / THEORY_ORDER * 100.0
            if math.isfinite(order)
            else float("nan")
        )
        order_dev_secondary = (
            abs(order - THEORY_ORDER_SECONDARY) / THEORY_ORDER_SECONDARY * 100.0
            if math.isfinite(order)
            else float("nan")
        )
        # ``uq.grid_convergence_index`` reports the GCI as a *fraction*; the
        # artefact reports per cent.
        raw_gci = block.get("gci_fine", float("nan"))
        gci_fine = (
            100.0 * raw_gci if isinstance(raw_gci, (int, float)) else float("nan")
        )
        # Robust attribution that does not need a valid observed order: the last
        # refinement increment is a lower-bound-flavoured scale for whatever
        # discretisation error is still in the fine grid.  Taking it at face
        # value twice gives the band within which the model term is unknown.
        last_increment = 100.0 * abs(fine - medium) / abs(ref) if ref else float("nan")
        model_bounds = (
            [
                decomposition["d_total_pct"] - 2.0 * last_increment,
                decomposition["d_total_pct"] + 2.0 * last_increment,
            ]
            if math.isfinite(last_increment)
            else [float("nan"), float("nan")]
        )
        quantities[quantity] = {
            "levels": values,
            "anchor": ref,
            "gci": block,
            "observed_order": order,
            "order_deviation_from_theory_pct": order_dev,
            "order_deviation_from_secondary_pct": order_dev_secondary,
            "order_within_tolerance": bool(
                math.isfinite(order_dev) and order_dev < ORDER_TOLERANCE * 100.0
            ),
            "gci_fine_pct": gci_fine,
            "gci_band_in_anchor_pct": (
                gci_fine * fine / ref
                if math.isfinite(gci_fine) and ref
                else float("nan")
            ),
            "gci_within_3pct": bool(
                math.isfinite(gci_fine) and gci_fine < GCI_ACCEPT_PCT
            ),
            "gci_within_1pct": bool(
                math.isfinite(gci_fine) and gci_fine < GCI_STRONG_PCT
            ),
            "decomposition": decomposition,
            "delta_vs_anchor_pct": decomposition["d_total_pct"],
            "abs_delta_vs_anchor_pct": abs(decomposition["d_total_pct"]),
            "last_increment_pct_of_anchor": last_increment,
            "model_bounds_anchor_pct": model_bounds,
            "model_term_outside_band": bool(
                math.isfinite(last_increment)
                and abs(decomposition["d_total_pct"]) > 2.0 * last_increment
            ),
        }

    return {
        "J": advance,
        "core_policy": policy,
        "advance_role": "mission" if advance in MISSION_J else "history-tie",
        "anchor": anchor,
        "levels": levels,
        "quantities": quantities,
        "physical_all_levels": all(level["physical"] for level in levels),
    }


# ---------------------------------------------------------------------------
# artefacts
# ---------------------------------------------------------------------------
def write_csvs(outdir: Path, cells: list[dict]) -> list[str]:
    written = []
    outdir.mkdir(parents=True, exist_ok=True)

    levels_path = outdir / "levels.csv"
    with levels_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "J",
                "core_policy",
                "level",
                "n_radial",
                "n_chord",
                "wake_steps",
                "core_ratio",
                "panels_per_blade",
                "KT",
                "10KQ",
                "eta0",
                "physical",
            ]
        )
        for cell in cells:
            for level in cell["levels"]:
                writer.writerow(
                    [
                        f"{cell['J']:g}",
                        cell["core_policy"],
                        level["level"],
                        level["n_radial"],
                        level["n_chord"],
                        level["wake_steps"],
                        level["core_ratio"],
                        level["n_panels_per_blade"],
                        f"{level['KT']:.10g}",
                        f"{level['10KQ']:.10g}",
                        f"{level['eta0']:.10g}",
                        level["physical"],
                    ]
                )
    written.append(str(levels_path.relative_to(REPO_ROOT)))

    gci_path = outdir / "gci_table.csv"
    with gci_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "J",
                "core_policy",
                "quantity",
                "observed_order",
                "order_dev_vs_p2_pct",
                "order_dev_vs_p1.5_pct",
                "extrapolated",
                "gci_fine_pct",
                "gci_band_in_anchor_pct",
                "d_total_pct",
                "d_discretisation_pct",
                "d_model_pct",
                "closure_residual_pct_points",
            ]
        )
        for cell in cells:
            for quantity in QUANTITIES:
                block = cell["quantities"][quantity]
                writer.writerow(
                    [
                        f"{cell['J']:g}",
                        cell["core_policy"],
                        quantity,
                        f"{block['observed_order']:.6g}",
                        f"{block['order_deviation_from_theory_pct']:.6g}",
                        f"{block['order_deviation_from_secondary_pct']:.6g}",
                        f"{block['gci'].get('extrapolated', float('nan')):.10g}",
                        f"{block['gci_fine_pct']:.6g}",
                        f"{block['gci_band_in_anchor_pct']:.6g}",
                        f"{block['decomposition']['d_total_pct']:.6g}",
                        f"{block['decomposition']['d_discretisation_pct']:.6g}",
                        f"{block['decomposition']['d_model_pct']:.6g}",
                        f"{block['decomposition']['closure_residual_pct_points']:.3g}",
                    ]
                )
    written.append(str(gci_path.relative_to(REPO_ROOT)))
    return written


def make_figures(outdir: Path, cells: list[dict]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional dependency
        return []

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    figures = []
    primary = [cell for cell in cells if cell["core_policy"] == PRIMARY_CORE_POLICY]

    # -- 1. K_T, 10 K_Q and eta0 against the anchor -------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    Js = sorted({cell["J"] for cell in primary})
    anchor_J = sorted(ANCHOR)
    for axis, quantity, ylabel in zip(
        axes, ("KT", "10KQ", "eta0"), ("$K_T$", "$10K_Q$", "$\\eta_0$")
    ):
        axis.plot(
            anchor_J,
            [
                anchor_values(j)[quantity] if quantity != "eta0" else anchor_eta0(j)
                for j in anchor_J
            ],
            "k-o",
            ms=4,
            lw=1.2,
            label="anchor (transcribed EFD)",
        )
        fine = [
            next(c for c in primary if c["J"] == j)["quantities"][quantity]["levels"][2]
            for j in Js
        ]
        axis.plot(Js, fine, "s--", color="#c0392b", ms=5, label="BEM fine grid (L3)")
        for j, value in zip(Js, fine):
            axis.annotate(
                f"J={j:g}",
                xy=(j, value),
                xytext=(4, -12),
                textcoords="offset points",
                fontsize=7,
            )
        axis.set_xlabel("$J$")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle(
        "KP505 open water: BEM lifting surface (L3 panel density) vs transcribed EFD anchor"
    )
    fig.tight_layout()
    path = figdir / "w20_kt_kq_eta_vs_anchor.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(str(path.relative_to(REPO_ROOT)))

    # -- 2. order and GCI per advance ratio ---------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0))
    width = 0.35
    xs = list(range(len(primary)))
    for offset, quantity, color in (
        (-width / 2, "KT", "#2980b9"),
        (width / 2, "10KQ", "#e67e22"),
    ):
        orders = [cell["quantities"][quantity]["observed_order"] for cell in primary]
        gcis = [cell["quantities"][quantity]["gci_fine_pct"] for cell in primary]
        ax1.bar([x + offset for x in xs], orders, width, label=quantity, color=color)
        ax2.bar([x + offset for x in xs], gcis, width, label=quantity, color=color)
    ax1.axhline(
        THEORY_ORDER,
        color="k",
        ls="--",
        lw=1.2,
        label=f"theory order $p_{{th}}$={THEORY_ORDER:g}",
    )
    ax1.axhspan(
        THEORY_ORDER * (1 - ORDER_TOLERANCE),
        THEORY_ORDER * (1 + ORDER_TOLERANCE),
        color="grey",
        alpha=0.2,
        label="±30 % band",
    )
    ax2.axhline(GCI_ACCEPT_PCT, color="r", ls="--", lw=1.2, label="3 % acceptance")
    ax2.axhline(GCI_STRONG_PCT, color="k", ls=":", lw=1.0, label="1 %")
    for axis, title, ylabel in (
        (ax1, "observed order of the fine triplet", "observed $p$"),
        (
            ax2,
            "fine-grid GCI (safety factor %.1f)" % DEFAULT_SAFETY_FACTOR,
            "GCI$_{fine}$ / %",
        ),
    ):
        axis.set_xticks(xs)
        axis.set_xticklabels([f"J={cell['J']:g}" for cell in primary])
        axis.set_title(title, fontsize=10)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3, axis="y")
        axis.legend(fontsize=7)
    fig.tight_layout()
    path = figdir / "w20_order_and_gci.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(str(path.relative_to(REPO_ROOT)))

    # -- 3. decomposition stacked bars --------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    for axis, quantity in zip(axes, ("KT", "10KQ")):
        totals = [
            cell["quantities"][quantity]["decomposition"]["d_total_pct"]
            for cell in primary
        ]
        discs = [
            cell["quantities"][quantity]["decomposition"]["d_discretisation_pct"]
            for cell in primary
        ]
        models = [
            cell["quantities"][quantity]["decomposition"]["d_model_pct"]
            for cell in primary
        ]
        axis.bar(xs, discs, width=0.5, label="discretisation term", color="#7f8c8d")
        axis.bar(
            xs, models, width=0.5, bottom=discs, label="model term", color="#16a085"
        )
        axis.plot(xs, totals, "ko", ms=5, label="total vs anchor")
        axis.axhline(0.0, color="k", lw=0.8)
        axis.set_xticks(xs)
        axis.set_xticklabels([f"J={cell['J']:g}" for cell in primary])
        axis.set_ylabel("per cent of anchor")
        axis.set_title(
            f"{quantity}: deviation split (primary core policy)", fontsize=10
        )
        axis.grid(alpha=0.3, axis="y")
        axis.legend(fontsize=7)
    fig.tight_layout()
    path = figdir / "w20_deviation_split.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(str(path.relative_to(REPO_ROOT)))
    return figures


# ---------------------------------------------------------------------------
# criteria / verdict
# ---------------------------------------------------------------------------
def build_criteria(cells: list[dict]) -> list[dict]:
    primary = {
        cell["J"]: cell for cell in cells if cell["core_policy"] == PRIMARY_CORE_POLICY
    }
    similarity = {
        cell["J"]: cell for cell in cells if cell["core_policy"] == "similarity"
    }
    criteria = []

    def add(cid, statement, threshold, measured, passed):
        criteria.append(
            {
                "id": cid,
                "statement": statement,
                "threshold": threshold,
                "measured": measured,
                "pass": bool(passed),
            }
        )

    mission_cells = [primary[j] for j in MISSION_J]
    finite = all(
        math.isfinite(cell["quantities"][q]["levels"][i])
        for cell in mission_cells
        for q in ("KT", "10KQ")
        for i in range(3)
    )
    add(
        "W1-coverage",
        "three mission advance ratios x three panel densities x K_T/10K_Q, all finite",
        "9 level values per quantity, all finite",
        {
            "n_levels": sum(len(c["levels"]) for c in mission_cells),
            "J": list(MISSION_J),
            "densities": [level["label"] for level in PANEL_LEVELS],
            "all_finite": finite,
        },
        finite,
    )

    physical = all(cell["physical_all_levels"] for cell in mission_cells)
    add(
        "W2-physical-admissibility",
        "every level yields K_T > 0 and resisting K_Q > 0 at the mission advance ratios",
        "all levels physical",
        {j: primary[j]["physical_all_levels"] for j in MISSION_J},
        physical,
    )

    order_rows = {
        f"{q}@J={j:g}": {
            "observed_order": primary[j]["quantities"][q]["observed_order"],
            "deviation_pct": primary[j]["quantities"][q][
                "order_deviation_from_theory_pct"
            ],
            "pass": primary[j]["quantities"][q]["order_within_tolerance"],
        }
        for q in ("KT", "10KQ")
        for j in MISSION_J
    }
    order_pass = all(row["pass"] for row in order_rows.values())
    add(
        "W3-observed-order-vs-theory",
        f"observed order within {ORDER_TOLERANCE * 100:.0f} % of p_th = {THEORY_ORDER:g} at every mission J",
        f"|p_obs - {THEORY_ORDER:g}| / {THEORY_ORDER:g} < {ORDER_TOLERANCE:.2f}",
        order_rows,
        order_pass,
    )

    gci_rows = {
        f"{q}@J={j:g}": {
            "gci_fine_pct": primary[j]["quantities"][q]["gci_fine_pct"],
            "within_3pct": primary[j]["quantities"][q]["gci_within_3pct"],
            "within_1pct": primary[j]["quantities"][q]["gci_within_1pct"],
        }
        for q in ("KT", "10KQ")
        for j in MISSION_J
    }
    add(
        "W4-gci-tabulated-and-tested",
        "fine-grid GCI is tabulated for both quantities at every mission J and the 3 % acceptance is answered",
        "tabulated (no threshold on GCI itself; the answer is reported)",
        gci_rows,
        True,
    )

    closure = max(
        abs(primary[j]["quantities"][q]["decomposition"]["closure_residual_pct_points"])
        for j in ALL_J
        for q in ("KT", "10KQ")
    )
    add(
        "W5-decomposition-closure",
        "discretisation + model term equals the total anchor deviation",
        f"residual <= {DECOMPOSITION_TOL:g} pct points",
        {"max_residual_pct_points": closure},
        closure <= DECOMPOSITION_TOL,
    )

    design = primary[DESIGN_J]
    dkt = abs(design["quantities"]["KT"]["delta_vs_anchor_pct"])
    dkq = abs(design["quantities"]["10KQ"]["delta_vs_anchor_pct"])
    add(
        "W6-design-point-3pct-question",
        "|dK_T| and |dK_Q| at J = 0.70 are reported and the 3 % question is answered explicitly",
        "|dK_T| < 3 % and |dK_Q| < 3 % (threshold never relaxed)",
        {
            "J": DESIGN_J,
            "abs_dKT_pct": dkt,
            "abs_dKQ_pct": dkq,
            "KT_pass": dkt < DESIGN_TOL_PCT,
            "KQ_pass": dkq < DESIGN_TOL_PCT,
            "both_pass": dkt < DESIGN_TOL_PCT and dkq < DESIGN_TOL_PCT,
        },
        dkt < DESIGN_TOL_PCT and dkq < DESIGN_TOL_PCT,
    )

    agreement = {
        f"{q}@J={j:g}": {
            "fixed_within": primary[j]["quantities"][q]["order_within_tolerance"],
            "similarity_within": similarity[j]["quantities"][q][
                "order_within_tolerance"
            ],
        }
        for q in ("KT", "10KQ")
        for j in MISSION_J
    }
    agree = all(
        row["fixed_within"] == row["similarity_within"] for row in agreement.values()
    )
    inert = bool(PROBE.get("policy_range_is_inert"))
    add(
        "W7-core-policy-agreement",
        "the observed-order verdict does not depend on the vortex-core policy",
        "identical W3 verdict under both policies AND the policies are distinguishable",
        {
            "verdicts": agreement,
            "verdicts_agree": agree,
            "core_policy_inert_at_these_densities": inert,
            "interpretation": (
                "NOT TESTABLE: the two policies return bit-identical K_T/10K_Q at the L1 and "
                "L3 densities (see core_inertness_probe), so agreement cannot be evidence of "
                "robustness and the criterion is failed closed"
                if inert
                else "the policies move the answer and are compared above"
            ),
        },
        bool(agree and not inert),
    )
    band = {
        f"{q}@J={j:g}": {
            "d_total_pct": primary[j]["quantities"][q]["delta_vs_anchor_pct"],
            "last_increment_pct_of_anchor": primary[j]["quantities"][q][
                "last_increment_pct_of_anchor"
            ],
            "model_bounds_anchor_pct": primary[j]["quantities"][q][
                "model_bounds_anchor_pct"
            ],
            "model_outside_band": primary[j]["quantities"][q][
                "model_term_outside_band"
            ],
        }
        for q in ("KT", "10KQ")
        for j in ALL_J
    }
    add(
        "W8-band-attribution",
        "a decomposition that does not require a valid observed order is reported for every cell",
        "last-increment band and model bounds present for KT and 10KQ at every J",
        band,
        all(
            math.isfinite(row["last_increment_pct_of_anchor"]) for row in band.values()
        ),
    )
    return criteria


def assemble(outdir: Path) -> dict:
    cells = []
    for advance in ALL_J:
        for policy in CORE_POLICIES:
            cell = cell_study(advance, policy)
            print(
                f"  J={advance:g} {policy:<10} KT={cell['levels'][2]['KT']:.5f} "
                f"10KQ={cell['levels'][2]['10KQ']:.5f} "
                f"p_KT={cell['quantities']['KT']['observed_order']:.3f}",
                flush=True,
            )
            cells.append(cell)

    global PROBE
    print("  history tie (base configuration) ...", flush=True)
    tie = history_tie()
    print("  inertness probe ...", flush=True)
    PROBE = inertness_probe()

    data_files = write_csvs(outdir, cells)
    figures = make_figures(outdir, cells)
    criteria = build_criteria(cells)
    primary = {
        cell["J"]: cell for cell in cells if cell["core_policy"] == PRIMARY_CORE_POLICY
    }

    try:
        commit = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # pragma: no cover
        commit = "unknown"

    decomposition_table = {
        f"{q}@J={j:g}": primary[j]["quantities"][q]["decomposition"]
        for j in ALL_J
        for q in ("KT", "10KQ")
    }
    payload = {
        "schema": SCHEMA,
        "topic": "propeller-openwater-bem / W20 item 5: three-density GCI and error decomposition",
        "generated_by": "tensorbem.benchmarks.kp505_w20_gci",
        "regenerate": "python -m tensorbem.benchmarks.kp505_w20_gci",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repo_commit": commit,
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "solver": {
            "module": "src/tensorbem/propeller.py",
            "entry_point": "open_water_point",
            "wake_model": HELD_FIXED["wake_model"],
            "gci_tool": "src/tensorbem/uq.py:grid_convergence_index",
            "safety_factor": DEFAULT_SAFETY_FACTOR,
        },
        "anchor": {
            "source": "benchmarks/anchors/kp505_openwater.json",
            "kind": "SECONDARY_REPRODUCTION",
            "values": {f"{j:g}": anchor_values(j) for j in ALL_J},
            "eta0": {f"{j:g}": anchor_eta0(j) for j in ALL_J},
            "transcription": "papers/data/w20-anchors/kp505_transcription.json",
        },
        "design": {
            "mission_J": list(MISSION_J),
            "history_tie_J": HISTORY_J,
            "design_J": DESIGN_J,
            "panel_levels": list(PANEL_LEVELS),
            "refinement_ratio": REFINEMENT_RATIO,
            "core_policies": CORE_POLICIES,
            "primary_core_policy": PRIMARY_CORE_POLICY,
            "held_fixed": HELD_FIXED,
            "theory_order": THEORY_ORDER,
            "theory_order_secondary": THEORY_ORDER_SECONDARY,
            "order_tolerance": ORDER_TOLERANCE,
            "gci_accept_pct": GCI_ACCEPT_PCT,
            "design_tolerance_pct": DESIGN_TOL_PCT,
        },
        "cells": cells,
        "decomposition": decomposition_table,
        "history_tie": tie,
        "core_inertness_probe": PROBE,
        "criteria": criteria,
        "verdict": {
            "all_criteria_pass": all(c["pass"] for c in criteria),
            "pass_by_id": {c["id"]: c["pass"] for c in criteria},
            "history_29_13pct_reproduction": tie,
            "history_dKT_at_L3_lattice_pct": primary[HISTORY_J]["quantities"]["KT"][
                "delta_vs_anchor_pct"
            ],
            "design_point": {
                "dKT_pct": primary[DESIGN_J]["quantities"]["KT"]["delta_vs_anchor_pct"],
                "dKQ_pct": primary[DESIGN_J]["quantities"]["10KQ"][
                    "delta_vs_anchor_pct"
                ],
                "gci_KT_pct": primary[DESIGN_J]["quantities"]["KT"]["gci_fine_pct"],
                "gci_10KQ_pct": primary[DESIGN_J]["quantities"]["10KQ"]["gci_fine_pct"],
                "answer_under_3pct": (
                    abs(primary[DESIGN_J]["quantities"]["KT"]["delta_vs_anchor_pct"])
                    < DESIGN_TOL_PCT
                    and abs(
                        primary[DESIGN_J]["quantities"]["10KQ"]["delta_vs_anchor_pct"]
                    )
                    < DESIGN_TOL_PCT
                ),
            },
        },
        "data_files": data_files,
        "figures": figures,
        "limitations": [
            "The anchor is a SECONDARY_REPRODUCTION of the KRISO KP505 open-water test "
            "(see papers/data/w20-anchors/kp505_transcription.json); the model term computed "
            "here therefore contains both the lattice's formulation error and the anchor's "
            "own transcription uncertainty.",
            "Three levels give one observed order with no way to check that the asymptotic "
            "range has been reached; the four-level single-axis ladders of "
            "benchmarks/verified/propeller_openwater/kp505_gci_ladder.json already showed "
            "that the two overlapping triplets of the same ladder can disagree, so the "
            "fine-grid GCI here is a lower-bound-flavoured estimate, not a proof.",
            "eta0 is a derived ratio of K_T and 10 K_Q and is reported for completeness only; "
            "no criterion is attached to it.",
        ],
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / ARTEFACT_NAME).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", nargs="?", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args(argv)
    payload = assemble(args.outdir)
    print(
        json.dumps(
            {
                "written": str(args.outdir / ARTEFACT_NAME),
                "criteria": payload["verdict"]["pass_by_id"],
                "all_criteria_pass": payload["verdict"]["all_criteria_pass"],
                "design_point": payload["verdict"]["design_point"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
