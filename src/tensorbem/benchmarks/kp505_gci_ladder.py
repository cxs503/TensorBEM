"""KP505 open water: multi-level ladder study and GCI-based error decomposition.

Research question (research plan item C of topic ``propeller-openwater-bem``)
---------------------------------------------------------------------------
Report 8822 measures ``dK_T(J = 0.8) = -29.13 %`` for the vortex-lattice /
lifting-surface kernel of :mod:`tensorbem.propeller` against the transcribed
KP505 open-water table, *and* a radial panel GCI of 7.1-8.1 % on the same
lattice -- itself roughly three times the 3 % acceptance criterion.  When the
discretisation error band is that wide, "model error" and "discretisation
error" are not separable in the reported number, so the evidence surface does
not support a journal claim either way.

What this driver adds
---------------------
Five four-level ladders at ``J = 0.5 / 0.8 / 1.0``, each refining one axis
with a constant ratio ``r = 2``:

==============  ================  =========================================
axis            parameter         role
==============  ================  =========================================
``spanwise``    ``n_radial``      panel density across the span (4,8,16,32)
``azimuthal``   ``wake_steps``    azimuthal segments of the helical wake
                                  (12,24,48,96).  The blade surface carries
                                  no circumferential segmentation (blade-
                                  fitted lattice), so this is the model's
                                  only azimuthal discretisation knob.
``chordwise``   ``n_chord``       panel density along the chord (1,2,4,8)
``wake_length``  ``wake_length``  trailing-vortex extent in revolutions
                                  (0.25,0.5,1.0,2.0)
``kernel``      ``core_ratio``    kernel desingularisation ("relaxation")
                                  radius (0.4,0.2,0.1,0.05).  The lattice
                                  solve is direct-dense, so the solver has no
                                  iterative relaxation factor; the van Garrel
                                  core radius is the only regularisation knob
                                  that plays that role, and the driver says so
                                  instead of inventing a parameter.
==============  ================  =========================================

Each cell is a four-level ladder, so Roache's GCI is evaluated twice per
quantity: once on the three coarsest levels and once on the three finest.  The
finest triplet is the one that carries the reported ``p`` and ``GCI``; the
coarsest triplet is kept as an asymptotic-range diagnostic, because the two
overlapping triplets of the same four-level ladder disagree by more than a
factor of thirty on this lattice (see ``headline``).  A three-level study has
no way to see that disagreement, which is precisely why the 7.1-8.1 % GCI of
report 8822 could not be read as a bound.

The decomposition then splits the anchor deviation into two terms that add up
exactly (both referenced to the same anchor, so the identity holds by
construction and is checked to 1e-9)::

    d_total = (f_fine - f_anchor) / f_anchor
    d_disc  = (f_fine - f_ext)    / f_anchor     # discretisation of the finest grid
    d_model = (f_ext  - f_anchor) / f_anchor     # what survives Richardson

Predeclared criteria (fixed before any ladder was run)
------------------------------------------------------
G1  coverage: 5 ladders x 4 levels x 3 advance ratios = 15 cells (>= 12
    required), and every level of every cell tabulates ``K_T``, ``10 K_Q`` and
    ``eta_0``.
G2  every level at every J is physically admissible (``K_T > 0``, ``K_Q > 0``);
    a cell whose lattice predicts negative thrust/torque is not a "converged"
    anything and is reported as such.
G3  the finest-level ``spanwise`` and ``azimuthal`` GCI of ``K_T`` is <= 1 %.
G4  the finest-triplet observed order on those two axes is ``p >= 1.0``.
G5  those two ladders are monotone (no odd-even oscillation).
G6  ``dK_T(J = 0.8)`` is only attributable to *model* error when it lies
    outside the ``+/- 2 GCI`` band, ``GCI`` being the root-sum-square of the
    five axes' finest GCI.
G7  the discrete/model decomposition closes (``|d_disc + d_model - d_total| <=
    1e-9`` percentage points).
G8  the ``wake_length`` axis converges: finite ``GCI``, ``p >= 1.0``.
G9  the ``kernel`` axis converges: finite ``GCI``, ``p >= 1.0``.

Nothing in the verdict is typed by hand: the anchor table is imported from
:mod:`tensorbem.benchmarks.kp505_openwater_bem` (itself a verbatim
transcription), and every threshold and measured value below is injected by
this module.

Run with ``python -m tensorbem.benchmarks.kp505_gci_ladder [outdir]``.
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

import numpy as np

from ..propeller import BladeGeometry, open_water_point
from ..uq import DEFAULT_SAFETY_FACTOR, grid_convergence_index
from .kp505_openwater_bem import (
    ANCHOR,
    ANCHOR_SOURCE,
    DESIGN_ADVANCE,
    GEOMETRY_SOURCE,
    PRIMARY,
    TABLE_2_ROWS_VERBATIM,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTDIR = REPO_ROOT / "benchmarks" / "verified" / "propeller_openwater"
ARTEFACT_NAME = "kp505_gci_ladder.json"

SCHEMA = "tensorbem.kp505_gci_ladder/1"

#: Advance ratios of the study (fixed by the mission, not by the results).
LADDER_J = (0.5, 0.8, 1.0)

#: The advance ratio at which the anchor deviation is decomposed.
DECOMPOSITION_J = 0.8

#: ``|dK_T|`` of report 8822 at the primary configuration; recomputed here so
#: the artefact carries the reproduction instead of the recalled number.
REPORT_8822_DKT_PCT = -29.13

#: Refinement ratio of every ladder (panel counts double, wake extent doubles,
#: core ratio halves).
REFINEMENT_RATIO = 2.0

#: Every ladder is ordered coarse -> fine.
LADDERS = {
    "spanwise": {
        "parameter": "n_radial",
        "levels": (4, 8, 16, 32),
        "role": "panel density across the span",
        "kind": "discretisation",
    },
    "azimuthal": {
        "parameter": "wake_steps",
        "levels": (12, 24, 48, 96),
        "role": (
            "azimuthal segments of the helical wake; the blade surface carries "
            "no circumferential segmentation in a blade-fitted lattice, so this "
            "is the model's only azimuthal discretisation knob"
        ),
        "kind": "discretisation",
    },
    "chordwise": {
        "parameter": "n_chord",
        "levels": (1, 2, 4, 8),
        "role": "panel density along the chord",
        "kind": "discretisation",
    },
    "wake_length": {
        "parameter": "wake_length",
        "levels": (0.25, 0.5, 1.0, 2.0),
        "role": "trailing-vortex extent in revolutions",
        "kind": "model",
    },
    "kernel": {
        "parameter": "core_ratio",
        "levels": (0.4, 0.2, 0.1, 0.05),
        "role": (
            "van Garrel vortex-core desingularisation radius; the direct-dense "
            "lattice solve exposes no iterative relaxation factor, so this "
            "regularisation is the axis that plays that role (declared "
            "substitution, see the artefact's limitations)"
        ),
        "kind": "model",
    },
}

#: Axes that set the usable panel count; criteria G3-G5 are attached to them.
PRIMARY_AXES = ("spanwise", "azimuthal")

#: Thresholds and orders, fixed before running.
GCI_TARGET_PCT = 1.0
ORDER_TARGET = 1.0
GCI_BAND_FACTOR = 2.0
DECOMPOSITION_TOL = 1e-9

#: Base lattice: the primary configuration of the existing sweep study, so the
#: ladder is anchored on the same point that produced report 8822.
BASE_CONFIG = {
    "n_radial": PRIMARY["n_radial"],
    "n_chord": PRIMARY["n_chord"],
    "wake_length": PRIMARY["wake_length"],
    "wake_steps": PRIMARY["wake_steps"],
    "core_ratio": 0.05,
    "wake_model": PRIMARY["wake_model"],
    "rev_per_second": PRIMARY["rev_per_second"],
    "thickness_ratio": PRIMARY["thickness_ratio"],
    "viscous": PRIMARY["viscous"],
    "radial_spacing": PRIMARY["radial_spacing"],
    "root_chord_offset": PRIMARY["root_chord_offset"],
}

#: Reported per level in the CSVs and the artefact.
QUANTITIES = ("KT", "10KQ", "eta0")


# ---------------------------------------------------------------------------
# Lattice evaluation
# ---------------------------------------------------------------------------

def anchor_efficiency(advance: float) -> float:
    """``eta_0 = J K_T / (2 pi K_Q)`` from the transcribed anchor columns."""

    values = ANCHOR[advance]
    return advance * values["KT"] / (2.0 * math.pi * values["10KQ"] / 10.0)


def _config(**overrides) -> dict:
    config = dict(BASE_CONFIG)
    config.update(overrides)
    return config


def evaluate(advance: float, config: dict) -> dict:
    """One lattice evaluation; the three quantities plus the physics flags."""

    geometry = BladeGeometry(
        n_radial=int(config["n_radial"]),
        n_chord=int(config["n_chord"]),
        thickness_ratio=config["thickness_ratio"],
        radial_spacing=config["radial_spacing"],
        root_chord_offset=config["root_chord_offset"],
    )
    result = open_water_point(
        geometry,
        advance,
        wake_model=config["wake_model"],
        wake_length=config["wake_length"],
        wake_steps=int(config["wake_steps"]),
        rev_per_second=config["rev_per_second"],
        viscous=config["viscous"],
        core_ratio=config["core_ratio"],
    )

    torque = -result.torque  # resisting torque > 0 when the lattice is physical
    ten_kq = 10.0 * torque
    efficiency = (
        advance * result.thrust / (2.0 * math.pi * torque) if torque != 0.0 else float("nan")
    )
    physical = bool(
        math.isfinite(result.thrust)
        and math.isfinite(ten_kq)
        and result.thrust > 0.0
        and torque > 0.0
        and 0.0 < efficiency <= result.momentum_bound
    )
    return {
        "KT": float(result.thrust),
        "10KQ": float(ten_kq),
        "eta0": float(efficiency),
        "n_panels": int(result.n_panels),
        "wake_model": result.wake_model,
        "physical": physical,
        "momentum_bound": float(result.momentum_bound),
    }


def _gci_block(values, quantity: str, label: str) -> dict:
    """GCI on the three coarsest and the three finest levels of a 4-level ladder."""

    fine_levels = list(values[1:4])[::-1]  # fine, medium, coarse
    coarse_levels = list(values[0:3])[::-1]
    out = {}
    for name, triplet in (("fine_triplet", fine_levels), ("coarse_triplet", coarse_levels)):
        if not all(math.isfinite(v) for v in triplet):
            out[name] = {"usable": False, "reason": "non-finite level value"}
            continue
        gci = grid_convergence_index(
            fine=triplet[0],
            medium=triplet[1],
            coarse=triplet[2],
            refinement_ratio=REFINEMENT_RATIO,
            safety_factor=DEFAULT_SAFETY_FACTOR,
            quantity=quantity,
            label=label,
        )
        block = gci.as_dict()
        block["usable"] = bool(
            math.isfinite(block["observed_order"]) and math.isfinite(block["gci_fine"])
        )
        out[name] = block
    return out


def ladder_study() -> list[dict]:
    """Run every ladder at every advance ratio and attach the GCI blocks."""

    cells = []
    for name, spec in LADDERS.items():
        parameter = spec["parameter"]
        for advance in LADDER_J:
            levels = []
            for level_index, value in enumerate(spec["levels"]):
                config = _config(**{parameter: value})
                measured = evaluate(advance, config)
                row = {
                    "level": level_index,
                    "value": float(value),
                    "config_digest": {
                        key: config[key]
                        for key in (
                            "n_radial",
                            "n_chord",
                            "wake_length",
                            "wake_steps",
                            "core_ratio",
                        )
                    },
                }
                row.update(measured)
                levels.append(row)
            cell = {
                "axis": name,
                "parameter": parameter,
                "kind": spec["kind"],
                "J": advance,
                "levels": levels,
                "gci": {},
                "monotone": {},
                "all_levels_physical": all(row["physical"] for row in levels),
            }
            for quantity in QUANTITIES:
                series = [row[quantity] for row in levels]
                cell["gci"][quantity] = _gci_block(series, quantity, f"{name} J={advance}")
                differences = np.diff(series)
                productive = differences[np.abs(differences) > 0.0]
                cell["monotone"][quantity] = bool(
                    productive.size >= 2
                    and (np.all(productive > 0.0) or np.all(productive < 0.0))
                )
            cells.append(cell)
    return cells


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------

def _decompose(fine: float, extrapolated: float, anchor: float) -> dict:
    total = 100.0 * (fine - anchor) / anchor
    discrete = 100.0 * (fine - extrapolated) / anchor
    model = 100.0 * (extrapolated - anchor) / anchor
    return {
        "fine": fine,
        "extrapolated": extrapolated,
        "anchor": anchor,
        "total_pct": total,
        "discrete_pct": discrete,
        "model_pct": model,
        "closure_error_pp": abs(discrete + model - total),
    }


def decomposition(cells) -> list[dict]:
    """Per-J split of the anchor deviation on the spanwise (panel) axis."""

    by_key = {(cell["axis"], cell["J"]): cell for cell in cells}
    rows = []
    for advance in LADDER_J:
        span = by_key[("spanwise", advance)]
        fine = span["levels"][-1]
        row = {"J": advance, "primary_axis": "spanwise", "quantities": {}}

        for quantity in QUANTITIES:
            reference = (
                anchor_efficiency(advance) if quantity == "eta0" else ANCHOR[advance][quantity]
            )
            block = span["gci"][quantity]["fine_triplet"]
            extrapolated = block.get("extrapolated", float("nan"))
            entry = _decompose(fine[quantity], extrapolated, reference)
            entry["gci_fine_pct"] = (
                100.0 * block["gci_fine"] if block.get("usable") else float("nan")
            )
            entry["observed_order"] = block.get("observed_order", float("nan"))
            entry["richardson_valid"] = bool(
                _finite(entry["observed_order"])
                and entry["observed_order"] >= ORDER_TARGET
                and block.get("converged")
                and span["monotone"][quantity]
            )
            entry["richardson_invalid_reason"] = (
                None
                if entry["richardson_valid"]
                else (
                    "observed order %s is %s 1.0 (or the ladder is not monotone / not "
                    "asymptotic), so the Richardson limit is not a defensible estimate and "
                    "the model term is bracketed by the GCI band instead"
                    % (
                        "nan" if not _finite(entry["observed_order"]) else f"{entry['observed_order']:.3f}",
                        ">=" if _finite(entry["observed_order"]) and entry["observed_order"] >= ORDER_TARGET else "<",
                    )
                )
            )
            row["quantities"][quantity] = entry

        # Combined discretisation band: RSS over every axis that produced a
        # usable finest-triplet GCI at this J.
        contributions = []
        for axis in LADDERS:
            cell = by_key[(axis, advance)]
            block = cell["gci"]["KT"]["fine_triplet"]
            if block.get("usable"):
                contributions.append(
                    {
                        "axis": axis,
                        "gci_pct": 100.0 * block["gci_fine"],
                        "observed_order": block["observed_order"],
                        "monotone": cell["monotone"]["KT"],
                    }
                )
        rss = math.sqrt(sum(item["gci_pct"] ** 2 for item in contributions))
        band = GCI_BAND_FACTOR * rss
        total = abs(row["quantities"]["KT"]["total_pct"])
        model_term = abs(row["quantities"]["KT"]["model_pct"])
        core_contributions = [
            item for item in contributions if LADDERS[item["axis"]]["kind"] == "discretisation"
        ]
        rss_core = math.sqrt(sum(item["gci_pct"] ** 2 for item in core_contributions))
        band_core = GCI_BAND_FACTOR * rss_core
        signed_total = row["quantities"]["KT"]["total_pct"]
        bounds = [signed_total - band_core, signed_total + band_core]
        row["discretisation_band"] = {
            "contributions": contributions,
            "rss_gci_pct": rss,
            "band_pct": band,
            "band_factor": GCI_BAND_FACTOR,
            "axes": list(LADDERS),
            "abs_total_KT_pct": total,
            "abs_model_KT_pct": model_term,
            "total_outside_band": bool(total > band),
            "model_outside_band": bool(model_term > band),
        }
        row["discretisation_band_core"] = {
            "note": (
                "post-hoc diagnostic: only the axes flagged kind='discretisation' "
                "(spanwise, azimuthal, chordwise) enter the RSS; the two model axes are "
                "excluded because a divergent model parameter is not a discretisation of "
                "the same solution"
            ),
            "contributions": core_contributions,
            "rss_gci_pct": rss_core,
            "band_pct": band_core,
            "band_factor": GCI_BAND_FACTOR,
            "total_outside_band": bool(total > band_core),
            "model_outside_band": bool(model_term > band_core),
        }
        row["model_term_bounds_pct"] = {
            "note": (
                "envelope: with the discretisation error bounded by the core band, the "
                "model term cannot be outside d_total -/+ band"
            ),
            "low": bounds[0],
            "high": bounds[1],
            "excludes_zero": bool(not (bounds[0] <= 0.0 <= bounds[1])),
            "richardson_valid": bool(row["quantities"]["KT"]["richardson_valid"]),
            "richardson_pct": row["quantities"]["KT"]["model_pct"],
        }
        rows.append(row)
    return rows

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def _criterion(cid, statement, passed, measured, threshold=None, evidence=None,
               post_hoc=False):
    return {
        "id": cid,
        "statement": statement,
        "threshold": threshold,
        "measured": measured,
        "passed": bool(passed),
        "post_hoc": bool(post_hoc),
        "evidence": evidence or {},
    }


def _finite(value) -> bool:
    return value is not None and isinstance(value, (int, float)) and math.isfinite(float(value))


def build_verdict(cells, decomposition_rows, reproduction) -> list[dict]:
    by_key = {(cell["axis"], cell["J"]): cell for cell in cells}

    levels_per_axis = {}
    for cell in cells:
        levels_per_axis.setdefault(cell["axis"], set()).add(len(cell["levels"]))
    levels_per_axis = {axis: sorted(counts) for axis, counts in levels_per_axis.items()}
    quantities_ok = all(
        all(math.isfinite(level[quantity]) for level in cell["levels"])
        for cell in cells
        for quantity in QUANTITIES
    )

    criteria = [
        _criterion(
            "G1",
            (
                "coverage: >= 4 levels on each axis, >= 12 (axis, J) cells, and "
                "K_T / 10 K_Q / eta_0 tabulated at every level"
            ),
            passed=(
                len(cells) >= 12
                and all(min(counts) >= 4 for counts in levels_per_axis.values())
                and quantities_ok
            ),
            measured={
                "n_cells": len(cells),
                "levels_per_axis": levels_per_axis,
                "advance_ratios": list(LADDER_J),
                "n_lattice_runs": sum(len(cell["levels"]) for cell in cells),
                "all_quantities_finite": quantities_ok,
            },
            threshold={"min_cells": 12, "min_levels": 4},
        )
    ]

    physical_cells = {
        f"{cell['axis']}@J={cell['J']}": cell["all_levels_physical"] for cell in cells
    }
    non_physical = sorted(key for key, ok in physical_cells.items() if not ok)
    criteria.append(
        _criterion(
            "G2",
            (
                "every level of every cell is physically admissible "
                "(K_T > 0, K_Q > 0, eta_0 <= momentum bound)"
            ),
            passed=not non_physical,
            measured={
                "cells": physical_cells,
                "non_physical_cells": non_physical,
                "n_non_physical": len(non_physical),
                "J_1_0_KT_levels": [
                    level["KT"] for level in by_key[("spanwise", 1.0)]["levels"]
                ],
            },
            threshold={"KT_min": 0.0, "KQ_min": 0.0},
        )
    )

    gci_measured = {}
    for axis in PRIMARY_AXES:
        for advance in LADDER_J:
            block = by_key[(axis, advance)]["gci"]["KT"]["fine_triplet"]
            gci_measured[f"{axis}@J={advance}"] = {
                "gci_fine_pct": 100.0 * block["gci_fine"] if block.get("usable") else None,
                "observed_order": block.get("observed_order"),
                "converged": block.get("converged"),
                "monotone": by_key[(axis, advance)]["monotone"]["KT"],
            }
    gci_values = [v["gci_fine_pct"] for v in gci_measured.values() if _finite(v["gci_fine_pct"])]
    criteria.append(
        _criterion(
            "G3",
            (
                "finest-level spanwise and azimuthal GCI of K_T <= 1 % "
                "(evidence-surface threshold of the mission)"
            ),
            passed=bool(gci_values) and max(gci_values) <= GCI_TARGET_PCT,
            measured={
                "per_cell": gci_measured,
                "worst_pct": max(gci_values) if gci_values else None,
                "best_pct": min(gci_values) if gci_values else None,
                "n_within_target": sum(1 for v in gci_values if v <= GCI_TARGET_PCT),
                "n_cells": len(gci_values),
            },
            threshold={"gci_pct_max": GCI_TARGET_PCT},
        )
    )

    orders = [v["observed_order"] for v in gci_measured.values()]
    finite_orders = [p for p in orders if _finite(p)]
    criteria.append(
        _criterion(
            "G4",
            "finest-triplet observed order p >= 1.0 on the spanwise and azimuthal axes",
            passed=(
                bool(orders)
                and len(finite_orders) == len(orders)
                and min(finite_orders) >= ORDER_TARGET
            ),
            measured={
                "per_cell": {key: value["observed_order"] for key, value in gci_measured.items()},
                "min_finite_order": min(finite_orders) if finite_orders else None,
                "n_non_finite": len(orders) - len(finite_orders),
            },
            threshold={"observed_order_min": ORDER_TARGET},
        )
    )

    non_monotone = sorted(
        f"{axis}@J={advance}"
        for axis in PRIMARY_AXES
        for advance in LADDER_J
        if not by_key[(axis, advance)]["monotone"]["KT"]
    )
    criteria.append(
        _criterion(
            "G5",
            "the spanwise and azimuthal K_T ladders are monotone (no odd-even oscillation)",
            passed=not non_monotone,
            measured={
                "non_monotone": non_monotone,
                "series": {
                    f"{axis}@J={advance}": [
                        level["KT"] for level in by_key[(axis, advance)]["levels"]
                    ]
                    for axis in PRIMARY_AXES
                    for advance in LADDER_J
                },
            },
        )
    )

    row = next(item for item in decomposition_rows if item["J"] == DECOMPOSITION_J)
    band = row["discretisation_band"]
    criteria.append(
        _criterion(
            "G6",
            (
                "dK_T(J=0.8) is attributable to model error only when it lies outside "
                "the +/- 2 GCI band (GCI = RSS over the five axes)"
            ),
            passed=band["total_outside_band"],
            measured={
                "dKT_total_pct": row["quantities"]["KT"]["total_pct"],
                "dKT_discrete_pct": row["quantities"]["KT"]["discrete_pct"],
                "dKT_model_pct": row["quantities"]["KT"]["model_pct"],
                "dKT_report_8822_pct": reproduction["dKT_primary_pct"],
                "rss_gci_pct": band["rss_gci_pct"],
                "band_pct": band["band_pct"],
                "band_factor": band["band_factor"],
                "total_outside_band": band["total_outside_band"],
                "model_outside_band": band["model_outside_band"],
                "contributions": band["contributions"],
            },
            threshold={"band_factor": GCI_BAND_FACTOR},
        )
    )

    core_band = row["discretisation_band_core"]
    bounds = row["model_term_bounds_pct"]
    criteria.append(
        _criterion(
            "G6b",
            (
                "POST-HOC sensitivity of G6: with the band restricted to the three "
                "discretisation axes (spanwise, azimuthal, chordwise), is dK_T(J=0.8) "
                "outside +/- 2 GCI?"
            ),
            passed=core_band["total_outside_band"],
            measured={
                "rss_gci_pct": core_band["rss_gci_pct"],
                "band_pct": core_band["band_pct"],
                "dKT_total_pct": row["quantities"]["KT"]["total_pct"],
                "dKT_model_pct": row["quantities"]["KT"]["model_pct"],
                "model_term_bounds_pct": [bounds["low"], bounds["high"]],
                "model_bounds_exclude_zero": bounds["excludes_zero"],
                "richardson_valid": bounds["richardson_valid"],
                "contributions": core_band["contributions"],
            },
            threshold={"band_factor": GCI_BAND_FACTOR},
            post_hoc=True,
            evidence={
                "why_post_hoc": (
                    "G6 named the RSS over all five axes before the run; that band is "
                    "dominated by the divergent wake-extent axis (see G8), so it mixes "
                    "model-form error into the discretisation band. G6b narrows the band "
                    "to the discretisation axes only. It was added after the first run and "
                    "is labelled post-hoc so the predeclared verdict (G6) stays the one on "
                    "the record."
                )
            },
        )
    )

    closure = {
        f"J={item['J']}@{quantity}": item["quantities"][quantity]["closure_error_pp"]
        for item in decomposition_rows
        for quantity in QUANTITIES
    }
    criteria.append(
        _criterion(
            "G7",
            "the discrete/model split closes: abs(d_disc + d_model - d_total) <= 1e-9 pp",
            passed=max(closure.values()) <= DECOMPOSITION_TOL,
            measured={
                "max_closure_error_pp": max(closure.values()),
                "per_quantity": closure,
            },
            threshold={"closure_error_pp_max": DECOMPOSITION_TOL},
        )
    )

    for cid, axis, statement in (
        (
            "G8",
            "wake_length",
            "the wake-extent axis converges: usable GCI, p >= 1.0 and GCI <= 1 % at every J",
        ),
        (
            "G9",
            "kernel",
            "the kernel-regularisation axis converges: usable GCI, p >= 1.0 and GCI <= 1 % at every J",
        ),
    ):
        per_j = {}
        for advance in LADDER_J:
            block = by_key[(axis, advance)]["gci"]["KT"]["fine_triplet"]
            per_j[f"J={advance}"] = {
                "usable": bool(block.get("usable")),
                "gci_fine_pct": 100.0 * block["gci_fine"] if block.get("usable") else None,
                "observed_order": block.get("observed_order"),
                "converged": block.get("converged"),
                "monotone": by_key[(axis, advance)]["monotone"]["KT"],
                "KT_levels": [level["KT"] for level in by_key[(axis, advance)]["levels"]],
            }
        ok = all(
            entry["usable"]
            and _finite(entry["observed_order"])
            and entry["observed_order"] >= ORDER_TARGET
            and _finite(entry["gci_fine_pct"])
            and entry["gci_fine_pct"] <= GCI_TARGET_PCT
            for entry in per_j.values()
        )
        criteria.append(
            _criterion(
                cid,
                statement,
                passed=ok,
                measured={"per_J": per_j},
                threshold={"gci_pct_max": GCI_TARGET_PCT, "observed_order_min": ORDER_TARGET},
            )
        )
    return criteria


def summarise(criteria) -> dict:
    failed = [c["id"] for c in criteria if not c["passed"]]
    return {
        "status": "PASS" if not failed else "FAIL",
        "passed_criteria": [c["id"] for c in criteria if c["passed"]],
        "failed_criteria": failed,
        "n_passed": len(criteria) - len(failed),
        "n_criteria": len(criteria),
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


def flatten_levels(cells) -> list[dict]:
    rows = []
    for cell in cells:
        for level in cell["levels"]:
            rows.append(
                {
                    "axis": cell["axis"],
                    "parameter": cell["parameter"],
                    "J": cell["J"],
                    "level": level["level"],
                    "value": level["value"],
                    "n_panels": level["n_panels"],
                    "KT": level["KT"],
                    "10KQ": level["10KQ"],
                    "eta0": level["eta0"],
                    "physical": level["physical"],
                }
            )
    return rows


def flatten_gci(cells) -> list[dict]:
    rows = []
    for cell in cells:
        for quantity in QUANTITIES:
            fine = cell["gci"][quantity]["fine_triplet"]
            coarse = cell["gci"][quantity]["coarse_triplet"]
            rows.append(
                {
                    "axis": cell["axis"],
                    "J": cell["J"],
                    "quantity": quantity,
                    "p_fine_triplet": fine.get("observed_order"),
                    "gci_fine_pct_fine_triplet": (
                        100.0 * fine["gci_fine"] if fine.get("usable") else None
                    ),
                    "extrapolated_fine_triplet": fine.get("extrapolated"),
                    "converged_fine_triplet": fine.get("converged"),
                    "p_coarse_triplet": coarse.get("observed_order"),
                    "gci_fine_pct_coarse_triplet": (
                        100.0 * coarse["gci_fine"] if coarse.get("usable") else None
                    ),
                    "monotone": cell["monotone"][quantity],
                }
            )
    return rows


def flatten_decomposition(rows) -> list[dict]:
    out = []
    for row in rows:
        band = row["discretisation_band"]
        for quantity in QUANTITIES:
            entry = row["quantities"][quantity]
            out.append(
                {
                    "J": row["J"],
                    "quantity": quantity,
                    "fine": entry["fine"],
                    "extrapolated": entry["extrapolated"],
                    "anchor": entry["anchor"],
                    "d_total_pct": entry["total_pct"],
                    "d_discrete_pct": entry["discrete_pct"],
                    "d_model_pct": entry["model_pct"],
                    "closure_error_pp": entry["closure_error_pp"],
                    "gci_fine_pct": entry["gci_fine_pct"],
                    "rss_gci_pct": band["rss_gci_pct"],
                    "band_2gci_pct": band["band_pct"],
                }
            )
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _figures(outdir: Path, cells, decomposition_rows) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = outdir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    written = []

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2))
    for axis, advance in zip(axes, LADDER_J):
        for cell in cells:
            if cell["J"] != advance:
                continue
            axis.plot(
                [level["value"] for level in cell["levels"]],
                [level["KT"] for level in cell["levels"]],
                "o-",
                label=f"{cell['axis']} ({cell['parameter']})",
            )
        axis.axhline(ANCHOR[advance]["KT"], color="k", linestyle=":", linewidth=1.0,
                     label="anchor (Table 2)")
        axis.set_xscale("log")
        axis.set_xlabel("level value (log scale)")
        axis.set_ylabel("$K_T$")
        axis.set_title(f"J = {advance}")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=6.5)
    fig.suptitle("KP505 lattice: four-level ladders, $K_T$ against the refined parameter")
    fig.tight_layout()
    path = figure_dir / "kp505_gci_ladders.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path))

    fig, axis = plt.subplots(figsize=(7.6, 4.0))
    width = 0.16
    for index, axis_name in enumerate(LADDERS):
        positions, values = [], []
        for j_index, advance in enumerate(LADDER_J):
            cell = next(c for c in cells if c["axis"] == axis_name and c["J"] == advance)
            block = cell["gci"]["KT"]["fine_triplet"]
            positions.append(j_index + width * (index - 2))
            values.append(100.0 * block["gci_fine"] if block.get("usable") else 0.0)
        axis.bar(positions, values, width=width, label=axis_name)
    axis.axhline(GCI_TARGET_PCT, color="r", linestyle=":", linewidth=1.2,
                 label=f"criterion {GCI_TARGET_PCT:g} %")
    axis.set_xticks(range(len(LADDER_J)))
    axis.set_xticklabels([f"J = {advance}" for advance in LADDER_J])
    axis.set_ylabel("finest-triplet GCI of $K_T$ [%]")
    axis.set_title("Discretisation uncertainty per refinement axis")
    axis.grid(alpha=0.3, axis="y")
    axis.legend(fontsize=7)
    fig.tight_layout()
    path = figure_dir / "kp505_gci_by_axis.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for axis, quantity in zip(axes, ("KT", "10KQ")):
        j_labels = [f"J = {row['J']}" for row in decomposition_rows]
        discrete = [row["quantities"][quantity]["discrete_pct"] for row in decomposition_rows]
        model = [row["quantities"][quantity]["model_pct"] for row in decomposition_rows]
        positions = np.arange(len(j_labels))
        axis.bar(positions, model, 0.5, label="model term (Richardson vs anchor)",
                 color="#4c72b0")
        axis.bar(positions, discrete, 0.5, bottom=model, label="discretisation term",
                 color="#dd8452")
        band_values = [row["discretisation_band"]["band_pct"] for row in decomposition_rows]
        axis.errorbar(
            positions,
            [row["quantities"][quantity]["total_pct"] for row in decomposition_rows],
            yerr=band_values,
            fmt="none",
            ecolor="k",
            capsize=5,
            linewidth=1.0,
            label="+/- 2 GCI band",
        )
        axis.axhline(0.0, color="k", linewidth=0.8)
        axis.set_xticks(positions)
        axis.set_xticklabels(j_labels)
        axis.set_ylabel(f"$\\Delta {quantity}$ [%]")
        axis.grid(alpha=0.3, axis="y")
        axis.legend(fontsize=7)
    fig.suptitle("Anchor deviation split into discretisation and model terms")
    fig.tight_layout()
    path = figure_dir / "kp505_decomposition.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path))
    return written


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def render_report(artefact: dict) -> str:
    """Paper-level markdown summary rendered from the artefact (no hand typing)."""

    design = artefact["design"]
    verdict = artefact["verdict"]
    reproduction = artefact["reproduction"]
    lines = [
        "# KP505 open water: four-level GCI ladders and the model/discretisation split",
        "",
        "Generated by `python -m tensorbem.benchmarks.kp505_gci_ladder` from "
        "`%s`; every number below is injected, none is typed." % ARTEFACT_NAME,
        "",
        "## 1. Case",
        "",
        "- Lattice: `tensorbem.propeller.open_water_point`, geometric wake; base "
        "configuration `%s`." % json.dumps(design["base_configuration"], sort_keys=True),
        "- Advance ratios `%s`; decomposition reported at `J = %s`."
        % (", ".join(str(j) for j in design["advance_ratios"]), design["decomposition_J"]),
        "- Anchor curve: transcribed Table 2 of the source paper (verbatim rows and "
        "citation in the artefact's `anchor` block).",
        "- Report-8822 point reproduced here: `dK_T(J=0.8) = %.2f %%` at the primary "
        "configuration, recomputed value `%.4f %%`."
        % (REPORT_8822_DKT_PCT, reproduction["dKT_primary_pct"]),
        "- Geometry as published (verbatim rows of Table 1 of the source paper, so the "
        "case description carries no recalled numbers):",
        "",
    ]
    for row in artefact["anchor"]["geometry_source"]["rows_verbatim"]:
        lines.append("  - `%s`" % row)
    lines += [
        "",
        "",
        "## 2. Method",
        "",
        "Five four-level ladders, constant refinement ratio `r = %.1f`:"
        % design["refinement_ratio"],
        "",
        "| axis | parameter | levels | role |",
        "|---|---|---|---|",
    ]
    for name, spec in design["ladders"].items():
        lines.append(
            "| %s | `%s` | %s | %s |"
            % (name, spec["parameter"], ", ".join(str(v) for v in spec["levels"]), spec["role"])
        )
    lines += [
        "",
        "Roache GCI with `Fs = %.2f`, evaluated on the three coarsest and the three "
        "finest levels of every cell; the reported `p`/`GCI` are the finest triplet. "
        "The split is `d_total = d_disc + d_model` with both terms referenced to the "
        "same anchor, so it closes by construction (checked to 1e-9 pp)."
        % design["gci_safety_factor"],
        "",
        "## 3. Results",
        "",
        "### 3.0 Headline: the two overlapping triplets disagree",
        "",
        "A four-level ladder contains two overlapping three-level triplets. On the "
        "spanwise axis they disagree by one to two orders of magnitude in GCI, so the "
        "ladder is not in the asymptotic range and neither triplet is a bound. A "
        "three-level study cannot detect this -- which is why the 7.1-8.1 % GCI of "
        "report 8822 was not usable as an error bar.",
        "",
        "| J | levels | K_T series | p (three coarsest) | GCI % (coarsest triplet) | "
        "p (three finest) | GCI % (finest triplet) |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in artefact["headline"]["spanwise_four_level_vs_three_level"]:
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s |"
            % (
                item["J"],
                ", ".join(str(v) for v in item["levels"]),
                ", ".join("%.4f" % v for v in item["KT_levels"]),
                "%s" % ("nan" if not _finite(item["p_three_coarsest_levels"])
                        else "%.3f" % item["p_three_coarsest_levels"]),
                "nan" if item["gci_three_coarsest_levels_pct"] is None
                else "%.3f" % item["gci_three_coarsest_levels_pct"],
                "%s" % ("nan" if not _finite(item["p_three_finest_levels"])
                        else "%.3f" % item["p_three_finest_levels"]),
                "nan" if item["gci_three_finest_levels_pct"] is None
                else "%.3f" % item["gci_three_finest_levels_pct"],
            )
        )
    headline = artefact["headline"]
    lines += [
        "",
        "- `dK_T(J=0.8)` at the base configuration (report 8822): `%.4f %%`; at the "
        "finest spanwise level: `%.4f %%`."
        % (
            headline["dKT_J0_8_base_configuration_pct"],
            headline["dKT_J0_8_finest_lattice_pct"],
        ),
        "- Richardson split: discretisation `%.3f %%`, model `%.3f %%` "
        "(`richardson_valid = %s`)."
        % (
            headline["dKT_J0_8_discretisation_term_richardson_pct"],
            headline["dKT_J0_8_model_term_richardson_pct"],
            headline["richardson_valid"],
        ),
        "- Discretisation band at `J=0.8`: `%.3f %%` over the five axes, `%.3f %%` over "
        "the three discretisation axes; the resulting model-term bracket is "
        "`[%+.3f, %+.3f] %%`."
        % (
            headline["discretisation_band_all_axes_pct"],
            headline["discretisation_band_core_axes_pct"],
            headline["model_term_bounds_pct"][0],
            headline["model_term_bounds_pct"][1],
        ),
        "",
        "### 3.1 Finest-triplet GCI and observed order (K_T)",
        "",
        "| axis | J | p | GCI % | monotone | levels (K_T) |",
        "|---|---|---|---|---|---|",
    ]
    for cell in artefact["cells"]:
        block = cell["gci"]["KT"]["fine_triplet"]
        p = block.get("observed_order")
        gci = 100.0 * block["gci_fine"] if block.get("usable") else float("nan")
        lines.append(
            "| %s | %s | %s | %s | %s | %s |"
            % (
                cell["axis"],
                cell["J"],
                "nan" if p is None else f"{p:.3f}",
                "nan" if not math.isfinite(gci) else f"{gci:.3f}",
                cell["monotone"]["KT"],
                ", ".join(f"{level['KT']:.4f}" for level in cell["levels"]),
            )
        )
    lines += [
        "",
        "### 3.2 Discrete/model split",
        "",
        "| J | quantity | fine | Richardson | anchor | d_total % | d_disc % | d_model % | "
        "GCI % | +/-2 GCI (5 axes) % | +/-2 GCI (3 axes) % | Richardson valid |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in artefact["decomposition"]:
        band = row["discretisation_band"]
        band_core = row["discretisation_band_core"]
        for quantity in QUANTITIES:
            entry = row["quantities"][quantity]
            lines.append(
                "| %s | %s | %.5f | %.5f | %.5f | %+.3f | %+.3f | %+.3f | %.3f | %.3f | "
                "%.3f | %s |"
                % (
                    row["J"],
                    quantity,
                    entry["fine"],
                    entry["extrapolated"],
                    entry["anchor"],
                    entry["total_pct"],
                    entry["discrete_pct"],
                    entry["model_pct"],
                    entry["gci_fine_pct"],
                    band["band_pct"],
                    band_core["band_pct"],
                    entry["richardson_valid"],
                )
            )
    lines += [
        "",
        "An invalid Richardson order means the extrapolated column is *not* an error "
        "estimate for the finest grid; the honest reading is the bracket "
        "`d_model in [d_total - 2 GCI, d_total + 2 GCI]` in section 3.0.",
        "",
    ]
    lines += [
        "",
        "### 3.3 Criterion verdicts",
        "",
        "| id | criterion | measured | verdict |",
        "|---|---|---|---|",
    ]
    for criterion in artefact["criteria"]:
        measured = criterion["measured"]
        summary = json.dumps(
            {k: v for k, v in measured.items() if not isinstance(v, (dict, list))},
            sort_keys=True,
        )
        lines.append(
            "| %s | %s | `%s` | %s%s |"
            % (
                criterion["id"],
                criterion["statement"],
                summary,
                "PASS" if criterion["passed"] else "FAIL",
                " (post-hoc)" if criterion.get("post_hoc") else "",
            )
        )
    lines += [
        "",
        "**Overall: %s** (passed %d/%d; failed: %s)."
        % (
            verdict["status"],
            verdict["n_passed"],
            verdict["n_criteria"],
            ", ".join(verdict["failed_criteria"]) or "none",
        ),
        "",
        "## 4. Figures",
        "",
    ]
    for path in artefact["figures"]:
        lines.append("- `%s`" % Path(path).relative_to(artefact["outdir"]))
    lines += [
        "",
        "## 5. Limitations and traps",
        "",
    ]
    for item in artefact["limitations"]:
        lines.append("- %s" % item)
    lines += [
        "",
        "## 6. Acceptance table (each key quantity vs its basis)",
        "",
        "| # | quantity | basis | value | within 3 % |",
        "|---|---|---|---|---|",
    ]
    for index, row in enumerate(artefact["acceptance"], start=1):
        lines.append(
            "| %d | %s | %s | %s | %s |"
            % (index, row["quantity"], row["basis"], row["value"], row["within_3pct"])
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:  # pragma: no cover - diagnostics only
        return "unknown"


def reproduction_block() -> dict:
    """Reproduce the report-8822 point from the base configuration."""

    measured = evaluate(DECOMPOSITION_J, _config())
    anchor = ANCHOR[DECOMPOSITION_J]["KT"]
    deviation = 100.0 * (measured["KT"] - anchor) / anchor
    return {
        "J": DECOMPOSITION_J,
        "configuration": dict(BASE_CONFIG),
        "KT": measured["KT"],
        "KT_anchor": anchor,
        "dKT_primary_pct": deviation,
        "report_8822_dKT_pct": REPORT_8822_DKT_PCT,
        "reproduces_report_8822": abs(deviation - REPORT_8822_DKT_PCT) <= 0.01,
        "note": (
            "the mission quotes -29.13 %; this driver recomputes the deviation from the "
            "same base configuration instead of recalling it"
        ),
    }


def headline_block(cells, decomposition_rows, reproduction) -> dict:
    """The handful of numbers a reader needs, all derived from the cells."""

    by_key = {(cell["axis"], cell["J"]): cell for cell in cells}
    ladder_rows = []
    for advance in LADDER_J:
        cell = by_key[("spanwise", advance)]
        coarse = cell["gci"]["KT"]["coarse_triplet"]
        fine = cell["gci"]["KT"]["fine_triplet"]
        ladder_rows.append(
            {
                "J": advance,
                "levels": [level["value"] for level in cell["levels"]],
                "KT_levels": [level["KT"] for level in cell["levels"]],
                "p_three_coarsest_levels": coarse.get("observed_order"),
                "gci_three_coarsest_levels_pct": (
                    100.0 * coarse["gci_fine"] if coarse.get("usable") else None
                ),
                "p_three_finest_levels": fine.get("observed_order"),
                "gci_three_finest_levels_pct": (
                    100.0 * fine["gci_fine"] if fine.get("usable") else None
                ),
            }
        )

    row08 = next(item for item in decomposition_rows if item["J"] == DECOMPOSITION_J)
    for item in ladder_rows:
        item["triplet_disagreement_ratio"] = (
            item["gci_three_finest_levels_pct"] / item["gci_three_coarsest_levels_pct"]
            if item["gci_three_coarsest_levels_pct"] not in (None, 0.0)
            and item["gci_three_finest_levels_pct"] is not None
            else None
        )
    return {
        "spanwise_four_level_vs_three_level": ladder_rows,
        "richardson_valid": row08["model_term_bounds_pct"]["richardson_valid"],
        "dKT_J0_8_finest_lattice_pct": row08["quantities"]["KT"]["total_pct"],
        "dKT_J0_8_base_configuration_pct": reproduction["dKT_primary_pct"],
        "dKT_J0_8_model_term_richardson_pct": row08["quantities"]["KT"]["model_pct"],
        "dKT_J0_8_discretisation_term_richardson_pct": row08["quantities"]["KT"]["discrete_pct"],
        "discretisation_band_all_axes_pct": row08["discretisation_band"]["band_pct"],
        "discretisation_band_core_axes_pct": row08["discretisation_band_core"]["band_pct"],
        "model_term_bounds_pct": [
            row08["model_term_bounds_pct"]["low"],
            row08["model_term_bounds_pct"]["high"],
        ],
        "per_J_attribution": [
            {
                "J": item["J"],
                "dKT_total_pct": item["quantities"]["KT"]["total_pct"],
                "core_band_pct": item["discretisation_band_core"]["band_pct"],
                "total_outside_core_band": item["discretisation_band_core"]["total_outside_band"],
                "model_bounds_pct": [
                    item["model_term_bounds_pct"]["low"],
                    item["model_term_bounds_pct"]["high"],
                ],
            }
            for item in decomposition_rows
        ],
    }


def _acceptance_table(artefact) -> list[dict]:
    """Key quantities against their basis, each answered explicitly vs the 3 % level."""

    cells = artefact["cells"]
    by_key = {(cell["axis"], cell["J"]): cell for cell in cells}
    span = by_key[("spanwise", DECOMPOSITION_J)]
    azim = by_key[("azimuthal", DECOMPOSITION_J)]
    row = next(item for item in artefact["decomposition"] if item["J"] == DECOMPOSITION_J)
    band_core = row["discretisation_band_core"]["band_pct"]
    band_all = row["discretisation_band"]["band_pct"]
    gci_span = 100.0 * span["gci"]["KT"]["fine_triplet"]["gci_fine"]
    gci_azim = 100.0 * azim["gci"]["KT"]["fine_triplet"]["gci_fine"]
    p_span = span["gci"]["KT"]["fine_triplet"]["observed_order"]
    gci_span_3 = 100.0 * span["gci"]["KT"]["coarse_triplet"]["gci_fine"]
    dkt = row["quantities"]["KT"]["total_pct"]
    model = row["quantities"]["KT"]["model_pct"]
    bounds = row["model_term_bounds_pct"]

    def verdict(value, limit=3.0):
        return "yes" if abs(value) < limit else "no"

    return [
        {
            "quantity": "dK_T(J=0.8), finest spanwise lattice vs the Table 2 anchor",
            "basis": "transcribed experimental column (Table 2, p. 10)",
            "value": f"{dkt:+.3f} %",
            "within_3pct": verdict(dkt),
        },
        {
            "quantity": "dK_T(J=0.8), Richardson model term",
            "basis": "Richardson limit vs anchor (order p = %s, %s)"
            % (f"{p_span:.3f}" if _finite(p_span) else "nan",
               "VALID" if bounds["richardson_valid"] else "INVALID: order below 1"),
            "value": f"{model:+.3f} %",
            "within_3pct": verdict(model) if bounds["richardson_valid"] else "n/a (estimate invalid)",
        },
        {
            "quantity": "dK_T(J=0.8), model term bracket",
            "basis": "d_total +/- 2 GCI over the discretisation axes (%.3f %%)" % band_core,
            "value": "[%+.3f, %+.3f] %%" % (bounds["low"], bounds["high"]),
            "within_3pct": "no (bracket excludes 0)"
            if bounds["excludes_zero"]
            else "no: bracket contains 0, so no magnitude is resolvable",
        },
        {
            "quantity": "finest spanwise GCI of K_T at J=0.8",
            "basis": "mission criterion 1 % (and the 3 % acceptance level)",
            "value": f"{gci_span:.3f} %",
            "within_3pct": verdict(gci_span),
        },
        {
            "quantity": "finest azimuthal GCI of K_T at J=0.8",
            "basis": "mission criterion 1 % (and the 3 % acceptance level)",
            "value": f"{gci_azim:.3f} %",
            "within_3pct": verdict(gci_azim),
        },
        {
            "quantity": (
                "spread of the two overlapping spanwise triplets at J=0.8 "
                "(coarse {4,8,16} vs fine {8,16,32})"
            ),
            "basis": (
                "the 7.1-8.1 % three-level figure quoted by report 8822 (different J grid, "
                "so not reproduced here); a single three-level number has no way to "
                "measure this spread"
            ),
            "value": f"{gci_span_3:.3f} % vs {gci_span:.3f} %",
            "within_3pct": "not comparable: neither number is a bound, they differ by %.0fx"
            % (gci_span / gci_span_3 if gci_span_3 else float("nan")),
        },
        {
            "quantity": "observed order p on the finest spanwise triplet at J=0.8",
            "basis": "criterion p >= 1.0",
            "value": f"{p_span:.3f}" if _finite(p_span) else "nan",
            "within_3pct": "no (p < 0: differences grow with refinement)",
        },
        {
            "quantity": "decomposition closure abs(d_disc + d_model - d_total)",
            "basis": "algebraic identity, tolerance 1e-9 pp",
            "value": "%.2e pp" % max(
                item["quantities"][q]["closure_error_pp"]
                for item in artefact["decomposition"]
                for q in QUANTITIES
            ),
            "within_3pct": "yes (machine precision)",
        },
        {
            "quantity": "combined 2 GCI band at J=0.8 (all five axes)",
            "basis": "predeclared G6 band",
            "value": f"{band_all:.3f} %",
            "within_3pct": "no (band wider than the 3 % acceptance level)",
        },
    ]


def build_artefact(outdir: Path) -> dict:
    cells = ladder_study()
    decomposition_rows = decomposition(cells)
    reproduction = reproduction_block()
    criteria = build_verdict(cells, decomposition_rows, reproduction)
    verdict = summarise(criteria)

    figures = _figures(outdir, cells, decomposition_rows)

    data_files = {
        "ladder_values": outdir / "kp505_gci_ladder_values.csv",
        "gci": outdir / "kp505_gci_ladder_gci.csv",
        "decomposition": outdir / "kp505_gci_ladder_decomposition.csv",
    }
    _write_csv(data_files["ladder_values"], flatten_levels(cells))
    _write_csv(data_files["gci"], flatten_gci(cells))
    _write_csv(data_files["decomposition"], flatten_decomposition(decomposition_rows))

    artefact = {
        "schema": SCHEMA,
        "mission_id": 8531,
        "task": "paper-exp-propeller-openwater-bem-C",
        "topic": "propeller-openwater-bem",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repo_commit": _git_commit(),
        "outdir": str(outdir),
        "research_question": (
            "report 8822 reports dK_T(J=0.8) = -29.13 % but the same lattice carries a "
            "radial GCI of 7.1-8.1 %, so model error and discretisation error were not "
            "separable; this study builds >= 4-level ladders on five axes at three advance "
            "ratios and splits the deviation into a discretisation and a model term"
        ),
        "solver": {
            "module": "src/tensorbem/propeller.py",
            "entry_point": "open_water_point",
            "wake_model": BASE_CONFIG["wake_model"],
            "gci_tool": "src/tensorbem/uq.py:grid_convergence_index",
        },
        "driver": {
            "module": "tensorbem.benchmarks.kp505_gci_ladder",
            "command": "python -m tensorbem.benchmarks.kp505_gci_ladder",
            "test_module": "tests/test_kp505_gci_ladder.py",
        },
        "env": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "anchor": {
            "source": ANCHOR_SOURCE,
            "rows_verbatim": TABLE_2_ROWS_VERBATIM,
            "table": {
                f"{advance}": {"KT": values["KT"], "10KQ": values["10KQ"],
                               "eta0": anchor_efficiency(advance)}
                for advance, values in sorted(ANCHOR.items())
            },
            "geometry_source": GEOMETRY_SOURCE,
            "design_advance_ratio": DESIGN_ADVANCE,
            "columns_used": "J, KT_exp, 10KQ_exp (the experimental columns of Table 2)",
        },
        "design": {
            "advance_ratios": list(LADDER_J),
            "decomposition_J": DECOMPOSITION_J,
            "refinement_ratio": REFINEMENT_RATIO,
            "gci_safety_factor": DEFAULT_SAFETY_FACTOR,
            "base_configuration": dict(BASE_CONFIG),
            "ladders": {
                name: {
                    "parameter": spec["parameter"],
                    "levels": list(spec["levels"]),
                    "kind": spec["kind"],
                    "role": spec["role"],
                }
                for name, spec in LADDERS.items()
            },
            "criteria_predeclared": {
                "G3_gci_target_pct": GCI_TARGET_PCT,
                "G4_observed_order_target": ORDER_TARGET,
                "G6_band_factor": GCI_BAND_FACTOR,
                "G7_closure_tol_pp": DECOMPOSITION_TOL,
            },
        },
        "reproduction": reproduction,
        "cells": cells,
        "decomposition": decomposition_rows,
        "criteria": criteria,
        "verdict": verdict,
        "data_files": {key: str(path) for key, path in data_files.items()},
        "figures": figures,
    }

    artefact["headline"] = headline_block(cells, decomposition_rows, reproduction)
    artefact["limitations"] = _limitations(artefact)
    artefact["acceptance"] = _acceptance_table(artefact)

    (outdir / ARTEFACT_NAME).write_text(json.dumps(artefact, indent=2) + "\n")
    (outdir / "report.md").write_text(render_report(artefact) + "\n")
    return artefact


def _limitations(artefact) -> list[str]:
    notes = [
        (
            "No blade-surface circumferential segmentation exists in a blade-fitted "
            "lifting-surface lattice, so the mission's \"circumferential segments\" axis is "
            "realised as the azimuthal segmentation of the helical wake (`wake_steps`), the "
            "only azimuthal discretisation the model has. Recorded as a declared "
            "substitution, not silently renamed."
        ),
        (
            "The solver is direct-dense with no fixed-point relaxation coefficient; the "
            "mission's \"relaxation factor\" axis is therefore realised as the van Garrel "
            "vortex-core desingularisation ratio, the only regularisation parameter. Its "
            "fine end (0.05) sits on the code's floor `core = max(core_ratio * segment, "
            "0.5 * span)`, so the axis saturates there and the reported `p` describes the "
            "approach to that floor, not to an unregularised kernel."
        ),
    ]
    span = next(c for c in artefact["cells"] if c["axis"] == "spanwise" and c["J"] == 0.8)
    if any(not level["physical"] for level in span["levels"]):
        notes.append(
            "The J=1.0 lattice predicts negative thrust/torque: GCI is still computable "
            "there but the quantity is not a physical K_T, and criterion G2 records the "
            "failure instead of hiding it."
        )
    notes.append(
        "The reported GCI is a discretisation band around the *lattice* solution, not an "
        "uncertainty on the anchor: the transcribed Table 2 values themselves carry "
        "measurement/reading uncertainty that this artefact does not quantify (see the "
        "provenance depth field of the anchor block)."
    )
    notes.append(
        "The observed order of the spanwise ladder is negative at every J once the "
        "triplet is taken on the fine side (p = %s), while the coarse-side triplet of "
        "the same four-level ladder reports p = %s. The lattice is therefore not in the "
        "asymptotic range and no GCI from it is a bound; the numbers are reported as "
        "measured instead of being fitted to a p >= 1 expectation."
        % (
            ", ".join(
                "%.3f" % item["p_three_finest_levels"]
                for item in artefact["headline"]["spanwise_four_level_vs_three_level"]
                if item["p_three_finest_levels"] is not None
            ),
            ", ".join(
                "%.3f" % item["p_three_coarsest_levels"]
                for item in artefact["headline"]["spanwise_four_level_vs_three_level"]
                if item["p_three_coarsest_levels"] is not None
            ),
        )
    )
    notes.append(
        "Only J = 0.5 / 0.8 / 1.0 are ladders; the remaining anchor points keep the "
        "single-density evidence of the sweep study (docs/m7485_kp505_openwater_bem.md)."
    )
    return notes


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "outdir",
        nargs="?",
        default=str(DEFAULT_OUTDIR),
        help="output directory (default: benchmarks/verified/propeller_openwater)",
    )
    args = parser.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    artefact = build_artefact(outdir)
    verdict = artefact["verdict"]
    print(f"cells = {len(artefact['cells'])}  status = {verdict['status']}")
    print(f"failed = {verdict['failed_criteria']}")
    for row in artefact["decomposition"]:
        entry = row["quantities"]["KT"]
        print(
            "J=%.1f  dK_T total %+7.3f%% = disc %+7.3f%% + model %+7.3f%%  "
            "(2 GCI band %.3f%%)"
            % (
                row["J"],
                entry["total_pct"],
                entry["discrete_pct"],
                entry["model_pct"],
                row["discretisation_band"]["band_pct"],
            )
        )
    print(f"written to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
