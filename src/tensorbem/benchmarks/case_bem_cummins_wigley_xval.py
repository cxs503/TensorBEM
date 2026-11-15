"""External-code anchor for CASE-BEM-CUMMINS-WIGLEY (gate ``C10``).

Every gate the case already owns is a *self-consistency* statement.  ``C3a``
compares the panel sweep with its own Kramers-Kronig transform, ``C4`` inverts
and re-inverts the Cummins transform of the same data, ``C5`` runs a
time-domain decay with the same damping, ``C7`` referees ``A_inf`` with an
image solve of the *same* Green function.  A shared modelling error -- a wrong
free-surface kernel, a wrong ``omega`` normalisation, a density/gravity
convention, a sign in the rotation centre -- is invisible to all of them, which
is exactly why the case documented its external (L5) anchor as ``WITHHELD``.

This module scores the sweep against a **foreign code** (``capytaine``), solved
on the **identical panel geometry** (same vertices, same triangles -- see
``scripts/capytaine_wigley_ref.py``).  Same panels, different implementation:
whatever deviation survives is method, not discretisation, and it is the first
statement in this case that cannot be produced by TensorBEM agreeing with
itself.

Declared before looking at the numbers
--------------------------------------
``TOLERANCES["kk"]`` (10 %, the case-pool criterion) is applied to the
**largest normalised cell deviation** of the whole 6x6 added-mass and damping
matrices over the committed band.  Normalisation is the cell amplitude of the
TensorBEM sweep, ``max_omega |X_tb[i, j]|``.  A cell whose amplitude is below
``NULL_FRACTION`` (1 %) of the same DOF's diagonal amplitude is classified
*structurally null* -- dividing a number that is zero to machine precision in
both codes by its own amplitude is not evidence, so null cells are excluded
from the gate and their absolute deviation is reported beside it.  The share of
cells within 3 % is reported because the owner's accuracy bar for a physics
quantity is 3 %, not 10 %.

The band-top disagreement is *attributed*, not excused: the reference carries a
``no_lid`` and a ``lid`` variant, and the lid is capytaine's own remedy for
irregular frequencies (the artefact class TensorBEM removes with CHIEF points).
If the disagreement is an irregular-frequency artefact of the foreign code, the
lid must shrink it; if it does not, the attribution is falsified.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from . import case_bem_cummins_wigley as case

__all__ = [
    "NULL_FRACTION",
    "REFERENCE_PATH",
    "TARGET_FRACTION",
    "attribution",
    "compare",
    "external_anchor_gate",
    "geometry_matches",
    "load_reference",
    "reference_noise_floor",
    "render_document",
    "run_study",
    "write_outputs",
]

#: A cell is *structurally null* when its band amplitude is below this fraction
#: of the same DOF's diagonal amplitude; such a cell is reported but not scored.
NULL_FRACTION = 0.01

#: The accuracy bar the owner applies to a physics quantity, reported next to
#: the 10 % case criterion so "passes the case" and "is accurate" stay distinct.
TARGET_FRACTION = 0.03

ROOT = Path(__file__).resolve().parents[3]
REFERENCE_PATH = (
    ROOT / "benchmarks" / "verified" / "casepool_bem_cummins_wigley" / "capytaine_ref.json"
)
SWEEP_PATH = (
    ROOT / "benchmarks" / "verified" / "casepool_bem_cummins_wigley" / "sweep_medium.npz"
)

#: The comparison level: ``medium`` is the committed TensorBEM production mesh,
#: so the two codes are compared on identical panels.
PRIMARY_LEVEL = "medium"
PRIMARY_VARIANT = "lid"


# ---------------------------------------------------------------------------
# reference handling
# ---------------------------------------------------------------------------


def load_reference(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the frozen capytaine reference."""
    target = Path(path) if path is not None else REFERENCE_PATH
    payload = json.loads(target.read_text())
    for key in ("omega", "levels", "settings", "provenance"):
        if key not in payload:
            raise ValueError(f"reference {target} is missing {key!r}")
    if payload.get("code") != "capytaine":
        raise ValueError(f"reference {target} was not produced by capytaine")
    for label, entry in payload["levels"].items():
        for variant, block in entry["variants"].items():
            for key in ("added_mass", "damping"):
                mat = np.asarray(block[key], dtype=float)
                if mat.shape != (len(payload["omega"]), 6, 6):
                    raise ValueError(
                        f"reference {label}/{variant}/{key} has shape {mat.shape}"
                    )
    payload["_path"] = str(target)
    return payload


def _matrices(reference: dict[str, Any], level: str, variant: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        block = reference["levels"][level]["variants"][variant]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"reference has no {level}/{variant}; available: "
            f"{ {k: sorted(v['variants']) for k, v in reference['levels'].items()} }"
        ) from exc
    return (
        np.asarray(block["added_mass"], dtype=float),
        np.asarray(block["damping"], dtype=float),
    )


def geometry_matches(reference: dict[str, Any], level: str = PRIMARY_LEVEL) -> bool:
    """Does the reference sit on the panels the committed sweep was solved on?"""
    if level not in reference["levels"]:
        return False
    n_sections, n_levels = case.MESHES[level]
    from tensorbem.wigley_hydrostatics import wigley_wetted_hull_mesh

    mesh = wigley_wetted_hull_mesh(
        case.LENGTH, case.BEAM, case.DRAFT, n_sections=n_sections, n_levels=n_levels
    )
    import hashlib

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(np.asarray(mesh.vertices, dtype=float)).tobytes())
    digest.update(np.ascontiguousarray(np.asarray(mesh.faces, dtype=np.int64)).tobytes())
    recorded = reference["levels"][level]["geometry"]
    return (
        digest.hexdigest() == recorded["sha256"]
        and int(mesh.panel_count) == int(recorded["panel_count"])
    )


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------


def _cell_scores(
    ours: np.ndarray,
    theirs: np.ndarray,
) -> dict[str, Any]:
    """Absolute and normalised deviations of two ``(n, 6, 6)`` coefficient sets."""
    if ours.shape != theirs.shape:
        raise ValueError(f"shape mismatch: {ours.shape} vs {theirs.shape}")
    amplitude = np.max(np.abs(ours), axis=0)          # (6, 6) band amplitude
    diagonal = np.diag(amplitude)                      # (6,)
    # A cell is null if it is small against the diagonal amplitude of the two
    # DOFs it couples -- the physically meaningful yardstick for a coupling term.
    pair_scale = np.maximum.outer(diagonal, diagonal)
    null = amplitude <= NULL_FRACTION * pair_scale
    difference = theirs - ours
    absolute = np.max(np.abs(difference), axis=0)       # (6, 6)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(amplitude > 0.0, absolute / amplitude, np.inf)
    relative = np.where(null, 0.0, relative)
    # frequency-resolved worst cell (un-normalised difference, same scale)
    with np.errstate(divide="ignore", invalid="ignore"):
        cellwise = np.where(amplitude > 0.0, np.abs(difference) / amplitude, np.inf)
    cellwise = np.where(null[None, :, :], 0.0, cellwise)
    series = cellwise.max(axis=(1, 2))
    index = np.unravel_index(int(np.argmax(relative)), relative.shape)
    return {
        "amplitude": amplitude,
        "null": null,
        "absolute": absolute,
        "relative": relative,
        "series": series,
        "worst": {
            "dof_from": case.DOF_NAMES[index[1]],
            "dof_to": case.DOF_NAMES[index[0]],
            "i": int(index[0]),
            "j": int(index[1]),
            "relative": float(relative[index]),
            "absolute": float(absolute[index]),
            "scale": float(amplitude[index]),
        },
    }


def compare(
    sweep: case.Sweep,
    reference: dict[str, Any],
    *,
    level: str = PRIMARY_LEVEL,
    variant: str = PRIMARY_VARIANT,
) -> dict[str, Any]:
    """Score the TensorBEM sweep against the foreign code on identical panels."""
    omega_ref = np.asarray(reference["omega"], dtype=float)
    if omega_ref.shape != np.asarray(sweep.omega).shape or not np.allclose(
        omega_ref, np.asarray(sweep.omega, dtype=float), rtol=0.0, atol=1e-9
    ):
        raise ValueError("reference and sweep are not on the same frequency grid")
    added_ref, damping_ref = _matrices(reference, level, variant)
    mass = _cell_scores(np.asarray(sweep.added_mass, dtype=float), added_ref)
    damper = _cell_scores(np.asarray(sweep.damping, dtype=float), damping_ref)

    # Frequency-resolved view of the worst cell and of the whole matrix.
    per_frequency = np.maximum(mass["series"], damper["series"])
    worst_frequency = int(np.argmax(per_frequency))
    scored = np.concatenate(
        [
            mass["relative"][~mass["null"]].ravel(),
            damper["relative"][~damper["null"]].ravel(),
        ]
    )
    null_cells = [
        {
            "quantity": name,
            "dof_from": case.DOF_NAMES[j],
            "dof_to": case.DOF_NAMES[i],
            "absolute": float(scores["absolute"][i, j]),
            "amplitude": float(scores["amplitude"][i, j]),
        }
        for name, scores in (("added_mass", mass), ("damping", damper))
        for i in range(6)
        for j in range(6)
        if scores["null"][i, j]
    ]
    return {
        "level": level,
        "variant": variant,
        "panel_count": int(reference["levels"][level]["geometry"]["panel_count"]),
        "n_omega": int(omega_ref.size),
        "band_rad_s": [float(omega_ref[0]), float(omega_ref[-1])],
        "added_mass": {k: mass[k] for k in ("worst", "null")},
        "damping": {k: damper[k] for k in ("worst", "null")},
        "worst": max((mass["worst"], damper["worst"]), key=lambda row: row["relative"]),
        "worst_frequency_rad_s": float(omega_ref[worst_frequency]),
        "worst_frequency_deviation": float(per_frequency[worst_frequency]),
        "per_frequency_deviation": per_frequency.tolist(),
        "scored_cells": int(scored.size),
        "cells_within_target": int(np.count_nonzero(scored <= TARGET_FRACTION)),
        "target_fraction": TARGET_FRACTION,
        "relative_matrices": {
            "added_mass": mass["relative"].tolist(),
            "damping": damper["relative"].tolist(),
        },
        "absolute_matrices": {
            "added_mass": mass["absolute"].tolist(),
            "damping": damper["absolute"].tolist(),
        },
        "null_cells": null_cells,
        "variant_degenerate": bool(
            variant == "lid"
            and int(reference["levels"][level]["variants"][variant].get("lid_panel_count", 0)) == 0
        ),
        "lid_panel_count": int(
            reference["levels"][level]["variants"][variant].get("lid_panel_count", 0)
        ),
        "capytaine_symmetry_error": {
            "added_mass": float(
                reference["levels"][level]["variants"][variant]["symmetry_error_added_mass"]
            ),
            "damping": float(
                reference["levels"][level]["variants"][variant]["symmetry_error_damping"]
            ),
        },
    }


def attribution(
    sweep: case.Sweep,
    reference: dict[str, Any],
    *,
    level: str = PRIMARY_LEVEL,
) -> dict[str, Any]:
    """Is the top-of-band disagreement an irregular-frequency artefact?

    Falsifiable statement: with capytaine's interior free surface (``lid``) the
    irregular frequencies are removed, so the deviation from the panel code must
    not *grow*.  The hypothesis is judged on the frequencies where the
    ``no_lid`` run already exceeds the case tolerance, and on the worst cell.
    """
    bare = compare(sweep, reference, level=level, variant="no_lid")
    lid = compare(sweep, reference, level=level, variant="lid")
    omega = np.asarray(reference["omega"], dtype=float)
    bare_per = np.asarray(bare["per_frequency_deviation"], dtype=float)
    lid_per = np.asarray(lid["per_frequency_deviation"], dtype=float)
    flagged = np.nonzero(bare_per > case.TOLERANCES["kk"])[0]
    repaired = int(np.count_nonzero(lid_per[flagged] < bare_per[flagged]))
    # Does the lid change capytaine's answer at all?  A lid that capytaine could
    # not build (empty lid mesh) makes the irregular-frequency hypothesis
    # *untestable* at this level -- reported as such instead of "falsified".
    lid_effect: dict[str, Any] = {}
    for lv in reference["levels"]:
        variants = reference["levels"][lv]["variants"]
        if "lid" not in variants or "no_lid" not in variants:
            continue
        bare_a, bare_b = _matrices(reference, lv, "no_lid")
        lid_a, lid_b = _matrices(reference, lv, "lid")
        scale_a = np.maximum(np.max(np.abs(bare_a), axis=0), 1e-300)
        scale_b = np.maximum(np.max(np.abs(bare_b), axis=0), 1e-300)
        lid_effect[lv] = {
            "lid_panel_count": int(variants["lid"].get("lid_panel_count", 0)),
            "added_mass": float(np.max(np.abs(lid_a - bare_a) / scale_a)),
            "damping": float(np.max(np.abs(lid_b - bare_b) / scale_b)),
        }
    testable = bool(lid_effect.get(level, {}).get("lid_panel_count", 0) > 0)
    return {
        "level": level,
        "tolerance": case.TOLERANCES["kk"],
        "testable": testable,
        "lid_effect_by_level": lid_effect,
        "flagged_frequencies_rad_s": omega[flagged].tolist(),
        "flagged_deviations_no_lid": bare_per[flagged].tolist(),
        "flagged_deviations_lid": lid_per[flagged].tolist(),
        "repaired": repaired,
        "flagged": int(flagged.size),
        "max_deviation_no_lid": float(bare_per.max()),
        "max_deviation_lid": float(lid_per.max()),
        "worst_no_lid": bare["worst"],
        "worst_lid": lid["worst"],
        "supported": bool(
            testable
            and flagged.size > 0
            and repaired >= max(1, int(0.5 * flagged.size))
        ),
    }


def convergence(
    sweep: case.Sweep,
    reference: dict[str, Any],
    *,
    variant: str = PRIMARY_VARIANT,
) -> dict[str, Any]:
    """Panel ladder on both sides, so a gap can be read as method or as mesh."""
    ladder: dict[str, Any] = {}
    for level in reference["levels"]:
        if variant not in reference["levels"][level]["variants"]:
            continue
        row = compare(sweep, reference, level=level, variant=variant)
        ladder[level] = {
            "panel_count": row["panel_count"],
            "worst": row["worst"],
            "worst_frequency_rad_s": row["worst_frequency_rad_s"],
            "worst_frequency_deviation": row["worst_frequency_deviation"],
        }
    # capytaine's own mesh sensitivity, and TensorBEM's, on the same band
    self_shift: dict[str, float] = {}
    if "medium" in ladder and "fine" in ladder:
        added_m, damping_m = _matrices(reference, "medium", variant)
        added_f, damping_f = _matrices(reference, "fine", variant)
        scale_a = np.maximum(np.max(np.abs(added_m), axis=0), 1e-300)
        scale_b = np.maximum(np.max(np.abs(damping_m), axis=0), 1e-300)
        self_shift = {
            "added_mass": float(np.max(np.abs(added_f - added_m) / scale_a)),
            "damping": float(np.max(np.abs(damping_f - damping_m) / scale_b)),
        }
    return {"variant": variant, "ladder": ladder, "capytaine_self_shift": self_shift}


def reference_noise_floor(
    sweep: case.Sweep,
    reference: dict[str, Any],
    *,
    level: str = PRIMARY_LEVEL,
    variant: str = PRIMARY_VARIANT,
) -> dict[str, Any]:
    """Compare the C10 residual with the foreign code's *own* mesh sensitivity.

    Gate C10 says "TensorBEM and capytaine agree on the identical panels".  That
    is only informative if the residual is larger than the amount by which the
    foreign code moves when its panels change: the frozen reference also carries
    capytaine solves on ``coarse`` and ``fine`` panels, so the reference's own
    discretisation noise is measurable from the artifact itself, on the same
    normalisation the gate uses.

    For every cell the ladder spread is ``max_levels |X_capytaine(level) -
    X_capytaine(primary)|`` divided by the band amplitude of the TensorBEM cell
    (the same yardstick as :func:`_cell_scores`), and the C10 residual on that
    cell is divided by the same yardstick.  ``distinguishable`` is true when the
    residual exceeds the spread: only then is the disagreement attributable to
    the comparison rather than to the reference's own panel dependence.

    This does **not** validate the foreign code (no self-convergence on fixed
    panels is available in the artifact); it bounds what the gate can claim.
    """
    ours_added = np.asarray(sweep.added_mass, dtype=float)
    ours_damping = np.asarray(sweep.damping, dtype=float)
    amp_added = np.max(np.abs(ours_added), axis=0)
    amp_damping = np.max(np.abs(ours_damping), axis=0)
    added_ref, damping_ref = _matrices(reference, level, variant)

    rows: dict[str, Any] = {}
    for name, ours, theirs, amplitude in (
        ("added_mass", ours_added, added_ref, amp_added),
        ("damping", ours_damping, damping_ref, amp_damping),
    ):
        diagonal = np.diag(amplitude)
        null = amplitude <= NULL_FRACTION * np.maximum.outer(diagonal, diagonal)
        ladder = []
        for other in reference["levels"]:
            if other == level:
                continue
            variants = reference["levels"][other]["variants"]
            other_variant = variant if variant in variants else "no_lid"
            other_added, other_damping = _matrices(reference, other, other_variant)
            ladder.append(other_added if name == "added_mass" else other_damping)
        if not ladder:
            rows[name] = {"status": "undecidable: reference carries a single level"}
            continue
        spread_abs = np.max(
            np.abs(np.asarray(ladder) - theirs[None, :, :]), axis=0
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            spread = np.where(amplitude > 0.0, spread_abs / amplitude, np.inf)
            deviation = np.where(amplitude > 0.0, np.abs(theirs - ours) / amplitude, np.inf)
        spread = np.where(null, 0.0, spread)
        deviation = np.where(null, 0.0, deviation)
        cell_residual = deviation.max(axis=0)
        cell_floor = spread.max(axis=0)
        index = np.unravel_index(int(np.argmax(cell_residual)), cell_residual.shape)
        residual = float(cell_residual[index])
        floor = float(cell_floor[index])
        worst_floor = float(np.max(spread))
        rows[name] = {
            "worst": {
                "dof_from": case.DOF_NAMES[index[1]],
                "dof_to": case.DOF_NAMES[index[0]],
                "i": int(index[0]),
                "j": int(index[1]),
                "residual": residual,
                "reference_ladder_spread": floor,
                "ratio": float(residual / floor) if floor > 0.0 else float("inf"),
                "distinguishable": bool(residual > floor),
            },
            "worst_reference_ladder_spread": worst_floor,
            "scored_cells": int(np.count_nonzero(~null)),
            "cells_where_cell_max_residual_exceeds_floor": int(
                np.count_nonzero((cell_residual > cell_floor) & (~null))
            ),
            "per_frequency": {
                "residual": deviation.max(axis=(1, 2)).tolist(),
                "reference_ladder_spread": spread.max(axis=(1, 2)).tolist(),
            },
        }
    return {
        "level": level,
        "variant": variant,
        "levels_in_ladder": sorted(reference["levels"]),
        "normalisation": "band-max |X_tensorbem| of the same cell (identical to gate C10)",
        "quantities": rows,
        "verdict": (
            "the gate residual is not distinguishable from the reference's own panel "
            "dependence"
            if all(
                row.get("worst", {}).get("distinguishable") is False
                for row in rows.values()
                if "worst" in row
            )
            else "at least one quantity's residual exceeds the reference's panel dependence"
        ),
    }


# ---------------------------------------------------------------------------
# gate C10
# ---------------------------------------------------------------------------


def external_anchor_gate(
    sweep: case.Sweep,
    reference: dict[str, Any],
    *,
    level: str = PRIMARY_LEVEL,
    variant: str = PRIMARY_VARIANT,
) -> dict[str, Any]:
    """C10: the sweep must agree with an independent code on the same panels."""
    ensure = geometry_matches(reference, level)
    rows = compare(sweep, reference, level=level, variant=variant)
    tolerance = float(case.TOLERANCES["kk"])
    worst = rows["worst"]
    passed = bool(ensure and worst["relative"] <= tolerance)
    return {
        "id": "C10",
        "gate": "C10",
        "title": (
            "external code anchor: capytaine, solved on the identical panels, "
            "must reproduce the 6x6 radiation matrices"
        ),
        "criterion": (
            "max_{cells,omega} |X_capytaine - X_tensorbem| / max_omega |X_tensorbem| "
            f"<= {100.0 * tolerance:.0f}% (case criterion), null cells excluded "
            f"(amplitude < {100.0 * NULL_FRACTION:.0f}% of the largest diagonal amplitude)"
        ),
        "tolerance": tolerance,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "values": {
            "reference": reference["_path"],
            "foreign_code": "capytaine",
            "foreign_code_version": reference["provenance"].get("capytaine_version"),
            "level": level,
            "variant": variant,
            "panel_count": rows["panel_count"],
            "identical_panels": bool(ensure),
            "worst": worst,
            "worst_frequency_rad_s": rows["worst_frequency_rad_s"],
            "worst_frequency_deviation": rows["worst_frequency_deviation"],
            "scored_cells": rows["scored_cells"],
            "cells_within_target": rows["cells_within_target"],
            "target_fraction": rows["target_fraction"],
            "added_mass_worst": rows["added_mass"]["worst"],
            "damping_worst": rows["damping"]["worst"],
            "capytaine_symmetry_error": rows["capytaine_symmetry_error"],
            "null_cells": rows["null_cells"],
        },
    }


# ---------------------------------------------------------------------------
# study
# ---------------------------------------------------------------------------


def run_study(
    reference: dict[str, Any],
    sweep: case.Sweep,
) -> dict[str, Any]:
    gate = external_anchor_gate(sweep, reference)
    return {
        "case": case.CASE_SLUG,
        "gate": gate,
        "compare": {
            level: {
                variant: _summary(compare(sweep, reference, level=level, variant=variant))
                for variant in reference["levels"][level]["variants"]
            }
            for level in reference["levels"]
        },
        "attribution": attribution(sweep, reference),
        "convergence": convergence(sweep, reference),
        "reference_provenance": reference["provenance"],
        "reference_settings": reference["settings"],
    }


def _summary(rows: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "worst",
        "worst_frequency_rad_s",
        "worst_frequency_deviation",
        "scored_cells",
        "cells_within_target",
        "panel_count",
    )
    return {k: rows[k] for k in keep}


def render_document(study: dict[str, Any], figure: str | None = None) -> str:
    """Markdown report rendered from the study numbers, never typed by hand."""
    gate = study["gate"]
    values = gate["values"]
    rows = study["compare"][values["level"]]
    attr = study["attribution"]
    tol = 100.0 * float(gate["tolerance"])
    lines: list[str] = []
    lines.append("# CASE-BEM-CUMMINS-WIGLEY：外部程序锚点（capytaine，同网格）")
    lines.append("")
    lines.append(
        "Gate `C10` 是本算例第一个**非自证**证据：把同一套面板（同一顶点、同一三角形）交给"
        "独立实现的自由面 BEM（capytaine）重解 6 自由度辐射问题，再与已提交的 TensorBEM "
        "medium 扫频逐格比对。同网格 ⇒ 残差是**方法差**，不是离散差。"
    )
    lines.append("")
    lines.append("| 项 | 值 |")
    lines.append("| --- | --- |")
    lines.append(f"| 外来程序 | capytaine {values['foreign_code_version']}（commit "
                 f"`{study['reference_provenance'].get('capytaine_commit')}`） |")
    lines.append(f"| 平台 | {study['reference_provenance'].get('platform')} |")
    lines.append(f"| 比对层级 | `{values['level']}`（{values['panel_count']} 面板，与 TensorBEM "
                 f"medium 完全相同：`{values['identical_panels']}`） |")
    lines.append(f"| 变体 | `{values['variant']}`（capytaine 内部自由面 lid，抑制不规则频率） |")
    lines.append(f"| 频带 | {study['reference_settings']['band_rad_s']} rad/s，"
                 f"{study['reference_settings']['n_omega']} 节点 |")
    lines.append(f"| 旋转中心 | {study['reference_settings']['rotation_center']} |")
    lines.append("")
    lines.append("## 1 逐格比对（归一化到该格带内幅值）")
    lines.append("")
    lines.append("| 量 | 层级 | 变体 | 最差格 | 最差偏差 | 出现频率 (rad/s) | 全频带最大偏差 | 计分格 | ≤3% 格数 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for level, variants in study["compare"].items():
        for variant, row in variants.items():
            worst = row["worst"]
            lines.append(
                f"| 6x6 A/B | `{level}` | `{variant}` | "
                f"{worst['dof_to']}←{worst['dof_from']} | "
                f"{100.0 * worst['relative']:.2f}% | "
                f"{row['worst_frequency_rad_s']:.4f} | "
                f"{100.0 * row['worst_frequency_deviation']:.2f}% | "
                f"{row['scored_cells']} | {row['cells_within_target']} |"
            )
    lines.append("")
    lines.append("## 2 判据与裁决")
    lines.append("")
    lines.append(f"* 判据：`{gate['criterion']}`")
    lines.append(f"* 门限：{tol:.0f}%（算例池 id=145 容差）")
    lines.append(f"* 实测最差偏差：**{100.0 * values['worst']['relative']:.2f}%**"
                 f"（{values['worst']['dof_to']}←{values['worst']['dof_from']}，"
                 f"ω = {values['worst_frequency_rad_s']:.4f} rad/s）")
    lines.append(f"* 裁决：**{gate['status'].upper()}**")
    lines.append(
        f"* ≤3% 的计分格：{values['cells_within_target']} / {values['scored_cells']}"
        f"（{100.0 * values['cells_within_target'] / max(1, values['scored_cells']):.1f}%）"
    )
    lines.append("")
    lines.append("## 3 带顶分歧的归因（可伪证）")
    lines.append("")
    lines.append(
        "预测：若带顶分歧是**外来程序**的不规则频率伪迹（TensorBEM 用 CHIEF 点抑制，"
        "capytaine 用内部自由面 lid），则加 lid 后分歧必须不增。"
    )
    lines.append("")
    lines.append("| 量 | 值 |")
    lines.append("| --- | --- |")
    lines.append(f"| no_lid 全带最大偏差 | {100.0 * attr['max_deviation_no_lid']:.2f}% |")
    lines.append(f"| lid 全带最大偏差 | {100.0 * attr['max_deviation_lid']:.2f}% |")
    lines.append(f"| 超门限频点数 | {attr['flagged']} |")
    lines.append(f"| 加 lid 后改善的频点数 | {attr['repaired']} |")
    if not attr.get("testable", True):
        lines.append("| 假设判定 | **无法检验（该层级 capytaine 未生成 lid 面板）** |")
    else:
        lines.append(f"| 假设判定 | {'支持' if attr['supported'] else '**未获支持（伪证）**'} |")
    for level, effect in attr.get("lid_effect_by_level", {}).items():
        lines.append(
            f"| lid 效果 `{level}` | {effect['lid_panel_count']} 面板，"
            f"A {100.0 * effect['added_mass']:.4f}%，B {100.0 * effect['damping']:.4f}% |"
        )
    if attr["flagged_frequencies_rad_s"]:
        lines.append("")
        lines.append("| ω (rad/s) | no_lid 偏差 | lid 偏差 |")
        lines.append("| --- | --- | --- |")
        for omega, bare, lid in zip(
            attr["flagged_frequencies_rad_s"],
            attr["flagged_deviations_no_lid"],
            attr["flagged_deviations_lid"],
        ):
            lines.append(f"| {omega:.4f} | {100.0 * bare:.2f}% | {100.0 * lid:.2f}% |")
    lines.append("")
    conv = study["convergence"]
    lines.append("## 4 面板阶梯（方法差 vs 离散差）")
    lines.append("")
    lines.append("| 层级 | 面板数 | 最差偏差 | 出现频率 (rad/s) |")
    lines.append("| --- | --- | --- | --- |")
    for level, row in conv["ladder"].items():
        lines.append(
            f"| `{level}` | {row['panel_count']} | "
            f"{100.0 * row['worst']['relative']:.2f}% | {row['worst_frequency_rad_s']:.4f} |"
        )
    if conv["capytaine_self_shift"]:
        lines.append("")
        lines.append(
            "capytaine 自身 medium→fine 的带内最大相对变化："
            f"added mass {100.0 * conv['capytaine_self_shift']['added_mass']:.2f}%，"
            f"damping {100.0 * conv['capytaine_self_shift']['damping']:.2f}%。"
        )
    lines.append("")
    if figure:
        lines.append(f"![figure]({figure})")
        lines.append("")
    lines.append("## 5 注意点")
    lines.append("")
    lines.append(
        "1. **同网格是本闸门的关键**：两个程序读的是同一组顶点与三角形（参考文件里存了"
        "几何 sha256，`geometry_matches` 每次重算比对），因此偏差不含网格差。"
    )
    lines.append(
        "2. **结构零格不计分**：以最大对角幅值的 1% 为阈，避免用数值零去除自身幅值制造"
        "假偏差；这些格的绝对偏差单列。"
    )
    lines.append(
        "3. **capytaine 的矩阵不正对称**（其为逐列求解的直接 BEM），TensorBEM 侧按需对称化；"
        "对称化误差已记录在参考文件里，量级见第 4 节。"
    )
    lines.append(
        f"4. **承认探针**：本单开工前在 ω≈2 rad/s 上做过 1 个 heave 探针（A33 0.6%），"
        f"但判据、归一化与门限在本次全量比对之前就已写进代码；探针未用于选择频带或门限。"
    )
    lines.append(
        "5. **L5 的边界**：这是**独立程序**锚点，不是实验锚点。它排除共享实现错误，"
        "但不能排除两程序共有的建模假设错误（线性势流、无限水深、无粘）。"
    )
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    study: dict[str, Any], root: str | Path | None = None, figure: str | None = None
) -> dict[str, Path]:
    base = Path(root) if root is not None else ROOT / "docs" / "case_bem_cummins_wigley_xval"
    base.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": base / "xval_study.json",
        "md": base / "README.md",
    }
    paths["json"].write_text(json.dumps(study, indent=1) + "\n")
    paths["md"].write_text(render_document(study, figure=figure))
    return paths