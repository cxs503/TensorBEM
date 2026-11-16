"""``CASE-BEM-CUMMINS-WIGLEY``: closure-band gates judged with an **independent** ``A_inf``.

Why this script exists
----------------------
The extended-band closure run (``docs/dev_report_m23130_cummins_wigley_band_closure_executed.md``)
left three gates failing on the 69-point band ``[0.10, 20] rad/s``:
``G3_kramers_kronig`` 33 %, ``G4_free_decay`` 18 %, ``G5_round_trip`` 33 %.
The G5 attribution (``docs/dev_report_m21140_cummins_g5_ainf_attribution.md``)
closed with a statement of the remaining gap, quoted verbatim:

    「需要独立的 A_∞ 锚（ω→∞ 极限的 Laplace 解，或文献值），而不是又一个从同一 (A,B) 造出来的估计器」

That anchor exists in the repository but **has never been used to judge these
gates**: :func:`tensorbem.added_mass_limit.added_mass_limit` returns
``A(omega -> inf)`` from the antisymmetric-image solve of the Dirichlet
calm-water surface (``phi = 0`` on ``z = 0``, closed exactly by the doubled
body).  It shares no code and no data with the frequency sweep: no tail fit, no
free parameter, no band.

Two things are needed before it can judge anything:

1. **The reference point.** The sweep is taken about
   ``ROTATION_CENTER = (0, 0, -T/2)``; :func:`tensorbem.benchmarks.case_bem_cummins_wigley.added_mass_inf_anchor`
   solves about ``CENTER = (0, 0, 0)`` (the waterline).  ``A_55`` is *not*
   invariant under that shift (it moves with the ``A_35`` coupling), so the
   image matrix is re-expressed with
   :func:`tensorbem.added_mass_limit.shift_added_mass` before any comparison.
   The shift is validated against a *direct* image solve about the sweep's
   centre.
2. **An uncertainty.** A single number would just replace one unquantified
   ``A_inf`` with another.  The image value is therefore computed on a mesh
   ladder and carried as an interval.

The gates themselves are **not redefined**: :func:`analysis_gates` gains an
``added_mass_inf`` override, so the predeclared statistics (``G3`` = max KK
residual normalised by ``max|A - A_inf|``, ``G5`` = band-limited round trip)
are recomputed with the physical constant.  Both evaluations are reported side
by side; a passing number does not rewrite the predeclared verdict, it
*attributes* it.

Usage::

    PYTHONPATH=src python3 scripts/cummins_closure_ainf_independent.py --stage cache
    PYTHONPATH=src python3 scripts/cummins_closure_ainf_independent.py --stage audit
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

from tensorbem.added_mass_limit import added_mass_limit, shift_added_mass
from tensorbem.benchmarks import case_bem_cummins_wigley as anchor_case
from tensorbem.benchmarks import cummins_wigley_kk_causality as case
from tensorbem.causality import kkr_added_mass
from tensorbem.wigley3_freq import (
    WIGLEY3_B,
    WIGLEY3_L,
    WIGLEY3_T,
    wigley3_wetted_hull_mesh,
)

CASE_DIR = pathlib.Path("benchmarks/verified/casepool_bem_cummins_wigley")
SWEEP_NPZ = CASE_DIR / "sweep_closure_medium800.npz"
OUT_JSON = CASE_DIR / "ainf_independent_gates.json"

#: Image-mesh ladder: the sweep is solved on ``(20, 10)`` (800 panels), the two
#: coarser/finer rungs bracket it so the ``A_inf`` value carries an interval.
LADDER = ((20, 6), (20, 10), (30, 12))

#: The frame check re-solves the image problem about the sweep's own centre and
#: compares it with the waterline solve pushed through the re-reference.
FRAME_CHECK_MESH = (20, 10)

#: ``A_inf`` probe for the admissibility window, as a fraction of the band-top
#: added mass of each DOF (a scale-free sweep, no cherry-picked endpoints).
SCAN_FRACTIONS = np.linspace(0.80, 1.20, 81)


def parse_ladder(spec: str) -> tuple[tuple[int, int], ...]:
    """``"20x6,20x10"`` -> ``((20, 6), (20, 10))``."""
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        a, _, b = chunk.partition("x")
        out.append((int(a), int(b)))
    if not out:
        raise SystemExit("empty mesh ladder")
    return tuple(out)


def _log(message: str) -> None:
    print(f"[ainf] {message}", flush=True)


# ---------------------------------------------------------------------------
# stage 1: freeze the closure sweep in-tree
# ---------------------------------------------------------------------------
def cache_sweep(points_dir: pathlib.Path, level: str) -> dict:
    """Assemble the closure sweep from the single-frequency cache and freeze it.

    The 69 point solves live in a per-mission artifacts directory; the gates
    cannot be re-judged later without them, so they are frozen as one ``npz``
    with their provenance (mesh, rotation centre, per-point linear residual).
    """
    points = case.load_points(points_dir, level)
    omega = np.asarray(sorted(points), dtype=float)
    sweep = case.sweep_from_points(points, omega)
    added = np.asarray(sweep["added_mass"], dtype=float)
    damping = np.asarray(sweep["damping"], dtype=float)
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        SWEEP_NPZ,
        omega=omega,
        added_mass=added,
        damping=damping,
        relative_residual=np.asarray(sweep["relative_residual"], dtype=float),
        wall_seconds=np.asarray(sweep["wall_seconds"], dtype=float),
        rotation_center=np.asarray(sweep["rotation_center"], dtype=float),
        level=np.asarray(sweep["level"]),
        level_key=np.asarray(sweep["level_key"]),
        density=np.asarray(float(sweep["density"])),
        gravity=np.asarray(float(sweep["gravity"])),
        source=np.asarray(str(points_dir)),
    )
    _log(f"froze {omega.size} points -> {SWEEP_NPZ}")
    return {"n_points": int(omega.size), "omega_min": float(omega[0]),
            "omega_max": float(omega[-1]), "path": str(SWEEP_NPZ)}


def load_sweep() -> dict:
    if not SWEEP_NPZ.exists():
        raise SystemExit(f"missing {SWEEP_NPZ}; run --stage cache first")
    with np.load(SWEEP_NPZ) as data:
        omega = data["omega"]
        added = data["added_mass"]
        damping = data["damping"]
        return {
            "omega": omega.tolist(),
            "added_mass": added.tolist(),
            "damping": damping.tolist(),
            "relative_residual": data["relative_residual"].tolist(),
            "rotation_center": data["rotation_center"].tolist(),
            "level": str(data["level"]),
            "level_key": str(data["level_key"]),
            "density": float(data["density"]),
            "gravity": float(data["gravity"]),
        }


# ---------------------------------------------------------------------------
# stage 2: the independent A_inf, its uncertainty and its frame
# ---------------------------------------------------------------------------
IMAGE_CACHE = pathlib.Path("artifacts/ainf_independent_cache")


def _cache_path(n_sections: int, n_levels: int, center) -> pathlib.Path:
    tag = "_".join(f"{float(c):+.5f}" for c in center)
    return IMAGE_CACHE / f"image_{n_sections}x{n_levels}_c{tag}.npz"


def image_added_mass(n_sections: int, n_levels: int, center, *, use_cache: bool = True) -> dict:
    """``A(omega -> inf)`` of the six modes from the antisymmetric-image solve.

    The solve is cached on disk keyed by ``(mesh, centre)``: the ladder is the
    expensive part of this audit, and a re-run must not pay for it twice.
    """
    path = _cache_path(n_sections, n_levels, center)
    if use_cache and path.exists():
        with np.load(path) as data:
            return {
                "center": [float(v) for v in center],
                "n_sections": int(data["n_sections"]),
                "n_levels": int(data["n_levels"]),
                "panel_count": int(data["panel_count"]),
                "doubled_panel_count": int(data["doubled_panel_count"]),
                "matrix": data["matrix"],
                "max_relative_residual_l2": float(data["max_relative_residual_l2"]),
                "symmetry_error": float(data["symmetry_error"]),
                "condition_number": float(data["condition_number"]),
                "wall_seconds": float(data["wall_seconds"]),
                "cached": True,
            }
    mesh = wigley3_wetted_hull_mesh(
        WIGLEY3_L, WIGLEY3_B, WIGLEY3_T, n_sections=n_sections, n_levels=n_levels
    )
    started = time.perf_counter()
    result = added_mass_limit(mesh, "infinite_frequency", center=center,
                              density=case.DENSITY)
    payload = {
        "center": [float(v) for v in center],
        "n_sections": int(n_sections),
        "n_levels": int(n_levels),
        "panel_count": int(mesh.panel_count),
        "doubled_panel_count": int(result.doubled_panel_count),
        "matrix": np.asarray(result.added_mass_matrix, dtype=float),
        "max_relative_residual_l2": float(result.max_relative_residual_l2),
        "symmetry_error": float(result.symmetry_error),
        "condition_number": float(result.condition_number),
        "wall_seconds": time.perf_counter() - started,
        "cached": False,
    }
    if use_cache:
        IMAGE_CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            matrix=payload["matrix"],
            n_sections=n_sections,
            n_levels=n_levels,
            panel_count=payload["panel_count"],
            doubled_panel_count=payload["doubled_panel_count"],
            max_relative_residual_l2=payload["max_relative_residual_l2"],
            symmetry_error=payload["symmetry_error"],
            condition_number=payload["condition_number"],
            wall_seconds=payload["wall_seconds"],
        )
    return payload


def ainf_anchor(center, ladder=LADDER, frame_mesh=FRAME_CHECK_MESH) -> dict:
    """Image ``A_inf`` ladder, re-expressed about ``center`` (the sweep's frame)."""
    rows = []
    for n_sections, n_levels in ladder:
        row = image_added_mass(n_sections, n_levels, anchor_case.CENTER)
        row["shifted_matrix"] = shift_added_mass(
            row.pop("matrix"), anchor_case.CENTER, center
        )
        row["diagonal"] = np.diag(row["shifted_matrix"]).tolist()
        rows.append(row)
        _log(f"ladder {(n_sections, n_levels)}: panels={row['panel_count']} "
             f"wall={row['wall_seconds']:.1f}s")

    # Frame check: a direct solve about the sweep's centre must equal the
    # waterline solve pushed through the rigid-body re-reference.
    target = tuple(frame_mesh)
    direct = image_added_mass(target[0], target[1], center)
    reference_row = next(row for row in rows
                         if (row["n_sections"], row["n_levels"]) == target)
    shifted = np.asarray(reference_row["shifted_matrix"], dtype=float)
    frame_error = float(np.max(np.abs(direct["matrix"] - shifted))
                        / max(float(np.max(np.abs(direct["matrix"]))), 1e-30))
    direct_diagonal = np.diag(direct["matrix"]).tolist()

    diagonal_ladder = np.asarray([row["diagonal"] for row in rows], dtype=float)
    finest = diagonal_ladder[-1]
    spread = diagonal_ladder.max(axis=0) - diagonal_ladder.min(axis=0)
    return {
        "method": "antisymmetric image (phi = 0 on the calm waterplane) about z = 0, "
                  "re-expressed about the sweep rotation centre",
        "anchor_center": list(anchor_case.CENTER),
        "sweep_center": [float(v) for v in center],
        "ladder": [{k: v for k, v in row.items() if k != "shifted_matrix"} for row in rows],
        "diagonal_at_sweep_center_finest": finest.tolist(),
        "ladder_spread": spread.tolist(),
        "ladder_spread_relative": (spread / np.maximum(np.abs(finest), 1e-30)).tolist(),
        "interval": [[float(lo), float(hi)] for lo, hi in
                     zip(diagonal_ladder.min(axis=0), diagonal_ladder.max(axis=0))],
        "frame_check": {
            "mesh": list(target),
            "max_relative_difference_direct_vs_transform": frame_error,
            "direct_diagonal": direct_diagonal,
            "transformed_diagonal": reference_row["diagonal"],
            "identity_check_diagonal_only": bool(
                np.allclose(direct_diagonal, np.asarray(reference_row["diagonal"]))),
        },
        "infinite_frequency_sphere_self_check": anchor_case.anchor_self_check(),
    }


# ---------------------------------------------------------------------------
# stage 3: the gates, twice
# ---------------------------------------------------------------------------
def _trim(study: dict) -> dict:
    """Drop the heavy kernel/profile arrays, keep every scalar the report uses."""
    keep = ("dof", "dof_label", "added_mass_inf", "added_mass_inf_source",
            "added_mass_inf_tail_fit", "added_mass_inf_tail_fraction",
            "kernel_diagnostics")
    out = {key: study[key] for key in keep if key in study}
    out["G2"] = {key: study["G2"][key]
                 for key in ("rel_err", "K_at_zero_sampled", "K_at_zero_integral")}
    out["G3"] = {key: study["G3"][key]
                 for key in ("rel_err", "rms", "interior_max", "argmax_omega")}
    out["G3b"] = {key: study["G3b"][key] for key in ("rel_err", "scale")}
    out["G5"] = {"rel_err": study["G5"]["rel_err"], "band": study["G5"]["band"]}
    return out


def ainf_scan(sweep: dict, dof: int, reference: float) -> dict:
    """``G3`` and ``G5`` as a function of the assumed ``A_inf`` for one DOF.

    The predeclared statistics both carry ``A_inf`` (``G3`` also divides by
    ``max|A - A_inf|``), so the admissibility window of each gate is a
    well-defined interval in ``A_inf``; the physical estimate is then either
    inside it or not.  No value is chosen by looking at the answer.
    """
    omega, added, damping = case._as_diag(sweep, dof)
    kk_omega = 0.5 * (omega[:-1] + omega[1:])
    measured = 0.5 * (added[:-1] + added[1:])
    values = np.unique(np.concatenate([reference * SCAN_FRACTIONS, [reference]]))
    rows = []
    for a_inf in values:
        pred = np.array([kkr_added_mass(omega, damping, float(w), added_mass_inf=float(a_inf))
                         for w in kk_omega])
        scale = max(float(np.max(np.abs(measured - a_inf))), 1e-30)
        rows.append(float(np.max(np.abs(pred - measured)) / scale))
    residual = np.asarray(rows, dtype=float)
    order = np.argsort(values)

    def window(mask) -> list[list[float]] | None:
        if not mask.any():
            return None
        segments, start = [], None
        for index, flag in enumerate(mask):
            if flag and start is None:
                start = index
            elif not flag and start is not None:
                segments.append([float(values[start]), float(values[index - 1])])
                start = None
        if start is not None:
            segments.append([float(values[start]), float(values[-1])])
        return segments

    return {
        "dof": int(dof),
        "dof_label": case.DOF_LABELS[dof],
        "reference_ainf": float(reference),
        "fractions": [float(v / reference) for v in values[order]],
        "ainf": values[order].tolist(),
        "G3": residual[order].tolist(),
        "best_ainf": float(values[int(np.argmin(residual))]),
        "best_G3": float(np.min(residual)),
        "G3_pass_window": window(residual <= case.GATES["G3_kramers_kronig"]["tol"]),
    }


def evaluate(sweep: dict, anchor: dict) -> dict:
    centre = tuple(float(v) for v in sweep["rotation_center"])
    sigma = np.asarray(anchor["interval"], dtype=float)
    rows = []
    for dof in range(6):
        baseline = _trim(case.analysis_gates(sweep, dof))
        anchored = _trim(case.analysis_gates(
            sweep, dof, added_mass_inf=float(anchor["diagonal_at_sweep_center_finest"][dof])))
        omega, added, _ = case._as_diag(sweep, dof)
        top_two = added[-2:]
        rows.append({
            "dof": dof,
            "dof_label": case.DOF_LABELS[dof],
            "a_inf_tail_fit": baseline["added_mass_inf_tail_fit"],
            "a_inf_image": float(anchor["diagonal_at_sweep_center_finest"][dof]),
            "a_inf_image_interval": [float(sigma[dof, 0]), float(sigma[dof, 1])],
            "a_inf_relative_change": float(
                (anchor["diagonal_at_sweep_center_finest"][dof]
                 - baseline["added_mass_inf_tail_fit"])
                / max(abs(baseline["added_mass_inf_tail_fit"]), 1e-30)),
            "band_top": {"omega_max": float(omega[-1]), "A_at_last": float(top_two[-1]),
                         "A_at_second_last": float(top_two[0]),
                         "rising_at_band_top": bool(top_two[-1] > top_two[0]),
                         "tail_fit_below_band_top": bool(
                             baseline["added_mass_inf_tail_fit"] < top_two[-1])},
            "baseline": baseline,
            "anchored": anchored,
            "G3_tail": baseline["G3"]["rel_err"],
            "G3_image": anchored["G3"]["rel_err"],
            "G5_tail": baseline["G5"]["rel_err"],
            "G5_image": anchored["G5"]["rel_err"],
            "scan": ainf_scan(sweep, dof, float(anchor["diagonal_at_sweep_center_finest"][dof])),
        })
        _log(f"{case.DOF_LABELS[dof]:6s} A_inf tail={baseline['added_mass_inf_tail_fit']:9.3f} "
             f"image={anchor['diagonal_at_sweep_center_finest'][dof]:9.3f} | "
             f"G3 {baseline['G3']['rel_err']:.4f} -> {anchored['G3']['rel_err']:.4f} | "
             f"G5 {baseline['G5']['rel_err']:.4f} -> {anchored['G5']['rel_err']:.4f}")

    # G4 travels through A_inf too: it is the instantaneous inertia of the
    # Cummins equation, so the same replacement changes the time-domain decay.
    heave_study_tail = case.analysis_gates(sweep, case.DOF_INDEX["heave"])
    heave_study_image = case.analysis_gates(
        sweep, case.DOF_INDEX["heave"],
        added_mass_inf=float(anchor["diagonal_at_sweep_center_finest"][2]))
    g4_tail = case.heave_decay_gate(heave_study_tail, sweep)
    g4_image = case.heave_decay_gate(heave_study_image, sweep)

    def verdict(name: str) -> dict:
        tol = case.GATES[name]["tol"]
        return {"tol": tol,
                "tail_passed": bool(max(row["G3_tail"] if name.startswith("G3")
                                        else row["G5_tail"] for row in rows) <= tol),
                "image_passed": bool(max(row["G3_image"] if name.startswith("G3")
                                         else row["G5_image"] for row in rows) <= tol)}

    g3 = verdict("G3_kramers_kronig")
    g5 = verdict("G5_round_trip")
    g4 = {"tol": case.GATES["G4_free_decay"]["tol"],
          "tail_rel_err": g4_tail["rel_err"], "image_rel_err": g4_image["rel_err"],
          "tail_passed": bool(abs(g4_tail["rel_err"]) <= case.GATES["G4_free_decay"]["tol"]),
          "image_passed": bool(abs(g4_image["rel_err"]) <= case.GATES["G4_free_decay"]["tol"])}
    worst_g3 = max(rows, key=lambda row: row["G3_image"])
    return {
        "rotation_center": list(centre),
        "rows": rows,
        "G3": g3,
        "G5": g5,
        "G4": g4,
        "G4_detail": {"tail": {k: g4_tail[k] for k in ("zeta_decay", "zeta_half_power",
                                                       "added_mass_inf", "rel_err")},
                      "image": {k: g4_image[k] for k in ("zeta_decay", "zeta_half_power",
                                                         "added_mass_inf", "rel_err")}},
        "verdict": {
            "gates": ["G3_kramers_kronig", "G4_free_decay", "G5_round_trip"],
            "as_declared_tail_fit": "FAIL",
            "with_image_ainf": "FAIL" if not (g3["image_passed"] and g4["image_passed"]
                                              and g5["image_passed"]) else "PASS",
            "worst_G3_dof_with_image": worst_g3["dof_label"],
            "worst_G3_with_image": worst_g3["G3_image"],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cache", "audit", "all"), default="all")
    parser.add_argument("--points-dir",
                        default="/root/tp-agent/work/m23130/artifacts/"
                                "case_bem_cummins_wigley_ext/points")
    parser.add_argument("--level", default="medium")
    parser.add_argument("--ladder", default=",".join(f"{a}x{b}" for a, b in LADDER),
                        help="image mesh ladder, e.g. 20x6,20x10,30x12")
    parser.add_argument("--frame-mesh", default="20x10",
                        help="mesh solved directly about the sweep centre as a frame check")
    parser.add_argument("--out", default=str(OUT_JSON))
    args = parser.parse_args(argv)

    started = time.perf_counter()
    cache_info = None
    if args.stage in ("cache", "all"):
        cache_info = cache_sweep(pathlib.Path(args.points_dir), args.level)
    if args.stage == "cache":
        print(json.dumps(cache_info, indent=1))
        return 0

    sweep = load_sweep()
    centre = tuple(float(v) for v in sweep["rotation_center"])
    _log(f"sweep: {len(sweep['omega'])} points {sweep['omega'][0]:g}..{sweep['omega'][-1]:g} "
         f"rad/s, centre {centre}")
    anchor = ainf_anchor(centre, ladder=parse_ladder(args.ladder),
                         frame_mesh=parse_ladder(args.frame_mesh)[0])
    evaluation = evaluate(sweep, anchor)
    payload = {
        "case": "CASE-BEM-CUMMINS-WIGLEY",
        "question": "do the failing closure-band gates (G3/G4/G5) survive when the "
                    "added-mass asymptote is taken from the independent image solve "
                    "instead of the in-band tail fit?",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sweep": {
            "path": str(SWEEP_NPZ),
            "n_points": len(sweep["omega"]),
            "omega_min": sweep["omega"][0],
            "omega_max": sweep["omega"][-1],
            "level": sweep["level"],
            "max_linear_residual": max(sweep["relative_residual"]),
            "cache": cache_info,
        },
        "gate_definitions": {name: case.GATES[name] for name in
                             ("G3_kramers_kronig", "G4_free_decay", "G5_round_trip")},
        "a_inf_anchor": anchor,
        "evaluation": evaluation,
        "wall_seconds": time.perf_counter() - started,
    }
    out = pathlib.Path(args.out)
    out.write_text(json.dumps(payload, indent=1))
    _log(f"wrote {out} in {payload['wall_seconds']:.1f}s")
    print(json.dumps(payload["evaluation"]["verdict"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())