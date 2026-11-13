"""算例 CASE-BEM-SUBMERGED-SPHERE-HAVELOCK: submerged-sphere radiation, frequency domain.

Case-pool wrapper around the P190-N03 studies in
:mod:`tensorbem.benchmarks.submerged_sphere_havelock`.  It runs

* the panel-refinement study (icosphere subdivisions 1/2/3 = 80/320/1280 panels),
* the frequency sweep ``ka = 0.1 ... 5.0`` at the production mesh (320 panels),
* the low-frequency ``A33'`` extrapolation (``ka -> 0``),
* a submergence sweep ``f/a = 1.5 / 2.0 / 3.0`` at ``ka = 1.0``, and
* the reference-integrity audit of the multipole reference,

then judges every item against the case criterion.

Pre-declared criteria (fixed before the numbers are inspected, no adaptation)
-----------------------------------------------------------------------------
``primary`` (the case tolerance): for the production mesh (320 panels, ``f/a =
2.0``) every radiation coefficient in the sweep -- ``A33'``, ``B33'``, ``A11'``,
``B11'`` -- must satisfy ``|got - ref| / |ref| < 5%``.  The same 5% band is
applied to every refinement level, every submergence ratio and to the
extrapolated low-frequency ``A33'``.
``supplementary`` (interpretation only, *never* used to flip a verdict): the
error normalised by the largest ``|ref|`` met in the same sweep,
``|got - ref| / max|ref|``, so that the very small high-``ka`` damping values
(``B33' ~ 1e-6``) are not judged by a relative metric that is dominated by
discretisation noise.
``stopping rule``: one production mesh, one sweep; no parameter is refined or
dropped after the errors are seen.

Reference provenance (zero transcription)
-----------------------------------------
The target values are *evaluated*, never copied: they come from the closed-form
truncated multipole solution :func:`tensorbem.analytic.submerged_sphere_wave_multipole`
(Havelock's frequency-domain multipole method -- Havelock 1955, *Proc. R. Soc.
Lond. A* 231, 1-13; Thorne 1953; Wang 1986; Linton 1991), whose truncation is
shown to be converged and which reproduces the classical unbounded-fluid value
``A' = 0.5`` / ``C_M = 1.5`` (Lamb 1932 art. 92) in the deep-submergence limit.
See :mod:`tensorbem.benchmarks.submerged_sphere_havelock_integrity`.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorbem.analytic import submerged_sphere_wave_multipole
from tensorbem.benchmarks import submerged_sphere_havelock_integrity as integrity
from tensorbem.benchmarks.submerged_sphere_havelock import (
    run_convergence_study,
    run_frequency_sweep,
    run_low_frequency_limit,
)
from tensorbem.free_surface_solver3d import (
    assemble_free_surface_operators_3d,
    solve_radiation_3d,
)
from tensorbem.geometry3d import submerged_sphere_mesh

CASE_TOLERANCE = 0.05
#: ``A / (rho * volume)`` normalisation used by every coefficient below.
COEFFICIENTS = ("a33", "b33", "a11", "b11")


def _bem_coefficients(rad: Any, ref: Any, density: float) -> dict[str, float]:
    volume = ref.displaced_volume
    omega = ref.omega
    return {
        "a33": float(rad.added_mass[2, 2] / (density * volume)),
        "b33": float(rad.damping[2, 2] / (density * omega * volume)),
        "a11": float(rad.added_mass[0, 0] / (density * volume)),
        "b11": float(rad.damping[0, 0] / (density * omega * volume)),
    }


def _reference_coefficients(ref: Any) -> dict[str, float]:
    return {
        "a33": float(ref.heave_added_mass_dimless),
        "b33": float(ref.heave_damping_dimless),
        "a11": float(ref.surge_added_mass_dimless),
        "b11": float(ref.surge_damping_dimless),
    }


def run_submergence_sweep(
    submergence_ratios: Sequence[float] = (1.5, 2.0, 3.0),
    ka: float = 1.0,
    radius: float = 1.0,
    subdivision: int = 2,
    density: float = 1025.0,
    gravity: float = 9.80665,
) -> list[dict[str, Any]]:
    """BEM vs multipole over the submergence ratio ``f/a`` at fixed ``ka``."""
    rows: list[dict[str, Any]] = []
    for ratio in submergence_ratios:
        submergence = float(ratio) * radius
        nu = float(ka) / radius
        ref = submerged_sphere_wave_multipole(
            ka, radius=radius, submergence=submergence,
            density=density, gravity=gravity,
        )
        mesh = submerged_sphere_mesh(subdivision, radius=radius, submergence=submergence)
        ops = assemble_free_surface_operators_3d(mesh, nu)
        rad = solve_radiation_3d(
            mesh, nu, center=(0.0, 0.0, -submergence),
            density=density, gravity=gravity, operators=ops,
        )
        got = _bem_coefficients(rad, ref, density)
        exact = _reference_coefficients(ref)
        rows.append(
            {
                "submergence_ratio": float(ratio),
                "ka": float(ka),
                "n_panels": mesh.panel_count,
                "h_mean": mesh.h_mean,
                "got": got,
                "ref": exact,
                "rel_error": {k: abs(got[k] - exact[k]) / max(abs(exact[k]), 1.0e-30) for k in COEFFICIENTS},
                "condition_number": rad.condition_number,
                "relative_residual": rad.relative_residual,
            }
        )
    return rows


def run_area_deficit_diagnostic(
    convergence: dict[str, Any],
    radius: float = 1.0,
    submergence: float = 2.0,
) -> dict[str, Any]:
    """EXPLORATORY (post-hoc): is the BEM bias explained by the polyhedral area deficit?

    An icosphere of subdivision ``s`` is an *inscribed* polyhedron: its wetted
    area ``A_poly`` is smaller than the sphere's ``4 pi a**2`` by
    ``deficit = 1 - A_poly/(4 pi a**2)``.  The BEM integrates the body boundary
    condition over that polyhedron, so the radiation coefficients inherit the
    deficit.  This diagnostic correlates the measured relative errors with the
    deficit level by level and reports the ratio.

    The hypothesis was formed *after* the errors were seen and the ratios are
    fitted on this single case (three refinement levels); the numbers here are
    descriptive/predictive for this geometry and must not be used to relax the
    pre-declared 5% criterion.
    """
    sphere_area = 4.0 * np.pi * radius**2
    rows = []
    for level in convergence["levels"]:
        mesh = submerged_sphere_mesh(
            level["subdivision"], radius=radius, submergence=submergence
        )
        verts = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        tri = 0.5 * np.linalg.norm(
            np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]]),
            axis=1,
        )
        poly_area = float(tri.sum())
        deficit = 1.0 - poly_area / sphere_area
        rows.append(
            {
                "subdivision": level["subdivision"],
                "n_panels": level["n_panels"],
                "polyhedral_area": poly_area,
                "sphere_area": sphere_area,
                "area_deficit": deficit,
                "a33_rel_error": level["a33_rel_error"],
                "b33_rel_error": level["b33_rel_error"],
                "a11_rel_error": level["a11_rel_error"],
                "b11_rel_error": level["b11_rel_error"],
                "ratio_a33_over_deficit": level["a33_rel_error"] / deficit if deficit else None,
                "ratio_b33_over_deficit": level["b33_rel_error"] / deficit if deficit else None,
                "ratio_a11_over_deficit": level["a11_rel_error"] / deficit if deficit else None,
                "ratio_b11_over_deficit": level["b11_rel_error"] / deficit if deficit else None,
            }
        )
    ratios_a = [r["ratio_a33_over_deficit"] for r in rows if r["ratio_a33_over_deficit"]]
    ratios_b = [r["ratio_b33_over_deficit"] for r in rows if r["ratio_b33_over_deficit"]]
    # Predict, from the measured ratios, the smallest subdivision whose damping
    # error would fall inside the pre-declared band.
    predicted = None
    if ratios_b:
        b_ratio = float(np.mean(ratios_b))
        for row in rows:
            if b_ratio * row["area_deficit"] < CASE_TOLERANCE:
                predicted = {
                    "subdivision": row["subdivision"],
                    "n_panels": row["n_panels"],
                    "predicted_b33_rel_error": b_ratio * row["area_deficit"],
                }
                break
    return {
        "status": "exploratory_post_hoc",
        "hypothesis": (
            "the radiation-coefficient bias of the inscibed icosphere is proportional to the "
            "polyhedral area deficit 1 - A_poly/(4 pi a^2)"
        ),
        "rows": rows,
        "ratio_a33_over_deficit_mean": float(np.mean(ratios_a)) if ratios_a else None,
        "ratio_b33_over_deficit_mean": float(np.mean(ratios_b)) if ratios_b else None,
        "smallest_level_inside_tolerance_by_rule": predicted,
    }


def _verdicts(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per-item and aggregate pass/fail against :data:`CASE_TOLERANCE`."""
    failures = [
        {"item": it["item"], "quantity": it["quantity"], "rel_error": it["rel_error"]}
        for it in items
        if not it["rel_error"] < CASE_TOLERANCE
    ]
    worst = max(items, key=lambda it: it["rel_error"]) if items else None
    return {
        "tolerance": CASE_TOLERANCE,
        "n_items": len(items),
        "n_failures": len(failures),
        "passed": not failures,
        "failures": failures,
        "worst_item": (
            {
                "item": worst["item"],
                "quantity": worst["quantity"],
                "rel_error": worst["rel_error"],
                "got": worst["got"],
                "ref": worst["ref"],
            }
            if worst
            else None
        ),
    }


def _collect_items(
    convergence: dict[str, Any], sweep: Sequence[dict[str, Any]], low_freq: dict[str, Any],
    submergence: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for level in convergence["levels"]:
        for key in COEFFICIENTS:
            items.append(
                {
                    "item": "refinement/sub%d" % level["subdivision"],
                    "quantity": key,
                    "got": level["%s_bem" % key],
                    "ref": level["%s_exact" % key],
                    "rel_error": level["%s_rel_error" % key],
                }
            )
    for row in sweep:
        for key in COEFFICIENTS:
            items.append(
                {
                    "item": "sweep/ka=%.3g" % row["ka"],
                    "quantity": key,
                    "got": row["%s_bem" % key],
                    "ref": row["%s_exact" % key],
                    "rel_error": row["%s_rel_error" % key],
                }
            )
    items.append(
        {
            "item": "low_frequency/extrapolated_a33",
            "quantity": "a33",
            "got": low_freq["bem_extrapolated_limit"],
            "ref": low_freq["exact_limit"],
            "rel_error": low_freq["limit_relative_error"],
        }
    )
    for row in submergence:
        for key in COEFFICIENTS:
            items.append(
                {
                    "item": "submergence/f_over_a=%.2f" % row["submergence_ratio"],
                    "quantity": key,
                    "got": row["got"][key],
                    "ref": row["ref"][key],
                    "rel_error": row["rel_error"][key],
                }
            )
    return items


def _supplementary_scale_aware(
    sweep: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Error normalised by ``max|ref|`` in the same sweep (interpretation only)."""
    rows: list[dict[str, Any]] = []
    for key in COEFFICIENTS:
        scale = max(abs(row["%s_exact" % key]) for row in sweep)
        for row in sweep:
            rows.append(
                {
                    "quantity": key,
                    "ka": row["ka"],
                    "reference_scale": scale,
                    "abs_error": abs(row["%s_bem" % key] - row["%s_exact" % key]),
                    "scale_normalised_error": abs(row["%s_bem" % key] - row["%s_exact" % key]) / scale,
                }
            )
    return rows


def run_case(
    *,
    quick: bool = False,
    radius: float = 1.0,
    submergence: float = 2.0,
    density: float = 1025.0,
    gravity: float = 9.80665,
    subdivisions: Sequence[int] = (1, 2, 3),
    ka_values: Sequence[float] | None = None,
    ka_low: Sequence[float] | None = None,
    submergence_ratios: Sequence[float] = (1.5, 2.0, 3.0),
) -> dict[str, Any]:
    """Run the whole case and return the JSON-ready artifact."""
    ka_values = tuple(ka_values) if ka_values is not None else (
        (0.5, 1.0, 2.0) if quick else (0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
    )
    ka_low = tuple(ka_low) if ka_low is not None else ((0.05, 0.1) if quick else (0.01, 0.02, 0.05, 0.1, 0.2))
    if quick:
        subdivisions = tuple(s for s in subdivisions if s <= 2)

    convergence = run_convergence_study(
        ka=0.8, radius=radius, submergence=submergence, density=density, gravity=gravity,
        subdivisions=tuple(subdivisions),
    )

    sweep = run_frequency_sweep(
        ka_values=ka_values, radius=radius, submergence=submergence,
        density=density, gravity=gravity,
    )
    low_freq = run_low_frequency_limit(
        ka_values=ka_low, radius=radius, submergence=submergence,
        density=density, gravity=gravity,
    )
    sub_sweep = run_submergence_sweep(
        submergence_ratios=submergence_ratios, ka=1.0, radius=radius,
        density=density, gravity=gravity,
    )
    ref_integrity = integrity.run(
        ka_values=ka_values, radius=radius, submergence=submergence,
        density=density, gravity=gravity,
    )

    deficit = run_area_deficit_diagnostic(
        convergence, radius=radius, submergence=submergence
    )

    items = _collect_items(convergence, sweep, low_freq, sub_sweep)
    production = [
        it for it in items if it["item"].startswith("sweep/")
    ]
    return {
        "case": "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
        "benchmark": "submerged_sphere_havelock_radiation_case",
        "geometry": {
            "body": "sphere",
            "radius": radius,
            "submergence": submergence,
            "submergence_ratio": submergence / radius,
            "density": density,
            "gravity": gravity,
            "displaced_volume": (4.0 / 3.0) * np.pi * radius**3,
        },
        "nondimensional_parameters": {
            "ka": list(ka_values),
            "submergence_ratio_f_over_a": [submergence / radius] + [float(r) for r in submergence_ratios],
            "normalisation": "A' = A/(rho*volume), B' = B/(rho*omega*volume)",
            "panel_subdivisions": list(subdivisions),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "criteria": {
            "primary": "per-item relative error |got-ref|/|ref| < 5%% (production mesh 320 panels)",
            "supplementary": "|got-ref|/max|ref| in the same sweep (interpretation only)",
            "stopping_rule": "one production mesh, one sweep, no adaptation after seeing errors",
        },
        "convergence_study": convergence,
        "frequency_sweep": sweep,
        "low_frequency_limit": low_freq,
        "submergence_sweep": sub_sweep,
        "reference_integrity": ref_integrity,
        "area_deficit_diagnostic": deficit,
        "judgement": {
            "all_items": _verdicts(items),
            "production_mesh": _verdicts(production),
        },
        "supplementary_scale_aware_errors": _supplementary_scale_aware(sweep),
    }


def main() -> None:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--submergence", type=float, default=2.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_case(quick=args.quick, radius=args.radius, submergence=args.submergence)
    text = json.dumps(result, indent=2, default=float)
    print(text)
    if args.output:
        args.output.write_text(text)


if __name__ == "__main__":  # pragma: no cover
    main()