"""Regenerate the CASE-BEM-CUMMINS-WIGLEY published verdict from the committed
sweep artifacts (verification-debt fix).

The committed ``case_result.json`` was published from an older sweep whose
Kramers-Kronig residual could not be certified (``worst_relative_residual =
inf``) and whose ``mesh_convergence`` used an older ``coarse_vs_medium``
schema, while the committed ``sweep_medium.npz`` / ``sweep_coarse.npz`` were
later refreshed to a converged, CHIEF-regularised solve that gives a finite
residual (``C3 = 0.036``, well inside the 10 % case tolerance).  The two were
therefore out of sync and ``test_production_artifact_replays_to_the_published_verdict``
failed.

Every gate in this case is a *function of the tabulated coefficients* plus the
hull constants (see ``gates_from_artifact``), so the verdict is re-derivable
from the committed ``.npz`` alone without re-solving the BEM system.  This
script rebuilds ``case_result.json`` / ``CASE.md`` from the committed sweeps so
the published verdict matches the committed evidence exactly.
"""

from __future__ import annotations

import json
import pathlib
import platform
import time

import numpy as np

from tensorbem.benchmarks import case_bem_cummins_wigley as case

ARTIFACT_DIR = pathlib.Path(
    "benchmarks/verified/casepool_bem_cummins_wigley"
).resolve()


def main() -> None:
    coarse = case.load_sweep(ARTIFACT_DIR / "sweep_coarse.npz")
    medium = case.load_sweep(ARTIFACT_DIR / "sweep_medium.npz")

    results: dict = {
        "case": case.CASE_SLUG,
        "mission": "m21100",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "hull": {
            "form": "mathematical Wigley hull y = (B/2)(1-(2x/L)^2)(1-(z/T)^2)",
            "length_m": case.LENGTH,
            "beam_m": case.BEAM,
            "draft_m": case.DRAFT,
            "density_kg_m3": case.DENSITY,
            "gravity_m_s2": case.GRAVITY,
            "block_coefficient_analytic": case.CB_ANALYTIC,
            "displaced_volume_analytic_m3": case.VOLUME_ANALYTIC,
        },
        "settings": {
            "band_rad_s": [float(case.BAND[0]), float(case.BAND[1])],
            "n_omega": int(medium.omega.size),
            "refine_peaks": 0,
            "meshes": {label: list(case.MESHES[label]) for label in ("coarse", "medium")},
            "chief_points": "auto",
            "tolerances": case.TOLERANCES,
            "free_decay_dof": case.DOF_NAMES[2],
            "dt": case.DT,
            "duration": case.DURATION,
            "tau_max": case.TAU_MAX,
        },
        "sweeps": {
            "coarse": coarse.summary(),
            "medium": medium.summary(),
        },
    }

    # --- gates, exactly as run_case would assemble them from a solved sweep ---
    results["gates"] = {"C1": case.geometry_gate()}
    production = medium
    results["gates"]["C0"] = case.solver_validity_gate(production)
    results["gates"]["C0_coarse"] = case.solver_validity_gate(coarse)
    if not results["gates"]["C0"]["passed"]:
        for key in ("C2", "C3", "C4", "C5", "C6a", "C6b"):
            results["gates"][key] = {
                "id": key,
                "title": f"{key} voided: the sweep failed gate C0",
                "threshold": "not evaluated",
                "passed": False,
                "voided": True,
                "values": {"voided_by": "C0", "note": "the BIE solve did not converge"},
            }
    else:
        results["gates"]["C2"] = case.panel_adequacy_gate(production)
        results["gates"]["C3"] = case.kk_gate(production)
        results["gates"]["C3_coarse"] = case.kk_gate(coarse)
        results["gates"]["C4"] = case.round_trip_gate(production, dof=2)
        results["gates"]["C5"] = case.free_decay_gate(production, dof=2)
        results["gates"]["C6a"] = case.negative_control_gate(production, dof=2)
        results["gates"]["C6b"] = case.kk_gate(
            production,
            corrupt=True,
            window=[
                all(row["decidable_mask"][k] for row in results["gates"]["C3"]["values"]["per_dof"])
                for k in range(len(results["gates"]["C3"]["values"]["per_dof"][0]["decidable_mask"]))
            ],
        )
        results["mesh_convergence"] = case._mesh_convergence(coarse, production)

    results["verdict"] = case.case_verdict(results["gates"])

    # --- figure (KK closure residual per DOF) ---
    figure_name = case._make_figure(
        {"coarse": coarse, "medium": medium}, results, ARTIFACT_DIR
    )
    results["figure"] = figure_name

    (ARTIFACT_DIR / "case_result.json").write_text(
        json.dumps(results, indent=2, sort_keys=False), encoding="utf-8"
    )
    (ARTIFACT_DIR / "CASE.md").write_text(
        case.render_markdown(results), encoding="utf-8"
    )

    print("verdict:", json.dumps(results["verdict"], indent=2))
    for k, v in results["gates"].items():
        print(
            f"  {k}: passed={v.get('passed')} "
            f"worst={v.get('values', {}).get('worst_relative_residual')}"
        )
    print("figure:", figure_name)


if __name__ == "__main__":
    main()
