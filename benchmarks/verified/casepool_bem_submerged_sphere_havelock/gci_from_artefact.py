#!/usr/bin/env python3
"""GCI / reference-validation on the band that the committed artefact pins.

``gci_continuum.py`` recomputes the whole ladder (the 1280-panel rungs cost
~4 min/point).  This entry reuses the two fine rungs that the committed
``numbers.json`` already contains (320 and 1280 panels, ka = 0.1 ... 3) and only
*computes* the missing 80-panel rung (a couple of seconds per wavenumber), then
runs the identical V&V algebra.  It exists so the reference-validation study can
be reproduced on a loaded machine, and so the study is not blocked behind the
cluster queue; the full-band numbers come from ``gci_continuum.py``.

Usage
    PYTHONPATH=src python3 \\
        benchmarks/verified/casepool_bem_submerged_sphere_havelock/gci_from_artefact.py \\
        --numbers docs/case_bem_submerged_sphere_havelock/numbers.json \\
        --outdir docs/case_bem_submerged_sphere_havelock/gci --figure
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from tensorbem.benchmarks import submerged_sphere_havelock as library


def _load_gci():
    path = Path(__file__).with_name("gci_continuum.py")
    spec = importlib.util.spec_from_file_location("havelock_gci_continuum", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build(numbers_path: Path, subdivisions: Sequence[int] = (1, 2, 3)) -> dict[str, Any]:
    gci = _load_gci()
    numbers = json.loads(numbers_path.read_text())
    coarse_rows = {float(r["ka"]): r for r in numbers["band_sweep_economy"]}
    fine_rows = {float(r["ka"]): r for r in (numbers["band_sweep_case"] or [])}
    ka_values = sorted(set(coarse_rows) & set(fine_rows))

    log: list[str] = []
    started = time.perf_counter()
    cells: list[dict[str, Any]] = []
    for ka in ka_values:
        started_cell = time.perf_counter()
        rows = library.run_frequency_sweep(
            ka_values=(ka,), radius=gci.RADIUS, submergence=gci.SUBMERGENCE,
            subdivision=1, density=gci.DENSITY, gravity=gci.GRAVITY,
        )
        cell_row = dict(rows[0])
        log.append(f"ka={ka:g} subdiv=1 ({cell_row['n_panels']} panels): "
                   f"{time.perf_counter() - started_cell:.2f} s")
        for quantity in gci.QUANTITIES:
            values = [cell_row[f"{quantity}_bem"],
                      coarse_rows[ka][f"{quantity}_bem"],
                      fine_rows[ka][f"{quantity}_bem"]]
            ref = fine_rows[ka][f"{quantity}_exact"]
            cells.append({
                "ka": ka,
                "quantity": quantity,
                "panels": [cell_row["n_panels"], coarse_rows[ka]["n_panels"],
                           fine_rows[ka]["n_panels"]],
                "levels": [
                    {"subdivision": 1, "n_panels": cell_row["n_panels"],
                     "value": cell_row[f"{quantity}_bem"]},
                    {"subdivision": 2, "n_panels": coarse_rows[ka]["n_panels"],
                     "value": coarse_rows[ka][f"{quantity}_bem"]},
                    {"subdivision": 3, "n_panels": fine_rows[ka]["n_panels"],
                     "value": fine_rows[ka][f"{quantity}_bem"]},
                ],
                "source": ["computed here", f"{numbers_path.name} (economy tier)",
                           f"{numbers_path.name} (case tier)"],
                **gci.cell_study(values, ref),
            })

    resolved = [c for c in cells if c["order_resolved"]]
    orders = [c["observed_order"] for c in resolved]
    cont_err = [c["err_continuum_vs_ref"] for c in cells
                if c["err_continuum_vs_ref"] == c["err_continuum_vs_ref"]]
    ratios = []
    for c in resolved:
        if c["err_fine_vs_ref"] > 0.0:
            predicted = gci.REFINEMENT_RATIO ** c["observed_order"] * c["err_fine_vs_ref"]
            if predicted > 0.0:
                ratios.append(c["err_coarse_vs_ref"] / predicted)

    worst = max((c for c in cells if c["err_continuum_vs_ref"] == c["err_continuum_vs_ref"]),
                key=lambda c: c["err_continuum_vs_ref"])
    gates = {
        "G1_order_in_range": {
            "gate": list(gci.GATE_ORDER),
            "n_resolved": len(resolved),
            "n_cells": len(cells),
            "n_failures": sum(1 for c in resolved if not c["gate_order"]),
            "order_min": min(orders) if orders else None,
            "order_max": max(orders) if orders else None,
            "order_median": sorted(orders)[len(orders) // 2] if orders else None,
            "unresolved_cells": [f"ka={c['ka']:g} {c['quantity']}"
                                 for c in cells if not c["order_resolved"]],
            "verdict": "PASS" if resolved and all(c["gate_order"] for c in resolved) else "FAIL",
        },
        "G2_continuum_vs_reference": {
            "gate": gci.GATE_CONTINUUM_VS_REF,
            "max_rel_error": max(cont_err) if cont_err else None,
            "worst_cell": {"ka": worst["ka"], "quantity": worst["quantity"],
                           "err": worst["err_continuum_vs_ref"]},
            "verdict": ("PASS" if cont_err and max(cont_err) < gci.GATE_CONTINUUM_VS_REF
                        else "FAIL"),
        },
        "G3_asme_validation_at_1280": {
            "criterion": "E = |f_1280 - ref|/|ref| <= U_V = 1.25*GCI_1280",
            "n_cells": len(cells),
            "n_failures": sum(1 for c in cells if not c["gate_validation"]),
            "max_err_fine_vs_ref": max(c["err_fine_vs_ref"] for c in cells),
            "max_gci_fine": max(c["gci_fine"] for c in cells),
            "verdict": "PASS" if all(c["gate_validation"] for c in cells) else "FAIL",
        },
        "G4_economy_error_is_grid_scaling": {
            "gate_factor": gci.GATE_ECONOMY_PREDICTION_FACTOR,
            "ratio_min": min(ratios) if ratios else None,
            "ratio_max": max(ratios) if ratios else None,
            "n_cells": len(ratios),
            "verdict": ("PASS" if ratios and all(
                1.0 / gci.GATE_ECONOMY_PREDICTION_FACTOR <= q <= gci.GATE_ECONOMY_PREDICTION_FACTOR
                for q in ratios) else "FAIL"),
        },
    }
    verdict = "PASS" if all(g["verdict"] == "PASS" for g in gates.values()) else "FAIL"
    return {
        "case": "CASE-BEM-SUBMERGED-SPHERE-HAVELOCK",
        "study": "grid convergence (Richardson / GCI / ASME V&V 20), artefact-backed band",
        "provenance_note": (
            "The 320-panel and 1280-panel rungs are taken from the committed case "
            "artefact; the 80-panel rung is computed here. The full 7-wavenumber "
            "band (adding ka=4,5 on the case mesh) is produced by gci_continuum.py."
        ),
        "protocol": {
            "ka_band": ka_values,
            "subdivisions": list(subdivisions),
            "panels": [80, 320, 1280],
            "refinement_ratio_linear": gci.REFINEMENT_RATIO,
            "safety_factor": gci.SAFETY_FACTOR,
            "radius": gci.RADIUS,
            "submergence": gci.SUBMERGENCE,
            "density": gci.DENSITY,
            "gravity": gci.GRAVITY,
            "gates": {
                "G1_order_in_range": list(gci.GATE_ORDER),
                "G2_continuum_vs_reference": gci.GATE_CONTINUUM_VS_REF,
                "G3_asme_validation": "E <= 1.25*GCI_1280",
                "G4_economy_error_is_grid_scaling": gci.GATE_ECONOMY_PREDICTION_FACTOR,
            },
        },
        "environment": gci._env(),
        "cells": cells,
        "gates": gates,
        "verdict": verdict,
        "runtime_seconds": time.perf_counter() - started,
        "log": log,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--numbers", type=Path,
                        default=Path("docs/case_bem_submerged_sphere_havelock/numbers.json"))
    parser.add_argument("--outdir", type=Path,
                        default=Path("docs/case_bem_submerged_sphere_havelock/gci"))
    parser.add_argument("--figure", action="store_true")
    args = parser.parse_args(argv)

    gci = _load_gci()
    result = build(args.numbers)
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "gci.json").write_text(json.dumps(result, indent=2, default=str))
    gci.write_csv(args.outdir / "table_gci.csv", result["cells"])
    (args.outdir / "GCI.md").write_text(gci.render_markdown(result))
    if args.figure:
        gci.make_figure(result, args.outdir / "fig5_gci.png")
    for name, gate in result["gates"].items():
        print(f"{name}: {gate['verdict']}")
    print(f"overall: {result['verdict']}  ({result['runtime_seconds']:.1f} s)")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())