#!/usr/bin/env python3
"""Generate the panel operator order benchmark deliverables."""
import json
import time
from pathlib import Path

import numpy as np

from tensorbem.benchmarks.panel_operator_order import (
    run_operator_study,
    run_energy_identity_study,
    evaluate_criteria,
    write_report,
    write_figures,
    CRITERIA,
)

def main():
    outdir = Path("benchmarks/verified/panel_operator_order")
    outdir.mkdir(parents=True, exist_ok=True)
    figures_dir = outdir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("Panel Operator Order Benchmark")
    print("=" * 70)
    
    # Run operator study with levels 1-2 (80 and 320 panels)
    # Levels 3-4 are too slow for local execution
    print("\n[1/4] Running operator study (subdivisions 1-2)...")
    t0 = time.time()
    operator = run_operator_study(
        subdivisions=[1, 2],
        quadrature_order=8,
        panel_stride=1,
    )
    print(f"  Operator study completed in {time.time()-t0:.1f}s")
    print(f"  Panels: {[row['panels'] for row in operator['rows']]}")
    print(f"  err_flux: {[row['err_flux'] for row in operator['rows']]}")
    print(f"  err_potential: {[row['err_potential'] for row in operator['rows']]}")
    
    # Run energy identity study with levels 2-3 (64 and 256 panels for hemisphere)
    print("\n[2/4] Running energy identity study (subdivisions 2-3)...")
    t0 = time.time()
    energy = run_energy_identity_study(
        subdivisions=[2, 3],
        ka=1.2,
        n_azimuth=36,
    )
    print(f"  Energy study completed in {time.time()-t0:.1f}s")
    print(f"  Panels: {[row['panels'] for row in energy['rows']]}")
    if energy['rows']:
        print(f"  Max rel diff: {[row['max_abs_relative_difference'] for row in energy['rows']]}")
    
    # Evaluate criteria
    print("\n[3/4] Evaluating criteria...")
    criteria = evaluate_criteria(operator, energy)
    print(f"  Verdicts: {criteria['verdicts']}")
    
    # Build the full payload
    import platform
    import sys
    payload = {
        "provenance": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "command": "python benchmarks/verified/panel_operator_order/generate_deliverables.py",
        },
        "criteria_definitions": CRITERIA,
        "operator": operator,
        "energy": energy,
        "criteria": criteria,
    }
    
    # Write result.json
    result_path = outdir / "result.json"
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[4/4] Writing deliverables...")
    print(f"  {result_path}")
    
    # Write gates_verified.json
    gates = {
        "campaign_node": "IN49-W3-N03",
        "mission_id": 5758,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "criteria_verdicts": criteria["verdicts"],
        "criteria_summary": criteria["summary"],
        "operator_study": {
            "subdivisions": [row["subdivisions"] for row in operator["rows"]],
            "panels": [row["panels"] for row in operator["rows"]],
            "err_flux": [row["err_flux"] for row in operator["rows"]],
            "err_potential": [row["err_potential"] for row in operator["rows"]],
            "slopes": operator["slopes"],
        },
        "energy_study": {
            "subdivisions": [row["subdivisions"] for row in energy["rows"]],
            "panels": [row["panels"] for row in energy["rows"]],
            "max_rel_diff": [row["max_abs_relative_difference"] for row in energy["rows"]],
            "monotone": criteria["C4"].get("monotone_decrease"),
        },
        "self_term": {
            "polar_max_rel_error": operator["self_term"]["polar_max_rel_error"],
            "coarse_max_rel_error": operator["self_term"]["coarse_max_rel_error"],
            "gate": CRITERIA["C3_self_term_analytic_vs_numeric"]["tolerance"],
        },
        "notes": {
            "operator_levels": "Only subdivisions 1-2 (80, 320 panels) computed due to local timeout; full 4-level sweep requires cluster",
            "energy_levels": "Subdivisions 2-3 (64, 256 hemisphere panels)",
            "linear_panel": "NOT_SUPPORTED - constant-panel collocation only",
        },
    }
    gates_path = outdir / "gates_verified.json"
    gates_path.write_text(json.dumps(gates, indent=2), encoding="utf-8")
    print(f"  {gates_path}")
    
    # Write markdown report
    report_path = outdir / "report.md"
    write_report(operator, energy, criteria, report_path, payload["provenance"])
    print(f"  {report_path}")
    
    # Generate figures
    figure_paths = write_figures(operator, energy, criteria, figures_dir)
    for path in figure_paths:
        print(f"  {path}")
    
    print("\n" + "=" * 70)
    print("Deliverables generated successfully!")
    print("=" * 70)
    
    # Print summary
    print("\nSummary:")
    print(f"  C1a (constant-panel slope): {criteria['verdicts']['C1a']}")
    print(f"  C1b (linear-panel slope): {criteria['verdicts']['C1b']}")
    print(f"  C2 (monotone decrease): {criteria['verdicts']['C2']}")
    print(f"  C3 (self-term): {criteria['verdicts']['C3']}")
    print(f"  C4 (energy identity): {criteria['verdicts']['C4']}")
    print(f"  C5 (negative control): {criteria['verdicts']['C5']}")
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
