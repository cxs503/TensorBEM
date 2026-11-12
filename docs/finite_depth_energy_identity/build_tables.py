"""Build report tables from the finite-depth energy identity convergence artifact."""

from __future__ import annotations

import json
from pathlib import Path

ARTIFACT = Path(__file__).resolve().parents[2] / "outputs" / "finite_depth_energy_identity" / "convergence.json"


def _fmt(val: float, prec: int = 4) -> str:
    return f"{val:.{prec}e}"


def blocks(payload: dict) -> dict[str, str]:
    """Return named table blocks for embedding in report.md."""
    rows = payload["rows"]
    deep_rows = payload["deep_water_control"]
    summaries = payload["summaries"]
    deep_summaries = payload["deep_water_summaries"]

    # Table 1: Shallow depth heave convergence
    shallow_heave = [r for r in rows if r["depth_label"] == "shallow" and r["motion"] == "heave"]
    shallow_heave.sort(key=lambda r: r["panels"])
    t1_lines = ["| Panels | A_33 | B_near | B_far | Offset | Offset_quad |",
                "|-------:|-----:|-------:|------:|-------:|------------:|"]
    for r in shallow_heave:
        t1_lines.append(f"| {r['panels']} | {_fmt(r['added_mass'])} | {_fmt(r['damping_near'])} | {_fmt(r['damping_far'])} | {_fmt(r['energy_offset'])} | {_fmt(r['energy_offset_quadrature'])} |")
    t1 = "\n".join(t1_lines)

    # Table 2: Shallow depth sway convergence
    shallow_sway = [r for r in rows if r["depth_label"] == "shallow" and r["motion"] == "sway"]
    shallow_sway.sort(key=lambda r: r["panels"])
    t2_lines = ["| Panels | A_11 | B_near | B_far | Offset | Offset_quad |",
                "|-------:|-----:|-------:|------:|-------:|------------:|"]
    for r in shallow_sway:
        t2_lines.append(f"| {r['panels']} | {_fmt(r['added_mass'])} | {_fmt(r['damping_near'])} | {_fmt(r['damping_far'])} | {_fmt(r['energy_offset'])} | {_fmt(r['energy_offset_quadrature'])} |")
    t2 = "\n".join(t2_lines)

    # Table 3: Deep water control
    dw_heave = [r for r in deep_rows if r["motion"] == "heave"]
    dw_heave.sort(key=lambda r: r["panels"])
    t3_lines = ["| Panels | B_near | B_far | Offset |",
                "|-------:|-------:|------:|-------:|"]
    for r in dw_heave:
        t3_lines.append(f"| {r['panels']} | {_fmt(r['damping_near'])} | {_fmt(r['damping_far'])} | {_fmt(r['energy_offset'])} |")
    t3 = "\n".join(t3_lines)

    # Table 4: Summaries
    t4_lines = ["| Case | Median Order | Offset Limit | Near-vs-Far Limit |",
                "|------|-------------:|-------------:|------------------:|"]
    for key, s in summaries.items():
        t4_lines.append(f"| {key} | {s['median_order']:.2f} | {_fmt(s['offset_limit'])} | {_fmt(s['near_vs_far_limit_rel'])} |")
    t4 = "\n".join(t4_lines)

    # Table 5: Node sweep
    node_rows = payload.get("node_sweep", [])
    t5_lines = ["| Panel Nodes | Offset | Offset_quad | Far Reading Diff |",
                "|------------:|-------:|------------:|-----------------:|"]
    for r in node_rows:
        t5_lines.append(f"| {r['panel_nodes']} | {_fmt(r['energy_offset'])} | {_fmt(r['energy_offset_quadrature'])} | {_fmt(r['far_reading_difference'])} |")
    t5 = "\n".join(t5_lines)

    return {
        "shallow_heave_convergence": t1,
        "shallow_sway_convergence": t2,
        "deep_water_control": t3,
        "summaries": t4,
        "node_sweep": t5,
    }


if __name__ == "__main__":
    payload = json.loads(ARTIFACT.read_text())
    for name, block in blocks(payload).items():
        print(f"<!-- BEGIN:{name} -->")
        print(block)
        print(f"<!-- END:{name} -->")
        print()
