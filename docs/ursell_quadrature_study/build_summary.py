"""Build the Ursell quadrature-versus-discretisation tables from the raw runs.

The ``raw/`` directory holds the unedited benchmark JSON written by
``tensorbem.benchmarks.ursell_cylinder`` for two implementations of the same
2-D free-surface solver:

* ``current_*`` - this tree. The singular ``log r`` part of the source
  potential is integrated analytically over each straight panel and only the
  smooth free-surface remainder is quadratured (introduced in commit 267d5a3).
* ``legacy878834a_*`` - the tree at commit 878834a, i.e. the state that produced
  the numbers currently printed in ``docs/marine_bem_benchmarks.md`` §4.6. There
  the full Green function, singular part included, was handed to a 24-node
  Gauss-Legendre rule per panel.

Nothing here re-runs the solver: this script only reads the raw files so that
every number in the report has a committed, machine-readable origin.

Run: python docs/ursell_quadrature_study/build_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_RAW = _HERE / "raw"

_FIELDS = (
    "panels",
    "panel_added_mass",
    "panel_added_mass_relative_error",
    "panel_damping",
    "panel_damping_relative_error",
    "near_far_field_relative_difference",
    "panel_condition_number",
)

PROVENANCE = {
    "current": {
        "tree": "agent/m2223 worktree (HEAD of this branch)",
        "solver": "analytic log primitive + smooth free-surface quadrature",
        "command": "python -m tensorbem.benchmarks.ursell_cylinder --panels P --ka 0.6 "
        "--panel-nodes N --output raw/current_panelsP_nodesN.json",
        "note": "effective smooth quadrature nodes are min(panel_nodes, 6)",
    },
    "legacy878834a": {
        "tree": "git archive 878834a",
        "solver": "full Green function quadratured with panel_nodes Gauss-Legendre nodes",
        "command": "python -m tensorbem.benchmarks.ursell_cylinder --panels P --ka 0.6 "
        "--panel-nodes N --output raw/legacy878834a_panelsP_nodesN.json",
        "note": "878834a is the commit whose printed table is quoted in docs/ §4.6",
    },
}


def _load(pattern: str) -> list[dict]:
    rows = []
    for path in sorted(_RAW.glob(pattern)):
        record = json.loads(path.read_text(encoding="utf-8"))
        row = {field: record[field] for field in _FIELDS}
        row["source_file"] = path.name
        row["panel_nodes"] = int(path.stem.rsplit("nodes", 1)[1])
        rows.append(row)
    return rows


def build() -> dict:
    current = _load("current_*.json")
    legacy = _load("legacy878834a_*.json")

    panel_sweep = sorted(
        (row for row in current if row["panel_nodes"] == 24), key=lambda r: r["panels"]
    )
    node_sweep_current = sorted(
        (row for row in current if row["panels"] == 128), key=lambda r: r["panel_nodes"]
    )
    node_sweep_legacy = sorted(legacy, key=lambda r: r["panel_nodes"])

    def spread(rows: list[dict], field: str) -> float:
        values = [row[field] for row in rows]
        return max(values) - min(values)

    reference = {
        "added_mass": panel_sweep[0] and current[0]["panel_added_mass"] and None,
    }
    # The reference values are identical in every raw file (same analytic
    # construction); take them from one file rather than duplicating a literal.
    first = json.loads((_RAW / panel_sweep[0]["source_file"]).read_text(encoding="utf-8"))
    reference = {
        "added_mass": first["reference_added_mass"],
        "damping": first["reference_damping"],
        "method": first["reference_method"],
    }

    summary = {
        "reference": reference,
        "provenance": PROVENANCE,
        "panel_sweep_current_nodes24": panel_sweep,
        "node_sweep_current_panels128": node_sweep_current,
        "node_sweep_legacy_panels128": node_sweep_legacy,
        "node_sensitivity_current_panels128": {
            "added_mass_spread": spread(node_sweep_current, "panel_added_mass"),
            "damping_spread": spread(node_sweep_current, "panel_damping"),
            "added_mass_spread_relative": spread(
                node_sweep_current, "panel_added_mass"
            )
            / reference["added_mass"],
            "damping_spread_relative": spread(node_sweep_current, "panel_damping")
            / reference["damping"],
        },
        "node_sensitivity_legacy_panels128": {
            "added_mass_spread": spread(node_sweep_legacy, "panel_added_mass"),
            "damping_spread": spread(node_sweep_legacy, "panel_damping"),
            "added_mass_spread_relative": spread(node_sweep_legacy, "panel_added_mass")
            / reference["added_mass"],
            "damping_spread_relative": spread(node_sweep_legacy, "panel_damping")
            / reference["damping"],
        },
    }
    # Extrapolated continuum limit of the legacy sequence is the current
    # (quadrature-converged) value; record the residual gap explicitly.
    converged = node_sweep_current[0]["panel_added_mass"]
    summary["legacy_finest_gap_to_converged"] = {
        "added_mass": node_sweep_legacy[-1]["panel_added_mass"] - converged,
        "added_mass_relative": (
            node_sweep_legacy[-1]["panel_added_mass"] - converged
        )
        / converged,
        "damping": node_sweep_legacy[-1]["panel_damping"]
        - node_sweep_current[0]["panel_damping"],
        "damping_relative": (
            node_sweep_legacy[-1]["panel_damping"] - node_sweep_current[0]["panel_damping"]
        )
        / node_sweep_current[0]["panel_damping"],
    }
    return summary


def to_markdown(summary: dict) -> str:
    ref = summary["reference"]
    lines = [
        "# Ursell quadrature study - generated tables",
        "",
        "Generated by `docs/ursell_quadrature_study/build_summary.py` from the raw",
        "benchmark JSON in `raw/`. Do not edit by hand.",
        "",
        "Reference (Ursell multipole, independent of the BEM): "
        f"A = {ref['added_mass']:.4f}, B = {ref['damping']:.4f}",
        "",
        "## Current solver, panel refinement (panel_nodes = 24)",
        "",
        "| panels | A | err(A) | B | err(B) | near/far | kappa |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in summary["panel_sweep_current_nodes24"]:
        lines.append(
            "| {panels} | {A:.4f} | {eA:.3e} | {B:.4f} | {eB:.3e} | {nf:.3e} | {k:.2f} |".format(
                panels=row["panels"],
                A=row["panel_added_mass"],
                eA=row["panel_added_mass_relative_error"],
                B=row["panel_damping"],
                eB=row["panel_damping_relative_error"],
                nf=row["near_far_field_relative_difference"],
                k=row["panel_condition_number"],
            )
        )
    lines += [
        "",
        "## Quadrature-node sensitivity at 128 panels",
        "",
        "| panel_nodes | current A | current B | legacy A | legacy B |",
        "|---|---|---|---|---|",
    ]
    legacy_by_nodes = {row["panel_nodes"]: row for row in summary["node_sweep_legacy_panels128"]}
    for row in summary["node_sweep_current_panels128"]:
        legacy = legacy_by_nodes.get(row["panel_nodes"])
        lines.append(
            "| {n} | {ca:.6f} | {cb:.6f} | {la} | {lb} |".format(
                n=row["panel_nodes"],
                ca=row["panel_added_mass"],
                cb=row["panel_damping"],
                la=("-" if legacy is None else "{:.6f}".format(legacy["panel_added_mass"])),
                lb=("-" if legacy is None else "{:.6f}".format(legacy["panel_damping"])),
            )
        )
    cur = summary["node_sensitivity_current_panels128"]
    leg = summary["node_sensitivity_legacy_panels128"]
    gap = summary["legacy_finest_gap_to_converged"]
    lines += [
        "",
        f"Node-sweep spread, current: dA = {cur['added_mass_spread']:.3e} "
        f"({cur['added_mass_spread_relative']:.2e} relative), "
        f"dB = {cur['damping_spread']:.3e} ({cur['damping_spread_relative']:.2e} relative)",
        "",
        f"Node-sweep spread, legacy 878834a: dA = {leg['added_mass_spread']:.3e} "
        f"({leg['added_mass_spread_relative']:.2e} relative), "
        f"dB = {leg['damping_spread']:.3e} ({leg['damping_spread_relative']:.2e} relative)",
        "",
        f"Legacy finest (48 nodes) minus converged value: dA = {gap['added_mass']:.4f} "
        f"({gap['added_mass_relative']:.2e} relative), "
        f"dB = {gap['damping']:.4f} ({gap['damping_relative']:.2e} relative)",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    summary = build()
    (_HERE / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    markdown = to_markdown(summary)
    (_HERE / "summary.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()