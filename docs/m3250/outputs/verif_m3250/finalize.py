"""Judge the 3-D free-surface registry rows from the measured artefacts.

The registry's own ``check`` run did not finish inside this session (machine
load15 > 130: the same three studies took 61 s at load 75 and over 12 min when
started later). The artefacts in this directory were produced by exactly the
benchmark functions the registry's runner calls, so the judgement below uses the
registry's own ``_judge`` and the registry's own declared values against them.
The in-suite gate is ``tests/test_declared_numbers.py::
test_free_surface_3d_tables_are_reproduced``, which runs ``check`` directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_all  # noqa: E402
from tensorbem.benchmarks.declared_numbers import (  # noqa: E402
    _HEMISPHERE_ORDER_LABELS,
    _SPHERE_ORDER_LABELS,
    _flatten_refinement_levels,
    _judge,
    _reference_values,
    DECLARED_NUMBERS,
)

GROUPS = ("hemisphere3d", "hemisphere3d_chief", "submerged_sphere3d")


def chief_values(path: Path) -> dict[str, float]:
    row = json.loads(path.read_text())[0]
    values = {"exact_a33": float(row["exact_a33"]), "exact_b33": float(row["exact_b33"])}
    for branch in ("unregularized", "chief_regularized"):
        for name, value in row[branch].items():
            if isinstance(value, bool):
                continue
            values[f"{branch}.{name}"] = float(value)
        values[f"{branch}.b33_dimless_abs"] = abs(float(row[branch]["b33_dimless"]))
    values["state.condition_number_ratio"] = (
        float(row["chief_regularized"]["condition_number"])
        / float(row["unregularized"]["condition_number"])
    )
    return values


def main() -> None:
    hemisphere = json.loads((HERE / "hemisphere_refinement.json").read_text())
    sphere = json.loads((HERE / "submerged_sphere_refinement.json").read_text())

    measured: dict[str, dict[str, float]] = {}
    measured["hemisphere3d"] = dict(
        _flatten_refinement_levels(hemisphere["levels"], _HEMISPHERE_ORDER_LABELS),
        **_reference_values(hemisphere["exact_reference"]),
    )
    measured["submerged_sphere3d"] = dict(
        _flatten_refinement_levels(sphere["levels"], _SPHERE_ORDER_LABELS),
        **_reference_values(sphere["exact_reference"]),
    )
    measured["hemisphere3d_chief"] = chief_values(HERE / "hemisphere_chief.json")

    rows = []
    for entry in DECLARED_NUMBERS:
        if entry.group not in GROUPS:
            continue
        value = measured[entry.group][entry.quantity]
        deviation, verdict = _judge(entry, value)
        rows.append(
            {
                "key": entry.key,
                "description": entry.description,
                "source": entry.source,
                "group": entry.group,
                "quantity": entry.quantity,
                "comparison": entry.kind,
                "declared": entry.declared,
                "measured": value,
                "relative_deviation": deviation,
                "tolerance": entry.tolerance,
                "verdict": verdict,
            }
        )
    passed = sum(1 for row in rows if row["verdict"] == "PASS")
    report = {
        "benchmark": "declared_numbers_three_dimensional_free_surface",
        "method": (
            "artefact re-judgement: measured values from outputs/verif_m3250/*.json "
            "produced by tensorbem.benchmarks.{floating_hemisphere,"
            "submerged_sphere_wave}; declared values and _judge from "
            "tensorbem.benchmarks.declared_numbers"
        ),
        "summary": {
            "checked": len(rows),
            "passed": passed,
            "drifted": len(rows) - passed,
            "verdict": "PASS" if passed == len(rows) else "DRIFT",
            "drifted_keys": [row["key"] for row in rows if row["verdict"] != "PASS"],
        },
        "checks": rows,
    }
    (HERE / "fs3d_registry.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(report["summary"]["checked"], report["summary"]["verdict"], flush=True)
    for row in rows:
        if row["verdict"] != "PASS":
            print("DRIFT", row["key"], row["declared"], row["measured"], row["relative_deviation"])
    print("figure", build_all.figure(report), flush=True)
    build_all.inject(build_all.doc_section(report))


if __name__ == "__main__":
    main()