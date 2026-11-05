"""Turn the interior-harmonic study into registry-shaped rows (same judge as the registry)."""

from __future__ import annotations

import json
from pathlib import Path

from tensorbem.benchmarks.declared_numbers import DECLARED_NUMBERS, _judge

ROOT = Path("outputs/verif_m3059")


def main() -> None:
    study = json.loads((ROOT / "interior_harmonics_1280.json").read_text())
    measured: dict[tuple[str, int], dict] = {}
    for harmonic, payload in study["cases"].items():
        for result in payload["results"]:
            measured[(harmonic, int(result["panels"]))] = result

    rows = []
    for entry in DECLARED_NUMBERS:
        if entry.group != "interior_harmonics":
            continue
        harmonic = str(entry.parameters["harmonic"])
        subdivisions = int(entry.parameters["subdivisions"])
        panels = {0: 20, 1: 80, 2: 320, 3: 1280}[subdivisions]
        result = measured[(harmonic, panels)]
        value = float(result[entry.quantity])
        deviation, verdict = _judge(entry, value)
        rows.append(
            {
                "key": entry.key,
                "description": entry.description,
                "source": entry.source,
                "group": entry.group,
                "parameters": entry.parameters,
                "tier": entry.tier,
                "comparison": entry.kind,
                "declared": entry.declared,
                "measured": value,
                "relative_deviation": deviation,
                "tolerance": entry.tolerance,
                "verdict": verdict,
            }
        )
    drifted = [row["key"] for row in rows if row["verdict"] != "PASS"]
    report = {
        "benchmark": "declared_numbers_recomputation",
        "tiers": ["heavy"],
        "checks": rows,
        "summary": {
            "checked": len(rows),
            "passed": len(rows) - len(drifted),
            "drifted": len(drifted),
            "drifted_keys": drifted,
            "worst_relative_deviation": max(
                (abs(row["relative_deviation"]) for row in rows), default=0.0
            ),
            "verdict": "PASS" if not drifted else "DRIFT",
        },
        "provenance": {
            "runner": "tensorbem.benchmarks.interior_harmonics --subdivisions 2 3 --degrees 1 2 3",
            "artifact": "outputs/verif_m3059/interior_harmonics_1280.json",
        },
    }
    (ROOT / "harmonic_rows.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    for row in rows:
        print(row["key"], row["declared"], row["measured"], f"{row['relative_deviation']:+.2e}", row["verdict"])


if __name__ == "__main__":
    main()