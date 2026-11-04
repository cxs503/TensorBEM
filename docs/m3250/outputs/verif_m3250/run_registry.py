"""Run the extended registry on the standard tier and dump the verdict JSON."""

from __future__ import annotations

import json
from pathlib import Path

from tensorbem.benchmarks.declared_numbers import DECLARED_NUMBERS, check

OUT = Path(__file__).resolve().parent

GROUPS = ("hemisphere3d", "hemisphere3d_chief", "submerged_sphere3d")


def main() -> None:
    entries = tuple(e for e in DECLARED_NUMBERS if e.group in GROUPS)
    report = check(tiers=("standard",), entries=entries)
    (OUT / "fs3d_registry.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(report["summary"]["checked"], report["summary"]["verdict"])
    for row in report["checks"]:
        if row["verdict"] != "PASS":
            print("DRIFT", row["key"], row["declared"], row["measured"], row["relative_deviation"])


if __name__ == "__main__":
    main()