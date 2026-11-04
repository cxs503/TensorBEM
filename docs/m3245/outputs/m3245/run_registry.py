"""Recompute only the 3-D free-surface rows of the declared-number registry.

Writes ``outputs/m3245/free_surface3d_registry.json`` (zero hand-copy source for
the report and the docs table) and prints a compact table.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from tensorbem.benchmarks.declared_numbers import DECLARED_NUMBERS, check

GROUPS = ("hulme_hemisphere", "hulme_chief", "submerged_sphere_wave")
entries = tuple(entry for entry in DECLARED_NUMBERS if entry.group in GROUPS)

tiers = sorted({entry.tier for entry in entries})
t0 = time.perf_counter()
report = check(tiers=tiers, entries=entries)
elapsed = time.perf_counter() - t0

report["elapsed_seconds"] = elapsed
report["groups"] = list(GROUPS)
report["note"] = (
    "Only the 3-D free-surface groups are recomputed here; the full registry is "
    "run by `python -m tensorbem.benchmarks.declared_numbers`."
)

out = Path("outputs/m3245/free_surface3d_registry.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")

print(f"checked {report['summary']['checked']} rows in {elapsed:.1f} s -> {out}")
for row in report["checks"]:
    print(
        f"{row['verdict']:5s} {row['key']:70s} declared={row['declared']:.6g} "
        f"measured={row['measured']:.6g} dev={row['relative_deviation']:+.2e}"
    )
print(json.dumps(report["summary"], indent=2))