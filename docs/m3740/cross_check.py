"""Cross-check the two independent registrations that the merge put side by side.

main          registered tables 10-12 as groups ``hulme_hemisphere`` /
              ``hulme_chief`` / ``submerged_sphere`` (default tier, point values
              from the post-audit documentation).
agent/m3181   registered the same tables as groups ``hulme3d`` / ``hulme3d_chief``
              / ``wang_linton3d`` (heavy tier, values transcribed from the same
              document).

A semantic union keeps both, so the merged tree must quote one number per
physical cell.  This script pairs the cells by (panel index, quantity) and prints
the disagreements; it is the machine-checkable form of "no side was silently
dropped *and* no side silently overrides the other".

    PYTHONPATH=src python3 docs/m3740/cross_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorbem.benchmarks.declared_numbers import DECLARED_NUMBERS  # noqa: E402

MAIN_PREFIX = {"hulme_hemisphere": "hulme", "hulme_chief": "hulme_chief",
               "submerged_sphere": "wang_linton"}
M3181_GROUPS = ("hulme3d", "hulme3d_chief", "wang_linton3d")


def panel_index(quantity: str) -> str:
    """``sub2_heave_added_mass_rel_error`` -> ``('2', 'heave_added_mass_rel_error')``."""
    head, _, tail = quantity.partition("_")
    if head.startswith("sub") and head[3:].isdigit():
        return head[3:], tail
    return "", quantity


def m3181_key(prefix: str, quantity: str, parameters: dict) -> str | None:
    """Reconstruct the m3181-side registry key for a main-side row."""
    index, tail = panel_index(quantity)
    if prefix in {"hulme", "wang_linton"}:
        subdivisions = parameters.get("subdivisions") or []
        if not index or not subdivisions:
            return None
        # main names its rows after the *subdivision level*, m3181 after the
        # panel count, so the level -> panels table is the bridge.
        level = int(index)
        panels = {1: 16, 2: 64, 3: 256} if prefix == "hulme" else {0: 20, 1: 80, 2: 320}
        if level not in panels:
            return None
        return f"{prefix}_{tail}_{panels[level]}"
    # CHIEF table: main prefixes the frequency, m3181 states it once per group.
    if prefix == "hulme_chief":
        cleaned = tail.removeprefix("ka2p58_").removeprefix("ka2.58_")
        return f"hulme_chief_{cleaned}"
    return None


def main() -> int:
    m3181 = {entry.key: entry for entry in DECLARED_NUMBERS if entry.group in M3181_GROUPS}
    pairs, unmapped = [], []
    for entry in DECLARED_NUMBERS:
        if entry.group not in MAIN_PREFIX:
            continue
        key = m3181_key(MAIN_PREFIX[entry.group], entry.quantity, entry.parameters)
        if key is None or key not in m3181:
            unmapped.append(entry.key)
            continue
        other = m3181[key]
        same = (
            abs(entry.declared - other.declared)
            <= 1e-9 * max(abs(entry.declared), 1e-30)
            or entry.declared == other.declared
        )
        pairs.append(
            {
                "main_key": entry.key,
                "m3181_key": key,
                "main_declared": entry.declared,
                "m3181_declared": other.declared,
                "main_tolerance": entry.tolerance,
                "m3181_tolerance": other.tolerance,
                "same_declared": bool(same),
            }
        )
    mismatched = [row for row in pairs if not row["same_declared"]]
    report = {
        "paired_cells": len(pairs),
        "identical_declared": len(pairs) - len(mismatched),
        "mismatched": mismatched,
        "main_rows_without_counterpart": unmapped,
        "m3181_rows_total": len(m3181),
    }
    out = HERE / "outputs" / "cross_check_two_registrations.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "mismatched"}, indent=2))
    for row in mismatched:
        print("MISMATCH", row)
    return 0 if not mismatched else 1


if __name__ == "__main__":
    raise SystemExit(main())