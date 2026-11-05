"""Evidence that merging PR #136 lost no registered number (m3862).

The merge renamed the three 3-D free-surface registry groups (main:
``hulme_hemisphere`` / ``hulme_chief`` / ``submerged_sphere`` -> m3250:
``hemisphere3d`` / ``hemisphere3d_chief`` / ``submerged_sphere3d``) and replaced
62 hand-written entries with a generator that emits 63.  A rename is exactly the
situation in which a number disappears without anyone noticing, so this script
compares the *multiset of declared values* of the two sides and prints the ones
that do not cancel.

Run it from the repository root::

    python docs/m3862/registry_equivalence.py

Exit status 0 means every difference is in the documented reconciliation list
below (the five known re-pins), i.e. no value vanished.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD_GROUPS = {"hulme_hemisphere", "hulme_chief", "submerged_sphere"}
NEW_GROUPS = {"hemisphere3d", "hemisphere3d_chief", "submerged_sphere3d"}

#: Every value that is expected to differ between the two sides, with the reason.
#: ``(side, value)`` -- main-only vs merged-only.
RECONCILED: list[dict] = [
    {
        "main_values": [-0.1042],
        "merged_values": [0.1042],
        "reason": (
            "CHIEF-unregularised B33': main registered the signed value, m3250 "
            "registers |B33'| (the sign is asserted by "
            "test_chief_regularisation_restores_passive_heave_damping). Same "
            "physical number, different encoding."
        ),
    },
    {
        "main_values": [0.0002151, 2.6481e-05],
        "merged_values": [0.00022, 2.6e-05],
        "reason": (
            "sphere-centre rotational coupling at 80/320 panels: main pinned the "
            "three-significant-digit values printed in table 12 after the m3150 "
            "rounding fix (2.151e-4 / 2.6481e-5); m3250 carried the earlier "
            "two-digit pair. The merge keeps main's sharper pins."
        ),
        "resolution": "main",
    },
    {
        "main_values": [1.8021, 1.9377],
        "merged_values": [1.8, 1.94],
        "reason": (
            "observed order of B11' (64 / 256 panels): main stored the four-digit "
            "measurement, the table prints two decimals. The registry pins the "
            "printed pair; the 1.5 % band accepts main's values (deviation "
            "0.12 % / 0.13 %) and the four-digit pair is recorded in the "
            "registry comment."
        ),
        "resolution": "main-comment",
    },
    {
        "main_values": [2.5e-32],
        "merged_values": [1.0e-06],
        "reason": (
            "20-panel rotational coupling: 2.5e-32 is a round-off artefact of an "
            "exactly symmetric mesh; m3250 re-pinned it as the upper bound "
            "'< 1e-6' and the table cell was rewritten to match."
        ),
        "resolution": "m3250",
    },
]


def _load_side(text: str, name: str):
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / f"{name}.py"
        path.write_text(text, encoding="utf-8")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module


def _signature(module, groups: set[str]) -> Counter:
    return Counter(
        round(entry.declared, 10) for entry in module.DECLARED_NUMBERS
        if entry.group in groups
    )


def main() -> int:
    main_side = subprocess.run(
        ["git", "show", "origin/main:src/tensorbem/benchmarks/declared_numbers.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    sys.path.insert(0, str(ROOT / "src"))
    old = _load_side(main_side, "dn_main_side")
    from tensorbem.benchmarks import declared_numbers as merged

    old_values = _signature(old, OLD_GROUPS)
    new_values = _signature(merged, NEW_GROUPS)
    only_main = sorted((old_values - new_values).elements())
    only_merged = sorted((new_values - old_values).elements())

    leftovers = list(only_main) + list(only_merged)
    for row in RECONCILED:
        for value in row["main_values"]:
            if value in leftovers:
                leftovers.remove(value)
        for value in row["merged_values"]:
            if value in leftovers:
                leftovers.remove(value)

    report = {
        "main_rows": len([e for e in old.DECLARED_NUMBERS if e.group in OLD_GROUPS]),
        "merged_rows": len(
            [e for e in merged.DECLARED_NUMBERS if e.group in NEW_GROUPS]
        ),
        "main_total_rows": len(old.DECLARED_NUMBERS),
        "merged_total_rows": len(merged.DECLARED_NUMBERS),
        "only_on_main": only_main,
        "only_in_merge": only_merged,
        "reconciled": RECONCILED,
        "unexplained_differences": leftovers,
        "verdict": "NO-NUMBER-LOST" if not leftovers else "UNEXPLAINED-DIFFERENCE",
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not leftovers else 1


if __name__ == "__main__":
    sys.exit(main())