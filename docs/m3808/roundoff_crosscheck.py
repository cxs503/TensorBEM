"""Cross-version reproduction check for ``docs/finite_depth3d.md`` (mission m3808).

Question this script answers with numbers instead of adjectives: *did the m3409
rewrite of the propagating mode (``cosh(k0*(z+h))`` -> log-scaled exponentials)
change the kernel, or only its roundoff?*

It reads the pin artifact recorded **before** the main merge
(``git show <base>:docs/finite_depth3d_pins/check.json``) and the one written by
the current tree, pairs the rows, and reports per-row movement in units of the
double-precision machine epsilon ``eps = 2.22e-16``.  A neutral rewrite leaves
every row bit-identical except the ones whose value is cancellation noise.

Usage::

    PYTHONPATH=src python3 docs/m3808/roundoff_crosscheck.py --base ab65abd
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CURRENT = REPO_ROOT / "docs" / "finite_depth3d_pins" / "check.json"
OUT = REPO_ROOT / "docs" / "finite_depth3d_pins" / "roundoff_crosscheck.json"
EPS = sys.float_info.epsilon


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_git_payload(rev: str, path: Path) -> dict:
    rel = path.relative_to(REPO_ROOT).as_posix()
    blob = subprocess.run(
        ["git", "show", f"{rev}:{rel}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(blob)


def key(row: dict) -> str:
    return f"{row['table']}/{row['key']}/{row['quantity']}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="ab65abd", help="revision recorded before the merge")
    args = parser.parse_args(argv)

    before = {key(r): r for r in load_git_payload(args.base, CURRENT)["rows"]}
    after = {key(r): r for r in load_payload(CURRENT)["rows"]}
    missing = sorted(set(before) ^ set(after))
    if missing:
        raise SystemExit(f"row sets differ between revisions: {missing[:5]}")

    rows = []
    identical = 0
    for name in sorted(before):
        old, new = before[name], after[name]
        if old["recomputed"] == new["recomputed"]:
            identical += 1
        shift = abs(new["recomputed"] - old["recomputed"])
        scale = max(abs(old["recomputed"]), abs(new["recomputed"]), 1e-300)
        ulp = math.ulp(scale)
        roundoff = "residual" in name and 0.0 < abs(old["declared"]) <= 1.0e-12
        rel = shift / scale
        if shift == 0.0:
            category = "bit-identical"
        elif shift <= 8.0 * ulp:
            # last one or two bits of the printed quantity itself
            category = "last-bit"
        elif roundoff:
            # the value is cancellation noise: only its absolute size matters
            category = "cancellation"
        elif rel <= 1.0e-6:
            # differences of two O(1) kernels amplify eps-level input shifts to
            # ~1e-9 relative; six orders inside the printed 3-digit tolerance
            category = "amplified-roundoff"
        else:
            category = "physical-change"
        rows.append(
            {
                "row": name,
                "declared": old["declared"],
                "recomputed_before": old["recomputed"],
                "recomputed_after": new["recomputed"],
                "abs_shift": shift,
                "rel_shift": shift / scale,
                "shift_in_ulps": shift / ulp,
                # a physical O(1) kernel value cancels to ~1e-15; one unit of
                # that cancellation noise is eps * O(1) = eps
                "shift_in_eps": shift / EPS,
                "category": category,
                "verdict_before": old["verdict"],
                "verdict_after": new["verdict"],
            }
        )

    categories: dict[str, int] = {}
    for row in rows:
        categories[row["category"]] = categories.get(row["category"], 0) + 1
    moved = [r for r in rows if r["category"] != "bit-identical"]
    payload = {
        "schema": "tensorbem.finite_depth3d_roundoff_crosscheck/1",
        "base_revision": args.base,
        "rows": len(rows),
        "bit_identical": identical,
        "moved": len(moved),
        "categories": dict(sorted(categories.items())),
        "eps": EPS,
        "max_shift_in_eps": max((r["shift_in_eps"] for r in rows), default=0.0),
        "worst_non_cancellation_rel_shift": max(
            (r["rel_shift"] for r in rows if r["category"] != "cancellation"), default=0.0
        ),
        "rows_over_printed_tolerance": [
            r["row"] for r in rows if r["verdict_after"] == "DRIFT"
        ],
        "moved_rows": moved,
        "all_verdicts_after": sorted({r["verdict_after"] for r in rows}),
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"{payload['rows']} rows, {identical} bit-identical, {len(moved)} moved")
    print(f"categories: {payload['categories']}")
    for r in moved:
        print(
            f"  [{r['category']}] {r['row']}: before={r['recomputed_before']:.6e} "
            f"after={r['recomputed_after']:.6e} rel={r['rel_shift']:.2e} "
            f"({r['shift_in_ulps']:.2f} ulp of the value, {r['shift_in_eps']:.2f} eps)"
        )
    print(
        f"worst non-cancellation relative shift "
        f"{payload['worst_non_cancellation_rel_shift']:.2e} "
        f"(printed tolerance 5e-3); drift rows after: "
        f"{payload['rows_over_printed_tolerance'] or 'none'} "
        f"-> {OUT.relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())