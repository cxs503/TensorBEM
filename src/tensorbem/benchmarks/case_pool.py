#!/usr/bin/env python3
"""Anchor registry: dispatch the case-pool anchors this package implements.

Why a registry
--------------
The platform (``tensor_case_pool``) schedules anchors by ``case_code``; a
scheduling layer must be able to go from a string to a runnable case without
importing a file by path.  This module is that layer: it holds the
``case_code -> implementation`` map, validates the anchor, and exposes a CLI
(``python -m tensorbem.benchmarks.case_pool --anchor CASE-BEM-HULME-HEMISPHERE``)
that the thin per-anchor driver scripts and the tests both call.

Registered anchors
------------------
``CASE-BEM-HULME-HEMISPHERE``
    :mod:`tensorbem.benchmarks.hulme_hemisphere_case` -- floating hemisphere
    radiation/diffraction against the Hulme (1982) multi-pole reference,
    ``tensor_case_pool.id = 138``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from tensorbem.benchmarks import hulme_hemisphere_case

#: ``case_code -> module``.  A module must expose ``run`` and ``main``.
ANCHORS: dict[str, Any] = {
    hulme_hemisphere_case.CASE_CODE: hulme_hemisphere_case,
}


def get_anchor(case_code: str) -> Any:
    """Return the implementation module registered for ``case_code``."""
    try:
        return ANCHORS[case_code]
    except KeyError:
        raise KeyError(
            f"unknown case-pool anchor {case_code!r}; registered: {sorted(ANCHORS)}"
        ) from None


def run_anchor(case_code: str, out: str | Path, **kwargs: Any) -> Any:
    """Run a registered anchor, returning whatever its ``run`` returns."""
    return get_anchor(case_code).run(Path(out), **kwargs)


def list_anchors() -> list[dict[str, str]]:
    """Inventory for the platform / docs: code, module, case-pool id, tolerance."""
    inventory = []
    for code, module in sorted(ANCHORS.items()):
        inventory.append(
            {
                "case_code": code,
                "module": f"tensorbem.benchmarks.{module.__name__.rsplit('.', 1)[-1]}",
                "case_pool_id": str(getattr(module, "CASE_POOL_ID", "")),
                "tolerance": f"<{getattr(module, 'GATE', 0.0):.0%}",
            }
        )
    return inventory


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", help="case_code to run")
    parser.add_argument("--out", default="", help="output directory for the run artefacts")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--list", action="store_true", help="print the registry inventory")
    args = parser.parse_args(argv)

    if args.list or not args.anchor:
        print(json.dumps(list_anchors(), indent=2))
        return 0

    module = get_anchor(args.anchor)
    out = args.out or f"benchmarks/verified/{args.anchor.lower().replace('-', '_')}/evidence"
    argv_out = ["--out", out, "--shard", str(args.shard), "--shards", str(args.shards)]
    if args.merge:
        argv_out.append("--merge")
    return int(module.main(argv_out))


if __name__ == "__main__":
    raise SystemExit(main())