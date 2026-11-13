#!/usr/bin/env python3
"""Thin driver for case-pool anchor ``BEM-HULME-HEMISPHERE`` (floating hemisphere).

The case itself is implemented once, in
:mod:`tensorbem.benchmarks.hulme_hemisphere_case`, and registered in
:mod:`tensorbem.benchmarks.case_pool`.  This file only keeps the historical
per-anchor path alive (``PYTHONPATH=src python3
benchmarks/verified/casepool_bem_hulme_hemisphere/run_case.py ...``) so the
reproduction command recorded in earlier artefacts still works, and so a reader
finds the anchor next to its evidence.

Reproduce::

    PYTHONPATH=src python3 benchmarks/verified/casepool_bem_hulme_hemisphere/run_case.py \
        --out benchmarks/verified/casepool_bem_hulme_hemisphere/evidence --shard 0 --shards 1
    PYTHONPATH=src python3 benchmarks/verified/casepool_bem_hulme_hemisphere/run_case.py \
        --out benchmarks/verified/casepool_bem_hulme_hemisphere/evidence --merge

Equivalent, without the per-anchor path::

    PYTHONPATH=src python3 -m tensorbem.benchmarks.case_pool \
        --anchor CASE-BEM-HULME-HEMISPHERE --out benchmarks/verified/casepool_bem_hulme_hemisphere/evidence
"""

from __future__ import annotations

import sys
from typing import Sequence

from tensorbem.benchmarks import case_pool, hulme_hemisphere_case

ANCHOR = hulme_hemisphere_case.CASE_CODE


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    return case_pool.main(["--anchor", ANCHOR, *args])


if __name__ == "__main__":
    raise SystemExit(main())