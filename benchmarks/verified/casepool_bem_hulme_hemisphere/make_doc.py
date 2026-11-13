#!/usr/bin/env python3
"""Render ``docs/bem_hulme_hemisphere_case.md`` from a completed case run.

Every number in the generated document is read out of ``results.json`` (and its
embedded ``benchmark`` table) produced by
:mod:`tensorbem.benchmarks.hulme_hemisphere_case` -- nothing is retyped by
hand.  The renderer itself lives in that module (``render_doc``) so that the
merge step and this script cannot drift apart.

::

    PYTHONPATH=src python3 benchmarks/verified/casepool_bem_hulme_hemisphere/make_doc.py \
        --results benchmarks/verified/casepool_bem_hulme_hemisphere/evidence/results.json \
        --doc docs/bem_hulme_hemisphere_case.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from tensorbem.benchmarks import hulme_hemisphere_case


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default="benchmarks/verified/casepool_bem_hulme_hemisphere/evidence/results.json",
    )
    parser.add_argument("--doc", default="docs/bem_hulme_hemisphere_case.md")
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.results).read_text(encoding="utf-8"))
    path = hulme_hemisphere_case.render_doc(payload, Path(args.doc))
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())