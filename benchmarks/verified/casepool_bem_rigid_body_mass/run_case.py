#!/usr/bin/env python3
"""Case-pool driver for ``CASE-BEM-RIGID-BODY-MASS`` (算例池·刚体质量矩阵).

The physics and the case cycle live in
:mod:`tensorbem.benchmarks.rigid_body_mass_case`; this driver adds the two
things a case-pool run needs on top of it:

* **a re-derived verdict** -- it runs
  :mod:`tensorbem.benchmarks.rigid_body_mass_gate`, which recomputes the <3 %
  criterion and the assembly invariants from the committed matrices instead of
  quoting the verdict stored in ``run.json`` (see that module's "Scope and
  honesty" note);
* **an exit-code contract** -- exit 0 on PASS, 1 on FAIL, so a scheduler can
  treat the anchor as a signal and not as prose.

One-shot reproduction of the committed artefacts::

    PYTHONPATH=src python3 benchmarks/verified/casepool_bem_rigid_body_mass/run_case.py \\
        --refresh

``--refresh`` first re-runs ``python -m tensorbem.benchmarks.rigid_body_mass_case``
(0.2 s; convergence ladder 20 -> 5120 panels), then gates the fresh artefact.
Without ``--refresh`` the driver only gates what is committed, which is the
mode the registry ``--check`` path uses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:  # allow plain ``python3 <this file>``
    sys.path.insert(0, str(REPO / "src"))

from tensorbem.benchmarks import rigid_body_mass_gate as gate  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", default=gate.DEFAULT_ARTIFACT,
                        help="committed run.json (repo-relative)")
    parser.add_argument("--out", default=gate.DEFAULT_VERDICT,
                        help="verdict.json to write (repo-relative)")
    parser.add_argument("--refresh", action="store_true",
                        help="re-run the case module before gating")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    verdict = gate.main([
        "--artifact", args.artifact,
        "--out", args.out,
        *(["--refresh"] if args.refresh else []),
        *(["--quiet"] if args.quiet else []),
    ])
    if not args.quiet:
        payload = json.loads((REPO / args.out).read_text(encoding="utf-8"))
        print(json.dumps({"verdict": payload["verdict"],
                          "failed_checks": payload["failed_checks"]}, ensure_ascii=False))
    return verdict


if __name__ == "__main__":
    raise SystemExit(main())