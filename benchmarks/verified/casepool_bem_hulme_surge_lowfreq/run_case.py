#!/usr/bin/env python3
"""Case-pool driver: horizontal (surge) long-wave anchor of the floating hemisphere.

Thin wrapper so the case directory follows the ``run_case.py`` convention of the
other ``benchmarks/verified/*`` packages; all physics lives in :mod:`limits`.

Run::

    PYTHONPATH=src python benchmarks/verified/casepool_bem_hulme_surge_lowfreq/run_case.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import limits  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(limits.main())