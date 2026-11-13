#!/usr/bin/env python3
"""Independent re-run of the finest barge level (640 panels) for the m17208 audit.

Re-executes ``tensorbem.benchmarks.barge_crosscheck.run_level(LEVELS[-1])`` from
the current source tree and compares every heave added-mass / damping cell with
the committed artifact ``benchmarks/verified/barge_crosscheck/results.json``.
Writes ``repro_finest.json`` next to this file.

Run:  PYTHONPATH=src python3 artifacts/m17208_audit/repro_finest.py
"""

from __future__ import annotations

import json
import pathlib

from tensorbem.benchmarks.barge_crosscheck import LEVELS, run_level

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = REPO / "benchmarks/verified/barge_crosscheck/results.json"


def main() -> int:
    stored = json.loads(RESULTS.read_text())["levels"][-1]
    fresh = run_level(LEVELS[-1])
    rows = {}
    for route in ("bare", "lid"):
        for nu_key, comps in stored[route].items():
            for quantity in ("added_mass", "damping"):
                got = fresh[route][nu_key][quantity]
                ref = comps[quantity]
                rows[f"{route}/{nu_key}/{quantity}"] = {
                    "fresh": got,
                    "committed": ref,
                    "relative": abs(got - ref) / max(abs(ref), 1.0),
                }
    worst = max(row["relative"] for row in rows.values())
    payload = {
        "level": list(LEVELS[-1]),
        "panels": stored["panels"],
        "cells": rows,
        "worst_relative_vs_committed": worst,
        "reproduced": worst < 1e-10,
    }
    (HERE / "repro_finest.json").write_text(json.dumps(payload, indent=1, sort_keys=True))
    print("panels", stored["panels"], "worst_rel", worst, "reproduced", payload["reproduced"])
    return 0 if payload["reproduced"] else 1


if __name__ == "__main__":
    raise SystemExit(main())