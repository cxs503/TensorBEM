#!/usr/bin/env python3
"""m17083 — run the verification commands for the inventory + tooling-portability fix
and record their exact outcome in docs/m17083_upstream_delta/verification.json.

Nothing here is transcribed by hand: counts come from parsing pytest's own summary.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "m17083_upstream_delta")

SUITES = {
    "portability_gate": ["tests/test_tooling_path_portability.py"],
    "window_modules": [
        "tests/test_numeric_evidence.py",
        "tests/test_steady_kelvin_identity.py",
        "tests/test_wavemaker_transfer.py",
        "tests/test_shape_gradient.py",
    ],
    "ported_legacy_test": ["tests/test_wigley3_freq_domain.py"],
}

SUMMARY = re.compile(
    r"(?:(\d+) failed)?,?\s*(?:(\d+) passed)?,?\s*(?:(\d+) xfailed)?.*?in ([\d.]+)s")


def run(paths: list[str]) -> dict:
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *paths]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=3600)
    tail = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    line = ""
    for l in reversed(tail):
        if "passed" in l or "failed" in l or "error" in l:
            line = l
            break
    m = SUMMARY.search(line)
    get = lambda i: int(m.group(i)) if (m and m.group(i)) else 0  # noqa: E731
    return {
        "cmd": " ".join(cmd),
        "exit_code": proc.returncode,
        "summary_line": line.strip(),
        "failed": get(1),
        "passed": get(2),
        "xfailed": get(3),
        "seconds": float(m.group(4)) if m else None,
    }


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    res = {name: run(paths) for name, paths in SUITES.items()}
    suites = [r for r in res.values() if isinstance(r, dict) and "exit_code" in r]
    res["all_green"] = all(r["exit_code"] == 0 for r in suites)
    total = sum(r["passed"] for r in suites)
    res["totals"] = {
        "passed": total,
        "failed": sum(r["failed"] for r in suites),
        "xfailed": sum(r["xfailed"] for r in suites),
    }
    with open(os.path.join(OUT, "verification.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k in ("totals", "all_green")}, indent=2))
    for name, r in res.items():
        if isinstance(r, dict) and "exit_code" in r:
            print(f"{name:22s} exit={r['exit_code']} {r['summary_line']}")
    return 0 if res["all_green"] else 1


if __name__ == "__main__":
    raise SystemExit(main())